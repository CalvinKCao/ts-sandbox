"""Channel-Independent DiT for high-variate time series diffusion.

each variate's 2D image processed independently thru a shared transformer,
with cross-variate attention layers to capture inter-variate correlations.

Key idea: instead of stacking 861 variates as channels (impossible),
treat each variate as a separate (B, 1, H, W) image. batch all variates
together as (B*V, C_per_var, H, W), run shared DiT, then reshape back.
cross-variate attention operates on pooled per-variate summaries.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def _modulate(x, shift, scale):
    """AdaLN modulation: x * (1 + scale) + shift"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _timestep_embedding(t, dim, max_period=10000):
    """sinusoidal timestep embeddings ala DDPM / attention-is-all-you-need"""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


# ============================================================================
# building blocks
# ============================================================================

class Attention(nn.Module):
    """multi-head self-attn using scaled_dot_product_attention (flash-compatible)."""

    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = attn_drop

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each (B, heads, N, head_dim)

        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


class DiTBlock(nn.Module):
    """standard DiT block with AdaLN-Zero conditioning on the timestep."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, num_heads, attn_drop=drop)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)

        # adaln-zero: 6 modulation vectors (shift/scale/gate for attn + mlp)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, c):
        # x: (BV, N, D), c: (BV, D) — timestep conditioning
        mods = self.adaLN(c).chunk(6, dim=-1)
        s1, sc1, g1, s2, sc2, g2 = mods

        h = _modulate(self.norm1(x), s1, sc1)
        h = self.attn(h)
        x = x + g1.unsqueeze(1) * h

        h = _modulate(self.norm2(x), s2, sc2)
        h = self.mlp(h)
        x = x + g2.unsqueeze(1) * h
        return x


class CrossVariateBlock(nn.Module):
    """cross-variate attention: pool spatial tokens -> attend across variates -> broadcast.

    operates on pooled per-variate summaries so the attention matrix is (B, V, V)
    instead of (B, V*N, V*N) which would be insane for V=861.
    """

    def __init__(self, dim, num_heads, drop=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, num_heads, attn_drop=drop)
        self.proj = nn.Linear(dim, dim)
        # start with zero gate so cross-var is a no-op initially
        self.gate = nn.Parameter(torch.zeros(dim))

    def forward(self, x, B, V):
        # x: (B*V, N, D)
        BV, N, D = x.shape
        x_4d = x.view(B, V, N, D)

        # pool over spatial patches → per-variate summary
        summary = x_4d.mean(dim=2)  # (B, V, D)
        summary_n = self.norm(summary)

        # attend across variates
        delta = self.attn(summary_n)  # (B, V, D)
        delta = self.proj(delta)

        # gated additive broadcast back to spatial tokens
        x_4d = x_4d + (self.gate * delta).unsqueeze(2)
        return x_4d.reshape(BV, N, D)


# ============================================================================
# main backbone
# ============================================================================

class ChannelIndependentDiT(nn.Module):
    """Channel-Independent Diffusion Transformer for high-variate time series.

    Each variate is processed as a separate 1-channel 2D image through a shared
    patchified transformer. Cross-variate attention is inserted periodically.

    backbone interface matches U-Net: forward(x, t, cond, encoder_hidden_states)
    so it's a drop-in replacement inside DiffusionTSF.
    """

    def __init__(
        self,
        image_height: int = 64,
        patch_size: Tuple[int, int] = (8, 8),
        embed_dim: int = 256,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        in_channels: int = 2,
        cond_channels: int = 1,
        out_channels: int = 1,
        n_variates: int = 1,
        cross_variate_every: int = 3,  # insert cross-var attn every N blocks. 0=disable
        dropout: float = 0.1,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.image_height = image_height
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.in_channels = in_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.n_variates = n_variates
        self.gradient_checkpointing = gradient_checkpointing
        self.cross_variate_every = cross_variate_every

        pH, pW = patch_size

        # patch projection convolutions
        self.x_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cond_embed = nn.Conv2d(cond_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        # learned positional embeddings (big enough for most configs)
        # 8192 tokens handles ~1200-wide images with 8px patches at H=64
        self.pos_embed = nn.Parameter(torch.zeros(1, 8192, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # timestep MLP
        self.t_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

        # transformer blocks + cross-variate blocks
        self.blocks = nn.ModuleList()
        self.cross_var_indices = set()
        self.cross_var_blocks = nn.ModuleDict()
        for i in range(depth):
            self.blocks.append(DiTBlock(embed_dim, num_heads, mlp_ratio, drop=dropout))
            if cross_variate_every > 0 and (i + 1) % cross_variate_every == 0:
                self.cross_var_blocks[str(i)] = CrossVariateBlock(embed_dim, num_heads, drop=dropout)
                self.cross_var_indices.add(i)

        # final output: norm + adaln + linear projection to patch pixels
        self.final_norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, 2 * embed_dim))
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)

        self.head = nn.Linear(embed_dim, out_channels * pH * pW)
        # zero-init output so model starts as identity (noise in, noise out)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"ChannelIndependentDiT: {n_params/1e6:.1f}M params | "
            f"depth={depth} dim={embed_dim} heads={num_heads} "
            f"patch={patch_size} cross_var_every={cross_variate_every} "
            f"n_variates={n_variates}"
        )

    def _patchify(self, x, conv):
        """(B, C, H, W) -> (B, num_patches, embed_dim)"""
        return conv(x).flatten(2).transpose(1, 2)

    def _unpatchify(self, tokens, grid_h, grid_w):
        """(B, num_patches, embed_dim) -> (B, out_channels, H_grid, W_grid)"""
        B = tokens.shape[0]
        pH, pW = self.patch_size
        tokens = self.head(tokens)  # (B, N, out_ch * pH * pW)
        tokens = tokens.view(B, grid_h, grid_w, self.out_channels, pH, pW)
        tokens = tokens.permute(0, 3, 1, 4, 2, 5).contiguous()
        return tokens.view(B, self.out_channels, grid_h * pH, grid_w * pW)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        args:
            x: (B*V, in_channels, H, W) batched per-variate noisy future images
            t: (B,) or (B*V,) diffusion timestep
            cond: (B*V, cond_channels, H, W_cond) batched per-variate conditioning
            encoder_hidden_states: ignored (kept for interface compat w/ UNet)
        returns:
            (B*V, out_channels, H, W) noise prediction per variate
        """
        BV, C, H, W = x.shape
        V = self.n_variates
        B = BV // V
        pH, pW = self.patch_size

        # --- pad to patch-aligned dims ---
        pad_h = (pH - H % pH) % pH
        pad_w = (pW - W % pW) % pW
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        Wc = cond.shape[3]
        pad_wc = (pW - Wc % pW) % pW
        if pad_h or pad_wc:
            cond = F.pad(cond, (0, pad_wc, 0, pad_h), mode='reflect')

        Hp, Wp = x.shape[2], x.shape[3]
        gh, gw = Hp // pH, Wp // pW
        # print(f"[ci_dit] x_padded={x.shape}, cond_padded={cond.shape}, grid=({gh},{gw})")

        # --- patchify ---
        x_tok = self._patchify(x, self.x_embed)      # (BV, Nx, D)
        c_tok = self._patchify(cond, self.cond_embed)  # (BV, Nc, D)
        Nx = x_tok.shape[1]
        Nc = c_tok.shape[1]

        # positional embeddings: [cond_pos | x_pos]
        total_tok = Nc + Nx
        pos = self.pos_embed[:, :total_tok]
        c_tok = c_tok + pos[:, :Nc]
        x_tok = x_tok + pos[:, Nc:total_tok]

        tokens = torch.cat([c_tok, x_tok], dim=1)  # (BV, Nc+Nx, D)

        # --- timestep embedding ---
        # t might be (B,) or (B*V,). normalize to (BV,)
        if t.shape[0] != BV:
            t_emb = _timestep_embedding(t, self.embed_dim)
            t_emb = self.t_embed(t_emb)
            t_emb = t_emb.unsqueeze(1).expand(-1, V, -1).reshape(BV, -1)
        else:
            t_emb = _timestep_embedding(t, self.embed_dim)
            t_emb = self.t_embed(t_emb)

        # --- transformer ---
        for i, block in enumerate(self.blocks):
            if self.gradient_checkpointing and self.training:
                tokens = checkpoint(block, tokens, t_emb, use_reentrant=False)
            else:
                tokens = block(tokens, t_emb)

            # cross-variate attention after this block?
            if i in self.cross_var_indices:
                cv = self.cross_var_blocks[str(i)]
                if self.gradient_checkpointing and self.training:
                    tokens = checkpoint(cv, tokens, B, V, use_reentrant=False)
                else:
                    tokens = cv(tokens, B, V)

        # --- extract future tokens + unpatchify ---
        x_tok = tokens[:, Nc:]  # (BV, Nx, D)

        shift, scale = self.final_adaLN(t_emb).chunk(2, dim=-1)
        x_tok = _modulate(self.final_norm(x_tok), shift, scale)
        out = self._unpatchify(x_tok, gh, gw)

        # remove padding
        if pad_h or pad_w:
            out = out[:, :, :H, :W]

        return out
