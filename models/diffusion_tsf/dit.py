"""Diffusion Transformer (DiT) backbone for 2D occupancy-map diffusion.

Drop-in replacement for ``ConditionalUNet2D`` where the training loop calls::

    forward(x, t, cond, encoder_hidden_states=None) -> (B, out_channels, H, W)

``B`` is the batch dimension (either ``B`` in multi-channel mode or ``B*V`` in
factorized mode). ``encoder_hidden_states`` has shape ``(B, V, ctx_dim)`` when
cross-attention is enabled.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=t.device, dtype=torch.float32)
        / max(half, 1)
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class _SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} not divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.drop = drop

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop if self.training else 0.0)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class _CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, 2 * dim, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.drop = drop

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        _, M, _ = ctx.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(ctx).reshape(B, M, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop if self.training else 0.0)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class _MLP(nn.Module):
    def __init__(self, dim: int, hidden: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


class _DiTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, drop: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = _SelfAttention(dim, num_heads, drop=drop)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = _MLP(dim, int(dim * mlp_ratio), drop=drop)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        s1, sc1, g1, s2, sc2, g2 = self.adaLN(c).chunk(6, dim=-1)
        x = x + g1.unsqueeze(1) * self.attn(_modulate(self.norm1(x), s1, sc1))
        x = x + g2.unsqueeze(1) * self.mlp(_modulate(self.norm2(x), s2, sc2))
        return x


class _DiTCrossAttnBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, drop: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = _SelfAttention(dim, num_heads, drop=drop)
        self.norm_x = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = _CrossAttention(dim, num_heads, drop=drop)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = _MLP(dim, int(dim * mlp_ratio), drop=drop)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor, ctx: Optional[torch.Tensor]) -> torch.Tensor:
        mods = self.adaLN(c).chunk(9, dim=-1)
        s1, sc1, g1, sx, scx, gx, s2, sc2, g2 = mods
        x = x + g1.unsqueeze(1) * self.self_attn(_modulate(self.norm1(x), s1, sc1))
        if ctx is not None:
            x = x + gx.unsqueeze(1) * self.cross_attn(_modulate(self.norm_x(x), sx, scx), ctx)
        x = x + g2.unsqueeze(1) * self.mlp(_modulate(self.norm2(x), s2, sc2))
        return x


class FactorizedDiT(nn.Module):
    """Patch DiT with AdaLN-Zero and one bottleneck cross-attention to iTrans tokens."""

    def __init__(
        self,
        in_channels: int,
        cond_channels: int,
        out_channels: int = 1,
        image_height: int = 64,
        patch_size: Tuple[int, int] = (8, 8),
        embed_dim: int = 384,
        depth: int = 8,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        context_dim: int = 256,
        max_pos_tokens: int = 8192,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        pH, pW = patch_size
        if image_height % pH != 0:
            raise ValueError(f"image_height={image_height} not divisible by patch_height={pH}")
        self.image_height = image_height
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.out_channels = out_channels
        self.gradient_checkpointing = gradient_checkpointing

        self.x_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cond_embed = nn.Conv2d(cond_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        self.pos_x = nn.Parameter(torch.zeros(1, max_pos_tokens, embed_dim))
        self.pos_cond = nn.Parameter(torch.zeros(1, max_pos_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_x, std=0.02)
        nn.init.trunc_normal_(self.pos_cond, std=0.02)

        self.t_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

        self.ctx_proj = nn.Linear(context_dim, embed_dim)
        self.ctx_norm = nn.LayerNorm(embed_dim, eps=1e-6)

        self.bottleneck_idx = depth // 2
        self.blocks = nn.ModuleList()
        for i in range(depth):
            if i == self.bottleneck_idx:
                self.blocks.append(_DiTCrossAttnBlock(embed_dim, num_heads, mlp_ratio, dropout))
            else:
                self.blocks.append(_DiTBlock(embed_dim, num_heads, mlp_ratio, dropout))

        self.final_norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, 2 * embed_dim))
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)

        self.head = nn.Linear(embed_dim, out_channels * pH * pW)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _patchify(self, img: torch.Tensor, proj: nn.Conv2d) -> Tuple[torch.Tensor, int, int]:
        h = proj(img)
        gh, gw = h.shape[-2], h.shape[-1]
        return h.flatten(2).transpose(1, 2), gh, gw

    def _unpatchify(self, tokens: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
        B = tokens.shape[0]
        pH, pW = self.patch_size
        h = self.head(tokens).view(B, gh, gw, self.out_channels, pH, pW)
        h = h.permute(0, 3, 1, 4, 2, 5).contiguous()
        return h.view(B, self.out_channels, gh * pH, gw * pW)

    def _pad_to_patch(self, img: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        pH, pW = self.patch_size
        H, W = img.shape[-2], img.shape[-1]
        pad_h = (pH - H % pH) % pH
        pad_w = (pW - W % pW) % pW
        if pad_h or pad_w:
            img = F.pad(img, (0, pad_w, 0, pad_h), mode="reflect")
        return img, pad_h, pad_w

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        Bsz, _, H, W = x.shape

        x_p, pad_h, pad_w = self._pad_to_patch(x)
        cond_p, _, _ = self._pad_to_patch(cond)

        x_tok, gh, gw = self._patchify(x_p, self.x_embed)
        c_tok, _, _ = self._patchify(cond_p, self.cond_embed)

        Nx, Nc = x_tok.shape[1], c_tok.shape[1]
        if Nx > self.pos_x.shape[1] or Nc > self.pos_cond.shape[1]:
            raise RuntimeError(
                f"DiT pos table too small: need Nx={Nx}, Nc={Nc}, "
                f"have {self.pos_x.shape[1]}. Increase max_pos_tokens."
            )
        x_tok = x_tok + self.pos_x[:, :Nx]
        c_tok = c_tok + self.pos_cond[:, :Nc]
        tokens = torch.cat([c_tok, x_tok], dim=1)

        if t.shape[0] != Bsz:
            raise ValueError(f"timestep batch {t.shape[0]} != batch {Bsz}")
        t_emb = self.t_embed(_timestep_embedding(t, self.embed_dim))

        ctx_proj: Optional[torch.Tensor] = None
        if encoder_hidden_states is not None:
            if encoder_hidden_states.shape[0] != Bsz:
                raise ValueError(
                    f"encoder_hidden_states batch {encoder_hidden_states.shape[0]} != {Bsz}"
                )
            ctx_proj = self.ctx_norm(self.ctx_proj(encoder_hidden_states))

        for i, block in enumerate(self.blocks):
            if i == self.bottleneck_idx:
                if self.gradient_checkpointing and self.training:
                    tokens = checkpoint(block, tokens, t_emb, ctx_proj, use_reentrant=False)
                else:
                    tokens = block(tokens, t_emb, ctx_proj)
            else:
                if self.gradient_checkpointing and self.training:
                    tokens = checkpoint(block, tokens, t_emb, use_reentrant=False)
                else:
                    tokens = block(tokens, t_emb)

        x_out = tokens[:, Nc:]
        shift, scale = self.final_adaLN(t_emb).chunk(2, dim=-1)
        x_out = _modulate(self.final_norm(x_out), shift, scale)
        out = self._unpatchify(x_out, gh, gw)

        if pad_h or pad_w:
            out = out[:, :, :H, :W]
        return out
