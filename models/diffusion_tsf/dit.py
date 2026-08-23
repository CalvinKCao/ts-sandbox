"""Factorized Diffusion Transformer (DiT) backbone for time-series occupancy maps.

Factorized per-variate DiT denoiser for binary CDF images. Call signature:

    forward(x, t, cond, encoder_hidden_states=None) -> (BV, out_channels, H, W)

Design (matches user spec):
- Per-variate factorized: input is (BV, in_channels, H, W_fut); one variate per
  forward pass with shared weights across the BV batch.
- No internal cross-variate mixing. Variates only couple through the bottleneck
  cross-attention to the (BV, V, ctx_dim) iTransformer tokens.
- AdaLN-Zero time conditioning in every block (DiT paper, Peebles & Xie 2023).
- Visual conditioning (past 2D) enters by patchifying separately and concatenating
  the cond patches into the sequence: [cond_patches | x_patches].
- Guidance ghost image is concatenated to x along the channel dim *before* this
  module is called (same as U-Net path), so it just shows up as extra in_channels.
- Single iTrans cross-attention site at depth // 2 (the "bottleneck").
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # AdaLN-Zero: x * (1 + scale) + shift, broadcast over the token axis
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
    """Multi-head self-attention via scaled_dot_product_attention (flash-friendly)."""

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
    """Multi-head cross-attention: queries from x, keys/values from a context tensor."""

    def __init__(self, dim: int, num_heads: int, drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, 2 * dim, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.drop = drop

    def forward(
        self,
        x: torch.Tensor,
        ctx: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        context_window_indices: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False,
    ):
        B, N, C = x.shape
        ctx_batch, M, _ = ctx.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(ctx).reshape(
            ctx_batch, M, 2, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        if context_window_indices is not None:
            if context_window_indices.shape != (B,):
                raise ValueError(
                    "context_window_indices must have one entry per query row, got "
                    f"{tuple(context_window_indices.shape)} for B={B}"
                )
            # Project K/V once per parent, then index them to patch rows. This
            # keeps the shared frozen context semantics but reduces the former
            # per-parent nonzero/SDPA loop to one batched attention call.
            k = k.index_select(0, context_window_indices)
            v = v.index_select(0, context_window_indices)
            bias_rows = (
                attn_bias[:, None, None, :].to(dtype=q.dtype)
                if attn_bias is not None
                else None
            )
            if return_attn_weights:
                logits = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
                if bias_rows is not None:
                    logits = logits + bias_rows
                weights = torch.softmax(logits, dim=-1)
                out = torch.matmul(weights, v)
                mean_weights = weights.mean(dim=(1, 2))
            else:
                out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=bias_rows,
                    dropout_p=self.drop if self.training else 0.0,
                )
            out = out.transpose(1, 2).reshape(B, N, C)
            out = self.proj(out)
            if return_attn_weights:
                return out, mean_weights
            return out
        if ctx_batch != B:
            raise ValueError(f"ctx batch {ctx_batch} != query batch {B}")
        if return_attn_weights:
            scale = self.head_dim ** -0.5
            attn_logits = torch.matmul(q, k.transpose(-2, -1)) * scale
            if attn_bias is not None:
                attn_logits = attn_logits + attn_bias[:, None, None, :]
            attn_weights = torch.softmax(attn_logits, dim=-1)
            out = torch.matmul(attn_weights, v)
            out = out.transpose(1, 2).reshape(B, N, C)
            out = self.proj(out)
            # mean over heads and query tokens -> (B, M)
            mean_weights = attn_weights.mean(dim=(1, 2))
            return out, mean_weights
        if attn_bias is not None:
            attn_bias = attn_bias[:, None, None, :].to(dtype=q.dtype)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_bias,
            dropout_p=self.drop if self.training else 0.0,
        )
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
    """Standard DiT block: AdaLN-Zero(self-attn) + AdaLN-Zero(MLP)."""

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
    """Bottleneck variant: adds cross-attention to encoder_hidden_states.

    Layer order: AdaLN(self-attn) -> AdaLN(cross-attn to ctx) -> AdaLN(MLP).
    Each sub-layer has its own AdaLN-Zero gate, so the cross-attn starts as a
    no-op at init and the network can choose how much to lean on it.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        drop: float,
        enable_cross_scale_attention: bool = False,
        target_context_bias: float = 0.0,
    ):
        super().__init__()
        self.enable_cross_scale_attention = enable_cross_scale_attention
        self.target_context_bias = float(target_context_bias)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = _SelfAttention(dim, num_heads, drop=drop)
        self.norm_x = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = _CrossAttention(dim, num_heads, drop=drop)
        if enable_cross_scale_attention:
            self.norm_s = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.scale_attn = _CrossAttention(dim, num_heads, drop=drop)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = _MLP(dim, int(dim * mlp_ratio), drop=drop)
        num_mods = 12 if enable_cross_scale_attention else 9
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, num_mods * dim))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        ctx: Optional[torch.Tensor],
        scale_indices: Optional[torch.Tensor] = None,
        variate_indices: Optional[torch.Tensor] = None,
        token_variate_ids: Optional[torch.Tensor] = None,
        context_window_indices: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False,
    ):
        mods = self.adaLN(c).chunk(12 if self.enable_cross_scale_attention else 9, dim=-1)
        if self.enable_cross_scale_attention:
            s1, sc1, g1, sx, scx, gx, ss, scs, gs, s2, sc2, g2 = mods
        else:
            s1, sc1, g1, sx, scx, gx, s2, sc2, g2 = mods
        x = x + g1.unsqueeze(1) * self.self_attn(_modulate(self.norm1(x), s1, sc1))
        cross_attn_weights = None
        if ctx is not None:
            attn_bias = None
            if self.target_context_bias != 0.0 and ctx.shape[1] > 1:
                if variate_indices is None:
                    raise ValueError("variate_indices are required for target-context attention bias.")
                if variate_indices.shape[0] != x.shape[0]:
                    raise ValueError(
                        f"variate_indices batch {variate_indices.shape[0]} != x batch {x.shape[0]}"
                    )
                target_ids = variate_indices.long()
                if token_variate_ids is not None:
                    if token_variate_ids.shape[0] != ctx.shape[1]:
                        raise ValueError(
                            f"token_variate_ids length {token_variate_ids.shape[0]} "
                            f"!= ctx tokens {ctx.shape[1]}"
                        )
                    attn_bias = torch.zeros(
                        x.shape[0],
                        ctx.shape[1],
                        device=x.device,
                        dtype=x.dtype,
                    )
                    own_mask = token_variate_ids[None, :] == target_ids[:, None]
                    attn_bias = attn_bias.masked_fill(own_mask, self.target_context_bias)
                else:
                    if target_ids.min() < 0 or target_ids.max() >= ctx.shape[1]:
                        raise ValueError(
                            f"variate_indices must be in [0, {ctx.shape[1] - 1}] for ctx tokens."
                        )
                    attn_bias = torch.zeros(
                        x.shape[0],
                        ctx.shape[1],
                        device=x.device,
                        dtype=x.dtype,
                    )
                    attn_bias.scatter_(1, target_ids.unsqueeze(1), self.target_context_bias)
            cross_in = _modulate(self.norm_x(x), sx, scx)
            if return_attn_weights:
                cross_out, cross_attn_weights = self.cross_attn(
                    cross_in,
                    ctx,
                    attn_bias=attn_bias,
                    context_window_indices=context_window_indices,
                    return_attn_weights=True,
                )
            else:
                cross_out = self.cross_attn(
                    cross_in,
                    ctx,
                    attn_bias=attn_bias,
                    context_window_indices=context_window_indices,
                )
            x = x + gx.unsqueeze(1) * cross_out
        if self.enable_cross_scale_attention:
            if scale_indices is None:
                raise ValueError("scale_indices are required for cross-scale attention.")
            if x.shape[0] % 2 != 0:
                raise ValueError("cross-scale attention expects paired coarse/fine batch items.")
            expected = torch.tensor([0, 1], device=scale_indices.device, dtype=scale_indices.dtype)
            if not torch.equal(scale_indices.reshape(-1, 2), expected.view(1, 2).expand(x.shape[0] // 2, -1)):
                raise ValueError("cross-scale attention expects adjacent [coarse, fine] scale ordering.")
            grouped = x.reshape(-1, 2, x.shape[1], x.shape[2])
            other_scale = grouped.flip(1).reshape_as(x)
            x = x + gs.unsqueeze(1) * self.scale_attn(
                _modulate(self.norm_s(x), ss, scs),
                other_scale,
            )
        x = x + g2.unsqueeze(1) * self.mlp(_modulate(self.norm2(x), s2, sc2))
        if return_attn_weights:
            return x, cross_attn_weights
        return x


class FactorizedDiT(nn.Module):
    """Per-variate DiT backbone with bottleneck cross-attention to frozen tokens.

    Inputs:
        x: (BV, in_channels, H, W_fut) noisy future canvas plus stage auxiliaries
        t: (BV,) diffusion timestep
        cond: (BV, cond_channels, H, W_fut) visual conditioning (past 2D resized)
        encoder_hidden_states: (BV, V, ctx_dim) or None — cross-variate token memory

    Returns:
        (BV, out_channels, H, W_fut) noise prediction
    """

    def __init__(
        self,
        in_channels: int,
        cond_channels: int,
        out_channels: int = 1,
        image_height: int = 32,
        patch_size: Tuple[int, int] = (8, 8),
        embed_dim: int = 384,
        depth: int = 8,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        context_dim: int = 256,
        max_pos_tokens: int = 8192,
        gradient_checkpointing: bool = False,
        cond_patch_size: Optional[Tuple[int, int]] = None,
        use_scale_embedding: bool = False,
        enable_cross_scale_attention: bool = False,
        use_variate_embedding: bool = False,
        max_variates: int = 512,
        cross_variate_context_bias: float = 0.0,
        use_patch_abs_embedding: bool = False,
        max_coarse_bins: int = 16,
        max_horizon_steps: int = 1024,
        use_horizon_chunk_embedding: bool = False,
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
        self.use_scale_embedding = use_scale_embedding
        self.enable_cross_scale_attention = enable_cross_scale_attention
        self.use_variate_embedding = use_variate_embedding
        self.use_patch_abs_embedding = use_patch_abs_embedding
        self.use_horizon_chunk_embedding = use_horizon_chunk_embedding
        self.cond_patch_size = cond_patch_size or patch_size

        self.x_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cond_embed = nn.Conv2d(
            cond_channels,
            embed_dim,
            kernel_size=self.cond_patch_size,
            stride=self.cond_patch_size,
        )

        # Separate learned positional embeddings for cond vs x slots so the model
        # can distinguish them even though they share the same sequence axis.
        self.pos_x = nn.Parameter(torch.zeros(1, max_pos_tokens, embed_dim))
        self.pos_cond = nn.Parameter(torch.zeros(1, max_pos_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_x, std=0.02)
        nn.init.trunc_normal_(self.pos_cond, std=0.02)

        self.t_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        if use_scale_embedding:
            self.scale_embed = nn.Embedding(2, embed_dim)
        else:
            self.scale_embed = None
        if use_variate_embedding:
            self.variate_embed = nn.Embedding(max_variates, embed_dim)
        else:
            self.variate_embed = None
        if use_patch_abs_embedding:
            self.coarse_bin_embed = nn.Embedding(max_coarse_bins, embed_dim)
            self.horizon_time_embed = nn.Embedding(max_horizon_steps, embed_dim)
        else:
            self.coarse_bin_embed = None
            self.horizon_time_embed = None
        if use_horizon_chunk_embedding:
            self.horizon_chunk_mlp = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim),
            )
            nn.init.zeros_(self.horizon_chunk_mlp[-1].weight)
            nn.init.zeros_(self.horizon_chunk_mlp[-1].bias)
        else:
            self.horizon_chunk_mlp = None

        self.ctx_proj = nn.Linear(context_dim, embed_dim)
        self.ctx_norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # Bottleneck position: middle of the stack. One cross-attn block.
        self.bottleneck_idx = depth // 2
        self.blocks = nn.ModuleList()
        for i in range(depth):
            if i == self.bottleneck_idx:
                self.blocks.append(
                    _DiTCrossAttnBlock(
                        embed_dim,
                        num_heads,
                        mlp_ratio,
                        dropout,
                        enable_cross_scale_attention=enable_cross_scale_attention,
                        target_context_bias=cross_variate_context_bias,
                    )
                )
            else:
                self.blocks.append(_DiTBlock(embed_dim, num_heads, mlp_ratio, dropout))

        self.final_norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, 2 * embed_dim))
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)

        self.head = nn.Linear(embed_dim, out_channels * pH * pW)
        # zero-init head: model starts as identity (noise in -> noise out)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _patchify(self, img: torch.Tensor, proj: nn.Conv2d) -> Tuple[torch.Tensor, int, int]:
        """(B, C, H, W) -> (B, gh*gw, D), with (gh, gw) returned for unpatchify."""
        h = proj(img)  # (B, D, gh, gw)
        gh, gw = h.shape[-2], h.shape[-1]
        return h.flatten(2).transpose(1, 2), gh, gw

    def _unpatchify(self, tokens: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
        B = tokens.shape[0]
        pH, pW = self.patch_size
        h = self.head(tokens).view(B, gh, gw, self.out_channels, pH, pW)
        h = h.permute(0, 3, 1, 4, 2, 5).contiguous()
        return h.view(B, self.out_channels, gh * pH, gw * pW)

    @staticmethod
    def _pad_to_patch(
        img: torch.Tensor,
        patch_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, int, int]:
        pH, pW = patch_size
        H, W = img.shape[-2], img.shape[-1]
        pad_h = (pH - H % pH) % pH
        pad_w = (pW - W % pW) % pW
        if pad_h or pad_w:
            img = F.pad(img, (0, pad_w, 0, pad_h), mode="reflect")
        return img, pad_h, pad_w

    def encode_horizon_chunk(
        self,
        t0: torch.Tensor,
        horizon: torch.Tensor,
        inner: int,
    ) -> torch.Tensor:
        """Sinusoid of t0/H and t1/H through a zero-init MLP → (N, embed_dim)."""
        if self.horizon_chunk_mlp is None:
            raise RuntimeError("horizon chunk embed requested but DiT was built without it")
        if t0.shape != horizon.shape:
            raise ValueError(
                f"t0 shape {tuple(t0.shape)} != horizon shape {tuple(horizon.shape)}"
            )
        h = horizon.to(dtype=torch.float32).clamp(min=1.0)
        t0_f = t0.to(dtype=torch.float32)
        t1_f = t0_f + float(inner)
        e0 = _timestep_embedding(t0_f / h, self.embed_dim)
        e1 = _timestep_embedding(t1_f / h, self.embed_dim)
        return self.horizon_chunk_mlp(e0 + e1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        scale_indices: Optional[torch.Tensor] = None,
        variate_indices: Optional[torch.Tensor] = None,
        token_variate_ids: Optional[torch.Tensor] = None,
        context_window_indices: Optional[torch.Tensor] = None,
        patch_coarse_bin: Optional[torch.Tensor] = None,
        patch_time0: Optional[torch.Tensor] = None,
        horizon_chunk_emb: Optional[torch.Tensor] = None,
        return_cross_attn_weights: bool = False,
    ):
        BV, _, H, W = x.shape
        self._diag_cross_attn_weights = None

        x_p, pad_h, pad_w = self._pad_to_patch(x, self.patch_size)
        cond_p, _, _ = self._pad_to_patch(cond, self.cond_patch_size)

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

        if self.use_patch_abs_embedding:
            if patch_coarse_bin is None or patch_time0 is None:
                raise ValueError(
                    "patch_coarse_bin and patch_time0 are required when "
                    "use_patch_abs_embedding=True"
                )
            if patch_coarse_bin.shape[0] != BV or patch_time0.shape[0] != BV:
                raise ValueError(
                    f"patch location batch mismatch: bins={tuple(patch_coarse_bin.shape)} "
                    f"time0={tuple(patch_time0.shape)} BV={BV}"
                )
            # Crop-level absolute ids broadcast over all target tokens.
            abs_emb = (
                self.coarse_bin_embed(patch_coarse_bin.long())
                + self.horizon_time_embed(patch_time0.long())
            ).unsqueeze(1)
            x_tok = x_tok + abs_emb

        tokens = torch.cat([c_tok, x_tok], dim=1)  # (BV, Nc + Nx, D)

        if self.variate_embed is not None:
            if variate_indices is None:
                raise ValueError("variate_indices are required when variate embeddings are enabled.")
            if variate_indices.shape[0] != BV:
                raise ValueError(f"variate_indices batch {variate_indices.shape[0]} != BV {BV}")
            v_emb = self.variate_embed(variate_indices.long()).unsqueeze(1)
            tokens = tokens + v_emb

        if t.shape[0] != BV:
            raise ValueError(f"timestep batch {t.shape[0]} != BV {BV}")
        t_emb = self.t_embed(_timestep_embedding(t, self.embed_dim))  # (BV, D)
        if self.scale_embed is not None:
            if scale_indices is None:
                raise ValueError("scale_indices are required when scale embeddings are enabled.")
            if scale_indices.shape[0] != BV:
                raise ValueError(f"scale_indices batch {scale_indices.shape[0]} != BV {BV}")
            t_emb = t_emb + self.scale_embed(scale_indices.long())
        if self.use_horizon_chunk_embedding:
            if horizon_chunk_emb is None:
                raise ValueError(
                    "horizon_chunk_emb is required when use_horizon_chunk_embedding=True"
                )
            if horizon_chunk_emb.shape != (BV, self.embed_dim):
                raise ValueError(
                    f"horizon_chunk_emb must be {(BV, self.embed_dim)}, "
                    f"got {tuple(horizon_chunk_emb.shape)}"
                )
            t_emb = t_emb + horizon_chunk_emb
        elif horizon_chunk_emb is not None:
            raise ValueError(
                "horizon_chunk_emb was passed but this DiT has no horizon-chunk embed"
            )

        ctx_proj: Optional[torch.Tensor] = None
        if encoder_hidden_states is not None:
            if context_window_indices is not None and context_window_indices.shape != (BV,):
                raise ValueError(
                    "context_window_indices must have one entry per DiT row, got "
                    f"{tuple(context_window_indices.shape)} for BV={BV}"
                )
            if context_window_indices is None:
                if encoder_hidden_states.shape[0] != BV:
                    raise ValueError(
                        "encoder_hidden_states batch "
                        f"{encoder_hidden_states.shape[0]} != DiT batch {BV}"
                    )
            ctx_proj = self.ctx_norm(self.ctx_proj(encoder_hidden_states))  # (BV, V, D)

        if horizon_chunk_emb is not None:
            extra = horizon_chunk_emb.unsqueeze(1)
            if ctx_proj is None:
                ctx_proj = extra
            elif context_window_indices is not None:
                b_ctx = ctx_proj.shape[0]
                window_extra = extra.new_zeros(b_ctx, extra.shape[-1])
                window_extra[context_window_indices] = extra.squeeze(1)
                ctx_proj = torch.cat([ctx_proj, window_extra.unsqueeze(1)], dim=1)
            else:
                if extra.shape[0] != ctx_proj.shape[0]:
                    raise ValueError(
                        f"horizon_chunk_emb batch {extra.shape[0]} != ctx batch {ctx_proj.shape[0]}"
                    )
                ctx_proj = torch.cat([ctx_proj, extra], dim=1)
            if token_variate_ids is not None:
                token_variate_ids = torch.cat(
                    [token_variate_ids, token_variate_ids.new_full((1,), -1)], dim=0
                )

        for i, block in enumerate(self.blocks):
            if i == self.bottleneck_idx:
                if self.gradient_checkpointing and self.training:
                    if context_window_indices is None:
                        tokens = checkpoint(
                            block,
                            tokens,
                            t_emb,
                            ctx_proj,
                            scale_indices,
                            variate_indices,
                            use_reentrant=False,
                        )
                    else:
                        cross_block = block

                        def checkpointed_cross_block(
                            block_tokens,
                            block_t_emb,
                            block_ctx,
                            block_scale,
                            block_variate,
                            block_context_windows,
                        ):
                            # checkpoint replays this closure during backward,
                            # after the outer loop has advanced to a different
                            # block. Capture the bottleneck block explicitly.
                            return cross_block(
                                block_tokens,
                                block_t_emb,
                                block_ctx,
                                block_scale,
                                block_variate,
                                token_variate_ids=token_variate_ids,
                                context_window_indices=block_context_windows,
                            )

                        tokens = checkpoint(
                            checkpointed_cross_block,
                            tokens,
                            t_emb,
                            ctx_proj,
                            scale_indices,
                            variate_indices,
                            context_window_indices,
                            use_reentrant=False,
                        )
                elif return_cross_attn_weights:
                    tokens, attn_w = block(
                        tokens, t_emb, ctx_proj, scale_indices, variate_indices,
                        token_variate_ids=token_variate_ids,
                        context_window_indices=context_window_indices,
                        return_attn_weights=True,
                    )
                    self._diag_cross_attn_weights = attn_w
                else:
                    tokens = block(
                        tokens, t_emb, ctx_proj, scale_indices, variate_indices,
                        token_variate_ids=token_variate_ids,
                        context_window_indices=context_window_indices,
                    )
            else:
                if self.gradient_checkpointing and self.training:
                    tokens = checkpoint(block, tokens, t_emb, use_reentrant=False)
                else:
                    tokens = block(tokens, t_emb)

        x_out = tokens[:, Nc:]  # (BV, Nx, D), drop cond slots
        shift, scale = self.final_adaLN(t_emb).chunk(2, dim=-1)
        x_out = _modulate(self.final_norm(x_out), shift, scale)
        out = self._unpatchify(x_out, gh, gw)

        if pad_h or pad_w:
            out = out[:, :, :H, :W]
        return out
