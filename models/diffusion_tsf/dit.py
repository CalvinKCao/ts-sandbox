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
        ctx_key_padding_mask: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False,
    ):
        B, N, C = x.shape
        ctx_batch, M, _ = ctx.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(ctx).reshape(
            ctx_batch, M, 2, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        pad_bias = None
        if ctx_key_padding_mask is not None:
            if ctx_key_padding_mask.shape != (ctx_batch, M):
                raise ValueError(
                    f"ctx_key_padding_mask must be {(ctx_batch, M)}, "
                    f"got {tuple(ctx_key_padding_mask.shape)}"
                )
            pad_bias = torch.zeros(
                ctx_batch, M, device=ctx.device, dtype=q.dtype,
            )
            pad_bias = pad_bias.masked_fill(ctx_key_padding_mask, torch.finfo(q.dtype).min)
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
            if pad_bias is not None:
                pad_rows = pad_bias.index_select(0, context_window_indices)[:, None, None, :]
                bias_rows = pad_rows if bias_rows is None else bias_rows + pad_rows
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
        if pad_bias is not None:
            attn_bias = pad_bias if attn_bias is None else attn_bias + pad_bias
        if return_attn_weights:
            scale = self.head_dim ** -0.5
            attn_logits = torch.matmul(q, k.transpose(-2, -1)) * scale
            if attn_bias is not None:
                attn_logits = attn_logits + attn_bias[:, None, None, :]
            attn_weights = torch.softmax(attn_logits, dim=-1)
            out = torch.matmul(attn_weights, v)
            out = out.transpose(1, 2).reshape(B, N, C)
            out = self.proj(out)
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


def _softmax_self_attn_layer(layer_idx: int, use_linear_attn: bool) -> bool:
    """3:1 hybrid: softmax when (layer_idx + 1) % 4 == 0; else linear if flag on."""
    if not use_linear_attn:
        return True
    return (layer_idx + 1) % 4 == 0


def _use_rational_cross_attn(
    layer_idx: int,
    use_linear_cross_attn: bool,
    use_linear_attn: bool,
) -> bool:
    if not use_linear_cross_attn:
        return False
    if not use_linear_attn:
        return True
    return not _softmax_self_attn_layer(layer_idx, True)


def _gdn_scan_sequential(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Left-to-right gated delta-rule reference. Python loop over N; tests only.

    q,k,v: (B, H, N, D); decay, beta: (B, H, N) → y: (B, H, N, D)
    S_i = decay * S_{i-1} + k_i * beta * (v_i - k_i^T * decay * S_{i-1})
    y_i = q_i^T S_i
    """
    B, H, N, D = q.shape
    if k.shape != q.shape or v.shape != q.shape:
        raise ValueError(f"q/k/v shape mismatch: {tuple(q.shape)} {tuple(k.shape)} {tuple(v.shape)}")
    if decay.shape != (B, H, N) or beta.shape != (B, H, N):
        raise ValueError(
            f"decay/beta must be {(B, H, N)}, got {tuple(decay.shape)} {tuple(beta.shape)}"
        )
    S = q.new_zeros(B, H, D, D)
    y = q.new_empty(B, H, N, D)
    for i in range(N):
        dec = decay[:, :, i]
        b = beta[:, :, i]
        ki = k[:, :, i]
        vi = v[:, :, i]
        dS = S * dec[:, :, None, None]
        kT_S = torch.einsum("bhd,bhde->bhe", ki, dS)
        S = dS + torch.einsum("bhd,bhe->bhde", ki * b[:, :, None], vi - kT_S)
        y[:, :, i] = torch.einsum("bhd,bhde->bhe", q[:, :, i], S)
    return y


@torch.compiler.disable
def _gdn_fused_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """fp32 chunked parallel GDN scan (Hillis–Steele over C=16, n_chunks
    state carries). Not a Python loop over N. Electricity-scale N keeps
    this path. compiler-disable so outer DiT torch.compile does not unroll N.
    Extra activation vs a sequential loop, and tiny drift vs left-to-right,
    are acceptable.
    """
    orig_dtype = q.dtype
    y = _gdn_chunked_scan(
        q.float(), k.float(), v.float(), decay.float(), beta.float(),
        chunk_size=16,
    )
    return y.to(dtype=orig_dtype)


def _inclusive_scan_ab(A: torch.Tensor, B: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Hillis–Steele inclusive scan of S_i = A_i S_{i-1} + B_i along dim=-3.

    combine(left, right) = (A_r @ A_l, A_r @ B_l + B_r). Loop is log2(C), not N.
    """
    C = A.shape[-3]
    if B.shape != A.shape:
        raise ValueError(f"A/B shape mismatch: {tuple(A.shape)} {tuple(B.shape)}")
    D = A.shape[-1]
    if A.shape[-1] != A.shape[-2]:
        raise ValueError(f"A must be square on last two dims, got {tuple(A.shape)}")
    n = 1 if C <= 1 else 1 << (C - 1).bit_length()
    if n != C:
        pad = n - C
        I = torch.eye(D, device=A.device, dtype=A.dtype)
        A = F.pad(A, (0, 0, 0, 0, 0, pad))
        B = F.pad(B, (0, 0, 0, 0, 0, pad))
        A[..., C:, :, :] = I
        B[..., C:, :, :] = 0
    A = A.clone()
    B = B.clone()
    offset = 1
    while offset < n:
        A_old = A.clone()
        B_old = B.clone()
        A_l = A_old[..., :-offset, :, :]
        B_l = B_old[..., :-offset, :, :]
        A_r = A_old[..., offset:, :, :]
        B_r = B_old[..., offset:, :, :]
        A[..., offset:, :, :] = A_r @ A_l
        B[..., offset:, :, :] = torch.matmul(A_r, B_l) + B_r
        offset *= 2
    return A[..., :C, :, :], B[..., :C, :, :]


def _gdn_chunked_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 16,
) -> torch.Tensor:
    """Chunked parallel GDN scan. Intra-chunk is a log-C associative scan, not a
    Python loop over sequence length N. Inter-chunk is n_chunks = ceil(N/C)
    state carries (electricity-scale N keeps this path; do not special-case
    weather). Extra activation memory vs a sequential loop, and tiny drift vs
    left-to-right recurrence, are acceptable for this ablation.
    """
    if chunk_size < 1 or (chunk_size & (chunk_size - 1)) != 0:
        raise ValueError(f"chunk_size must be a power of 2, got {chunk_size}")
    B, H, N, D = q.shape
    if k.shape != q.shape or v.shape != q.shape:
        raise ValueError(f"q/k/v shape mismatch: {tuple(q.shape)} {tuple(k.shape)} {tuple(v.shape)}")
    if decay.shape != (B, H, N) or beta.shape != (B, H, N):
        raise ValueError(
            f"decay/beta must be {(B, H, N)}, got {tuple(decay.shape)} {tuple(beta.shape)}"
        )
    pad = (chunk_size - N % chunk_size) % chunk_size
    if pad:
        q = F.pad(q, (0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad))
        beta = F.pad(beta, (0, pad))
        decay = F.pad(decay, (0, pad), value=1.0)
    Np = q.shape[2]
    n_chunks = Np // chunk_size
    q = q.view(B, H, n_chunks, chunk_size, D)
    k = k.view(B, H, n_chunks, chunk_size, D)
    v = v.view(B, H, n_chunks, chunk_size, D)
    decay = decay.view(B, H, n_chunks, chunk_size)
    beta = beta.view(B, H, n_chunks, chunk_size)
    I = torch.eye(D, device=q.device, dtype=q.dtype)
    k_outer = k.unsqueeze(-1) * k.unsqueeze(-2)
    A = decay[..., None, None] * (I - beta[..., None, None] * k_outer)
    Bmat = (beta[..., None] * k).unsqueeze(-1) * v.unsqueeze(-2)
    A_scan, B_scan = _inclusive_scan_ab(A, Bmat)
    S = q.new_zeros(B, H, D, D)
    ys = []
    for ci in range(n_chunks):
        A_c = A_scan[:, :, ci]
        B_c = B_scan[:, :, ci]
        S_pos = torch.matmul(A_c, S.unsqueeze(2)) + B_c
        ys.append(torch.einsum("bhcd,bhcdf->bhcf", q[:, :, ci], S_pos))
        S = S_pos[:, :, -1]
    y = torch.cat(ys, dim=2)
    return y[:, :, :N]


def _flip_target_seq(x: torch.Tensor, num_cond_tokens: int, seq_dim: int = 2) -> torch.Tensor:
    """Flip only the target slice; prefix is never reversed (DeltaFlow-P)."""
    n = x.shape[seq_dim]
    if num_cond_tokens < 0 or num_cond_tokens > n:
        raise ValueError(f"num_cond_tokens={num_cond_tokens} out of range for N={n}")
    pre = x.narrow(seq_dim, 0, num_cond_tokens)
    tgt = x.narrow(seq_dim, num_cond_tokens, n - num_cond_tokens)
    return torch.cat([pre, torch.flip(tgt, dims=[seq_dim])], dim=seq_dim)


class MixFFN(nn.Module):
    """SegFormer Mix-FFN (arXiv:2105.15203): 1x1 expand → DWConv 3x3 → SiLU → 1x1.

    Applied on independent real 2D grids (cond vs x). Shared weights. Do not
    reshape concatenated cond|x into one fake grid (ghost neighbors at the join).
    """

    def __init__(self, dim: int, expand: int = 4):
        super().__init__()
        if expand < 1:
            raise ValueError(f"MixFFN expand must be >= 1, got {expand}")
        hidden = dim * expand
        self.fc1 = nn.Linear(dim, hidden)
        self.dwconv = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def _grid(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        B, N, _ = x.shape
        if N != height * width:
            raise ValueError(
                f"MixFFN slice length {N} != H*W={height}*{width}={height * width}"
            )
        h = self.fc1(x)
        hidden = h.shape[-1]
        h = h.transpose(1, 2).reshape(B, hidden, height, width)
        h = F.silu(self.dwconv(h))
        h = h.flatten(2).transpose(1, 2)
        return self.fc2(h)

    def forward(
        self,
        x: torch.Tensor,
        num_cond_tokens: int,
        shape_cond: Tuple[int, int],
        shape_x: Tuple[int, int],
    ) -> torch.Tensor:
        cond = x[:, :num_cond_tokens]
        xt = x[:, num_cond_tokens:]
        hc, wc = int(shape_cond[0]), int(shape_cond[1])
        hx, wx = int(shape_x[0]), int(shape_x[1])
        out_c = self._grid(cond, hc, wc)
        out_x = self._grid(xt, hx, wx)
        return torch.cat([out_c, out_x], dim=1)


class NaLaAdaptiveDeltaFlowP(nn.Module):
    """NaLa query-norm + prefix-preserving bidirectional gated delta scan."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} not divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.a_proj = nn.Linear(dim, num_heads, bias=True)
        self.b_proj = nn.Linear(dim, num_heads, bias=True)
        self.g_proj = nn.Linear(dim, dim, bias=True)
        self.u_a = nn.Linear(dim, num_heads, bias=True)
        self.u_b = nn.Linear(dim, num_heads, bias=True)
        nn.init.zeros_(self.u_a.weight)
        nn.init.zeros_(self.u_a.bias)
        nn.init.zeros_(self.u_b.weight)
        nn.init.zeros_(self.u_b.bias)
        self.out_rms = nn.RMSNorm(dim, eps=1e-6)
        self.out_proj = nn.Linear(dim, dim)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        return x.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        B, H, N, D = x.shape
        return x.permute(0, 2, 1, 3).reshape(B, N, H * D)

    def _scan(self, q, k, v, decay, beta) -> torch.Tensor:
        return _gdn_fused_scan(q, k, v, decay, beta)

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        num_cond_tokens: int,
        shape_cond: Tuple[int, int],
        shape_x: Tuple[int, int],
    ) -> torch.Tensor:
        B, N, C = x.shape
        hc, wc = int(shape_cond[0]), int(shape_cond[1])
        hx, wx = int(shape_x[0]), int(shape_x[1])
        if num_cond_tokens != hc * wc:
            raise ValueError(
                f"num_cond_tokens={num_cond_tokens} != shape_cond H*W={hc}*{wc}"
            )
        if N - num_cond_tokens != hx * wx:
            raise ValueError(
                f"x tokens {N - num_cond_tokens} != shape_x H*W={hx}*{wx}"
            )
        if t_emb.shape != (B, C):
            raise ValueError(f"t_emb must be {(B, C)}, got {tuple(t_emb.shape)}")
        q = self._heads(self.q_proj(x))
        k = F.normalize(self._heads(self.k_proj(x)), dim=-1)
        v = self._heads(self.v_proj(x))
        g = self.g_proj(x)
        a = self.a_proj(x).permute(0, 2, 1)
        b = self.b_proj(x).permute(0, 2, 1)
        u_a = self.u_a(t_emb).unsqueeze(-1)
        u_b = self.u_b(t_emb).unsqueeze(-1)
        decay = torch.exp(-F.softplus(a + u_a))
        beta = torch.sigmoid(b + u_b)
        q_rms = torch.sqrt(q.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        q_hat = q / q_rms
        y_fwd = self._scan(q_hat, k, v, decay, beta)
        q_b = _flip_target_seq(q_hat, num_cond_tokens)
        k_b = _flip_target_seq(k, num_cond_tokens)
        v_b = _flip_target_seq(v, num_cond_tokens)
        decay_b = _flip_target_seq(decay, num_cond_tokens)
        beta_b = _flip_target_seq(beta, num_cond_tokens)
        y_b_scanned = self._scan(q_b, k_b, v_b, decay_b, beta_b)
        y_b_tgt = torch.flip(y_b_scanned[:, :, num_cond_tokens:], dims=[2])
        y_bwd = torch.cat([y_b_scanned[:, :, :num_cond_tokens], y_b_tgt], dim=2)
        y = 0.5 * (y_fwd + y_bwd)
        y = y * q_rms
        y = self._merge(y)
        y = self.out_rms(y)
        y = y * F.silu(g)
        return self.out_proj(y)


class RationalKernelCrossAttention(nn.Module):
    """Cross-attention with phi(z)=z^2+z+1. Fail-fast on weights / nonzero bias."""

    def __init__(self, dim: int, num_heads: int, drop: float = 0.0):
        super().__init__()
        if drop != 0.0:
            raise ValueError("RationalKernelCrossAttention does not support dropout")
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, 2 * dim, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.scale_factor = nn.Parameter(torch.tensor(0.01))

    @staticmethod
    def _phi(z: torch.Tensor) -> torch.Tensor:
        return z * z + z + 1

    def forward(
        self,
        x: torch.Tensor,
        ctx: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        context_window_indices: Optional[torch.Tensor] = None,
        ctx_key_padding_mask: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False,
    ):
        if return_attn_weights:
            raise ValueError("RationalKernelCrossAttention does not support return_attn_weights")
        if attn_bias is not None and bool(attn_bias.ne(0).any()):
            raise ValueError("RationalKernelCrossAttention does not support nonzero attn_bias")
        B, N, C = x.shape
        ctx_batch, M, _ = ctx.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(ctx).reshape(
            ctx_batch, M, 2, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        pad_mask = None
        if ctx_key_padding_mask is not None:
            if ctx_key_padding_mask.shape != (ctx_batch, M):
                raise ValueError(
                    f"ctx_key_padding_mask must be {(ctx_batch, M)}, "
                    f"got {tuple(ctx_key_padding_mask.shape)}"
                )
            pad_mask = ctx_key_padding_mask
        if context_window_indices is not None:
            if context_window_indices.shape != (B,):
                raise ValueError(
                    "context_window_indices must have one entry per query row, got "
                    f"{tuple(context_window_indices.shape)} for B={B}"
                )
            k = k.index_select(0, context_window_indices)
            v = v.index_select(0, context_window_indices)
            if pad_mask is not None:
                pad_mask = pad_mask.index_select(0, context_window_indices)
        elif ctx_batch != B:
            raise ValueError(f"ctx batch {ctx_batch} != query batch {B}")
        q = self._phi(q.float())
        k = self._phi(k.float())
        v = v.float()
        if pad_mask is not None:
            fill = pad_mask[:, None, :, None]
            k = k.masked_fill(fill, 0)
            v = v.masked_fill(fill, 0)
        kv = torch.einsum("bhmd,bhmv->bhdv", k, v)
        ksum = k.sum(dim=2)
        num = torch.einsum("bhnd,bhdv->bhnv", q, kv)
        den = torch.einsum("bhnd,bhd->bhn", q, ksum).unsqueeze(-1)
        den = den.clamp_min(1e-6)
        out = self.scale_factor.float() * (num / den)
        out = out.to(dtype=x.dtype)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class _DiTBlock(nn.Module):
    """Standard DiT block: AdaLN-Zero(self-attn) + AdaLN-Zero(MLP)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        drop: float,
        use_linear_self: bool = False,
        use_mix_ffn: bool = False,
    ):
        super().__init__()
        self.use_linear_self = bool(use_linear_self)
        self.use_mix_ffn = bool(use_mix_ffn)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        if use_linear_self:
            self.attn = NaLaAdaptiveDeltaFlowP(dim, num_heads)
        else:
            self.attn = _SelfAttention(dim, num_heads, drop=drop)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        if use_mix_ffn:
            self.mlp = MixFFN(dim, expand=4)
        else:
            self.mlp = _MLP(dim, int(dim * mlp_ratio), drop=drop)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        num_cond_tokens: Optional[int] = None,
        shape_cond: Optional[Tuple[int, int]] = None,
        shape_x: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        s1, sc1, g1, s2, sc2, g2 = self.adaLN(c).chunk(6, dim=-1)
        h1 = _modulate(self.norm1(x), s1, sc1)
        if self.use_linear_self:
            if num_cond_tokens is None or shape_cond is None or shape_x is None:
                raise ValueError("linear self-attn requires num_cond_tokens, shape_cond, shape_x")
            attn_out = self.attn(h1, c, num_cond_tokens, shape_cond, shape_x)
        else:
            attn_out = self.attn(h1)
        x = x + g1.unsqueeze(1) * attn_out
        h2 = _modulate(self.norm2(x), s2, sc2)
        if self.use_mix_ffn:
            if num_cond_tokens is None or shape_cond is None or shape_x is None:
                raise ValueError("MixFFN requires num_cond_tokens, shape_cond, shape_x")
            mlp_out = self.mlp(h2, num_cond_tokens, shape_cond, shape_x)
        else:
            mlp_out = self.mlp(h2)
        x = x + g2.unsqueeze(1) * mlp_out
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
        use_linear_self: bool = False,
        use_rational_cross: bool = False,
        use_mix_ffn: bool = False,
    ):
        super().__init__()
        self.enable_cross_scale_attention = enable_cross_scale_attention
        self.target_context_bias = float(target_context_bias)
        self.use_linear_self = bool(use_linear_self)
        self.use_rational_cross = bool(use_rational_cross)
        self.use_mix_ffn = bool(use_mix_ffn)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        if use_linear_self:
            self.self_attn = NaLaAdaptiveDeltaFlowP(dim, num_heads)
        else:
            self.self_attn = _SelfAttention(dim, num_heads, drop=drop)
        self.norm_x = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        if use_rational_cross:
            self.cross_attn = RationalKernelCrossAttention(dim, num_heads, drop=drop)
        else:
            self.cross_attn = _CrossAttention(dim, num_heads, drop=drop)
        if enable_cross_scale_attention:
            self.norm_s = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.scale_attn = _CrossAttention(dim, num_heads, drop=drop)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        if use_mix_ffn:
            self.mlp = MixFFN(dim, expand=4)
        else:
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
        ctx_key_padding_mask: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False,
        num_cond_tokens: Optional[int] = None,
        shape_cond: Optional[Tuple[int, int]] = None,
        shape_x: Optional[Tuple[int, int]] = None,
    ):
        mods = self.adaLN(c).chunk(12 if self.enable_cross_scale_attention else 9, dim=-1)
        if self.enable_cross_scale_attention:
            s1, sc1, g1, sx, scx, gx, ss, scs, gs, s2, sc2, g2 = mods
        else:
            s1, sc1, g1, sx, scx, gx, s2, sc2, g2 = mods
        h1 = _modulate(self.norm1(x), s1, sc1)
        if self.use_linear_self:
            if num_cond_tokens is None or shape_cond is None or shape_x is None:
                raise ValueError("linear self-attn requires num_cond_tokens, shape_cond, shape_x")
            self_out = self.self_attn(h1, c, num_cond_tokens, shape_cond, shape_x)
        else:
            self_out = self.self_attn(h1)
        x = x + g1.unsqueeze(1) * self_out
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
                    ctx_key_padding_mask=ctx_key_padding_mask,
                    return_attn_weights=True,
                )
            else:
                cross_out = self.cross_attn(
                    cross_in,
                    ctx,
                    attn_bias=attn_bias,
                    context_window_indices=context_window_indices,
                    ctx_key_padding_mask=ctx_key_padding_mask,
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
        h2 = _modulate(self.norm2(x), s2, sc2)
        if self.use_mix_ffn:
            if num_cond_tokens is None or shape_cond is None or shape_x is None:
                raise ValueError("MixFFN requires num_cond_tokens, shape_cond, shape_x")
            mlp_out = self.mlp(h2, num_cond_tokens, shape_cond, shape_x)
        else:
            mlp_out = self.mlp(h2)
        x = x + g2.unsqueeze(1) * mlp_out
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
        use_linear_attn: bool = False,
        use_linear_cross_attn: bool = False,
        use_attn_res: bool = False,
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
        self.use_linear_attn = bool(use_linear_attn)
        self.use_linear_cross_attn = bool(use_linear_cross_attn)
        self.use_attn_res = bool(use_attn_res)
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
        self.softmax_self_layers = [
            _softmax_self_attn_layer(i, self.use_linear_attn) for i in range(depth)
        ]
        self.blocks = nn.ModuleList()
        for i in range(depth):
            lin_self = not self.softmax_self_layers[i]
            mix = self.use_linear_attn
            if i == self.bottleneck_idx:
                self.blocks.append(
                    _DiTCrossAttnBlock(
                        embed_dim,
                        num_heads,
                        mlp_ratio,
                        dropout,
                        enable_cross_scale_attention=enable_cross_scale_attention,
                        target_context_bias=cross_variate_context_bias,
                        use_linear_self=lin_self,
                        use_rational_cross=_use_rational_cross_attn(
                            i, self.use_linear_cross_attn, self.use_linear_attn,
                        ),
                        use_mix_ffn=mix,
                    )
                )
            else:
                self.blocks.append(
                    _DiTBlock(
                        embed_dim,
                        num_heads,
                        mlp_ratio,
                        dropout,
                        use_linear_self=lin_self,
                        use_mix_ffn=mix,
                    )
                )

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
        ctx_key_padding_mask: Optional[torch.Tensor] = None,
        return_cross_attn_weights: bool = False,
    ):
        BV, _, H, W = x.shape
        self._diag_cross_attn_weights = None

        x_p, pad_h, pad_w = self._pad_to_patch(x, self.patch_size)
        cond_p, _, _ = self._pad_to_patch(cond, self.cond_patch_size)

        x_tok, gh, gw = self._patchify(x_p, self.x_embed)
        c_tok, c_gh, c_gw = self._patchify(cond_p, self.cond_embed)

        Nx, Nc = x_tok.shape[1], c_tok.shape[1]
        if Nx != gh * gw:
            raise ValueError(f"x tokens {Nx} != gh*gw={gh}*{gw}")
        if Nc != c_gh * c_gw:
            raise ValueError(f"cond tokens {Nc} != c_gh*c_gw={c_gh}*{c_gw}")
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

        geom_kw = {}
        if self.use_linear_attn:
            geom_kw = {
                "num_cond_tokens": Nc,
                "shape_cond": (c_gh, c_gw),
                "shape_x": (gh, gw),
            }

        attn_res_src = None
        for i, block in enumerate(self.blocks):
            if i == self.bottleneck_idx:
                if self.gradient_checkpointing and self.training:
                    if context_window_indices is None:
                        if geom_kw:
                            tokens = checkpoint(
                                block,
                                tokens,
                                t_emb,
                                ctx_proj,
                                scale_indices,
                                variate_indices,
                                None,
                                None,
                                None,
                                False,
                                geom_kw["num_cond_tokens"],
                                geom_kw["shape_cond"],
                                geom_kw["shape_x"],
                                use_reentrant=False,
                            )
                        else:
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
                        lin_nc = geom_kw.get("num_cond_tokens")
                        lin_sc = geom_kw.get("shape_cond")
                        lin_sx = geom_kw.get("shape_x")

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
                                ctx_key_padding_mask=ctx_key_padding_mask,
                                num_cond_tokens=lin_nc,
                                shape_cond=lin_sc,
                                shape_x=lin_sx,
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
                        ctx_key_padding_mask=ctx_key_padding_mask,
                        return_attn_weights=True,
                        **geom_kw,
                    )
                    self._diag_cross_attn_weights = attn_w
                else:
                    tokens = block(
                        tokens, t_emb, ctx_proj, scale_indices, variate_indices,
                        token_variate_ids=token_variate_ids,
                        context_window_indices=context_window_indices,
                        ctx_key_padding_mask=ctx_key_padding_mask,
                        **geom_kw,
                    )
            else:
                if self.gradient_checkpointing and self.training:
                    if geom_kw:
                        tokens = checkpoint(
                            block,
                            tokens,
                            t_emb,
                            geom_kw["num_cond_tokens"],
                            geom_kw["shape_cond"],
                            geom_kw["shape_x"],
                            use_reentrant=False,
                        )
                    else:
                        tokens = checkpoint(block, tokens, t_emb, use_reentrant=False)
                else:
                    tokens = block(tokens, t_emb, **geom_kw)
            if self.use_attn_res:
                if self.softmax_self_layers[i]:
                    attn_res_src = tokens
                elif attn_res_src is not None:
                    tokens = tokens + attn_res_src

        x_out = tokens[:, Nc:]  # (BV, Nx, D), drop cond slots
        shift, scale = self.final_adaLN(t_emb).chunk(2, dim=-1)
        x_out = _modulate(self.final_norm(x_out), shift, scale)
        out = self._unpatchify(x_out, gh, gw)

        if pad_h or pad_w:
            out = out[:, :, :H, :W]
        return out
