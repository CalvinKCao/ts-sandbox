"""Past-anchored tie-aware ordinal normalization (no window/instance norm).

Each variate builds a sorted unique-value ladder from the lookback only.
Tied values share one rank/bin. Values map to [-max_scale, max_scale] by rank.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class OrdinalLadder:
    """Per-(batch, variate) past uniquified values, padded to n_unique_max."""

    values: torch.Tensor  # (B, V, K)
    n_unique: torch.Tensor  # (B, V) int64
    max_scale: float
    tie_atol: float


def _unique_sorted_1d(x: torch.Tensor, tie_atol: float) -> Tuple[torch.Tensor, int]:
    """Sorted unique values with atol tie merge. x: (T,)"""
    if x.numel() == 0:
        return x.new_zeros(0), 0
    xs, _ = torch.sort(x.reshape(-1))
    groups = [xs[0].item()]
    for v in xs[1:].tolist():
        if abs(v - groups[-1]) > tie_atol:
            groups.append(v)
    uniq = x.new_tensor(groups)
    return uniq, int(uniq.numel())


def _value_to_rank(values: torch.Tensor, x: torch.Tensor, tie_atol: float) -> torch.Tensor:
    """Map x to integer rank using past uniquified ladder."""
    k = values.shape[0]
    if k <= 1:
        return torch.zeros_like(x, dtype=torch.long)
    ranks = torch.zeros_like(x, dtype=torch.long)
    for j in range(k):
        if j == k - 1:
            mask = x >= (values[j] - tie_atol)
        else:
            mid = (values[j] + values[j + 1]) * 0.5
            if j == 0:
                mask = x < mid
            else:
                mask = (x >= (values[j] - tie_atol)) & (x < mid)
        ranks = torch.where(mask, torch.full_like(ranks, j), ranks)
    return ranks.clamp(0, k - 1)


def _rank_to_ordinal(rank: torch.Tensor, n_unique: int, max_scale: float) -> torch.Tensor:
    if n_unique <= 1:
        return torch.zeros_like(rank, dtype=torch.float32)
    denom = float(max(n_unique - 1, 1))
    return (2.0 * max_scale * rank.to(torch.float32) / denom) - max_scale


def build_ladder_from_past(
    past: torch.Tensor,
    *,
    max_scale: float,
    tie_atol: float,
) -> OrdinalLadder:
    """Build padded ladder from lookback past (B, V, L)."""
    if past.dim() == 2:
        past = past.unsqueeze(1)
    b, v, _l = past.shape
    n_unique_list = []
    uniq_rows = []
    k_max = 1
    for bi in range(b):
        for vi in range(v):
            u, k = _unique_sorted_1d(past[bi, vi], tie_atol)
            n_unique_list.append(k)
            k_max = max(k_max, k)
            uniq_rows.append(u)
    padded = past.new_zeros(b, v, k_max)
    n_unique = past.new_zeros(b, v, dtype=torch.long)
    idx = 0
    for bi in range(b):
        for vi in range(v):
            u = uniq_rows[idx]
            k = len(u)
            if k > 0:
                padded[bi, vi, :k] = u
            n_unique[bi, vi] = max(k, 1)
            idx += 1
    return OrdinalLadder(values=padded, n_unique=n_unique, max_scale=float(max_scale), tie_atol=float(tie_atol))


def encode_with_ladder(
    x: torch.Tensor,
    ladder: OrdinalLadder,
) -> torch.Tensor:
    """Map values x (B,V,T) to ordinal-normalized space using past ladder."""
    if x.dim() == 2:
        x = x.unsqueeze(1)
    b, v, t = x.shape
    out = x.new_zeros(b, v, t)
    ms = ladder.max_scale
    atol = ladder.tie_atol
    for bi in range(b):
        for vi in range(v):
            k = int(ladder.n_unique[bi, vi].item())
            uniq = ladder.values[bi, vi, :k]
            ranks = _value_to_rank(uniq, x[bi, vi], atol)
            out[bi, vi] = _rank_to_ordinal(ranks, k, ms)
    return out


def decode_with_ladder(
    ordinal: torch.Tensor,
    ladder: OrdinalLadder,
) -> torch.Tensor:
    """Inverse ordinal map back to value space (global z-score)."""
    if ordinal.dim() == 2:
        ordinal = ordinal.unsqueeze(1)
    b, v, t = ordinal.shape
    out = ordinal.new_zeros(b, v, t)
    ms = ladder.max_scale
    atol = ladder.tie_atol
    for bi in range(b):
        for vi in range(v):
            k = int(ladder.n_unique[bi, vi].item())
            uniq = ladder.values[bi, vi, :k]
            if k <= 0:
                continue
            if k == 1:
                out[bi, vi] = uniq[0]
                continue
            cont_rank = (ordinal[bi, vi] + ms) / (2.0 * ms) * float(k - 1)
            j = cont_rank.round().clamp(0, k - 1).long()
            out[bi, vi] = uniq[j]
            # atol tie groups already collapsed in uniq
            _ = atol
    return out


def ordinal_encode(
    past: torch.Tensor,
    future: Optional[torch.Tensor],
    *,
    max_scale: float,
    tie_atol: float = 1e-6,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], OrdinalLadder]:
    ladder = build_ladder_from_past(past, max_scale=max_scale, tie_atol=tie_atol)
    past_ord = encode_with_ladder(past, ladder)
    fut_ord = encode_with_ladder(future, ladder) if future is not None else None
    return past_ord, fut_ord, ladder


def ordinal_decode(
    past_ord: torch.Tensor,
    future_ord: Optional[torch.Tensor],
    ladder: OrdinalLadder,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    past_val = decode_with_ladder(past_ord, ladder)
    fut_val = decode_with_ladder(future_ord, ladder) if future_ord is not None else None
    return past_val, fut_val
