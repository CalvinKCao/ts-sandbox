"""Global training-set ordinal encoding (no window/instance norm).

Each variate builds a sorted unique-value ladder from the full training split.
Tied values share one rank. Values map to integer ranks 0..K-1 as floats.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch


@dataclass
class OrdinalLadder:
    """Global per-variate uniquified value ladder, padded to n_unique_max."""

    values: torch.Tensor  # (B, V, K)
    n_unique: torch.Tensor  # (B, V) int64
    tie_atol: float
    precomputed_ranks: Optional[torch.Tensor] = None  # (T, V) float32, optional

    def rank_max_per_variate(self) -> torch.Tensor:
        """Inclusive max rank index per variate (0 when only one unique value).

        Always uses row 0 so batch-expanded ladders (expand_batch) still broadcast
        as (V,) / (1, V, 1), not (B*V,).
        """
        return (self.n_unique[0] - 1).clamp_min(0)

    def z_envelope(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-variate train ladder min/max z-scores, shape (V,)."""
        v = self.values.shape[1]
        k_max = int(self.n_unique[0].max().item())
        mins = []
        maxs = []
        for vi in range(v):
            k = int(self.n_unique[0, vi].item())
            uniq = self.values[0, vi, :k]
            mins.append(uniq[0])
            maxs.append(uniq[k - 1])
        return torch.stack(mins), torch.stack(maxs)

    def expand_batch(self, batch_size: int) -> "OrdinalLadder":
        if self.values.shape[0] == batch_size:
            return self
        if self.values.shape[0] != 1:
            raise ValueError(f"cannot expand ladder batch {self.values.shape[0]} -> {batch_size}")
        return OrdinalLadder(
            values=self.values.expand(batch_size, -1, -1),
            n_unique=self.n_unique.expand(batch_size, -1),
            tie_atol=self.tie_atol,
            precomputed_ranks=self.precomputed_ranks,
        )


def _unique_sorted_1d_np(x: np.ndarray, tie_atol: float) -> Tuple[np.ndarray, int]:
    if x.size == 0:
        return np.zeros(0, dtype=np.float64), 0
    xs = np.sort(x.reshape(-1))
    groups = [float(xs[0])]
    for v in xs[1:]:
        if abs(float(v) - groups[-1]) > tie_atol:
            groups.append(float(v))
    uniq = np.asarray(groups, dtype=np.float64)
    return uniq, int(uniq.size)


def _value_to_rank_slow(values: torch.Tensor, x: torch.Tensor, tie_atol: float) -> torch.Tensor:
    """Reference implementation: full (N,K) distance matrix."""
    k = values.shape[0]
    if k <= 1:
        return torch.zeros_like(x, dtype=torch.long)
    flat = x.reshape(-1)
    dist = (flat.unsqueeze(-1) - values.unsqueeze(0)).abs()
    tie_hit = dist <= tie_atol
    has_tie = tie_hit.any(dim=-1)
    tie_rank = tie_hit.int().argmax(dim=-1)
    nearest = dist.argmin(dim=-1)
    ranks = torch.where(has_tie, tie_rank, nearest)
    return ranks.reshape(x.shape).clamp(0, k - 1)


def _value_to_rank(values: torch.Tensor, x: torch.Tensor, tie_atol: float) -> torch.Tensor:
    """Map x to ladder rung index via nearest uniquified value (no semi-infinite outer bins)."""
    k = values.shape[0]
    if k <= 1:
        return torch.zeros_like(x, dtype=torch.long)
    flat = x.reshape(-1).contiguous()
    idx_r = torch.searchsorted(values, flat)
    left_i = (idx_r - 1).clamp(0, k - 1)
    right_i = idx_r.clamp(0, k - 1)

    left_d = (flat - values[left_i]).abs()
    right_d = (flat - values[right_i]).abs()

    left_tie = left_d <= tie_atol
    right_tie = right_d <= tie_atol

    ranks = torch.where(
        left_tie,
        left_i,
        torch.where(
            right_tie,
            right_i,
            torch.where(left_d <= right_d, left_i, right_i),
        ),
    )
    return ranks.reshape(x.shape).clamp(0, k - 1)


def precompute_ranks_for_array(
    z_data: Union[torch.Tensor, np.ndarray],
    ladder: OrdinalLadder,
) -> torch.Tensor:
    """Precompute ordinal ranks for full z-scored array (T, V)."""
    if isinstance(z_data, np.ndarray):
        arr = torch.from_numpy(z_data.astype(np.float32, copy=False))
    else:
        arr = z_data.detach().cpu().float()
    if arr.ndim == 1:
        arr = arr.unsqueeze(-1)
    _t, v = arr.shape
    out = torch.zeros(_t, v, dtype=torch.float32)
    atol = ladder.tie_atol
    for vi in range(v):
        k = int(ladder.n_unique[0, vi].item())
        uniq = ladder.values[0, vi, :k]
        out[:, vi] = _value_to_rank(uniq, arr[:, vi], atol).float()
    return out


def build_global_ladder_from_training(
    train_data: Union[torch.Tensor, np.ndarray],
    *,
    tie_atol: float = 1e-6,
    precompute_ranks_for: Optional[Union[torch.Tensor, np.ndarray]] = None,
) -> OrdinalLadder:
    """Build one global ladder per variate from the full z-scored training array (T, V)."""
    if isinstance(train_data, np.ndarray):
        arr = train_data.astype(np.float64, copy=False)
    else:
        arr = train_data.detach().cpu().numpy().astype(np.float64, copy=False)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    _t, v = arr.shape
    uniq_rows = []
    n_unique_list = []
    k_max = 1
    for vi in range(v):
        u, k = _unique_sorted_1d_np(arr[:, vi], tie_atol)
        n_unique_list.append(max(k, 1))
        k_max = max(k_max, max(k, 1))
        uniq_rows.append(u)
    padded = torch.zeros(1, v, k_max, dtype=torch.float32)
    n_unique = torch.zeros(1, v, dtype=torch.long)
    for vi in range(v):
        u = uniq_rows[vi]
        k = len(u)
        if k > 0:
            padded[0, vi, :k] = torch.from_numpy(u.astype(np.float32))
        n_unique[0, vi] = max(k, 1)

    precomputed = None
    if precompute_ranks_for is not None:
        ladder_stub = OrdinalLadder(
            values=padded, n_unique=n_unique, tie_atol=float(tie_atol),
        )
        precomputed = precompute_ranks_for_array(precompute_ranks_for, ladder_stub)

    return OrdinalLadder(
        values=padded,
        n_unique=n_unique,
        tie_atol=float(tie_atol),
        precomputed_ranks=precomputed,
    )


def rank_max_broadcast(ladder: OrdinalLadder, x: torch.Tensor) -> torch.Tensor:
    """Per-variate max rank as a broadcastable divisor/multiplier for x."""
    rm = ladder.rank_max_per_variate().to(device=x.device, dtype=x.dtype).reshape(-1).clamp_min(1.0)
    if x.dim() == 4:
        return rm.view(1, -1, 1, 1)
    if x.dim() == 3:
        return rm.view(1, -1, 1)
    if x.dim() == 2:
        return rm.view(-1, 1)
    return rm


def ranks_to_unit(ordinal: torch.Tensor, ladder: OrdinalLadder) -> torch.Tensor:
    return ordinal / rank_max_broadcast(ladder, ordinal)


def ranks_from_unit(ordinal_unit: torch.Tensor, ladder: OrdinalLadder) -> torch.Tensor:
    return ordinal_unit * rank_max_broadcast(ladder, ordinal_unit)


def ranks_to_unit_numpy(rank_np: np.ndarray, ladder: OrdinalLadder) -> np.ndarray:
    rm = ladder.rank_max_per_variate().numpy().astype(np.float64).reshape(-1)
    rm = np.maximum(rm, 1.0)
    return (rank_np / rm[np.newaxis, :]).astype(np.float32)


def encode_with_ladder(
    x: torch.Tensor,
    ladder: OrdinalLadder,
) -> torch.Tensor:
    """Map values x (B,V,T) to raw integer ranks as float using the global ladder."""
    if x.dim() == 2:
        x = x.unsqueeze(1)
    b, v, _t = x.shape
    ladder = ladder.expand_batch(b)
    out = x.new_zeros(b, v, _t)
    atol = ladder.tie_atol
    for vi in range(v):
        k = int(ladder.n_unique[0, vi].item())
        uniq = ladder.values[0, vi, :k].to(device=x.device, dtype=x.dtype)
        ranks = _value_to_rank(uniq, x[:, vi], atol)
        out[:, vi] = ranks.to(torch.float32)
    return out


def decode_with_ladder(
    ordinal: torch.Tensor,
    ladder: OrdinalLadder,
) -> torch.Tensor:
    """Inverse ordinal map back to global z-score."""
    n_variates = int(ladder.values.shape[1])
    ordinal = _ensure_bvt(ordinal, n_variates=n_variates)
    ordinal = torch.nan_to_num(ordinal, nan=0.0, posinf=0.0, neginf=0.0)
    b, v, t = ordinal.shape
    ladder = ladder.expand_batch(b)
    out = ordinal.new_zeros(b, v, t)
    for vi in range(v):
        k = int(ladder.n_unique[0, vi].item())
        uniq = ladder.values[0, vi, :k].to(device=ordinal.device, dtype=ordinal.dtype)
        if k <= 1:
            out[:, vi] = uniq[0]
            continue
        j = ordinal[:, vi].round().clamp(0, k - 1).long()
        out[:, vi] = uniq.gather(0, j.reshape(-1)).reshape(b, t)
    return out


def _ensure_bvt(
    x: torch.Tensor,
    *,
    n_variates: Optional[int] = None,
) -> torch.Tensor:
    """Return (B, V, T), transposing (B, T, V) when needed."""
    if x.dim() == 2:
        d0, d1 = x.shape
        if n_variates is not None:
            if d0 == n_variates:
                return x.unsqueeze(0)
            if d1 == n_variates:
                return x.transpose(0, 1).unsqueeze(0)
        return x.unsqueeze(0) if d0 <= d1 else x.transpose(0, 1).unsqueeze(0)
    if x.dim() != 3:
        raise ValueError(f"expected (B,V,T) or (V,T), got {tuple(x.shape)}")
    b, d0, d1 = x.shape
    if n_variates is not None:
        if d0 == n_variates:
            return x
        if d1 == n_variates:
            return x.transpose(1, 2).contiguous()
    return x if d0 <= d1 else x.transpose(1, 2).contiguous()


def shift_window_to_ordinal_envelope(
    past: torch.Tensor,
    future: Optional[torch.Tensor],
    ladder: OrdinalLadder,
    *,
    margin_frac: float = 0.05,
    check_lookback_only: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Shift entire window (past+future) so values fit inside train ladder envelope.

    Only applies when lookback has OOD timesteps (outside train min/max).
    Target: [train_min + margin*span, train_max - margin*span] per variate.

    Returns ``ood_shift`` with shape (B, V, 1) to subtract after ordinal decode.
    """
    n_variates = int(ladder.values.shape[1])
    single_batch = past.dim() == 2
    past = _ensure_bvt(past, n_variates=n_variates)
    if future is not None:
        future = _ensure_bvt(future, n_variates=n_variates)

    train_min, train_max = ladder.z_envelope()
    device = past.device
    dtype = past.dtype
    train_min = train_min.to(device=device, dtype=dtype)
    train_max = train_max.to(device=device, dtype=dtype)
    span = (train_max - train_min).clamp_min(1e-8)
    margin = margin_frac * span
    lo = train_min + margin
    hi = train_max - margin

    past_out = past.clone()
    fut_out = future.clone() if future is not None else None

    if future is not None:
        combined = torch.cat([past, future], dim=-1)
    else:
        combined = past

    b, v, _ = past.shape
    ood_shift = torch.zeros(b, v, 1, device=device, dtype=dtype)
    for vi in range(v):
        if check_lookback_only:
            past_v = past[:, vi]
            ood = (past_v < train_min[vi]) | (past_v > train_max[vi])
            if not ood.any():
                continue
        win = combined[:, vi].reshape(b, -1)
        wmin = win.min(dim=-1, keepdim=True).values
        wmax = win.max(dim=-1, keepdim=True).values
        shift = torch.zeros_like(wmin)
        shift = torch.where(wmax > hi[vi], hi[vi] - wmax, shift)
        shift = torch.where(wmin < lo[vi], lo[vi] - wmin, shift)
        ood_shift[:, vi, 0] = shift.squeeze(-1)
        past_out[:, vi] = past_out[:, vi] + shift
        if fut_out is not None:
            fut_out[:, vi] = fut_out[:, vi] + shift

    if single_batch:
        past_out = past_out.squeeze(0)
        if fut_out is not None:
            fut_out = fut_out.squeeze(0)
        ood_shift = ood_shift.squeeze(0)
    return past_out, fut_out, ood_shift


def ordinal_encode(
    past: torch.Tensor,
    future: Optional[torch.Tensor],
    *,
    ladder: OrdinalLadder,
    apply_ood_shift: bool = False,
    margin_frac: float = 0.05,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], OrdinalLadder, torch.Tensor]:
    ood_shift: Optional[torch.Tensor] = None
    if apply_ood_shift:
        past, future, ood_shift = shift_window_to_ordinal_envelope(
            past, future, ladder, margin_frac=margin_frac,
        )
    if past.dim() == 2:
        batch_size = 1
    else:
        batch_size = past.shape[0]
    ladder_b = ladder.expand_batch(batch_size)
    if ood_shift is None:
        device = past.device if past.dim() == 3 else ladder_b.values.device
        dtype = past.dtype if past.dim() >= 2 else ladder_b.values.dtype
        ood_shift = torch.zeros(
            batch_size,
            int(ladder_b.values.shape[1]),
            1,
            device=device,
            dtype=dtype,
        )
        if past.dim() == 2:
            ood_shift = ood_shift.squeeze(0)
    past_ord = encode_with_ladder(past, ladder_b)
    fut_ord = encode_with_ladder(future, ladder_b) if future is not None else None
    return past_ord, fut_ord, ladder_b, ood_shift


def ordinal_decode(
    past_ord: torch.Tensor,
    future_ord: Optional[torch.Tensor],
    ladder: OrdinalLadder,
    *,
    ood_shift: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    past_val = decode_with_ladder(past_ord, ladder)
    fut_val = decode_with_ladder(future_ord, ladder) if future_ord is not None else None
    if ood_shift is not None and fut_val is not None:
        shift = ood_shift
        if shift.dim() == 2:
            shift = shift.unsqueeze(-1)
        fut_val = fut_val - shift.to(device=fut_val.device, dtype=fut_val.dtype)
    return past_val, fut_val
