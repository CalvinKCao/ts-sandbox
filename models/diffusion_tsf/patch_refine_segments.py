"""Unique absolute patch-refine segments + previous-stride teacher force."""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .patch_refine_geometry import PatchLocation

AllowedParentVariates = Dict[int, Set[int]]
AllowedSegmentVariates = Dict[int, Set[int]]


def iter_unique_segment_starts(
    series_len: int,
    *,
    lookback: int,
    horizon: int,
    overlap: int,
    patch_width: int,
    segment_stride: int = 1,
) -> List[int]:
    """Absolute left edges ``t`` of every unique patch that fits in some parent.

    ``segment_stride`` should match the data-subset sample stride (e.g. dynamic
    480) so sparse series do not explode into dense absolute indices.
    """
    segment_stride = max(1, int(segment_stride))
    patch_width = int(patch_width)
    fut_w = int(horizon) + int(overlap)
    if patch_width > fut_w:
        raise ValueError(f"patch_width {patch_width} > future width {fut_w}")
    # Absolute future of parent S covers [S+lb-K, S+lb+hz).
    t_min = int(lookback) - int(overlap)
    t_max = int(series_len) - patch_width
    # Need at least one parent: S in [0, series_len-lb-hz] with patch inside future.
    max_S = int(series_len) - int(lookback) - int(horizon)
    if max_S < 0 or t_max < t_min:
        return []
    starts: List[int] = []
    for t in range(t_min, t_max + 1, segment_stride):
        # Parent S must satisfy: S+lb-K <= t <= S+lb+hz-patch_width
        # => t - (lb+hz-pw) <= S <= t - (lb-K)
        s_lo = t - (int(lookback) + int(horizon) - patch_width)
        s_hi = t - (int(lookback) - int(overlap))
        s_lo = max(0, s_lo)
        s_hi = min(max_S, s_hi)
        if s_lo <= s_hi:
            starts.append(int(t))
    return starts


def parent_starts_for_segment(
    t: int,
    *,
    lookback: int,
    horizon: int,
    overlap: int,
    patch_width: int,
    series_len: int,
) -> List[int]:
    max_S = int(series_len) - int(lookback) - int(horizon)
    if max_S < 0:
        return []
    s_lo = t - (int(lookback) + int(horizon) - int(patch_width))
    s_hi = t - (int(lookback) - int(overlap))
    s_lo = max(0, s_lo)
    s_hi = min(max_S, s_hi)
    if s_lo > s_hi:
        return []
    return list(range(s_lo, s_hi + 1))


def sample_parent_start(
    t: int,
    *,
    epoch: int,
    series_id: int,
    lookback: int,
    horizon: int,
    overlap: int,
    patch_width: int,
    series_len: int,
    allowed_parent_starts: Optional[Sequence[int]] = None,
    allowed_parent_variates: Optional[AllowedParentVariates] = None,
) -> int:
    parents = parent_starts_for_segment(
        t,
        lookback=lookback,
        horizon=horizon,
        overlap=overlap,
        patch_width=patch_width,
        series_len=series_len,
    )
    if allowed_parent_variates is not None:
        parents = [
            s
            for s in parents
            if int(s) in allowed_parent_variates and allowed_parent_variates[int(s)]
        ]
    elif allowed_parent_starts is not None:
        allow = set(int(s) for s in allowed_parent_starts)
        parents = [s for s in parents if int(s) in allow]
    if not parents:
        raise RuntimeError(f"no valid parent for absolute segment t={t}")
    seed_bytes = hashlib.sha256(
        f"patch-seg|{series_id}|{epoch}|{t}".encode("utf-8")
    ).digest()
    rng = np.random.RandomState(int.from_bytes(seed_bytes[:4], "little"))
    return int(parents[int(rng.randint(0, len(parents)))])


def compress_prev_refine_32_to_16(prev_hir: torch.Tensor) -> torch.Tensor:
    """Pool a hir prev-crop vertically onto 16 coarse rows.

    Historically named for 32→16 (factor 2). Also accepts any height divisible
    by 16 (e.g. 64→16 with factor 4) via vertical avg-pool.
    """
    h = int(prev_hir.shape[-2])
    if h < 16 or h % 16 != 0:
        raise ValueError(
            f"prev refine crop height must be a positive multiple of 16, got {tuple(prev_hir.shape)}"
        )
    kh = h // 16
    flat = prev_hir.reshape(-1, 1, h, prev_hir.shape[-1])
    out = F.avg_pool2d(flat, kernel_size=(kh, 1))
    return out.reshape(*prev_hir.shape[:-2], 16, prev_hir.shape[-1])


def prev_primary_row0(
    coarse_edges: torch.Tensor,
    loc: PatchLocation,
    *,
    canvas_height: int,
    patch_height: int,
    patch_width: int,
    col_stride: int,
) -> Optional[int]:
    """Row0 the previous stride primary would use (coarse-edge centered).

    Matches AR infer, which compresses the prior primary prediction in that
    patch's own frame. Returns ``None`` when the prev window is OOB.
    """
    prev_col0 = int(loc.col0) - int(col_stride)
    width = int(coarse_edges.shape[-1])
    if prev_col0 < 0 or prev_col0 + patch_width > width:
        return None
    max_row0 = int(canvas_height) - int(patch_height)
    anchor = prev_col0 + int(patch_width) // 2
    edge = int(coarse_edges[loc.batch_index, loc.variate_index, anchor].item())
    return max(0, min(edge - int(patch_height) // 2, max_row0))


def extract_prev_refine_crops(
    hir_canvas: torch.Tensor,
    locations: Sequence[PatchLocation],
    *,
    patch_height: int,
    patch_width: int,
    col_stride: int,
    coarse_edges: torch.Tensor,
    canvas_height: Optional[int] = None,
) -> torch.Tensor:
    """Teacher-force crops at ``[col0 - col_stride, col0 - col_stride + pw)``.

    For pw=8, stride=6 this is absolute ``[t-6, t+2)`` relative to the current
    patch left edge. Crops use the **previous primary's row0** (coarse-centered
    at the prev mid-column), matching AR infer which compresses the prior
    primary prediction in its own frame. Missing / OOB → zeros.
    Returns ``(N,1,ph,pw)``.
    """
    n = len(locations)
    device = hir_canvas.device
    dtype = hir_canvas.dtype
    out = torch.zeros(n, 1, patch_height, patch_width, device=device, dtype=dtype)
    if canvas_height is None:
        canvas_height = int(hir_canvas.shape[-2])
    width = int(hir_canvas.shape[-1])
    for i, loc in enumerate(locations):
        prev_col0 = int(loc.col0) - int(col_stride)
        if prev_col0 < 0 or prev_col0 + patch_width > width:
            continue
        prev_row0 = prev_primary_row0(
            coarse_edges,
            loc,
            canvas_height=canvas_height,
            patch_height=patch_height,
            patch_width=patch_width,
            col_stride=col_stride,
        )
        if prev_row0 is None:
            continue
        crop = hir_canvas[
            loc.batch_index,
            loc.variate_index,
            prev_row0 : prev_row0 + patch_height,
            prev_col0 : prev_col0 + patch_width,
        ]
        if crop.shape[-2] != patch_height or crop.shape[-1] != patch_width:
            continue
        out[i, 0] = crop
    return out


def locations_for_fixed_col0(
    coarse_edges: torch.Tensor,
    col0: torch.Tensor,
    *,
    canvas_height: int,
    patch_height: int,
    patch_width: int,
    hir_canvas: Optional[torch.Tensor] = None,
    variate_keep: Optional[torch.Tensor] = None,
) -> List[PatchLocation]:
    """One boundary-centered crop per kept (B,V) at the given per-batch ``col0``.

    Centers vertically on the coarse edge at the patch mid-column. When
    ``hir_canvas`` is provided and that crop would miss every hi-res GT
    transition in ``[col0, col0+pw)``, recenters on the hir edge at the
    mid-column (same fallback idea as ``select_patch_locations``).

    ``variate_keep`` optional bool ``(B,V)`` — skip active vars marked False
    (flatline undersample per active variate).
    """
    if col0.ndim != 1:
        raise ValueError(f"col0 must be (B,), got {tuple(col0.shape)}")
    B, V, W = coarse_edges.shape
    if col0.numel() != B:
        raise ValueError(f"col0 length {col0.numel()} != batch {B}")
    if hir_canvas is not None and hir_canvas.shape[:2] != (B, V):
        raise ValueError(
            f"hir_canvas batch/vars {tuple(hir_canvas.shape[:2])} != coarse {(B, V)}"
        )
    keep = None
    if variate_keep is not None:
        keep = variate_keep.to(dtype=torch.bool)
        if keep.shape != (B, V):
            raise ValueError(
                f"variate_keep shape {tuple(keep.shape)} != batch/vars {(B, V)}"
            )
    max_row0 = canvas_height - patch_height
    max_col0 = W - patch_width
    locations: List[PatchLocation] = []
    for bi in range(B):
        c0 = int(col0[bi].item())
        if c0 < 0 or c0 > max_col0:
            raise ValueError(f"col0={c0} out of range [0, {max_col0}] for W={W}")
        anchor = c0 + patch_width // 2
        for vi in range(V):
            if keep is not None and not bool(keep[bi, vi].item()):
                continue
            edge = int(coarse_edges[bi, vi, anchor].item())
            row0 = max(0, min(edge - patch_height // 2, max_row0))
            if hir_canvas is not None:
                hir_edges = hir_canvas[bi, vi, :, c0 : c0 + patch_width].sum(dim=0) - 1
                in_view = (hir_edges >= row0) & (hir_edges < row0 + patch_height)
                if not bool(in_view.any()):
                    he = int(hir_edges[patch_width // 2].item())
                    he = max(0, min(he, canvas_height - 1))
                    row0 = max(0, min(he - patch_height // 2, max_row0))
            locations.append(
                PatchLocation(
                    flat_index=bi * V + vi,
                    batch_index=bi,
                    variate_index=vi,
                    row0=row0,
                    col0=c0,
                )
            )
    return locations


def select_primary_ar_locations(
    coarse_edges: torch.Tensor,
    *,
    canvas_height: int,
    patch_height: int,
    patch_width: int,
    col_stride: int,
) -> List[PatchLocation]:
    """One coarse-centered crop per (B,V) at each primary stride ``col0``.

    Unique ``col0`` on the stride grid — no fallback duplicates.
    """
    if coarse_edges.ndim != 3:
        raise ValueError(f"coarse_edges must be (B,V,W), got {tuple(coarse_edges.shape)}")
    batch_size, n_variates, width = coarse_edges.shape
    max_row0 = canvas_height - patch_height
    max_col0 = width - patch_width
    if max_col0 < 0:
        raise ValueError(f"patch width {patch_width} exceeds future width {width}")
    primary_starts = list(range(0, max_col0 + 1, col_stride))
    locations: List[PatchLocation] = []
    for bi in range(batch_size):
        for vi in range(n_variates):
            flat_index = bi * n_variates + vi
            edges = coarse_edges[bi, vi]
            for col0 in primary_starts:
                anchor = col0 + patch_width // 2
                row0 = max(0, min(int(edges[anchor].item()) - patch_height // 2, max_row0))
                locations.append(
                    PatchLocation(
                        flat_index=flat_index,
                        batch_index=bi,
                        variate_index=vi,
                        row0=row0,
                        col0=col0,
                    )
                )
    return locations


def _coverage_mask_for_locations(
    coarse_edges: torch.Tensor,
    locations: Sequence[PatchLocation],
    *,
    patch_height: int,
    patch_width: int,
) -> torch.Tensor:
    """Bool ``(B,V,W)`` — True where a location sees the coarse boundary."""
    B, V, W = coarse_edges.shape
    covered = torch.zeros(B, V, W, device=coarse_edges.device, dtype=torch.bool)
    for loc in locations:
        cols = torch.arange(
            loc.col0,
            loc.col0 + patch_width,
            device=coarse_edges.device,
        )
        edges = coarse_edges[loc.batch_index, loc.variate_index, cols]
        in_rows = (edges >= loc.row0) & (edges < loc.row0 + patch_height)
        covered[loc.batch_index, loc.variate_index, cols[in_rows]] = True
    return covered


def select_coverage_gap_locations(
    coarse_edges: torch.Tensor,
    primary_locations: Sequence[PatchLocation],
    *,
    canvas_height: int,
    patch_height: int,
    patch_width: int,
) -> List[PatchLocation]:
    """Blanked-prev fallbacks only for timesteps AR primaries leave uncovered.

    Coverage gaps are OOB / boundary too hi/lo for the 32-high primary patches —
    not every off-stride ``col0`` from the old dense fallback set.
    """
    if coarse_edges.ndim != 3:
        raise ValueError(f"coarse_edges must be (B,V,W), got {tuple(coarse_edges.shape)}")
    batch_size, n_variates, width = coarse_edges.shape
    max_row0 = canvas_height - patch_height
    max_col0 = width - patch_width
    covered = _coverage_mask_for_locations(
        coarse_edges,
        primary_locations,
        patch_height=patch_height,
        patch_width=patch_width,
    )
    # Dedup against primaries (and among gap fills) by (B,V,row0,col0).
    seen: set[Tuple[int, int, int, int]] = {
        (loc.batch_index, loc.variate_index, loc.row0, loc.col0)
        for loc in primary_locations
    }
    gap: List[PatchLocation] = []
    for bi in range(batch_size):
        for vi in range(n_variates):
            flat_index = bi * n_variates + vi
            edges = coarse_edges[bi, vi]
            while not bool(covered[bi, vi].all()):
                timestep = int((~covered[bi, vi]).nonzero(as_tuple=False)[0].item())
                col0 = max(0, min(timestep - patch_width // 2, max_col0))
                row0 = max(
                    0,
                    min(int(edges[timestep].item()) - patch_height // 2, max_row0),
                )
                key = (bi, vi, row0, col0)
                was = bool(covered[bi, vi, timestep])
                if key not in seen:
                    seen.add(key)
                    gap.append(
                        PatchLocation(
                            flat_index=flat_index,
                            batch_index=bi,
                            variate_index=vi,
                            row0=row0,
                            col0=col0,
                        )
                    )
                cols = torch.arange(col0, col0 + patch_width, device=edges.device)
                in_rows = (edges[cols] >= row0) & (edges[cols] < row0 + patch_height)
                covered[bi, vi, cols[in_rows]] = True
                if not bool(covered[bi, vi, timestep]) and not was:
                    raise RuntimeError(
                        f"failed to cover boundary at B={bi} V={vi} t={timestep}"
                    )
    return gap


def group_locations_by_col0(
    locations: Sequence[PatchLocation],
) -> List[Tuple[int, List[PatchLocation]]]:
    """Group locations into ``(col0, locs)`` sorted left→right for batched AR."""
    by_col: dict[int, List[PatchLocation]] = {}
    for loc in locations:
        by_col.setdefault(int(loc.col0), []).append(loc)
    return sorted(by_col.items(), key=lambda kv: kv[0])


def partition_primary_and_gap(
    locations: Sequence[PatchLocation],
    *,
    col_stride: int,
) -> Tuple[List[PatchLocation], List[PatchLocation]]:
    """Deprecated split by ``col0 % stride``; prefer coverage-based gap fill.

    Kept for callers/tests: primary = unique stride-grid col0 (first wins);
    gap = everything else. Prefer ``select_primary_ar_locations`` +
    ``select_coverage_gap_locations`` for unique-seg infer.
    """
    primary: List[PatchLocation] = []
    gap: List[PatchLocation] = []
    seen_primary: set[Tuple[int, int, int]] = set()
    for loc in locations:
        if int(loc.col0) % int(col_stride) == 0:
            key = (loc.batch_index, loc.variate_index, int(loc.col0))
            if key in seen_primary:
                gap.append(loc)
                continue
            seen_primary.add(key)
            primary.append(loc)
        else:
            gap.append(loc)
    primary.sort(key=lambda loc: (loc.batch_index, loc.variate_index, loc.col0))
    return primary, gap


def _normalize_var_allow_map(
    allowed: Optional[Dict[int, Sequence[int]]],
    *,
    name: str,
) -> Optional[Dict[int, Set[int]]]:
    if allowed is None:
        return None
    out: Dict[int, Set[int]] = {}
    for key, vars_ in allowed.items():
        vs = {int(v) for v in vars_}
        if not vs:
            continue
        out[int(key)] = vs
    if not out:
        raise RuntimeError(f"{name} is empty")
    return out


def _normalize_allowed_parent_variates(
    allowed_parent_variates: Optional[Dict[int, Sequence[int]]],
) -> Optional[AllowedParentVariates]:
    return _normalize_var_allow_map(
        allowed_parent_variates, name="allowed_parent_variates"
    )


def _normalize_allowed_segment_variates(
    allowed_segment_variates: Optional[Dict[int, Sequence[int]]],
) -> Optional[AllowedSegmentVariates]:
    return _normalize_var_allow_map(
        allowed_segment_variates, name="allowed_segment_variates"
    )


class UniquePatchSegmentDataset(Dataset):
    """Index unique absolute patch starts; resample parent window each epoch."""

    def __init__(
        self,
        data: torch.Tensor,
        *,
        lookback: int,
        horizon: int,
        overlap: int,
        patch_width: int,
        segment_stride: int = 1,
        series_id: int = 0,
        rank_data: Optional[torch.Tensor] = None,
        allowed_parent_starts: Optional[Sequence[int]] = None,
        allowed_parent_variates: Optional[Dict[int, Sequence[int]]] = None,
        allowed_segment_variates: Optional[Dict[int, Sequence[int]]] = None,
    ):
        if data.ndim != 2:
            raise ValueError(f"data must be (T,V), got {tuple(data.shape)}")
        if (
            allowed_segment_variates is not None
            and allowed_parent_variates is not None
        ):
            raise ValueError(
                "pass only one of allowed_segment_variates / allowed_parent_variates"
            )
        if (
            allowed_segment_variates is not None
            and allowed_parent_starts is not None
        ):
            raise ValueError(
                "allowed_segment_variates already filters crops; do not also pass "
                "allowed_parent_starts"
            )
        self.data = data if isinstance(data, torch.Tensor) else torch.tensor(data)
        self.rank_data = rank_data
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.overlap = int(overlap)
        self.patch_width = int(patch_width)
        self.segment_stride = max(1, int(segment_stride))
        self.series_id = int(series_id)
        self.epoch = 0
        # Matches TimeSeriesDataset: loaders must not re-run ordinal_encode on ranks.
        self.yields_ordinal_ranks = self.rank_data is not None
        self.n_variates = int(self.data.shape[1])
        self.allowed_segment_variates = _normalize_allowed_segment_variates(
            allowed_segment_variates
        )
        self.allowed_parent_variates = _normalize_allowed_parent_variates(
            allowed_parent_variates
        )
        self.allowed_parent_starts: Optional[set[int]]
        if self.allowed_parent_variates is not None:
            self.allowed_parent_starts = set(self.allowed_parent_variates.keys())
        elif allowed_parent_starts is None:
            self.allowed_parent_starts = None
        else:
            self.allowed_parent_starts = {int(s) for s in allowed_parent_starts}
            if not self.allowed_parent_starts:
                raise RuntimeError(
                    "UniquePatchSegmentDataset: allowed_parent_starts is empty"
                )
        starts = iter_unique_segment_starts(
            int(self.data.shape[0]),
            lookback=self.lookback,
            horizon=self.horizon,
            overlap=self.overlap,
            patch_width=self.patch_width,
            segment_stride=self.segment_stride,
        )
        if self.allowed_segment_variates is not None:
            allow_t = self.allowed_segment_variates
            self.segment_starts = [int(t) for t in starts if int(t) in allow_t]
        elif self.allowed_parent_starts is not None:
            filtered: List[int] = []
            for t in starts:
                parents = parent_starts_for_segment(
                    int(t),
                    lookback=self.lookback,
                    horizon=self.horizon,
                    overlap=self.overlap,
                    patch_width=self.patch_width,
                    series_len=int(self.data.shape[0]),
                )
                if self.allowed_parent_variates is not None:
                    if any(
                        int(s) in self.allowed_parent_variates
                        and self.allowed_parent_variates[int(s)]
                        for s in parents
                    ):
                        filtered.append(int(t))
                elif any(int(s) in self.allowed_parent_starts for s in parents):
                    filtered.append(int(t))
            self.segment_starts = filtered
        else:
            self.segment_starts = starts
        if not self.segment_starts:
            raise RuntimeError("UniquePatchSegmentDataset: zero valid segments")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.segment_starts)

    def _variate_keep_for_parent(self, parent_start: int) -> torch.Tensor:
        mask = torch.ones(self.n_variates, dtype=torch.bool)
        if self.allowed_parent_variates is None:
            return mask
        allowed = self.allowed_parent_variates.get(int(parent_start))
        if not allowed:
            raise RuntimeError(
                f"parent_start={parent_start} has no allowed active variates"
            )
        mask.zero_()
        for v in allowed:
            if v < 0 or v >= self.n_variates:
                raise IndexError(
                    f"allowed variate {v} out of range for V={self.n_variates}"
                )
            mask[v] = True
        return mask

    def _variate_keep_for_segment(self, segment_start: int) -> torch.Tensor:
        mask = torch.ones(self.n_variates, dtype=torch.bool)
        if self.allowed_segment_variates is None:
            return mask
        allowed = self.allowed_segment_variates.get(int(segment_start))
        if not allowed:
            raise RuntimeError(
                f"segment_start={segment_start} has no allowed active variates"
            )
        mask.zero_()
        for v in allowed:
            if v < 0 or v >= self.n_variates:
                raise IndexError(
                    f"allowed variate {v} out of range for V={self.n_variates}"
                )
            mask[v] = True
        return mask

    def __getitem__(self, idx: int):
        t = int(self.segment_starts[idx])
        S = sample_parent_start(
            t,
            epoch=self.epoch,
            series_id=self.series_id,
            lookback=self.lookback,
            horizon=self.horizon,
            overlap=self.overlap,
            patch_width=self.patch_width,
            series_len=int(self.data.shape[0]),
            allowed_parent_starts=(
                sorted(self.allowed_parent_starts)
                if self.allowed_parent_starts is not None
                and self.allowed_parent_variates is None
                else None
            ),
            allowed_parent_variates=self.allowed_parent_variates,
        )
        source = self.rank_data if self.rank_data is not None else self.data
        past = source[S : S + self.lookback].T
        fut_start = S + self.lookback - self.overlap
        future = source[fut_start : S + self.lookback + self.horizon].T
        col0 = t - fut_start
        if self.allowed_segment_variates is not None:
            keep = self._variate_keep_for_segment(t)
            return past, future, torch.tensor(col0, dtype=torch.long), keep
        if self.allowed_parent_variates is not None:
            keep = self._variate_keep_for_parent(S)
            return past, future, torch.tensor(col0, dtype=torch.long), keep
        return past, future, torch.tensor(col0, dtype=torch.long)


def wrap_timeseries_as_unique_segments(
    ts_ds: Dataset,
    *,
    patch_width: int,
    segment_stride: int = 1,
    series_id: int = 0,
    allowed_parent_starts: Optional[Sequence[int]] = None,
    allowed_parent_variates: Optional[Dict[int, Sequence[int]]] = None,
    allowed_segment_variates: Optional[Dict[int, Sequence[int]]] = None,
) -> UniquePatchSegmentDataset:
    """Rebuild a ``TimeSeriesDataset`` as unique absolute patch segments."""
    data = ts_ds.data
    rank = getattr(ts_ds, "rank_data", None)
    return UniquePatchSegmentDataset(
        data,
        lookback=int(ts_ds.lookback),
        horizon=int(ts_ds.horizon),
        overlap=int(ts_ds.lookback_overlap),
        patch_width=int(patch_width),
        segment_stride=int(segment_stride),
        series_id=int(series_id),
        rank_data=rank,
        allowed_parent_starts=allowed_parent_starts,
        allowed_parent_variates=allowed_parent_variates,
        allowed_segment_variates=allowed_segment_variates,
    )
