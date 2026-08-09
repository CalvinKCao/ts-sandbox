"""GT flatline detection for patch_refine train undersampling.

Flat run (same defs as temp/scripts/etth2_coarse_flat_run_stats.py):
  ≥ min_run consecutive identical coarse H bins AND continuous GT range
  ≤ flat_eps_frac × coarse_bin_width over that run.

Refine trains on unique absolute patch crops (e.g. pw=6 coarse columns).
Flat / wiggle / undersample are per (segment_start, active_variate), with the
true-flat predicate restricted to that crop's time span — not the full H=96
parent horizon.
"""

from __future__ import annotations

import logging
import math
import random
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

logger = logging.getLogger(__name__)

DEFAULT_MIN_RUN = 3
DEFAULT_EPS_FRAC = 0.25
# Keep distinct from patch_refine_finetune_window_fraction's seed+17.
DEFAULT_SEED_OFFSET = 91


def find_same_bin_runs(
    bins_1d: np.ndarray, min_run: int
) -> List[Tuple[int, int, int]]:
    n = int(bins_1d.shape[0])
    out: List[Tuple[int, int, int]] = []
    i = 0
    while i < n:
        j = i + 1
        while j < n and int(bins_1d[j]) == int(bins_1d[i]):
            j += 1
        if j - i >= min_run:
            out.append((i, j, int(bins_1d[i])))
        i = j
    return out


def window_norm_z_and_coarse_bins(
    past: np.ndarray,
    future: np.ndarray,
    *,
    max_scale: float,
    coarse_h: int,
    std_floor: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """past/future (V, T) → z/bins (V, T_future). Matches train window-norm."""
    if past.ndim != 2 or future.ndim != 2:
        raise ValueError(f"expected (V,T), got past={past.shape} future={future.shape}")
    if past.shape[0] != future.shape[0]:
        raise ValueError(f"V mismatch past={past.shape} future={future.shape}")
    if max_scale <= 0.0 or coarse_h <= 0:
        raise RuntimeError(f"bad lattice max_scale={max_scale} coarse_h={coarse_h}")
    if std_floor <= 0.0:
        raise RuntimeError(f"std_floor must be > 0, got {std_floor}")

    past_t = torch.as_tensor(past, dtype=torch.float32)
    fut_t = torch.as_tensor(future, dtype=torch.float32)
    center = past_t.mean(dim=-1)
    std = past_t.std(dim=-1).clamp_min(std_floor)
    z = ((fut_t - center.unsqueeze(-1)) / std.unsqueeze(-1)).detach().cpu().numpy()
    z_clip = np.clip(z, -max_scale, max_scale)
    pos = (z_clip + max_scale) / (2.0 * max_scale) * coarse_h
    bins = np.floor(pos).astype(np.int64)
    bins = np.clip(bins, 0, coarse_h - 1)
    return z.astype(np.float64, copy=False), bins


def variate_has_true_flatline(
    z_h: np.ndarray,
    bins_h: np.ndarray,
    *,
    flat_eps: float,
    min_run: int,
) -> bool:
    """True if this 1d span has a true-flat same-bin run (any length series)."""
    if z_h.ndim != 1 or bins_h.ndim != 1:
        raise ValueError(f"expected (T,), got z={z_h.shape} bins={bins_h.shape}")
    if z_h.shape != bins_h.shape:
        raise ValueError(f"shape mismatch z={z_h.shape} bins={bins_h.shape}")
    for a, b, _bid in find_same_bin_runs(bins_h, min_run):
        seg = z_h[a:b]
        if float(seg.max() - seg.min()) <= flat_eps:
            return True
    return False


def horizon_flatline_per_variate(
    z_vh: np.ndarray,
    bins_vh: np.ndarray,
    *,
    flat_eps: float,
    min_run: int,
) -> np.ndarray:
    """Bool mask (V,) — True when that active variate has a true-flat run."""
    if z_vh.ndim != 2 or bins_vh.ndim != 2:
        raise ValueError(f"expected (V,T), got z={z_vh.shape} bins={bins_vh.shape}")
    if z_vh.shape != bins_vh.shape:
        raise ValueError(f"shape mismatch z={z_vh.shape} bins={bins_vh.shape}")
    v = int(z_vh.shape[0])
    out = np.zeros(v, dtype=bool)
    for i in range(v):
        out[i] = variate_has_true_flatline(
            z_vh[i], bins_vh[i], flat_eps=flat_eps, min_run=min_run
        )
    return out


def horizon_has_true_flatline(
    z_vh: np.ndarray,
    bins_vh: np.ndarray,
    *,
    flat_eps: float,
    min_run: int,
) -> bool:
    """Legacy OR-across-vars: True if any variate is flat. Prefer per-variate APIs."""
    return bool(
        horizon_flatline_per_variate(
            z_vh, bins_vh, flat_eps=flat_eps, min_run=min_run
        ).any()
    )


def classify_timeseries_flatline_windows(
    ts_ds: Dataset,
    *,
    max_scale: float,
    coarse_h: int,
    std_floor: float,
    forecast_length: int,
    lookback_overlap: int,
    min_run: int = DEFAULT_MIN_RUN,
    flat_eps_frac: float = DEFAULT_EPS_FRAC,
) -> np.ndarray:
    """Deprecated: bool mask (N, V) over full H forecast. Prefer crop APIs."""
    n = len(ts_ds)
    if n <= 0:
        raise RuntimeError("empty train dataset for flatline classification")
    if forecast_length <= 0:
        raise ValueError(f"forecast_length must be > 0, got {forecast_length}")
    if lookback_overlap < 0:
        raise ValueError(f"lookback_overlap must be >= 0, got {lookback_overlap}")

    bin_width = 2.0 * float(max_scale) / float(coarse_h)
    flat_eps = float(flat_eps_frac) * bin_width

    item0 = ts_ds[0]
    if not isinstance(item0, (tuple, list)) or len(item0) < 2:
        raise TypeError(
            f"expected (past, future[, ...]) from train ds, got {type(item0)}"
        )
    past0 = np.asarray(item0[0], dtype=np.float32)
    if past0.ndim != 2:
        raise ValueError(f"expected past (V,T), got {past0.shape}")
    n_vars = int(past0.shape[0])
    if n_vars <= 0:
        raise RuntimeError("zero variates in train window")
    mask = np.zeros((n, n_vars), dtype=bool)

    for i in range(n):
        item = ts_ds[i]
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            raise TypeError(
                f"expected (past, future[, ...]) from train ds, got {type(item)}"
            )
        past = np.asarray(item[0], dtype=np.float32)
        future = np.asarray(item[1], dtype=np.float32)
        if past.ndim != 2 or future.ndim != 2:
            raise ValueError(
                f"window {i}: expected past/future (V,T), got {past.shape}/{future.shape}"
            )
        if int(past.shape[0]) != n_vars:
            raise ValueError(
                f"window {i}: V={past.shape[0]} != first-window V={n_vars}"
            )
        fut_w = int(future.shape[-1])
        need = int(lookback_overlap) + int(forecast_length)
        if fut_w < need:
            raise RuntimeError(
                f"window {i}: future width {fut_w} < overlap+horizon {need}"
            )
        hz = future[:, -int(forecast_length) :]
        z, bins = window_norm_z_and_coarse_bins(
            past,
            hz,
            max_scale=max_scale,
            coarse_h=coarse_h,
            std_floor=std_floor,
        )
        mask[i] = horizon_flatline_per_variate(
            z, bins, flat_eps=flat_eps, min_run=int(min_run)
        )
    return mask


def classify_unique_segment_flatline_crops(
    data: np.ndarray,
    segment_starts: Sequence[int],
    *,
    lookback: int,
    horizon: int,
    overlap: int,
    patch_width: int,
    max_scale: float,
    coarse_h: int,
    std_floor: float,
    min_run: int = DEFAULT_MIN_RUN,
    flat_eps_frac: float = DEFAULT_EPS_FRAC,
) -> np.ndarray:
    """Bool mask (N_seg, V): flat if true-flat run exists inside the pw crop.

    Window-norm affine uses the leftmost valid parent for each absolute segment
    (deterministic; unique-seg resamples parents at train time).
    """
    from models.diffusion_tsf.patch_refine_segments import parent_starts_for_segment

    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"expected series (T,V), got {data.shape}")
    series_len, n_vars = int(data.shape[0]), int(data.shape[1])
    if n_vars <= 0:
        raise RuntimeError("zero variates in series")
    pw = int(patch_width)
    if pw <= 0:
        raise ValueError(f"patch_width must be > 0, got {pw}")
    if int(min_run) > pw:
        raise RuntimeError(
            f"min_run={min_run} > patch_width={pw}: no crop can ever be flat"
        )

    bin_width = 2.0 * float(max_scale) / float(coarse_h)
    flat_eps = float(flat_eps_frac) * bin_width
    starts = [int(t) for t in segment_starts]
    n = len(starts)
    mask = np.zeros((n, n_vars), dtype=bool)

    for i, t in enumerate(starts):
        if t < 0 or t + pw > series_len:
            raise IndexError(
                f"segment t={t} pw={pw} out of range for series_len={series_len}"
            )
        parents = parent_starts_for_segment(
            t,
            lookback=int(lookback),
            horizon=int(horizon),
            overlap=int(overlap),
            patch_width=pw,
            series_len=series_len,
        )
        if not parents:
            raise RuntimeError(f"no valid parent for absolute segment t={t}")
        S = int(parents[0])
        past = data[S : S + int(lookback)].T
        crop = data[t : t + pw].T
        if past.shape != (n_vars, int(lookback)):
            raise RuntimeError(f"past shape {past.shape} at S={S} t={t}")
        if crop.shape != (n_vars, pw):
            raise RuntimeError(f"crop shape {crop.shape} at t={t}")
        z, bins = window_norm_z_and_coarse_bins(
            past,
            crop,
            max_scale=max_scale,
            coarse_h=coarse_h,
            std_floor=std_floor,
        )
        mask[i] = horizon_flatline_per_variate(
            z, bins, flat_eps=flat_eps, min_run=int(min_run)
        )
    return mask


def select_flatline_undersample_pairs(
    flat_mask: np.ndarray,
    *,
    keep_frac: float,
    seed: int,
) -> List[Tuple[int, int]]:
    """Keep all wiggle (i,v); randomly keep keep_frac of flatline (i,v) pairs."""
    if not math.isfinite(keep_frac) or keep_frac <= 0.0 or keep_frac > 1.0:
        raise ValueError(
            f"patch_refine_flatline_keep_frac must be in (0, 1], got {keep_frac!r}"
        )
    flat_mask = np.asarray(flat_mask, dtype=bool)
    if flat_mask.ndim != 2:
        raise ValueError(f"expected flat_mask (N,V), got shape={flat_mask.shape}")
    n, v = flat_mask.shape
    if keep_frac >= 1.0:
        return [(i, j) for i in range(n) for j in range(v)]

    wiggle_pairs = [
        (int(i), int(j))
        for i, j in zip(*np.nonzero(~flat_mask))
    ]
    flat_pairs = [
        (int(i), int(j))
        for i, j in zip(*np.nonzero(flat_mask))
    ]
    if not flat_pairs:
        return sorted(wiggle_pairs)
    k = max(1, int(round(len(flat_pairs) * keep_frac)))
    k = min(k, len(flat_pairs))
    rng = random.Random(int(seed))
    kept_flat = rng.sample(flat_pairs, k) if k > 0 else []
    return sorted(wiggle_pairs + kept_flat)


def select_flatline_undersample_indices(
    flat_mask: np.ndarray,
    *,
    keep_frac: float,
    seed: int,
) -> List[int]:
    """Deprecated window-level helper. Prefer select_flatline_undersample_pairs."""
    flat_mask = np.asarray(flat_mask, dtype=bool)
    if flat_mask.ndim == 2:
        flat_mask = flat_mask.any(axis=1)
    if flat_mask.ndim != 1:
        raise ValueError(f"expected flat_mask (N,) or (N,V), got {flat_mask.shape}")
    if not math.isfinite(keep_frac) or keep_frac <= 0.0 or keep_frac > 1.0:
        raise ValueError(
            f"patch_refine_flatline_keep_frac must be in (0, 1], got {keep_frac!r}"
        )
    n = int(flat_mask.shape[0])
    wiggle_idx = np.flatnonzero(~flat_mask).tolist()
    flat_idx = np.flatnonzero(flat_mask).tolist()
    if keep_frac >= 1.0:
        return list(range(n))
    if not flat_idx:
        return sorted(wiggle_idx)
    k = max(1, int(round(len(flat_idx) * keep_frac))) if flat_idx else 0
    k = min(k, len(flat_idx))
    rng = random.Random(int(seed))
    kept_flat = rng.sample(flat_idx, k) if k > 0 else []
    return sorted(wiggle_idx + kept_flat)


def parent_starts_from_timeseries_indices(
    ts_ds: Dataset, indices: Sequence[int]
) -> List[int]:
    """Map TimeSeriesDataset indices → absolute parent starts (idx * stride)."""
    stride = int(getattr(ts_ds, "stride", 1))
    out: List[int] = []
    for idx in indices:
        i = int(idx)
        if i < 0 or i >= len(ts_ds):
            raise IndexError(f"window index {i} out of range for len={len(ts_ds)}")
        out.append(i * stride)
    return out


def allowed_parent_variates_from_pairs(
    ts_ds: Dataset, pairs: Sequence[Tuple[int, int]]
) -> Dict[int, Set[int]]:
    """Deprecated: parent-level allow-set. Prefer allowed_segment_variates_from_pairs."""
    stride = int(getattr(ts_ds, "stride", 1))
    out: Dict[int, Set[int]] = {}
    n = len(ts_ds)
    for wi, vi in pairs:
        i = int(wi)
        v = int(vi)
        if i < 0 or i >= n:
            raise IndexError(f"window index {i} out of range for len={n}")
        if v < 0:
            raise IndexError(f"variate index {v} must be >= 0")
        parent = i * stride
        out.setdefault(parent, set()).add(v)
    return out


def allowed_segment_variates_from_pairs(
    segment_starts: Sequence[int],
    pairs: Sequence[Tuple[int, int]],
) -> Dict[int, Set[int]]:
    """Map absolute segment_start → set of allowed active variate indices."""
    starts = [int(t) for t in segment_starts]
    out: Dict[int, Set[int]] = {}
    n = len(starts)
    for si, vi in pairs:
        i = int(si)
        v = int(vi)
        if i < 0 or i >= n:
            raise IndexError(f"segment index {i} out of range for n={n}")
        if v < 0:
            raise IndexError(f"variate index {v} must be >= 0")
        out.setdefault(starts[i], set()).add(v)
    return out


def _per_variate_stats(
    flat_mask: np.ndarray, kept_pairs: Sequence[Tuple[int, int]]
) -> List[dict]:
    n, v = flat_mask.shape
    kept_set = {(int(i), int(j)) for i, j in kept_pairs}
    rows: List[dict] = []
    for vi in range(v):
        n_flat = int(flat_mask[:, vi].sum())
        n_wiggle = int((~flat_mask[:, vi]).sum())
        n_flat_kept = sum(
            1 for i in range(n) if flat_mask[i, vi] and (i, vi) in kept_set
        )
        n_wiggle_kept = sum(
            1 for i in range(n) if (not flat_mask[i, vi]) and (i, vi) in kept_set
        )
        n_kept = n_flat_kept + n_wiggle_kept
        rows.append(
            {
                "variate": int(vi),
                "n_flatline": n_flat,
                "n_wiggle": n_wiggle,
                "n_flatline_kept": n_flat_kept,
                "n_wiggle_kept": n_wiggle_kept,
                "n_kept": n_kept,
                "flat_keep_rate": (n_flat_kept / n_flat) if n_flat else float("nan"),
                "pct_kept": (100.0 * n_kept / n) if n else float("nan"),
            }
        )
    return rows


def undersample_flatline_refine_crops(
    ts_ds: Dataset,
    *,
    patch_width: int,
    segment_stride: int,
    max_scale: float,
    coarse_h: int,
    std_floor: float,
    keep_frac: float,
    seed: int,
    min_run: int = DEFAULT_MIN_RUN,
    flat_eps_frac: float = DEFAULT_EPS_FRAC,
) -> Tuple[Dict[int, List[int]], dict]:
    """Keep wiggle refine crops + keep_frac of flat crops; return segment allow-set.

    Unit is (absolute_segment_start, active_variate) with flatness judged only on
    the ``patch_width`` crop span. Does not drop whole H-horizon parents.
    """
    from models.diffusion_tsf.patch_refine_segments import iter_unique_segment_starts

    data = getattr(ts_ds, "data", None)
    if data is None:
        raise TypeError(
            "undersample_flatline_refine_crops requires a dataset with .data (T,V)"
        )
    data_np = (
        data.detach().cpu().numpy()
        if isinstance(data, torch.Tensor)
        else np.asarray(data, dtype=np.float32)
    )
    lookback = int(ts_ds.lookback)
    horizon = int(ts_ds.horizon)
    overlap = int(ts_ds.lookback_overlap)
    pw = int(patch_width)
    seg_stride = max(1, int(segment_stride))

    segment_starts = iter_unique_segment_starts(
        int(data_np.shape[0]),
        lookback=lookback,
        horizon=horizon,
        overlap=overlap,
        patch_width=pw,
        segment_stride=seg_stride,
    )
    if not segment_starts:
        raise RuntimeError("zero unique refine segments for flatline undersample")

    flat_mask = classify_unique_segment_flatline_crops(
        data_np,
        segment_starts,
        lookback=lookback,
        horizon=horizon,
        overlap=overlap,
        patch_width=pw,
        max_scale=max_scale,
        coarse_h=coarse_h,
        std_floor=std_floor,
        min_run=min_run,
        flat_eps_frac=flat_eps_frac,
    )
    pairs = select_flatline_undersample_pairs(
        flat_mask, keep_frac=keep_frac, seed=seed
    )
    if not pairs:
        raise RuntimeError("flatline undersample kept zero (segment, variate) crops")

    allowed = allowed_segment_variates_from_pairs(segment_starts, pairs)
    allowed_lists = {
        int(t): sorted(int(x) for x in vs) for t, vs in sorted(allowed.items())
    }
    n_flat = int(flat_mask.sum())
    n_wiggle = int((~flat_mask).sum())
    n_flat_kept = sum(1 for i, v in pairs if flat_mask[i, v])
    n_wiggle_kept = sum(1 for i, v in pairs if not flat_mask[i, v])
    per_var = _per_variate_stats(flat_mask, pairs)
    stats = {
        "semantics": "per_refine_crop",
        "patch_width": pw,
        "segment_stride": seg_stride,
        "n_segments": int(flat_mask.shape[0]),
        "n_variates": int(flat_mask.shape[1]),
        "n_crops": int(flat_mask.size),
        "n_flatline": n_flat,
        "n_wiggle": n_wiggle,
        "n_flatline_kept": n_flat_kept,
        "n_wiggle_kept": n_wiggle_kept,
        "n_kept_crops": len(pairs),
        "n_segments_kept": len(allowed_lists),
        "keep_frac": float(keep_frac),
        "seed": int(seed),
        "min_run": int(min_run),
        "flat_eps_frac": float(flat_eps_frac),
        "flat_keep_rate": (n_flat_kept / n_flat) if n_flat else float("nan"),
        "allowed_segment_variates": allowed_lists,
        "kept_pairs": [(int(i), int(v)) for i, v in pairs],
        "per_variate": per_var,
        # Backward-compatible aliases used by older log lines / verify scripts.
        "n_windows": int(flat_mask.shape[0]),
        "n_pairs": int(flat_mask.size),
        "n_kept_pairs": len(pairs),
        "n_kept": len(pairs),
    }
    logger.info(
        "  [flatline_undersample] crops: segments=%d vars=%d crops=%d "
        "flat=%d wiggle=%d kept_flat=%d kept_wiggle=%d kept_crops=%d "
        "segments_kept=%d pw=%d stride=%d keep_frac=%.3f seed=%s "
        "flat_keep_rate=%.3f",
        stats["n_segments"],
        stats["n_variates"],
        stats["n_crops"],
        n_flat,
        n_wiggle,
        n_flat_kept,
        n_wiggle_kept,
        stats["n_kept_crops"],
        stats["n_segments_kept"],
        pw,
        seg_stride,
        keep_frac,
        seed,
        stats["flat_keep_rate"],
    )
    for row in per_var:
        logger.info(
            "  [flatline_undersample] var=%d flat=%d wiggle=%d kept_flat=%d "
            "kept_wiggle=%d kept=%d flat_keep_rate=%.3f pct_kept=%.1f",
            row["variate"],
            row["n_flatline"],
            row["n_wiggle"],
            row["n_flatline_kept"],
            row["n_wiggle_kept"],
            row["n_kept"],
            row["flat_keep_rate"],
            row["pct_kept"],
        )
    return allowed_lists, stats


def undersample_flatline_train_windows(
    train_ds: Dataset,
    *,
    max_scale: float,
    coarse_h: int,
    std_floor: float,
    forecast_length: int,
    lookback_overlap: int,
    keep_frac: float,
    seed: int,
    min_run: int = DEFAULT_MIN_RUN,
    flat_eps_frac: float = DEFAULT_EPS_FRAC,
) -> Tuple[Dataset, dict]:
    """Deprecated full-horizon (window, var) undersample. Prefer crop API."""
    flat_mask = classify_timeseries_flatline_windows(
        train_ds,
        max_scale=max_scale,
        coarse_h=coarse_h,
        std_floor=std_floor,
        forecast_length=forecast_length,
        lookback_overlap=lookback_overlap,
        min_run=min_run,
        flat_eps_frac=flat_eps_frac,
    )
    pairs = select_flatline_undersample_pairs(
        flat_mask, keep_frac=keep_frac, seed=seed
    )
    if not pairs:
        raise RuntimeError("flatline undersample kept zero (window, variate) pairs")

    window_indices = sorted({int(i) for i, _v in pairs})
    allowed = allowed_parent_variates_from_pairs(train_ds, pairs)
    n_flat = int(flat_mask.sum())
    n_wiggle = int((~flat_mask).sum())
    n_flat_kept = sum(1 for i, v in pairs if flat_mask[i, v])
    n_wiggle_kept = sum(1 for i, v in pairs if not flat_mask[i, v])
    per_var = _per_variate_stats(flat_mask, pairs)
    stats = {
        "semantics": "per_parent_horizon_deprecated",
        "n_windows": int(flat_mask.shape[0]),
        "n_variates": int(flat_mask.shape[1]),
        "n_pairs": int(flat_mask.size),
        "n_flatline": n_flat,
        "n_wiggle": n_wiggle,
        "n_flatline_kept": n_flat_kept,
        "n_wiggle_kept": n_wiggle_kept,
        "n_kept_pairs": len(pairs),
        "n_windows_kept": len(window_indices),
        "n_kept": len(window_indices),
        "keep_frac": float(keep_frac),
        "seed": int(seed),
        "min_run": int(min_run),
        "flat_eps_frac": float(flat_eps_frac),
        "flat_keep_rate": (n_flat_kept / n_flat) if n_flat else float("nan"),
        "parent_starts": parent_starts_from_timeseries_indices(train_ds, window_indices),
        "allowed_parent_variates": {
            int(p): sorted(int(x) for x in vs) for p, vs in sorted(allowed.items())
        },
        "kept_pairs": [(int(i), int(v)) for i, v in pairs],
        "per_variate": per_var,
    }
    logger.info(
        "  [flatline_undersample/deprecated_horizon] windows=%d vars=%d pairs=%d "
        "flat=%d wiggle=%d kept_flat=%d kept_wiggle=%d kept_pairs=%d "
        "windows_kept=%d keep_frac=%.3f seed=%s flat_keep_rate=%.3f",
        stats["n_windows"],
        stats["n_variates"],
        stats["n_pairs"],
        n_flat,
        n_wiggle,
        n_flat_kept,
        n_wiggle_kept,
        stats["n_kept_pairs"],
        stats["n_windows_kept"],
        keep_frac,
        seed,
        stats["flat_keep_rate"],
    )
    return Subset(train_ds, window_indices), stats
