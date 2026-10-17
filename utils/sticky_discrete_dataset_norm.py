"""Sticky instance-norm floors + discrete coarse-bin affine (raw series).

Discrete (<4 unique values) takes precedence: skip per-window IN and map the
two most common values onto adjacent coarse-bin midpoints.

Sticky (consecutive-equal fraction > threshold, not discrete) keep instance
norm, with a per-variate std floor so <5% of train windows clip at ±max_scale.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from utils.hybrid_flat_dataset_norm import consecutive_flat_fractions


def coarse_bin_centers(max_scale: float, coarse_height: int) -> np.ndarray:
    h = int(coarse_height)
    ms = float(max_scale)
    if h < 2:
        raise ValueError(f"coarse_height must be >= 2, got {h}")
    if not np.isfinite(ms) or ms <= 0.0:
        raise ValueError(f"max_scale must be > 0, got {max_scale!r}")
    width = (2.0 * ms) / h
    return (np.arange(h, dtype=np.float64) + 0.5) * width - ms


def n_unique_per_variate(train_raw: np.ndarray) -> np.ndarray:
    x = np.asarray(train_raw)
    if x.ndim != 2:
        raise ValueError(f"train_raw must be (T, V), got {x.shape}")
    return np.array([np.unique(x[:, v]).size for v in range(x.shape[1])], dtype=np.int64)


def _two_most_common(series: np.ndarray) -> np.ndarray:
    vals, counts = np.unique(np.asarray(series), return_counts=True)
    order = np.argsort(-counts)
    return vals[order]


def _affine_value_to_center(value: float, center: float) -> tuple[float, float]:
    return float(value) - float(center), 1.0


def _affine_two_values_to_centers(
    v0: float, v1: float, c0: float, c1: float,
) -> tuple[float, float]:
    if v0 == v1:
        raise ValueError("need two distinct values for a 2-point affine")
    if c0 == c1:
        raise ValueError("need two distinct bin centers")
    std = (v1 - v0) / (c1 - c0)
    if std == 0.0 or not np.isfinite(std):
        raise ValueError(f"degenerate discrete affine std={std}")
    mean = v0 - c0 * std
    return float(mean), float(std)


def sticky_std_floor_for_series(
    series: np.ndarray,
    *,
    lookback: int,
    horizon: int,
    max_scale: float,
    clip_coverage: float,
    default_floor: float,
) -> Dict[str, float]:
    """Smallest floor (at least default) so >=clip_coverage windows avoid ±MS clip."""
    x = np.asarray(series, dtype=np.float64).reshape(-1)
    lb = int(lookback)
    hz = int(horizon)
    ms = float(max_scale)
    cov = float(clip_coverage)
    floor0 = float(default_floor)
    if lb < 1 or hz < 0:
        raise ValueError(f"lookback>0 and horizon>=0 required, got {lb}, {hz}")
    if not np.isfinite(ms) or ms <= 0.0:
        raise ValueError(f"max_scale must be > 0, got {max_scale!r}")
    if not (0.0 < cov <= 1.0):
        raise ValueError(f"clip_coverage must be in (0, 1], got {clip_coverage!r}")
    if not np.isfinite(floor0) or floor0 <= 0.0:
        raise ValueError(f"default_floor must be > 0, got {default_floor!r}")
    win = lb + hz
    if x.shape[0] < win:
        raise ValueError(f"series length {x.shape[0]} < lookback+horizon {win}")
    windows = np.lib.stride_tricks.sliding_window_view(x, win)
    past = windows[:, :lb]
    center = past.mean(axis=1)
    std = past.std(axis=1)
    max_abs = np.max(np.abs(windows - center[:, None]), axis=1)
    needed = max_abs / ms
    already_ok = std >= needed
    n_win = int(windows.shape[0])
    n_ok_target = int(np.ceil(cov * n_win))
    n_already = int(already_ok.sum())
    if n_already >= n_ok_target:
        floor = floor0
    else:
        n_need = n_ok_target - n_already
        candidates = np.sort(needed[~already_ok])
        if n_need > candidates.size:
            raise RuntimeError(
                f"cannot keep {cov:.2%} windows clip-free; "
                f"n_win={n_win} already_ok={n_already}"
            )
        floor = max(floor0, float(candidates[n_need - 1]))
    used = np.maximum(std, floor)
    clip_frac = float((used < needed).mean())
    if clip_frac + 1e-12 > (1.0 - cov):
        raise RuntimeError(
            f"sticky floor {floor:.6g} still clips {clip_frac:.4%} of windows "
            f"(need < {1.0 - cov:.4%}); n_win={n_win}"
        )
    return {
        "floor": floor,
        "clip_frac": clip_frac,
        "n_windows": float(n_win),
        "already_ok_frac": float(n_already) / float(n_win),
    }


def build_sticky_discrete_affine(
    train_raw: np.ndarray,
    *,
    lookback: int,
    horizon: int,
    max_scale: float,
    coarse_height: int,
    sticky_frac_threshold: float,
    clip_coverage: float,
    discrete_max_uniques: int,
    default_std_floor: float,
    empiric_eps: float = 1e-8,
) -> Dict[str, Any]:
    x = np.asarray(train_raw, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"train_raw must be (T, V), got {x.shape}")
    n_t, n_v = x.shape
    if n_t < 2:
        raise ValueError(f"need >=2 timesteps, got {n_t}")
    if not (0.0 < float(sticky_frac_threshold) < 1.0):
        raise ValueError(
            f"sticky_frac_threshold must be in (0, 1), got {sticky_frac_threshold!r}"
        )
    max_u = int(discrete_max_uniques)
    if max_u < 1:
        raise ValueError(f"discrete_max_uniques must be >= 1, got {max_u}")

    emp_mean = x.mean(axis=0, keepdims=True)
    emp_std = x.std(axis=0, keepdims=True) + float(empiric_eps)
    mean = emp_mean.copy()
    std = emp_std.copy()
    sticky_frac = consecutive_flat_fractions(x)
    n_unique = n_unique_per_variate(x)
    discrete_mask = n_unique <= max_u
    sticky_mask = (sticky_frac > float(sticky_frac_threshold)) & (~discrete_mask)
    skip_mask = discrete_mask.copy()
    floors = np.full(n_v, float(default_std_floor), dtype=np.float64)
    centers = coarse_bin_centers(max_scale, coarse_height)
    # Adjacent mid-canvas bins so two levels stay distinct and on-ladder.
    j0 = int(coarse_height) // 2 - 1
    c_lo, c_hi = float(centers[j0]), float(centers[j0 + 1])
    discrete_details: List[Dict[str, Any]] = []
    sticky_details: List[Dict[str, Any]] = []

    for v in range(n_v):
        if discrete_mask[v]:
            ranked = _two_most_common(x[:, v])
            if ranked.size == 1:
                mean[0, v], std[0, v] = _affine_value_to_center(float(ranked[0]), c_lo)
                mapped = [float(ranked[0])]
            else:
                pair = np.sort(ranked[:2])
                mean[0, v], std[0, v] = _affine_two_values_to_centers(
                    float(pair[0]), float(pair[1]), c_lo, c_hi,
                )
                mapped = [float(pair[0]), float(pair[1])]
            discrete_details.append({
                "variate": int(v),
                "n_unique": int(n_unique[v]),
                "mapped_values": mapped,
                "bin_centers": [c_lo, c_hi] if ranked.size > 1 else [c_lo],
            })
            continue
        if sticky_mask[v]:
            detail = sticky_std_floor_for_series(
                x[:, v],
                lookback=lookback,
                horizon=horizon,
                max_scale=max_scale,
                clip_coverage=clip_coverage,
                default_floor=default_std_floor,
            )
            floors[v] = detail["floor"]
            sticky_details.append({"variate": int(v), **detail, "sticky_frac": float(sticky_frac[v])})

    return {
        "mean": mean.astype(np.float64),
        "std": std.astype(np.float64),
        "emp_std": emp_std.astype(np.float64),
        "skip_window_norm_variate_mask": [bool(x) for x in skip_mask.tolist()],
        "window_norm_std_floor_per_variate": [float(x) for x in floors.tolist()],
        "discrete_mask": discrete_mask.astype(bool),
        "sticky_mask": sticky_mask.astype(bool),
        "sticky_frac": sticky_frac.astype(np.float64),
        "n_unique": n_unique,
        "discrete_details": discrete_details,
        "sticky_details": sticky_details,
        "lookback": int(lookback),
        "horizon": int(horizon),
        "max_scale": float(max_scale),
        "coarse_height": int(coarse_height),
        "sticky_frac_threshold": float(sticky_frac_threshold),
        "clip_coverage": float(clip_coverage),
        "discrete_max_uniques": max_u,
        "sticky_discrete_dataset_norm": True,
    }


def clamp_past_std(
    past_std: "torch.Tensor",
    scalar_floor: float,
    per_variate_floor: Optional[Sequence[float]] = None,
) -> "torch.Tensor":
    """Clamp lookback std with a scalar floor or a length-V floor list."""
    import torch

    if per_variate_floor is None:
        return past_std.clamp_min(float(scalar_floor))
    if len(per_variate_floor) != int(past_std.shape[1]):
        raise ValueError(
            f"window_norm_std_floor_per_variate length {len(per_variate_floor)} "
            f"!= num_variables {int(past_std.shape[1])}"
        )
    floors = torch.as_tensor(
        list(per_variate_floor), device=past_std.device, dtype=past_std.dtype,
    ).view(1, -1, 1)
    return past_std.clamp_min(floors)
