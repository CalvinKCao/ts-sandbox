"""Hybrid flat-variate dataset-level affine for window-norm canvas leaves.

Flat detection (train raw, before any affine):
  flat_frac[v] = mean(value[T] == value[T-1]) over train timesteps T>=1
  is_flat[v]   = flat_frac[v] > frac_threshold  (default 0.5)

For flat variates only: skip per-window / instance norm. Apply a single
global affine ``(x - mean) / scale`` with train mean and a scale chosen so
that >= ``oob_coverage`` (default 0.99) of train lookback windows have
**no OOB** values vs the canvas ladder.

OOB definition (matches training encode after the affine, with identity
window-norm): a window is OOB if any lookback value satisfies
``|x_aff| > max_scale`` (those values clip in ``encode_absolute_hir_cdf`` /
``TimeSeriesTo2D`` before binning onto ``[-max_scale, max_scale]``).

Scale formula for flat variate ``v``:
  max_abs_dev[t] = max_t'|x[t:t+L] - mean|
  needed[t]      = max_abs_dev[t] / max_scale
  scale          = max(percentile(needed, 100*oob_coverage), eps)

Non-flat variates keep empirical train std and existing window norm.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def consecutive_flat_fractions(train_raw: np.ndarray) -> np.ndarray:
    """Per-variate fraction of timesteps with value[T] == value[T-1]. Shape (V,)."""
    x = np.asarray(train_raw, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"train_raw must be (T, V), got {x.shape}")
    if x.shape[0] < 2:
        raise ValueError(f"need >=2 timesteps for flat detection, got {x.shape[0]}")
    return (x[1:] == x[:-1]).mean(axis=0)


def detect_flat_variate_mask(
    train_raw: np.ndarray,
    *,
    frac_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (is_flat bool (V,), flat_frac float (V,))."""
    if not (0.0 < float(frac_threshold) < 1.0):
        raise ValueError(f"frac_threshold must be in (0,1), got {frac_threshold}")
    frac = consecutive_flat_fractions(train_raw)
    return frac > float(frac_threshold), frac.astype(np.float64)


def coverage_scale_for_lookbacks(
    series: np.ndarray,
    *,
    mean: float,
    lookback: int,
    max_scale: float,
    oob_coverage: float = 0.99,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Pick global scale so >=oob_coverage of lookback windows are in [-MS, MS]."""
    x = np.asarray(series, dtype=np.float64).reshape(-1)
    lb = int(lookback)
    ms = float(max_scale)
    cov = float(oob_coverage)
    if lb < 1:
        raise ValueError(f"lookback must be >=1, got {lb}")
    if ms <= 0.0:
        raise ValueError(f"max_scale must be >0, got {ms}")
    if not (0.0 < cov <= 1.0):
        raise ValueError(f"oob_coverage must be in (0,1], got {cov}")
    if x.shape[0] < lb:
        raise ValueError(f"series length {x.shape[0]} < lookback {lb}")

    n_win = x.shape[0] - lb + 1
    # Stride view: (n_win, lb)
    windows = np.lib.stride_tricks.sliding_window_view(x, lb)
    max_abs = np.max(np.abs(windows - float(mean)), axis=1)
    needed = max_abs / ms
    q = 100.0 * cov
    scale = float(np.percentile(needed, q))
    scale = max(scale, float(eps))
    # Verify coverage under chosen scale (ties at the percentile boundary).
    in_bounds = (max_abs / scale) <= ms + 1e-9
    achieved = float(in_bounds.mean())
    if achieved + 1e-12 < cov:
        # Fail-fast: percentile edge cases with many ties — bump to exact needed quantile.
        scale = float(np.quantile(needed, cov))
        scale = max(scale, float(eps))
        in_bounds = (max_abs / scale) <= ms + 1e-9
        achieved = float(in_bounds.mean())
    if achieved + 1e-12 < cov:
        raise RuntimeError(
            f"coverage scale {scale:.6g} only keeps {achieved:.4%} lookbacks in-ladder "
            f"(need {cov:.4%}); n_win={n_win}"
        )
    return {
        "scale": scale,
        "achieved_coverage": achieved,
        "n_windows": float(n_win),
        "p99_max_abs_dev": float(np.percentile(max_abs, 99.0)),
        "max_abs_dev": float(max_abs.max()),
    }


def build_hybrid_affine_scales(
    train_raw: np.ndarray,
    *,
    lookback: int,
    max_scale: float,
    frac_threshold: float = 0.5,
    oob_coverage: float = 0.99,
    empiric_eps: float = 1e-8,
) -> Dict[str, Any]:
    """Compute per-variate mean/scale + flat mask for hybrid dataset affine.

    Returns dict with mean (1,V), std/scale (1,V), flat_mask (V,), flat_frac (V,),
    and per-flat coverage diagnostics.
    """
    x = np.asarray(train_raw, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"train_raw must be (T, V), got {x.shape}")
    mean = x.mean(axis=0, keepdims=True)
    std_emp = x.std(axis=0, keepdims=True) + float(empiric_eps)
    flat_mask, flat_frac = detect_flat_variate_mask(x, frac_threshold=frac_threshold)
    scales = std_emp.copy()
    flat_details: List[Dict[str, Any]] = []
    for v, is_flat in enumerate(flat_mask.tolist()):
        if not is_flat:
            continue
        detail = coverage_scale_for_lookbacks(
            x[:, v],
            mean=float(mean[0, v]),
            lookback=lookback,
            max_scale=max_scale,
            oob_coverage=oob_coverage,
            eps=empiric_eps,
        )
        scales[0, v] = detail["scale"]
        flat_details.append({"variate": int(v), **detail, "emp_std": float(std_emp[0, v])})
    return {
        "mean": mean.astype(np.float64),
        "std": scales.astype(np.float64),  # stored as norm_std in metadata
        "emp_std": std_emp.astype(np.float64),
        "flat_mask": flat_mask.astype(bool),
        "flat_frac": flat_frac.astype(np.float64),
        "flat_details": flat_details,
        "lookback": int(lookback),
        "max_scale": float(max_scale),
        "frac_threshold": float(frac_threshold),
        "oob_coverage": float(oob_coverage),
    }


def skip_window_norm_mask_from_stats(stats: Dict[str, Any]) -> Optional[List[bool]]:
    mask = stats.get("flat_variate_mask")
    if mask is None:
        return None
    return [bool(x) for x in np.asarray(mask).reshape(-1).tolist()]


def apply_skip_window_norm_mask(
    center: "torch.Tensor",
    std: "torch.Tensor",
    mask: Optional[Sequence[bool]],
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """For flat (skip) variates: identity in dataset-affine space (center=0, std=1)."""
    if mask is None:
        return center, std
    import torch

    if len(mask) != center.shape[1]:
        raise ValueError(
            f"skip_window_norm_mask length {len(mask)} != num_variables {center.shape[1]}"
        )
    if not any(mask):
        return center, std
    m = torch.tensor(list(mask), device=center.device, dtype=torch.bool).view(1, -1, 1)
    center = torch.where(m, torch.zeros_like(center), center)
    std = torch.where(m, torch.ones_like(std), std)
    return center, std
