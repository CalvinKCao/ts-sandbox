"""Dataset affine that maps raw mode 0 to canvas floor (-max_scale).

x' = x / s_v - max_scale, with s_v = p99(train_v) / (2 * max_scale), so
raw 0 -> -MS and the 99th percentile of train timesteps hits +MS.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def build_mode_anchor_affine(
    train_raw: np.ndarray,
    *,
    max_scale: float,
) -> Dict[str, Any]:
    """Per-variate scale so <1% of train timesteps clip at the upper canvas bound.

    Inverse of the stored affine is x_raw = (x' + max_scale) * s_v, which is
    the usual z-score inverse with mean = max_scale * s_v and std = s_v.
    """
    x = np.asarray(train_raw, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"train_raw must be (T, V), got {x.shape}")
    ms = float(max_scale)
    if not np.isfinite(ms) or ms <= 0.0:
        raise ValueError(f"max_scale must be > 0, got {max_scale!r}")
    if x.shape[0] < 1:
        raise ValueError("train_raw needs at least 1 timestep")
    p99 = np.percentile(x, 99.0, axis=0, keepdims=True)
    bad = np.where(p99.reshape(-1) <= 0.0)[0].tolist()
    if bad:
        raise ValueError(
            f"mode_anchor_dataset_norm needs p99(train) > 0 so raw 0 can sit at "
            f"-max_scale; got <=0 at variates {bad}"
        )
    s_v = p99 / (2.0 * ms)
    mean = ms * s_v
    return {
        "mean": mean.astype(np.float64),
        "std": s_v.astype(np.float64),
        "p99": p99.astype(np.float64),
        "max_scale": ms,
        "mode_anchor_dataset_norm": True,
    }
