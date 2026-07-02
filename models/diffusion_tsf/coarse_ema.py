"""Causal EMA smoothing for decoded coarse 1D series (fine-stage target design)."""

from __future__ import annotations

from typing import Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def causal_ema_1d(
    series: ArrayLike,
    alpha: float,
    *,
    init: ArrayLike | None = None,
) -> np.ndarray:
    """Causal EMA along the last axis.

    ``s[t] = alpha * x[t] + (1 - alpha) * s[t-1]``. Higher ``alpha`` tracks the
    input more closely; lower ``alpha`` is smoother.

    ``init`` seeds ``s[0]``. When omitted, ``s[0] = x[0]``.
    """
    alpha = float(alpha)
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    x = _to_numpy(series).astype(np.float64, copy=False)
    out = np.empty_like(x)
    if init is None:
        out[..., 0] = x[..., 0]
        start = 1
    else:
        init_arr = _to_numpy(init).astype(np.float64, copy=False)
        out[..., 0] = init_arr
        start = 1
    for t in range(start, x.shape[-1]):
        out[..., t] = alpha * x[..., t] + (1.0 - alpha) * out[..., t - 1]
    return out


def causal_ema_with_past_seed(
    past_tail: ArrayLike,
    future_series: ArrayLike,
    alpha: float,
) -> np.ndarray:
    """EMA on ``future_series`` with ``s[0]`` seeded from the last past value.

    Prevents a discontinuity at the lookback→forecast handoff when smoothing
    decoded coarse steps for the fine-stage residual target.
    """
    past_tail = _to_numpy(past_tail)
    future_series = _to_numpy(future_series)
    if np.ndim(past_tail) == 0:
        seed = past_tail
    else:
        if past_tail.shape[:-1] != future_series.shape[:-1]:
            raise ValueError(
                f"past_tail and future_series batch dims differ: "
                f"{past_tail.shape[:-1]} vs {future_series.shape[:-1]}"
            )
        seed = past_tail[..., -1]
    return causal_ema_1d(future_series, alpha, init=seed)


def fine_residual_vs_smoothed_coarse(
    gt: ArrayLike,
    coarse_decoded: ArrayLike,
    *,
    past_tail: ArrayLike,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (raw_coarse, smoothed_coarse, residual) for fine-target design.

    ``residual = gt - smoothed_coarse``; conditioning still uses the unsmoothed
    coarse 2D map elsewhere in the pipeline.
    """
    gt_arr = _to_numpy(gt)
    coarse_arr = _to_numpy(coarse_decoded)
    smooth = causal_ema_with_past_seed(past_tail, coarse_arr, alpha)
    residual = gt_arr - smooth
    return coarse_arr, smooth, residual
