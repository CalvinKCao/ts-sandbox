"""Flatline-preserving blur for decoded coarse 1D (fine-stage target design).

Coarse decode is piecewise-constant. Symmetric blur on the raw series smears
plateaus; causal EMA shifts the curve forward. Instead:

1. RLE into constant runs.
2. Collapse each run to one skeleton sample (flatlines with length >= 2 are a
   single point — excluded from within-run mixing).
3. Symmetric blur on the skeleton (optionally left-padded with past seed).
4. Expand: every timestep in run *i* gets skeleton_blur[i] (flatlines restored).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from models.diffusion_tsf.coarse_ema import _to_numpy

ArrayLike = Union[np.ndarray, float]
KernelName = Literal["gaussian", "box"]


@dataclass(frozen=True)
class ConstantRun:
    start: int
    end: int
    value: float

    @property
    def length(self) -> int:
        return self.end - self.start

    def is_flatline(self, min_flat_len: int) -> bool:
        return self.length >= int(min_flat_len)


def segment_constant_runs(
    series: ArrayLike,
    *,
    atol: float = 0.0,
) -> List[ConstantRun]:
    x = _to_numpy(series).astype(np.float64, copy=False).reshape(-1)
    if x.size == 0:
        return []
    runs: List[ConstantRun] = []
    start = 0
    for i in range(1, x.size):
        if not np.isclose(x[i], x[start], atol=atol, rtol=0.0):
            runs.append(ConstantRun(start=start, end=i, value=float(x[start])))
            start = i
    runs.append(ConstantRun(start=start, end=x.size, value=float(x[start])))
    return runs


def _blur_kernel(radius: int, kernel: KernelName) -> np.ndarray:
    if radius < 1:
        return np.array([1.0], dtype=np.float64)
    size = 2 * int(radius) + 1
    if kernel == "box":
        k = np.ones(size, dtype=np.float64)
    elif kernel == "gaussian":
        t = np.arange(size, dtype=np.float64) - float(radius)
        sigma = max(float(radius) / 2.0, 1.0)
        k = np.exp(-0.5 * (t / sigma) ** 2)
    else:
        raise ValueError(f"unknown kernel {kernel!r}")
    k /= k.sum()
    return k


def symmetric_blur_1d(values: np.ndarray, *, radius: int, kernel: KernelName) -> np.ndarray:
    """Centered 1D blur — no forward phase shift."""
    if values.size == 0:
        return values.copy()
    if radius < 1:
        return values.copy()
    k = _blur_kernel(radius, kernel)
    pad = int(radius)
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, k, mode="valid")


def _skeleton_values_for_runs(series: np.ndarray, runs: Sequence[ConstantRun]) -> np.ndarray:
    return np.array([float(np.mean(series[r.start : r.end])) for r in runs], dtype=np.float64)


def flatline_preserving_blur(
    series: ArrayLike,
    *,
    flatline_source: Optional[ArrayLike] = None,
    past_seed: Optional[ArrayLike] = None,
    blur_radius: int = 1,
    min_flat_len: int = 2,
    atol: float = 0.0,
    kernel: KernelName = "gaussian",
) -> Tuple[np.ndarray, List[ConstantRun]]:
    """Blur ``series`` (typically decoded coarse) with flatline-aware collapse.

    Run boundaries come from ``flatline_source`` when set (use GT coarse+fine
    decode); otherwise from ``series``. Skeleton values are means of ``series``
    over each run span.

    Returns ``(smoothed, runs)`` on the same time grid as ``series``.
    """
    x = _to_numpy(series).astype(np.float64, copy=False).reshape(-1)
    source = x if flatline_source is None else _to_numpy(flatline_source).astype(np.float64).reshape(-1)
    if source.shape != x.shape:
        raise ValueError(f"flatline_source shape {source.shape} != series shape {x.shape}")
    runs = segment_constant_runs(source, atol=atol)
    if not runs:
        return x.copy(), runs

    skeleton = _skeleton_values_for_runs(x, runs)
    if past_seed is not None:
        seed = float(_to_numpy(past_seed).reshape(-1)[0])
        skeleton = np.concatenate([np.array([seed], dtype=np.float64), skeleton])
    blurred = symmetric_blur_1d(skeleton, radius=int(blur_radius), kernel=kernel)
    if past_seed is not None:
        blurred = blurred[1:]

    out = np.empty_like(x)
    for run, rep in zip(runs, blurred):
        out[run.start : run.end] = rep
    return out, runs


def fine_residual_vs_flatline_blur_coarse(
    gt: ArrayLike,
    coarse_decoded: ArrayLike,
    *,
    gt_combined: Optional[ArrayLike] = None,
    past_seed: Optional[ArrayLike] = None,
    blur_radius: int = 1,
    min_flat_len: int = 2,
    atol: float = 0.0,
    kernel: KernelName = "gaussian",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, List[ConstantRun]]:
    """Return ``(raw_coarse, smoothed_coarse, residual, runs)``.

    ``gt_combined`` should be the GT coarse+fine 1D decode used to find flatlines.
    """
    gt_arr = _to_numpy(gt).reshape(-1)
    coarse_arr = _to_numpy(coarse_decoded).reshape(-1)
    flat_src = gt_arr if gt_combined is None else _to_numpy(gt_combined).reshape(-1)
    smooth, runs = flatline_preserving_blur(
        coarse_arr,
        flatline_source=flat_src,
        past_seed=past_seed,
        blur_radius=blur_radius,
        min_flat_len=min_flat_len,
        atol=atol,
        kernel=kernel,
    )
    return coarse_arr, smooth, gt_arr - smooth, runs


def flatline_preserving_blur_torch(
    series: torch.Tensor,
    *,
    flatline_source: Optional[torch.Tensor] = None,
    past_seed: Optional[torch.Tensor] = None,
    blur_radius: int = 4,
    atol: float = 0.0,
    kernel: KernelName = "gaussian",
) -> torch.Tensor:
    """Batched flatline-preserving blur; same semantics as ``flatline_preserving_blur``."""
    if series.dim() < 1:
        raise ValueError(f"series must be at least 1D, got {series.shape}")
    orig_shape = series.shape
    t_len = int(orig_shape[-1])
    flat = series.reshape(-1, t_len)
    n_series = flat.shape[0]
    source = flat if flatline_source is None else flatline_source.reshape(-1, t_len)
    if source.shape != flat.shape:
        raise ValueError(f"flatline_source shape {source.shape} != series shape {flat.shape}")

    seeds: Optional[torch.Tensor] = None
    if past_seed is not None:
        seeds = past_seed.reshape(-1)
        if seeds.numel() not in (1, n_series):
            raise ValueError(f"past_seed must have 1 or {n_series} elements, got {seeds.numel()}")

    out = torch.empty_like(flat)
    for i in range(n_series):
        seed_val = None
        if seeds is not None:
            seed_val = float(seeds[i if seeds.numel() > 1 else 0].detach().cpu().item())
        smooth_np, _ = flatline_preserving_blur(
            flat[i].detach().cpu().numpy(),
            flatline_source=source[i].detach().cpu().numpy(),
            past_seed=seed_val,
            blur_radius=int(blur_radius),
            atol=float(atol),
            kernel=kernel,
        )
        out[i] = torch.from_numpy(smooth_np).to(device=flat.device, dtype=flat.dtype)
    return out.reshape(orig_shape)
