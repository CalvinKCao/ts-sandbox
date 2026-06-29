"""Fourier staged representation helpers (fixed cutoff from config)."""

from __future__ import annotations

import logging

import torch

from models.diffusion_tsf.pipeline.state import PipelineState

logger = logging.getLogger(__name__)


def _window_normalize(
    past: torch.Tensor,
    future: torch.Tensor,
    *,
    center_mode: str,
    std_floor: float,
) -> torch.Tensor:
    if center_mode == "last":
        center = past[..., -1:]
    elif center_mode == "mean":
        center = past.mean(dim=-1, keepdim=True)
    else:
        raise ValueError(f"unknown window_norm_center {center_mode!r}")
    std = past.std(dim=-1, keepdim=True).clamp_min(float(std_floor))
    return (future - center) / std


def ensure_fourier_frequency_calibration(state: PipelineState, *, force: bool = False) -> None:
    """No-op: cutoff comes from fourier_high_freq_percent in config."""
    if state.staged_representation != "fourier_frequency":
        return
    state.fourier_high_freq_cutoff_bins_per_variate = None
    state.fourier_fine_max_scale_per_variate = None
    logger.debug(
        "fourier_frequency: fixed cutoff high_freq_percent=%.2f",
        float(state.fourier_high_freq_percent),
    )
