"""Classifier-free guidance helpers for inference."""

from __future__ import annotations

from typing import Optional

CFG_SCALE_EPS = 1e-6


def cfg_mix_applies(cfg_scale: float, eps: float = CFG_SCALE_EPS) -> bool:
    """True when output should mix conditional and null forwards (w != 1)."""
    return abs(float(cfg_scale) - 1.0) > eps


def resolve_effective_cfg_scale(
    cfg_scale: Optional[float],
    config_scale: float,
    use_cfg_inference: bool,
) -> float:
    """Scale passed to sampling; 1.0 means conditional-only path."""
    if cfg_scale is not None:
        return float(cfg_scale)
    if use_cfg_inference:
        return float(config_scale)
    return 1.0
