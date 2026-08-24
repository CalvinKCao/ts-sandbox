"""
Diffusion-based Time Series Forecasting using 2D binary CDF images and FactorizedDiT.

Package import is lazy so login-node helpers (MMPD viz preflight) can load YAML
without importing torch.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "DiffusionTSFConfig",
    "TimeSeriesTo2D",
    "BinaryDiffusionScheduler",
    "DiffusionTSF",
    "compute_metrics",
    "shape_preservation_score",
]

_LAZY = {
    "DiffusionTSFConfig": ".config",
    "TimeSeriesTo2D": ".preprocessing",
    "BinaryDiffusionScheduler": ".diffusion",
    "DiffusionTSF": ".diffusion_model",
    "compute_metrics": ".metrics",
    "shape_preservation_score": ".metrics",
}


def __getattr__(name: str) -> Any:
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(mod, __name__), name)
    globals()[name] = value
    return value
