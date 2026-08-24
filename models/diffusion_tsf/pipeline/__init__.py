"""Modular pipeline for diffusion TSF experiments.

Package import is lazy so `python -m ...mmpd_viz_preflight` does not pull torch/numpy.
"""

from __future__ import annotations

from typing import Any

__all__ = ["PipelineState", "PipelinePhase", "Pipeline", "load_experiment_config"]

_LAZY = {
    "PipelineState": "models.diffusion_tsf.pipeline.state",
    "PipelinePhase": "models.diffusion_tsf.pipeline.phase",
    "Pipeline": "models.diffusion_tsf.pipeline.orchestrator",
    "load_experiment_config": "models.diffusion_tsf.pipeline.config",
}


def __getattr__(name: str) -> Any:
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(mod), name)
    globals()[name] = value
    return value
