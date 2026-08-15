"""Modular pipeline for diffusion TSF experiments.

Usage:
    from models.diffusion_tsf.pipeline import Pipeline, PipelineState, load_experiment_config
    from models.diffusion_tsf.pipeline.phases import PHASE_REGISTRY

    cfg = load_experiment_config(
        "configs/binary_window_norm_patch_refine_canvas128_p64x6.yaml",
        cli_overrides={...},
    )
    state = PipelineState.from_config(cfg)
    phases = [PHASE_REGISTRY[p["phase"]](**p) for p in cfg["phases"]]
    Pipeline(phases, state).run()
"""

from models.diffusion_tsf.pipeline.state import PipelineState
from models.diffusion_tsf.pipeline.phase import PipelinePhase
from models.diffusion_tsf.pipeline.orchestrator import Pipeline
from models.diffusion_tsf.pipeline.config import load_experiment_config

__all__ = ["PipelineState", "PipelinePhase", "Pipeline", "load_experiment_config"]
