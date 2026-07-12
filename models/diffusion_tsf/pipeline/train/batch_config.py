"""YAML-driven finetune micro-batch settings (replaces GPU batch probing)."""

from __future__ import annotations

from models.diffusion_tsf.pipeline.config import training_value
from models.diffusion_tsf.pipeline.state import PipelineState


def configured_finetune_micro_batch(state: PipelineState, smoke_test: bool) -> int:
    """Default micro-batch for fixed/lr_only paths and grad-accum multipliers."""
    default_bs = int(training_value(state, "diffusion_batch_size", 32))
    if smoke_test:
        return max(1, min(2, default_bs))

    finetune_sizes = training_value(state, "finetune_batch_sizes", None) or []
    if finetune_sizes:
        return max(1, max(int(s) for s in finetune_sizes))

    return max(1, default_bs)


def configured_max_diffusion_batch(state: PipelineState, smoke_test: bool) -> int:
    """Planning ceiling for lr_eff_batch_univariate (former GPU-probe upper bound)."""
    if smoke_test:
        return configured_finetune_micro_batch(state, smoke_test=True)

    explicit = training_value(state, "finetune_max_micro_batch", None)
    if explicit is not None:
        return max(1, int(explicit))

    return configured_finetune_micro_batch(state, smoke_test=False)
