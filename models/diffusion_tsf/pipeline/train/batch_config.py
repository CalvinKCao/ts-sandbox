"""YAML-driven finetune micro-batch settings (replaces GPU batch probing)."""

from __future__ import annotations

from typing import Any, Mapping, Optional

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


def configured_phase_micro_batch(
    state: PipelineState,
    smoke_test: bool,
    phase_overrides: Optional[Mapping[str, Any]] = None,
) -> int:
    """Phase micro-batch ceiling: per-dataset probe dump, else training YAML.

    ``batch_size_by_dataset`` is the usable window batch after the old GPU
    probe (max_fit * headroom). ``lr_only`` still splits that ceiling against
    ``target_univariate_batch``. Mutually exclusive with
    ``probe_train_batch_size``.
    """
    overrides = dict(phase_overrides or {})
    if bool(overrides.get("probe_train_batch_size", False)) and overrides.get(
        "batch_size_by_dataset"
    ):
        raise ValueError(
            "probe_train_batch_size and batch_size_by_dataset cannot both be set"
        )
    by_ds = overrides.get("batch_size_by_dataset")
    if by_ds is not None:
        if not isinstance(by_ds, dict) or not by_ds:
            raise ValueError(
                "batch_size_by_dataset must be a non-empty dataset -> int map"
            )
        if state.dataset not in by_ds:
            raise ValueError(
                f"batch_size_by_dataset missing {state.dataset!r}; "
                f"have {sorted(by_ds)}"
            )
        bs = max(1, int(by_ds[state.dataset]))
        if smoke_test:
            return max(1, min(2, bs))
        return bs
    return configured_finetune_micro_batch(state, smoke_test)


def configured_max_diffusion_batch(state: PipelineState, smoke_test: bool) -> int:
    """Planning ceiling for lr_eff_batch_univariate (former GPU-probe upper bound)."""
    if smoke_test:
        return configured_finetune_micro_batch(state, smoke_test=True)

    explicit = training_value(state, "finetune_max_micro_batch", None)
    if explicit is not None:
        return max(1, int(explicit))

    return configured_finetune_micro_batch(state, smoke_test=False)
