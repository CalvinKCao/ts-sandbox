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
    """Phase micro-batch ceiling: per-dataset U map, else window-unit YAML.

    ``max_univariate_micro_batch_by_dataset`` is max univariate DiT rows (U)
    that fit one fwd+bwd. ``batch_size_by_dataset`` is the older window-unit
    map. The two cannot be set together. Mutually exclusive with
    ``probe_train_batch_size``.
    """
    overrides = dict(phase_overrides or {})
    u_by_ds = overrides.get("max_univariate_micro_batch_by_dataset")
    by_ds = overrides.get("batch_size_by_dataset")
    if u_by_ds is not None and by_ds is not None:
        raise ValueError(
            "max_univariate_micro_batch_by_dataset and batch_size_by_dataset "
            "cannot both be set"
        )
    if bool(overrides.get("probe_train_batch_size", False)) and by_ds:
        raise ValueError(
            "probe_train_batch_size and batch_size_by_dataset cannot both be set"
        )
    if bool(overrides.get("probe_train_batch_size", False)) and u_by_ds:
        raise ValueError(
            "probe_train_batch_size and max_univariate_micro_batch_by_dataset "
            "cannot both be set"
        )
    chosen = u_by_ds if u_by_ds is not None else by_ds
    key_name = (
        "max_univariate_micro_batch_by_dataset"
        if u_by_ds is not None
        else "batch_size_by_dataset"
    )
    if chosen is not None:
        if not isinstance(chosen, dict) or not chosen:
            raise ValueError(f"{key_name} must be a non-empty dataset -> int map")
        if state.dataset not in chosen:
            raise ValueError(
                f"{key_name} missing {state.dataset!r}; have {sorted(chosen)}"
            )
        bs = max(1, int(chosen[state.dataset]))
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
