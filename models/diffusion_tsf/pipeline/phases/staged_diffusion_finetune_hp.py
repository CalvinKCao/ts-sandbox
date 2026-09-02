"""HP tuning for staged coarse/fine diffusion models; best trial checkpoint is final."""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import random
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from optuna.exceptions import TrialPruned
from optuna.pruners import HyperbandPruner, MedianPruner, NopPruner
from optuna.samplers import TPESampler
from optuna.trial import TrialState
from torch.utils.data import DataLoader, Subset

from models.diffusion_tsf import train_multivariate_pipeline as pipeline_mod
from models.diffusion_tsf.patch_refine_segments import wrap_timeseries_as_unique_segments
from models.diffusion_tsf.pipeline.phase import PipelinePhase
from models.diffusion_tsf.pipeline.config import training_value
from models.diffusion_tsf.pipeline.optuna_parallel import run_optuna_study
from models.diffusion_tsf.pipeline.phase_diagnostics import run_phase_start_diagnostics
from models.diffusion_tsf.pipeline.reused_paths import (
    find_reused_tuned_params_meta,
    reused_root,
    reused_stage_best_ckpt,
)
from models.diffusion_tsf.pipeline.state import PipelineState
from models.diffusion_tsf.pipeline.data_subset import put_subset_record, random_window_subset
from models.diffusion_tsf.pipeline import wandb_utils
from models.diffusion_tsf.pipeline.phases.staged_diffusion_pretrain import (
    _stage_pretrain_ckpt,
    discover_dataset_run_ckpt_dir,
    stage_state,
)
from models.diffusion_tsf.pipeline.train.batch_config import (
    configured_max_diffusion_batch,
    configured_phase_micro_batch,
)
from models.diffusion_tsf.pipeline.train.univariate_microbatch import (
    dataloader_windows_for_univariate_rows,
    require_probed_univariate_u,
)
from models.diffusion_tsf.pipeline.train.checkpointing import (
    EarlyStopping,
    ensure_checkpoint_dir,
    save_checkpoint,
)
from models.diffusion_tsf.pipeline.train.diffusion_loop import (
    DiffusionEpochMetrics,
    DiffusionTrainer,
    log_epoch_shard_contract,
    make_grouped_train_loader,
)
from models.diffusion_tsf.pipeline.train.cross_variate_cache import (
    CrossVariateTokenCache,
    count_cache_windows,
)
from models.diffusion_tsf.pipeline.visualize_utils import (
    _load_staged_diffusion_from_ckpt,
    run_real_dataset_phase_diagnostics,
)
from models.diffusion_tsf.train_window_aug import (
    maybe_wrap_train_window_aug,
    set_train_window_aug_epoch,
)

logger = logging.getLogger(__name__)


def _has_train_window_augmentation(dataset) -> bool:
    current = dataset
    while current is not None:
        if current.__class__.__name__ == "TrainWindowAugDataset":
            return True
        current = getattr(current, "dataset", None) or getattr(current, "base", None)
    return False


def _hybrid_norm_metadata(norm_stats: Dict[str, Any]) -> Dict[str, Any]:
    """Persist hybrid flat-variate affine so disc/MMPD rematerialize the same scales."""
    if not norm_stats.get("hybrid_flat_dataset_norm"):
        return {}
    out: Dict[str, Any] = {
        "hybrid_flat_dataset_norm": True,
        "flat_variate_mask": [bool(x) for x in norm_stats["flat_variate_mask"].tolist()],
        "flat_variate_frac": [float(x) for x in norm_stats["flat_variate_frac"].tolist()],
        "hybrid_flat_frac_threshold": float(norm_stats["hybrid_flat_frac_threshold"]),
        "hybrid_flat_oob_coverage": float(norm_stats["hybrid_flat_oob_coverage"]),
        "hybrid_flat_max_scale": float(norm_stats["hybrid_flat_max_scale"]),
        "hybrid_flat_lookback": int(norm_stats["hybrid_flat_lookback"]),
        "hybrid_flat_details": list(norm_stats.get("hybrid_flat_details") or []),
    }
    if norm_stats.get("emp_std") is not None:
        out["emp_std"] = norm_stats["emp_std"].tolist()
    return out


def _maybe_subsample_patch_refine_train_windows(state: PipelineState, train_ds):
    """Keep a seeded random fraction of train windows for patch_refine finetune only."""
    frac = float(getattr(state, "patch_refine_finetune_window_fraction", 1.0))
    if not math.isfinite(frac) or frac <= 0.0 or frac > 1.0:
        raise ValueError(
            f"patch_refine_finetune_window_fraction must be in (0, 1], got {frac!r}"
        )
    if frac >= 1.0:
        return train_ds
    n = len(train_ds)
    if n <= 1:
        return train_ds
    k = max(1, int(round(n * frac)))
    k = min(k, n)
    rng = random.Random(int(state.seed) + 17)
    indices = sorted(rng.sample(range(n), k))
    logger.info(
        "  [diffusion_patch_refine_finetune_hp] train window fraction=%.3f: %d/%d "
        "(seed=%s)",
        frac,
        k,
        n,
        state.seed,
    )
    return Subset(train_ds, indices)


def _unpack_patch_refine_batch(batch):
    """Unpack train/val patch-refine batches with their absolute crop column."""
    if len(batch) == 3:
        past, future, patch_col0 = batch
    elif len(batch) == 2:
        past, future = batch
        patch_col0 = None
    else:
        raise ValueError(f"unexpected patch_refine batch length {len(batch)}")
    return past, future, patch_col0



def _probe_max_univariate_micro_batch(
    *,
    model,
    lookback: int,
    horizon: int,
    overlap: int,
    n_variates: int,
    device: torch.device,
    stage: str,
    min_bs: int = 1,
    max_bs: int = 16,
    headroom: float = 0.85,
) -> Dict[str, int]:
    """Largest univariate row count U that fits one train forward+backward."""
    if device.type != "cuda":
        raise RuntimeError("univariate micro-batch probe requires CUDA")
    fut_w = int(horizon) + int(overlap)
    patch_w = int(getattr(model.config, "patch_refine_patch_width", 8))
    unique = bool(getattr(model.config, "patch_refine_unique_segments", False))
    n_variates = max(1, int(n_variates))

    def _fits(u: int) -> bool:
        n_win = dataloader_windows_for_univariate_rows(u, n_variates)
        past = torch.randn(n_win, n_variates, lookback, device=device)
        future = torch.randn(n_win, n_variates, fut_w, device=device)
        row_index = torch.arange(int(u), device=device)
        patch_col0 = None
        if stage == "patch_refine" and unique:
            max_c0 = max(0, fut_w - patch_w)
            patch_col0 = torch.zeros(n_win, device=device, dtype=torch.long)
            if max_c0 > 0:
                patch_col0 = torch.randint(0, max_c0 + 1, (n_win,), device=device)
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            model.train()
            model.zero_grad(set_to_none=True)
            loss = model.get_loss(
                past, future, patch_col0=patch_col0, include_anchor=True,
                univariate_row_index=row_index,
            )
            loss.backward()
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            return True
        except RuntimeError as exc:
            err = str(exc).lower()
            if (
                "out of memory" not in err
                and "checkpoint" not in err
            ):
                raise
            torch.cuda.empty_cache()
            return False

    lo = max(1, int(min_bs))
    hi = max(lo, int(max_bs))
    if not _fits(lo):
        logger.warning(
            "finetune U probe: min_u=%d already OOMs; max_fit=0 usable=0", lo,
        )
        torch.cuda.empty_cache()
        return {"max_fit": 0, "usable": 0}
    best = lo
    cand = lo
    while cand * 2 <= hi and _fits(cand * 2):
        cand *= 2
        best = cand
    left, right = best, min(hi, best * 2)
    while left < right:
        mid = (left + right + 1) // 2
        if _fits(mid):
            left = mid
        else:
            right = mid - 1
    best = left
    usable = max(1, int(best * float(headroom)))
    logger.info(
        "finetune U probe: max_fit=%d headroom=%.2f -> U=%d",
        best,
        headroom,
        usable,
    )
    torch.cuda.empty_cache()
    return {"max_fit": int(best), "usable": int(usable)}


def _probe_max_finetune_batch_size(
    *,
    model,
    lookback: int,
    horizon: int,
    overlap: int,
    n_variates: int,
    device: torch.device,
    stage: str,
    min_bs: int = 1,
    max_bs: int = 16,
    headroom: float = 0.85,
) -> int:
    """Usable univariate micro-batch U after headroom (legacy int return)."""
    return int(
        _probe_max_univariate_micro_batch(
            model=model,
            lookback=lookback,
            horizon=horizon,
            overlap=overlap,
            n_variates=n_variates,
            device=device,
            stage=stage,
            min_bs=min_bs,
            max_bs=max_bs,
            headroom=headroom,
        )["usable"]
    )


def _log_gpu_mem(tag: str) -> None:
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    logger.info("  [%s] gpu_mem MiB: alloc=%.0f reserved=%.0f peak=%.0f", tag, alloc, reserved, peak)


def resolve_diffusion_batch_and_accum(
    probed_max: int,
    multiplier: Optional[float],
) -> Dict[str, int]:
    """Split probed max micro-batch and grad-accum steps to hit target effective batch."""
    probed_max = max(1, int(probed_max))
    if multiplier is None or float(multiplier) <= 1.0:
        return {
            "batch_size": probed_max,
            "gradient_accumulation_steps": 1,
            "effective_batch_size": probed_max,
        }

    target_effective = max(probed_max, int(round(probed_max * float(multiplier))))
    for accum in range(1, target_effective + 1):
        if target_effective % accum != 0:
            continue
        micro = target_effective // accum
        if micro <= probed_max:
            return {
                "batch_size": micro,
                "gradient_accumulation_steps": accum,
                "effective_batch_size": micro * accum,
            }

    accum = max(1, math.ceil(target_effective / probed_max))
    micro = max(1, min(probed_max, target_effective // accum))
    return {
        "batch_size": micro,
        "gradient_accumulation_steps": accum,
        "effective_batch_size": micro * accum,
    }


def _apply_effective_batch_multiplier(
    params: Dict[str, Any],
    max_batch_size: int,
    state: PipelineState,
) -> Dict[str, Any]:
    out = dict(params)
    batch_info = resolve_diffusion_batch_and_accum(
        max_batch_size,
        state.extra.get("diffusion_effective_batch_multiplier"),
    )
    out.update(batch_info)
    return out


def _effective_batch_tune_range(
    probed_max: int,
    state: PipelineState,
) -> Tuple[int, int]:
    lo_frac = float(training_value(state, "finetune_hp_effective_batch_min_frac", 0.25))
    hi_frac = float(training_value(state, "finetune_hp_effective_batch_max_frac", 4.0))
    probed_max = max(1, int(probed_max))
    lo = max(1, int(round(probed_max * lo_frac)))
    hi = max(lo, int(round(probed_max * hi_frac)))
    return lo, hi


def _candidate_micro_batches(probed_max: int, *, min_micro: int) -> list[int]:
    """Micro-batch sizes that divide probed_max cleanly or sit on a power-of-2 grid."""
    probed_max = max(1, int(probed_max))
    min_micro = max(1, min(int(min_micro), probed_max))
    micros: set[int] = set()
    for div in range(min_micro, probed_max + 1):
        if probed_max % div == 0:
            micros.add(div)
    power = 1
    while power <= probed_max:
        if power >= min_micro:
            micros.add(power)
        power *= 2
    micros.add(probed_max)
    return sorted(micros, reverse=True)


def enumerate_nice_effective_batch_plans(
    probed_max: int,
    lo: int,
    hi: int,
    *,
    min_micro: Optional[int] = None,
    max_accum: int = 64,
) -> list[Dict[str, int]]:
    """Effective batch sizes reachable as micro*accum with micro<=probed_max (no micro=1 traps).

    For each effective size keep the plan with the largest micro batch so epoch work stays low.
    """
    probed_max = max(1, int(probed_max))
    lo = max(1, int(lo))
    hi = max(lo, int(hi))
    if min_micro is None:
        min_micro = max(4, probed_max // 8)

    best_for_effective: Dict[int, Dict[str, int]] = {}
    for micro in _candidate_micro_batches(probed_max, min_micro=min_micro):
        accum_hi = min(max_accum, hi // micro)
        for accum in range(1, accum_hi + 1):
            effective = micro * accum
            if effective < lo or effective > hi:
                continue
            plan = {
                "batch_size": micro,
                "gradient_accumulation_steps": accum,
                "effective_batch_size": effective,
            }
            prev = best_for_effective.get(effective)
            if prev is None or micro > prev["batch_size"]:
                best_for_effective[effective] = plan

    return [best_for_effective[k] for k in sorted(best_for_effective)]


def resolve_target_effective_batch(
    probed_max: int,
    target_effective: int,
    *,
    lo: Optional[int] = None,
    hi: Optional[int] = None,
) -> Dict[str, int]:
    """Map a target effective batch to micro-batch + grad-accum (micro <= probed_max)."""
    probed_max = max(1, int(probed_max))
    target_effective = max(1, int(target_effective))
    if lo is None:
        lo = max(1, int(round(probed_max * 0.25)))
    if hi is None:
        hi = max(lo, int(round(probed_max * 4.0)))

    plans = enumerate_nice_effective_batch_plans(probed_max, lo, hi)
    if not plans:
        return {
            "batch_size": probed_max,
            "gradient_accumulation_steps": 1,
            "effective_batch_size": probed_max,
        }

    for plan in plans:
        if plan["effective_batch_size"] == target_effective:
            return dict(plan)

    return min(
        plans,
        key=lambda p: (
            abs(p["effective_batch_size"] - target_effective),
            -p["batch_size"],
        ),
    )


def _suggest_nice_effective_batch(
    trial,
    probed_max: int,
    state: PipelineState,
    *,
    param_name: str = "effective_batch_size",
    lo: Optional[int] = None,
    hi: Optional[int] = None,
    min_micro: Optional[int] = None,
) -> Dict[str, int]:
    if lo is None or hi is None:
        range_lo, range_hi = _effective_batch_tune_range(probed_max, state)
        lo = range_lo if lo is None else lo
        hi = range_hi if hi is None else hi
    plans = enumerate_nice_effective_batch_plans(probed_max, lo, hi, min_micro=min_micro)
    if not plans:
        return {
            "batch_size": probed_max,
            "gradient_accumulation_steps": 1,
            "effective_batch_size": probed_max,
        }
    effectives = [p["effective_batch_size"] for p in plans]
    chosen = trial.suggest_categorical(param_name, effectives)
    for plan in plans:
        if plan["effective_batch_size"] == chosen:
            return dict(plan)
    return dict(plans[0])


def _n_variates_for_batch(state: PipelineState) -> int:
    idxs = getattr(state, "variate_indices", None)
    if idxs:
        return max(1, len(idxs))
    return max(1, int(getattr(state, "n_variates", 1) or 1))


def _plan_univariate_effective_batch(
    *,
    probed_max_windows: int,
    n_variates: int,
    target_univariate: int,
    smoke_test: bool = False,
) -> Dict[str, int]:
    """Map target univariate batch U to micro-U + grad-accum.

    ``probed_max_windows`` is the max univariate row count that fits one
    fwd+bwd (legacy kwarg name). ``batch_size`` is that micro U, not window B.
    """
    n_variates = max(1, int(n_variates))
    probed_u = max(1, int(probed_max_windows))
    target_u = max(1, int(target_univariate))
    if smoke_test:
        micro = min(probed_u, target_u, 2)
        return {
            "batch_size": micro,
            "gradient_accumulation_steps": 1,
            "effective_batch_size": micro,
            "effective_univariate_batch": micro,
            "target_univariate_batch": int(target_univariate),
            "univariate_row_micro_batch": True,
            "dataloader_windows": dataloader_windows_for_univariate_rows(micro, n_variates),
        }

    micro = min(probed_u, target_u)
    accum = max(1, math.ceil(target_u / micro))
    actual_u = micro * accum
    return {
        "batch_size": micro,
        "gradient_accumulation_steps": accum,
        "effective_batch_size": actual_u,
        "effective_univariate_batch": actual_u,
        "target_univariate_batch": target_u,
        "univariate_row_micro_batch": True,
        "dataloader_windows": dataloader_windows_for_univariate_rows(micro, n_variates),
    }


def _effective_univariate_batch_grid(
    state: PipelineState,
    phase_overrides: Dict[str, Any],
) -> Tuple[list[int], Optional[int]]:
    """Return (grid, probed_U or None). Grid is [U, 2U, 4U] when multipliers are set."""
    multipliers = phase_overrides.get("effective_univariate_batch_multipliers")
    raw_grid = phase_overrides.get("effective_univariate_batch_grid")
    if multipliers is not None and raw_grid is not None:
        raise ValueError(
            "cannot set both effective_univariate_batch_grid and "
            "effective_univariate_batch_multipliers"
        )
    if multipliers is not None:
        by_u = phase_overrides.get("max_univariate_micro_batch_by_dataset")
        if not isinstance(by_u, dict) or not by_u:
            raise ValueError(
                "effective_univariate_batch_multipliers requires "
                "max_univariate_micro_batch_by_dataset"
            )
        if state.dataset not in by_u:
            raise ValueError(
                f"max_univariate_micro_batch_by_dataset missing {state.dataset!r}; "
                f"have {sorted(by_u)}"
            )
        probed_u = require_probed_univariate_u(by_u[state.dataset], dataset=state.dataset)
        grid = sorted({max(1, int(m)) * probed_u for m in multipliers})
        if not grid:
            raise ValueError("effective_univariate_batch_multipliers is empty")
        return grid, probed_u
    if not raw_grid:
        raise ValueError(
            "search_space=lr_eff_batch_univariate requires "
            "effective_univariate_batch_grid or effective_univariate_batch_multipliers"
        )
    grid = sorted({int(x) for x in raw_grid})
    if not grid:
        raise ValueError("effective_univariate_batch_grid is empty")
    return grid, None


def _resolved_enqueue_trials(
    phase_overrides: Dict[str, Any],
    state: PipelineState,
) -> Optional[List[Dict[str, Any]]]:
    """Rewrite donor enqueue so 1x U is per-dataset, not a hardcoded 336."""
    trials = phase_overrides.get("enqueue_trials")
    if not trials:
        return trials
    multipliers = phase_overrides.get("effective_univariate_batch_multipliers")
    by_u = phase_overrides.get("max_univariate_micro_batch_by_dataset")
    if not multipliers or not isinstance(by_u, dict):
        return trials
    if state.dataset not in by_u:
        raise ValueError(
            f"max_univariate_micro_batch_by_dataset missing {state.dataset!r}; "
            f"have {sorted(by_u)}"
        )
    probed_u = require_probed_univariate_u(by_u[state.dataset], dataset=state.dataset)
    out = []
    for trial_params in trials:
        row = dict(trial_params)
        row["effective_univariate_batch"] = probed_u
        out.append(row)
    return out


def _suggest_lr_eff_batch_univariate(
    trial,
    state: PipelineState,
    max_batch_size: int,
    smoke_test: bool,
    phase_overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """Tune LR + effective univariate batch (U rows); other diffusion HP fixed."""
    lr_min = float(phase_overrides["hp_lr_min"])
    lr_max = float(phase_overrides["hp_lr_max"])
    grid, probed_u = _effective_univariate_batch_grid(state, phase_overrides)

    n_vars = _n_variates_for_batch(state)
    base_ms = float(state.max_scale_by_dataset.get(state.dataset, state.max_scale))
    target_u = int(trial.suggest_categorical("effective_univariate_batch", grid))
    plan_ceiling = int(probed_u) if probed_u is not None else int(max_batch_size)
    batch_plan = _plan_univariate_effective_batch(
        probed_max_windows=plan_ceiling,
        n_variates=n_vars,
        target_univariate=target_u,
        smoke_test=smoke_test,
    )
    lr = float(trial.suggest_float("learning_rate", lr_min, lr_max, log=True))

    return {
        "learning_rate": lr,
        "batch_size": int(batch_plan["batch_size"]),
        "gradient_accumulation_steps": int(batch_plan["gradient_accumulation_steps"]),
        "effective_batch_size": int(batch_plan["effective_batch_size"]),
        "effective_univariate_batch": int(batch_plan["effective_univariate_batch"]),
        "target_univariate_batch": int(batch_plan["target_univariate_batch"]),
        "univariate_row_micro_batch": True,
        "dataloader_windows": int(batch_plan["dataloader_windows"]),
        "ema_decay": float(phase_overrides.get("ema_decay", 0.995)),
        "binary_noise_schedule": str(phase_overrides.get("binary_noise_schedule", "linear")),
        "loss_weighting": str(phase_overrides.get("loss_weighting", "min_snr")),
        "min_snr_gamma": float(phase_overrides.get("min_snr_gamma", 2.0)),
        "prediction_target": str(phase_overrides.get("prediction_target", "x0")),
        "max_scale": base_ms,
        "binary_length_mode": str(
            phase_overrides.get("binary_length_mode", state.binary_length_mode)
        ),
        "binary_length_g": float(
            phase_overrides.get("binary_length_g", state.binary_length_g)
        ),
        "binary_length_scale": float(
            phase_overrides.get("binary_length_scale", state.binary_length_scale)
        ),
    }


def _suggest_lr_eff_batch_g(
    trial,
    state: PipelineState,
    max_batch_size: int,
    smoke_test: bool,
    phase_overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """Tune LR + univariate batch + continuous length_g in [hp_g_min, hp_g_max]."""
    params = _suggest_lr_eff_batch_univariate(
        trial, state, max_batch_size, smoke_test, phase_overrides,
    )
    g_lo = float(phase_overrides.get("hp_g_min", 1.0))
    g_hi = float(phase_overrides.get("hp_g_max", 10.0))
    if g_hi < g_lo:
        raise ValueError(f"hp_g_max ({g_hi}) must be >= hp_g_min ({g_lo})")
    params["binary_length_g"] = float(trial.suggest_float("binary_length_g", g_lo, g_hi))
    params["binary_length_mode"] = "power"
    return params


def _suggest_lr_eff_batch_univariate_ema(
    trial,
    state: PipelineState,
    max_batch_size: int,
    smoke_test: bool,
    phase_overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """Tune only LR, univariate effective batch, and EMA for staged diffusion."""
    params = _suggest_lr_eff_batch_univariate(
        trial, state, max_batch_size, smoke_test, phase_overrides,
    )
    raw_grid = phase_overrides.get("ema_decay_grid")
    if not raw_grid:
        raise ValueError(
            "search_space=lr_eff_batch_univariate_ema requires ema_decay_grid in phase YAML"
        )
    grid = sorted({float(x) for x in raw_grid})
    if any(value < 0.0 or value >= 1.0 for value in grid):
        raise ValueError(f"ema_decay_grid must be in [0, 1), got {grid}")
    params["ema_decay"] = float(trial.suggest_categorical("ema_decay", grid))
    return params


def _fraction_subset(ds, fraction: float, seed: int):
    """Deterministic subset of a dataset (same idea as staged_eval)."""
    n = len(ds)
    frac = float(fraction)
    if frac >= 1.0:
        return ds
    if frac <= 0.0:
        raise ValueError(f"fraction must be in (0, 1], got {frac}")
    keep = max(1, int(round(n * frac)))
    rng = np.random.default_rng(int(seed))
    idx = sorted(rng.choice(n, size=min(keep, n), replace=False).tolist())
    return Subset(ds, idx)





def _suggest_full_diffusion_params(
    trial,
    state: PipelineState,
    max_batch_size: int,
    *,
    tune_effective_batch: bool,
) -> Dict[str, Any]:
    base_ms = float(state.max_scale_by_dataset.get(state.dataset, state.max_scale))
    if training_value(state, "max_scale_tuning", False):
        rng = training_value(state, "max_scale_tuning_range", [2.5, 14.0])
        ms = trial.suggest_float("max_scale", float(rng[0]), float(rng[1]))
    else:
        ms = base_ms

    params: Dict[str, Any] = {
        "learning_rate": trial.suggest_float("learning_rate", 3e-6, 8e-4, log=True),
        "ema_decay": trial.suggest_categorical("ema_decay", [0.0, 0.99, 0.995, 0.999]),
        "binary_noise_schedule": trial.suggest_categorical(
            "binary_noise_schedule", ["linear", "cosine"]
        ),
        "loss_weighting": trial.suggest_categorical("loss_weighting", ["none", "min_snr"]),
        "prediction_target": trial.suggest_categorical("prediction_target", ["x0", "epsilon"]),
        "max_scale": ms,
    }
    params["min_snr_gamma"] = (
        trial.suggest_float("min_snr_gamma", 1.0, 10.0, log=True)
        if params["loss_weighting"] == "min_snr"
        else 5.0
    )

    if tune_effective_batch:
        params.update(_suggest_nice_effective_batch(trial, max_batch_size, state))
    else:
        batch_grid = [b for b in (4, 8, 16, 32, 48, 64, 96, 128) if b <= max_batch_size]
        if not batch_grid:
            batch_grid = [max(1, max_batch_size)]
        params["batch_size"] = trial.suggest_categorical("batch_size", batch_grid)
        params = _apply_effective_batch_multiplier(params, max_batch_size, state)

    return params


def _resolve_best_trial_ckpt(
    study,
    trials_dir: str,
    subset_dir: str,
    best_trial_num: int,
) -> str:
    """Locate the best Optuna trial checkpoint on disk."""
    candidates = [
        os.path.join(trials_dir, f"trial_{best_trial_num}_best.pt"),
        os.path.join(subset_dir, f"_diff_ft_trial_{best_trial_num}_best.pt"),
    ]
    best_trial = study.best_trial
    if best_trial is not None and int(best_trial.number) == int(best_trial_num):
        user_ckpt = best_trial.user_attrs.get("ckpt_path")
        if user_ckpt:
            candidates.insert(0, str(user_ckpt))

    for path in candidates:
        if path and os.path.isfile(path):
            return path

    for trial_dir in (trials_dir, subset_dir):
        if not os.path.isdir(trial_dir):
            continue
        for fn in sorted(os.listdir(trial_dir)):
            if fn == f"trial_{best_trial_num}_best.pt" or fn == f"_diff_ft_trial_{best_trial_num}_best.pt":
                return os.path.join(trial_dir, fn)
    raise RuntimeError(
        f"Best trial checkpoint missing for trial {best_trial_num}; "
        f"checked {candidates} under {trials_dir}"
    )


def _cleanup_trial_ckpts(trials_dir: str, subset_dir: str, *, keep: str) -> None:
    keep_abs = os.path.abspath(keep)
    for trial_dir in (trials_dir, subset_dir):
        if not os.path.isdir(trial_dir):
            continue
        for fn in os.listdir(trial_dir):
            if not (fn.startswith("trial_") or fn.startswith("_diff_ft_trial_")):
                continue
            if not fn.endswith("_best.pt"):
                continue
            path = os.path.abspath(os.path.join(trial_dir, fn))
            if path == keep_abs:
                continue
            try:
                os.remove(path)
            except OSError:
                pass


TUNED_MODEL_KEYS = (
    "max_scale",
    "binary_noise_schedule",
    "binary_length_mode",
    "binary_length_g",
    "binary_length_scale",
    "prediction_target",
    "loss_weighting",
    "min_snr_gamma",
    "dit_dropout",
)


def _stage_subset_dir(
    state: PipelineState,
    stage: str,
    *,
    checkpoint_dir: Optional[str] = None,
) -> str:
    subset_id = state.subset_id or state.dataset
    return os.path.join(checkpoint_dir or state.checkpoint_dir, subset_id, stage)


def _stage_best_ckpt(
    state: PipelineState,
    stage: str,
    *,
    checkpoint_dir: Optional[str] = None,
) -> str:
    return os.path.join(_stage_subset_dir(state, stage, checkpoint_dir=checkpoint_dir), "best.pt")


def _model_kwargs_from_tuned(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not params:
        return {}
    return {key: params[key] for key in TUNED_MODEL_KEYS if key in params}


def _state_anchor_kwargs(state: PipelineState) -> Dict[str, Any]:
    if not state.deterministic_anchor_loss:
        return {"use_deterministic_anchor_loss": False}
    return {
        "use_deterministic_anchor_loss": True,
        "deterministic_anchor_lambda": float(state.deterministic_anchor_lambda),
        "deterministic_anchor_alpha": float(state.deterministic_anchor_alpha),
    }


def _with_state_anchor_params(params: Dict[str, Any], state: PipelineState) -> Dict[str, Any]:
    out = dict(params)
    out.update(_state_anchor_kwargs(state))
    return out


def _load_reused_stage_params(
    state: PipelineState,
    *,
    stage: str,
    subset_id: str,
    source_config: str,
) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
    reused_meta = find_reused_tuned_params_meta(source_config, subset_id, stage)
    if reused_meta:
        with open(reused_meta, encoding="utf-8") as f:
            source_meta = json.load(f)
        params = dict(source_meta.get("tuned_params") or {})
        if not params:
            raise ValueError(f"No tuned_params in {reused_meta}")
        policy_ms = float(state.max_scale_by_dataset.get(state.dataset, state.max_scale))
        old_ms = params.get("max_scale")
        params["max_scale"] = policy_ms
        params.setdefault("min_snr_gamma", 5.0)
        return params, reused_root(), {**source_meta, "reused_max_scale_previous": old_ms}

    source_dir = discover_dataset_run_ckpt_dir(state, source_config)
    meta_path = os.path.join(source_dir, subset_id, stage, "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Missing {stage} metadata for reuse: {meta_path} "
            f"(from *-{state.dataset}-{source_config})"
        )
    with open(meta_path, encoding="utf-8") as f:
        source_meta = json.load(f)
    params = dict(source_meta.get("tuned_params") or {})
    if not params:
        raise ValueError(f"No tuned_params in {meta_path}")
    policy_ms = float(state.max_scale_by_dataset.get(state.dataset, state.max_scale))
    old_ms = params.get("max_scale")
    params["max_scale"] = policy_ms
    params.setdefault("min_snr_gamma", 5.0)
    return params, source_dir, {**source_meta, "reused_max_scale_previous": old_ms}


def _suggest_reduced_hp_params(
    trial,
    state: PipelineState,
    max_batch_size: int,
    smoke_test: bool,
    phase_overrides: Dict[str, Any],
) -> Dict[str, Any]:
    lr_min = float(phase_overrides["hp_lr_min"])
    lr_max = float(phase_overrides["hp_lr_max"])
    min_micro = int(phase_overrides.get("min_micro_batch", 4))
    min_snr_gamma = float(phase_overrides.get("min_snr_gamma", 2.0))

    base_ms = float(state.max_scale_by_dataset.get(state.dataset, state.max_scale))
    if training_value(state, "max_scale_tuning", False):
        rng = training_value(state, "max_scale_tuning_range", [2.5, 14.0])
        ms = trial.suggest_float("max_scale", float(rng[0]), float(rng[1]))
    else:
        ms = base_ms

    preset = trial.suggest_categorical("ema_prediction_preset", ["x0_0995", "epsilon_0999"])
    if preset == "x0_0995":
        ema_decay, prediction_target = 0.995, "x0"
    else:
        ema_decay, prediction_target = 0.999, "epsilon"

    params: Dict[str, Any] = {
        "learning_rate": trial.suggest_float("learning_rate", lr_min, lr_max, log=True),
        "ema_decay": ema_decay,
        "binary_noise_schedule": "linear",
        "loss_weighting": "min_snr",
        "min_snr_gamma": min_snr_gamma,
        "prediction_target": prediction_target,
        "max_scale": ms,
    }

    if smoke_test:
        smoke_hi = max(min_micro, min(4, max_batch_size))
        params.update(
            _suggest_nice_effective_batch(
                trial,
                max_batch_size,
                state,
                param_name="effective_batch_size",
                lo=1,
                hi=smoke_hi,
                min_micro=min_micro,
            )
        )
        return params

    params.update(
        _suggest_nice_effective_batch(
            trial, max_batch_size, state, min_micro=min_micro,
        )
    )
    return params


def _build_fixed_hp_params(
    state: PipelineState,
    max_batch_size: int,
    smoke_test: bool,
    phase_overrides: Dict[str, Any],
) -> Dict[str, Any]:
    fixed = dict(phase_overrides.get("fixed_tuned_params") or {})
    by_ds = phase_overrides.get("fixed_tuned_params_by_dataset") or {}
    if isinstance(by_ds, dict) and state.dataset in by_ds:
        fixed.update(dict(by_ds[state.dataset] or {}))
    if not fixed and not by_ds:
        raise ValueError(
            "search_space=fixed requires fixed_tuned_params "
            "(and/or fixed_tuned_params_by_dataset) in phase YAML"
        )
    if not fixed:
        raise ValueError(
            f"search_space=fixed: no fixed_tuned_params for dataset={state.dataset!r}"
        )
    missing_keys = [k for k in ("learning_rate",) if k not in fixed]
    if missing_keys:
        known = sorted(str(k) for k in by_ds.keys()) if isinstance(by_ds, dict) else []
        raise ValueError(
            f"search_space=fixed: dataset={state.dataset!r} missing required "
            f"keys {missing_keys} after merging fixed_tuned_params"
            f"{'' if not known else f' (HPs defined for: {known})'}"
        )
    params = dict(fixed)
    params.setdefault(
        "max_scale",
        float(state.max_scale_by_dataset.get(state.dataset, state.max_scale)),
    )

    target_u = params.pop("target_univariate_batch", None)
    if target_u is None:
        target_u = params.pop("effective_univariate_batch", None)
    if target_u is not None:
        batch_plan = _plan_univariate_effective_batch(
            probed_max_windows=max_batch_size,
            n_variates=_n_variates_for_batch(state),
            target_univariate=int(target_u),
            smoke_test=smoke_test,
        )
        params.update(batch_plan)
        if smoke_test:
            params["batch_size"] = min(int(params.get("batch_size", 1)), 2)
            params["gradient_accumulation_steps"] = 1
            params["effective_batch_size"] = int(params["batch_size"])
            params["effective_univariate_batch"] = int(params["batch_size"])
        return params

    if "batch_size" in params:
        micro = max(1, int(params["batch_size"]))
        accum = max(1, int(params.get("gradient_accumulation_steps", 1)))
        if smoke_test:
            micro = min(micro, 2)
            accum = 1
        params["batch_size"] = micro
        params["gradient_accumulation_steps"] = accum
        params["effective_batch_size"] = micro * accum
        return params

    if smoke_test:
        params["batch_size"] = min(int(params.get("batch_size", 1)), 2)
        params["gradient_accumulation_steps"] = 1
        params["effective_batch_size"] = int(params["batch_size"])
        return params
    return _apply_effective_batch_multiplier(params, max_batch_size, state)


def _suggest_staged_params(
    trial,
    state: PipelineState,
    max_batch_size: int,
    smoke_test: bool,
    search_space: str = "default",
    phase_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    overrides = phase_overrides or {}

    if search_space == "reduced_hp":
        return _suggest_reduced_hp_params(
            trial, state, max_batch_size, smoke_test, overrides,
        )

    if search_space == "lr_eff_batch_univariate":
        return _suggest_lr_eff_batch_univariate(
            trial, state, max_batch_size, smoke_test, overrides,
        )

    if search_space == "lr_eff_batch_g":
        return _suggest_lr_eff_batch_g(
            trial, state, max_batch_size, smoke_test, overrides,
        )

    if search_space == "lr_eff_batch_univariate_ema":
        return _suggest_lr_eff_batch_univariate_ema(
            trial, state, max_batch_size, smoke_test, overrides,
        )

    if search_space == "fixed":
        raise RuntimeError("_suggest_staged_params must not be called for search_space=fixed")

    if search_space == "full_with_batch":
        if smoke_test:
            smoke_hi = max(2, min(4, max_batch_size))
            params = _suggest_full_diffusion_params(
                trial, state, max_batch_size, tune_effective_batch=False,
            )
            params.update(
                _suggest_nice_effective_batch(
                    trial,
                    max_batch_size,
                    state,
                    param_name="effective_batch_size",
                    lo=1,
                    hi=smoke_hi,
                )
            )
            return params
        return _suggest_full_diffusion_params(
            trial, state, max_batch_size, tune_effective_batch=True,
        )

    base_ms = float(state.max_scale_by_dataset.get(state.dataset, state.max_scale))
    if training_value(state, "max_scale_tuning", False):
        rng = training_value(state, "max_scale_tuning_range", [2.5, 14.0])
        ms = trial.suggest_float("max_scale", float(rng[0]), float(rng[1]))
    else:
        ms = base_ms

    if search_space == "lr_only":
        if "hp_lr_min" not in overrides or "hp_lr_max" not in overrides:
            raise ValueError("search_space=lr_only requires phase hp_lr_min and hp_lr_max")
        lr_min = float(overrides["hp_lr_min"])
        lr_max = float(overrides["hp_lr_max"])
        if lr_max < lr_min:
            raise ValueError(f"hp_lr_max ({lr_max}) must be >= hp_lr_min ({lr_min})")
        lr = float(trial.suggest_float("learning_rate", lr_min, lr_max, log=True))
        params = {
            "learning_rate": lr,
            "ema_decay": float(overrides.get("ema_decay", state.extra.get("diffusion_ema_decay", 0.0))),
            "binary_noise_schedule": str(overrides.get("binary_noise_schedule", state.binary_noise_schedule)),
            "loss_weighting": str(overrides.get("loss_weighting", state.loss_weighting)),
            "min_snr_gamma": float(overrides.get("min_snr_gamma", state.min_snr_gamma)),
            "prediction_target": str(overrides.get("prediction_target", state.prediction_target)),
            "max_scale": ms,
            "binary_length_mode": str(
                overrides.get("binary_length_mode", state.binary_length_mode)
            ),
            "binary_length_g": float(
                overrides.get("binary_length_g", state.binary_length_g)
            ),
            "binary_length_scale": float(
                overrides.get("binary_length_scale", state.binary_length_scale)
            ),
        }
        target_u = overrides.get("target_univariate_batch")
        if target_u is not None:
            batch_plan = _plan_univariate_effective_batch(
                probed_max_windows=max_batch_size,
                n_variates=_n_variates_for_batch(state),
                target_univariate=int(target_u),
                smoke_test=smoke_test,
            )
            params.update(batch_plan)
            return params
        return _apply_effective_batch_multiplier(
            {
                **params,
                "batch_size": max(1, max_batch_size),
            },
            max_batch_size,
            state,
        )

    if smoke_test:
        return _apply_effective_batch_multiplier(
            {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-4, log=True),
                "batch_size": min(max(1, max_batch_size), 2),
                "ema_decay": 0.0,
                "binary_noise_schedule": state.binary_noise_schedule,
                "loss_weighting": state.loss_weighting,
                "min_snr_gamma": float(state.min_snr_gamma),
                "prediction_target": state.prediction_target,
                "max_scale": ms,
            },
            max_batch_size,
            state,
        )

    return _suggest_full_diffusion_params(
        trial, state, max_batch_size, tune_effective_batch=False,
    )


class _BaseStagedDiffusionFinetuneHPPhase(PipelinePhase):
    stage = ""

    def _copy_reused_stage_checkpoint(self, state: PipelineState) -> None:
        reuse_dir = self.get("reuse_checkpoint_dir")
        if not reuse_dir:
            return
        reuse_dir = os.path.abspath(str(reuse_dir))
        src_best = _stage_best_ckpt(state, self.stage, checkpoint_dir=reuse_dir)
        src_meta = os.path.join(
            _stage_subset_dir(state, self.stage, checkpoint_dir=reuse_dir),
            "metadata.json",
        )
        if not os.path.isfile(src_best) or not os.path.isfile(src_meta):
            raise FileNotFoundError(
                f"{self.name} reuse_checkpoint_dir={reuse_dir!r} missing "
                f"{self.stage} best.pt/metadata.json (looked for {src_best})"
            )
        dest_dir = _stage_subset_dir(state, self.stage)
        dest_best = _stage_best_ckpt(state, self.stage)
        dest_meta = os.path.join(dest_dir, "metadata.json")
        if os.path.isfile(dest_best) and os.path.isfile(dest_meta):
            return
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy2(src_best, dest_best)
        shutil.copy2(src_meta, dest_meta)
        logger.info("  [%s] copied %s checkpoint from %s", self.name, self.stage, reuse_dir)

    def should_skip(self, state: PipelineState) -> bool:
        # retrain=true forces a fresh train on a new run, but --resume must still
        # honor local best.pt+metadata so we can finish eval after quota crashes.
        if self.get("retrain", False) and not bool(getattr(state, "resume", False)):
            return False
        self._copy_reused_stage_checkpoint(state)
        best_pt = _stage_best_ckpt(state, self.stage)
        meta = os.path.join(_stage_subset_dir(state, self.stage), "metadata.json")
        if os.path.exists(best_pt) and os.path.exists(meta):
            try:
                # Corrupt/truncated saves (disk quota) look like a cache hit otherwise.
                torch.load(best_pt, map_location="cpu", weights_only=False)
            except Exception as e:
                logger.warning(
                    "  [%s] ignoring unreadable cache %s: %s", self.name, best_pt, e,
                )
                return False
            # Search@N + refit@M: only skip once the long refit has finished.
            if self.get("refit_best_max_epochs") is not None:
                try:
                    with open(meta, encoding="utf-8") as f:
                        meta_obj = json.load(f)
                except Exception as e:
                    logger.warning("  [%s] ignoring unreadable meta %s: %s", self.name, meta, e)
                    return False
                if not meta_obj.get("refit_completed"):
                    logger.info(
                        "  [%s] search ckpt present but refit_best_max_epochs pending; not skipping",
                        self.name,
                    )
                    return False
            logger.info("  [%s] cached: %s", self.name, best_pt)
            params = None
            try:
                with open(meta) as f:
                    params = json.load(f).get("tuned_params")
            except Exception as e:
                logger.warning("Failed to load tuned params from %s: %s", meta, e)
            self._record_finetune_result(state, best_pt, params)
            self._apply_tuned_length_to_state(state, params)
            return True
        return False

    def _record_finetune_result(
        self,
        state: PipelineState,
        checkpoint_path: str,
        params: Optional[Dict[str, Any]],
    ) -> None:
        if self.stage == "coarse":
            state.diffusion_coarse_finetune_ckpt = checkpoint_path
            state.coarse_finetune_best_params = params
        elif self.stage == "patch_refine":
            state.diffusion_patch_refine_finetune_ckpt = checkpoint_path
            state.patch_refine_finetune_best_params = params
        else:
            raise ValueError(f"unsupported finetune stage: {self.stage!r}")

    def on_skip(self, state: PipelineState) -> PipelineState:
        best_pt = _stage_best_ckpt(state, self.stage)
        if not os.path.exists(best_pt):
            return state
        meta_path = os.path.join(_stage_subset_dir(state, self.stage), "metadata.json")
        best_params: Dict[str, Any] = {}
        try:
            with open(meta_path, encoding="utf-8") as f:
                best_params = dict(json.load(f).get("tuned_params") or {})
        except Exception as e:
            logger.warning("Failed to load tuned params from %s: %s", meta_path, e)
        self._apply_tuned_length_to_state(state, best_params)
        try:
            self._log_post_finetune_viz_and_diagnostics(
                state,
                final_ckpt=best_pt,
                best_params=best_params,
            )
        except Exception as e:
            logger.warning("[%s] cached-phase viz/wandb log failed: %s", self.name, e, exc_info=True)
        return state

    def _log_post_finetune_viz_and_diagnostics(
        self,
        state: PipelineState,
        *,
        final_ckpt: str,
        best_params: Dict[str, Any],
        train_ds=None,
    ) -> None:
        if state.smoke_test:
            return

        needs_guidance = state.needs_guidance
        if not needs_guidance:
            return

        variate_indices = state.variate_indices
        if not variate_indices:
            raise ValueError(
                f"[{self.name}] Missing resolved variate_indices in state for dataset {state.dataset!r}. "
                "Data subset policy must be resolved before running phase."
            )
        subset_meta = state.data_subset_resolved or {}
        train_stride = int(subset_meta.get("train_stride", state.window_stride))
        test_stride = int(subset_meta.get("test_stride", 1))
        if train_ds is None:
            train_ds, _, _, norm_stats = pipeline_mod.load_dataset(
                state, state.dataset,
                variate_indices,
                stride=train_stride,
                test_stride=test_stride,
                ordinal_tie_atol=float(state.ordinal_tie_atol),
                use_ordinal_window_norm=state.use_ordinal_window_norm,
            )
            if norm_stats.get("ordinal_ladder") is not None:
                state.ordinal_ladder = norm_stats["ordinal_ladder"]
            if norm_stats.get("hybrid_flat_dataset_norm"):
                state.extra["hybrid_flat_norm_stats"] = {
                    k: norm_stats[k]
                    for k in (
                        "flat_variate_mask",
                        "flat_variate_frac",
                        "hybrid_flat_details",
                        "emp_std",
                    )
                    if k in norm_stats
                }

        ft_guidance_ckpt = state.guidance_finetune_ckpt
        if not ft_guidance_ckpt or not os.path.exists(ft_guidance_ckpt):
            ft_guidance_ckpt = state.default_guidance_finetune_ckpt_path()
        if not os.path.exists(ft_guidance_ckpt):
            logger.warning("[%s] viz skipped: guidance ckpt missing (%s)", self.name, ft_guidance_ckpt)
            return

        n_iv = len(variate_indices)
        device = state.resolve_device()
        coarse_ft = state.diffusion_coarse_finetune_ckpt or _stage_best_ckpt(state, "coarse")

        try:
            finetuned_model, _ = _load_staged_diffusion_from_ckpt(
                ckpt_path=final_ckpt,
                stage=self.stage,
                itrans_ckpt_path=ft_guidance_ckpt,
                n_vars=n_iv,
                device=device,
                tuned_params=best_params,
                guidance_type=state.guidance_type,
                state=state,
            )
            diag = run_real_dataset_phase_diagnostics(
                state,
                train_ds=train_ds,
                model=finetuned_model,
                itrans_ckpt_path=ft_guidance_ckpt,
                stage=self.stage,
                diffusion_ckpt_path=final_ckpt,
                coarse_ckpt_path=None,
                tag=f"diffusion_{self.stage}_finetune",
                include_phase_start=False,
            )
            wandb_utils.log_phase_diagnostics_result(diag)
            del finetuned_model
        except Exception as e:
            logger.warning("[%s] post-finetune diagnostics failed: %s", self.name, e, exc_info=True)

    def _pretrained_ckpt(self, state: PipelineState) -> Optional[str]:
        if bool(self.get("from_random_init", False)):
            logger.info(
                "  [%s] from_random_init=true; skipping staged pretrain load",
                self.name,
            )
            return None
        attr = {
            "coarse": state.diffusion_coarse_pretrain_ckpt,
            "patch_refine": state.diffusion_patch_refine_pretrain_ckpt,
        }[self.stage]
        candidates = [
            self.get("pretrained_ckpt"),
            attr,
            _stage_pretrain_ckpt(state, self.stage),
        ]
        for path in candidates:
            if path and os.path.exists(path):
                return path
        raise FileNotFoundError(
            f"{self.name} requires a staged {self.stage} pretrain checkpoint. "
            f"Expected one of: {', '.join(str(p) for p in candidates if p)}"
        )

    def _build_model(
        self,
        *,
        state: PipelineState,
        n_iv: int,
        itrans_guidance,
        device: torch.device,
        params: Dict[str, Any],
    ):
        ds_lb, ds_hz = pipeline_mod.dataset_window_lengths(state, state.dataset)
        model_state = stage_state(state, self.stage, honor_dataset_windows=True)
        model_kwargs = pipeline_mod.anchor_kwargs_from_params(model_state, params)
        model_kwargs.update(_state_anchor_kwargs(state))
        model_kwargs.update(_model_kwargs_from_tuned(params))
        return pipeline_mod.create_diffusion_model(
            model_state,
            n_variates=n_iv,
            lookback=ds_lb,
            horizon=ds_hz,
            guidance_model=itrans_guidance,
            diffusion_stage=self.stage,
            **model_kwargs,
        ).to(device)

    def _make_pruner(self, max_epochs: int):
        pruner_name = str(self.get("pruner", "hyperband")).lower()
        if pruner_name in ("none", "nop", "off"):
            return NopPruner()
        if pruner_name == "median":
            return MedianPruner(
                n_startup_trials=int(self.get("pruner_n_startup_trials", 2)),
                n_warmup_steps=int(self.get("pruner_n_warmup_steps", 0)),
            )
        if pruner_name == "hyperband":
            return HyperbandPruner(
                min_resource=int(self.get("pruner_n_warmup_steps", 1)),
                max_resource=max_epochs, reduction_factor=3,
            )
        raise ValueError(
            f"{self.name}: unknown pruner={pruner_name!r} "
            "(expected none, median, or hyperband)"
        )

    def _inject_length_params(self, params: Dict[str, Any], state: PipelineState) -> Dict[str, Any]:
        out = dict(params)
        out.setdefault("binary_length_mode", state.binary_length_mode)
        out.setdefault("binary_length_g", float(state.binary_length_g))
        out.setdefault("binary_length_scale", float(state.binary_length_scale))
        return out

    @staticmethod
    def _build_lr_scheduler(optimizer, state: PipelineState, max_epochs: int, learning_rate: float):
        scheduler_type = str(training_value(state, "lr_scheduler_type", "none"))
        warmup_epochs = min(
            int(training_value(state, "lr_warmup_epochs", 0)),
            max(0, max_epochs - 1),
        )
        if scheduler_type == "cosine":
            if warmup_epochs > 0:
                return torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[
                        torch.optim.lr_scheduler.LinearLR(
                            optimizer, start_factor=0.1, total_iters=warmup_epochs,
                        ),
                        torch.optim.lr_scheduler.CosineAnnealingLR(
                            optimizer,
                            T_max=max_epochs - warmup_epochs,
                            eta_min=learning_rate * 0.01,
                        ),
                    ],
                    milestones=[warmup_epochs],
                )
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max_epochs, eta_min=learning_rate * 0.01,
            )
        if scheduler_type == "linear":
            if warmup_epochs > 0:
                return torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[
                        torch.optim.lr_scheduler.LinearLR(
                            optimizer, start_factor=0.1, total_iters=warmup_epochs,
                        ),
                        torch.optim.lr_scheduler.LinearLR(
                            optimizer,
                            start_factor=1.0,
                            end_factor=0.01,
                            total_iters=max_epochs - warmup_epochs,
                        ),
                    ],
                    milestones=[warmup_epochs],
                )
            return torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=0.01, total_iters=max_epochs,
            )
        return None

    @staticmethod
    def _apply_tuned_length_to_state(state: PipelineState, params: Optional[Dict[str, Any]]) -> None:
        """Keep the winning length schedule in shared state for later phases."""
        if not params:
            return
        if "binary_length_mode" in params:
            state.binary_length_mode = str(params["binary_length_mode"])
        if "binary_length_g" in params:
            g = float(params["binary_length_g"])
            state.binary_length_g = g
            by_g = dict(getattr(state, "binary_length_g_by_dataset", None) or {})
            if state.dataset:
                by_g[str(state.dataset)] = g
            state.binary_length_g_by_dataset = by_g
        if "binary_length_scale" in params:
            state.binary_length_scale = float(params["binary_length_scale"])

    def _refit_best_if_configured(
        self,
        *,
        state: PipelineState,
        train_ds,
        val_ds,
        best_params: Dict[str, Any],
        diff_ckpt: Optional[str],
        ft_guidance_ckpt: str,
        device: torch.device,
        variate_indices,
        final_ckpt: str,
        hp_best_val_loss: float,
        best_trial_num: int,
        search_space: str,
        search_max_epochs: int,
        search_patience: int,
        subset_dir: str,
        subset_id: str,
        subset_meta: Dict[str, Any],
        norm_stats: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], float, int, bool]:
        """Train the Optuna winner for refit_best_max_epochs.

        Default continues the search checkpoint. ``refit_from_pretrain: true``
        loads synthetic pretrain weights instead (fresh optimizer / EMA).

        Returns (best_params, final_val, final_epoch, refit_completed).
        """
        refit_epochs = self.get("refit_best_max_epochs")
        if refit_epochs is None:
            return best_params, float(hp_best_val_loss), 0, False
        refit_epochs = int(refit_epochs)
        if refit_epochs < 1:
            raise ValueError(f"refit_best_max_epochs must be >= 1, got {refit_epochs}")
        if state.smoke_test:
            refit_epochs = 1
        refit_patience = int(self.get("refit_best_patience", refit_epochs))
        if state.smoke_test:
            refit_patience = 1
        best_params = _with_state_anchor_params(
            self._inject_length_params(best_params, state), state,
        )
        from_pretrain = bool(self.get("refit_from_pretrain", False))
        if from_pretrain and not (diff_ckpt and os.path.isfile(diff_ckpt)):
            raise FileNotFoundError(
                f"{self.name} refit_from_pretrain requires pretrain ckpt, got {diff_ckpt!r}"
            )
        logger.info(
            "  [%s] refit_best: search_epochs=%d -> refit_epochs=%d patience=%d "
            "lr=%.2e g=%s from_pretrain=%s",
            self.name,
            search_max_epochs,
            refit_epochs,
            refit_patience,
            float(best_params.get("learning_rate", 0.0)),
            best_params.get("binary_length_g"),
            from_pretrain,
        )
        # Persist search winner before long refit so --resume can skip Optuna.
        meta_pending: Dict[str, Any] = {
            "subset_id": subset_id,
            "dataset_name": state.dataset,
            "variate_indices": list(variate_indices),
            "norm_mean": norm_stats["mean"].tolist(),
            "norm_std": norm_stats["std"].tolist(),
            "tuned_params": best_params,
            "best_trial": best_trial_num,
            "hp_best_val_loss": float(hp_best_val_loss),
            "best_val_loss": float(hp_best_val_loss),
            "diffusion_stage": self.stage,
            "staged_representation": state.staged_representation,
            "search_space": search_space,
            "max_epochs": search_max_epochs,
            "patience": search_patience,
            "refit_best_max_epochs": refit_epochs,
            "refit_from_pretrain": from_pretrain,
            "refit_completed": False,
        }
        meta_pending.update(_hybrid_norm_metadata(norm_stats))
        put_subset_record(meta_pending, state.dataset, subset_meta)
        with open(os.path.join(subset_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta_pending, f, indent=2, sort_keys=True)

        final_val, final_epoch = self._train_once(
            state=state,
            train_ds=train_ds,
            val_ds=val_ds,
            params=best_params,
            pretrained_path=diff_ckpt if from_pretrain else None,
            guidance_checkpoint=ft_guidance_ckpt,
            device=device,
            variate_indices=variate_indices,
            ckpt_path=final_ckpt,
            max_epochs=refit_epochs,
            patience=refit_patience,
            trial=None,
            resume_ckpt=None if from_pretrain else final_ckpt,
        )
        return best_params, float(final_val), int(final_epoch), True

    def _train_once(
        self,
        *,
        state: PipelineState,
        train_ds,
        val_ds,
        params: Dict[str, Any],
        pretrained_path: Optional[str],
        guidance_checkpoint: str,
        device: torch.device,
        variate_indices,
        ckpt_path: Optional[str],
        max_epochs: int,
        patience: int,
        trial=None,
        guidance=None,
        pretrained_state_dict: Optional[Dict[str, Any]] = None,
        resume_ckpt: Optional[str] = None,
    ) -> Tuple[float, int]:
        params = _with_state_anchor_params(params, state)
        n_iv = len(variate_indices)
        batch_size = int(params["batch_size"])
        n_groups_cfg = int(training_value(state, "train_epoch_groups", 1))
        if n_groups_cfg < 1:
            raise ValueError(
                f"training.train_epoch_groups must be >= 1, got {n_groups_cfg!r}"
            )
        raw_max_bytes = training_value(state, "train_epoch_max_bytes", None)
        max_bytes = None if raw_max_bytes is None else int(raw_max_bytes)
        if len(train_ds) == 0:
            raise ValueError(f"{self.stage} train set is empty")
        trial_label = f"trial={trial.number}" if trial is not None else "trial=single"
        _log_gpu_mem(f"{self.stage}/{trial_label}/start")

        ds_lb, ds_hz = pipeline_mod.dataset_window_lengths(state, state.dataset)
        if guidance is None and guidance_checkpoint:
            guidance = pipeline_mod.load_wrapped_guidance(
                state, guidance_checkpoint, n_iv, device,
                guidance_type=state.guidance_type,
                dataset_lookback=ds_lb, dataset_horizon=ds_hz,
            )
        model = self._build_model(
            state=state, n_iv=n_iv, itrans_guidance=guidance, device=device, params=params,
        )
        try:
            resume_state = None
            start_epoch = 0
            if resume_ckpt:
                if not os.path.isfile(resume_ckpt):
                    raise FileNotFoundError(f"{self.name} resume_ckpt missing: {resume_ckpt}")
                resume_state = torch.load(
                    resume_ckpt, map_location=device, weights_only=False,
                )
                pipeline_mod.load_diffusion_state_keep_attached_guidance(
                    model, resume_state["model_state_dict"],
                )
                start_epoch = int(resume_state.get("epoch", -1)) + 1
                if start_epoch < 0:
                    raise RuntimeError(f"{self.name} resume ckpt has invalid epoch={resume_state.get('epoch')!r}")
                logger.info(
                    "  [%s] resume weights from %s at epoch %d/%d",
                    self.name, resume_ckpt, start_epoch, max_epochs,
                )
                if start_epoch >= int(max_epochs):
                    resume_val = float(resume_state["val_loss"])
                    resume_best_epoch = int(resume_state.get("epoch", -1)) + 1
                    logger.info(
                        "  [%s] resume already at epoch %d >= max_epochs=%d; keeping ckpt val=%.4f",
                        self.name, start_epoch, max_epochs, resume_val,
                    )
                    return resume_val, resume_best_epoch
            elif pretrained_path or pretrained_state_dict is not None:
                if pretrained_state_dict is None:
                    pretrained_state_dict = torch.load(
                        pretrained_path, map_location=device, weights_only=False,
                    )["model_state_dict"]
                pipeline_mod.load_diffusion_state_keep_attached_guidance(
                    model, pretrained_state_dict,
                )
            else:
                logger.info("  [%s] random init (no pretrain ckpt)", self.name)

            if (
                bool(self.get("probe_train_batch_size", False))
                and not state.smoke_test
            ):
                probed = _probe_max_finetune_batch_size(
                    model=model,
                    lookback=int(ds_lb),
                    horizon=int(ds_hz),
                    overlap=int(state.lookback_overlap),
                    n_variates=n_iv,
                    device=device,
                    stage=self.stage,
                    min_bs=1,
                    max_bs=int(self.get("probe_train_batch_size_max", 16)),
                    headroom=float(self.get("probe_train_batch_size_headroom", 0.85)),
                )
                if probed != batch_size or params.get("target_univariate_batch") is not None:
                    logger.info(
                        "  [%s] train batch probe: yaml/plan batch_size=%d -> probed=%d",
                        self.name,
                        batch_size,
                        probed,
                    )
                    target_u = params.get("target_univariate_batch")
                    if target_u is not None:
                        batch_plan = _plan_univariate_effective_batch(
                            probed_max_windows=int(probed),
                            n_variates=n_iv,
                            target_univariate=int(target_u),
                            smoke_test=False,
                        )
                        params.update(batch_plan)
                        batch_size = int(params["batch_size"])
                    else:
                        params["batch_size"] = int(probed)
                        params["gradient_accumulation_steps"] = 1
                        params["effective_batch_size"] = int(probed)
                        params["effective_univariate_batch"] = int(probed)
                        batch_size = int(probed)

            trial_seed = int(state.seed)
            if trial is not None:
                trial_seed = trial_seed + 17 * int(trial.number)
            u_micro = int(params["batch_size"])
            use_u_rows = bool(params.get("univariate_row_micro_batch")) or (
                self.get("max_univariate_micro_batch_by_dataset") is not None
            )
            loader_bs = (
                dataloader_windows_for_univariate_rows(u_micro, n_iv)
                if use_u_rows
                else u_micro
            )
            train_loader, n_groups, window_nbytes, group_nbytes = make_grouped_train_loader(
                train_ds,
                batch_size=loader_bs,
                n_groups=n_groups_cfg,
                seed=trial_seed,
                max_bytes=max_bytes,
                n_variates=n_iv,
                lookback=int(ds_lb),
                horizon=int(ds_hz),
                overlap=int(state.lookback_overlap),
                smoke_test=bool(state.smoke_test),
            )
            val_loader = DataLoader(
                val_ds, batch_size=loader_bs, shuffle=False, num_workers=0,
            )
            if len(train_loader) == 0:
                raise ValueError(
                    f"{self.stage} train set has {len(train_ds)} windows, "
                    f"smaller than dataloader_windows={loader_bs}"
                )
            logger.info(
                "  [%s/%s] %s START epochs=%d patience=%d lr=%.2e U=%d windows=%d accum=%d "
                "train_epoch_groups=%d train_n=%d train_batches=%d val_batches=%d "
                "window_bytes=%s group_bytes=%s g=%s",
                self.name, self.stage, trial_label, max_epochs, patience,
                float(params["learning_rate"]), u_micro, loader_bs,
                int(params.get("gradient_accumulation_steps", 1)),
                n_groups, len(train_ds), len(train_loader), len(val_loader),
                window_nbytes, group_nbytes,
                params.get("binary_length_g"),
            )
            log_epoch_shard_contract(
                name=f"{self.name}/{self.stage}",
                n_groups=n_groups,
                max_epochs=max_epochs,
                patience=patience,
            )

            token_cache = getattr(self, "_phase_token_cache", None)
            if (
                not state.disable_cross_attention
                and bool(self.get("cache_cross_variate_tokens", True))
                and token_cache is None
            ):
                token_cache = CrossVariateTokenCache(
                    model=model,
                    device=device,
                    storage=str(self.get("cross_variate_token_cache_storage", "pinned_cpu")),
                    token_kind="mixed",  # frozen encoder tokens; adapter runs live each step
                )
                stable_train = not _has_train_window_augmentation(train_ds)
                n_cache = count_cache_windows(val_ds)
                if stable_train:
                    n_cache += count_cache_windows(train_ds)
                token_cache.reserve(n_cache)
                if stable_train:
                    token_cache.precompute_dataset(train_ds, batch_size=loader_bs)
                else:
                    logger.info(
                        "  [%s/%s] train token cache disabled: train-window augmentation changes past inputs",
                        self.name, self.stage,
                    )
                token_cache.precompute_dataset(val_ds, batch_size=loader_bs)
                token_cache.release_encoder()
                self._phase_token_cache = token_cache
                self._phase_cache_train_enabled = stable_train
            train_token_cache = (
                token_cache if getattr(self, "_phase_cache_train_enabled", False) else None
            )

            optimizer = torch.optim.AdamW(
                model.parameters(), lr=float(params["learning_rate"]),
            )
            if resume_state is not None and resume_state.get("optimizer_state_dict") is not None:
                optimizer.load_state_dict(resume_state["optimizer_state_dict"])
            scheduler = self._build_lr_scheduler(
                optimizer, state, max_epochs, float(params["learning_rate"]),
            )
            if scheduler is not None and start_epoch > 0:
                for _ in range(start_epoch):
                    scheduler.step()
            saved_ckpt = False

            def set_training_epoch(loader: DataLoader, epoch: int) -> None:
                batch_sampler = getattr(loader, "batch_sampler", None)
                if hasattr(batch_sampler, "set_epoch"):
                    batch_sampler.set_epoch(epoch)
                set_train_window_aug_epoch(loader, epoch)
                epoch_ds = train_ds
                while hasattr(epoch_ds, "dataset") and not hasattr(epoch_ds, "set_epoch"):
                    epoch_ds = epoch_ds.dataset
                if hasattr(epoch_ds, "set_epoch"):
                    epoch_ds.set_epoch(epoch)

            def save_best(metrics: DiffusionEpochMetrics) -> None:
                nonlocal saved_ckpt
                if ckpt_path is None:
                    return
                config = {
                    "tuned_params": dict(params),
                    "diffusion_stage": self.stage,
                    "best_epoch": metrics.epoch,
                    "selection_metric": "val_loss",
                }
                save_checkpoint(
                    pipeline_mod.unwrap_model(model), optimizer, metrics.epoch - 1,
                    metrics.train_loss, metrics.selection_score, config, ckpt_path,
                )
                saved_ckpt = True
                if trial is not None:
                    trial.set_user_attr("ckpt_path", ckpt_path)

            def report_epoch(metrics: DiffusionEpochMetrics) -> None:
                _log_gpu_mem(f"{self.stage}/{trial_label}/ep{metrics.epoch}")
                if trial is None:
                    return
                trial.report(metrics.selection_score, metrics.epoch - 1)
                if trial.should_prune():
                    logger.info(
                        "  [%s/%s] %s epoch %d/%d PRUNED val=%.4f",
                        self.name, self.stage, trial_label, metrics.epoch, max_epochs,
                        metrics.selection_score,
                    )
                    raise TrialPruned()

            trainer = DiffusionTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                optimizer=optimizer,
                accum_steps=int(params.get("gradient_accumulation_steps", 1)),
                clip_grad=1.0,
                scheduler=scheduler,
                ema_decay=float(params.get("ema_decay", 0.0)),
                unpack_batch=_unpack_patch_refine_batch,
                set_loader_mode=lambda current_model, loader, eval_mode=False: (
                    pipeline_mod._set_ordinal_loader_mode(
                        state, current_model, loader, eval_mode=eval_mode,
                    )
                ),
                set_training_epoch=set_training_epoch,
                sequential_anchor_backward=(
                    self.stage == "patch_refine"
                    and bool(getattr(state, "sequential_anchor_backward", False))
                ),
                deterministic_anchor_every_n_batches=(
                    state.deterministic_anchor_every_n_batches
                ),
                train_token_cache=train_token_cache,
                val_token_cache=token_cache,
                log_prefix=f"{self.name}/{self.stage}/{trial_label}",
                univariate_micro_batch=u_micro if use_u_rows else None,
            )
            early_stopping = EarlyStopping(patience=patience)
            if resume_state is not None:
                early_stopping.best_loss = float(resume_state["val_loss"])
            result = trainer.fit(
                max_epochs=max_epochs,
                start_epoch=start_epoch,
                initial_best_val=(
                    None if resume_state is None else float(resume_state["val_loss"])
                ),
                initial_best_epoch=(
                    0 if resume_state is None else int(resume_state.get("epoch", -1)) + 1
                ),
                early_stopping=early_stopping,
                on_best=save_best,
                on_epoch_end=report_epoch,
            )
            if ckpt_path and result.best_epoch > 0 and not saved_ckpt:
                if not (resume_ckpt and os.path.isfile(ckpt_path)):
                    raise RuntimeError(
                        f"{trial_label}: best_val={result.best_val:.4f} at epoch {result.best_epoch} "
                        f"but no checkpoint was written to {ckpt_path}"
                    )
            if ckpt_path and saved_ckpt and not os.path.isfile(ckpt_path):
                raise RuntimeError(
                    f"{trial_label}: expected checkpoint at {ckpt_path} after save"
                )
            if ckpt_path:
                hist_path = os.path.join(os.path.dirname(ckpt_path), "val_loss_history.json")
                with open(hist_path, "w", encoding="utf-8") as hf:
                    json.dump(
                        {
                            "stage": self.stage,
                            "trial_label": trial_label,
                            "dataset": state.dataset,
                            "seed": int(state.seed),
                            "length_mode": params.get("binary_length_mode", "none"),
                            "length_g": params.get("binary_length_g", 1.0),
                            "length_scale": params.get("binary_length_scale", 1.0),
                            "binary_noise_schedule": params.get("binary_noise_schedule"),
                            "selection_metric": "val_loss",
                            "best_val": result.best_val,
                            "best_epoch": result.best_epoch,
                            "epochs": [
                                {
                                    "epoch": metrics.epoch,
                                    "train_loss": metrics.train_loss,
                                    "val_loss": metrics.val_loss,
                                    "selection_score": metrics.selection_score,
                                    "best_val": metrics.best_val,
                                    "lr": metrics.lr,
                                    "saved": metrics.saved,
                                }
                                for metrics in result.history
                            ],
                        },
                        hf,
                        indent=2,
                    )
            logger.info(
                "  [%s/%s] %s DONE best_val=%.4f best_epoch=%d total_time=%.1fs",
                self.name, self.stage, trial_label, result.best_val, result.best_epoch,
                result.elapsed_seconds,
            )
            return result.best_val, result.best_epoch
        finally:
            del model, guidance
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def execute(self, state: PipelineState) -> PipelineState:
        subset_id = state.subset_id or state.dataset
        variate_indices = state.variate_indices
        if not variate_indices:
            raise ValueError(
                f"[{self.name}] Missing resolved variate_indices in state for dataset {state.dataset!r}. "
                "Data subset policy must be resolved before running phase."
            )
        subset_meta = state.data_subset_resolved or {}
        train_stride = int(subset_meta.get("train_stride", state.window_stride))
        test_stride = int(subset_meta.get("test_stride", 1))

        ft_guidance_ckpt = state.guidance_finetune_ckpt
        if not ft_guidance_ckpt or not os.path.exists(ft_guidance_ckpt):
            ft_guidance_ckpt = state.default_guidance_finetune_ckpt_path()
        needs_guidance = state.needs_guidance
        if needs_guidance and not os.path.exists(ft_guidance_ckpt):
            raise RuntimeError(
                f"{self.name} requires finetuned guidance ({state.guidance_type}), got: {ft_guidance_ckpt}"
            )
        if not needs_guidance:
            ft_guidance_ckpt = ""
        if self.stage == "patch_refine" and not state.diffusion_coarse_finetune_ckpt:
            raise RuntimeError("patch_refine staged tuning requires completed coarse best model first")
        diff_ckpt = self._pretrained_ckpt(state)

        device = state.resolve_device()
        n_iv = len(variate_indices)
        train_ds, val_ds, _, norm_stats = pipeline_mod.load_dataset(
            state, state.dataset,
            variate_indices,
            stride=train_stride,
            test_stride=test_stride,
            ordinal_tie_atol=float(state.ordinal_tie_atol),
            use_ordinal_window_norm=state.use_ordinal_window_norm,
        )
        # Unique-segment wrap rebuilds from ts_ds.data. Cap windows after that
        # so a random_window_subset Subset is never passed into the wrap.
        seg_stride = max(1, int(train_stride))
        if (
            self.stage == "patch_refine"
            and bool(getattr(state, "patch_refine_unique_segments", False))
        ):
            train_ds = wrap_timeseries_as_unique_segments(
                train_ds,
                patch_width=int(getattr(state, "patch_refine_patch_width", 8)),
                segment_stride=seg_stride,
                series_id=0,
            )
            val_ds = wrap_timeseries_as_unique_segments(
                val_ds,
                patch_width=int(getattr(state, "patch_refine_patch_width", 8)),
                segment_stride=seg_stride,
                series_id=1,
            )
            logger.info(
                "  [%s] unique patch segments enabled "
                "(segment_stride=%d train=%d val=%d)",
                self.name,
                seg_stride,
                len(train_ds),
                len(val_ds),
            )
        train_ds = random_window_subset(
            train_ds,
            subset_meta.get("train_max_windows"),
            int(state.seed) + 17,
            label=f"{self.name}/train",
        )
        val_ds = random_window_subset(
            val_ds,
            subset_meta.get("val_max_windows"),
            int(state.seed) + 29,
            label=f"{self.name}/val",
        )
        if norm_stats.get("hybrid_flat_dataset_norm"):
            flat_mask = [bool(x) for x in norm_stats["flat_variate_mask"].tolist()]
            state.extra["hybrid_flat_norm_stats"] = {
                k: norm_stats[k]
                for k in (
                    "flat_variate_mask",
                    "flat_variate_frac",
                    "hybrid_flat_details",
                    "emp_std",
                )
                if k in norm_stats
            }
            logger.info(
                "  [%s] hybrid flat dataset-norm: flat_mask=%s frac=%s details=%s",
                self.name,
                flat_mask,
                [round(float(x), 4) for x in norm_stats["flat_variate_frac"].tolist()],
                norm_stats.get("hybrid_flat_details"),
            )
        if self.stage == "patch_refine":
            train_ds = _maybe_subsample_patch_refine_train_windows(state, train_ds)
            patch_frac = float(getattr(state, "patch_refine_finetune_patch_fraction", 1.0))
            if not math.isfinite(patch_frac) or patch_frac <= 0.0 or patch_frac > 1.0:
                raise ValueError(
                    f"patch_refine_finetune_patch_fraction must be in (0, 1], got {patch_frac!r}"
                )
            if patch_frac < 1.0 and not bool(getattr(state, "patch_refine_unique_segments", False)):
                raise ValueError(
                    "patch_refine_finetune_patch_fraction < 1 requires "
                    "patch_refine_unique_segments=true"
                )
            if patch_frac < 1.0:
                logger.info(
                    "  [%s] train unique-seg crop fraction=%.3f (stride on variate index; "
                    "independent of window_fraction=%.3f)",
                    self.name,
                    patch_frac,
                    float(getattr(state, "patch_refine_finetune_window_fraction", 1.0)),
                )
        if norm_stats.get("ordinal_ladder") is not None:
            state.extra["global_ordinal_ladder"] = norm_stats["ordinal_ladder"]
        aug_cfg = training_value(state, "train_window_aug", None) or {}
        train_ds = maybe_wrap_train_window_aug(
            train_ds,
            enabled=bool(aug_cfg.get("enabled", False)),
            apply_prob=float(aug_cfg.get("apply_prob", 0.5)),
            seed=int(state.seed),
            ladder=norm_stats.get("ordinal_ladder"),
            acf_threshold=float(aug_cfg.get("acf_threshold", 0.35)),
            excluded_names=aug_cfg.get("exclude_names", ()),
        )
        if state.smoke_test:
            train_ds = Subset(train_ds, list(range(min(1, len(train_ds)))))
            val_ds = Subset(val_ds, list(range(min(1, len(val_ds)))))
        logger.info(
            "  [%s] train/val windows=%d/%d",
            self.name, len(train_ds), len(val_ds),
        )



        if not diff_ckpt:
            logger.info(
                "  [%s] phase-start diagnostics skipped (from_random_init / no pretrain ckpt)",
                self.name,
            )
        else:
            try:
                probe_model, _ = _load_staged_diffusion_from_ckpt(
                    ckpt_path=diff_ckpt,
                    stage=self.stage,
                    itrans_ckpt_path=ft_guidance_ckpt,
                    n_vars=n_iv,
                    device=device,
                    guidance_type=state.guidance_type,
                    state=state,
                )
                ckpt_info = []
                if ft_guidance_ckpt and os.path.exists(ft_guidance_ckpt):
                    ckpt_info.append(
                        {
                            "kind": state.guidance_type,
                            "path": ft_guidance_ckpt,
                            "n_variates": n_iv,
                            "lookback": int(state.lookback_length),
                            "horizon": int(state.forecast_length),
                        }
                    )
                ckpt_info.append(
                    {
                        "kind": f"diffusion_pretrain_{self.stage}",
                        "path": diff_ckpt,
                        "n_variates": n_iv,
                        "lookback": int(state.lookback_length),
                        "horizon": int(state.forecast_length),
                    }
                )
                phase_start = run_phase_start_diagnostics(
                    state,
                    phase_name=self.name,
                    models=[probe_model],
                    model_labels=[f"diffusion_{self.stage}"],
                    datasets=[train_ds],
                    dataset_prefixes=["dataset"],
                    ckpt_info=ckpt_info,
                )
                wandb_utils.log_phase_diagnostics_result({"summary": phase_start})
                del probe_model
            except Exception as e:
                logger.warning("[%s] phase-start diagnostics failed: %s", self.name, e, exc_info=True)
        ds_lb, ds_hz = pipeline_mod.dataset_window_lengths(state, state.dataset)
        micro_ceiling = configured_max_diffusion_batch(state, state.smoke_test)
        default_micro = configured_phase_micro_batch(
            state, state.smoke_test, self.overrides,
        )
        logger.info(
            "  [%s] finetune micro-batch default=%d ceiling=%d dataset=%s (YAML; no GPU probe)",
            self.name,
            default_micro,
            micro_ceiling,
            state.dataset,
        )

        accum_mult = state.extra.get("diffusion_effective_batch_multiplier")
        if accum_mult is not None and float(accum_mult) > 1.0:
            batch_plan = resolve_diffusion_batch_and_accum(default_micro, accum_mult)
            logger.info(
                "  [%s] grad accum: base_micro=%d multiplier=%s -> micro=%d accum=%d effective=%d",
                self.name,
                default_micro,
                accum_mult,
                batch_plan["batch_size"],
                batch_plan["gradient_accumulation_steps"],
                batch_plan["effective_batch_size"],
            )

        reuse_from = self.get("reuse_tuned_params_from")
        by_dataset = self.get("reuse_tuned_params_from_by_dataset") or {}
        if isinstance(by_dataset, dict) and state.dataset in by_dataset:
            reuse_from = by_dataset[state.dataset]
        retrain_reused = bool(reuse_from) and bool(self.get("retrain", False))

        max_epochs = int(self.require("max_epochs"))
        patience = int(self.require("patience"))
        if state.smoke_test:
            max_epochs = patience = 1

        subset_dir = _stage_subset_dir(state, self.stage)
        ensure_checkpoint_dir(final_ckpt := _stage_best_ckpt(state, self.stage))
        trials_dir = os.path.join(subset_dir, "_trials")
        ensure_checkpoint_dir(os.path.join(trials_dir, "_trial.pt"))

        reuse_meta: Dict[str, Any] = {}
        hp_best_val_loss: Optional[float] = None
        best_trial_num = -1
        final_val = float("nan")
        final_epoch = 0
        search_space = "lr_only"
        refit_completed = False
        pending_refit = False
        best_params: Dict[str, Any] = {}
        meta_path = os.path.join(subset_dir, "metadata.json")

        if (
            not reuse_from
            and self.get("refit_best_max_epochs") is not None
            and os.path.isfile(final_ckpt)
            and os.path.isfile(meta_path)
        ):
            try:
                with open(meta_path, encoding="utf-8") as f:
                    prev_meta = json.load(f)
            except Exception as e:
                prev_meta = {}
                logger.warning("  [%s] could not read %s for pending refit: %s", self.name, meta_path, e)
            if prev_meta.get("tuned_params") and not prev_meta.get("refit_completed"):
                best_params = dict(prev_meta["tuned_params"])
                hp_best_val_loss = float(
                    prev_meta.get("hp_best_val_loss")
                    or prev_meta.get("best_val_loss")
                    or float("nan")
                )
                best_trial_num = int(prev_meta.get("best_trial", -1))
                search_space = str(
                    prev_meta.get("search_space") or self.get("search_space") or "lr_only"
                ).lower()
                pending_refit = True
                logger.info(
                    "  [%s] resuming pending refit (search already done, trial=%d)",
                    self.name,
                    best_trial_num,
                )

        if reuse_from:
            best_params, source_dir, reuse_meta = _load_reused_stage_params(
                state, stage=self.stage, subset_id=subset_id, source_config=str(reuse_from),
            )
            search_space = str(reuse_meta.get("search_space") or self.get("search_space") or "lr_only").lower()
            tuned_bs = int(best_params.get("batch_size", default_micro))
            best_params["batch_size"] = min(tuned_bs, micro_ceiling)
            best_params = _with_state_anchor_params(best_params, state)
            if retrain_reused:
                final_val, final_epoch = self._train_once(
                    state=state,
                    train_ds=train_ds,
                    val_ds=val_ds,
                    params=best_params,
                    pretrained_path=diff_ckpt,
                    guidance_checkpoint=ft_guidance_ckpt,
                    device=device,
                    variate_indices=variate_indices,
                    ckpt_path=final_ckpt,
                    max_epochs=max_epochs,
                    patience=patience,
                    trial=None,
                )
                hp_best_val_loss = float(final_val)
                logger.info(
                    "  [%s] retrained %s with reused HP from %s (lr=%s)",
                    self.name,
                    self.stage,
                    source_dir,
                    best_params.get("learning_rate"),
                )
            else:
                src_best = os.path.join(source_dir, subset_id, self.stage, "best.pt")
                if not os.path.exists(src_best):
                    raise FileNotFoundError(f"Missing reused staged checkpoint: {src_best}")
                if not os.path.exists(final_ckpt):
                    shutil.copy2(src_best, final_ckpt)
                hp_best_val_loss = float(
                    reuse_meta.get("best_val_loss")
                    or reuse_meta.get("hp_best_val_loss")
                    or float("nan")
                )
                final_val = hp_best_val_loss
                final_epoch = int(reuse_meta.get("best_epoch", 0))
                logger.info("  [%s] reused %s from %s", self.name, self.stage, source_dir)
        elif pending_refit:
            logger.info("  [%s] skipping Optuna; using cached search winner for refit", self.name)
        else:
            n_trials = int(self.require("n_trials"))
            if state.smoke_test:
                n_trials = 1
            search_space = str(self.require("search_space")).lower()
            if search_space not in {
                "default",
                "lr_only",
                "full_with_batch",
                "reduced_hp",
                "lr_eff_batch_univariate",
                "lr_eff_batch_univariate_ema",
                "lr_eff_batch_g",
                "fixed",
            }:
                raise ValueError(f"Unknown staged diffusion search_space={search_space!r}")
            if search_space in {"reduced_hp", "lr_only"}:
                for key in ("hp_lr_min", "hp_lr_max"):
                    if self.get(key) is None:
                        raise ValueError(f"search_space={search_space} requires phase {key}")
            if search_space in {
                "lr_eff_batch_univariate",
                "lr_eff_batch_univariate_ema",
                "lr_eff_batch_g",
            }:
                required = ["hp_lr_min", "hp_lr_max"]
                if search_space == "lr_eff_batch_univariate_ema":
                    required.append("ema_decay_grid")
                for key in required:
                    if self.get(key) is None:
                        raise ValueError(
                            f"search_space={search_space} requires phase {key}"
                        )
                has_grid = self.get("effective_univariate_batch_grid") is not None
                has_mult = self.get("effective_univariate_batch_multipliers") is not None
                if has_grid and has_mult:
                    raise ValueError(
                        f"search_space={search_space} cannot set both "
                        "effective_univariate_batch_grid and "
                        "effective_univariate_batch_multipliers"
                    )
                if not has_grid and not has_mult:
                    raise ValueError(
                        f"search_space={search_space} requires "
                        "effective_univariate_batch_grid or "
                        "effective_univariate_batch_multipliers"
                    )
            if search_space == "fixed" and not (
                self.get("fixed_tuned_params") or self.get("fixed_tuned_params_by_dataset")
            ):
                raise ValueError(
                    "search_space=fixed requires fixed_tuned_params "
                    "and/or fixed_tuned_params_by_dataset in phase YAML"
                )

            if search_space == "fixed":
                best_params = _build_fixed_hp_params(
                    state, default_micro, state.smoke_test, self.overrides,
                )
                best_params = _with_state_anchor_params(best_params, state)
                final_val, final_epoch = self._train_once(
                    state=state,
                    train_ds=train_ds,
                    val_ds=val_ds,
                    params=best_params,
                    pretrained_path=diff_ckpt,
                    guidance_checkpoint=ft_guidance_ckpt,
                    device=device,
                    variate_indices=variate_indices,
                    ckpt_path=final_ckpt,
                    max_epochs=max_epochs,
                    patience=patience,
                    trial=None,
                )
                hp_best_val_loss = float(final_val)
                best_trial_num = 0
                logger.info(
                    "  [%s] fixed HP train done: val=%.4f epoch=%d lr=%.2e micro_bs=%d",
                    self.name,
                    hp_best_val_loss,
                    final_epoch,
                    float(best_params.get("learning_rate", 0.0)),
                    int(best_params.get("batch_size", 1)),
                )
            else:
                phase = self

                def objective_builder(_worker_id: int):
                    dev = state.resolve_device()
                    worker_guidance = None
                    if ft_guidance_ckpt:
                        worker_guidance = pipeline_mod.load_wrapped_guidance(
                            state, ft_guidance_ckpt,
                            n_iv,
                            dev,
                            guidance_type=state.guidance_type,
                            dataset_lookback=ds_lb,
                            dataset_horizon=ds_hz,
                        )
                    if diff_ckpt:
                        worker_pretrained = torch.load(
                            diff_ckpt, map_location=dev, weights_only=False,
                        )["model_state_dict"]
                    else:
                        worker_pretrained = None

                    def objective(trial):
                        plan_batch = (
                            default_micro if search_space == "lr_only" else micro_ceiling
                        )
                        params = _suggest_staged_params(
                            trial,
                            state,
                            plan_batch,
                            state.smoke_test,
                            search_space=search_space,
                            phase_overrides=phase.overrides,
                        )
                        trial.set_user_attr("full_params", dict(params))
                        if search_space in {
                            "full_with_batch",
                            "reduced_hp",
                            "lr_eff_batch_univariate",
                            "lr_eff_batch_univariate_ema",
                            "lr_eff_batch_g",
                        } and not state.smoke_test:
                            micro = int(params["batch_size"])
                            accum = int(params.get("gradient_accumulation_steps", 1))
                            # Univariate plans may use micro=1 and large accum to hit U=B*C.
                            min_micro = (
                                1
                                if search_space in {
                                    "lr_eff_batch_univariate",
                                    "lr_eff_batch_univariate_ema",
                                    "lr_eff_batch_g",
                                }
                                else 4
                            )
                            max_accum = (
                                2048
                                if search_space in {
                                    "lr_eff_batch_univariate",
                                    "lr_eff_batch_univariate_ema",
                                    "lr_eff_batch_g",
                                }
                                else 512
                            )
                            if micro < min_micro or accum > max_accum:
                                raise RuntimeError(
                                    f"Degenerate batch plan micro_bs={micro} accum={accum} "
                                    f"(effective={micro * accum}); stale Optuna journal or planner bug"
                                )
                        logger.info(
                            "  [%s] Optuna trial %d/%d suggested lr=%.2e micro_bs=%d "
                            "accum=%d effective_bs=%d univariate_U=%s (target_U=%s) g=%s",
                            phase.name,
                            trial.number + 1,
                            n_trials,
                            float(params["learning_rate"]),
                            int(params["batch_size"]),
                            int(params.get("gradient_accumulation_steps", 1)),
                            int(params.get("effective_batch_size", params["batch_size"])),
                            params.get("effective_univariate_batch", "-"),
                            params.get("target_univariate_batch", "-"),
                            params.get("binary_length_g", "-"),
                        )
                        trial_ckpt = os.path.join(
                            trials_dir, f"trial_{trial.number}_best.pt",
                        )
                        trial_t0 = time.perf_counter()
                        try:
                            best_val, best_ep = phase._train_once(
                                state=state,
                                train_ds=train_ds,
                                val_ds=val_ds,
                                params=params,
                                pretrained_path=diff_ckpt,
                                guidance_checkpoint=ft_guidance_ckpt,
                                device=dev,
                                variate_indices=variate_indices,
                                ckpt_path=trial_ckpt,
                                max_epochs=max_epochs,
                                patience=patience,
                                trial=trial,
                                guidance=worker_guidance,
                                pretrained_state_dict=worker_pretrained,
                            )
                        except torch.cuda.OutOfMemoryError:
                            logger.warning(
                                "  [%s] trial %d OOM (batch=%s), pruning",
                                phase.name, trial.number, params.get("batch_size"),
                            )
                            # _train_once's finally deletes the model, but the
                            # CUDA caching allocator only releases after the
                            # frame unwinds — empty here or the next trial
                            # starts already near the L40S ceiling.
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            raise TrialPruned() from None
                        except TrialPruned:
                            logger.info(
                                "  [%s] Optuna trial %d pruned after %.1fs",
                                phase.name, trial.number, time.perf_counter() - trial_t0,
                            )
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            raise
                        trial.set_user_attr("best_epoch", best_ep)
                        logger.info(
                            "  [%s] Optuna trial %d finished best_val=%.4f best_epoch=%d time=%.1fs",
                            phase.name,
                            trial.number,
                            best_val,
                            best_ep,
                            time.perf_counter() - trial_t0,
                        )
                        return best_val

                    return objective

                def _retain_complete_trial_ckpts(study, _trial) -> None:
                    # Keep every COMPLETE trial weight so --resume can still
                    # promote whichever trial wins after more trials land.
                    # Drop only pruned/failed mid-run checkpoints.
                    keep_nums = {
                        int(t.number)
                        for t in study.get_trials(
                            deepcopy=False, states=(TrialState.COMPLETE,),
                        )
                    }
                    keep_names = {f"trial_{n}_best.pt" for n in keep_nums}
                    keep_names |= {f"_diff_ft_trial_{n}_best.pt" for n in keep_nums}
                    for trial_dir in (trials_dir, subset_dir):
                        if not os.path.isdir(trial_dir):
                            continue
                        for fn in os.listdir(trial_dir):
                            if not fn.endswith("_best.pt"):
                                continue
                            if not (
                                fn.startswith("trial_")
                                or fn.startswith("_diff_ft_trial_")
                            ):
                                continue
                            if fn in keep_names:
                                continue
                            path = os.path.join(trial_dir, fn)
                            try:
                                os.remove(path)
                            except OSError:
                                pass

                logger.info(
                    "  [%s] Optuna study start: n_trials=%d max_epochs=%d patience=%d",
                    self.name, n_trials, max_epochs, patience,
                )
                study_t0 = time.perf_counter()
                pruner = self._make_pruner(max_epochs)
                study = run_optuna_study(
                    study_name=f"{state.experiment_name}-{self.stage}-hp",
                    checkpoint_dir=subset_dir,
                    n_trials=n_trials,
                    direction="minimize",
                    objective_builder=objective_builder,
                    sampler=TPESampler(seed=state.seed, multivariate=True, group=True),
                    pruner=pruner,
                    sampler_seed=state.seed,
                    callbacks=[_retain_complete_trial_ckpts],
                    enqueue_trials=_resolved_enqueue_trials(self.overrides, state),
                )
                try:
                    best_trial = study.best_trial
                except ValueError as e:
                    raise RuntimeError(
                        f"All {self.stage} diffusion HP trials failed for {subset_id}"
                    ) from e

                best_params = dict(best_trial.user_attrs.get("full_params") or best_trial.params)
                best_params.setdefault("min_snr_gamma", 5.0)
                best_params.setdefault(
                    "max_scale",
                    float(state.max_scale_by_dataset.get(state.dataset, state.max_scale)),
                )
                best_params = _with_state_anchor_params(best_params, state)
                hp_best_val_loss = float(study.best_value)
                best_trial_num = int(best_trial.number)
                final_epoch = int(best_trial.user_attrs.get("best_epoch", 0))
                logger.info(
                    "  [%s] Optuna study done in %.1fs: best_trial=%d best_val=%.4f best_epoch=%d lr=%.2e",
                    self.name,
                    time.perf_counter() - study_t0,
                    best_trial_num,
                    hp_best_val_loss,
                    final_epoch,
                    float(best_params.get("learning_rate", 0.0)),
                )

                src = _resolve_best_trial_ckpt(
                    study, trials_dir, subset_dir, best_trial_num,
                )
                shutil.copy2(src, final_ckpt)
                if not os.path.isfile(final_ckpt):
                    raise RuntimeError(f"Failed to promote best trial checkpoint to {final_ckpt}")
                final_val = hp_best_val_loss
                _cleanup_trial_ckpts(trials_dir, subset_dir, keep=src)

        if not reuse_from and self.get("refit_best_max_epochs") is not None:
            if hp_best_val_loss is None:
                raise RuntimeError(f"{self.name}: refit_best_max_epochs set but no HP winner available")
            best_params, final_val, final_epoch, refit_completed = self._refit_best_if_configured(
                state=state,
                train_ds=train_ds,
                val_ds=val_ds,
                best_params=best_params,
                diff_ckpt=diff_ckpt,
                ft_guidance_ckpt=ft_guidance_ckpt,
                device=device,
                variate_indices=variate_indices,
                final_ckpt=final_ckpt,
                hp_best_val_loss=float(hp_best_val_loss),
                best_trial_num=best_trial_num,
                search_space=search_space,
                search_max_epochs=max_epochs,
                search_patience=patience,
                subset_dir=subset_dir,
                subset_id=subset_id,
                subset_meta=subset_meta,
                norm_stats=norm_stats,
            )

        meta_out: Dict[str, Any] = {
            "subset_id": subset_id,
            "dataset_name": state.dataset,
            "variate_indices": variate_indices,
            "norm_mean": norm_stats["mean"].tolist(),
            "norm_std": norm_stats["std"].tolist(),
            "tuned_params": best_params,
            "best_trial": best_trial_num,
            "hp_best_val_loss": hp_best_val_loss,
            "best_val_loss": float(final_val),
            "best_selection_score": float(final_val),
            "best_epoch": int(final_epoch),
            "diffusion_stage": self.stage,
            "staged_representation": state.staged_representation,
            "search_space": search_space,
            "selection_metric": "val_loss",
            "max_epochs": (
                int(self.get("refit_best_max_epochs"))
                if refit_completed
                else max_epochs
            ),
            "patience": (
                int(self.get("refit_best_patience", self.get("refit_best_max_epochs")))
                if refit_completed
                else patience
            ),
            "search_max_epochs": max_epochs,
            "search_patience": patience,
        }
        meta_out.update(_hybrid_norm_metadata(norm_stats))
        if self.get("refit_best_max_epochs") is not None:
            meta_out["refit_best_max_epochs"] = int(self.get("refit_best_max_epochs"))
            meta_out["refit_completed"] = bool(refit_completed)
        if reuse_from:
            meta_out.update({
                "reuse_tuned_params_from": str(reuse_from),
                "retrain_reused_params": bool(self.get("retrain", False)),
                "reused_max_scale_policy": best_params.get("max_scale"),
                "reused_max_scale_previous": reuse_meta.get("reused_max_scale_previous"),
            })
        with open(os.path.join(subset_dir, "metadata.json"), "w", encoding="utf-8") as f:
            put_subset_record(meta_out, state.dataset, subset_meta)
            json.dump(meta_out, f, indent=2, sort_keys=True)

        self._record_finetune_result(state, final_ckpt, best_params)
        self._apply_tuned_length_to_state(state, best_params)

        wandb_utils.log_summary({
            f"hp/{self.stage}_diff_ft_best_val_loss": final_val,
            f"hp/{self.stage}_diff_ft_hp_best_val_loss": hp_best_val_loss,
            f"hp/{self.stage}_diff_ft_best_trial": best_trial_num,
            f"hp/{self.stage}_diff_ft_best_lr": best_params.get("learning_rate"),
            f"hp/{self.stage}_diff_ft_batch_size": best_params.get("batch_size"),
            f"hp/{self.stage}_diff_ft_effective_univariate_batch": best_params.get(
                "effective_univariate_batch"
            ),
            f"hp/{self.stage}_diff_ft_target_univariate_batch": best_params.get(
                "target_univariate_batch"
            ),
            f"hp/{self.stage}_diff_ft_max_scale": best_params.get("max_scale"),
            f"hp/{self.stage}_diff_ft_refit_completed": bool(refit_completed),
            f"hp/{self.stage}_diff_ft_binary_length_g": best_params.get("binary_length_g"),
        })

        self._log_post_finetune_viz_and_diagnostics(
            state,
            final_ckpt=final_ckpt,
            best_params=best_params,
            train_ds=train_ds,
        )

        token_cache = getattr(self, "_phase_token_cache", None)
        if token_cache is not None:
            token_cache.release()
            del self._phase_token_cache
            self._phase_cache_train_enabled = False

        return state


class CoarseDiffusionFinetuneHPPhase(_BaseStagedDiffusionFinetuneHPPhase):
    name = "diffusion_coarse_finetune_hp"
    stage = "coarse"


class PatchRefineDiffusionFinetuneHPPhase(_BaseStagedDiffusionFinetuneHPPhase):
    name = "diffusion_patch_refine_finetune_hp"
    stage = "patch_refine"
