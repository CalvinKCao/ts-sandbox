"""HP tuning for staged coarse/fine diffusion models; best trial checkpoint is final."""

from __future__ import annotations

import json
import logging
import math
import os
import time
from typing import Any, Dict, Optional, Tuple

import torch
from optuna.exceptions import TrialPruned
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from torch.utils.data import DataLoader, Subset

from models.diffusion_tsf.pipeline.phase import PipelinePhase
from models.diffusion_tsf.pipeline.config import training_value
from models.diffusion_tsf.pipeline.state import PipelineState
from models.diffusion_tsf.pipeline import wandb_utils
from models.diffusion_tsf.pipeline.haar_frequency_calibration import ensure_haar_frequency_calibration
from models.diffusion_tsf.pipeline.fourier_frequency_calibration import ensure_fourier_frequency_calibration
from models.diffusion_tsf.pipeline.phases.staged_diffusion_pretrain import (
    _stage_pretrain_ckpt,
    discover_dataset_run_ckpt_dir,
    patch_stage_globals,
)

logger = logging.getLogger(__name__)


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
    "prediction_target",
    "loss_weighting",
    "min_snr_gamma",
    "dit_dropout",
)


def _stage_subset_dir(state: PipelineState, stage: str) -> str:
    subset_id = state.subset_id or state.dataset
    return os.path.join(state.checkpoint_dir, subset_id, stage)


def _stage_best_ckpt(state: PipelineState, stage: str) -> str:
    return os.path.join(_stage_subset_dir(state, stage), "best.pt")


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


class _Ema:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if torch.is_floating_point(v)
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        state = model.state_dict()
        for key, avg in self.shadow.items():
            avg.mul_(self.decay).add_(state[key].detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def swap_in(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        state = model.state_dict()
        backup = {key: state[key].detach().clone() for key in self.shadow}
        for key, avg in self.shadow.items():
            state[key].copy_(avg)
        return backup

    @torch.no_grad()
    def restore(self, model: torch.nn.Module, backup: Dict[str, torch.Tensor]) -> None:
        state = model.state_dict()
        for key, value in backup.items():
            state[key].copy_(value)


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
    if not fixed:
        raise ValueError("search_space=fixed requires fixed_tuned_params in phase YAML")
    params = dict(fixed)
    params.setdefault(
        "max_scale",
        float(state.max_scale_by_dataset.get(state.dataset, state.max_scale)),
    )
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
    from models.diffusion_tsf.train_multivariate_pipeline import (
        FINETUNE_HP_LR_MAX,
        FINETUNE_HP_LR_MIN,
    )

    overrides = phase_overrides or {}

    if search_space == "reduced_hp":
        return _suggest_reduced_hp_params(
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
        from models.diffusion_tsf.train_multivariate_pipeline import (
            FINETUNE_HP_LR_MAX,
            FINETUNE_HP_LR_MIN,
        )

        if FINETUNE_HP_LR_MIN == FINETUNE_HP_LR_MAX:
            lr = float(FINETUNE_HP_LR_MIN)
        else:
            lr = trial.suggest_float(
                "learning_rate", FINETUNE_HP_LR_MIN, FINETUNE_HP_LR_MAX, log=True
            )
        return _apply_effective_batch_multiplier(
            {
                "learning_rate": lr,
                "batch_size": max(1, max_batch_size),
                "ema_decay": float(state.extra.get("diffusion_ema_decay", 0.0)),
                "binary_noise_schedule": state.binary_noise_schedule,
                "loss_weighting": state.loss_weighting,
                "min_snr_gamma": float(state.min_snr_gamma),
                "prediction_target": state.prediction_target,
                "max_scale": ms,
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

    def should_skip(self, state: PipelineState) -> bool:
        if self.stage == "finer" and not getattr(state, "use_triple_scale", False):
            logger.info("  [%s] skipping: use_triple_scale=False", self.name)
            return True
        best_pt = _stage_best_ckpt(state, self.stage)
        meta = os.path.join(_stage_subset_dir(state, self.stage), "metadata.json")
        if os.path.exists(best_pt) and os.path.exists(meta):
            logger.info("  [%s] cached: %s", self.name, best_pt)
            params = None
            try:
                with open(meta) as f:
                    params = json.load(f).get("tuned_params")
            except Exception as e:
                logger.warning("Failed to load tuned params from %s: %s", meta, e)
            if self.stage == "coarse":
                state.diffusion_coarse_finetune_ckpt = best_pt
                state.coarse_finetune_best_params = params
            elif self.stage == "fine":
                state.diffusion_fine_finetune_ckpt = best_pt
                state.fine_finetune_best_params = params
            else:
                state.diffusion_finer_finetune_ckpt = best_pt
                state.finer_finetune_best_params = params
            return True
        return False

    def _pretrained_ckpt(self, state: PipelineState) -> str:
        attr = {
            "coarse": state.diffusion_coarse_pretrain_ckpt,
            "fine": state.diffusion_fine_pretrain_ckpt,
            "finer": state.diffusion_finer_pretrain_ckpt,
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
        from models.diffusion_tsf.train_multivariate_pipeline import (
            anchor_kwargs_from_params,
            create_diffusion_model,
            dataset_window_lengths,
        )

        ds_lb, ds_hz = dataset_window_lengths(state.dataset)
        model_kwargs = anchor_kwargs_from_params(params)
        model_kwargs.update(_state_anchor_kwargs(state))
        model_kwargs.update(_model_kwargs_from_tuned(params))
        return create_diffusion_model(
            n_variates=n_iv,
            lookback=ds_lb,
            horizon=ds_hz,
            guidance_model=itrans_guidance,
            diffusion_stage=self.stage,
            use_guidance_channel=state.use_guidance_channel,
            **model_kwargs,
        ).to(device)

    def _train_once(
        self,
        *,
        state: PipelineState,
        train_ds,
        val_ds,
        params: Dict[str, Any],
        pretrained_path: str,
        itrans_checkpoint: str,
        device: torch.device,
        variate_indices,
        ckpt_path: Optional[str],
        max_epochs: int,
        patience: int,
        trial=None,
    ) -> Tuple[float, int]:
        from models.diffusion_tsf.train_multivariate_pipeline import (
            EarlyStopping,
            amp_context,
            dataset_window_lengths,
            load_diffusion_state_keep_attached_guidance,
            load_wrapped_guidance,
            save_checkpoint,
            unwrap_model,
        )

        params = _with_state_anchor_params(params, state)
        n_iv = len(variate_indices)
        batch_size = int(params["batch_size"])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        n_train_batches = len(train_loader)
        n_val_batches = len(val_loader)
        trial_label = (
            f"trial={trial.number}" if trial is not None else "trial=single"
        )
        logger.info(
            "  [%s/%s] %s START epochs=%d patience=%d lr=%.2e bs=%d accum=%d "
            "train_batches=%d val_batches=%d",
            self.name,
            self.stage,
            trial_label,
            max_epochs,
            patience,
            float(params["learning_rate"]),
            batch_size,
            int(params.get("gradient_accumulation_steps", 1)),
            n_train_batches,
            n_val_batches,
        )
        _log_gpu_mem(f"{self.stage}/{trial_label}/start")

        ds_lb, ds_hz = dataset_window_lengths(state.dataset)
        guidance = load_wrapped_guidance(
            itrans_checkpoint,
            n_iv,
            device,
            guidance_type=state.guidance_type,
            dataset_lookback=ds_lb,
            dataset_horizon=ds_hz,
        )
        model = self._build_model(
            state=state,
            n_iv=n_iv,
            itrans_guidance=guidance,
            device=device,
            params=params,
        )
        try:
            ckpt = torch.load(pretrained_path, map_location=device, weights_only=False)
            load_diffusion_state_keep_attached_guidance(model, ckpt["model_state_dict"])

            optimizer = torch.optim.AdamW(model.parameters(), lr=float(params["learning_rate"]))

            lr_scheduler_type = str(training_value(state, "lr_scheduler_type", "none"))
            warmup_epochs = int(training_value(state, "lr_warmup_epochs", 0))
            warmup_epochs = min(warmup_epochs, max(0, max_epochs - 1))
            
            scheduler = None
            if lr_scheduler_type == "cosine":
                if warmup_epochs > 0:
                    scheduler = torch.optim.lr_scheduler.SequentialLR(
                        optimizer,
                        schedulers=[
                            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs),
                            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs - warmup_epochs, eta_min=float(params["learning_rate"]) * 0.01)
                        ],
                        milestones=[warmup_epochs]
                    )
                else:
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=float(params["learning_rate"]) * 0.01)
            elif lr_scheduler_type == "linear":
                if warmup_epochs > 0:
                    scheduler = torch.optim.lr_scheduler.SequentialLR(
                        optimizer,
                        schedulers=[
                            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs),
                            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=max_epochs - warmup_epochs)
                        ],
                        milestones=[warmup_epochs]
                    )
                else:
                    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=max_epochs)

            early_stop = EarlyStopping(patience=patience)
            ema = _Ema(model, float(params.get("ema_decay", 0.0))) if params.get("ema_decay", 0.0) else None
            accum_steps = max(1, int(params.get("gradient_accumulation_steps", 1)))
            best_val = float("inf")
            best_epoch = 0
            saved_ckpt = False
            train_log_stride = max(1, n_train_batches // 4)
            val_log_stride = max(1, n_val_batches // 2)
            epoch_t0 = time.perf_counter()

            for epoch in range(max_epochs):
                epoch_start = time.perf_counter()
                logger.info(
                    "  [%s/%s] %s epoch %d/%d train_start",
                    self.name, self.stage, trial_label, epoch + 1, max_epochs,
                )
                model.train()
                from models.diffusion_tsf.train_multivariate_pipeline import _set_ordinal_loader_mode

                _set_ordinal_loader_mode(model, train_loader, eval_mode=False)
                train_loss = 0.0
                n_train = 0
                optimizer.zero_grad(set_to_none=True)
                for batch_idx, (past, future) in enumerate(train_loader):
                    if batch_idx == 0 or (batch_idx + 1) % train_log_stride == 0 or batch_idx + 1 == n_train_batches:
                        logger.info(
                            "  [%s/%s] %s epoch %d/%d train_batch %d/%d",
                            self.name, self.stage, trial_label,
                            epoch + 1, max_epochs, batch_idx + 1, n_train_batches,
                        )
                    past, future = past.to(device), future.to(device)
                    with amp_context():
                        loss = model.get_loss(past, future) / accum_steps
                    loss.backward()
                    if (batch_idx + 1) % accum_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        if ema is not None:
                            ema.update(model)
                    train_loss += float(loss.item()) * accum_steps
                    n_train += 1
                if accum_steps > 1 and len(train_loader) % accum_steps != 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if ema is not None:
                        ema.update(model)

                train_loss_avg = train_loss / max(n_train, 1)
                train_elapsed = time.perf_counter() - epoch_start
                logger.info(
                    "  [%s/%s] %s epoch %d/%d train_done loss=%.4f time=%.1fs",
                    self.name, self.stage, trial_label,
                    epoch + 1, max_epochs, train_loss_avg, train_elapsed,
                )

                if scheduler is not None:
                    scheduler.step()

                backup = ema.swap_in(model) if ema is not None else None
                model.eval()
                _set_ordinal_loader_mode(model, val_loader, eval_mode=True)
                val_loss = 0.0
                n_val = 0
                val_start = time.perf_counter()
                logger.info(
                    "  [%s/%s] %s epoch %d/%d val_start",
                    self.name, self.stage, trial_label, epoch + 1, max_epochs,
                )
                with torch.no_grad():
                    for val_idx, (past, future) in enumerate(val_loader):
                        if val_idx == 0 or (val_idx + 1) % val_log_stride == 0 or val_idx + 1 == n_val_batches:
                            logger.info(
                                "  [%s/%s] %s epoch %d/%d val_batch %d/%d",
                                self.name, self.stage, trial_label,
                                epoch + 1, max_epochs, val_idx + 1, n_val_batches,
                            )
                        past, future = past.to(device), future.to(device)
                        with amp_context():
                            loss = model.get_loss(past, future)
                        val_loss += float(loss.item())
                        n_val += 1
                val_loss /= max(n_val, 1)
                val_elapsed = time.perf_counter() - val_start
                lr_now = float(optimizer.param_groups[0]["lr"])
                saved = val_loss < best_val
                if saved:
                    best_val = val_loss
                    best_epoch = epoch + 1
                    config = {
                        "tuned_params": dict(params),
                        "diffusion_stage": self.stage,
                        "best_epoch": best_epoch,
                    }
                    if ckpt_path:
                        save_checkpoint(
                            unwrap_model(model),
                            optimizer,
                            epoch,
                            train_loss_avg,
                            val_loss,
                            config,
                            ckpt_path,
                        )
                        saved_ckpt = True
                        if trial is not None:
                            trial.set_user_attr("ckpt_path", ckpt_path)
                if backup is not None:
                    ema.restore(model, backup)

                logger.info(
                    "  [%s/%s] %s epoch %d/%d done train=%.4f val=%.4f best=%.4f "
                    "best_ep=%d lr=%.2e saved=%s train_t=%.1fs val_t=%.1fs epoch_t=%.1fs",
                    self.name, self.stage, trial_label,
                    epoch + 1, max_epochs,
                    train_loss_avg, val_loss, best_val, best_epoch, lr_now,
                    saved, train_elapsed, val_elapsed, time.perf_counter() - epoch_start,
                )
                _log_gpu_mem(f"{self.stage}/{trial_label}/ep{epoch + 1}")

                if trial is not None:
                    trial.report(val_loss, epoch)
                    if trial.should_prune():
                        logger.info(
                            "  [%s/%s] %s epoch %d/%d PRUNED val=%.4f",
                            self.name, self.stage, trial_label, epoch + 1, max_epochs, val_loss,
                        )
                        raise TrialPruned()
                if early_stop(val_loss):
                    logger.info(
                        "  [%s/%s] %s epoch %d/%d EARLY_STOP val=%.4f best=%.4f best_ep=%d",
                        self.name, self.stage, trial_label,
                        epoch + 1, max_epochs, val_loss, best_val, best_epoch,
                    )
                    break

            total_elapsed = time.perf_counter() - epoch_t0
            if ckpt_path and best_epoch > 0 and not saved_ckpt:
                raise RuntimeError(
                    f"{trial_label}: best_val={best_val:.4f} at epoch {best_epoch} "
                    f"but no checkpoint was written to {ckpt_path}"
                )
            if ckpt_path and saved_ckpt and not os.path.isfile(ckpt_path):
                raise RuntimeError(
                    f"{trial_label}: expected checkpoint at {ckpt_path} after save"
                )
            logger.info(
                "  [%s/%s] %s DONE best_val=%.4f best_epoch=%d total_time=%.1fs",
                self.name, self.stage, trial_label, best_val, best_epoch, total_elapsed,
            )
            return best_val, best_epoch
        finally:
            del model, guidance
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def execute(self, state: PipelineState) -> PipelineState:
        ensure_haar_frequency_calibration(state)
        ensure_fourier_frequency_calibration(state)
        from models.diffusion_tsf.train_multivariate_pipeline import (
            diffusion_probe_max_candidate,
            dataset_window_lengths,
            generate_dataset_job,
            load_dataset,
            load_wrapped_guidance,
            select_diffusion_batch_size,
        )
        import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod

        patch_stage_globals(pipeline_mod, state, self.stage, honor_dataset_windows=True)

        subset_id = state.subset_id or state.dataset
        variate_indices = state.variate_indices
        if variate_indices is None:
            variate_indices = generate_dataset_job(state.dataset)["variate_indices"]
        subset_meta = state.data_subset_resolved or {}
        train_stride = int(subset_meta.get("train_stride", state.window_stride))
        test_stride = int(subset_meta.get("test_stride", 1))

        ft_guidance_ckpt = state.guidance_finetune_ckpt
        if not ft_guidance_ckpt or not os.path.exists(ft_guidance_ckpt):
            ft_guidance_ckpt = state.default_guidance_finetune_ckpt_path()
        if not os.path.exists(ft_guidance_ckpt):
            raise RuntimeError(
                f"{self.name} requires finetuned guidance ({state.guidance_type}), got: {ft_guidance_ckpt}"
            )
        if self.stage == "fine" and not state.diffusion_coarse_finetune_ckpt:
            raise RuntimeError("fine staged tuning requires completed coarse best model first")
        if self.stage == "finer":
            if not state.use_triple_scale:
                raise RuntimeError("finer staged tuning requires use_triple_scale=True")
            if not state.diffusion_fine_finetune_ckpt:
                raise RuntimeError("finer staged tuning requires completed fine best model first")
        diff_ckpt = self._pretrained_ckpt(state)

        device = state.resolve_device()
        n_iv = len(variate_indices)
        train_ds, val_ds, _, norm_stats = load_dataset(
            state.dataset,
            variate_indices,
            stride=train_stride,
            test_stride=test_stride,
            ordinal_tie_atol=float(state.ordinal_tie_atol),
        )
        if norm_stats.get("ordinal_ladder") is not None:
            state.extra["global_ordinal_ladder"] = norm_stats["ordinal_ladder"]
        if state.smoke_test:
            train_ds = Subset(train_ds, list(range(min(4, len(train_ds)))))
            val_ds = Subset(val_ds, list(range(min(2, len(val_ds)))))
        logger.info(
            "  [%s] train/val windows=%d/%d",
            self.name, len(train_ds), len(val_ds),
        )

        from models.diffusion_tsf.pipeline.phase_diagnostics import run_phase_start_diagnostics
        from models.diffusion_tsf.pipeline.visualize_utils import (
            _load_staged_diffusion_from_ckpt,
            run_real_dataset_phase_diagnostics,
            run_staged_finetune_visualizations,
        )

        try:
            probe_model, _ = _load_staged_diffusion_from_ckpt(
                ckpt_path=diff_ckpt,
                stage=self.stage,
                itrans_ckpt_path=ft_guidance_ckpt,
                n_vars=n_iv,
                device=device,
                guidance_type=state.guidance_type,
            )
            phase_start = run_phase_start_diagnostics(
                state,
                phase_name=self.name,
                models=[probe_model],
                model_labels=[f"diffusion_{self.stage}"],
                datasets=[train_ds],
                dataset_prefixes=["dataset"],
                ckpt_info=[
                    {
                        "kind": state.guidance_type,
                        "path": ft_guidance_ckpt,
                        "n_variates": n_iv,
                        "lookback": int(state.lookback_length),
                        "horizon": int(state.forecast_length),
                    },
                    {
                        "kind": f"diffusion_pretrain_{self.stage}",
                        "path": diff_ckpt,
                        "n_variates": n_iv,
                        "lookback": int(state.lookback_length),
                        "horizon": int(state.forecast_length),
                    },
                ],
            )
            wandb_utils.log_phase_diagnostics_result({"summary": phase_start})
            del probe_model
        except Exception as e:
            logger.warning("[%s] phase-start diagnostics failed: %s", self.name, e, exc_info=True)

        batch_probe_ds = train_ds
        ds_lb, ds_hz = dataset_window_lengths(state.dataset)
        ft_guidance = load_wrapped_guidance(
            ft_guidance_ckpt,
            n_iv,
            device,
            guidance_type=state.guidance_type,
            dataset_lookback=ds_lb,
            dataset_horizon=ds_hz,
        )
        max_batch = select_diffusion_batch_size(
            phase_name=f"{self.stage.title()} Diff FT ({subset_id})",
            dataset=batch_probe_ds,
            device=device,
            itrans_guidance=ft_guidance,
            max_candidate=diffusion_probe_max_candidate(n_iv, state.smoke_test),
            smoke_test=state.smoke_test,
        )
        del ft_guidance
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        accum_mult = state.extra.get("diffusion_effective_batch_multiplier")
        if accum_mult is not None and float(accum_mult) > 1.0:
            batch_plan = resolve_diffusion_batch_and_accum(max_batch, accum_mult)
            logger.info(
                "  [%s] grad accum: probed_max=%d multiplier=%s -> micro=%d accum=%d effective=%d",
                self.name,
                max_batch,
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
        from models.diffusion_tsf.train_multivariate_pipeline import ensure_checkpoint_dir

        ensure_checkpoint_dir(final_ckpt := _stage_best_ckpt(state, self.stage))
        trials_dir = os.path.join(subset_dir, "_trials")
        ensure_checkpoint_dir(os.path.join(trials_dir, "_trial.pt"))

        reuse_meta: Dict[str, Any] = {}
        hp_best_val_loss: Optional[float] = None
        best_trial_num = -1
        final_val = float("nan")
        final_epoch = 0
        search_space = "lr_only"

        if reuse_from:
            best_params, source_dir, reuse_meta = _load_reused_stage_params(
                state, stage=self.stage, subset_id=subset_id, source_config=str(reuse_from),
            )
            search_space = str(reuse_meta.get("search_space") or self.get("search_space") or "lr_only").lower()
            tuned_bs = int(best_params.get("batch_size", max_batch))
            best_params["batch_size"] = min(tuned_bs, max_batch)
            best_params = _with_state_anchor_params(best_params, state)
            if retrain_reused:
                final_val, final_epoch = self._train_once(
                    state=state,
                    train_ds=train_ds,
                    val_ds=val_ds,
                    params=best_params,
                    pretrained_path=diff_ckpt,
                    itrans_checkpoint=ft_guidance_ckpt,
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
                    import shutil
                    shutil.copy2(src_best, final_ckpt)
                hp_best_val_loss = float(
                    reuse_meta.get("best_val_loss")
                    or reuse_meta.get("hp_best_val_loss")
                    or float("nan")
                )
                final_val = hp_best_val_loss
                final_epoch = int(reuse_meta.get("best_epoch", 0))
                logger.info("  [%s] reused %s from %s", self.name, self.stage, source_dir)
        else:
            n_trials = int(self.require("n_trials"))
            if state.smoke_test:
                n_trials = 1
            search_space = str(self.require("search_space")).lower()
            if search_space not in {
                "default", "lr_only", "full_with_batch", "reduced_hp", "fixed",
            }:
                raise ValueError(f"Unknown staged diffusion search_space={search_space!r}")
            if search_space == "reduced_hp":
                for key in ("hp_lr_min", "hp_lr_max"):
                    if self.get(key) is None:
                        raise ValueError(f"search_space=reduced_hp requires phase {key}")
            if search_space == "fixed" and not self.get("fixed_tuned_params"):
                raise ValueError("search_space=fixed requires fixed_tuned_params in phase YAML")

            if search_space == "fixed":
                best_params = _build_fixed_hp_params(
                    state, max_batch, state.smoke_test, self.overrides,
                )
                best_params = _with_state_anchor_params(best_params, state)
                final_val, final_epoch = self._train_once(
                    state=state,
                    train_ds=train_ds,
                    val_ds=val_ds,
                    params=best_params,
                    pretrained_path=diff_ckpt,
                    itrans_checkpoint=ft_guidance_ckpt,
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
                from models.diffusion_tsf.pipeline.optuna_parallel import run_optuna_study

                phase = self

                def objective_builder(_worker_id: int):
                    dev = state.resolve_device()

                    def objective(trial):
                        params = _suggest_staged_params(
                            trial,
                            state,
                            max_batch,
                            state.smoke_test,
                            search_space=search_space,
                            phase_overrides=phase.overrides,
                        )
                        trial.set_user_attr("full_params", dict(params))
                        if search_space in {"full_with_batch", "reduced_hp"} and not state.smoke_test:
                            micro = int(params["batch_size"])
                            accum = int(params.get("gradient_accumulation_steps", 1))
                            if micro < 4 or accum > 64:
                                raise RuntimeError(
                                    f"Degenerate batch plan micro_bs={micro} accum={accum} "
                                    f"(effective={micro * accum}); stale Optuna journal or planner bug"
                                )
                        logger.info(
                            "  [%s] Optuna trial %d/%d suggested lr=%.2e micro_bs=%d "
                            "accum=%d effective_bs=%d",
                            phase.name,
                            trial.number + 1,
                            n_trials,
                            float(params["learning_rate"]),
                            int(params["batch_size"]),
                            int(params.get("gradient_accumulation_steps", 1)),
                            int(params.get("effective_batch_size", params["batch_size"])),
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
                                itrans_checkpoint=ft_guidance_ckpt,
                                device=dev,
                                variate_indices=variate_indices,
                                ckpt_path=trial_ckpt,
                                max_epochs=max_epochs,
                                patience=patience,
                                trial=trial,
                            )
                        except torch.cuda.OutOfMemoryError:
                            logger.warning(
                                "  [%s] trial %d OOM (batch=%s), pruning",
                                phase.name, trial.number, params.get("batch_size"),
                            )
                            raise TrialPruned() from None
                        except TrialPruned:
                            logger.info(
                                "  [%s] Optuna trial %d pruned after %.1fs",
                                phase.name, trial.number, time.perf_counter() - trial_t0,
                            )
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

                logger.info(
                    "  [%s] Optuna study start: n_trials=%d max_epochs=%d patience=%d",
                    self.name, n_trials, max_epochs, patience,
                )
                study_t0 = time.perf_counter()
                study = run_optuna_study(
                    study_name=f"{state.experiment_name}-{self.stage}-hp",
                    checkpoint_dir=subset_dir,
                    n_trials=n_trials,
                    parallel_workers=state.parallel_optuna_workers,
                    direction="minimize",
                    objective_builder=objective_builder,
                    sampler=TPESampler(seed=state.seed, multivariate=True, group=True),
                    pruner=HyperbandPruner(
                        min_resource=1, max_resource=max_epochs, reduction_factor=3,
                    ),
                    sampler_seed=state.seed,
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

                import shutil
                src = _resolve_best_trial_ckpt(
                    study, trials_dir, subset_dir, best_trial_num,
                )
                shutil.copy2(src, final_ckpt)
                if not os.path.isfile(final_ckpt):
                    raise RuntimeError(f"Failed to promote best trial checkpoint to {final_ckpt}")
                final_val = hp_best_val_loss
                _cleanup_trial_ckpts(trials_dir, subset_dir, keep=src)

        meta_out: Dict[str, Any] = {
            "subset_id": subset_id,
            "dataset_name": state.dataset,
            "variate_indices": variate_indices,
            "data_subset": subset_meta,
            "norm_mean": norm_stats["mean"].tolist(),
            "norm_std": norm_stats["std"].tolist(),
            "tuned_params": best_params,
            "best_trial": best_trial_num,
            "hp_best_val_loss": hp_best_val_loss,
            "best_val_loss": float(final_val),
            "best_epoch": int(final_epoch),
            "diffusion_stage": self.stage,
            "staged_representation": state.staged_representation,
            "haar_high_freq_levels": int(state.haar_high_freq_levels),
            "haar_high_freq_percent": float(state.haar_high_freq_percent),
            "haar_fine_max_scale": float(state.haar_fine_max_scale),
            "haar_fine_scale_quantile": float(state.haar_fine_scale_quantile),
            "fourier_high_freq_cutoff_bin": int(state.fourier_high_freq_cutoff_bin),
            "fourier_high_freq_percent": float(state.fourier_high_freq_percent),
            "fourier_fine_max_scale": float(state.fourier_fine_max_scale),
            "fourier_flatline_atol": float(state.fourier_flatline_atol),
            "coarse_flatline_blur_fine_target": bool(state.coarse_flatline_blur_fine_target),
            "coarse_flatline_blur_radius": int(state.coarse_flatline_blur_radius),
            "coarse_flatline_blur_kernel": str(state.coarse_flatline_blur_kernel),
            "search_space": search_space,
            "max_epochs": max_epochs,
            "patience": patience,
        }
        if reuse_from:
            meta_out.update({
                "reuse_tuned_params_from": str(reuse_from),
                "retrain_reused_params": bool(self.get("retrain", False)),
                "reused_max_scale_policy": best_params.get("max_scale"),
                "reused_max_scale_previous": reuse_meta.get("reused_max_scale_previous"),
            })
        with open(os.path.join(subset_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta_out, f, indent=2, sort_keys=True)

        if self.stage == "coarse":
            state.diffusion_coarse_finetune_ckpt = final_ckpt
            state.coarse_finetune_best_params = best_params
        elif self.stage == "fine":
            state.diffusion_fine_finetune_ckpt = final_ckpt
            state.fine_finetune_best_params = best_params
        else:
            state.diffusion_finer_finetune_ckpt = final_ckpt
            state.finer_finetune_best_params = best_params

        wandb_utils.log_summary({
            f"hp/{self.stage}_diff_ft_best_val_loss": final_val,
            f"hp/{self.stage}_diff_ft_hp_best_val_loss": hp_best_val_loss,
            f"hp/{self.stage}_diff_ft_best_trial": best_trial_num,
            f"hp/{self.stage}_diff_ft_best_lr": best_params.get("learning_rate"),
            f"hp/{self.stage}_diff_ft_batch_size": best_params.get("batch_size"),
            f"hp/{self.stage}_diff_ft_max_scale": best_params.get("max_scale"),
        })

        coarse_ft = state.diffusion_coarse_finetune_ckpt or _stage_best_ckpt(state, "coarse")
        guidance_ckpt = state.guidance_finetune_ckpt or state.default_guidance_finetune_ckpt_path()
        if self.stage == "fine" and coarse_ft and final_ckpt and guidance_ckpt and os.path.exists(guidance_ckpt):
            try:
                viz_paths = run_staged_finetune_visualizations(
                    state,
                    coarse_ckpt_path=coarse_ft,
                    fine_ckpt_path=final_ckpt if self.stage == "fine" else _stage_best_ckpt(state, "fine"),
                    itrans_ckpt_path=guidance_ckpt,
                    tuned_params=best_params,
                    tag="staged_diffusion_finetuned",
                )
                wandb_utils.log_visualization_paths(
                    viz_paths, wandb_key="viz/staged_diffusion_finetuned",
                )
            except Exception as e:
                logger.warning("Staged finetune viz failed: %s", e, exc_info=True)

        if not state.smoke_test:
            try:
                finetuned_model, _ = _load_staged_diffusion_from_ckpt(
                    ckpt_path=final_ckpt,
                    stage=self.stage,
                    itrans_ckpt_path=ft_guidance_ckpt,
                    n_vars=n_iv,
                    device=device,
                    tuned_params=best_params,
                )
                diag = run_real_dataset_phase_diagnostics(
                    state,
                    train_ds=train_ds,
                    model=finetuned_model,
                    itrans_ckpt_path=ft_guidance_ckpt,
                    stage=self.stage,
                    diffusion_ckpt_path=final_ckpt,
                    coarse_ckpt_path=coarse_ft if self.stage == "fine" else None,
                    tag=f"diffusion_{self.stage}_finetune",
                    include_phase_start=False,
                )
                wandb_utils.log_phase_diagnostics_result(diag)
                del finetuned_model
            except Exception as e:
                logger.warning("[%s] post-finetune diagnostics failed: %s", self.name, e, exc_info=True)

        return state


class CoarseDiffusionFinetuneHPPhase(_BaseStagedDiffusionFinetuneHPPhase):
    name = "diffusion_coarse_finetune_hp"
    stage = "coarse"


class FineDiffusionFinetuneHPPhase(_BaseStagedDiffusionFinetuneHPPhase):
    name = "diffusion_fine_finetune_hp"
    stage = "fine"


class FinerDiffusionFinetuneHPPhase(_BaseStagedDiffusionFinetuneHPPhase):
    name = "diffusion_finer_finetune_hp"
    stage = "finer"
