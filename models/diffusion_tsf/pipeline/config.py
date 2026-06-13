"""YAML experiment config loader — YAML is the single source of truth."""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, Optional, Set

import yaml

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)

REQUIRED_TOP_LEVEL = ("experiment", "phases", "training", "visualization")

REQUIRED_EXPERIMENT_KEYS = (
    "name",
    "dataset",
    "n_variates",
    "seed",
    "diffusion_type",
    "model_type",
    "image_height",
    "coarse_image_height",
    "fine_image_height",
    "finer_image_height",
    "max_scale",
    "max_scale_by_dataset",
    "window_norm_std_floor",
    "dit_patch_size",
    "dit_embed_dim",
    "dit_depth",
    "dit_num_heads",
    "dit_mlp_ratio",
    "dit_dropout",
    "use_dual_scale",
    "use_triple_scale",
    "diffusion_stage",
    "dual_scale_fine_weight",
    "dual_scale_independent_timesteps",
    "use_guidance_channel",
    "cfg_dropout",
    "deterministic_anchor_loss",
    "deterministic_anchor_lambda",
    "deterministic_anchor_alpha",
    "eval_sampler",
    "disable_cross_attention",
    "cross_variate_context_bias",
    "use_window_normalization",
    "zero_guidance_forecast",
    "lookback_length",
    "forecast_length",
    "lookback_overlap",
    "itrans_d_model",
    "itrans_d_ff",
    "itrans_e_layers",
    "itrans_n_heads",
    "binary_noise_schedule",
    "binary_num_steps",
    "binary_beta_start",
    "binary_beta_end",
    "prediction_target",
    "loss_weighting",
    "min_snr_gamma",
    "use_coordinate_channel",
    "window_stride",
    "data_subset",
)

REQUIRED_TRAINING_KEYS = (
    "pretrain_epochs",
    "pretrain_diffusion_epochs",
    "pretrain_diffusion_max_epochs",
    "pretrain_synthetic_override",
    "synthetic_samples_full_cap",
    "synthetic_samples_hp_tune",
    "synthetic_samples_diff_tune",
    "synthetic_samples_min",
    "n_itrans_hp_trials",
    "n_diffusion_hp_trials",
    "n_finetune_hp_trials",
    "itrans_hp_pretrain_max_epochs",
    "itrans_hp_finetune_max_epochs",
    "diffusion_hp_patience",
    "hp_tune_epochs",
    "hp_tune_patience",
    "itrans_real_cold_start",
    "itrans_paper_batch_size",
    "itrans_paper_lr_grid",
    "itrans_paper_dropout",
    "diffusion_batch_size",
    "diffusion_batch_sizes",
    "finetune_batch_sizes",
    "diffusion_probe_target_effective_batch",
    "diffusion_probe_max_batch_cap",
    "diffusion_probe_min_batch",
    "finetune_hp_lr_min",
    "finetune_hp_lr_max",
    "use_amp",
    "use_gradient_checkpointing",
    "unet_max_chunk_size",
    "eval_num_samples",
    "emd_lambda",
    "guidance_penalty_weight",
    "past_loss_weight",
    "anchor_hp_lambda_min",
    "anchor_hp_lambda_max",
    "anchor_hp_alpha_min",
    "anchor_hp_alpha_max",
    "lr_scheduler_type",
    "lr_warmup_epochs",
    "max_scale_tuning",
    "max_scale_tuning_range",
)

REQUIRED_VISUALIZATION_KEYS = (
    "enabled",
    "n_samples",
    "n_dual_scale_vars",
    "jpeg_dpi",
    "dual_scale_sampler",
    "dual_scale_inference_steps",
)

# Maps YAML training keys -> train_multivariate_pipeline module attribute names.
_TRAINING_GLOBAL_MAP: Dict[str, str] = {
    "pretrain_epochs": "PRETRAIN_EPOCHS",
    "pretrain_diffusion_epochs": "PRETRAIN_DIFFUSION_EPOCHS",
    "pretrain_diffusion_max_epochs": "PRETRAIN_DIFFUSION_MAX_EPOCHS",
    "pretrain_synthetic_override": "PRETRAIN_SYNTHETIC_SAMPLES_OVERRIDE",
    "synthetic_samples_full_cap": "SYNTHETIC_SAMPLES_CAP",
    "synthetic_samples_hp_tune": "SYNTHETIC_SAMPLES_HP_TUNE",
    "synthetic_samples_diff_tune": "SYNTHETIC_SAMPLES_DIFF_TUNE",
    "synthetic_samples_min": "SYNTHETIC_SAMPLES_MIN",
    "n_itrans_hp_trials": "N_ITRANS_HP_TRIALS",
    "n_diffusion_hp_trials": "N_DIFFUSION_HP_TRIALS",
    "n_finetune_hp_trials": "N_FINETUNE_HP_TRIALS",
    "itrans_hp_pretrain_max_epochs": "ITRANS_HP_PRETRAIN_MAX_EPOCHS",
    "itrans_hp_finetune_max_epochs": "ITRANS_HP_FINETUNE_MAX_EPOCHS",
    "diffusion_hp_patience": "DIFFUSION_HP_PATIENCE",
    "hp_tune_epochs": "HP_TUNE_EPOCHS",
    "hp_tune_patience": "HP_TUNE_PATIENCE",
    "itrans_real_cold_start": "ITRANS_REAL_COLD_START",
    "itrans_paper_batch_size": "ITRANS_PAPER_BATCH_SIZE",
    "itrans_paper_lr_grid": "ITRANS_PAPER_LR_GRID",
    "itrans_paper_dropout": "ITRANS_PAPER_DROPOUT",
    "diffusion_batch_size": "DIFFUSION_BATCH_SIZE",
    "diffusion_batch_sizes": "DIFFUSION_BATCH_SIZES",
    "finetune_batch_sizes": "FINETUNE_BATCH_SIZES",
    "diffusion_probe_target_effective_batch": "DIFFUSION_PROBE_TARGET_EFFECTIVE_BATCH",
    "diffusion_probe_max_batch_cap": "DIFFUSION_PROBE_MAX_BATCH_CAP",
    "diffusion_probe_min_batch": "DIFFUSION_PROBE_MIN_BATCH",
    "finetune_hp_lr_min": "FINETUNE_HP_LR_MIN",
    "finetune_hp_lr_max": "FINETUNE_HP_LR_MAX",
    "use_amp": "USE_AMP",
    "use_gradient_checkpointing": "USE_GRADIENT_CHECKPOINTING",
    "unet_max_chunk_size": "UNET_MAX_CHUNK_SIZE",
    "eval_num_samples": "EVAL_NUM_SAMPLES",
    "emd_lambda": "EMD_LAMBDA",
    "guidance_penalty_weight": "GUIDANCE_PENALTY_WEIGHT",
    "past_loss_weight": "PAST_LOSS_WEIGHT",
    "anchor_hp_lambda_min": "ANCHOR_HP_LAMBDA_MIN",
    "anchor_hp_lambda_max": "ANCHOR_HP_LAMBDA_MAX",
    "anchor_hp_alpha_min": "ANCHOR_HP_ALPHA_MIN",
    "anchor_hp_alpha_max": "ANCHOR_HP_ALPHA_MAX",
    "lr_scheduler_type": "LR_SCHEDULER_TYPE",
    "lr_warmup_epochs": "LR_WARMUP_EPOCHS",
    "max_scale_tuning": "MAX_SCALE_TUNING",
    "max_scale_tuning_range": "MAX_SCALE_TUNING_RANGE",
}


def _is_phase_list(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    return all(isinstance(item, dict) and "phase" in item for item in value)


def _merge_phase_lists(base: list, override: list) -> list:
    """Merge phase override entries into base by ``phase`` name."""
    if not _is_phase_list(base):
        return list(override)
    if not _is_phase_list(override):
        return list(override)

    merged: Dict[str, Dict[str, Any]] = {}
    order: list[str] = []
    for entry in base:
        name = str(entry["phase"])
        merged[name] = dict(entry)
        order.append(name)

    for entry in override:
        name = str(entry["phase"])
        if name in merged:
            merged[name] = _deep_merge(merged[name], entry)
        else:
            merged[name] = dict(entry)
            order.append(name)

    return [merged[name] for name in order]


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k == "phases" and _is_phase_list(out.get(k)) and _is_phase_list(v):
            out[k] = _merge_phase_lists(out[k], v)
        elif k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _resolve_config_path(path: str, relative_to: str) -> str:
    if os.path.isabs(path):
        return os.path.abspath(path)
    for base in (os.path.dirname(relative_to), os.path.join(_REPO_ROOT, "configs"), _REPO_ROOT):
        candidate = os.path.normpath(os.path.join(base, path))
        if os.path.isfile(candidate):
            return candidate
    return os.path.normpath(os.path.join(os.path.dirname(relative_to), path))


def _load_yaml_tree(path: str, seen: Optional[Set[str]] = None) -> Dict[str, Any]:
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"config not found: {path}")
    seen = seen or set()
    if path in seen:
        raise ValueError(f"extends cycle detected at {path}")
    seen.add(path)

    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping: {path}")

    extends = raw.pop("extends", [])
    if isinstance(extends, str):
        extends = [extends]

    merged: Dict[str, Any] = {}
    for ext in extends:
        ext_path = _resolve_config_path(str(ext), path)
        merged = _deep_merge(merged, _load_yaml_tree(ext_path, seen))
    merged = _deep_merge(merged, raw)
    return merged


def validate_config(cfg: Dict[str, Any]) -> None:
    missing_top = [k for k in REQUIRED_TOP_LEVEL if k not in cfg]
    if missing_top:
        raise ValueError(f"config missing top-level section(s): {missing_top}")

    exp = cfg["experiment"]
    if not isinstance(exp, dict):
        raise ValueError("experiment must be a mapping")
    missing_exp = [k for k in REQUIRED_EXPERIMENT_KEYS if k not in exp]
    if missing_exp:
        raise ValueError(f"experiment missing required key(s): {missing_exp}")

    training = cfg["training"]
    if not isinstance(training, dict):
        raise ValueError("training must be a mapping")
    missing_train = [k for k in REQUIRED_TRAINING_KEYS if k not in training]
    if missing_train:
        raise ValueError(f"training missing required key(s): {missing_train}")

    phases = cfg["phases"]
    if not isinstance(phases, list) or not phases:
        raise ValueError("phases must be a non-empty list")
    for i, phase in enumerate(phases):
        if not isinstance(phase, dict) or "phase" not in phase:
            raise ValueError(f"phases[{i}] must be a mapping with a 'phase' key")

    viz = cfg["visualization"]
    if not isinstance(viz, dict):
        raise ValueError("visualization must be a mapping")
    missing_viz = [k for k in REQUIRED_VISUALIZATION_KEYS if k not in viz]
    if missing_viz:
        raise ValueError(f"visualization missing required key(s): {missing_viz}")


def load_experiment_config(
    yaml_path: str,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load experiment config from YAML (with extends), validate, apply CLI overrides."""
    if not yaml_path:
        raise ValueError("--config is required")

    cfg = _load_yaml_tree(os.path.abspath(yaml_path))
    cfg["_yaml_path"] = os.path.abspath(yaml_path)

    if cli_overrides:
        exp = dict(cfg.get("experiment", {}))
        exp.update(cli_overrides)
        cfg["experiment"] = exp

    validate_config(cfg)
    return cfg


def apply_training_config_to_module(mod: Any, cfg: Optional[Dict[str, Any]], state: Any = None) -> None:
    """Push training section from merged YAML onto pipeline module globals."""
    training = (cfg or {}).get("training")
    if not isinstance(training, dict):
        raise ValueError("merged config missing training section")
    for yaml_key, attr in _TRAINING_GLOBAL_MAP.items():
        if yaml_key not in training:
            raise KeyError(f"training.{yaml_key} required")
        setattr(mod, attr, training[yaml_key])
    if state is not None and hasattr(mod, "diffusion_probe_max_candidate"):
        n_v = getattr(state, "n_variates", getattr(mod, "N_VARIATES", 7))
        smoke = getattr(state, "smoke_test", False)
        mod._yaml_diffusion_probe_max_candidate = mod.diffusion_probe_max_candidate(n_v, smoke)


def build_wandb_config(
    cfg: Dict[str, Any],
    state: Any,
    *,
    phase_name: Optional[str] = None,
    phase_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from models.diffusion_tsf.train_multivariate_pipeline import get_git_info, get_system_info

    wandb_cfg = copy.deepcopy({k: v for k, v in cfg.items() if not k.startswith("_")})
    exp = dict(wandb_cfg.get("experiment") or {})
    exp.update({
        "checkpoint_dir": state.checkpoint_dir,
        "results_dir": state.results_dir,
        "datasets_dir": state.datasets_dir,
        "smoke_test": state.smoke_test,
        "resume": state.resume,
        "subset_id": state.subset_id,
        "variate_indices": state.variate_indices,
    })
    wandb_cfg["experiment"] = exp

    runtime: Dict[str, Any] = {
        "phase": phase_name,
        "parallel_optuna_workers": int(getattr(state, "parallel_optuna_workers", 1)),
    }
    if phase_overrides:
        runtime["phase_overrides"] = copy.deepcopy(phase_overrides)
    runtime.update(get_git_info())
    runtime.update(get_system_info())
    n_v = getattr(state, "n_variates", 7)
    smoke = getattr(state, "smoke_test", False)
    from models.diffusion_tsf.train_multivariate_pipeline import diffusion_probe_max_candidate
    runtime["diffusion_probe_max_candidate_default_v"] = diffusion_probe_max_candidate(n_v, smoke)
    wandb_cfg["runtime"] = runtime
    if cfg.get("_yaml_path"):
        wandb_cfg["_yaml_path"] = cfg["_yaml_path"]
    return wandb_cfg


def visualization_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return dict(cfg["visualization"])
