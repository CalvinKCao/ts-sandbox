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

GEOMETRY_KEYS = (
    "lookback_length",
    "forecast_length",
    "lookback_overlap",
    "diffusion_lookback_cap",
    "diffusion_chunk_horizon",
    "representation_time_stride",
    "past_cond_resize_to_horizon",
    "itrans_lookback_length",
    "mmpd_patch_size",
)

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
    "max_scale",
    "max_scale_by_dataset",
    "window_norm_std_floor",
    "window_norm_center",
    "dit_patch_size",
    "dit_cond_patch_size",
    "patch_refine_canvas_height",
    "patch_refine_patch_height",
    "patch_refine_patch_width",
    "patch_refine_col_stride",
    "patch_refine_unique_segments",
    "patch_refine_prev_cond_dropout",
    "dit_embed_dim",
    "dit_depth",
    "dit_num_heads",
    "dit_mlp_ratio",
    "dit_dropout",
    "diffusion_stage",
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
    "diffusion_lookback_cap",
    "diffusion_chunk_horizon",
    "representation_time_stride",
    "past_cond_resize_to_horizon",
    "itrans_lookback_length",
    "itrans_d_model",
    "itrans_d_ff",
    "itrans_e_layers",
    "itrans_n_heads",
    "binary_noise_schedule",
    "binary_length_mode",
    "binary_length_g",
    "binary_length_scale",
    "binary_length_g_by_dataset",
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

REMOVED_EXPERIMENT_KEYS = frozenset({
    "finer_image_height",
    "use_triple_scale",
    "use_vertical_dual_concat",
    "use_channel_dual_concat",
})

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
    "finetune_max_micro_batch",
    "finetune_hp_lr_min",
    "finetune_hp_lr_max",
    "use_amp",
    "use_gradient_checkpointing",
    "unet_max_chunk_size",
    "sequential_anchor_backward",
    "eval_num_samples",
    "past_loss_weight",
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

WANDB_DEFAULTS: Dict[str, Any] = {
    "enabled": False,
    "project": "ts-sandbox-leaderboard",
    "group": None,
    "tags": None,
}

# CLI keys merged into experiment vs applied directly to PipelineState.
CLI_EXPERIMENT_KEYS = frozenset({
    "dataset",
    "n_variates",
    "variate_indices",
    "seed",
    "seeds",
    "subset_id",
    "ckpt_config",
    "walltime",
    "existing_ckpt_roots",
    "mmpd_root",
    "ordinal_disc_evaluator",
    "ordinal_binary_config",
    "disc_run",
    "raw_run",
    "slice_lengths",
})

CLI_STATE_KEYS = frozenset({
    "checkpoint_dir",
    "results_dir",
    "datasets_dir",
    "synth_cache_dir",
    "smoke_test",
    "resume",
    "fresh",
})

# Every training value is held by PipelineState.  Keeping this list explicit
# makes YAML validation and state construction use the same contract.
TRAINING_STATE_KEYS = REQUIRED_TRAINING_KEYS

TRAINING_EXTRA_KEYS = (
    "use_hardcoded_synthetic_hp",
    "fixed_synthetic_diff_hp",
    "skip_synthetic_tuning",
    "force_retrain_synthetic",
    "diffusion_ema_decay",
    "diffusion_effective_batch_multiplier",
)


def training_section(state: Any) -> Dict[str, Any]:
    cfg = getattr(state, "merged_config", None) or {}
    training = cfg.get("training")
    if not isinstance(training, dict):
        return {}
    return training


def training_value(state: Any, key: str, default: Any = None) -> Any:
    """Read training.<key> from merged YAML, else PipelineState field."""
    training = training_section(state)
    if key in training:
        return training[key]
    return getattr(state, key, default)


def apply_wandb_section_to_state(
    wandb_section: Optional[Dict[str, Any]],
    init_kwargs: Dict[str, Any],
) -> None:
    merged = dict(WANDB_DEFAULTS)
    if isinstance(wandb_section, dict):
        merged.update(wandb_section)

    init_kwargs["wandb_enabled"] = bool(merged["enabled"])
    init_kwargs["wandb_project"] = str(merged["project"])
    group = merged.get("group")
    init_kwargs["wandb_group"] = str(group) if group else None
    tags = merged.get("tags")
    init_kwargs["wandb_tags"] = list(tags) if tags else None


def wandb_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(WANDB_DEFAULTS)
    section = cfg.get("wandb")
    if isinstance(section, dict):
        merged.update(section)
    return merged


def apply_training_section_to_state(
    training: Dict[str, Any],
    init_kwargs: Dict[str, Any],
    extra: Dict[str, Any],
) -> None:
    for key in TRAINING_STATE_KEYS:
        if key not in training:
            continue
        value = training[key]
        if key in ("n_itrans_hp_trials", "n_diffusion_hp_trials", "n_finetune_hp_trials", "lr_warmup_epochs"):
            init_kwargs[key] = int(value)
        elif key == "max_scale_tuning":
            init_kwargs[key] = bool(value)
        elif key == "max_scale_tuning_range":
            init_kwargs[key] = [float(x) for x in value]
        else:
            init_kwargs[key] = value
    for key in TRAINING_EXTRA_KEYS:
        if key in training:
            extra[key] = training[key]


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


def normalize_guidance_phases(
    phases: list,
    guidance_type: str,
    *,
    experiment: Optional[Dict[str, Any]] = None,
) -> list:
    """Normalize merged phase lists for the live guidance and patch-refine path."""
    by_name: Dict[str, Dict[str, Any]] = {}
    for entry in phases:
        name = str(entry["phase"])
        if guidance_type == "patch_decoder" and name == "itrans_finetune_hp":
            continue
        by_name[name] = dict(entry)
    exp = experiment or {}
    # Match DiffusionTSFConfig / PipelineState defaults (use_guidance_channel=True).
    needs_guidance = bool(exp.get("use_guidance_channel", True)) or not bool(
        exp.get("disable_cross_attention", False)
    )
    if not needs_guidance:
        by_name.pop("patch_guidance_finetune_hp", None)
        by_name.pop("itrans_finetune_hp", None)
    preferred = (
        "staged_diffusion_pretrain",
        "patch_guidance_finetune_hp",
        "diffusion_coarse_finetune_hp",
        "diffusion_patch_refine_finetune_hp",
        "staged_eval",
    )
    ordered = [by_name[n] for n in preferred if n in by_name]
    seen = {str(p["phase"]) for p in ordered}
    for entry in phases:
        name = str(entry["phase"])
        if name not in seen and name != "itrans_finetune_hp" and name in by_name:
            ordered.append(dict(by_name[name]))
            seen.add(name)
    return ordered


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
    # Leaf opt-in: replace inherited phase list instead of merging by phase name.
    replace_phases = bool(raw.pop("replace_phases", False))
    phases_override = raw.get("phases") if replace_phases else None

    merged: Dict[str, Any] = {}
    for ext in extends:
        ext_path = _resolve_config_path(str(ext), path)
        merged = _deep_merge(merged, _load_yaml_tree(ext_path, seen))
    merged = _deep_merge(merged, raw)
    if replace_phases:
        if not _is_phase_list(phases_override):
            raise ValueError(f"replace_phases=true requires a non-empty phases list: {path}")
        merged["phases"] = [dict(p) for p in phases_override]
    return _apply_geometry_block(merged)


def _apply_geometry_block(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Merge optional ``geometry:`` into ``experiment`` (geometry wins for its keys).

    Leaf overrides of parent geometry must set ``geometry:`` (not only ``experiment:``),
    e.g. uncompressed sets ``geometry.representation_time_stride: 1``.
    """
    geom = cfg.get("geometry")
    if geom is None:
        return cfg
    if not isinstance(geom, dict):
        raise ValueError("geometry must be a mapping")
    unknown = set(geom) - set(GEOMETRY_KEYS)
    if unknown:
        raise ValueError(f"unknown geometry key(s): {sorted(unknown)}")
    exp = dict(cfg.get("experiment") or {})
    for key in GEOMETRY_KEYS:
        if key in geom:
            exp[key] = geom[key]
    cfg["experiment"] = exp
    return cfg


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
    removed_exp = sorted(set(exp) & REMOVED_EXPERIMENT_KEYS)
    if removed_exp:
        raise ValueError(
            f"experiment uses removed representation setting(s): {removed_exp}"
        )

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
        state_overrides: Dict[str, Any] = {}
        for key, value in cli_overrides.items():
            if key in CLI_EXPERIMENT_KEYS:
                exp[key] = value
            elif key in CLI_STATE_KEYS:
                state_overrides[key] = value
            else:
                raise ValueError(
                    f"unsupported CLI override {key!r}; wandb settings belong in the YAML wandb: section"
                )
        cfg["experiment"] = exp
        if state_overrides:
            cfg["_cli_state_overrides"] = state_overrides

    validate_config(cfg)
    guidance_type = str(cfg.get("experiment", {}).get("guidance_type", "patch_decoder"))
    cfg["phases"] = normalize_guidance_phases(
        cfg["phases"],
        guidance_type,
        experiment=cfg.get("experiment") or {},
    )
    return cfg


def apply_cli_state_overrides(state: Any, cfg: Dict[str, Any]) -> None:
    """Apply runtime CLI flags onto PipelineState after YAML load."""
    from dataclasses import fields

    overrides = cfg.get("_cli_state_overrides") or {}
    known_fields = {f.name for f in fields(state)}
    for key, value in overrides.items():
        if key not in known_fields:
            raise ValueError(f"unsupported CLI state override: {key!r}")
        setattr(state, key, value)


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
        "lookback_length": state.lookback_length,
        "forecast_length": state.forecast_length,
    })
    wandb_cfg["experiment"] = exp
    if getattr(state, "dataset", None):
        wandb_cfg["dataset"] = state.dataset
    wandb_cfg["leaderboard_lookback"] = state.lookback_length
    wandb_cfg["leaderboard_horizon"] = state.forecast_length

    runtime: Dict[str, Any] = {
        "phase": phase_name,
        "parallel_optuna_workers": int(getattr(state, "parallel_optuna_workers", 1)),
    }
    if phase_overrides:
        runtime["phase_overrides"] = copy.deepcopy(phase_overrides)
    runtime.update(get_git_info())
    runtime.update(get_system_info())
    wandb_cfg["runtime"] = runtime
    if cfg.get("_yaml_path"):
        wandb_cfg["_yaml_path"] = cfg["_yaml_path"]
        try:
            from utils.leaderboard_config_nicknames import leaderboard_nickname

            nick = leaderboard_nickname(yaml_path=cfg["_yaml_path"])
            if nick:
                wandb_cfg["config_nickname"] = nick
        except Exception:
            pass
    return wandb_cfg


def visualization_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return dict(cfg["visualization"])


def logging_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return dict(cfg.get("logging", {"diagnostics_enabled": True}))
