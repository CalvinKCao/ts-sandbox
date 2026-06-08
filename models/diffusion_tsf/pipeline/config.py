"""YAML experiment config loader.

Loads a YAML file, merges with pipeline_config.py defaults, and
applies CLI overrides on top. The merged YAML is the single source of
truth for wandb ``config`` on pipeline runs.
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, Optional

import yaml

from models.diffusion_tsf import pipeline_config as pc

# Defaults sourced from pipeline_config.py — only the knobs that the
# binary-anchor DiT pipeline actually uses.
_DEFAULTS: Dict[str, Any] = {
    "experiment": {
        "name": "experiment",
        "dataset": "ETTh1",
        "n_variates": 7,
        "seed": 42,
        "diffusion_type": "binary",
        "model_type": "dit",
        "image_height": 16,
        "coarse_image_height": 16,
        "fine_image_height": 16,
        "finer_image_height": 16,
        "max_scale": 3.5,
        "max_scale_by_dataset": {},
        "window_norm_std_floor": 1e-8,
        "dit_patch_size": [8, 8],
        "dit_embed_dim": 384,
        "dit_depth": 8,
        "dit_num_heads": 6,
        "dit_mlp_ratio": 4.0,
        "dit_dropout": 0.0,
        "use_dual_scale": True,
        "use_triple_scale": False,
        "diffusion_stage": "joint",
        "dual_scale_fine_weight": 0.75,
        "dual_scale_independent_timesteps": True,
        "use_guidance_channel": True,
        "cfg_dropout": 0.1,
        "cfg_scale": 2.0,
        "use_cfg_inference": False,
        "deterministic_anchor_loss": True,
        "deterministic_anchor_lambda": 0.99,
        "deterministic_anchor_alpha": 0.0,
        "eval_sampler": "anchor",
        "disable_cross_attention": False,
        "cross_variate_context_bias": 0.0,
        "use_window_normalization": True,
        "zero_guidance_forecast": False,
        "lookback_length": 96,
        "forecast_length": 96,
        "lookback_overlap": 8,
        "window_stride": 1,
        "data_subset": {
            "enabled": False,
        },
    },
    "phases": [
        {"phase": "itrans_hp_pretrain", "n_trials": 10, "max_epochs": 10},
        {"phase": "diffusion_hp_pretrain", "n_trials": 8, "max_epochs": 5, "patience": 4},
        {"phase": "itrans_finetune_hp", "n_trials": 10, "max_epochs": 10, "cold_start": True},
        {"phase": "diffusion_finetune_hp", "n_trials": 5, "max_epochs": 20, "patience": 15},
        {"phase": "eval", "n_samples": 100, "probabilistic_n_samples": 100},
    ],
    "training": {
        "pretrain_epochs": pc.PRETRAIN_EPOCHS,
        "pretrain_diffusion_epochs": pc.PRETRAIN_DIFFUSION_EPOCHS,
        "pretrain_diffusion_max_epochs": pc.PRETRAIN_DIFFUSION_MAX_EPOCHS,
        "pretrain_synthetic_override": pc.PRETRAIN_SYNTHETIC_SAMPLES_OVERRIDE,
        "synthetic_samples_full_cap": pc.SYNTHETIC_SAMPLES_CAP,
        "synthetic_samples_hp_tune": pc.SYNTHETIC_SAMPLES_HP_TUNE,
        "synthetic_samples_diff_tune": pc.SYNTHETIC_SAMPLES_DIFF_TUNE,
        "n_itrans_hp_trials": pc.N_ITRANS_HP_TRIALS,
        "n_diffusion_hp_trials": pc.N_DIFFUSION_HP_TRIALS,
        "n_finetune_hp_trials": pc.N_FINETUNE_HP_TRIALS,
        "itrans_hp_pretrain_max_epochs": pc.ITRANS_HP_PRETRAIN_MAX_EPOCHS,
        "itrans_hp_finetune_max_epochs": pc.ITRANS_HP_FINETUNE_MAX_EPOCHS,
        "diffusion_hp_max_epochs": pc.DIFFUSION_HP_MAX_EPOCHS,
        "diffusion_hp_patience": pc.DIFFUSION_HP_PATIENCE,
        "hp_tune_epochs": pc.HP_TUNE_EPOCHS,
        "hp_tune_patience": pc.HP_TUNE_PATIENCE,
        "itrans_real_cold_start": pc.ITRANS_REAL_COLD_START,
        "itrans_paper_batch_size": pc.ITRANS_PAPER_BATCH_SIZE,
        "itrans_paper_lr_grid": list(pc.ITRANS_PAPER_LR_GRID),
        "itrans_paper_dropout": pc.ITRANS_PAPER_DROPOUT,
        "diffusion_batch_sizes": list(pc.DIFFUSION_BATCH_SIZES),
        "finetune_batch_sizes": list(pc.FINETUNE_BATCH_SIZES),
        "diffusion_probe_target_effective_batch": pc.DIFFUSION_PROBE_TARGET_EFFECTIVE_BATCH,
        "diffusion_probe_max_batch_cap": pc.DIFFUSION_PROBE_MAX_BATCH_CAP,
        "finetune_hp_lr_min": pc.FINETUNE_HP_LR_MIN,
        "finetune_hp_lr_max": pc.FINETUNE_HP_LR_MAX,
        "use_amp": pc.USE_AMP,
        "use_gradient_checkpointing": pc.USE_GRADIENT_CHECKPOINTING,
        "unet_max_chunk_size": pc.UNET_MAX_CHUNK_SIZE,
        "eval_num_samples": pc.EVAL_NUM_SAMPLES,
    },
    "visualization": {
        "enabled": True,
        "n_samples": 3,
        "n_dual_scale_vars": 3,
        "jpeg_dpi": 100,
        "dual_scale_sampler": "anchor",
        "dual_scale_inference_steps": 20,
    },
}

# Maps YAML training keys -> train_multivariate_pipeline module attribute names.
_TRAINING_GLOBAL_MAP: Dict[str, str] = {
    "pretrain_epochs": "PRETRAIN_EPOCHS",
    "pretrain_diffusion_epochs": "PRETRAIN_DIFFUSION_EPOCHS",
    "pretrain_diffusion_max_epochs": "PRETRAIN_DIFFUSION_MAX_EPOCHS",
    "pretrain_synthetic_override": "PRETRAIN_SYNTHETIC_SAMPLES_OVERRIDE",
    "synthetic_samples_full_cap": "SYNTHETIC_SAMPLES_CAP",
    "synthetic_samples_hp_tune": "SYNTHETIC_SAMPLES_HP_TUNE",
    "synthetic_samples_diff_tune": "SYNTHETIC_SAMPLES_DIFF_TUNE",
    "n_itrans_hp_trials": "N_ITRANS_HP_TRIALS",
    "n_diffusion_hp_trials": "N_DIFFUSION_HP_TRIALS",
    "n_finetune_hp_trials": "N_FINETUNE_HP_TRIALS",
    "itrans_hp_pretrain_max_epochs": "ITRANS_HP_PRETRAIN_MAX_EPOCHS",
    "itrans_hp_finetune_max_epochs": "ITRANS_HP_FINETUNE_MAX_EPOCHS",
    "diffusion_hp_max_epochs": "DIFFUSION_HP_MAX_EPOCHS",
    "diffusion_hp_patience": "DIFFUSION_HP_PATIENCE",
    "hp_tune_epochs": "HP_TUNE_EPOCHS",
    "hp_tune_patience": "HP_TUNE_PATIENCE",
    "itrans_real_cold_start": "ITRANS_REAL_COLD_START",
    "itrans_paper_batch_size": "ITRANS_PAPER_BATCH_SIZE",
    "itrans_paper_lr_grid": "ITRANS_PAPER_LR_GRID",
    "itrans_paper_dropout": "ITRANS_PAPER_DROPOUT",
    "diffusion_batch_sizes": "DIFFUSION_BATCH_SIZES",
    "finetune_batch_sizes": "FINETUNE_BATCH_SIZES",
    "diffusion_probe_target_effective_batch": "DIFFUSION_PROBE_TARGET_EFFECTIVE_BATCH",
    "diffusion_probe_max_batch_cap": "DIFFUSION_PROBE_MAX_BATCH_CAP",
    "finetune_hp_lr_min": "FINETUNE_HP_LR_MIN",
    "finetune_hp_lr_max": "FINETUNE_HP_LR_MAX",
    "use_amp": "USE_AMP",
    "use_gradient_checkpointing": "USE_GRADIENT_CHECKPOINTING",
    "unet_max_chunk_size": "UNET_MAX_CHUNK_SIZE",
    "eval_num_samples": "EVAL_NUM_SAMPLES",
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a copy of *base*."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_experiment_config(
    yaml_path: Optional[str] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load experiment config from YAML, merge with defaults and CLI overrides.

    Priority (highest wins): CLI overrides > YAML > built-in defaults.
    """
    cfg = dict(_DEFAULTS)

    if yaml_path is not None:
        with open(yaml_path) as f:
            yaml_cfg = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, yaml_cfg)
        cfg["_yaml_path"] = os.path.abspath(yaml_path)

    # CLI overrides go into the experiment section
    if cli_overrides:
        exp = dict(cfg.get("experiment", {}))
        exp.update(cli_overrides)
        cfg["experiment"] = exp

    return cfg


def apply_training_config_to_module(mod: Any, cfg: Optional[Dict[str, Any]], state: Any = None) -> None:
    """Push ``training`` section from merged YAML onto pipeline module globals."""
    training = (cfg or {}).get("training") or {}
    for yaml_key, attr in _TRAINING_GLOBAL_MAP.items():
        if yaml_key in training:
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
    """Build wandb config as a copy of merged YAML plus runtime metadata."""
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
        "ddp_enabled": False,
        "world_size": 1,
    }
    if phase_overrides:
        runtime["phase_overrides"] = copy.deepcopy(phase_overrides)
    try:
        from models.diffusion_tsf.train_multivariate_pipeline import _ddp_enabled, get_world_size
        runtime["ddp_enabled"] = bool(_ddp_enabled)
        runtime["world_size"] = int(get_world_size())
    except Exception:
        pass
    runtime.update(get_git_info())
    runtime.update(get_system_info())
    n_v = getattr(state, "n_variates", 7)
    smoke = getattr(state, "smoke_test", False)
    try:
        from models.diffusion_tsf.train_multivariate_pipeline import diffusion_probe_max_candidate
        runtime["diffusion_probe_max_candidate_default_v"] = diffusion_probe_max_candidate(n_v, smoke)
    except Exception:
        pass
    wandb_cfg["runtime"] = runtime
    if cfg.get("_yaml_path"):
        wandb_cfg["_yaml_path"] = cfg["_yaml_path"]
    return wandb_cfg


def visualization_settings(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return merged visualization knobs from YAML."""
    defaults = dict(_DEFAULTS.get("visualization") or {})
    if cfg:
        defaults.update(cfg.get("visualization") or {})
    return defaults


def sync_globals_to_merged_config(
    mod: Any,
    yaml_path: Optional[str] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Overlay live pipeline module globals onto merged YAML defaults."""
    cfg = load_experiment_config(yaml_path, cli_overrides)
    exp = cfg.setdefault("experiment", {})
    exp.update({
        "lookback_length": getattr(mod, "LOOKBACK_LENGTH", exp.get("lookback_length")),
        "forecast_length": getattr(mod, "FORECAST_LENGTH", exp.get("forecast_length")),
        "lookback_overlap": getattr(mod, "LOOKBACK_OVERLAP", exp.get("lookback_overlap", 8)),
        "image_height": getattr(mod, "IMAGE_HEIGHT", exp.get("image_height")),
        "coarse_image_height": getattr(mod, "COARSE_IMAGE_HEIGHT", exp.get("coarse_image_height")),
        "fine_image_height": getattr(mod, "FINE_IMAGE_HEIGHT", exp.get("fine_image_height")),
        "finer_image_height": getattr(mod, "FINER_IMAGE_HEIGHT", exp.get("finer_image_height")),
        "max_scale": getattr(mod, "MAX_SCALE", exp.get("max_scale")),
        "window_norm_std_floor": getattr(mod, "WINDOW_NORM_STD_FLOOR", exp.get("window_norm_std_floor")),
        "use_dual_scale": getattr(mod, "USE_DUAL_SCALE", exp.get("use_dual_scale")),
        "use_triple_scale": getattr(mod, "USE_TRIPLE_SCALE", exp.get("use_triple_scale")),
        "diffusion_stage": getattr(mod, "DIFFUSION_STAGE", exp.get("diffusion_stage")),
        "dual_scale_fine_weight": getattr(mod, "DUAL_SCALE_FINE_WEIGHT", exp.get("dual_scale_fine_weight")),
        "dual_scale_independent_timesteps": getattr(
            mod, "DUAL_SCALE_INDEPENDENT_TIMESTEPS", exp.get("dual_scale_independent_timesteps")
        ),
        "cfg_dropout": getattr(mod, "CFG_DROPOUT", exp.get("cfg_dropout")),
        "cfg_scale": getattr(mod, "CFG_SCALE", exp.get("cfg_scale")),
        "use_cfg_inference": getattr(mod, "USE_CFG_INFERENCE", exp.get("use_cfg_inference")),
        "cross_variate_context_bias": getattr(
            mod, "CROSS_VARIATE_CONTEXT_BIAS", exp.get("cross_variate_context_bias")
        ),
        "n_variates": getattr(mod, "N_VARIATES", exp.get("n_variates")),
        "diffusion_type": getattr(mod, "DIFFUSION_TYPE", exp.get("diffusion_type")),
        "model_type": getattr(mod, "MODEL_TYPE", exp.get("model_type")),
        "deterministic_anchor_loss": getattr(
            mod, "DETERMINISTIC_ANCHOR_LOSS", exp.get("deterministic_anchor_loss")
        ),
        "deterministic_anchor_lambda": getattr(
            mod, "DETERMINISTIC_ANCHOR_LAMBDA", exp.get("deterministic_anchor_lambda")
        ),
        "deterministic_anchor_alpha": getattr(
            mod, "DETERMINISTIC_ANCHOR_ALPHA", exp.get("deterministic_anchor_alpha")
        ),
        "eval_sampler": getattr(mod, "EVAL_SAMPLER", exp.get("eval_sampler")),
        "disable_cross_attention": getattr(mod, "DISABLE_CROSS_ATTENTION", exp.get("disable_cross_attention")),
        "use_window_normalization": getattr(
            mod, "USE_WINDOW_NORMALIZATION", exp.get("use_window_normalization")
        ),
        "zero_guidance_forecast": getattr(mod, "ZERO_GUIDANCE_FORECAST", exp.get("zero_guidance_forecast")),
        "window_stride": getattr(mod, "WINDOW_STRIDE", exp.get("window_stride", 1)),
        "dit_patch_size": list(getattr(mod, "DIT_PATCH_SIZE", exp.get("dit_patch_size", [8, 8]))),
        "dit_embed_dim": getattr(mod, "DIT_EMBED_DIM", exp.get("dit_embed_dim")),
        "dit_depth": getattr(mod, "DIT_DEPTH", exp.get("dit_depth")),
        "dit_num_heads": getattr(mod, "DIT_NUM_HEADS", exp.get("dit_num_heads")),
        "dit_mlp_ratio": getattr(mod, "DIT_MLP_RATIO", exp.get("dit_mlp_ratio")),
        "dit_dropout": getattr(mod, "DIT_DROPOUT", exp.get("dit_dropout")),
    })
    training = cfg.setdefault("training", {})
    for yaml_key, attr in _TRAINING_GLOBAL_MAP.items():
        if hasattr(mod, attr):
            training[yaml_key] = getattr(mod, attr)
    if hasattr(mod, "resolve_pretrain_virtual_dataset_size"):
        training["pretrain_virtual_samples"] = mod.resolve_pretrain_virtual_dataset_size(False)
    training["anchor_hp_lambda_min"] = getattr(mod, "ANCHOR_HP_LAMBDA_MIN", None)
    training["anchor_hp_lambda_max"] = getattr(mod, "ANCHOR_HP_LAMBDA_MAX", None)
    training["anchor_hp_alpha_min"] = getattr(mod, "ANCHOR_HP_ALPHA_MIN", None)
    training["anchor_hp_alpha_max"] = getattr(mod, "ANCHOR_HP_ALPHA_MAX", None)
    return cfg


class WandbStateShim:
    """Minimal state object for legacy init_wandb."""

    def __init__(self, mod: Any, **overrides: Any):
        self.checkpoint_dir = overrides.get("checkpoint_dir", getattr(mod, "CHECKPOINT_DIR", "."))
        self.results_dir = overrides.get("results_dir", getattr(mod, "RESULTS_DIR", "."))
        self.datasets_dir = overrides.get("datasets_dir", getattr(mod, "DATASETS_DIR", "."))
        self.smoke_test = overrides.get("smoke_test", False)
        self.resume = overrides.get("resume", False)
        self.seed = overrides.get("seed", 42)
        self.subset_id = overrides.get("subset_id")
        self.variate_indices = overrides.get("variate_indices")
        self.n_variates = overrides.get("n_variates", getattr(mod, "N_VARIATES", 7))
        self.dataset = overrides.get("dataset", "ETTh1")


def _dataset_n_variates(dataset: str) -> int:
    """Quick lookup of native variate count for known datasets."""
    _MAP = {
        "ETTh1": 7, "ETTh2": 7, "ETTm1": 7, "ETTm2": 7,
        "illness": 7, "exchange_rate": 8, "weather": 21,
        "electricity": 321, "traffic": 862, "PeMS": 307, "solar_Alabama": 137,
        "dalia": 5,
    }
    return _MAP.get(dataset, 7)
