"""YAML experiment config loader.

Loads a YAML file, merges with pipeline_config.py defaults, and
applies CLI overrides on top.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import yaml

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
        "dit_patch_size": [8, 8],
        "use_dual_scale": True,
        "dual_scale_fine_weight": 0.75,
        "dual_scale_independent_timesteps": True,
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

    # CLI overrides go into the experiment section
    if cli_overrides:
        exp = dict(cfg.get("experiment", {}))
        exp.update(cli_overrides)
        cfg["experiment"] = exp

    return cfg


def _dataset_n_variates(dataset: str) -> int:
    """Quick lookup of native variate count for known datasets."""
    _MAP = {
        "ETTh1": 7, "ETTh2": 7, "ETTm1": 7, "ETTm2": 7,
        "illness": 7, "exchange_rate": 8, "weather": 21,
        "electricity": 321, "traffic": 862, "PeMS": 307, "solar_Alabama": 137,
        "dalia": 5,
    }
    return _MAP.get(dataset, 7)
