"""Shared mutable state passed between pipeline phases.

Replaces the module-level globals in train_multivariate_pipeline.py with
an explicit dataclass. Phases read what they need and write back the
artifacts they produce (checkpoint paths, best HP params, etc.).
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from models.diffusion_tsf.pipeline.config import (
    REQUIRED_EXPERIMENT_KEYS,
    apply_training_section_to_state,
    apply_wandb_section_to_state,
)


@dataclass
class PipelineState:
    """Everything that flows between phases."""

    # -- Experiment identity (frozen after init) --
    experiment_name: str = "experiment"
    dataset: str = "ETTh1"
    n_variates: int = 7
    seed: int = 42
    smoke_test: bool = False
    parallel_optuna_workers: int = 1

    # -- Model / diffusion knobs --
    diffusion_type: str = "binary"
    d3pm_transition_max: float = 0.3
    d3pm_transition_min: float = 1e-5
    d3pm_neighbor_kernel: str = "gaussian"
    d3pm_noise_schedule: str = "sqrt_linear"
    d3pm_loss_type: str = "cross_entropy"
    model_type: str = "dit"
    image_height: int = 32
    coarse_image_height: int = 16
    fine_image_height: int = 16
    finer_image_height: int = 16
    max_scale: float = 3.5
    max_scale_by_dataset: Dict[str, float] = field(default_factory=dict)
    dit_patch_size: Tuple[int, int] = (8, 8)
    dit_embed_dim: int = 384
    dit_depth: int = 8
    dit_num_heads: int = 6
    dit_mlp_ratio: float = 4.0
    dit_dropout: float = 0.0
    use_dual_scale: bool = False
    use_triple_scale: bool = False
    diffusion_stage: str = "joint"
    dual_scale_fine_weight: float = 0.5
    dual_scale_independent_timesteps: bool = True
    use_guidance_channel: bool = True
    cfg_dropout: float = 0.1
    deterministic_anchor_loss: bool = False
    deterministic_anchor_lambda: float = 0.99
    deterministic_anchor_alpha: float = 0.5
    binary_anchor_input_mode: str = "stationary_flat"
    eval_sampler: str = "dpmpp"
    disable_cross_attention: bool = False
    cross_variate_context_bias: float = 0.0
    use_window_normalization: bool = True
    window_norm_center: str = "mean"
    window_norm_std_floor: float = 1e-8
    zero_guidance_forecast: bool = False

    # -- Sequence geometry --
    lookback_length: int = 96
    forecast_length: int = 96
    lookback_overlap: int = 8
    diffusion_lookback_cap: int = 0
    diffusion_chunk_horizon: int = 0
    itrans_lookback_length: Optional[int] = None
    itrans_d_model: int = 512
    itrans_d_ff: int = 512
    itrans_e_layers: int = 4
    itrans_n_heads: int = 8
    binary_noise_schedule: str = "linear"
    prediction_target: str = "epsilon"
    loss_weighting: str = "none"
    min_snr_gamma: float = 5.0
    use_coordinate_channel: bool = True
    window_stride: int = 1
    binary_num_steps: int = 1000
    binary_beta_start: float = 1e-5
    binary_beta_end: float = 0.5
    lr_scheduler_type: str = "none"
    lr_warmup_epochs: int = 0
    max_scale_tuning: bool = False
    max_scale_tuning_range: List[float] = field(default_factory=lambda: [2.5, 14.0])
    n_itrans_hp_trials: int = 10
    n_diffusion_hp_trials: int = 10
    n_finetune_hp_trials: int = 10

    # -- Paths --
    checkpoint_dir: str = "./results/ckpts"
    results_dir: str = "./results/datasets"
    synth_cache_dir: Optional[str] = None
    datasets_dir: str = "./datasets"

    # -- Variate selection --
    variate_indices: Optional[List[int]] = None
    subset_id: Optional[str] = None
    data_subset: Dict[str, Any] = field(default_factory=dict)
    data_subset_resolved: Dict[str, Any] = field(default_factory=dict)

    # -- Device --
    device: Optional[torch.device] = None

    # -- Wandb --
    wandb_enabled: bool = False
    wandb_project: str = "ts-sandbox-leaderboard"
    wandb_group: Optional[str] = None
    wandb_tags: Optional[List[str]] = None
    wandb_phase_run_ids: Dict[str, str] = field(default_factory=dict)

    # -- Resume / fresh --
    resume: bool = False
    fresh: bool = False

    # -- Mutable: populated by phases as they produce artifacts --
    itrans_pretrain_ckpt: Optional[str] = None
    diffusion_pretrain_ckpt: Optional[str] = None
    diffusion_coarse_pretrain_ckpt: Optional[str] = None
    diffusion_fine_pretrain_ckpt: Optional[str] = None
    diffusion_finer_pretrain_ckpt: Optional[str] = None
    itrans_finetune_ckpt: Optional[str] = None
    diffusion_finetune_ckpt: Optional[str] = None
    diffusion_coarse_finetune_ckpt: Optional[str] = None
    diffusion_fine_finetune_ckpt: Optional[str] = None
    diffusion_finer_finetune_ckpt: Optional[str] = None

    itrans_best_params: Optional[Dict[str, Any]] = None
    diffusion_best_params: Optional[Dict[str, Any]] = None
    finetune_best_params: Optional[Dict[str, Any]] = None
    coarse_finetune_best_params: Optional[Dict[str, Any]] = None
    fine_finetune_best_params: Optional[Dict[str, Any]] = None
    finer_finetune_best_params: Optional[Dict[str, Any]] = None

    # Phase-level overrides from YAML (list of dicts)
    phase_configs: List[Dict[str, Any]] = field(default_factory=list)

    # Full merged YAML (experiment + phases + training + visualization)
    merged_config: Optional[Dict[str, Any]] = None

    # Extra kwargs that don't map to a field (forwarded to phases)
    extra: Dict[str, Any] = field(default_factory=dict)

    def seed_everything(self) -> None:
        """Pin RNG state for reproducibility."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def resolve_device(self) -> torch.device:
        if self.device is not None:
            return self.device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self.device

    def ensure_dirs(self) -> None:
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "PipelineState":
        """Build state from a merged experiment config dict."""
        exp = dict(cfg["experiment"])
        if "experiment_name" not in exp and "name" in exp:
            exp["experiment_name"] = exp["name"]
        missing = [k for k in REQUIRED_EXPERIMENT_KEYS if k not in exp]
        if missing:
            raise ValueError(f"experiment missing required key(s): {missing}")

        known_fields = {f.name for f in cls.__dataclass_fields__.values()}

        init_kwargs: Dict[str, Any] = {}
        extra: Dict[str, Any] = {}
        for k, v in exp.items():
            if k in known_fields:
                init_kwargs[k] = v
            elif k != "name":
                extra[k] = v
        if "name" in exp and "experiment_name" not in init_kwargs:
            init_kwargs["experiment_name"] = exp["name"]

        if "dit_patch_size" in init_kwargs:
            init_kwargs["dit_patch_size"] = tuple(int(x) for x in init_kwargs["dit_patch_size"])
        for key in ("image_height", "coarse_image_height", "fine_image_height", "finer_image_height"):
            if key in init_kwargs:
                init_kwargs[key] = int(init_kwargs[key])
        for key in ("dit_embed_dim", "dit_depth", "dit_num_heads", "itrans_d_model", "itrans_d_ff", "itrans_e_layers", "itrans_n_heads"):
            if key in init_kwargs:
                init_kwargs[key] = int(init_kwargs[key])
        for key in ("dit_mlp_ratio", "dit_dropout"):
            if key in init_kwargs:
                init_kwargs[key] = float(init_kwargs[key])
        if "max_scale" in init_kwargs:
            init_kwargs["max_scale"] = float(init_kwargs["max_scale"])
        if "max_scale_by_dataset" in init_kwargs:
            init_kwargs["max_scale_by_dataset"] = {
                str(k): float(v) for k, v in init_kwargs["max_scale_by_dataset"].items()
            }
        if "window_norm_std_floor" in init_kwargs:
            init_kwargs["window_norm_std_floor"] = float(init_kwargs["window_norm_std_floor"])
        if "min_snr_gamma" in init_kwargs:
            init_kwargs["min_snr_gamma"] = float(init_kwargs["min_snr_gamma"])
        if "use_coordinate_channel" in init_kwargs:
            init_kwargs["use_coordinate_channel"] = bool(init_kwargs["use_coordinate_channel"])

        training = cfg.get("training", {})
        if not isinstance(training, dict):
            training = {}
        apply_training_section_to_state(training, init_kwargs, extra)
        apply_wandb_section_to_state(cfg.get("wandb"), init_kwargs)

        init_kwargs["extra"] = extra
        init_kwargs["phase_configs"] = cfg.get("phases", [])
        init_kwargs["merged_config"] = cfg
        return cls(**init_kwargs)
