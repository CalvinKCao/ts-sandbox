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
    use_ordinal_window_norm: bool = False
    ordinal_tie_atol: float = 1e-6
    model_type: str = "dit"
    image_height: int = 32
    coarse_image_height: int = 16
    fine_image_height: int = 16
    finer_image_height: int = 16
    max_scale: float = 3.5
    max_scale_by_dataset: Dict[str, float] = field(default_factory=dict)
    staged_representation: str = "value_precision"
    dit_patch_size: Tuple[int, int] = (8, 8)
    dit_embed_dim: int = 384
    dit_depth: int = 8
    dit_num_heads: int = 6
    dit_mlp_ratio: float = 4.0
    dit_dropout: float = 0.0
    use_triple_scale: bool = False
    use_vertical_dual_concat: bool = False
    use_channel_dual_concat: bool = False
    diffusion_stage: str = "joint"
    use_guidance_channel: bool = True
    guidance_placement: str = "canvas"
    guidance_type: str = "patch_decoder"
    mmpd_patch_size: int = 12
    cfg_dropout: float = 0.1
    deterministic_anchor_loss: bool = False
    deterministic_anchor_lambda: float = 0.99
    deterministic_anchor_alpha: float = 0.5
    binary_anchor_input_mode: str = "stationary_flat"
    binary_use_boundary_weighted_bce: bool = False
    binary_cdf_distance_alpha: float = 1.0
    binary_use_normalized_value_width_weighted_bce: bool = False
    anchor_mse_proxy_lambda: float = 0.5
    eval_sampler: str = "quad_t"
    disable_cross_attention: bool = False
    cross_variate_context_bias: float = 0.0
    use_window_normalization: bool = True
    window_norm_center: str = "mean"
    window_norm_std_floor: float = 1e-8
    window_norm_low_var_threshold: float = 0.0
    window_norm_low_var_unit_std: float = 1.0
    window_norm_low_var_unit_std_by_variate: Dict[str, List[float]] = field(default_factory=dict)
    window_norm_low_var_unit_std_by_dataset: Dict[str, float] = field(default_factory=dict)
    lookback_overlap_center_shift: bool = False
    zero_guidance_forecast: bool = False
    use_raw_lookback_cond_channel: bool = False

    # -- Sequence geometry --
    lookback_length: int = 96
    forecast_length: int = 96
    lookback_overlap: int = 8
    diffusion_lookback_cap: int = 0
    diffusion_chunk_horizon: int = 0
    representation_time_stride: int = 1
    past_cond_resize_to_horizon: bool = True
    itrans_lookback_length: Optional[int] = None
    itrans_d_model: int = 512
    itrans_d_ff: int = 512
    itrans_e_layers: int = 4
    itrans_n_heads: int = 8
    binary_noise_schedule: str = "linear"
    binary_length_mode: str = "none"
    binary_length_g: float = 1.0
    binary_length_scale: float = 1.0
    # Optional per-dataset override applied when dataset is known (see patch_globals).
    binary_length_g_by_dataset: Dict[str, float] = field(default_factory=dict)
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
    wandb_run_id: Optional[str] = None

    # -- Resume / fresh --
    resume: bool = False
    fresh: bool = False

    # -- Mutable: populated by phases as they produce artifacts --
    itrans_pretrain_ckpt: Optional[str] = None
    diffusion_pretrain_ckpt: Optional[str] = None
    diffusion_coarse_pretrain_ckpt: Optional[str] = None
    diffusion_fine_pretrain_ckpt: Optional[str] = None
    diffusion_finer_pretrain_ckpt: Optional[str] = None
    diffusion_vertical_dual_pretrain_ckpt: Optional[str] = None
    diffusion_channel_dual_pretrain_ckpt: Optional[str] = None
    itrans_finetune_ckpt: Optional[str] = None
    patch_guidance_finetune_ckpt: Optional[str] = None
    diffusion_finetune_ckpt: Optional[str] = None
    diffusion_coarse_finetune_ckpt: Optional[str] = None
    diffusion_fine_finetune_ckpt: Optional[str] = None
    diffusion_finer_finetune_ckpt: Optional[str] = None
    diffusion_vertical_dual_finetune_ckpt: Optional[str] = None
    diffusion_channel_dual_finetune_ckpt: Optional[str] = None

    itrans_best_params: Optional[Dict[str, Any]] = None
    diffusion_best_params: Optional[Dict[str, Any]] = None
    finetune_best_params: Optional[Dict[str, Any]] = None
    coarse_finetune_best_params: Optional[Dict[str, Any]] = None
    fine_finetune_best_params: Optional[Dict[str, Any]] = None
    finer_finetune_best_params: Optional[Dict[str, Any]] = None
    vertical_dual_finetune_best_params: Optional[Dict[str, Any]] = None
    channel_dual_finetune_best_params: Optional[Dict[str, Any]] = None

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

    @property
    def guidance_finetune_ckpt(self) -> Optional[str]:
        if self.guidance_type == "patch_decoder":
            return self.patch_guidance_finetune_ckpt
        return self.itrans_finetune_ckpt

    def default_guidance_finetune_ckpt_path(self) -> str:
        subset_id = self.subset_id or self.dataset
        if self.guidance_type == "patch_decoder":
            return os.path.join(self.checkpoint_dir, f"{subset_id}_patch_guidance.pt")
        return os.path.join(self.checkpoint_dir, f"{subset_id}_itransformer_finetuned.pt")

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
        if "binary_length_g_by_dataset" in init_kwargs:
            init_kwargs["binary_length_g_by_dataset"] = {
                str(k): float(v) for k, v in (init_kwargs["binary_length_g_by_dataset"] or {}).items()
            }
        if "window_norm_std_floor" in init_kwargs:
            init_kwargs["window_norm_std_floor"] = float(init_kwargs["window_norm_std_floor"])
        for key in ("window_norm_low_var_threshold", "window_norm_low_var_unit_std"):
            if key in init_kwargs:
                init_kwargs[key] = float(init_kwargs[key])
        if "window_norm_low_var_unit_std_by_variate" in init_kwargs:
            by_ds = init_kwargs["window_norm_low_var_unit_std_by_variate"] or {}
            init_kwargs["window_norm_low_var_unit_std_by_variate"] = {
                str(k): [float(v) for v in vals]
                for k, vals in by_ds.items()
            }
        if "window_norm_low_var_unit_std_by_dataset" in init_kwargs:
            by_ds = init_kwargs["window_norm_low_var_unit_std_by_dataset"] or {}
            init_kwargs["window_norm_low_var_unit_std_by_dataset"] = {
                str(k): float(v) for k, v in by_ds.items()
            }
        if "representation_time_stride" in init_kwargs:
            init_kwargs["representation_time_stride"] = int(init_kwargs["representation_time_stride"])
        if "past_cond_resize_to_horizon" in init_kwargs:
            init_kwargs["past_cond_resize_to_horizon"] = bool(init_kwargs["past_cond_resize_to_horizon"])
        if "min_snr_gamma" in init_kwargs:
            init_kwargs["min_snr_gamma"] = float(init_kwargs["min_snr_gamma"])
        if "use_coordinate_channel" in init_kwargs:
            init_kwargs["use_coordinate_channel"] = bool(init_kwargs["use_coordinate_channel"])
        if "use_raw_lookback_cond_channel" in init_kwargs:
            init_kwargs["use_raw_lookback_cond_channel"] = bool(init_kwargs["use_raw_lookback_cond_channel"])

        training = cfg.get("training", {})
        if not isinstance(training, dict):
            training = {}
        apply_training_section_to_state(training, init_kwargs, extra)
        apply_wandb_section_to_state(cfg.get("wandb"), init_kwargs)

        init_kwargs["extra"] = extra
        init_kwargs["phase_configs"] = cfg.get("phases", [])
        init_kwargs["merged_config"] = cfg
        return cls(**init_kwargs)
