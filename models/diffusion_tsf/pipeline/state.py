"""Shared mutable state passed between pipeline phases.

Replaces the module-level globals in train_multivariate_pipeline.py with
an explicit dataclass. Phases read what they need and write back the
artifacts they produce (checkpoint paths, best HP params, etc.).
"""

from __future__ import annotations

import math
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
    eval_bench: bool = False
    eval_max_windows: Optional[int] = None
    eval_max_steps: Optional[int] = None
    parallel_optuna_workers: int = 1

    # -- Model / diffusion knobs --
    diffusion_type: str = "binary"
    use_ordinal_window_norm: bool = False
    ordinal_ood_shift_causal_only: bool = False
    ordinal_tie_atol: float = 1e-6
    # Derived from the training split when ordinal normalization is enabled.
    ordinal_ladder: Optional[Any] = None
    model_type: str = "dit"
    image_height: int = 16
    coarse_image_height: int = 16
    fine_image_height: int = 16
    max_scale: float = 3.5
    max_scale_by_dataset: Dict[str, float] = field(default_factory=dict)
    staged_representation: str = "value_precision"
    dit_patch_size: Tuple[int, int] = (8, 8)
    dit_embed_dim: int = 384
    dit_depth: int = 8
    dit_num_heads: int = 6
    dit_mlp_ratio: float = 4.0
    dit_dropout: float = 0.0
    use_patch_refine_stage: bool = False
    diffusion_stage: str = "coarse"
    patch_refine_canvas_height: int = 256
    patch_refine_patch_height: int = 32
    patch_refine_patch_width: int = 8
    patch_refine_col_stride: int = 6
    patch_refine_unique_segments: bool = False
    # Fraction of train windows used only during patch_refine finetune (1.0 = all).
    patch_refine_finetune_window_fraction: float = 1.0
    # Train-only fraction of unique-seg (B,V) crops kept inside each window.
    # Independent of window_fraction. 1.0 = all variate-crops.
    patch_refine_finetune_patch_fraction: float = 1.0
    dit_cond_patch_size: Tuple[int, int] = (8, 8)
    guidance_type: str = "itransformer"
    mmpd_patch_size: int = 12
    cfg_dropout: float = 0.1
    deterministic_anchor_loss: bool = False
    deterministic_anchor_lambda: float = 0.99
    deterministic_anchor_alpha: float = 0.5
    # Apply the expensive deterministic-anchor forward every N train batches.
    # One preserves the standard combined objective on every batch.
    deterministic_anchor_every_n_batches: int = 1
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
    # ETTh2-style hybrid: flat vars → dataset-level affine only (no window norm).
    hybrid_flat_dataset_norm: bool = False
    hybrid_flat_frac_threshold: float = 0.5
    hybrid_flat_oob_coverage: float = 0.99
    # Derived with the dataset-affine hybrid normalization and copied into
    # DiffusionTSFConfig when a model is built.
    skip_window_norm_variate_mask: Optional[List[bool]] = None
    lookback_overlap_center_shift: bool = False
    use_raw_lookback_cond_channel: bool = False

    # -- Sequence geometry --
    lookback_length: int = 96
    forecast_length: int = 96
    lookback_overlap: int = 8
    diffusion_lookback_cap: int = 0
    horizon_stitch: bool = False
    horizon_chunk_inner: int = 96
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
    # Optional per-dataset override applied when constructing a model.
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
    # 1 = each train epoch walks the full packed batch list. N>1 splits that
    # list into N groups; epoch e uses group e % N (full cycle still covers all).
    # After each cycle, indices are reshuffled and batches are repacked.
    train_epoch_groups: int = 1
    # If set, N is computed after the batch-size probe so each packed group is
    # <= this many bytes (fp32 past+future). Wins over train_epoch_groups > 1.
    train_epoch_max_bytes: Optional[int] = None
    max_scale_tuning: bool = False
    max_scale_tuning_range: List[float] = field(default_factory=lambda: [2.5, 14.0])
    n_itrans_hp_trials: int = 10
    n_diffusion_hp_trials: int = 10
    n_finetune_hp_trials: int = 10

    # -- Training knobs --
    pretrain_epochs: int = 10
    pretrain_diffusion_epochs: int = 20
    pretrain_diffusion_max_epochs: int = 20
    pretrain_synthetic_override: Optional[int] = None
    synthetic_samples_full_cap: int = 50_000
    synthetic_samples_hp_tune: int = 20_000
    synthetic_samples_diff_tune: int = 10_000
    synthetic_samples_min: int = 4_096
    itrans_hp_pretrain_max_epochs: int = 10
    itrans_hp_finetune_max_epochs: int = 10
    diffusion_hp_patience: int = 4
    hp_tune_epochs: int = 20
    hp_tune_patience: int = 15
    itrans_real_cold_start: bool = True
    itrans_paper_batch_size: int = 32
    itrans_paper_lr_grid: List[float] = field(
        default_factory=lambda: [1e-3, 5e-4, 1e-4]
    )
    itrans_paper_dropout: float = 0.1
    diffusion_batch_size: int = 32
    diffusion_batch_sizes: List[int] = field(default_factory=lambda: [16])
    finetune_batch_sizes: List[int] = field(default_factory=lambda: [4, 8, 16])
    finetune_max_micro_batch: Optional[int] = None
    finetune_hp_lr_min: float = 3e-6
    finetune_hp_lr_max: float = 2e-4
    use_amp: bool = True
    use_gradient_checkpointing: bool = True
    # Compile FactorizedDiT (eager geometry/validation, compiled denoiser).
    # Smoke tests skip this even when true. Requires CUDA.
    torch_compile: bool = False
    unet_max_chunk_size: int = 128
    sequential_anchor_backward: bool = False
    eval_num_samples: int = 30
    past_loss_weight: float = 0.3
    train_window_aug: Dict[str, Any] = field(default_factory=dict)

    # -- Paths --
    checkpoint_dir: str = "./results/ckpts"
    results_dir: str = "./results/datasets"
    synth_cache_dir: Optional[str] = None
    datasets_dir: str = "./datasets"

    # -- Variate selection --
    variate_indices: Optional[List[int]] = None
    subset_id: Optional[str] = None
    data_subset_by_dataset: Dict[str, Any] = field(default_factory=dict)
    data_subset_resolved: Dict[str, Any] = field(default_factory=dict)
    dataset_shape_cache: Dict[Tuple[str, str], Tuple[int, int]] = field(
        default_factory=dict
    )

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

    # -- Refactored YAML options --
    seeds: List[int] = field(default_factory=lambda: [42])
    ckpt_config: Optional[str] = None
    walltime: str = "3:00:00"

    # -- Mutable: populated by phases as they produce artifacts --
    itrans_pretrain_ckpt: Optional[str] = None
    diffusion_pretrain_ckpt: Optional[str] = None
    diffusion_coarse_pretrain_ckpt: Optional[str] = None
    diffusion_fine_pretrain_ckpt: Optional[str] = None
    diffusion_patch_refine_pretrain_ckpt: Optional[str] = None
    itrans_finetune_ckpt: Optional[str] = None
    diffusion_finetune_ckpt: Optional[str] = None
    diffusion_coarse_finetune_ckpt: Optional[str] = None
    diffusion_fine_finetune_ckpt: Optional[str] = None
    diffusion_patch_refine_finetune_ckpt: Optional[str] = None

    itrans_best_params: Optional[Dict[str, Any]] = None
    diffusion_best_params: Optional[Dict[str, Any]] = None
    finetune_best_params: Optional[Dict[str, Any]] = None
    coarse_finetune_best_params: Optional[Dict[str, Any]] = None
    fine_finetune_best_params: Optional[Dict[str, Any]] = None
    patch_refine_finetune_best_params: Optional[Dict[str, Any]] = None

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
        return self.itrans_finetune_ckpt

    @property
    def needs_guidance(self) -> bool:
        """True when bottleneck cross-attention needs a frozen encoder checkpoint."""
        return not bool(self.disable_cross_attention)

    def default_guidance_finetune_ckpt_path(self) -> str:
        subset_id = self.subset_id or self.dataset
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
        if "dit_cond_patch_size" in init_kwargs and init_kwargs["dit_cond_patch_size"] is not None:
            init_kwargs["dit_cond_patch_size"] = tuple(
                int(x) for x in init_kwargs["dit_cond_patch_size"]
            )
        if "horizon_stitch" in init_kwargs:
            from models.diffusion_tsf.config import parse_horizon_stitch
            init_kwargs["horizon_stitch"] = parse_horizon_stitch(init_kwargs["horizon_stitch"])
        for key in (
            "image_height",
            "coarse_image_height",
            "fine_image_height",
            "patch_refine_canvas_height",
            "patch_refine_patch_height",
            "patch_refine_patch_width",
            "patch_refine_col_stride",
        ):
            if key in init_kwargs:
                init_kwargs[key] = int(init_kwargs[key])
        if "patch_refine_unique_segments" in init_kwargs:
            init_kwargs["patch_refine_unique_segments"] = bool(
                init_kwargs["patch_refine_unique_segments"]
            )
        if "patch_refine_finetune_window_fraction" in init_kwargs:
            init_kwargs["patch_refine_finetune_window_fraction"] = float(
                init_kwargs["patch_refine_finetune_window_fraction"]
            )
        if "patch_refine_finetune_patch_fraction" in init_kwargs:
            frac = float(init_kwargs["patch_refine_finetune_patch_fraction"])
            if not math.isfinite(frac) or frac <= 0.0 or frac > 1.0:
                raise ValueError(
                    f"patch_refine_finetune_patch_fraction must be in (0, 1], got {frac!r}"
                )
            init_kwargs["patch_refine_finetune_patch_fraction"] = frac
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
        if "deterministic_anchor_every_n_batches" in init_kwargs:
            interval = int(init_kwargs["deterministic_anchor_every_n_batches"])
            if interval < 1:
                raise ValueError("deterministic_anchor_every_n_batches must be >= 1")
            init_kwargs["deterministic_anchor_every_n_batches"] = interval
        if "use_coordinate_channel" in init_kwargs:
            init_kwargs["use_coordinate_channel"] = bool(init_kwargs["use_coordinate_channel"])
        if "use_raw_lookback_cond_channel" in init_kwargs:
            init_kwargs["use_raw_lookback_cond_channel"] = bool(init_kwargs["use_raw_lookback_cond_channel"])

        training = cfg.get("training", {})
        if not isinstance(training, dict):
            training = {}
        apply_training_section_to_state(training, init_kwargs, extra)
        apply_wandb_section_to_state(cfg.get("wandb"), init_kwargs)

        if cfg.get("_yaml_path"):
            extra["config_path"] = cfg["_yaml_path"]
        init_kwargs["extra"] = extra
        init_kwargs["phase_configs"] = cfg.get("phases", [])
        init_kwargs["merged_config"] = cfg
        return cls(**init_kwargs)
