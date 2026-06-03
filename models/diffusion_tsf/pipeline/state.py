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


@dataclass
class PipelineState:
    """Everything that flows between phases."""

    # -- Experiment identity (frozen after init) --
    experiment_name: str = "experiment"
    dataset: str = "ETTh1"
    n_variates: int = 7
    seed: int = 42
    smoke_test: bool = False

    # -- Model / diffusion knobs --
    diffusion_type: str = "binary"
    model_type: str = "dit"
    image_height: int = 32
    max_scale: float = 3.5
    max_scale_by_dataset: Dict[str, float] = field(default_factory=dict)
    dit_patch_size: Tuple[int, int] = (8, 8)
    use_dual_scale: bool = False
    diffusion_stage: str = "joint"
    dual_scale_fine_weight: float = 0.5
    dual_scale_independent_timesteps: bool = True
    use_guidance_channel: bool = True
    cfg_dropout: float = 0.1
    cfg_scale: float = 2.0
    use_cfg_inference: bool = False
    deterministic_anchor_loss: bool = False
    deterministic_anchor_lambda: float = 0.99
    deterministic_anchor_alpha: float = 0.5
    eval_sampler: str = "dpmpp"
    disable_cross_attention: bool = False
    cross_variate_context_bias: float = 0.0
    use_window_normalization: bool = True
    zero_guidance_forecast: bool = False

    # -- Sequence geometry --
    lookback_length: int = 96
    forecast_length: int = 96
    lookback_overlap: int = 8
    window_stride: int = 1

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
    wandb_project: str = "diffusion-tsf"
    wandb_group: Optional[str] = None

    # -- Resume / fresh --
    resume: bool = False
    fresh: bool = False

    # -- Mutable: populated by phases as they produce artifacts --
    itrans_pretrain_ckpt: Optional[str] = None
    diffusion_pretrain_ckpt: Optional[str] = None
    diffusion_coarse_pretrain_ckpt: Optional[str] = None
    diffusion_fine_pretrain_ckpt: Optional[str] = None
    itrans_finetune_ckpt: Optional[str] = None
    diffusion_finetune_ckpt: Optional[str] = None
    diffusion_coarse_finetune_ckpt: Optional[str] = None
    diffusion_fine_finetune_ckpt: Optional[str] = None

    itrans_best_params: Optional[Dict[str, Any]] = None
    diffusion_best_params: Optional[Dict[str, Any]] = None
    finetune_best_params: Optional[Dict[str, Any]] = None
    coarse_finetune_best_params: Optional[Dict[str, Any]] = None
    fine_finetune_best_params: Optional[Dict[str, Any]] = None

    # Phase-level overrides from YAML (list of dicts)
    phase_configs: List[Dict[str, Any]] = field(default_factory=list)

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
        exp = cfg.get("experiment", {})
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}

        init_kwargs: Dict[str, Any] = {}
        extra: Dict[str, Any] = {}
        for k, v in exp.items():
            if k in known_fields:
                init_kwargs[k] = v
            else:
                extra[k] = v

        if "dit_patch_size" in init_kwargs:
            init_kwargs["dit_patch_size"] = tuple(int(x) for x in init_kwargs["dit_patch_size"])
        if "max_scale" in init_kwargs:
            init_kwargs["max_scale"] = float(init_kwargs["max_scale"])
        if "max_scale_by_dataset" in init_kwargs:
            init_kwargs["max_scale_by_dataset"] = {
                str(k): float(v) for k, v in init_kwargs["max_scale_by_dataset"].items()
            }

        init_kwargs["extra"] = extra
        init_kwargs["phase_configs"] = cfg.get("phases", [])
        return cls(**init_kwargs)
