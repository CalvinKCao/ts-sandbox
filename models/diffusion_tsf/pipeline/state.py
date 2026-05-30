"""Shared mutable state passed between pipeline phases.

Replaces the module-level globals in train_multivariate_pipeline.py with
an explicit dataclass. Phases read what they need and write back the
artifacts they produce (checkpoint paths, best HP params, etc.).
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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
    diffusion_type: str = "gaussian"
    model_type: str = "dit"
    prediction_mode: str = "epsilon"
    image_height: int = 32
    deterministic_anchor_loss: bool = False
    deterministic_anchor_lambda: float = 0.99
    deterministic_anchor_alpha: float = 0.5
    eval_sampler: str = "dpmpp"
    disable_cross_attention: bool = False
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
    itrans_finetune_ckpt: Optional[str] = None
    diffusion_finetune_ckpt: Optional[str] = None

    itrans_best_params: Optional[Dict[str, Any]] = None
    diffusion_best_params: Optional[Dict[str, Any]] = None
    finetune_best_params: Optional[Dict[str, Any]] = None

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

        init_kwargs["extra"] = extra
        init_kwargs["phase_configs"] = cfg.get("phases", [])
        return cls(**init_kwargs)
