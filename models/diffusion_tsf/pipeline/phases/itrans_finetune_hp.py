"""Phase: iTransformer encoder finetune for DiT cross-attn tokens (not 2D ghost)."""

from __future__ import annotations

import logging
import os
import shutil

from models.diffusion_tsf.pipeline.phase import PipelinePhase
from models.diffusion_tsf.pipeline.state import PipelineState
from models.diffusion_tsf.pipeline import wandb_utils

logger = logging.getLogger(__name__)


class ITransFinetuneHPPhase(PipelinePhase):
    name = "itrans_finetune_hp"

    def should_skip(self, state: PipelineState) -> bool:
        if state.guidance_type != "itransformer":
            logger.info("  [%s] skipping: guidance_type=%s", self.name, state.guidance_type)
            return True
        ckpt = state.default_guidance_finetune_ckpt_path()
        if os.path.exists(ckpt):
            state.itrans_finetune_ckpt = ckpt
            logger.info("  [%s] cached: %s", self.name, ckpt)
            return True
        return False

    def execute(self, state: PipelineState) -> PipelineState:
        from models.diffusion_tsf.train_multivariate_pipeline import (
            run_itransformer_finetune_hp_tuning,
        )

        subset_id = state.subset_id or state.dataset
        ft_ckpt = state.default_guidance_finetune_ckpt_path()
        variate_indices = state.variate_indices
        if not variate_indices:
            raise ValueError(
                f"[{self.name}] Missing resolved variate_indices in state for dataset {state.dataset!r}."
            )
        subset_meta = state.data_subset_resolved or {}
        train_stride = int(subset_meta.get("train_stride", state.window_stride))
        test_stride = int(subset_meta.get("test_stride", 1))
        n_trials = int(self.require("n_trials"))
        if state.smoke_test:
            n_trials = 1

        pretrained = ""
        best_params, tune_ckpt_path = run_itransformer_finetune_hp_tuning(
            state,
            dataset_name=state.dataset,
            variate_indices=variate_indices,
            pretrained_ckpt=pretrained,
            n_trials=n_trials,
            device=state.resolve_device(),
            smoke_test=state.smoke_test,
            checkpoint_dir=state.checkpoint_dir,
            subset_id=subset_id,
            train_stride=train_stride,
            test_stride=test_stride,
            parallel_workers=state.parallel_optuna_workers,
        )
        hp_best = os.path.join(state.checkpoint_dir, f"{subset_id}_itrans_ft_hp_best.pt")
        src = tune_ckpt_path if tune_ckpt_path and os.path.exists(tune_ckpt_path) else hp_best
        if os.path.exists(src) and src != ft_ckpt:
            os.makedirs(os.path.dirname(ft_ckpt), exist_ok=True)
            shutil.copy2(src, ft_ckpt)
        if not os.path.exists(ft_ckpt):
            raise FileNotFoundError(f"{self.name} missing iTransformer finetune ckpt at {ft_ckpt}")
        state.itrans_finetune_ckpt = ft_ckpt
        wandb_utils.log_summary({
            "hp/itrans_ft_best_lr": best_params.get("learning_rate"),
        })
        logger.info("  [%s] finetuned iTransformer tokens → %s", self.name, ft_ckpt)
        return state
