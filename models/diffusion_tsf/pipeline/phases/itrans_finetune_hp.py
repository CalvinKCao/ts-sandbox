"""Phase 2A: iTransformer HP finetune on real data.

Cold-starts by default (ignores pretrained weights) since synthetic
pretrain on RealTS tends to converge near a unit-variance mean predictor.
"""

from __future__ import annotations

import json
import logging
import os
import shutil

from models.diffusion_tsf.pipeline.phase import PipelinePhase
from models.diffusion_tsf.pipeline.state import PipelineState
from models.diffusion_tsf.pipeline.phases.itrans_hp_pretrain import _patch_globals
from models.diffusion_tsf.pipeline import wandb_utils

logger = logging.getLogger(__name__)


class ITransFinetuneHPPhase(PipelinePhase):
    name = "itrans_finetune_hp"

    def should_skip(self, state: PipelineState) -> bool:
        subset_id = state.subset_id or state.dataset
        ckpt = os.path.join(state.checkpoint_dir, f"{subset_id}_itransformer_finetuned.pt")
        if os.path.exists(ckpt):
            logger.info(f"  [{self.name}] cached: {ckpt}")
            state.itrans_finetune_ckpt = ckpt
            return True
        return False

    def execute(self, state: PipelineState) -> PipelineState:
        from models.diffusion_tsf.train_multivariate_pipeline import (
            run_itransformer_finetune_hp_tuning,
            generate_dataset_job,
        )
        import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod

        _patch_globals(pipeline_mod, state)

        subset_id = state.subset_id or state.dataset
        variate_indices = state.variate_indices
        if variate_indices is None:
            variate_indices = generate_dataset_job(state.dataset)["variate_indices"]

        n_trials = self.get("n_trials", 10)
        if state.smoke_test:
            n_trials = 1

        pretrained_ckpt = state.itrans_pretrain_ckpt or ""
        cold_start = self.get("cold_start", True)

        best_params, tune_ckpt_path = run_itransformer_finetune_hp_tuning(
            dataset_name=state.dataset,
            variate_indices=variate_indices,
            pretrained_ckpt="" if cold_start else pretrained_ckpt,
            n_trials=n_trials,
            device=state.resolve_device(),
            smoke_test=state.smoke_test,
            checkpoint_dir=state.checkpoint_dir,
            subset_id=subset_id,
        )

        # Promote to canonical finetuned name
        ft_ckpt = os.path.join(state.checkpoint_dir, f"{subset_id}_itransformer_finetuned.pt")
        hp_best = os.path.join(state.checkpoint_dir, f"{subset_id}_itrans_ft_hp_best.pt")
        if os.path.exists(hp_best) and not os.path.exists(ft_ckpt):
            shutil.copy2(hp_best, ft_ckpt)

        state.itrans_finetune_ckpt = ft_ckpt

        wandb_utils.log_summary({
            "hp/itrans_ft_best_lr": best_params.get("learning_rate", None),
        })

        return state
