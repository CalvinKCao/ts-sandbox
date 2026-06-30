"""Phase: patch decoder + mixer HP finetune on real data (window-norm MSE)."""

from __future__ import annotations

import logging
import os
import shutil

from models.diffusion_tsf.pipeline.phase import PipelinePhase
from models.diffusion_tsf.pipeline.state import PipelineState
from models.diffusion_tsf.pipeline.globals_bridge import patch_globals
from models.diffusion_tsf.pipeline.phases.staged_diffusion_pretrain import (
    discover_dataset_run_ckpt_dir,
)
from models.diffusion_tsf.pipeline import wandb_utils

logger = logging.getLogger(__name__)


class PatchGuidanceFinetuneHPPhase(PipelinePhase):
    name = "patch_guidance_finetune_hp"

    def _try_reuse_patch_guidance_ckpt(
        self,
        state: PipelineState,
        *,
        fail_if_missing: bool,
    ) -> bool:
        subset_id = state.subset_id or state.dataset
        ft_ckpt = state.default_guidance_finetune_ckpt_path()
        if os.path.exists(ft_ckpt):
            state.patch_guidance_finetune_ckpt = ft_ckpt
            return True

        reuse_from = self.get("reuse_checkpoint_from_config")
        if not reuse_from:
            return False

        try:
            source_dir = discover_dataset_run_ckpt_dir(state, str(reuse_from))
        except FileNotFoundError:
            if fail_if_missing:
                raise
            return False

        src_ckpt = os.path.join(source_dir, f"{subset_id}_patch_guidance.pt")
        if not os.path.exists(src_ckpt):
            if fail_if_missing:
                raise FileNotFoundError(
                    f"Missing patch guidance finetune to reuse: {src_ckpt} "
                    f"(from *-{state.dataset}-{reuse_from})"
                )
            return False

        os.makedirs(os.path.dirname(ft_ckpt), exist_ok=True)
        shutil.copy2(src_ckpt, ft_ckpt)
        state.patch_guidance_finetune_ckpt = ft_ckpt
        logger.info("  [%s] reused finetuned patch guidance from %s", self.name, source_dir)
        return True

    def should_skip(self, state: PipelineState) -> bool:
        if state.guidance_type != "patch_decoder":
            logger.info("  [%s] skipping: guidance_type=%s", self.name, state.guidance_type)
            return True
        if self._try_reuse_patch_guidance_ckpt(state, fail_if_missing=False):
            ckpt = state.default_guidance_finetune_ckpt_path()
            logger.info("  [%s] cached: %s", self.name, ckpt)
            return True
        return False

    def execute(self, state: PipelineState) -> PipelineState:
        from models.diffusion_tsf.train_multivariate_pipeline import (
            run_patch_guidance_finetune_hp_tuning,
            generate_dataset_job,
        )
        import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod

        patch_globals(pipeline_mod, state)

        subset_id = state.subset_id or state.dataset
        ft_ckpt = state.default_guidance_finetune_ckpt_path()

        if self.get("reuse_checkpoint_from_config"):
            self._try_reuse_patch_guidance_ckpt(state, fail_if_missing=True)
            return state

        variate_indices = state.variate_indices
        if variate_indices is None:
            variate_indices = generate_dataset_job(state.dataset)["variate_indices"]
        subset_meta = state.data_subset_resolved or {}
        train_stride = int(subset_meta.get("train_stride", state.window_stride))
        test_stride = int(subset_meta.get("test_stride", 1))

        n_trials = int(self.require("n_trials"))
        if state.smoke_test:
            n_trials = 1

        best_params, tune_ckpt_path = run_patch_guidance_finetune_hp_tuning(
            dataset_name=state.dataset,
            variate_indices=variate_indices,
            n_trials=n_trials,
            device=state.resolve_device(),
            smoke_test=state.smoke_test,
            checkpoint_dir=state.checkpoint_dir,
            subset_id=subset_id,
            train_stride=train_stride,
            test_stride=test_stride,
            parallel_workers=state.parallel_optuna_workers,
        )

        hp_best = os.path.join(state.checkpoint_dir, f"{subset_id}_patch_guidance_hp_best.pt")
        if os.path.exists(hp_best) and not os.path.exists(ft_ckpt):
            shutil.copy2(hp_best, ft_ckpt)

        state.patch_guidance_finetune_ckpt = ft_ckpt

        wandb_utils.log_summary({
            "hp/patch_guidance_ft_best_lr": best_params.get("learning_rate"),
        })
        if tune_ckpt_path:
            logger.info("  [%s] best HP checkpoint: %s", self.name, tune_ckpt_path)

        return state
