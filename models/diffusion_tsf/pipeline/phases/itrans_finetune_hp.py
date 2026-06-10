"""Phase 2A: iTransformer HP finetune on real data.

Cold-starts by default (ignores pretrained weights) since synthetic
pretrain on RealTS tends to converge near a unit-variance mean predictor.
"""

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
from models.diffusion_tsf.pipeline.visualize_utils import run_itrans_checkpoint_visualizations

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

        patch_globals(pipeline_mod, state)

        subset_id = state.subset_id or state.dataset
        ft_ckpt = os.path.join(state.checkpoint_dir, f"{subset_id}_itransformer_finetuned.pt")

        reuse_from = self.get("reuse_checkpoint_from_config")
        if reuse_from:
            source_dir = discover_dataset_run_ckpt_dir(state, str(reuse_from))
            src_ckpt = os.path.join(source_dir, f"{subset_id}_itransformer_finetuned.pt")
            if not os.path.exists(src_ckpt):
                raise FileNotFoundError(
                    f"Missing iTransformer finetune to reuse: {src_ckpt} "
                    f"(from *-{state.dataset}-{reuse_from})"
                )
            if not os.path.exists(ft_ckpt):
                shutil.copy2(src_ckpt, ft_ckpt)
            state.itrans_finetune_ckpt = ft_ckpt
            logger.info("  [%s] reused finetuned iTransformer from %s", self.name, source_dir)
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

        pretrained_ckpt = state.itrans_pretrain_ckpt or ""
        cold_start = bool(self.require("cold_start"))

        best_params, tune_ckpt_path = run_itransformer_finetune_hp_tuning(
            dataset_name=state.dataset,
            variate_indices=variate_indices,
            pretrained_ckpt="" if cold_start else pretrained_ckpt,
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
        if os.path.exists(hp_best) and not os.path.exists(ft_ckpt):
            shutil.copy2(hp_best, ft_ckpt)

        state.itrans_finetune_ckpt = ft_ckpt

        wandb_utils.log_summary({
            "hp/itrans_ft_best_lr": best_params.get("learning_rate", None),
        })

        try:
            viz_paths = run_itrans_checkpoint_visualizations(
                state, ft_ckpt, tag="itrans_finetuned",
            )
            wandb_utils.log_visualization_paths(
                viz_paths, wandb_key="viz/itrans_finetuned",
            )
        except Exception as e:
            logger.warning("Finetuned iTransformer viz failed: %s", e, exc_info=True)

        return state
