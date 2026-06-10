"""Phase 1B: Diffusion HP tuning on synthetic data (with frozen iTransformer guidance).

Delegates to ``run_diffusion_hp_tuning``. Requires Phase 1A to have
populated ``state.itrans_pretrain_ckpt``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil

from models.diffusion_tsf.pipeline.phase import PipelinePhase
from models.diffusion_tsf.pipeline.state import PipelineState
from models.diffusion_tsf.pipeline.globals_bridge import patch_globals
from models.diffusion_tsf.pipeline import wandb_utils
from models.diffusion_tsf.pipeline.visualize_utils import run_pretrain_diffusion_visualizations

logger = logging.getLogger(__name__)


class DiffusionHPPretrainPhase(PipelinePhase):
    name = "diffusion_hp_pretrain"

    def should_skip(self, state: PipelineState) -> bool:
        ckpt = os.path.join(state.checkpoint_dir, "diffusion.pt")
        hp_json = os.path.join(state.checkpoint_dir, "diff_hp.json")
        if os.path.exists(ckpt) and os.path.exists(hp_json):
            logger.info(f"  [{self.name}] cached: {ckpt}")
            with open(hp_json) as f:
                state.diffusion_best_params = json.load(f)
            state.diffusion_pretrain_ckpt = ckpt
            return True
        return False

    def execute(self, state: PipelineState) -> PipelineState:
        from models.diffusion_tsf.train_multivariate_pipeline import (
            run_diffusion_hp_tuning,
        )
        import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod

        patch_globals(pipeline_mod, state, honor_dataset_windows=False)

        if not state.itrans_pretrain_ckpt or not os.path.exists(state.itrans_pretrain_ckpt):
            raise RuntimeError(
                f"Phase 1B requires iTransformer checkpoint from Phase 1A, "
                f"but got: {state.itrans_pretrain_ckpt}"
            )

        n_trials = int(self.require("n_trials"))
        if state.smoke_test:
            n_trials = 1

        best_params, tune_ckpt_path = run_diffusion_hp_tuning(
            itrans_checkpoint=state.itrans_pretrain_ckpt,
            n_trials=n_trials,
            smoke_test=state.smoke_test,
            checkpoint_dir=state.checkpoint_dir,
            parallel_workers=state.parallel_optuna_workers,
        )

        hp_json = os.path.join(state.checkpoint_dir, "diff_hp.json")
        with open(hp_json, "w") as f:
            json.dump(best_params, f, indent=2)

        # Promote to canonical name
        diff_ckpt = os.path.join(state.checkpoint_dir, "diffusion.pt")
        if tune_ckpt_path and os.path.exists(tune_ckpt_path) and not os.path.exists(diff_ckpt):
            shutil.copy2(tune_ckpt_path, diff_ckpt)

        state.diffusion_best_params = best_params
        state.diffusion_pretrain_ckpt = diff_ckpt

        wandb_utils.log_summary({
            "hp/best_val_loss": best_params.get("best_val_loss", None),
            "hp/best_lr": best_params.get("learning_rate", None),
        })

        try:
            viz_paths = run_pretrain_diffusion_visualizations(
                state,
                diff_ckpt_path=diff_ckpt,
                itrans_ckpt_path=state.itrans_pretrain_ckpt,
                tuned_params=best_params,
                tag="diffusion_synthetic_pretrain",
            )
            wandb_utils.log_visualization_paths(
                viz_paths, wandb_key="viz/diffusion_synthetic_pretrain",
            )
        except Exception as e:
            logger.warning("Synthetic-pretrain diffusion viz failed: %s", e, exc_info=True)

        return state
