"""Phase 1A: iTransformer HP tuning on synthetic data.

Delegates to the existing ``run_itransformer_hp_tuning`` function, then
writes the best checkpoint path and params into PipelineState.
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
from models.diffusion_tsf.pipeline.visualize_utils import run_itrans_checkpoint_visualizations

logger = logging.getLogger(__name__)


class ITransHPPretrainPhase(PipelinePhase):
    name = "itrans_hp_pretrain"

    def should_skip(self, state: PipelineState) -> bool:
        ckpt = os.path.join(state.checkpoint_dir, "itransformer.pt")
        hp_json = os.path.join(state.checkpoint_dir, "itrans_hp.json")
        if os.path.exists(ckpt) and os.path.exists(hp_json):
            logger.info(f"  [{self.name}] cached: {ckpt}")
            # Load cached params into state
            with open(hp_json) as f:
                state.itrans_best_params = json.load(f)
            state.itrans_pretrain_ckpt = ckpt
            return True
        return False

    def execute(self, state: PipelineState) -> PipelineState:
        # Lazy import to avoid circular deps / heavy module load at import time
        from models.diffusion_tsf.train_multivariate_pipeline import (
            run_itransformer_hp_tuning,
        )
        import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod

        # Temporarily set the module-level globals the old code reads
        patch_globals(pipeline_mod, state, honor_dataset_windows=False)

        n_trials = int(self.require("n_trials"))
        if state.smoke_test:
            n_trials = 1

        best_params, tune_ckpt_path = run_itransformer_hp_tuning(
            n_trials=n_trials,
            smoke_test=state.smoke_test,
            checkpoint_dir=state.checkpoint_dir,
            parallel_workers=state.parallel_optuna_workers,
        )

        # Save HP params
        hp_json = os.path.join(state.checkpoint_dir, "itrans_hp.json")
        with open(hp_json, "w") as f:
            json.dump(best_params, f, indent=2)

        # Promote best HP model to canonical name
        itrans_ckpt = os.path.join(state.checkpoint_dir, "itransformer.pt")
        if tune_ckpt_path and os.path.exists(tune_ckpt_path) and not os.path.exists(itrans_ckpt):
            shutil.copy2(tune_ckpt_path, itrans_ckpt)

        state.itrans_best_params = best_params
        state.itrans_pretrain_ckpt = itrans_ckpt

        wandb_utils.log_summary({
            "hp/best_val_loss": best_params.get("best_val_loss", None),
            "hp/best_lr": best_params.get("learning_rate", None),
        })

        try:
            viz_paths = run_itrans_checkpoint_visualizations(
                state, itrans_ckpt, tag="itrans_synthetic_pretrain",
            )
            wandb_utils.log_visualization_paths(
                viz_paths, wandb_key="viz/itrans_synthetic_pretrain",
            )
        except Exception as e:
            logger.warning("iTransformer synthetic-pretrain viz failed: %s", e, exc_info=True)

        return state
