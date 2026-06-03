"""Phase 1A: iTransformer HP tuning on synthetic data.

Delegates to the existing ``run_itransformer_hp_tuning`` function, then
writes the best checkpoint path and params into PipelineState.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from typing import Any

from models.diffusion_tsf.pipeline.phase import PipelinePhase
from models.diffusion_tsf.pipeline.state import PipelineState
from models.diffusion_tsf.pipeline import wandb_utils

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
            N_VARIATES,
        )
        import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod

        # Temporarily set the module-level globals the old code reads
        _patch_globals(pipeline_mod, state, honor_dataset_windows=False)

        n_trials = self.get("n_trials", 10)
        if state.smoke_test:
            n_trials = 1

        best_params, tune_ckpt_path = run_itransformer_hp_tuning(
            n_trials=n_trials,
            smoke_test=state.smoke_test,
            checkpoint_dir=state.checkpoint_dir,
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

        return state


def _patch_globals(
    mod: Any,
    state: PipelineState,
    *,
    honor_dataset_windows: bool = True,
) -> None:
    """Temporarily set module-level globals that old training code reads.

    This is the bridge between the new PipelineState and the existing
    global-based code. Will shrink as we migrate more code.

    Pretrain phases pass ``honor_dataset_windows=False`` so synthetic pretrain
    keeps YAML lengths; finetune/eval use per-dataset windows (e.g. DALIA 80/20).
    """
    mod.N_VARIATES = state.n_variates
    lookback = state.lookback_length
    forecast = state.forecast_length
    if honor_dataset_windows:
        from models.diffusion_tsf.train_multivariate_pipeline import dataset_window_lengths
        lookback, forecast = dataset_window_lengths(state.dataset)
    mod.LOOKBACK_LENGTH = lookback
    mod.FORECAST_LENGTH = forecast
    mod.ITRANSFORMER_SEQ_LEN = lookback
    mod.IMAGE_HEIGHT = state.image_height
    mod.MAX_SCALE = float(state.max_scale_by_dataset.get(state.dataset, state.max_scale))
    mod.DIT_PATCH_SIZE = tuple(state.dit_patch_size)
    mod.USE_DUAL_SCALE = state.use_dual_scale
    mod.DIFFUSION_STAGE = state.diffusion_stage
    mod.DUAL_SCALE_FINE_WEIGHT = state.dual_scale_fine_weight
    mod.DUAL_SCALE_INDEPENDENT_TIMESTEPS = state.dual_scale_independent_timesteps
    mod.USE_GUIDANCE_CHANNEL = state.use_guidance_channel
    mod.CFG_DROPOUT = state.cfg_dropout
    mod.CFG_SCALE = state.cfg_scale
    mod.USE_CFG_INFERENCE = state.use_cfg_inference
    mod.MODEL_TYPE = state.model_type
    mod.DIFFUSION_TYPE = state.diffusion_type
    mod.DETERMINISTIC_ANCHOR_LOSS = state.deterministic_anchor_loss
    mod.DETERMINISTIC_ANCHOR_LAMBDA = state.deterministic_anchor_lambda
    mod.DETERMINISTIC_ANCHOR_ALPHA = state.deterministic_anchor_alpha
    mod.EVAL_SAMPLER = state.eval_sampler
    mod.DISABLE_CROSS_ATTENTION = state.disable_cross_attention
    mod.CROSS_VARIATE_CONTEXT_BIAS = state.cross_variate_context_bias
    mod.USE_WINDOW_NORMALIZATION = state.use_window_normalization
    mod.WINDOW_NORM_STD_FLOOR = state.window_norm_std_floor
    mod.ZERO_GUIDANCE_FORECAST = state.zero_guidance_forecast
    mod.WINDOW_STRIDE = state.window_stride
    if state.checkpoint_dir:
        mod.CHECKPOINT_DIR = state.checkpoint_dir
    if state.results_dir:
        mod.RESULTS_DIR = state.results_dir
    if state.synth_cache_dir:
        mod.SYNTH_CACHE_DIR = state.synth_cache_dir
    mod.DATASETS_DIR = os.path.abspath(
        os.path.expanduser(state.datasets_dir or mod.DATASETS_DIR)
    )
