"""Bridge PipelineState + merged YAML onto legacy module globals."""

from __future__ import annotations

import os
from typing import Any

from models.diffusion_tsf.pipeline.config import apply_training_config_to_module
from models.diffusion_tsf.pipeline.state import PipelineState


def patch_globals(
    mod: Any,
    state: PipelineState,
    *,
    honor_dataset_windows: bool = True,
) -> None:
    """Set module-level globals that legacy training code still reads."""
    mod.N_VARIATES = state.n_variates
    lookback = state.lookback_length
    forecast = state.forecast_length
    if honor_dataset_windows:
        from models.diffusion_tsf.train_multivariate_pipeline import dataset_window_lengths

        lookback, forecast = dataset_window_lengths(state.dataset)
    mod.LOOKBACK_LENGTH = lookback
    mod.FORECAST_LENGTH = forecast
    mod.ITRANSFORMER_SEQ_LEN = lookback
    mod.LOOKBACK_OVERLAP = state.lookback_overlap
    mod.ITRANS_D_MODEL = state.itrans_d_model
    mod.ITRANS_D_FF = state.itrans_d_ff
    mod.ITRANS_E_LAYERS = state.itrans_e_layers
    mod.ITRANS_N_HEADS = state.itrans_n_heads
    mod.BINARY_NOISE_SCHEDULE = state.binary_noise_schedule
    mod.PREDICTION_TARGET = state.prediction_target
    mod.LOSS_WEIGHTING = state.loss_weighting
    mod.MIN_SNR_GAMMA = state.min_snr_gamma
    mod.USE_COORDINATE_CHANNEL = state.use_coordinate_channel
    mod.IMAGE_HEIGHT = state.image_height
    mod.COARSE_IMAGE_HEIGHT = state.coarse_image_height
    mod.FINE_IMAGE_HEIGHT = state.fine_image_height
    mod.FINER_IMAGE_HEIGHT = state.finer_image_height
    mod.MAX_SCALE = float(state.max_scale_by_dataset.get(state.dataset, state.max_scale))
    mod.DIT_PATCH_SIZE = tuple(state.dit_patch_size)
    mod.DIT_EMBED_DIM = state.dit_embed_dim
    mod.DIT_DEPTH = state.dit_depth
    mod.DIT_NUM_HEADS = state.dit_num_heads
    mod.DIT_MLP_RATIO = state.dit_mlp_ratio
    mod.DIT_DROPOUT = state.dit_dropout
    mod.USE_DUAL_SCALE = state.use_dual_scale
    mod.USE_TRIPLE_SCALE = state.use_triple_scale
    mod.DIFFUSION_STAGE = state.diffusion_stage
    mod.DUAL_SCALE_FINE_WEIGHT = state.dual_scale_fine_weight
    mod.DUAL_SCALE_INDEPENDENT_TIMESTEPS = state.dual_scale_independent_timesteps
    mod.USE_GUIDANCE_CHANNEL = state.use_guidance_channel
    mod.CFG_DROPOUT = state.cfg_dropout
    mod.MODEL_TYPE = state.model_type
    mod.DIFFUSION_TYPE = state.diffusion_type
    mod.D3PM_TRANSITION_MAX = state.d3pm_transition_max
    mod.D3PM_TRANSITION_MIN = state.d3pm_transition_min
    mod.D3PM_NEIGHBOR_KERNEL = state.d3pm_neighbor_kernel
    mod.D3PM_NOISE_SCHEDULE = state.d3pm_noise_schedule
    mod.D3PM_LOSS_TYPE = state.d3pm_loss_type
    mod.DETERMINISTIC_ANCHOR_LOSS = state.deterministic_anchor_loss
    mod.DETERMINISTIC_ANCHOR_LAMBDA = state.deterministic_anchor_lambda
    mod.DETERMINISTIC_ANCHOR_ALPHA = state.deterministic_anchor_alpha
    mod.BINARY_ANCHOR_INPUT_MODE = state.binary_anchor_input_mode
    mod.EVAL_SAMPLER = state.eval_sampler
    mod.DISABLE_CROSS_ATTENTION = state.disable_cross_attention
    mod.CROSS_VARIATE_CONTEXT_BIAS = state.cross_variate_context_bias
    mod.USE_WINDOW_NORMALIZATION = state.use_window_normalization
    mod.WINDOW_NORM_STD_FLOOR = state.window_norm_std_floor
    mod.ZERO_GUIDANCE_FORECAST = state.zero_guidance_forecast
    mod.WINDOW_STRIDE = state.window_stride
    mod.BINARY_NUM_STEPS = state.binary_num_steps
    mod.BINARY_BETA_START = state.binary_beta_start
    mod.BINARY_BETA_END = state.binary_beta_end
    mod.LR_SCHEDULER_TYPE = getattr(state, "lr_scheduler_type", "none")
    mod.LR_WARMUP_EPOCHS = getattr(state, "lr_warmup_epochs", 0)
    mod.MAX_SCALE_TUNING = getattr(state, "max_scale_tuning", False)
    mod.MAX_SCALE_TUNING_RANGE = getattr(state, "max_scale_tuning_range", [2.5, 14.0])
    if state.checkpoint_dir:
        mod.CHECKPOINT_DIR = state.checkpoint_dir
    if state.results_dir:
        mod.RESULTS_DIR = state.results_dir
    if state.synth_cache_dir:
        mod.SYNTH_CACHE_DIR = state.synth_cache_dir
    mod.DATASETS_DIR = os.path.abspath(
        os.path.expanduser(state.datasets_dir or mod.DATASETS_DIR)
    )
    apply_training_config_to_module(mod, state.merged_config, state)
