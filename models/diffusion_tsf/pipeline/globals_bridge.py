"""Bridge PipelineState + merged YAML onto legacy module globals."""

from __future__ import annotations

import os
from typing import Any

from models.diffusion_tsf.pipeline.config import (
    apply_training_config_to_module,
    training_value,
)
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
    if honor_dataset_windows and state.dataset == "dalia":
        from models.diffusion_tsf.train_multivariate_pipeline import dalia_window_lengths

        lookback, forecast = dalia_window_lengths()
    mod.LOOKBACK_LENGTH = lookback
    mod.FORECAST_LENGTH = forecast
    itrans_lb = state.itrans_lookback_length
    mod.ITRANSFORMER_SEQ_LEN = int(itrans_lb if itrans_lb is not None else lookback)
    mod.ITRANS_LOOKBACK_LENGTH = itrans_lb
    mod.LOOKBACK_OVERLAP = state.lookback_overlap
    mod.DIFFUSION_LOOKBACK_CAP = int(state.diffusion_lookback_cap)
    mod.DIFFUSION_CHUNK_HORIZON = int(state.diffusion_chunk_horizon)
    mod.REPRESENTATION_TIME_STRIDE = int(state.representation_time_stride)
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
    mod.STAGED_REPRESENTATION = state.staged_representation
    mod.HAAR_HIGH_FREQ_PERCENT = float(state.haar_high_freq_percent)
    mod.HAAR_HIGH_FREQ_LEVELS = int(state.haar_high_freq_levels)
    mod.HAAR_FINE_MAX_SCALE = float(state.haar_fine_max_scale)
    mod.FOURIER_HIGH_FREQ_PERCENT = float(state.fourier_high_freq_percent)
    mod.FOURIER_HIGH_FREQ_CUTOFF_BIN = int(state.fourier_high_freq_cutoff_bin)
    mod.FOURIER_FINE_MAX_SCALE = float(state.fourier_fine_max_scale)
    mod.FOURIER_FLATLINE_ATOL = float(state.fourier_flatline_atol)
    mod.FOURIER_HIGH_FREQ_CUTOFF_BINS_PER_VARIATE = (
        list(state.fourier_high_freq_cutoff_bins_per_variate)
        if state.fourier_high_freq_cutoff_bins_per_variate
        else None
    )
    mod.FOURIER_FINE_MAX_SCALE_PER_VARIATE = (
        list(state.fourier_fine_max_scale_per_variate)
        if state.fourier_fine_max_scale_per_variate
        else None
    )
    mod.COARSE_FLATLINE_BLUR_FINE_TARGET = bool(state.coarse_flatline_blur_fine_target)
    mod.COARSE_FLATLINE_BLUR_RADIUS = int(state.coarse_flatline_blur_radius)
    mod.COARSE_FLATLINE_BLUR_KERNEL = str(state.coarse_flatline_blur_kernel)
    mod.COARSE_FLATLINE_BLUR_ATOL = state.coarse_flatline_blur_atol
    mod.DIT_PATCH_SIZE = tuple(state.dit_patch_size)
    mod.DIT_EMBED_DIM = state.dit_embed_dim
    mod.DIT_DEPTH = state.dit_depth
    mod.DIT_NUM_HEADS = state.dit_num_heads
    mod.DIT_MLP_RATIO = state.dit_mlp_ratio
    mod.DIT_DROPOUT = state.dit_dropout
    mod.USE_TRIPLE_SCALE = state.use_triple_scale
    mod.DIFFUSION_STAGE = state.diffusion_stage
    mod.USE_GUIDANCE_CHANNEL = state.use_guidance_channel
    mod.GUIDANCE_TYPE = state.guidance_type
    mod.MMPD_PATCH_SIZE = int(state.mmpd_patch_size)
    mod.CFG_DROPOUT = state.cfg_dropout
    mod.MODEL_TYPE = state.model_type
    mod.DIFFUSION_TYPE = state.diffusion_type
    mod.USE_ORDINAL_WINDOW_NORM = state.use_ordinal_window_norm
    mod.ORDINAL_TIE_ATOL = state.ordinal_tie_atol
    mod.DETERMINISTIC_ANCHOR_LOSS = state.deterministic_anchor_loss
    mod.DETERMINISTIC_ANCHOR_LAMBDA = state.deterministic_anchor_lambda
    mod.DETERMINISTIC_ANCHOR_ALPHA = state.deterministic_anchor_alpha
    mod.BINARY_ANCHOR_INPUT_MODE = state.binary_anchor_input_mode
    mod.EVAL_SAMPLER = state.eval_sampler
    mod.DISABLE_CROSS_ATTENTION = state.disable_cross_attention
    mod.CROSS_VARIATE_CONTEXT_BIAS = state.cross_variate_context_bias
    mod.USE_WINDOW_NORMALIZATION = state.use_window_normalization
    mod.WINDOW_NORM_CENTER = state.window_norm_center
    mod.WINDOW_NORM_STD_FLOOR = state.window_norm_std_floor
    mod.WINDOW_NORM_LOW_VAR_THRESHOLD = state.window_norm_low_var_threshold
    unit_std = float(state.window_norm_low_var_unit_std)
    per_ds_unit = (state.window_norm_low_var_unit_std_by_dataset or {}).get(state.dataset)
    if per_ds_unit is not None:
        unit_std = float(per_ds_unit)
    mod.WINDOW_NORM_LOW_VAR_UNIT_STD = unit_std
    per_v = (state.window_norm_low_var_unit_std_by_variate or {}).get(state.dataset)
    mod.WINDOW_NORM_LOW_VAR_UNIT_STD_PER_VARIATE = list(per_v) if per_v else None
    mod.LOOKBACK_OVERLAP_CENTER_SHIFT = bool(state.lookback_overlap_center_shift)
    mod.ZERO_GUIDANCE_FORECAST = state.zero_guidance_forecast
    mod.USE_RAW_LOOKBACK_COND_CHANNEL = state.use_raw_lookback_cond_channel
    mod.WINDOW_STRIDE = state.window_stride
    mod.BINARY_NUM_STEPS = state.binary_num_steps
    mod.BINARY_BETA_START = state.binary_beta_start
    mod.BINARY_BETA_END = state.binary_beta_end
    mod.LR_SCHEDULER_TYPE = training_value(state, "lr_scheduler_type", "none")
    mod.LR_WARMUP_EPOCHS = int(training_value(state, "lr_warmup_epochs", 0))
    mod.MAX_SCALE_TUNING = bool(training_value(state, "max_scale_tuning", False))
    mod.MAX_SCALE_TUNING_RANGE = training_value(state, "max_scale_tuning_range", [2.5, 14.0])
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
