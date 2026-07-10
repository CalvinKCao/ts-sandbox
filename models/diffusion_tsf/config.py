"""
config for the diffusion tsf model.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple


@dataclass
class DiffusionTSFConfig:
    """Settings for binary CDF diffusion with FactorizedDiT."""

    # seq lens
    lookback_length: int = 512
    forecast_length: int = 96
    # YAML horizon (before overlap); used for AR when > diffusion_chunk_horizon.
    dataset_forecast_length: int = 0
    # Cap 2D past conditioning width; 0 = legacy min(past_len, target_width).
    diffusion_lookback_cap: int = 0
    # Fixed denoiser chunk width; 0 = use full dataset_forecast_length.
    diffusion_chunk_horizon: int = 0
    # Subsample timesteps before 2D encode (x[..., ::stride]); decode upsamples linearly.
    representation_time_stride: int = 1
    # When False, past 2D cond keeps native lookback width (e.g. 336); DiT cond tokens
    # are separate from the wider horizon canvas. Fine-stage horizon cond channels are
    # zero-padded to horizon width before channel concat, never bilinearly stretched.
    past_cond_resize_to_horizon: bool = True
    # iTransformer encoder length; None -> lookback_length.
    itrans_lookback_length: Optional[int] = None

    # Lookback overlap: predict the last K observed timesteps alongside the
    # future horizon to smooth the past/future boundary.
    lookback_overlap: int = 8
    past_loss_weight: float = 0.3

    # multivariate support
    num_variables: int = 1
    variate_factorized: bool = True
    use_variate_embedding: bool = True
    disable_cross_attention: bool = False
    cross_variate_context_bias: float = 0.0

    # 2d mapping (hard binary CDF, no vertical blur)
    image_height: int = 32
    coarse_image_height: int = 16
    fine_image_height: int = 16
    finer_image_height: int = 16
    max_scale: float = 3.5
    representation_mode: str = "cdf"  # pdf or cdf
    staged_representation: str = "value_precision"  # value_precision, haar_frequency, or fourier_frequency
    haar_high_freq_percent: float = 0.38
    haar_high_freq_levels: int = 0
    haar_fine_max_scale: float = 0.0
    fourier_high_freq_percent: float = 0.85
    fourier_high_freq_cutoff_bin: int = 0
    fourier_fine_max_scale: float = 0.0
    fourier_flatline_atol: float = 1e-8
    fourier_fft_edge_mode: str = "mirror_pad"
    fourier_mirror_pad_frac: float = 0.25
    fourier_high_freq_cutoff_bins_per_variate: Optional[List[int]] = None
    fourier_fine_max_scale_per_variate: Optional[List[float]] = None
    coarse_flatline_blur_fine_target: bool = False
    coarse_flatline_blur_radius: int = 4
    coarse_flatline_blur_kernel: str = "gaussian"
    coarse_flatline_blur_atol: Optional[float] = None

    # unified time axis (L+F vs Future-Only)
    unified_time_axis: bool = False

    diffusion_type: str = "binary"
    use_ordinal_window_norm: bool = False
    ordinal_tie_atol: float = 1e-6
    ordinal_ladder: Optional[Any] = None
    binary_num_steps: int = 1000
    binary_sample_steps: int = 20
    binary_beta_start: float = 1e-5
    binary_beta_end: float = 0.5
    binary_noise_schedule: str = "sqrt_linear"  # sqrt_linear, linear, cosine
    # Length-dependent β remap (diag / ablation). Default none = identity schedule.
    binary_length_mode: str = "none"  # none | power | scale
    binary_length_g: float = 1.0
    binary_length_scale: float = 1.0
    binary_boundary_weight: float = 1.0
    binary_background_weight: float = 0.1
    binary_boundary_width: int = 8
    binary_use_boundary_weighted_bce: bool = False
    diffusion_stage: str = "joint"  # joint, coarse, fine, finer
    use_triple_scale: bool = False

    # classifier-free guidance (training dropout only; inference is always conditional)
    cfg_dropout: float = 0.1

    # 2d augs (cutout)
    cutout_prob: float = 0.5
    cutout_min_masks: int = 1
    cutout_max_masks: int = 3
    cutout_shapes: List[Tuple[int, int]] = field(
        default_factory=lambda: [(16, 16), (32, 5)]
    )

    # decoding
    decode_temperature: float = 0.5

    emd_lambda: float = 0.2
    use_monotonicity_loss: bool = False
    monotonicity_weight: float = 1.0
    guidance_penalty_weight: float = 0.0

    # Deterministic anchor loss at max-noise stationary Bernoulli(0.5) state.
    use_deterministic_anchor_loss: bool = False
    deterministic_anchor_lambda: float = 0.99
    deterministic_anchor_alpha: float = 0.5
    binary_anchor_input_mode: str = "stationary_flat"  # stationary_flat or random_bits
    use_window_normalization: bool = True
    # Center for per-window norm: past mean (default) or last lookback timestep.
    window_norm_center: str = "mean"
    window_norm_std_floor: float = 1e-8
    # When past_std < threshold (z-score units), divide by unit_std instead of std_floor.
    window_norm_low_var_threshold: float = 0.0
    window_norm_low_var_unit_std: float = 1.0
    # Per local variate index (batch dim V); falls back to window_norm_low_var_unit_std.
    window_norm_low_var_unit_std_per_variate: Optional[List[float]] = None
    # Shift window center at decode so overlap preds align with past tail (quantization fix).
    lookback_overlap_center_shift: bool = False
    zero_guidance_forecast: bool = False
    prediction_target: str = "x0"  # x0 or epsilon (bit-flip mask)
    loss_weighting: str = "none"  # none or min_snr
    min_snr_gamma: float = 5.0

    model_type: str = "dit"

    # DiT backbone
    dit_patch_size: Tuple[int, int] = (8, 8)
    dit_embed_dim: int = 384
    dit_depth: int = 8
    dit_num_heads: int = 6
    dit_mlp_ratio: float = 4.0
    dit_dropout: float = 0.0

    # memory optimization
    use_gradient_checkpointing: bool = False
    unet_max_chunk_size: int = 128  # chunks BV through FactorizedDiT
    use_amp: bool = False

    # aux channels
    use_coordinate_channel: bool = True
    use_time_ramp: bool = False
    use_time_sine: bool = False
    use_value_channel: bool = False
    seasonal_period: int = 96
    # Extra visual cond: coarse CDF of lookback tail in dataset z-score space.
    use_raw_lookback_cond_channel: bool = False

    # Stage 1 guidance (ghost image + encoder tokens)
    use_guidance_channel: bool = True
    context_embedding_dim: int = 256
    guidance_type: str = "itransformer"  # itransformer | patch_decoder
    mmpd_patch_size: int = 12
    itrans_d_model: int = 512

    # train
    learning_rate: float = 2e-4
    batch_size: int = 8

    def __post_init__(self):
        assert self.image_height > 0
        assert self.max_scale > 0
        if self.diffusion_type != "binary":
            raise ValueError(f"diffusion_type must be 'binary', got {self.diffusion_type!r}")
        if self.use_ordinal_window_norm and self.use_window_normalization:
            raise ValueError(
                "use_ordinal_window_norm replaces window normalization; set use_window_normalization=false"
            )
        if self.binary_use_boundary_weighted_bce:
            raise ValueError(
                "Edge CDF boundary-weighted BCE is not supported for binary diffusion yet."
            )
        if self.binary_anchor_input_mode not in {"stationary_flat", "random_bits"}:
            raise ValueError(
                "binary_anchor_input_mode must be 'stationary_flat' or 'random_bits', "
                f"got {self.binary_anchor_input_mode!r}."
            )
        valid_stages = {"joint", "coarse", "fine", "finer"}
        if self.diffusion_stage not in valid_stages:
            raise ValueError(
                "diffusion_stage must be one of {'joint', 'coarse', 'fine', 'finer'}, "
                f"got {self.diffusion_stage!r}."
            )
        if self.diffusion_stage == "finer" and not self.use_triple_scale:
            raise ValueError("diffusion_stage='finer' requires use_triple_scale=True.")
        if self.staged_representation not in {"value_precision", "haar_frequency", "fourier_frequency"}:
            raise ValueError(
                "staged_representation must be 'value_precision', 'haar_frequency', or "
                f"'fourier_frequency', got {self.staged_representation!r}."
            )
        if self.staged_representation == "haar_frequency":
            if self.use_triple_scale:
                raise ValueError("haar_frequency staged representation supports only coarse/fine stages.")
            if not 0.0 < float(self.haar_high_freq_percent) <= 1.0:
                raise ValueError("haar_high_freq_percent must be in (0, 1].")
            if int(self.haar_high_freq_levels) < 0:
                raise ValueError("haar_high_freq_levels must be >= 0.")
            if float(self.haar_fine_max_scale) < 0.0:
                raise ValueError("haar_fine_max_scale must be >= 0.")
        if self.staged_representation == "fourier_frequency":
            if self.use_triple_scale:
                raise ValueError("fourier_frequency staged representation supports only coarse/fine stages.")
            if not 0.0 < float(self.fourier_high_freq_percent) <= 1.0:
                raise ValueError("fourier_high_freq_percent must be in (0, 1].")
            if int(self.fourier_high_freq_cutoff_bin) < 0:
                raise ValueError("fourier_high_freq_cutoff_bin must be >= 0.")
            if float(self.fourier_fine_max_scale) < 0.0:
                raise ValueError("fourier_fine_max_scale must be >= 0.")
            if float(self.fourier_flatline_atol) < 0.0:
                raise ValueError("fourier_flatline_atol must be >= 0.")
        if self.use_triple_scale and self.diffusion_stage == "joint":
            raise ValueError(
                "use_triple_scale has no joint forward path; use staged coarse/fine/finer."
            )
        if self.diffusion_stage in {"coarse", "fine", "finer"}:
            expected_height = {
                "coarse": self.coarse_image_height,
                "fine": self.fine_image_height,
                "finer": self.finer_image_height,
            }[self.diffusion_stage]
            if self.image_height != expected_height:
                raise ValueError(
                    f"staged {self.diffusion_stage} model expects image_height={expected_height}, "
                    f"got {self.image_height}."
                )
            if self.image_height % self.dit_patch_size[0] != 0:
                raise ValueError("staged image_height must divide dit_patch_size[0].")
            if self.coarse_image_height <= 0 or self.fine_image_height <= 0 or self.finer_image_height <= 0:
                raise ValueError("coarse/fine/finer image heights must be positive.")
            if self.coarse_image_height % self.dit_patch_size[0] != 0:
                raise ValueError("coarse_image_height must divide dit_patch_size[0].")
            if self.fine_image_height % self.dit_patch_size[0] != 0:
                raise ValueError("fine_image_height must divide dit_patch_size[0].")
            if self.use_triple_scale and self.finer_image_height % self.dit_patch_size[0] != 0:
                raise ValueError("finer_image_height must divide dit_patch_size[0].")
        if not 0.0 <= self.cfg_dropout <= 1.0:
            raise ValueError("cfg_dropout must be in [0, 1].")
        assert 0 <= self.cutout_prob <= 1
        assert 0.0 <= self.deterministic_anchor_lambda <= 1.0
        assert 0.0 <= self.deterministic_anchor_alpha < 1.0
        assert self.window_norm_std_floor > 0
        assert self.window_norm_low_var_threshold >= 0.0
        assert self.window_norm_low_var_unit_std > 0.0
        per_v = self.window_norm_low_var_unit_std_per_variate
        if per_v is not None:
            if len(per_v) < 1:
                raise ValueError("window_norm_low_var_unit_std_per_variate must be non-empty.")
            if any(float(u) <= 0.0 for u in per_v):
                raise ValueError("window_norm_low_var_unit_std_per_variate values must be > 0.")
        if self.window_norm_center not in {"mean", "last"}:
            raise ValueError(
                "window_norm_center must be 'mean' or 'last', "
                f"got {self.window_norm_center!r}."
            )
        if self.diffusion_type == "binary":
            if self.binary_noise_schedule not in {"sqrt_linear", "linear", "cosine"}:
                raise ValueError(
                    "binary_noise_schedule must be one of {'sqrt_linear', 'linear', 'cosine'}, "
                    f"got {self.binary_noise_schedule!r}."
                )
            if self.binary_length_mode not in {"none", "power", "scale"}:
                raise ValueError(
                    "binary_length_mode must be one of {'none', 'power', 'scale'}, "
                    f"got {self.binary_length_mode!r}."
                )
            if float(self.binary_length_g) <= 0:
                raise ValueError("binary_length_g must be > 0.")
            if float(self.binary_length_scale) <= 0:
                raise ValueError("binary_length_scale must be > 0.")
            if self.prediction_target not in {"x0", "epsilon"}:
                raise ValueError("prediction_target must be 'x0' or 'epsilon'.")
            if self.loss_weighting not in {"none", "min_snr"}:
                raise ValueError("loss_weighting must be 'none' or 'min_snr'.")
        if self.min_snr_gamma <= 0:
            raise ValueError("min_snr_gamma must be > 0.")
        assert self.representation_mode in ["pdf", "cdf"]
        if not self.variate_factorized:
            raise ValueError("variate_factorized=False is no longer supported.")
        if self.model_type != "dit":
            raise ValueError(f"model_type must be 'dit', got {self.model_type!r}")

    @property
    def dit_out_channels(self) -> int:
        return 2 if self.diffusion_type == "binary" else 1

    @property
    def bin_width(self) -> float:
        return (2 * self.max_scale) / self.image_height

    @property
    def bin_centers(self) -> List[float]:
        return [
            (j + 0.5) * self.bin_width - self.max_scale
            for j in range(self.image_height)
        ]

    @property
    def num_aux_channels(self) -> int:
        count = 0
        if self.use_coordinate_channel:
            count += 1
        if self.use_time_ramp:
            count += 1
        if self.use_time_sine:
            count += 1
        if self.use_value_channel:
            count += 1
        return count

    @property
    def backbone_in_channels(self) -> int:
        return 1 + self.num_aux_channels + self.guidance_channels

    @property
    def visual_cond_channels(self) -> int:
        per_scale = 1 + (1 if self.use_value_channel else 0)
        raw_extra = 1 if self.use_raw_lookback_cond_channel else 0
        if self.diffusion_stage == "coarse":
            return per_scale * (3 if self.use_triple_scale else 2) + raw_extra
        if self.diffusion_stage == "fine":
            return per_scale * (4 if self.use_triple_scale else 3) + raw_extra
        if self.diffusion_stage == "finer":
            return per_scale * 5 + raw_extra
        if self.use_triple_scale:
            return per_scale * 3 + raw_extra
        return per_scale + raw_extra

    @property
    def guidance_channels(self) -> int:
        if not self.use_guidance_channel:
            return 0
        return 1
