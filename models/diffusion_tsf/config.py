"""
config for the diffusion tsf model.
"""

import math
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
    image_height: int = 16
    coarse_image_height: int = 16
    fine_image_height: int = 16
    patch_refine_canvas_height: int = 256
    patch_refine_patch_height: int = 32
    patch_refine_patch_width: int = 8
    patch_refine_col_stride: int = 6
    # Unique absolute 8-step segments + AR prev-refine cond (see patch_refine_segments).
    patch_refine_unique_segments: bool = False
    patch_refine_prev_cond_dropout: float = 0.5
    max_scale: float = 3.5
    representation_mode: str = "cdf"  # pdf or cdf
    staged_representation: str = "value_precision"

    # unified time axis (L+F vs Future-Only)
    unified_time_axis: bool = False

    diffusion_type: str = "binary"
    use_ordinal_window_norm: bool = False
    # Derive any ordinal OOD envelope shift from the lookback alone so an
    # unseen future cannot change the forecast coordinate system.
    ordinal_ood_shift_causal_only: bool = False
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
    # Distance-weighted BCE: W=1+α|r−k| (quadratic miss penalty in BCE space).
    binary_use_boundary_weighted_bce: bool = False
    binary_cdf_distance_alpha: float = 1.0
    diffusion_stage: str = "coarse"  # coarse or patch_refine

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
    # Hybrid flat: per-variate skip of window/instance norm (identity in dataset-affine space).
    # Length must match num_variables when set. See utils/hybrid_flat_dataset_norm.py.
    skip_window_norm_variate_mask: Optional[List[bool]] = None
    hybrid_flat_dataset_norm: bool = False
    hybrid_flat_frac_threshold: float = 0.5
    hybrid_flat_oob_coverage: float = 0.99
    # Shift window center at decode so overlap preds align with past tail (quantization fix).
    lookback_overlap_center_shift: bool = False
    prediction_target: str = "x0"  # x0 or epsilon (bit-flip mask)
    loss_weighting: str = "none"  # none or min_snr
    min_snr_gamma: float = 5.0

    model_type: str = "dit"

    # DiT backbone
    dit_patch_size: Tuple[int, int] = (8, 8)
    dit_cond_patch_size: Optional[Tuple[int, int]] = None
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

    # Frozen patch-decoder encoder tokens for bottleneck cross-attention.
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
        if self.binary_cdf_distance_alpha < 0.0:
            raise ValueError(
                f"binary_cdf_distance_alpha must be >= 0, got {self.binary_cdf_distance_alpha}"
            )
        if self.binary_anchor_input_mode not in {"stationary_flat", "random_bits"}:
            raise ValueError(
                "binary_anchor_input_mode must be 'stationary_flat' or 'random_bits', "
                f"got {self.binary_anchor_input_mode!r}."
            )
        valid_stages = {"coarse", "patch_refine"}
        if self.diffusion_stage not in valid_stages:
            raise ValueError(
                "diffusion_stage must be one of "
                "{'coarse', 'patch_refine'}, "
                f"got {self.diffusion_stage!r}. "
                "fine/finer/vertical_dual/channel_dual paths were removed."
            )
        if self.staged_representation != "value_precision":
            raise ValueError(
                "staged_representation must be 'value_precision', "
                f"got {self.staged_representation!r}."
            )
        if self.diffusion_stage == "coarse":
            expected_height = self.coarse_image_height
            if self.image_height != expected_height:
                raise ValueError(
                    f"staged coarse model expects image_height={expected_height}, "
                    f"got {self.image_height}."
                )
            if self.image_height % self.dit_patch_size[0] != 0:
                raise ValueError("staged image_height must divide dit_patch_size[0].")
            if self.coarse_image_height <= 0 or self.fine_image_height <= 0:
                raise ValueError("coarse/fine image heights must be positive (fine used for past Hc∥Hf).")
            if self.coarse_image_height % self.dit_patch_size[0] != 0:
                raise ValueError("coarse_image_height must divide dit_patch_size[0].")
            if self.fine_image_height % self.dit_patch_size[0] != 0:
                raise ValueError("fine_image_height must divide dit_patch_size[0].")
        if self.diffusion_stage == "patch_refine":
            if self.image_height != int(self.patch_refine_patch_height):
                raise ValueError(
                    "patch_refine expects image_height == patch_refine_patch_height "
                    f"({self.patch_refine_patch_height}), got {self.image_height}."
                )
            if self.image_height % self.dit_patch_size[0] != 0:
                raise ValueError("patch_refine image_height must divide dit_patch_size[0].")
            if int(self.patch_refine_patch_width) % self.dit_patch_size[1] != 0:
                raise ValueError("patch_refine_patch_width must divide dit_patch_size[1].")
            if int(self.patch_refine_canvas_height) % int(self.coarse_image_height) != 0:
                raise ValueError(
                    "patch_refine_canvas_height must be divisible by coarse_image_height."
                )
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
    def data_occupancy_channels(self) -> int:
        """Noisy canvas occupancy channels."""
        return 1

    @property
    def dit_out_channels(self) -> int:
        # Binary: C x0 heads + C zt heads.
        if self.diffusion_type == "binary":
            return 2 * self.data_occupancy_channels
        return self.data_occupancy_channels

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
        patch_refine_extra = 3 if self.diffusion_stage == "patch_refine" else 0
        return (
            self.data_occupancy_channels
            + self.num_aux_channels
            + patch_refine_extra
        )

    @property
    def visual_cond_channels(self) -> int:
        per_scale = 1 + (1 if self.use_value_channel else 0)
        raw_extra = 1 if self.use_raw_lookback_cond_channel else 0
        if self.diffusion_stage == "patch_refine":
            return 1
        if self.diffusion_stage == "coarse":
            return per_scale * 2 + raw_extra
        raise ValueError(f"unsupported diffusion_stage={self.diffusion_stage!r}")
