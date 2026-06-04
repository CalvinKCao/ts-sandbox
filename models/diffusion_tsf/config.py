"""
config for the diffusion tsf model.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class DiffusionTSFConfig:
    """Settings for binary CDF diffusion with FactorizedDiT."""

    # seq lens
    lookback_length: int = 512
    forecast_length: int = 96

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
    max_scale: float = 3.5
    representation_mode: str = "cdf"  # pdf or cdf

    # unified time axis (L+F vs Future-Only)
    unified_time_axis: bool = False

    # binary diffusion
    diffusion_type: str = "binary"
    binary_num_steps: int = 1000
    binary_sample_steps: int = 20
    binary_beta_start: float = 1e-5
    binary_beta_end: float = 0.5
    binary_noise_schedule: str = "sqrt_linear"  # sqrt_linear, linear, cosine
    binary_boundary_weight: float = 1.0
    binary_background_weight: float = 0.1
    binary_boundary_width: int = 8
    binary_use_boundary_weighted_bce: bool = False
    diffusion_stage: str = "joint"  # joint, coarse, fine
    use_dual_scale: bool = False
    dual_scale_fine_weight: float = 0.5
    dual_scale_independent_timesteps: bool = True

    # classifier-free guidance
    cfg_dropout: float = 0.1
    cfg_scale: float = 2.0
    use_cfg_inference: bool = False

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

    # Deterministic anchor loss at max-noise Bernoulli clean-bit anchor.
    use_deterministic_anchor_loss: bool = False
    deterministic_anchor_lambda: float = 0.99
    deterministic_anchor_alpha: float = 0.5
    use_window_normalization: bool = True
    window_norm_std_floor: float = 1e-8
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

    # Stage 1 guidance (iTransformer ghost image + encoder tokens)
    use_guidance_channel: bool = True
    context_embedding_dim: int = 256
    itrans_d_model: int = 512

    # train
    learning_rate: float = 2e-4
    batch_size: int = 8

    def __post_init__(self):
        assert self.image_height > 0
        assert self.max_scale > 0
        assert self.diffusion_type == "binary", (
            f"Only binary diffusion is supported (got {self.diffusion_type!r})"
        )
        if self.binary_use_boundary_weighted_bce:
            raise ValueError(
                "Edge CDF boundary-weighted BCE is not supported for binary diffusion yet."
            )
        if self.diffusion_stage not in {"joint", "coarse", "fine"}:
            raise ValueError(
                "diffusion_stage must be one of {'joint', 'coarse', 'fine'}, "
                f"got {self.diffusion_stage!r}."
            )
        if self.diffusion_stage in {"coarse", "fine"}:
            if self.use_dual_scale:
                raise ValueError("staged coarse/fine models expect use_dual_scale=False.")
            if self.image_height != 16:
                raise ValueError("staged coarse/fine models expect image_height=16.")
            if self.image_height % self.dit_patch_size[0] != 0:
                raise ValueError("staged image_height must divide dit_patch_size[0].")
        if self.use_dual_scale:
            if self.image_height != 16:
                raise ValueError("use_dual_scale=True expects image_height=16.")
            if self.image_height % self.dit_patch_size[0] != 0:
                raise ValueError("dual-scale image_height must divide dit_patch_size[0].")
        if not 0.0 <= self.dual_scale_fine_weight <= 1.0:
            raise ValueError("dual_scale_fine_weight must be in [0, 1].")
        if not 0.0 <= self.cfg_dropout <= 1.0:
            raise ValueError("cfg_dropout must be in [0, 1].")
        if self.cfg_scale < 0.0:
            raise ValueError("cfg_scale must be >= 0.0.")
        assert 0 <= self.cutout_prob <= 1
        assert 0.0 <= self.deterministic_anchor_lambda <= 1.0
        assert 0.0 <= self.deterministic_anchor_alpha < 1.0
        assert self.window_norm_std_floor > 0
        if self.binary_noise_schedule not in {"sqrt_linear", "linear", "cosine"}:
            raise ValueError(
                "binary_noise_schedule must be one of {'sqrt_linear', 'linear', 'cosine'}, "
                f"got {self.binary_noise_schedule!r}."
            )
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
        if self.diffusion_stage == "coarse":
            return per_scale * 2
        if self.diffusion_stage == "fine":
            return per_scale * 3
        if self.use_dual_scale:
            return per_scale * 2
        return per_scale

    @property
    def guidance_channels(self) -> int:
        if not self.use_guidance_channel:
            return 0
        return 2 if self.use_dual_scale else 1
