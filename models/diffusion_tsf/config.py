"""
config for the diffusion tsf model.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class DiffusionTSFConfig:
    """settings for the model.

    seq lengths:
        lookback_length: historic context (default: 512)
        forecast_length: forecast horizon (default: 96)

    2D mapping:
        image_height: height of the 2D CDF occupancy map (default: 128)
        max_scale: for truncating z-scored values (default: 3.5)
        blur_kernel_size: gaussian blur kernel size (default: 31)
        blur_sigma: sigma for blur (default: 1.0)

    unet:
        unet_channels: channels at each level
        num_res_blocks: res blocks per level
        attention_levels: where to put attention

    diffusion:
        num_diffusion_steps: T (default: 1000)
        beta_start: start beta
        beta_end: end beta
        noise_schedule: "linear" or "cosine"

    sampling:
        ddim_steps: steps for DDIM
        ddim_eta: eta for DDIM (0 = deterministic, >0 = stochastic)

    train:
        learning_rate: lr
        batch_size: batch size
    """

    # seq lens
    lookback_length: int = 512
    forecast_length: int = 96

    # lookback overlap: predict last K observed timesteps alongside the horizon
    # to smooth the past/future boundary. first K steps are discarded at inference.
    lookback_overlap: int = 0
    past_loss_weight: float = 0.3

    # multivariate support
    num_variables: int = 1
    # process each variate independently through a shared unet instead of stacking as channels.
    # the unet sees (B*V, 1, H, W) per step; cross-variate info from bottleneck cross-attn.
    variate_factorized: bool = True

    # 2d mapping (CDF occupancy representation only)
    image_height: int = 64
    max_scale: float = 3.5
    blur_kernel_size: int = 31
    blur_sigma: float = 1.0

    # unified time axis (L+F vs Future-Only)
    unified_time_axis: bool = False

    # unet guts
    unet_channels: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    num_res_blocks: int = 2
    attention_levels: List[int] = field(default_factory=lambda: [1, 2])
    unet_kernel_size: Tuple[int, int] = (3, 3)
    use_dilated_middle: bool = False

    # diffusion params
    num_diffusion_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    noise_schedule: str = "linear"

    # ddim (eta=0 → deterministic, eta>0 → stochastic)
    ddim_steps: int = 50
    ddim_eta: float = 0.0

    # classifier-free guidance
    cfg_dropout: float = 0.1
    cfg_scale: float = 2.0

    # 2d augs (cutout)
    cutout_prob: float = 0.5
    cutout_min_masks: int = 1
    cutout_max_masks: int = 3
    cutout_shapes: List[Tuple[int, int]] = field(
        default_factory=lambda: [(16, 16), (32, 5)]
    )

    # decoding
    decode_temperature: float = 0.5
    decode_smoothing: bool = False

    # emd loss weight
    emd_lambda: float = 0.2

    # monotonicity regularization
    use_monotonicity_loss: bool = False
    monotonicity_weight: float = 1.0

    # backbone: "unet" or "transformer"
    model_type: str = "unet"

    # -----------------------------------------------------------------------
    # binary diffusion (BDPM-inspired)
    # set diffusion_type="binary" to swap gaussian noise for bit-flip XOR diffusion.
    # removes the need for gaussian blur — CDF images stay hard binary.
    # requires variate_factorized=True.
    # -----------------------------------------------------------------------
    diffusion_type: str = "gaussian"       # "gaussian" | "binary"
    binary_num_steps: int = 1000
    binary_sample_steps: int = 20
    binary_beta_start: float = 1e-5
    binary_beta_end: float = 0.5
    binary_boundary_weight: float = 1.0    # bce weight near cdf boundary
    binary_background_weight: float = 0.1  # bce weight far from boundary
    binary_boundary_width: int = 8         # rows within boundary that get high weight

    # transformer backbone params (DiffusionTransformer, not CI-DiT)
    transformer_embed_dim: int = 256
    transformer_depth: int = 6
    transformer_num_heads: int = 8
    transformer_patch_height: int = 16
    transformer_patch_width: int = 16
    transformer_dropout: float = 0.1

    # memory / compute optimization
    use_gradient_checkpointing: bool = False
    use_amp: bool = False
    # factor (K_h, K_w) conv into (K_h,1)+(1,K_w) — ~2.25x cheaper for (3,9) kernels
    separable_kernel: bool = False

    # aux channels
    use_coordinate_channel: bool = True
    use_time_ramp: bool = False
    use_time_sine: bool = False
    use_value_channel: bool = False
    seasonal_period: int = 96

    # conditioning mode
    conditioning_mode: str = "visual_concat"

    # Stage 1 guidance (iTransformerGuidance)
    use_guidance_channel: bool = False

    # hybrid 1D conditioning via cross-attention
    use_hybrid_condition: bool = True
    context_embedding_dim: int = 128
    context_input_channels: int = 2
    context_encoder_layers: int = 2

    # train
    learning_rate: float = 2e-4
    batch_size: int = 8

    def __post_init__(self):
        assert self.image_height > 0
        assert self.max_scale > 0
        assert self.blur_kernel_size % 2 == 1
        assert self.num_diffusion_steps > 0
        assert self.noise_schedule in ["linear", "cosine", "sigmoid", "quadratic"]
        assert 0 <= self.cutout_prob <= 1

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
        if self.use_coordinate_channel: count += 1
        if self.use_time_ramp: count += 1
        if self.use_time_sine: count += 1
        if self.use_value_channel: count += 1
        return count

    @property
    def backbone_in_channels(self) -> int:
        if self.variate_factorized:
            return 1 + self.num_aux_channels + (1 if self.use_guidance_channel else 0)
        base_channels = self.num_variables + self.num_aux_channels
        if self.use_guidance_channel:
            base_channels += self.num_variables
        return base_channels

    @property
    def visual_cond_channels(self) -> int:
        vars_per = 1 if self.variate_factorized else self.num_variables
        return vars_per + (1 if self.use_value_channel else 0)

    @property
    def guidance_channels(self) -> int:
        if not self.use_guidance_channel:
            return 0
        return 1 if self.variate_factorized else self.num_variables


@dataclass
class LatentDiffusionConfig(DiffusionTSFConfig):
    """DiffusionTSF hyperparameters plus VAE / latent-space fields."""

    latent_channels: int = 4
    kl_weight: float = 1e-4
    vae_lr: float = 1e-4
    vae_epochs: int = 50
    image_height: int = 128

    def __post_init__(self):
        super().__post_init__()
        if self.lookback_overlap % 4 != 0:
            raise ValueError("lookback_overlap must be divisible by 4 for latent overlap (K_lat = K/4)")
        if self.image_height % 4 != 0:
            raise ValueError("image_height must be divisible by 4 for the VAE (2× stride-2)")

    @property
    def latent_spatial_downsample(self) -> int:
        return 4

    @property
    def latent_image_height(self) -> int:
        return self.image_height // self.latent_spatial_downsample
