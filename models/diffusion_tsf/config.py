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
        image_height: height of the 2D thing (default: 128)
        max_scale: for truncating values (default: 3.5)
        blur_kernel_size: gaussian blur kernel size (default: 31)
        blur_sigma: sigma for blur (default: 1.0)
        representation_mode: "pdf" or "cdf" (occupancy map)
        
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
        ddim_eta: eta for ddim (0 = deterministic)
        
    train:
        learning_rate: lr
        batch_size: batch size
    """
    
    # seq lens
    lookback_length: int = 512
    forecast_length: int = 96
    
    # Lookback overlap: predict the last K observed timesteps alongside the
    # future horizon to smooth the past/future boundary. The diffusion model
    # denoises a (K+H)-wide region; during inference the first K are discarded.
    lookback_overlap: int = 8
    past_loss_weight: float = 0.3
    
    # multivariate support
    num_variables: int = 1  # how many variables (1 = uni, >1 = multi)
    # all V variates ride along as channels in a single U-Net pass
    # (RGB-style). cross-attn tokens are always produced when V>1.
    disable_cross_attention: bool = False
    
    # 2d mapping
    image_height: int = 64  # height of 2d rep (64 is faster)
    max_scale: float = 3.5  # MS param
    blur_kernel_size: int = 31
    blur_sigma: float = 1.0
    representation_mode: str = "cdf"  # pdf or cdf
    
    # unified time axis (L+F vs Future-Only)
    # if True: diffuse on (Lookback + Forecast) combined. 
    #          width = 512 + 96 = 608. slow but smooth.
    # if False: diffuse on Forecast only. much faster.
    unified_time_axis: bool = False
    
    # unet guts
    unet_channels: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    num_res_blocks: int = 2
    attention_levels: List[int] = field(default_factory=lambda: [1, 2])
    
    # kernel size - can be int or (h, w)
    unet_kernel_size: Tuple[int, int] = (3, 3)  
    
    # dilated middle bit for more temporal field
    use_dilated_middle: bool = False  
    
    # diffusion params
    num_diffusion_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    noise_schedule: str = "linear"  
    
    # ddim stuff
    ddim_steps: int = 50
    ddim_eta: float = 0.0  
    
    # classifier-free guidance
    cfg_dropout: float = 0.1  # drop conditioning prob
    cfg_scale: float = 2.0  # how hard to guide (1 = none)
    
    # 2d augs (cutout)
    cutout_prob: float = 0.5
    cutout_min_masks: int = 1
    cutout_max_masks: int = 3
    cutout_shapes: List[Tuple[int, int]] = field(
        default_factory=lambda: [(16, 16), (32, 5)]
    )
    
    # decoding
    decode_temperature: float = 0.5  # lower = sharper
    
    # emd loss weight
    emd_lambda: float = 0.2

    # monotonicity regulariztion (cdf mode)
    use_monotonicity_loss: bool = False
    monotonicity_weight: float = 1.0
    
    # penalty for deviating from itransformer guidance ghost image
    guidance_penalty_weight: float = 0.0

    # backbone: "unet" or "dit" (FactorizedDiT; works for multi-channel layout)
    model_type: str = "unet"

    dit_patch_size: Tuple[int, int] = (8, 8)
    dit_embed_dim: int = 384
    dit_depth: int = 8
    dit_num_heads: int = 6
    dit_mlp_ratio: float = 4.0
    dit_dropout: float = 0.0

    # memory optimization flags
    use_gradient_checkpointing: bool = False
    use_amp: bool = False  # bfloat16 mixed precision
    
    # aux channels
    use_coordinate_channel: bool = True  # vertical gradient
    use_time_ramp: bool = False  # linear ramp
    use_time_sine: bool = False  # sine wave
    use_value_channel: bool = False  # last past values
    seasonal_period: int = 96  
    
    # Stage 1 Guidance (e.g. iTransformer)
    # adds a "ghost image" to help the diffusion model
    use_guidance_channel: bool = True  
    
    # SpatialTransformerBlock (self+cross attn) always used at attention_levels.
    # use_hybrid_condition removed — attention_levels is the single knob.
    context_embedding_dim: int = 256
    itrans_d_model: int = 512         # must match iTransformer d_model at construction
    
    # train
    learning_rate: float = 2e-4
    batch_size: int = 8
    
    def __post_init__(self):
        """check if config is okay."""
        assert self.image_height > 0
        assert self.max_scale > 0
        assert self.blur_kernel_size % 2 == 1
        assert self.num_diffusion_steps > 0
        assert self.noise_schedule in ["linear", "cosine", "sigmoid", "quadratic"]
        assert 0 <= self.cutout_prob <= 1
        assert self.representation_mode in ["pdf", "cdf"]
        if self.model_type not in ("unet", "dit"):
            raise ValueError(f"model_type must be 'unet' or 'dit', got {self.model_type!r}")

    @property
    def bin_width(self) -> float:
        """width of each bin."""
        return (2 * self.max_scale) / self.image_height
    
    @property
    def bin_centers(self) -> List[float]:
        """centers of bins."""
        return [
            (j + 0.5) * self.bin_width - self.max_scale 
            for j in range(self.image_height)
        ]
    
    @property
    def num_aux_channels(self) -> int:
        """how many extra channels we got."""
        count = 0
        if self.use_coordinate_channel: count += 1
        if self.use_time_ramp: count += 1
        if self.use_time_sine: count += 1
        if self.use_value_channel: count += 1
        return count
    
    @property
    def backbone_in_channels(self) -> int:
        """total input channels for the backbone (V noisy + aux + V guidance)."""
        return (
            self.num_variables
            + self.num_aux_channels
            + (self.num_variables if self.use_guidance_channel else 0)
        )

    @property
    def visual_cond_channels(self) -> int:
        """channels for visual concat mode (one per variate, +1 if value channel)."""
        return self.num_variables + (1 if self.use_value_channel else 0)

    @property
    def guidance_channels(self) -> int:
        """guidance channels (one per variate)."""
        return self.num_variables if self.use_guidance_channel else 0


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

