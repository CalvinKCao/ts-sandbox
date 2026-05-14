"""
config for the diffusion tsf model.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


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
    # process each variate independently thru a shared unet instead of stacking as channels.
    # this is now the only supported U-Net mode.
    # ignored for V=1 (no-op). cross-attn tokens are always produced when V>1.
    variate_factorized: bool = True
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

    # which backbone: "unet" (ConditionalUNet2D) or "dit" (FactorizedDiT)
    model_type: str = "unet"

    # DiT backbone params (used when model_type == "dit").
    # Factorized per-variate, AdaLN-Zero time conditioning, single bottleneck
    # cross-attention to (BV, V, ctx_dim) iTransformer tokens at depth // 2.
    dit_patch_size: Tuple[int, int] = (8, 8)
    dit_embed_dim: int = 384
    dit_depth: int = 8
    dit_num_heads: int = 6
    dit_mlp_ratio: float = 4.0
    dit_dropout: float = 0.0

    
    # memory optimization flags
    use_gradient_checkpointing: bool = False
    unet_max_chunk_size: int = 128
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

    # ---- End-to-end joint iTransformer + diffusion training -----------------
    # When True, the iTransformer guidance model is trained jointly with the
    # diffusion backbone. Gradients flow back through the cross-attention
    # encoder-token path (always) and through the ghost-image path (only when
    # joint_use_ghost_image=True; even then the encode_to_2d's clamp+floor is
    # non-differentiable so the practical contribution there is limited).
    # An auxiliary forecast MSE loss anchors the iTransformer as a real
    # forecaster rather than letting it drift into a denoising-helper role.
    e2e_joint_training: bool = False

    # Whether to freeze the guidance model. Independent of e2e_joint_training
    # so that "old behavior" (e2e=False, frozen=True) is preserved by default.
    # Setting e2e_joint_training=True automatically implies freeze_guidance_model=False
    # via __post_init__ unless the user explicitly overrides it.
    freeze_guidance_model: bool = True

    # Auxiliary forecast loss weight (active only when e2e_joint_training=True).
    # Loss term = aux_forecast_loss_weight * MSE(itrans_forecast_norm, future_norm).
    # Should be ~1-2x the typical noise loss magnitude after a few hundred steps.
    aux_forecast_loss_weight: float = 1.0

    # Joint-training ghost-image variant toggle:
    #   True  (Option B): keep the ghost-image channel; .detach() guidance forecast
    #                     in the ghost-image branch so iTrans does NOT receive
    #                     gradient through encode_to_2d. Gradient still flows
    #                     via cross-attn tokens + aux forecast loss.
    #   False (Option C): drop the ghost-image channel entirely. Tokens-only.
    #                     Saves one U-Net input channel; cleanest gradient story.
    # Only takes effect when e2e_joint_training=True (otherwise legacy
    # use_guidance_channel logic governs the ghost-image channel).
    joint_use_ghost_image: bool = True

    # Number of iTrans-only warmup epochs at the start of joint training.
    # During warmup the diffusion loss is masked out and only aux_forecast_loss
    # is back-propagated, giving the cross-attention tokens meaningful structure
    # before the DiT starts learning to use them.
    itrans_warmup_epochs: int = 1

    # Separate learning rate for iTransformer params when e2e_joint_training=True.
    # Diffusion backbone uses ``learning_rate``. None -> use ``learning_rate``
    # for both param groups.
    itrans_lr: Optional[float] = 1e-4

    def __post_init__(self):
        """check if config is okay."""
        assert self.image_height > 0
        assert self.max_scale > 0
        assert self.blur_kernel_size % 2 == 1
        assert self.num_diffusion_steps > 0
        assert self.noise_schedule in ["linear", "cosine", "sigmoid", "quadratic"]
        assert 0 <= self.cutout_prob <= 1
        assert self.representation_mode in ["pdf", "cdf"]
        if not self.variate_factorized:
            raise ValueError("variate_factorized=False is no longer supported; use the factorized path.")
        if self.model_type not in ("unet", "dit"):
            raise ValueError(f"model_type must be 'unet' or 'dit', got {self.model_type!r}")
        if self.e2e_joint_training:
            # Joint training is meaningless if iTrans is frozen.
            self.freeze_guidance_model = False
        
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
    def emits_ghost_image(self) -> bool:
        """Whether a guidance ghost image is actually emitted as a U-Net input channel.

        In joint-training mode with joint_use_ghost_image=False (Option C, tokens-only)
        the ghost image is suppressed even if use_guidance_channel is True.
        """
        if not self.use_guidance_channel:
            return False
        if self.e2e_joint_training and not self.joint_use_ghost_image:
            return False
        return True

    @property
    def backbone_in_channels(self) -> int:
        """total input channels for the backbone."""
        # per-variate: 1 data ch + aux + optional 1 guidance ch
        return 1 + self.num_aux_channels + (1 if self.emits_ghost_image else 0)

    @property
    def visual_cond_channels(self) -> int:
        """channels for visual concat mode."""
        return 1 + (1 if self.use_value_channel else 0)

    @property
    def guidance_channels(self) -> int:
        """guidance channels actually concatenated to the backbone input."""
        return 1 if self.emits_ghost_image else 0
    
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

