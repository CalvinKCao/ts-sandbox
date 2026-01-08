"""
Configuration for Diffusion-based Time Series Forecasting.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class DiffusionTSFConfig:
    """Configuration for the Diffusion TSF model.
    
    Attributes:
        # Sequence lengths (from ViTime paper)
        lookback_length: int - Length of historical context (default: 512)
        forecast_length: int - Length of forecast horizon (default: 96)
        
        # 2D Mapping parameters
        image_height: int - Height of the 2D representation (default: 128)
        max_scale: float - MS parameter for value truncation (default: 3.5)
        blur_kernel_size: int - Vertical Gaussian blur kernel size (default: 31)
        blur_sigma: float - Sigma for Gaussian blur (default: 1.0)
        representation_mode: str - "pdf" (stripe) or "cdf" (occupancy map)
        
        # U-Net architecture
        unet_channels: List[int] - Channel dimensions at each level
        num_res_blocks: int - Number of residual blocks per level
        attention_levels: List[int] - Levels where attention is applied
        
        # Diffusion parameters
        num_diffusion_steps: int - T, number of diffusion steps (default: 1000)
        beta_start: float - Starting beta for noise schedule
        beta_end: float - Ending beta for noise schedule
        noise_schedule: str - Type of noise schedule ("linear" or "cosine")
        
        # DDIM sampling
        ddim_steps: int - Number of steps for accelerated DDIM sampling
        ddim_eta: float - Eta parameter for DDIM (0 = deterministic)
        
        # Training
        learning_rate: float - Learning rate for optimizer
        batch_size: int - Batch size for training
    """
    
    # Sequence lengths
    lookback_length: int = 512
    forecast_length: int = 96
    
    # Multivariate support
    num_variables: int = 1  # Number of time series variables (1 = univariate, >1 = multivariate)
    
    # 2D Mapping parameters
    image_height: int = 128
    max_scale: float = 3.5  # MS parameter from ViTime
    blur_kernel_size: int = 31
    blur_sigma: float = 1.0
    representation_mode: str = "pdf"  # "pdf" (stripe) or "cdf" (occupancy)
    
    # U-Net architecture
    # Default aligned with ViTime paper (~93M params target)
    unet_channels: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    num_res_blocks: int = 2
    attention_levels: List[int] = field(default_factory=lambda: [1, 2])  # Apply attention at deeper levels
    
    # U-Net kernel size configuration - allows rectangular kernels
    # Can be int (square) or tuple (height, width) for rectangular kernels
    # Height = value axis, Width = temporal axis
    # Example: (3, 5) uses 3-pixel height (value) and 5-pixel width (time)
    unet_kernel_size: Tuple[int, int] = (3, 3)  # Default square 3x3
    
    # Dilated middle block for expanded temporal receptive field
    # Uses exponentially increasing dilations (1, 2, 4, 8) on the time axis
    use_dilated_middle: bool = False  # Enable dilated convolutions in bottleneck
    
    # Diffusion parameters
    num_diffusion_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    noise_schedule: str = "linear"  # "linear" or "cosine"
    
    # DDIM sampling
    ddim_steps: int = 50
    ddim_eta: float = 0.0  # 0 = deterministic DDIM
    
    # Classifier-Free Guidance (CFG)
    cfg_dropout: float = 0.1  # Probability of dropping conditioning during training
    cfg_scale: float = 2.0  # Guidance scale during inference (1.0 = no guidance, >1 = stronger conditioning)
    
    # 2D augmentation (coarse dropout / cutout)
    cutout_prob: float = 0.5
    cutout_min_masks: int = 1
    cutout_max_masks: int = 3
    cutout_shapes: List[Tuple[int, int]] = field(
        default_factory=lambda: [(16, 16), (32, 5)]
    )
    
    # Decoding
    decode_temperature: float = 0.5  # Lower = sharper peaks in softmax (0.1-1.0 typical)
    decode_smoothing: bool = False  # Apply horizontal Gaussian smoothing at inference
    
    # EMD loss weighting
    emd_lambda: float = 0.2

    # Monotonicity regularization (CDF mode)
    use_monotonicity_loss: bool = False
    monotonicity_weight: float = 1.0
    
    # Model selector: "unet" (default) or "transformer"
    model_type: str = "unet"
    
    # Transformer (DiT-style) parameters
    transformer_embed_dim: int = 256
    transformer_depth: int = 6
    transformer_num_heads: int = 8
    transformer_patch_height: int = 16  # Patch size along value axis
    transformer_patch_width: int = 16   # Patch size along time axis
    transformer_dropout: float = 0.1
    
    # Spatial coordinate channel for vertical awareness
    use_coordinate_channel: bool = True  # Concatenate vertical gradient to input
    
    # Temporal coordinate channels for horizontal awareness (fixes phase drift in U-Net)
    use_time_ramp: bool = False  # Add linear ramp channel (-1 to +1 "progress bar")
    use_time_sine: bool = False  # Add sine wave channel (periodic "clock")
    # Value channel: shows the last forecast_length values from past as a 1D representation
    # Each column t contains past_norm[-(forecast_length-t)] broadcast across height
    # This gives recent value context without leaking future info (train/test consistent)
    use_value_channel: bool = False  # Add channel with recent past values (1D→2D broadcast)
    seasonal_period: int = 96  # Period for sine wave (e.g., 96 for hourly data with daily seasonality)
    
    # Conditioning Mode for Past Context
    # Controls how the past context is fed to the U-Net backbone:
    # - "visual_concat": Directly concatenate the masked past image (past 2D + zeros for future)
    #                    to the model input channels. This allows the model to explicitly "see"
    #                    the past trajectory pixels. Bypasses ConditioningEncoder.
    # - "vector_embedding": Use the ConditioningEncoder to extract local/global features from
    #                       the past image, then concatenate these encoded features to input.
    #                       This is the original approach.
    conditioning_mode: str = "visual_concat"  # "visual_concat" or "vector_embedding"
    
    # Hybrid "Visual Guide" Forecasting (Stage 1 Predictor → Diffusion Refinement)
    # When enabled, a coarse forecast from a Stage 1 model (e.g., iTransformer) is converted
    # to a 2D "ghost image" and concatenated to the U-Net input. This allows the diffusion
    # model to focus on refining texture/residuals rather than guessing the global trend.
    use_guidance_channel: bool = False  # Enable guidance channel from Stage 1 predictor
    
    # Hybrid 1D Cross-Attention Conditioning
    # Instead of only using 2D visual conditioning, also inject raw 1D numerical values
    # of the past sequence via Cross-Attention layers in the U-Net
    use_hybrid_condition: bool = True  # Enable 1D context encoder + cross-attention
    context_embedding_dim: int = 128  # Dimension of the 1D context embeddings
    context_input_channels: int = 2  # 1 for value + 1 for time index
    context_encoder_layers: int = 2  # Number of transformer layers in context encoder
    
    # Training
    learning_rate: float = 2e-4
    batch_size: int = 8
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.image_height > 0, "image_height must be positive"
        assert self.max_scale > 0, "max_scale must be positive"
        assert self.blur_kernel_size % 2 == 1, "blur_kernel_size must be odd"
        assert self.num_diffusion_steps > 0, "num_diffusion_steps must be positive"
        assert self.noise_schedule in ["linear", "cosine", "sigmoid", "quadratic"], "Invalid noise schedule"
        assert 0 <= self.cutout_prob <= 1, "cutout_prob must be in [0, 1]"
        assert self.cutout_min_masks > 0 and self.cutout_max_masks >= self.cutout_min_masks, "Invalid cutout mask counts"
        assert self.representation_mode in ["pdf", "cdf"], "representation_mode must be 'pdf' or 'cdf'"
        assert self.seasonal_period > 0, "seasonal_period must be positive"
        assert self.num_variables > 0, "num_variables must be positive"
        assert self.conditioning_mode in ["visual_concat", "vector_embedding"], \
            "conditioning_mode must be 'visual_concat' or 'vector_embedding'"
        # Validate kernel size
        kh, kw = self.unet_kernel_size
        assert kh > 0 and kw > 0, "unet_kernel_size dimensions must be positive"
        assert kh % 2 == 1 and kw % 2 == 1, "unet_kernel_size dimensions must be odd for symmetric padding"
    
    @property
    def bin_width(self) -> float:
        """Width of each bin in the 2D representation."""
        return (2 * self.max_scale) / self.image_height
    
    @property
    def bin_centers(self) -> List[float]:
        """Centers of each bin for inverse mapping."""
        return [
            (j + 0.5) * self.bin_width - self.max_scale 
            for j in range(self.image_height)
        ]
    
    @property
    def num_aux_channels(self) -> int:
        """Number of auxiliary channels (coordinate + time channels)."""
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
        """Total input channels for the backbone (data + auxiliary + guidance).
        
        This is the number of channels for the noisy future + auxiliary channels,
        NOT including conditioning channels which are added separately.
        
        If use_guidance_channel is True, adds num_variables extra channels for
        the 2D representation of the Stage 1 predictor's coarse forecast.
        """
        base_channels = self.num_variables + self.num_aux_channels
        if self.use_guidance_channel:
            # Add one guidance channel per variable (the "ghost image")
            base_channels += self.num_variables
        return base_channels
    
    @property
    def visual_cond_channels(self) -> int:
        """Number of visual conditioning channels in visual_concat mode.
        
        In visual_concat mode, the past 2D image (with same number of variables)
        is concatenated directly as conditioning channels.
        """
        return self.num_variables
    
    @property
    def guidance_channels(self) -> int:
        """Number of guidance channels from Stage 1 predictor.
        
        Returns num_variables if guidance is enabled, 0 otherwise.
        """
        return self.num_variables if self.use_guidance_channel else 0

