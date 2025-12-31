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
        blur_sigma: float - Sigma for Gaussian blur (default: 6.0)
        
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
    
    # 2D Mapping parameters
    image_height: int = 128
    max_scale: float = 3.5  # MS parameter from ViTime
    blur_kernel_size: int = 31
    blur_sigma: float = 6.0
    
    # U-Net architecture
    # Default aligned with ViTime paper (~93M params target)
    unet_channels: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    num_res_blocks: int = 2
    attention_levels: List[int] = field(default_factory=lambda: [1, 2])  # Apply attention at deeper levels
    
    # Diffusion parameters
    num_diffusion_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    noise_schedule: str = "linear"  # "linear" or "cosine"
    
    # DDIM sampling
    ddim_steps: int = 50
    ddim_eta: float = 0.0  # 0 = deterministic DDIM
    
    # Training
    learning_rate: float = 2e-4
    batch_size: int = 8
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.image_height > 0, "image_height must be positive"
        assert self.max_scale > 0, "max_scale must be positive"
        assert self.blur_kernel_size % 2 == 1, "blur_kernel_size must be odd"
        assert self.num_diffusion_steps > 0, "num_diffusion_steps must be positive"
        assert self.noise_schedule in ["linear", "cosine"], "Invalid noise schedule"
    
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

