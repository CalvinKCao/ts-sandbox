"""
Complete Diffusion-based Time Series Forecasting Model.

This module integrates:
- Preprocessing (normalization, 2D encoding, blur)
- Conditional U-Net
- DDPM/DDIM diffusion
- 2D to 1D decoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Optional, Tuple

# Handle imports for both module and script execution
try:
    from .config import DiffusionTSFConfig
    from .preprocessing import Standardizer, TimeSeriesTo2D, VerticalGaussianBlur
    from .unet import ConditionalUNet2D
    from .diffusion import DiffusionScheduler
except ImportError:
    from config import DiffusionTSFConfig
    from preprocessing import Standardizer, TimeSeriesTo2D, VerticalGaussianBlur
    from unet import ConditionalUNet2D
    from diffusion import DiffusionScheduler

logger = logging.getLogger(__name__)


class DiffusionTSF(nn.Module):
    """Diffusion-based Time Series Forecasting Model.
    
    Pipeline:
    1. Normalize input (past + future) using local mean/std
    2. Convert to 2D "stripe" representation
    3. Apply vertical Gaussian blur
    4. Train U-Net to denoise future conditioned on past
    5. At inference: generate future via DDPM/DDIM
    6. Decode 2D representation back to 1D
    """
    
    def __init__(self, config: DiffusionTSFConfig):
        """
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        
        # Preprocessing modules
        self.to_2d = TimeSeriesTo2D(
            height=config.image_height,
            max_scale=config.max_scale
        )
        self.blur = VerticalGaussianBlur(
            kernel_size=config.blur_kernel_size,
            sigma=config.blur_sigma
        )
        
        # U-Net for noise prediction
        self.unet = ConditionalUNet2D(
            in_channels=1,
            out_channels=1,
            channels=config.unet_channels,
            num_res_blocks=config.num_res_blocks,
            attention_levels=config.attention_levels,
            image_height=config.image_height
        )
        
        # Diffusion scheduler (not a nn.Module, managed separately)
        self.scheduler = DiffusionScheduler(
            num_steps=config.num_diffusion_steps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            schedule=config.noise_schedule
        )
        
        logger.info(f"DiffusionTSF initialized:")
        logger.info(f"  Lookback: {config.lookback_length}, Forecast: {config.forecast_length}")
        logger.info(f"  Image size: {config.image_height} x W")
        logger.info(f"  Diffusion steps: {config.num_diffusion_steps}")
    
    def to(self, device):
        """Move model and scheduler to device."""
        super().to(device)
        self.scheduler = self.scheduler.to(device)
        return self
    
    def _normalize_sequence(
        self,
        past: torch.Tensor,
        future: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Normalize sequences using past statistics.
        
        Args:
            past: Past sequence of shape (batch, past_len)
            future: Optional future sequence of shape (batch, future_len)
            
        Returns:
            (past_norm, future_norm, (mean, std))
        """
        # Compute statistics from past only
        mean = past.mean(dim=-1, keepdim=True)
        std = past.std(dim=-1, keepdim=True) + 1e-8
        
        past_norm = (past - mean) / std
        
        if future is not None:
            future_norm = (future - mean) / std
        else:
            future_norm = None
        
        return past_norm, future_norm, (mean, std)
    
    def _denormalize(
        self,
        x: torch.Tensor,
        stats: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Denormalize using stored statistics."""
        mean, std = stats
        return x * std + mean
    
    def encode_to_2d(self, x: torch.Tensor, scale_for_diffusion: bool = True) -> torch.Tensor:
        """Encode 1D time series to blurred 2D representation.
        
        Args:
            x: Normalized time series of shape (batch, seq_len)
            scale_for_diffusion: If True, scale output to [-1, 1] range for diffusion
            
        Returns:
            Blurred 2D image of shape (batch, 1, height, seq_len)
        """
        image = self.to_2d(x)
        blurred = self.blur(image)
        
        if scale_for_diffusion:
            # Scale from [0, ~0.03] to [-1, 1] for proper diffusion SNR
            # First normalize each column to sum to 1, then scale
            # The blur already creates a pseudo-probability, but values are small
            # Multiply by a factor to boost signal, then shift to [-1, 1]
            blurred = blurred * 30.0  # Now roughly [0, 1]
            blurred = blurred * 2.0 - 1.0  # Now [-1, 1]
        
        return blurred
    
    def decode_from_2d(self, image: torch.Tensor, unscale_from_diffusion: bool = True) -> torch.Tensor:
        """Decode 2D representation to 1D time series.
        
        Uses expectation over the probability distribution at each time step.
        
        Args:
            image: 2D image of shape (batch, 1, height, seq_len)
            unscale_from_diffusion: If True, unscale from [-1, 1] range
            
        Returns:
            Time series of shape (batch, seq_len)
        """
        if unscale_from_diffusion:
            # Reverse the scaling: [-1, 1] -> [0, ~0.03]
            image = (image + 1.0) / 2.0  # [-1, 1] -> [0, 1]
            image = image / 30.0  # [0, 1] -> [0, ~0.03]
        
        return self.to_2d.inverse(image)
    
    def forward(
        self,
        past: torch.Tensor,
        future: torch.Tensor,
        t: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass.
        
        Computes the diffusion loss:
        1. Normalize and encode to 2D
        2. Add noise to future image
        3. Predict noise with U-Net conditioned on past
        4. Return loss between true and predicted noise
        
        Args:
            past: Past sequence of shape (batch, past_len)
            future: Future sequence of shape (batch, future_len)
            t: Optional diffusion timesteps (sampled randomly if None)
            
        Returns:
            Dictionary with 'loss' and intermediate values for debugging
        """
        batch_size = past.shape[0]
        device = past.device
        
        # Normalize using past statistics
        past_norm, future_norm, stats = self._normalize_sequence(past, future)
        
        # Encode to 2D
        past_2d = self.encode_to_2d(past_norm)  # (batch, 1, H, past_len)
        future_2d = self.encode_to_2d(future_norm)  # (batch, 1, H, future_len)
        
        logger.debug(f"past_2d shape: {past_2d.shape}, future_2d shape: {future_2d.shape}")
        
        # Sample random timesteps if not provided
        if t is None:
            t = torch.randint(0, self.config.num_diffusion_steps, (batch_size,), device=device)
        
        # Add noise to future
        noisy_future, noise = self.scheduler.add_noise(future_2d, t)
        
        # Predict noise
        noise_pred = self.unet(noisy_future, t, past_2d)
        
        # L2 loss on noise
        loss = F.mse_loss(noise_pred, noise)
        
        return {
            'loss': loss,
            'noise': noise,
            'noise_pred': noise_pred,
            'past_2d': past_2d,
            'future_2d': future_2d,
            'noisy_future': noisy_future,
            't': t
        }
    
    @torch.no_grad()
    def generate(
        self,
        past: torch.Tensor,
        use_ddim: bool = True,
        num_ddim_steps: int = 50,
        eta: float = 0.0,
        verbose: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Generate future predictions given past context.
        
        Args:
            past: Past sequence of shape (batch, past_len)
            use_ddim: Whether to use accelerated DDIM sampling
            num_ddim_steps: Number of DDIM steps (ignored if use_ddim=False)
            eta: DDIM stochasticity parameter
            verbose: Whether to log progress
            
        Returns:
            Dictionary with predictions and intermediate values
        """
        batch_size = past.shape[0]
        device = past.device
        
        # Normalize past
        past_norm, _, stats = self._normalize_sequence(past)
        
        # Encode past to 2D
        past_2d = self.encode_to_2d(past_norm)
        
        # Shape for future image
        future_shape = (
            batch_size,
            1,
            self.config.image_height,
            self.config.forecast_length
        )
        
        # Generate via diffusion
        if use_ddim:
            future_2d = self.scheduler.sample_ddim(
                model=self.unet,
                shape=future_shape,
                cond=past_2d,
                num_steps=num_ddim_steps,
                eta=eta,
                device=device,
                verbose=verbose
            )
        else:
            future_2d = self.scheduler.sample_ddpm(
                model=self.unet,
                shape=future_shape,
                cond=past_2d,
                device=device,
                verbose=verbose
            )
        
        # Decode to 1D (normalized)
        future_norm = self.decode_from_2d(future_2d)
        
        # Denormalize
        future = self._denormalize(future_norm, stats)
        
        return {
            'prediction': future,
            'prediction_norm': future_norm,
            'future_2d': future_2d,
            'past_2d': past_2d
        }
    
    def get_loss(
        self,
        past: torch.Tensor,
        future: torch.Tensor
    ) -> torch.Tensor:
        """Convenience method to get just the loss for training.
        
        Args:
            past: Past sequence of shape (batch, past_len)
            future: Future sequence of shape (batch, future_len)
            
        Returns:
            Scalar loss value
        """
        outputs = self.forward(past, future)
        return outputs['loss']

