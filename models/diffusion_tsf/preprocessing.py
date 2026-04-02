"""
Preprocessing utilities for Diffusion TSF.

This module handles:
1. Standardization (z-score normalization)
2. 2D Encoding (the "Stripe" method)
3. Vertical Gaussian blur
4. Inverse mapping (2D back to 1D)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TimeSeriesTo2D(nn.Module):
    """Maps 1D time series to 2D occupancy (CDF-style) images along the value axis.

    For each time step, bins the normalized value and fills all rows from the bottom
    up to that bin — a soft cumulative / bar-stack view (not a one-hot stripe).
    """

    def __init__(self, height: int = 128, max_scale: float = 3.5):
        """
        Args:
            height: Height H of the 2D representation (number of bins)
            max_scale: MS parameter - values beyond [-MS, MS] are clipped
        """
        super().__init__()
        self.height = height
        self.max_scale = max_scale
        
        # Precompute bin centers for inverse mapping
        # Centers: (j + 0.5) * (2*MS/H) - MS for j in [0, H-1]
        bin_width = (2 * max_scale) / height
        bin_centers = torch.tensor([
            (j + 0.5) * bin_width - max_scale 
            for j in range(height)
        ])
        self.register_buffer('bin_centers', bin_centers)
        
        logger.info(f"TimeSeriesTo2D initialized: H={height}, MS={max_scale}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """1D normalized series → 2D occupancy map (values in [0, 1] per column).

        Univariate: (batch, seq_len) -> (batch, 1, height, seq_len)
        Multivariate: (batch, num_vars, seq_len) -> (batch, num_vars, height, seq_len)
        """
        # Handle univariate case: (batch, seq_len) -> (batch, 1, seq_len)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        batch_size, num_vars, seq_len = x.shape
        
        # Clip values to [-MS, MS] range
        x_clipped = torch.clamp(x, -self.max_scale, self.max_scale)
        
        # Calculate bin indices: (batch, num_vars, seq_len)
        # Formula: y = (x + MS) / (2*MS) * H, then clip to [0, H-1]
        bin_indices = ((x_clipped + self.max_scale) / (2 * self.max_scale) * self.height)
        bin_indices = torch.clamp(bin_indices.long(), 0, self.height - 1)

        height_range = torch.arange(self.height, device=x.device).view(1, 1, self.height, 1)
        filled = (height_range <= bin_indices.unsqueeze(2)).float()
        image = filled

        logger.debug(f"TimeSeriesTo2D: input {x.shape} -> output {image.shape}")
        return image
    
    def _decode_expectation_from_occupancy(
        self,
        cdf_map: torch.Tensor,
        sharpen_temp: Optional[float] = None,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Decode occupancy map via vertical gradient → normalized mass → expected bin index."""
        # Ensure valid CDF range
        cdf_map = torch.clamp(cdf_map, 0.0, 1.0)
        
        # Pad a zero row at the top so the final drop to 0 is captured
        cdf_padded = torch.cat(
            [cdf_map, torch.zeros_like(cdf_map[:, :, :1, :])],
            dim=2
        )
        
        # PDF is the positive drop between adjacent rows (bottom -> top)
        pdf = cdf_padded[:, :, :-1, :] - cdf_padded[:, :, 1:, :]
        pdf = F.relu(pdf)
        
        # Optional sharpening temperature (temperature < 1 sharpens)
        if sharpen_temp is not None and sharpen_temp != 1.0:
            power = 1.0 / max(sharpen_temp, eps)
            pdf = torch.pow(pdf, power)
        
        # Normalize per column
        pdf_sum = pdf.sum(dim=2, keepdim=True).clamp(min=eps)
        pdf = pdf / pdf_sum
        
        # Expectation over pixel indices (0 .. H-1)
        indices = torch.arange(self.height, device=cdf_map.device, dtype=cdf_map.dtype)
        indices = indices.view(1, 1, -1, 1)
        expected_idx = (pdf * indices).sum(dim=2)  # -> (batch, num_vars, seq_len)
        
        # Map back to normalized value space using existing scalar logic
        denom = float(self.height)
        normalized = expected_idx / max(denom, eps)
        x = normalized * (2 * self.max_scale) - self.max_scale
        return x
    
    def inverse(
        self,
        image: torch.Tensor,
        cdf_decoder: str = "mean",
        expectation_sharpen_temp: Optional[float] = None,
        squeeze_univariate: bool = True,
    ) -> torch.Tensor:
        """Occupancy map (per-column values in [0,1]) → 1D normalized series.

        cdf_decoder: 'mean' (column sum → value) or 'expectation' (gradient mass → expected bin).
        expectation_sharpen_temp: optional power scaling when cdf_decoder=='expectation'.
        """
        batch_size, num_vars, height, seq_len = image.shape
        squeeze_output = squeeze_univariate and (num_vars == 1)

        if cdf_decoder == "expectation":
            x = self._decode_expectation_from_occupancy(image, expectation_sharpen_temp)
        else:
            occupancy = torch.clamp(image, min=0.0, max=1.0)
            column_sum = occupancy.sum(dim=2)
            column_sum = torch.clamp(column_sum, 0.0, float(self.height))
            normalized = column_sum / float(self.height)
            x = normalized * (2 * self.max_scale) - self.max_scale

        if squeeze_output:
            x = x.squeeze(1)
        
        logger.debug(f"TimeSeriesTo2D.inverse: input {image.shape} -> output {x.shape}")
        return x


class VerticalGaussianBlur(nn.Module):
    """Applies 1D Gaussian blur ONLY along the height (value) axis.
    
    This creates a probability density without smearing temporal patterns.
    The blur kernel is strictly vertical (H x 1).
    
    From ViTime Section 3.4.2: Gaussian blur significantly reduces sparsity
    and increases local information density.
    """
    
    def __init__(self, kernel_size: int = 31, sigma: float = 6.0):
        """
        Args:
            kernel_size: Size of the Gaussian kernel (must be odd)
            sigma: Standard deviation of the Gaussian
        """
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        
        self.kernel_size = kernel_size
        self.sigma = sigma
        
        # Create 1D Gaussian kernel
        # Shape: (kernel_size,)
        x = torch.arange(kernel_size).float() - kernel_size // 2
        gaussian_1d = torch.exp(-x**2 / (2 * sigma**2))
        gaussian_1d = gaussian_1d / gaussian_1d.sum()  # Normalize
        
        # Reshape to 2D vertical kernel: (1, 1, kernel_size, 1)
        # This applies convolution only along height (dim 2)
        kernel = gaussian_1d.view(1, 1, kernel_size, 1)
        self.register_buffer('kernel', kernel)
        
        # Padding only along height dimension
        self.padding = (0, 0, kernel_size // 2, kernel_size // 2)
        
        logger.info(f"VerticalGaussianBlur initialized: kernel_size={kernel_size}, sigma={sigma}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply vertical Gaussian blur.
        
        Args:
            x: Image tensor of shape (batch, channels, height, width)
            
        Returns:
            Blurred image of same shape
        """
        batch, channels, height, width = x.shape
        
        # Pad along height dimension only
        x_padded = F.pad(x, self.padding, mode='reflect')
        
        # Apply convolution with vertical kernel
        # Process each channel separately (groups=channels)
        kernel = self.kernel.expand(channels, 1, self.kernel_size, 1)
        
        # Custom convolution for non-square kernel
        # We use conv2d with the vertical kernel
        blurred = F.conv2d(x_padded, kernel, groups=channels)
        
        logger.debug(f"VerticalGaussianBlur: input {x.shape} -> output {blurred.shape}")
        return blurred



