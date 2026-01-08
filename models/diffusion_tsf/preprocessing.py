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


class Standardizer:
    """Z-score standardization using local mean and standard deviation.
    
    Normalizes input windows to have zero mean and unit variance.
    Stores the normalization parameters for inverse transform.
    """
    
    def __init__(self, eps: float = 1e-8):
        """
        Args:
            eps: Small constant for numerical stability
        """
        self.eps = eps
        self.mean = None
        self.std = None
    
    def fit_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Compute mean/std from x and normalize.
        
        Args:
            x: Input tensor of shape (batch, seq_len) or (batch, channels, seq_len)
            
        Returns:
            Normalized tensor of same shape
        """
        # Handle both 2D and 3D inputs
        if x.dim() == 2:
            # (batch, seq_len) -> compute stats over seq_len
            self.mean = x.mean(dim=-1, keepdim=True)
            self.std = x.std(dim=-1, keepdim=True) + self.eps
        else:
            # (batch, channels, seq_len)
            self.mean = x.mean(dim=-1, keepdim=True)
            self.std = x.std(dim=-1, keepdim=True) + self.eps
        
        normalized = (x - self.mean) / self.std
        logger.debug(f"Standardizer: mean={self.mean.mean().item():.4f}, std={self.std.mean().item():.4f}")
        return normalized
    
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize using stored mean/std.
        
        Args:
            x: Input tensor
            
        Returns:
            Normalized tensor
        """
        if self.mean is None or self.std is None:
            raise ValueError("Standardizer must be fit before transform")
        return (x - self.mean) / self.std
    
    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize using stored mean/std.
        
        Args:
            x: Normalized tensor
            
        Returns:
            Original scale tensor
        """
        if self.mean is None or self.std is None:
            raise ValueError("Standardizer must be fit before inverse_transform")
        return x * self.std + self.mean


class TimeSeriesTo2D(nn.Module):
    """Maps 1D time series to 2D "stripe" binary images.
    
    For each time step t, calculates the bin index:
        y_t = clip((x_t / MS) * (H/2) + H/2, 0, H-1)
    
    Creates a binary image where only pixel (t, y_t) is 1.
    This is the "stripe" representation from ViTime.
    """
    
    def __init__(
        self,
        height: int = 128,
        max_scale: float = 3.5,
        representation_mode: str = "pdf"
    ):
        """
        Args:
            height: Height H of the 2D representation (number of bins)
            max_scale: MS parameter - values beyond [-MS, MS] are clipped
            representation_mode: "pdf" (stripe) or "cdf" (occupancy map)
        """
        super().__init__()
        self.height = height
        self.max_scale = max_scale
        assert representation_mode in ["pdf", "cdf"], "representation_mode must be 'pdf' or 'cdf'"
        self.representation_mode = representation_mode
        
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
        """Convert 1D time series to 2D binary stripe image.
        
        Supports both univariate and multivariate inputs:
        - Univariate: (batch, seq_len) -> (batch, 1, height, seq_len)
        - Multivariate: (batch, num_vars, seq_len) -> (batch, num_vars, height, seq_len)
        
        Args:
            x: Normalized time series of shape (batch, seq_len) or (batch, num_vars, seq_len)
            
        Returns:
            Binary image of shape (batch, num_vars, height, seq_len)
            Each column has exactly one 1 (one-hot along height) per variable
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
        
        if self.representation_mode == "pdf":
            # Create one-hot encoding along height dimension
            # Shape: (batch, num_vars, seq_len, height)
            one_hot = F.one_hot(bin_indices, num_classes=self.height).float()
            
            # Reshape to (batch, num_vars, height, seq_len)
            image = one_hot.permute(0, 1, 3, 2)
        else:
            # Occupancy/CDF: fill all pixels <= bin index
            # height_range: (1, 1, height, 1) for broadcasting
            height_range = torch.arange(self.height, device=x.device).view(1, 1, self.height, 1)
            # bin_indices: (batch, num_vars, 1, seq_len) for broadcasting
            filled = (height_range <= bin_indices.unsqueeze(2)).float()
            image = filled
        
        logger.debug(f"TimeSeriesTo2D: input {x.shape} -> output {image.shape}")
        return image
    
    def _decode_cdf_pdf_expectation(
        self,
        cdf_map: torch.Tensor,
        pdf_temperature: Optional[float] = None,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """Decode a CDF/occupancy map using a PDF expectation approach.
        
        Steps:
        1) Convert CDF → PDF via vertical gradient (drop between rows).
        2) ReLU to remove negative gradients (monotonicity violations).
        3) Normalize PDF column-wise.
        4) Compute expected pixel index and map back to value space.
        """
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
        if pdf_temperature is not None and pdf_temperature != 1.0:
            # Use a power transform to mimic temperature scaling
            power = 1.0 / max(pdf_temperature, eps)
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
        pdf_temperature: Optional[float] = None
    ) -> torch.Tensor:
        """Convert 2D probability image back to 1D time series.
        
        Uses expectation: x_t = sum_j P(j) * center(j)
        
        Supports both univariate and multivariate:
        - Univariate: (batch, 1, height, seq_len) -> (batch, seq_len)
        - Multivariate: (batch, num_vars, height, seq_len) -> (batch, num_vars, seq_len)
        
        Args:
            image: Probability image of shape (batch, num_vars, height, seq_len)
                   Each column should be a valid probability distribution
            
        Returns:
            Time series of shape (batch, num_vars, seq_len) or (batch, seq_len) if univariate
        """
        batch_size, num_vars, height, seq_len = image.shape
        squeeze_output = (num_vars == 1)
        
        if self.representation_mode == "pdf":
            # Normalize each column to be a probability distribution
            # Softmax along height (dim=2)
            prob = F.softmax(image, dim=2)
            
            # Compute expectation: sum_j P(j) * center(j)
            # bin_centers: (height,) -> (1, 1, height, 1)
            centers = self.bin_centers.view(1, 1, -1, 1)
            
            # Weighted sum: (batch, num_vars, seq_len)
            x = (prob * centers).sum(dim=2)
        else:
            if cdf_decoder == "pdf_expectation":
                x = self._decode_cdf_pdf_expectation(image, pdf_temperature)
            else:
                # Occupancy/CDF: value is the column sum mapped back to normalized range
                occupancy = torch.clamp(image, min=0.0, max=1.0)
                column_sum = occupancy.sum(dim=2)  # Sum along height
                column_sum = torch.clamp(column_sum, 0.0, float(self.height))
                normalized = column_sum / float(self.height)
                x = normalized * (2 * self.max_scale) - self.max_scale
        
        # For backwards compatibility, squeeze if univariate
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


class Preprocessing(nn.Module):
    """Complete preprocessing pipeline: normalize -> 2D encode -> blur.
    
    Combines all preprocessing steps into a single module.
    """
    
    def __init__(
        self,
        height: int = 128,
        max_scale: float = 3.5,
        blur_kernel_size: int = 31,
        blur_sigma: float = 1.0,
        representation_mode: str = "pdf"
    ):
        super().__init__()
        self.to_2d = TimeSeriesTo2D(
            height=height,
            max_scale=max_scale,
            representation_mode=representation_mode
        )
        self.blur = VerticalGaussianBlur(kernel_size=blur_kernel_size, sigma=blur_sigma)
        self.standardizer = Standardizer()
    
    def encode(self, x: torch.Tensor, fit_standardizer: bool = True) -> torch.Tensor:
        """Encode 1D time series to blurred 2D representation.
        
        Args:
            x: Time series of shape (batch, seq_len)
            fit_standardizer: If True, fit standardizer on this data
            
        Returns:
            Blurred 2D image of shape (batch, 1, height, seq_len)
        """
        # Normalize
        if fit_standardizer:
            x_norm = self.standardizer.fit_transform(x)
        else:
            x_norm = self.standardizer.transform(x)
        
        # Convert to 2D
        image = self.to_2d(x_norm)
        
        # Apply vertical blur
        blurred = self.blur(image)
        
        return blurred
    
    def decode(self, image: torch.Tensor) -> torch.Tensor:
        """Decode 2D representation back to 1D time series.
        
        Args:
            image: 2D probability image of shape (batch, 1, height, seq_len)
            
        Returns:
            Time series of shape (batch, seq_len) in original scale
        """
        # Convert to 1D (normalized scale)
        x_norm = self.to_2d.inverse(image)
        
        # Denormalize
        x = self.standardizer.inverse_transform(x_norm)
        
        return x

