"""
Preprocessing utilities for Diffusion TSF.

Handles:
1. 2D CDF occupancy map encoding
2. Vertical Gaussian blur
3. Inverse mapping (2D CDF back to 1D)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TimeSeriesTo2D(nn.Module):
    """Maps 1D time series to 2D CDF occupancy images.

    For each time step t, calculates the bin index and fills all pixels
    at or below that index (bottom-up CDF fill):
        column[y] = 1  if y <= bin_index(x_t)
                    0  otherwise

    The resulting image is a monotone-decreasing step function per column,
    fully saturated at the bottom and zeroing out above the data value.
    """

    def __init__(
        self,
        height: int = 128,
        max_scale: float = 3.5,
        # representation_mode param kept for checkpoint compat but ignored
        representation_mode: str = "cdf",
    ):
        """
        Args:
            height: H — number of bins (vertical resolution)
            max_scale: z-score values beyond [-MS, MS] are clipped
        """
        super().__init__()
        self.height = height
        self.max_scale = max_scale
        self.representation_mode = "cdf"  # always cdf

        bin_width = (2 * max_scale) / height
        bin_centers = torch.tensor([
            (j + 0.5) * bin_width - max_scale
            for j in range(height)
        ])
        self.register_buffer('bin_centers', bin_centers)

        logger.info(f"TimeSeriesTo2D initialized: H={height}, MS={max_scale}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert 1D time series to 2D CDF occupancy image.

        Univariate:    (batch, seq_len) → (batch, 1, height, seq_len)
        Multivariate:  (batch, V, seq_len) → (batch, V, height, seq_len)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        batch_size, num_vars, seq_len = x.shape

        x_clipped = torch.clamp(x, -self.max_scale, self.max_scale)
        bin_indices = ((x_clipped + self.max_scale) / (2 * self.max_scale) * self.height)
        bin_indices = torch.clamp(bin_indices.long(), 0, self.height - 1)

        # CDF fill: pixel (b, v, h, t) = 1 iff h <= bin_index
        height_range = torch.arange(self.height, device=x.device).view(1, 1, self.height, 1)
        image = (height_range <= bin_indices.unsqueeze(2)).float()

        logger.debug(f"TimeSeriesTo2D: input {x.shape} -> output {image.shape}")
        return image

    def _decode_cdf_pdf_expectation(
        self,
        cdf_map: torch.Tensor,
        pdf_temperature: Optional[float] = None,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """Decode CDF map via PDF expectation (CDF gradient → expectation)."""
        cdf_map = torch.clamp(cdf_map, 0.0, 1.0)
        cdf_padded = torch.cat(
            [cdf_map, torch.zeros_like(cdf_map[:, :, :1, :])],
            dim=2
        )
        pdf = cdf_padded[:, :, :-1, :] - cdf_padded[:, :, 1:, :]
        pdf = F.relu(pdf)

        if pdf_temperature is not None and pdf_temperature != 1.0:
            power = 1.0 / max(pdf_temperature, eps)
            pdf = torch.pow(pdf, power)

        pdf_sum = pdf.sum(dim=2, keepdim=True).clamp(min=eps)
        pdf = pdf / pdf_sum

        indices = torch.arange(self.height, device=cdf_map.device, dtype=cdf_map.dtype)
        indices = indices.view(1, 1, -1, 1)
        expected_idx = (pdf * indices).sum(dim=2)

        denom = float(self.height)
        normalized = expected_idx / max(denom, eps)
        x = normalized * (2 * self.max_scale) - self.max_scale
        return x

    def inverse(
        self,
        image: torch.Tensor,
        cdf_decoder: str = "mean",
        pdf_temperature: Optional[float] = None,
        squeeze_univariate: bool = True
    ) -> torch.Tensor:
        """Convert 2D CDF occupancy map back to 1D time series.

        Univariate:    (batch, 1, height, seq_len) → (batch, seq_len) if squeeze_univariate
        Multivariate:  (batch, V, height, seq_len) → (batch, V, seq_len)

        Args:
            image: CDF image in [0, 1] range
            cdf_decoder: 'mean' (column sum) or 'pdf_expectation' (gradient-based)
            pdf_temperature: sharpening for pdf_expectation mode
            squeeze_univariate: squeeze output for univariate case
        """
        batch_size, num_vars, height, seq_len = image.shape
        squeeze_output = squeeze_univariate and (num_vars == 1)

        if cdf_decoder == "pdf_expectation":
            x = self._decode_cdf_pdf_expectation(image, pdf_temperature)
        else:
            # default: column sum maps occupancy → value
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

    Preserves temporal sharpness while smoothing value-axis bin boundaries.
    Kernel is strictly vertical (H x 1).
    """

    def __init__(self, kernel_size: int = 31, sigma: float = 6.0):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.kernel_size = kernel_size
        self.sigma = sigma

        x = torch.arange(kernel_size).float() - kernel_size // 2
        gaussian_1d = torch.exp(-x**2 / (2 * sigma**2))
        gaussian_1d = gaussian_1d / gaussian_1d.sum()

        kernel = gaussian_1d.view(1, 1, kernel_size, 1)
        self.register_buffer('kernel', kernel)
        self.padding = (0, 0, kernel_size // 2, kernel_size // 2)

        logger.info(f"VerticalGaussianBlur initialized: kernel_size={kernel_size}, sigma={sigma}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        x_padded = F.pad(x, self.padding, mode='reflect')
        kernel = self.kernel.expand(channels, 1, self.kernel_size, 1)
        blurred = F.conv2d(x_padded, kernel, groups=channels)
        logger.debug(f"VerticalGaussianBlur: input {x.shape} -> output {blurred.shape}")
        return blurred
