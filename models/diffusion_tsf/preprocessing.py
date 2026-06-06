"""
Preprocessing utilities for Diffusion TSF.

2D encoding (stripe / hard CDF maps) and inverse mapping back to 1D.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class TimeSeriesTo2D(nn.Module):
    """Maps 1D time series to 2D occupancy (CDF-style) images along the value axis.

    For each time step, bins the normalized value and fills all rows from the bottom
    up to that bin — a soft cumulative / bar-stack view (not a one-hot stripe).
    """

    def __init__(self, height: int = 32, max_scale: float = 3.5):
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

    def _cdf_from_bin_indices(self, bin_indices: torch.Tensor, height: Optional[int] = None) -> torch.Tensor:
        height = int(height or self.height)
        height_range = torch.arange(height, device=bin_indices.device).view(1, 1, height, 1)
        return (height_range <= bin_indices.unsqueeze(2)).float()

    def _encode_values_in_range(
        self,
        x: torch.Tensor,
        *,
        value_range: float,
        height: int,
    ) -> torch.Tensor:
        x_clipped = torch.clamp(x, -value_range, value_range)
        pos = (x_clipped + value_range) / (2 * value_range) * height
        bin_indices = torch.clamp(pos.long(), 0, height - 1)
        return self._cdf_from_bin_indices(bin_indices, height=height)

    def encode_dual(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode values as full-range coarse CDF plus within-bin residual CDF."""
        return self.encode_dual_heights(x, coarse_height=self.height, fine_height=self.height)

    def encode_dual_heights(
        self,
        x: torch.Tensor,
        *,
        coarse_height: int,
        fine_height: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode full-range coarse and within-bin residual CDFs with independent heights."""
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x_clipped = torch.clamp(x, -self.max_scale, self.max_scale)
        coarse_pos = (x_clipped + self.max_scale) / (2 * self.max_scale) * coarse_height
        coarse_bin = torch.clamp(coarse_pos.long(), 0, coarse_height - 1)
        coarse = self._cdf_from_bin_indices(coarse_bin, height=coarse_height)

        coarse_width = (2 * self.max_scale) / coarse_height
        coarse_center = (coarse_bin.to(x_clipped.dtype) + 0.5) * coarse_width - self.max_scale
        residual = x_clipped - coarse_center
        residual_range = self.max_scale / coarse_height
        residual = torch.clamp(residual, -residual_range, residual_range)
        fine_pos = (residual + residual_range) / (2 * residual_range) * fine_height
        fine_bin = torch.clamp(fine_pos.long(), 0, fine_height - 1)
        fine = self._cdf_from_bin_indices(fine_bin, height=fine_height)
        return coarse, fine
    
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
        
        height = cdf_map.shape[2]
        # Expectation over pixel indices (0 .. H-1)
        indices = torch.arange(height, device=cdf_map.device, dtype=cdf_map.dtype)
        indices = indices.view(1, 1, -1, 1)
        expected_idx = (pdf * indices).sum(dim=2)  # -> (batch, num_vars, seq_len)
        
        # Map back to normalized value space using existing scalar logic
        denom = float(height)
        normalized = expected_idx / max(denom, eps)
        x = normalized * (2 * self.max_scale) - self.max_scale
        return x

    def _decode_occupancy_in_range(
        self,
        cdf_map: torch.Tensor,
        value_range: float,
        cdf_decoder: str = "mean",
        expectation_sharpen_temp: Optional[float] = None,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        cdf_map = torch.clamp(cdf_map, 0.0, 1.0)
        height = cdf_map.shape[2]
        centers = torch.linspace(
            -value_range + value_range / height,
            value_range - value_range / height,
            height,
            device=cdf_map.device,
            dtype=cdf_map.dtype,
        ).view(1, 1, -1, 1)

        if cdf_decoder in ("expectation", "pdf_expectation"):
            cdf_padded = torch.cat(
                [cdf_map, torch.zeros_like(cdf_map[:, :, :1, :])],
                dim=2,
            )
            pdf = F.relu(cdf_padded[:, :, :-1, :] - cdf_padded[:, :, 1:, :])
            if expectation_sharpen_temp is not None and expectation_sharpen_temp != 1.0:
                power = 1.0 / max(expectation_sharpen_temp, eps)
                pdf = torch.pow(pdf, power)
            pdf = pdf / pdf.sum(dim=2, keepdim=True).clamp(min=eps)
            return (pdf * centers).sum(dim=2)

        if cdf_decoder != "mean":
            raise ValueError(f"Unknown dual CDF decoder '{cdf_decoder}'")

        column_sum = cdf_map.sum(dim=2).clamp(1.0, float(height))
        bin_idx = (column_sum - 1.0).clamp(0.0, float(height - 1))
        return ((bin_idx + 0.5) / float(height) * (2 * value_range)) - value_range

    def decode_dual(
        self,
        coarse_map: torch.Tensor,
        fine_map: torch.Tensor,
        cdf_decoder: str = "mean",
        expectation_sharpen_temp: Optional[float] = None,
        squeeze_univariate: bool = True,
    ) -> torch.Tensor:
        """Decode full-range coarse CDF plus residual CDF back to normalized values."""
        if coarse_map.shape[:2] != fine_map.shape[:2] or coarse_map.shape[3] != fine_map.shape[3]:
            raise ValueError(f"coarse/fine shapes differ: {coarse_map.shape} vs {fine_map.shape}")
        batch_size, num_vars, coarse_height, seq_len = coarse_map.shape

        coarse_value = self._decode_occupancy_in_range(
            coarse_map,
            value_range=self.max_scale,
            cdf_decoder=cdf_decoder,
            expectation_sharpen_temp=expectation_sharpen_temp,
        )
        fine_value = self._decode_occupancy_in_range(
            fine_map,
            value_range=self.max_scale / coarse_height,
            cdf_decoder=cdf_decoder,
            expectation_sharpen_temp=expectation_sharpen_temp,
        )
        x = coarse_value + fine_value
        x = torch.clamp(x, -self.max_scale, self.max_scale)
        if squeeze_univariate and num_vars == 1:
            x = x.squeeze(1)
        logger.debug(
            "TimeSeriesTo2D.decode_dual: input %s/%s -> output %s",
            coarse_map.shape,
            fine_map.shape,
            x.shape,
        )
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
            column_sum = torch.clamp(column_sum, 0.0, float(height))
            normalized = column_sum / float(height)
            x = normalized * (2 * self.max_scale) - self.max_scale

        if squeeze_output:
            x = x.squeeze(1)
        
        logger.debug(f"TimeSeriesTo2D.inverse: input {image.shape} -> output {x.shape}")
        return x


