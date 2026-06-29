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
        
        logger.debug("TimeSeriesTo2D initialized: H=%s, MS=%s", height, max_scale)
    
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

    def _skyline_from_bin_indices(self, bin_indices: torch.Tensor, height: Optional[int] = None) -> torch.Tensor:
        height = int(height or self.height)
        bins = torch.clamp(bin_indices.long(), 0, height - 1)
        return F.one_hot(bins, num_classes=height).permute(0, 1, 3, 2).to(dtype=torch.float32)

    def encode_skyline(
        self,
        x: torch.Tensor,
        *,
        height: Optional[int] = None,
        value_range: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """1D series → one-hot skyline and integer bin indices per column."""
        height = int(height or self.height)
        value_range = float(value_range if value_range is not None else self.max_scale)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x_clipped = torch.clamp(x, -value_range, value_range)
        pos = (x_clipped + value_range) / (2 * value_range) * height
        bin_indices = torch.clamp(pos.long(), 0, height - 1)
        skyline = self._skyline_from_bin_indices(bin_indices, height=height)
        return skyline, bin_indices

    def encode_dual_skyline_heights(
        self,
        x: torch.Tensor,
        *,
        coarse_height: int,
        fine_height: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full-range coarse skyline plus within-bin residual skyline and bin indices."""
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x_clipped = torch.clamp(x, -self.max_scale, self.max_scale)
        coarse_pos = (x_clipped + self.max_scale) / (2 * self.max_scale) * coarse_height
        coarse_bin = torch.clamp(coarse_pos.long(), 0, coarse_height - 1)
        coarse = self._skyline_from_bin_indices(coarse_bin, height=coarse_height)

        coarse_width = (2 * self.max_scale) / coarse_height
        coarse_center = (coarse_bin.to(x_clipped.dtype) + 0.5) * coarse_width - self.max_scale
        residual = x_clipped - coarse_center
        residual_range = self.max_scale / coarse_height
        residual = torch.clamp(residual, -residual_range, residual_range)
        fine_pos = (residual + residual_range) / (2 * residual_range) * fine_height
        fine_bin = torch.clamp(fine_pos.long(), 0, fine_height - 1)
        fine = self._skyline_from_bin_indices(fine_bin, height=fine_height)
        return coarse, fine, coarse_bin, fine_bin

    def decode_skyline(
        self,
        skyline: torch.Tensor,
        *,
        value_range: Optional[float] = None,
        squeeze_univariate: bool = True,
    ) -> torch.Tensor:
        """One-hot skyline → 1D values via argmax bin centers."""
        value_range = float(value_range if value_range is not None else self.max_scale)
        height = skyline.shape[2]
        bin_indices = skyline.argmax(dim=2)
        bin_width = (2 * value_range) / height
        x = (bin_indices.to(skyline.dtype) + 0.5) * bin_width - value_range
        if squeeze_univariate and skyline.shape[1] == 1:
            x = x.squeeze(1)
        return x

    def decode_dual_skyline(
        self,
        coarse_sky: torch.Tensor,
        fine_sky: torch.Tensor,
        *,
        squeeze_univariate: bool = True,
    ) -> torch.Tensor:
        """Decode coarse + fine skylines by summing decoded residual values."""
        if coarse_sky.shape[:2] != fine_sky.shape[:2] or coarse_sky.shape[3] != fine_sky.shape[3]:
            raise ValueError(f"coarse/fine shapes differ: {coarse_sky.shape} vs {fine_sky.shape}")
        coarse_height = coarse_sky.shape[2]
        coarse_value = self.decode_skyline(
            coarse_sky,
            value_range=self.max_scale,
            squeeze_univariate=False,
        )
        fine_value = self.decode_skyline(
            fine_sky,
            value_range=self.max_scale / coarse_height,
            squeeze_univariate=False,
        )
        x = coarse_value + fine_value
        x = torch.clamp(x, -self.max_scale, self.max_scale)
        if squeeze_univariate and coarse_sky.shape[1] == 1:
            x = x.squeeze(1)
        return x

    def decode_continuous_bins(
        self,
        bin_indices: torch.Tensor,
        *,
        value_range: float,
        height: int,
        squeeze_univariate: bool = True,
    ) -> torch.Tensor:
        """Map continuous bin indices (incl. fractional) to normalized 1D values."""
        bin_width = (2 * float(value_range)) / int(height)
        x = (bin_indices.to(torch.float32) + 0.5) * bin_width - float(value_range)
        if squeeze_univariate and x.dim() >= 2 and x.shape[1] == 1:
            x = x.squeeze(1)
        return x

    def decode_dual_continuous_bins(
        self,
        coarse_bins: torch.Tensor,
        fine_bins: torch.Tensor,
        *,
        coarse_height: int,
        fine_height: int,
        squeeze_univariate: bool = True,
    ) -> torch.Tensor:
        """Decode summed coarse/fine continuous bin indices to normalized values."""
        coarse_value = self.decode_continuous_bins(
            coarse_bins,
            value_range=self.max_scale,
            height=coarse_height,
            squeeze_univariate=False,
        )
        fine_value = self.decode_continuous_bins(
            fine_bins,
            value_range=self.max_scale / coarse_height,
            height=fine_height,
            squeeze_univariate=False,
        )
        x = coarse_value + fine_value
        x = torch.clamp(x, -self.max_scale, self.max_scale)
        if squeeze_univariate and coarse_bins.dim() >= 2 and coarse_bins.shape[1] == 1:
            x = x.squeeze(1)
        return x

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

    @staticmethod
    def haar_detail_levels(seq_len: int) -> int:
        """Number of dyadic Haar detail levels after padding ``seq_len`` to a power of two."""
        if seq_len < 2:
            return 0
        padded = 1 << (int(seq_len) - 1).bit_length()
        return padded.bit_length() - 1

    @staticmethod
    def _pad_time_to_pow2(x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        seq_len = int(x.shape[-1])
        if seq_len < 1:
            raise ValueError("Haar split requires a non-empty time axis")
        padded_len = 1 << (seq_len - 1).bit_length()
        pad = padded_len - seq_len
        if pad == 0:
            return x, seq_len
        return F.pad(x, (0, pad), mode="replicate"), seq_len

    def haar_frequency_split_values(
        self,
        x: torch.Tensor,
        *,
        high_freq_levels: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split a series into low/high Haar reconstructions along the time axis.

        ``high_freq_levels`` counts finest Haar detail levels. The low component
        keeps the approximation plus remaining coarser details; the high
        component keeps only the selected finest details. Components sum back to
        the padded input up to floating point error.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x_pad, original_len = self._pad_time_to_pow2(x)
        levels = self.haar_detail_levels(x_pad.shape[-1])
        if levels == 0:
            return x[..., :original_len], torch.zeros_like(x[..., :original_len])
        high_freq_levels = int(max(1, min(high_freq_levels, levels)))

        details = []
        cur = x_pad
        inv_sqrt2 = 1.0 / (2.0 ** 0.5)
        while cur.shape[-1] >= 2:
            even = cur[..., 0::2]
            odd = cur[..., 1::2]
            details.append((even - odd) * inv_sqrt2)  # finest -> coarsest
            cur = (even + odd) * inv_sqrt2
        approx = cur

        zero_approx = torch.zeros_like(approx)
        low_details = []
        high_details = []
        for level_idx, detail in enumerate(details):
            is_high = level_idx < high_freq_levels
            low_details.append(torch.zeros_like(detail) if is_high else detail)
            high_details.append(detail if is_high else torch.zeros_like(detail))

        def reconstruct(start: torch.Tensor, detail_stack: list[torch.Tensor]) -> torch.Tensor:
            rec = start
            for detail in reversed(detail_stack):
                even = (rec + detail) * inv_sqrt2
                odd = (rec - detail) * inv_sqrt2
                out = torch.empty(*even.shape[:-1], even.shape[-1] * 2, device=even.device, dtype=even.dtype)
                out[..., 0::2] = even
                out[..., 1::2] = odd
                rec = out
            return rec[..., :original_len]

        low = reconstruct(approx, low_details)
        high = reconstruct(zero_approx, high_details)
        return low, high

    def fourier_frequency_split_values(
        self,
        x: torch.Tensor,
        *,
        cutoff_bin,
        flatline_atol: float,
        edge_mode: str = "mirror_pad",
        mirror_pad_frac: float = 0.25,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split a series into low/high Fourier reconstructions along the time axis."""
        from models.diffusion_tsf.fourier_frequency import fourier_frequency_split_torch

        if x.dim() == 2:
            x = x.unsqueeze(1)
        low, high = fourier_frequency_split_torch(
            x,
            cutoff_bin=cutoff_bin,
            flatline_atol=float(flatline_atol),
            edge_mode=edge_mode,
            mirror_pad_frac=mirror_pad_frac,
        )
        if low.shape[1] == 1 and x.shape[1] == 1:
            return low.squeeze(1), high.squeeze(1)
        return low, high

    def encode_fourier_frequency_heights(
        self,
        x: torch.Tensor,
        *,
        coarse_height: int,
        fine_height: int,
        cutoff_bin,
        fine_value_range,
        flatline_atol: float,
        edge_mode: str = "mirror_pad",
        mirror_pad_frac: float = 0.25,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        low, high = self.fourier_frequency_split_values(
            x,
            cutoff_bin=cutoff_bin,
            flatline_atol=flatline_atol,
            edge_mode=edge_mode,
            mirror_pad_frac=mirror_pad_frac,
        )
        if low.dim() == 2:
            low = low.unsqueeze(1)
            high = high.unsqueeze(1)
        if isinstance(fine_value_range, (list, tuple)):
            fine_ranges = [float(v) for v in fine_value_range]
        else:
            fine_ranges = [float(fine_value_range)] * int(low.shape[1])

        coarse = self._encode_values_in_range(
            low,
            value_range=self.max_scale,
            height=coarse_height,
        )
        fine_maps = []
        for vi in range(high.shape[1]):
            fine_maps.append(
                self._encode_values_in_range(
                    high[:, vi : vi + 1],
                    value_range=fine_ranges[vi],
                    height=fine_height,
                )
            )
        fine = torch.cat(fine_maps, dim=1)
        return coarse, fine

    def encode_haar_frequency_heights(
        self,
        x: torch.Tensor,
        *,
        coarse_height: int,
        fine_height: int,
        high_freq_levels: int,
        fine_value_range: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode Haar low/high reconstructions as staged hard binary CDF maps."""
        low, high = self.haar_frequency_split_values(x, high_freq_levels=high_freq_levels)
        coarse = self._encode_values_in_range(
            low,
            value_range=self.max_scale,
            height=coarse_height,
        )
        fine = self._encode_values_in_range(
            high,
            value_range=float(fine_value_range),
            height=fine_height,
        )
        return coarse, fine

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

    def encode_triple_heights(
        self,
        x: torch.Tensor,
        *,
        coarse_height: int,
        fine_height: int,
        finer_height: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode full-range, residual, and second residual CDFs with independent heights."""
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x_clipped = torch.clamp(x, -self.max_scale, self.max_scale)
        coarse_pos = (x_clipped + self.max_scale) / (2 * self.max_scale) * coarse_height
        coarse_bin = torch.clamp(coarse_pos.long(), 0, coarse_height - 1)
        coarse = self._cdf_from_bin_indices(coarse_bin, height=coarse_height)

        coarse_width = (2 * self.max_scale) / coarse_height
        coarse_center = (coarse_bin.to(x_clipped.dtype) + 0.5) * coarse_width - self.max_scale
        fine_residual_range = self.max_scale / coarse_height
        fine_residual = torch.clamp(
            x_clipped - coarse_center,
            -fine_residual_range,
            fine_residual_range,
        )
        fine_pos = (fine_residual + fine_residual_range) / (2 * fine_residual_range) * fine_height
        fine_bin = torch.clamp(fine_pos.long(), 0, fine_height - 1)
        fine = self._cdf_from_bin_indices(fine_bin, height=fine_height)

        fine_width = (2 * fine_residual_range) / fine_height
        fine_center = (fine_bin.to(x_clipped.dtype) + 0.5) * fine_width - fine_residual_range
        finer_residual_range = fine_residual_range / fine_height
        finer_residual = torch.clamp(
            fine_residual - fine_center,
            -finer_residual_range,
            finer_residual_range,
        )
        finer_pos = (finer_residual + finer_residual_range) / (2 * finer_residual_range) * finer_height
        finer_bin = torch.clamp(finer_pos.long(), 0, finer_height - 1)
        finer = self._cdf_from_bin_indices(finer_bin, height=finer_height)
        return coarse, fine, finer
    
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

    def decode_fourier_frequency_dual(
        self,
        coarse_map: torch.Tensor,
        fine_map: torch.Tensor,
        *,
        fine_value_range: float,
        cdf_decoder: str = "mean",
        expectation_sharpen_temp: Optional[float] = None,
        squeeze_univariate: bool = True,
    ) -> torch.Tensor:
        return self.decode_haar_frequency_dual(
            coarse_map,
            fine_map,
            fine_value_range=fine_value_range,
            cdf_decoder=cdf_decoder,
            expectation_sharpen_temp=expectation_sharpen_temp,
            squeeze_univariate=squeeze_univariate,
        )

    def decode_haar_frequency_dual(
        self,
        coarse_map: torch.Tensor,
        fine_map: torch.Tensor,
        *,
        fine_value_range: float,
        cdf_decoder: str = "mean",
        expectation_sharpen_temp: Optional[float] = None,
        squeeze_univariate: bool = True,
    ) -> torch.Tensor:
        """Decode Haar low/high CDF maps by summing the component values."""
        if coarse_map.shape[:2] != fine_map.shape[:2] or coarse_map.shape[3] != fine_map.shape[3]:
            raise ValueError(f"coarse/fine shapes differ: {coarse_map.shape} vs {fine_map.shape}")
        _batch_size, num_vars, _coarse_height, _seq_len = coarse_map.shape
        coarse_value = self._decode_occupancy_in_range(
            coarse_map,
            value_range=self.max_scale,
            cdf_decoder=cdf_decoder,
            expectation_sharpen_temp=expectation_sharpen_temp,
        )
        fine_value = self._decode_occupancy_in_range(
            fine_map,
            value_range=float(fine_value_range),
            cdf_decoder=cdf_decoder,
            expectation_sharpen_temp=expectation_sharpen_temp,
        )
        x = torch.clamp(coarse_value + fine_value, -self.max_scale, self.max_scale)
        if squeeze_univariate and num_vars == 1:
            x = x.squeeze(1)
        return x

    def decode_triple(
        self,
        coarse_map: torch.Tensor,
        fine_map: torch.Tensor,
        finer_map: torch.Tensor,
        cdf_decoder: str = "mean",
        expectation_sharpen_temp: Optional[float] = None,
        squeeze_univariate: bool = True,
    ) -> torch.Tensor:
        """Decode full-range coarse plus two residual CDF maps back to normalized values."""
        if (
            coarse_map.shape[:2] != fine_map.shape[:2]
            or coarse_map.shape[:2] != finer_map.shape[:2]
            or coarse_map.shape[3] != fine_map.shape[3]
            or coarse_map.shape[3] != finer_map.shape[3]
        ):
            raise ValueError(
                f"coarse/fine/finer shapes differ: "
                f"{coarse_map.shape} vs {fine_map.shape} vs {finer_map.shape}"
            )
        batch_size, num_vars, coarse_height, seq_len = coarse_map.shape
        fine_height = fine_map.shape[2]

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
        finer_value = self._decode_occupancy_in_range(
            finer_map,
            value_range=self.max_scale / (coarse_height * fine_height),
            cdf_decoder=cdf_decoder,
            expectation_sharpen_temp=expectation_sharpen_temp,
        )
        x = coarse_value + fine_value + finer_value
        x = torch.clamp(x, -self.max_scale, self.max_scale)
        if squeeze_univariate and num_vars == 1:
            x = x.squeeze(1)
        logger.debug(
            "TimeSeriesTo2D.decode_triple: input %s/%s/%s -> output %s",
            coarse_map.shape,
            fine_map.shape,
            finer_map.shape,
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


