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

    def encode_dual_heights_bounded(
        self,
        x: torch.Tensor,
        *,
        coarse_height: int,
        fine_height: int,
        value_min: float = 0.0,
        value_max_per_variate: torch.Tensor | list[float] | float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode coarse/fine CDF maps for values in [value_min, value_max] per variate."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        b, n_vars, seq_len = x.shape
        if isinstance(value_max_per_variate, (int, float)):
            vmax = torch.full((n_vars,), float(value_max_per_variate), device=x.device, dtype=x.dtype)
        elif isinstance(value_max_per_variate, list):
            vmax = torch.tensor(value_max_per_variate, device=x.device, dtype=x.dtype)
        else:
            vmax = value_max_per_variate.to(device=x.device, dtype=x.dtype).reshape(-1)
        if vmax.numel() != n_vars:
            raise ValueError(f"value_max_per_variate length {vmax.numel()} != num_vars {n_vars}")

        coarse_maps = []
        fine_maps = []
        vmin = float(value_min)
        for vi in range(n_vars):
            span = float(vmax[vi].item()) - vmin
            xi = x[:, vi : vi + 1]
            if span <= 0.0:
                coarse_bin = torch.zeros((b, 1, seq_len), device=x.device, dtype=torch.long)
                coarse_maps.append(self._cdf_from_bin_indices(coarse_bin, height=coarse_height))
                fine_maps.append(torch.zeros((b, 1, fine_height, seq_len), device=x.device, dtype=xi.dtype))
                continue
            x_clip = torch.clamp(xi, vmin, vmin + span)
            coarse_pos = (x_clip - vmin) / span * coarse_height
            coarse_bin = torch.clamp(coarse_pos.long(), 0, coarse_height - 1)
            coarse = self._cdf_from_bin_indices(coarse_bin, height=coarse_height)

            coarse_width = span / coarse_height
            coarse_center = (coarse_bin.to(x_clip.dtype) + 0.5) * coarse_width + vmin
            residual_range = coarse_width * 0.5
            residual = torch.clamp(x_clip - coarse_center, -residual_range, residual_range)
            fine_pos = (residual + residual_range) / (2 * residual_range) * fine_height
            fine_bin = torch.clamp(fine_pos.long(), 0, fine_height - 1)
            fine = self._cdf_from_bin_indices(fine_bin, height=fine_height)
            coarse_maps.append(coarse)
            fine_maps.append(fine)
        return torch.cat(coarse_maps, dim=1), torch.cat(fine_maps, dim=1)

    def _decode_occupancy_bounded(
        self,
        cdf_map: torch.Tensor,
        *,
        value_min: float,
        value_max: float,
        cdf_decoder: str = "mean",
        expectation_sharpen_temp: Optional[float] = None,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        cdf_map = torch.clamp(cdf_map, 0.0, 1.0)
        height = cdf_map.shape[2]
        span = float(value_max) - float(value_min)
        if span <= 0.0:
            return torch.full(
                (cdf_map.shape[0], cdf_map.shape[1], cdf_map.shape[3]),
                float(value_min),
                device=cdf_map.device,
                dtype=cdf_map.dtype,
            )
        centers = torch.linspace(
            float(value_min) + span / (2 * height),
            float(value_max) - span / (2 * height),
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
        return (bin_idx + 0.5) / float(height) * span + float(value_min)

    def decode_dual_heights_bounded(
        self,
        coarse_map: torch.Tensor,
        fine_map: torch.Tensor,
        *,
        value_min: float = 0.0,
        value_max_per_variate: torch.Tensor | list[float] | float,
        cdf_decoder: str = "mean",
        expectation_sharpen_temp: Optional[float] = None,
        squeeze_univariate: bool = True,
    ) -> torch.Tensor:
        if coarse_map.shape[:2] != fine_map.shape[:2] or coarse_map.shape[3] != fine_map.shape[3]:
            raise ValueError(f"coarse/fine shapes differ: {coarse_map.shape} vs {fine_map.shape}")
        _batch_size, num_vars, coarse_height, _seq_len = coarse_map.shape
        if isinstance(value_max_per_variate, (int, float)):
            vmax = torch.full((num_vars,), float(value_max_per_variate), device=coarse_map.device, dtype=coarse_map.dtype)
        elif isinstance(value_max_per_variate, list):
            vmax = torch.tensor(value_max_per_variate, device=coarse_map.device, dtype=coarse_map.dtype)
        else:
            vmax = value_max_per_variate.to(device=coarse_map.device, dtype=coarse_map.dtype).reshape(-1)

        coarse_vals = []
        fine_vals = []
        vmin = float(value_min)
        for vi in range(num_vars):
            span = float(vmax[vi].item()) - vmin
            fine_range = span / float(coarse_height) * 0.5 if span > 0.0 else 0.0
            coarse_vals.append(
                self._decode_occupancy_bounded(
                    coarse_map[:, vi : vi + 1],
                    value_min=vmin,
                    value_max=vmin + span,
                    cdf_decoder=cdf_decoder,
                    expectation_sharpen_temp=expectation_sharpen_temp,
                )
            )
            fine_vals.append(
                self._decode_occupancy_bounded(
                    fine_map[:, vi : vi + 1],
                    value_min=-fine_range,
                    value_max=fine_range,
                    cdf_decoder=cdf_decoder,
                    expectation_sharpen_temp=expectation_sharpen_temp,
                )
            )
        coarse_value = torch.cat(coarse_vals, dim=1)
        fine_value = torch.cat(fine_vals, dim=1)
        x = coarse_value + fine_value
        if squeeze_univariate and num_vars == 1:
            x = x.squeeze(1)
        return x

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

    @staticmethod
    def stack_vertical_dual(coarse: torch.Tensor, fine: torch.Tensor) -> torch.Tensor:
        """Stack coarse/fine CDF maps on the height axis → (..., Hc+Hf, W)."""
        if coarse.shape[:-2] != fine.shape[:-2] or coarse.shape[-1] != fine.shape[-1]:
            raise ValueError(f"coarse/fine shapes differ: {coarse.shape} vs {fine.shape}")
        return torch.cat([coarse, fine], dim=-2)

    @staticmethod
    def split_vertical_dual(
        canvas: torch.Tensor,
        coarse_height: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split stacked canvas into coarse [:Hc] and fine [Hc:]."""
        if canvas.shape[-2] <= int(coarse_height):
            raise ValueError(
                f"canvas height {canvas.shape[-2]} must exceed coarse_height={coarse_height}"
            )
        return canvas[..., :coarse_height, :], canvas[..., coarse_height:, :]

    @staticmethod
    def bin_indices_from_cdf(cdf_map: torch.Tensor) -> torch.Tensor:
        """Hard/soft CDF → per-column bin index k (column_sum - 1), shape (*batch, W)."""
        height = int(cdf_map.shape[-2])
        column_sum = cdf_map.sum(dim=-2).clamp(1.0, float(height))
        return (column_sum - 1.0).clamp(0.0, float(height - 1))

    @staticmethod
    def cdf_distance_weights(
        target_cdf: torch.Tensor,
        alpha: float = 1.0,
        *,
        coarse_height: Optional[int] = None,
    ) -> torch.Tensor:
        """Pixel weights W=1+α|r−k| for distance-weighted BCE.

        For a stacked vertical_dual canvas, pass coarse_height so each half
        gets its own staircase distance (fine uses local row index).
        target_cdf: (..., 1 or V, H, W) or (..., H, W) — weight matches shape.
        """
        if target_cdf.dim() < 2:
            raise ValueError(f"target_cdf must be at least 2D, got {target_cdf.shape}")
        H = int(target_cdf.shape[-2])
        W = int(target_cdf.shape[-1])
        device = target_cdf.device
        dtype = target_cdf.dtype
        alpha = float(alpha)

        def _weights_one(cdf: torch.Tensor) -> torch.Tensor:
            h = int(cdf.shape[-2])
            rows = torch.arange(h, device=device, dtype=dtype).view(
                *([1] * (cdf.dim() - 2)), h, 1
            )
            gt_k = TimeSeriesTo2D.bin_indices_from_cdf(cdf).to(dtype=dtype)
            gt_k = gt_k.unsqueeze(-2)  # (..., 1, W)
            return 1.0 + alpha * (rows - gt_k).abs()

        if coarse_height is None or int(coarse_height) <= 0 or int(coarse_height) >= H:
            return _weights_one(target_cdf)

        hc = int(coarse_height)
        coarse = target_cdf[..., :hc, :]
        fine = target_cdf[..., hc:, :]
        return torch.cat([_weights_one(coarse), _weights_one(fine)], dim=-2)

    def encode_vertical_dual_heights(
        self,
        x: torch.Tensor,
        *,
        coarse_height: int,
        fine_height: int,
    ) -> torch.Tensor:
        """Encode dual-scale CDFs and stack to (B, V, Hc+Hf, W)."""
        coarse, fine = self.encode_dual_heights(
            x, coarse_height=coarse_height, fine_height=fine_height,
        )
        return self.stack_vertical_dual(coarse, fine)

    def encode_vertical_dual_heights_bounded(
        self,
        x: torch.Tensor,
        *,
        coarse_height: int,
        fine_height: int,
        value_min: float = 0.0,
        value_max_per_variate: torch.Tensor,
    ) -> torch.Tensor:
        coarse, fine = self.encode_dual_heights_bounded(
            x,
            coarse_height=coarse_height,
            fine_height=fine_height,
            value_min=value_min,
            value_max_per_variate=value_max_per_variate,
        )
        return self.stack_vertical_dual(coarse, fine)

    def decode_vertical_dual(
        self,
        canvas: torch.Tensor,
        *,
        coarse_height: int,
        cdf_decoder: str = "mean",
        expectation_sharpen_temp: Optional[float] = None,
        squeeze_univariate: bool = True,
    ) -> torch.Tensor:
        """Split stacked canvas and decode_dual."""
        coarse, fine = self.split_vertical_dual(canvas, coarse_height)
        return self.decode_dual(
            coarse,
            fine,
            cdf_decoder=cdf_decoder,
            expectation_sharpen_temp=expectation_sharpen_temp,
            squeeze_univariate=squeeze_univariate,
        )

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


