"""Differentiable soft Gaussian renderer: 1D series -> vertical soft CDF maps."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftGaussianRenderer(nn.Module):
    """Map normalized 1D values to soft occupancy via vertical Gaussians.

    Each time column gets a Gaussian over value bins (mean = series value).
    Learned sigma controls vertical blur only; time axis stays sharp.
    """

    def __init__(self, height: int, max_scale: float = 3.5):
        super().__init__()
        self.height = height
        self.max_scale = max_scale
        self.log_sigma = nn.Parameter(torch.tensor(math.log(0.15)))

        bin_width = (2 * max_scale) / height
        bin_centers = torch.tensor(
            [(j + 0.5) * bin_width - max_scale for j in range(height)],
            dtype=torch.float32,
        )
        self.register_buffer("bin_centers", bin_centers)

    @property
    def sigma(self) -> torch.Tensor:
        return F.softplus(self.log_sigma) + 1e-4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, V, L) or (B, L) -> soft CDF (B, V, H, L) in [0, 1]."""
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x_clipped = torch.clamp(x, -self.max_scale, self.max_scale)
        sigma = self.sigma
        # (B, V, 1, L) vs (1, 1, H, 1) -> (B, V, H, L)
        diff = self.bin_centers.view(1, 1, -1, 1) - x_clipped.unsqueeze(2)
        pdf = torch.exp(-0.5 * (diff / sigma) ** 2)
        pdf = pdf / pdf.sum(dim=2, keepdim=True).clamp(min=1e-8)
        cdf = pdf.cumsum(dim=2)
        top = cdf[..., -1:, :].clamp(min=1e-8)
        cdf = cdf / top
        return cdf

    def to_diffusion_range(self, cdf: torch.Tensor) -> torch.Tensor:
        """Map [0, 1] CDF to [-1, 1] for diffusion-style tensors."""
        return cdf.clamp(0.0, 1.0) * 2.0 - 1.0
