"""PatchGAN regularizers for iTransformer forecasts."""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _dilations_for_receptive_field(receptive_field: int) -> List[int]:
    """Return dilations giving RFs close to 8, 16, or 32 with kernel size 3."""
    choices = {
        8: [1, 2],
        16: [1, 2, 4],
        32: [1, 2, 4, 8],
    }
    if receptive_field not in choices:
        raise ValueError(
            f"Unsupported PatchGAN receptive field {receptive_field}; "
            f"expected one of {sorted(choices)}."
        )
    return choices[receptive_field]


class SoftCDFBinning(nn.Module):
    """Differentiably render a 1D forecast as a per-column CDF occupancy map.

    For every scalar value x_t, the layer computes a soft assignment over fixed
    value-bin centers using squared distance logits, then cumulatively sums the
    mass from low to high bins. The result is differentiable with respect to the
    input series and, optionally, the temperature.

    Input:  ``(batch, variates, time)``
    Output: ``(batch, variates, bins, time)``
    """

    def __init__(
        self,
        num_bins: int = 32,
        value_min: float = -3.5,
        value_max: float = 3.5,
        temperature: float = 0.1,
        learnable_temperature: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if num_bins < 2:
            raise ValueError("num_bins must be at least 2")
        if value_min >= value_max:
            raise ValueError("value_min must be smaller than value_max")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        centers = torch.linspace(value_min, value_max, num_bins)
        self.register_buffer("bin_centers", centers.view(1, 1, num_bins, 1))
        self.eps = eps

        init_log_temp = math.log(float(temperature))
        if learnable_temperature:
            self.log_temperature = nn.Parameter(torch.tensor(init_log_temp))
        else:
            self.register_buffer("log_temperature", torch.tensor(init_log_temp))

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp_min(self.eps)

    def forward(self, series: torch.Tensor) -> torch.Tensor:
        if series.dim() == 2:
            series = series.unsqueeze(1)
        if series.dim() != 3:
            raise ValueError(
                "SoftCDFBinning expects series shaped (batch, variates, time) "
                f"or (batch, time), got {tuple(series.shape)}"
            )

        x = series.unsqueeze(2)
        logits = -((x - self.bin_centers.to(dtype=x.dtype, device=x.device)) ** 2)
        logits = logits / self.temperature.to(dtype=x.dtype, device=x.device)
        mass = torch.softmax(logits, dim=2)
        return mass.cumsum(dim=2)


class PatchGAN1D(nn.Module):
    """Temporal PatchGAN discriminator for forecasts shaped ``(B, V, T)``."""

    def __init__(
        self,
        in_channels: int,
        receptive_field: int = 16,
        base_channels: int = 64,
        max_channels: int = 256,
        use_spectral_norm: bool = True,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        channels = in_channels
        out_channels = base_channels

        for dilation in _dilations_for_receptive_field(receptive_field):
            conv = nn.Conv1d(
                channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            )
            if use_spectral_norm:
                conv = nn.utils.spectral_norm(conv)
            layers.extend([conv, nn.LeakyReLU(0.2, inplace=True)])
            channels = out_channels
            out_channels = min(out_channels * 2, max_channels)

        final = nn.Conv1d(channels, 1, kernel_size=3, padding=1)
        if use_spectral_norm:
            final = nn.utils.spectral_norm(final)
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, series: torch.Tensor) -> torch.Tensor:
        if series.dim() == 2:
            series = series.unsqueeze(1)
        if series.dim() != 3:
            raise ValueError(f"PatchGAN1D expects (B, V, T), got {tuple(series.shape)}")
        return self.net(series)


class PatchGAN2D(nn.Module):
    """Spatial PatchGAN discriminator for CDF maps shaped ``(B, V, H, T)``."""

    def __init__(
        self,
        in_channels: int,
        receptive_field: int = 16,
        base_channels: int = 64,
        max_channels: int = 256,
        use_spectral_norm: bool = True,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        channels = in_channels
        out_channels = base_channels

        for dilation in _dilations_for_receptive_field(receptive_field):
            conv = nn.Conv2d(
                channels,
                out_channels,
                kernel_size=3,
                padding=(1, dilation),
                dilation=(1, dilation),
            )
            if use_spectral_norm:
                conv = nn.utils.spectral_norm(conv)
            layers.extend([conv, nn.LeakyReLU(0.2, inplace=True)])
            channels = out_channels
            out_channels = min(out_channels * 2, max_channels)

        final = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        if use_spectral_norm:
            final = nn.utils.spectral_norm(final)
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, maps: torch.Tensor) -> torch.Tensor:
        if maps.dim() != 4:
            raise ValueError(f"PatchGAN2D expects (B, V, H, T), got {tuple(maps.shape)}")
        return self.net(maps)


def discriminator_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    """Non-saturating PatchGAN discriminator loss."""
    real_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
    fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    return 0.5 * (real_loss + fake_loss)


def generator_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    """Non-saturating generator loss."""
    return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad_(requires_grad)
