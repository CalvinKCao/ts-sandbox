"""
Convolutional VAE for 2D time-series images (B, 1, H, W) → latent (B, C_z, H/4, W/4).
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _num_groups(ch: int, preferred: int = 8) -> int:
    g = min(preferred, ch)
    while g > 1 and ch % g != 0:
        g -= 1
    return max(1, g)


class TimeSeriesVAE(nn.Module):
    """2D conv VAE with 4× spatial compression (two stride-2 stages)."""

    def __init__(self, latent_channels: int = 4, base_channels: int = 64):
        super().__init__()
        self.latent_channels = latent_channels
        c1, c2 = base_channels, base_channels * 2
        g1, g2 = _num_groups(c1), _num_groups(c2)

        self.enc1 = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(g1, c1),
            nn.SiLU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(g2, c2),
            nn.SiLU(),
        )
        self.enc_mid = nn.Sequential(
            nn.Conv2d(c2, c2, kernel_size=3, padding=1),
            nn.GroupNorm(g2, c2),
            nn.SiLU(),
        )
        self.to_mu = nn.Conv2d(c2, latent_channels, kernel_size=1)
        self.to_logvar = nn.Conv2d(c2, latent_channels, kernel_size=1)

        self.dec_in = nn.Sequential(
            nn.Conv2d(latent_channels, c2, kernel_size=3, padding=1),
            nn.GroupNorm(g2, c2),
            nn.SiLU(),
        )
        self.dec_up1 = nn.Sequential(
            nn.ConvTranspose2d(c2, c1, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(g1, c1),
            nn.SiLU(),
        )
        self.dec_up2 = nn.Sequential(
            nn.ConvTranspose2d(c1, c1, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(g1, c1),
            nn.SiLU(),
        )
        self.dec_out = nn.Conv2d(c1, 1, kernel_size=3, padding=1)

        self.register_buffer("scale_factor", torch.tensor(1.0))

    def encode_mu_logvar(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.enc1(x)
        h = self.enc2(h)
        h = self.enc_mid(h)
        return self.to_mu(h), self.to_logvar(h)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        mu, log_var = self.encode_mu_logvar(x)
        return self.reparameterize(mu, log_var) if sample else mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.dec_in(z)
        h = self.dec_up1(h)
        h = self.dec_up2(h)
        return torch.tanh(self.dec_out(h))

    def encode_to_scaled_latent(self, x: torch.Tensor, sample: bool = False) -> torch.Tensor:
        z = self.encode(x, sample=sample)
        return z / self.scale_factor.clamp(min=1e-8)

    def decode_from_scaled_latent(self, z_scaled: torch.Tensor) -> torch.Tensor:
        z = z_scaled * self.scale_factor
        return self.decode(z)

    def forward(self, x: torch.Tensor, kl_weight: float = 1e-4) -> Dict[str, torch.Tensor]:
        mu, log_var = self.encode_mu_logvar(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        recon_loss = F.mse_loss(recon, x)
        kl = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        loss = recon_loss + kl_weight * kl
        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "kl_loss": kl,
            "mu": mu,
            "log_var": log_var,
            "z": z,
            "recon": recon,
        }


@torch.no_grad()
def estimate_vae_scale_factor(
    vae: TimeSeriesVAE,
    images: torch.Tensor,
    max_batches: int = 4,
    batch_size: int = 16,
) -> torch.Tensor:
    vae.eval()
    device = next(vae.parameters()).device
    mus = []
    n = min(images.shape[0], max_batches * batch_size)
    for i in range(0, n, batch_size):
        batch = images[i : i + batch_size].to(device)
        mu, _ = vae.encode_mu_logvar(batch)
        mus.append(mu.detach().cpu())
    stacked = torch.cat(mus, dim=0)
    sf = stacked.std().clamp(min=1e-6)
    vae.scale_factor.copy_(sf.to(vae.scale_factor.device))
    logger.info("VAE scale_factor set to %.6f", float(sf))
    return vae.scale_factor.detach()
