"""DDPM in log-signature patch latent space with learned patch decoder."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.diffusion_tsf.signature_diffusion_backbone import PatchTokenTransformer
from models.diffusion_tsf.signature_latent import (
    LatentConfig,
    LogSigPatchDecoder,
    encode_series_logsig,
    fuse_point_channels,
    latent_dim,
    logsig_consistency_loss,
    overlap_add_patches,
    select_channels,
)
from models.diffusion_tsf.variate_subsets import generate_variate_subsets


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> np.ndarray:
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 1e-4, 0.999)


def _to_btc(x: torch.Tensor) -> torch.Tensor:
    """``[B, C, T]`` -> ``[B, T, C]``."""
    if x.dim() == 2:
        return x.unsqueeze(-1)
    if x.dim() != 3:
        raise ValueError(f"expected [B,C,T] or [B,T,C], got {tuple(x.shape)}")
    if x.shape[1] <= x.shape[2]:
        return x.permute(0, 2, 1)
    return x


@dataclass
class SignatureDiffusionConfig:
    n_channels: int = 7
    lookback: int = 96
    horizon: int = 96
    diff_steps: int = 100
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.1
    decoder_hidden: int = 256
    lambda_point: float = 1.0
    lambda_logsig_consistency: float = 0.25
    loss_type: str = "l1"
    latent: LatentConfig = field(default_factory=LatentConfig)
    subset_scheme: str = "all"
    subset_size: Optional[int] = None
    subset_stride: int = 1
    max_branches: int = 3
    fusion_mode: str = "precision"
    sample_steps: int = 20
    use_mom: bool = False
    mom_repeats: int = 3


class SignatureDiffusionModel(nn.Module):
    def __init__(self, cfg: SignatureDiffusionConfig):
        super().__init__()
        self.cfg = cfg
        self.n_channels = cfg.n_channels
        d_latent = latent_dim(cfg.latent, cfg.n_channels)

        self.denoiser = PatchTokenTransformer(
            latent_dim=d_latent,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            dropout=cfg.dropout,
        )
        self.decoder = LogSigPatchDecoder(
            d_latent,
            cfg.latent.patch_size,
            cfg.n_channels,
            hidden=cfg.decoder_hidden,
        )
        self._register_diffusion(cfg.diff_steps)

    def _register_diffusion(self, diff_steps: int) -> None:
        betas = cosine_beta_schedule(diff_steps)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)
        self.register_buffer("betas", torch.tensor(betas, dtype=torch.float32))
        self.register_buffer("alphas_cumprod", torch.tensor(alphas_cumprod, dtype=torch.float32))
        self.register_buffer(
            "sqrt_alphas_cumprod",
            torch.sqrt(torch.tensor(alphas_cumprod, dtype=torch.float32)),
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(torch.tensor(1.0 - alphas_cumprod, dtype=torch.float32)),
        )
        weights = np.sqrt(alphas_cumprod) / (2.0 * (1.0 - alphas_cumprod))
        if len(weights) > 1:
            weights[0] = weights[1]
        self.register_buffer("lvlb_weights", torch.tensor(weights, dtype=torch.float32))

    @property
    def diff_steps(self) -> int:
        return int(self.betas.shape[0])

    def q_sample(self, z0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        if noise is None:
            noise = torch.randn_like(z0)
        sa = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        so = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return sa * z0 + so * noise, noise

    def _latent_loss(self, pred: torch.Tensor, target: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.cfg.loss_type == "mse":
            per = F.mse_loss(pred, target, reduction="none").mean(dim=(1, 2))
        else:
            per = F.l1_loss(pred, target, reduction="none").mean(dim=(1, 2))
        w = self.lvlb_weights[t]
        return (w * per).mean()

    def _decode_horizon(self, z: torch.Tensor, horizon: int) -> torch.Tensor:
        patches = self.decoder(z)
        return overlap_add_patches(
            patches,
            horizon=horizon,
            patch_size=self.cfg.latent.patch_size,
            stride=self.cfg.latent.patch_stride,
        )

    def forward_branch(
        self,
        past_btc: torch.Tensor,
        future_btc: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        z0, _ = encode_series_logsig(future_btc, self.cfg.latent)
        cond, _ = encode_series_logsig(past_btc, self.cfg.latent)

        batch = z0.size(0)
        t = torch.randint(0, self.diff_steps, (batch,), device=z0.device, dtype=torch.long)
        z_noisy, _ = self.q_sample(z0, t)
        z_pred = self.denoiser(z_noisy, t, cond)

        loss_latent = self._latent_loss(z_pred, z0, t)
        y_hat = self._decode_horizon(z_pred, future_btc.size(1))
        loss_point = F.mse_loss(y_hat, future_btc)
        loss_cons = logsig_consistency_loss(self.decoder(z_pred), z_pred, self.cfg.latent)

        loss = (
            loss_latent
            + self.cfg.lambda_point * loss_point
            + self.cfg.lambda_logsig_consistency * loss_cons
        )
        metrics = {
            "loss": loss.detach(),
            "loss_latent": loss_latent.detach(),
            "loss_point": loss_point.detach(),
            "loss_logsig_cons": loss_cons.detach(),
        }
        return loss, metrics

    @torch.no_grad()
    def sample_branch(
        self,
        past_btc: torch.Tensor,
        horizon: int,
        *,
        n_samples: int = 1,
    ) -> torch.Tensor:
        cond, _ = encode_series_logsig(past_btc, self.cfg.latent)
        batch = past_btc.size(0)
        n_patches = cond.size(1)
        d = cond.size(2)
        device = past_btc.device

        outputs = []
        for _ in range(n_samples):
            z = torch.randn(batch, n_patches, d, device=device)
            steps = max(1, min(self.cfg.sample_steps, self.diff_steps))
            times = torch.linspace(self.diff_steps - 1, 0, steps, device=device).long()
            for i, t_val in enumerate(times):
                t = torch.full((batch,), int(t_val), device=device, dtype=torch.long)
                z_pred = self.denoiser(z, t, cond)
                if i < len(times) - 1:
                    t_prev = int(times[i + 1].item())
                    sa = self.sqrt_alphas_cumprod[t_val]
                    sa_p = self.sqrt_alphas_cumprod[t_prev]
                    so = self.sqrt_one_minus_alphas_cumprod[t_val]
                    so_p = self.sqrt_one_minus_alphas_cumprod[t_prev]
                    eps = (z - sa * z_pred) / so.clamp_min(1e-8)
                    z = sa_p * z_pred + so_p * eps
                else:
                    z = z_pred
            outputs.append(self._decode_horizon(z, horizon))

        stacked = torch.stack(outputs, dim=0)
        if self.cfg.use_mom and n_samples > 1:
            return stacked.median(dim=0).values
        return stacked.mean(dim=0)

    def forward(self, past: torch.Tensor, future: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        past_btc = _to_btc(past)
        future_btc = _to_btc(future)
        return self.forward_branch(past_btc, future_btc)

    @torch.no_grad()
    def predict(self, past: torch.Tensor, horizon: Optional[int] = None, *, n_variates: int) -> torch.Tensor:
        """Run branches that match this model's ``n_channels``; fuse in point space."""
        past_btc = _to_btc(past)
        horizon = horizon or self.cfg.horizon
        subsets = generate_variate_subsets(
            n_variates,
            scheme=self.cfg.subset_scheme,
            subset_size=self.cfg.subset_size,
            subset_stride=self.cfg.subset_stride,
            max_branches=self.cfg.max_branches,
        )
        branches: List[Tuple[Sequence[int], torch.Tensor]] = []
        for s in subsets:
            if len(s) != self.n_channels:
                continue
            past_s = select_channels(past_btc, s)
            pred = self.sample_branch(past_s, horizon, n_samples=1)
            branches.append((s, pred))

        if not branches:
            return torch.zeros(past_btc.size(0), horizon, n_variates, device=past_btc.device)

        if len(branches) == 1 and len(branches[0][0]) == n_variates:
            return branches[0][1]

        return fuse_point_channels(branches, n_channels=n_variates)
