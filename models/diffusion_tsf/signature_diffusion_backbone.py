"""Transformer denoiser over log-signature patch tokens (SimDiff-inspired layout)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PatchTokenTransformer(nn.Module):
    """Denoise future patch latents conditioned on past patch latents + diffusion time."""

    def __init__(
        self,
        latent_dim: int,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        max_tokens: int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.in_proj = nn.Linear(latent_dim, d_model)
        self.time_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, latent_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_tokens, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward(
        self,
        noisy_future: torch.Tensor,
        timesteps: torch.Tensor,
        cond_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            noisy_future: [B, P, D]
            timesteps: [B] integer diffusion indices
            cond_tokens: [B, Pc, D] encoded past patches
        Returns:
            predicted clean future latent [B, P, D]
        """
        batch, n_future, _ = noisy_future.shape
        _, n_cond, _ = cond_tokens.shape

        t_emb = self.time_proj(timesteps.float().view(batch, 1, 1))
        fut = self.in_proj(noisy_future) + t_emb
        cond = self.in_proj(cond_tokens) + t_emb

        tokens = torch.cat([cond, fut], dim=1)
        n_tok = tokens.size(1)
        if n_tok > self.pos_emb.size(1):
            raise ValueError(f"token count {n_tok} exceeds max_tokens {self.pos_emb.size(1)}")
        tokens = tokens + self.pos_emb[:, :n_tok, :]
        encoded = self.encoder(tokens)
        future_out = encoded[:, n_cond:, :]
        return self.out_proj(future_out)
