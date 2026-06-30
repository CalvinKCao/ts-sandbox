"""Mix per-channel past patch tokens with cross-variate self-attention."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


@dataclass
class PatchContextMixerConfig:
    d_in: int = 256
    d_model: int = 512
    d_out: int = 256
    n_layers: int = 4
    n_heads: int = 8
    d_ff: int = 512
    dropout: float = 0.1
    max_variates: int = 512
    max_past_patches: int = 64


class PatchContextMixer(nn.Module):
    """Self-attention over flattened (variate, past_patch) tokens."""

    def __init__(self, config: PatchContextMixerConfig):
        super().__init__()
        self.config = config
        self.in_proj = nn.Linear(config.d_in, config.d_model)
        self.channel_embed = nn.Embedding(config.max_variates, config.d_model)
        self.patch_slot_embed = nn.Embedding(config.max_past_patches, config.d_model)
        layer = TransformerEncoderLayer(
            config.d_model,
            config.n_heads,
            config.d_ff,
            config.dropout,
            batch_first=True,
        )
        self.encoder = TransformerEncoder(layer, config.n_layers)
        self.out_proj = nn.Linear(config.d_model, config.d_out)
        self.norm = nn.LayerNorm(config.d_out)

    def forward(
        self,
        past_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            past_tokens: (B, V, N_past, d_in)

        Returns:
            mixed: (B, M, d_out) where M = V * N_past
            token_variate_ids: (M,) long, variate index per token
        """
        batch_size, num_variates, num_patches, _ = past_tokens.shape
        device = past_tokens.device
        flat = past_tokens.reshape(batch_size, num_variates * num_patches, -1)
        x = self.in_proj(flat)

        var_ids = torch.arange(num_variates, device=device).repeat_interleave(num_patches)
        patch_ids = torch.arange(num_patches, device=device).repeat(num_variates)
        x = x + self.channel_embed(var_ids)[None, :, :] + self.patch_slot_embed(patch_ids)[None, :, :]

        x = self.encoder(x)
        x = self.norm(self.out_proj(x))
        return x, var_ids
