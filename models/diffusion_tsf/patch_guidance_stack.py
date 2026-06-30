"""Trainable decoder + patch mixer stack for guidance finetune."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.diffusion_tsf.backbones.mmpd_decoder import MMPDDecoder, MMPDDecoderConfig
from models.diffusion_tsf.patch_context_mixer import PatchContextMixer, PatchContextMixerConfig


@dataclass
class PatchGuidanceStackConfig:
    in_len: int
    out_len: int
    patch_size: int
    data_dim: int
    decoder_d_model: int = 256
    decoder_d_ff: int = 512
    decoder_n_heads: int = 4
    decoder_d_layers: int = 2
    decoder_dropout: float = 0.2
    mixer_d_model: int = 512
    mixer_n_layers: int = 4
    mixer_n_heads: int = 8
    mixer_d_ff: int = 512
    mixer_dropout: float = 0.1
    context_dim: int = 256

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PatchGuidanceStack(nn.Module):
    def __init__(self, config: PatchGuidanceStackConfig):
        super().__init__()
        self.config = config
        dec_cfg = MMPDDecoderConfig(
            in_len=config.in_len,
            out_len=config.out_len,
            patch_size=config.patch_size,
            data_dim=config.data_dim,
            d_model=config.decoder_d_model,
            d_ff=config.decoder_d_ff,
            n_heads=config.decoder_n_heads,
            d_layers=config.decoder_d_layers,
            dropout=config.decoder_dropout,
        )
        self.decoder = MMPDDecoder(dec_cfg)
        mixer_cfg = PatchContextMixerConfig(
            d_in=config.decoder_d_model,
            d_model=config.mixer_d_model,
            d_out=config.context_dim,
            n_layers=config.mixer_n_layers,
            n_heads=config.mixer_n_heads,
            d_ff=config.mixer_d_ff,
            dropout=config.mixer_dropout,
            max_variates=max(config.data_dim, 512),
        )
        self.mixer = PatchContextMixer(mixer_cfg)
        self.token_recon = nn.Linear(config.context_dim, config.decoder_d_model)

    def encode_context(self, past: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        past_tokens = self.decoder.encode_past_tokens(past)
        return self.mixer(past_tokens)

    def forecast(self, past: torch.Tensor) -> torch.Tensor:
        return self.decoder.forecast(past)

    def finetune_loss(
        self,
        past: torch.Tensor,
        target: torch.Tensor,
        *,
        recon_weight: float = 0.1,
    ) -> torch.Tensor:
        """MSE forecast + light past-token recon so mixer gets gradients."""
        forecast = self.forecast(past)
        loss = F.mse_loss(forecast, target)
        past_tokens = self.decoder.encode_past_tokens(past)
        mixed, _ = self.mixer(past_tokens)
        flat_tokens = past_tokens.reshape(past.shape[0], -1, past_tokens.shape[-1])
        recon = self.token_recon(mixed)
        loss = loss + recon_weight * F.mse_loss(recon, flat_tokens)
        return loss
