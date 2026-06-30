"""MMPD DecoderOnlyTransformer vendored for patch-token guidance."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import TransformerEncoder, TransformerEncoderLayer


@dataclass
class MMPDDecoderConfig:
    in_len: int
    out_len: int
    patch_size: int
    data_dim: int
    d_model: int = 256
    d_ff: int = 512
    n_heads: int = 4
    d_layers: int = 2
    dropout: float = 0.2


class _LearnablePositionEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        self.d_model = d_model
        self.position_embedding = nn.Parameter(torch.randn(max_len, d_model))

    def forward(self, seq_idxs: torch.Tensor) -> torch.Tensor:
        batch_size = seq_idxs.shape[0]
        table = self.position_embedding[None, :, :].expand(batch_size, -1, -1)
        idxs = seq_idxs[:, :, None].expand(-1, -1, self.d_model)
        return table.gather(1, idxs)


class _TransformerStack(nn.Module):
    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        layer = TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout, batch_first=True
        )
        self.layers = TransformerEncoder(layer, n_layers)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.layers(tokens)


class MMPDDecoder(nn.Module):
    """Patch-based decoder backbone (per-channel temporal attention)."""

    def __init__(self, config: MMPDDecoderConfig):
        super().__init__()
        self.config = config
        self.in_len = config.in_len
        self.out_len = config.out_len
        self.patch_size = config.patch_size
        self.data_dim = config.data_dim
        self.d_model = config.d_model

        self.patch_embedding = nn.Linear(self.patch_size, self.d_model, bias=False)
        self.position_embedding = _LearnablePositionEmbedding(self.d_model)
        self.learnable_patch = nn.Parameter(torch.randn(self.d_model))
        self.decoder = _TransformerStack(
            config.d_layers,
            config.d_model,
            config.n_heads,
            config.d_ff,
            config.dropout,
        )
        self.patch_readout = nn.Linear(self.d_model, self.patch_size, bias=True)

    def _pad_to_patch(self, x_seq: torch.Tensor) -> torch.Tensor:
        seq_len = x_seq.shape[-1]
        point_to_pad = (self.patch_size - (seq_len % self.patch_size)) % self.patch_size
        if point_to_pad > 0:
            x_seq = torch.cat([x_seq[..., :1].expand(-1, -1, point_to_pad), x_seq], dim=-1)
        return x_seq

    def _patchify(self, x_seq: torch.Tensor) -> tuple[torch.Tensor, int]:
        """x_seq: (B, V, L) -> patch_seq (B*V, N, P), N past patch count."""
        batch_size, data_dim, _ = x_seq.shape
        x_seq = self._pad_to_patch(x_seq)
        past_patch_num = x_seq.shape[-1] // self.patch_size
        x_flat = rearrange(x_seq, "b d l -> (b d) l")
        patch_seq = rearrange(x_flat, "b (n p) -> b n p", p=self.patch_size)
        return patch_seq, past_patch_num

    def encode_past_tokens(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Return past patch tokens (B, V, N_past, d_model)."""
        batch_size, data_dim, _ = x_seq.shape
        patch_seq, past_patch_num = self._patchify(x_seq)
        flatten_batch = patch_seq.shape[0]
        patch_embed = self.patch_embedding(patch_seq)
        in_idxs = torch.arange(past_patch_num, device=x_seq.device)[None, :].expand(
            flatten_batch, -1
        )
        pos_embed = self.position_embedding(in_idxs)
        tokens = patch_embed + pos_embed
        encoded = self.decoder(tokens)
        return rearrange(
            encoded,
            "(batch_size data_dim) patch_num d_model -> batch_size data_dim patch_num d_model",
            batch_size=batch_size,
            data_dim=data_dim,
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Future patch latents (B, V, N_fut, d_model)."""
        batch_size, data_dim, _ = x_seq.shape
        output_patch_num = math.ceil(self.out_len / self.patch_size)

        patch_seq, past_patch_num = self._patchify(x_seq)
        flatten_batch = patch_seq.shape[0]
        patch_embed = self.patch_embedding(patch_seq)
        in_idxs = torch.arange(past_patch_num, device=x_seq.device)[None, :].expand(
            flatten_batch, -1
        )
        in_pos_embed = self.position_embedding(in_idxs)
        input_embed = patch_embed + in_pos_embed

        out_idxs = torch.arange(
            past_patch_num, past_patch_num + output_patch_num, device=x_seq.device
        )[None, :].expand(flatten_batch, -1)
        out_pos_embed = self.position_embedding(out_idxs)
        out_patch_embed = out_pos_embed + self.learnable_patch[None, None, :]
        dec_in = torch.cat([input_embed, out_patch_embed], dim=1)
        dec_out = self.decoder(dec_in)[:, -output_patch_num:, :]
        return rearrange(
            dec_out,
            "(batch_size data_dim) out_len d_model -> batch_size data_dim out_len d_model",
            data_dim=data_dim,
        )

    def latents_to_series(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode patch latents to 1D series (B, V, out_len)."""
        patches = self.patch_readout(latents)
        series = rearrange(patches, "b v n p -> b v (n p)")
        return series[..., : self.out_len]

    def forecast(self, x_seq: torch.Tensor) -> torch.Tensor:
        return self.latents_to_series(self.forward(x_seq))
