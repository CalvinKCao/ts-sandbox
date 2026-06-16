"""UP2ME-style MaskAE backbone for MMPD.

Adapted from Thinklab-SJTU/UP2ME (ICML 2024) forecasting fine-tune path:
encode past patches, append future learnable tokens, mix with temporal-channel
layers, return future latent tokens for the MMPD projector.

Trained end-to-end with MMPD loss (no external UP2ME pretrain checkpoint).
Normalization is handled by MMPD's exp_forecast, so RevIN is omitted here.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn import TransformerEncoder, TransformerEncoderLayer


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


class _MAEEncoder(nn.Module):
    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        layer = TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout, batch_first=True
        )
        self.encoder_layers = TransformerEncoder(layer, n_layers)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.encoder_layers(tokens)


def _batch_cosine_similarity(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    inner = torch.einsum("bqd,bkd->bqk", x, y)
    x_norm = torch.norm(x, dim=-1, keepdim=True)
    y_norm = torch.norm(y, dim=-1, keepdim=True)
    denom = torch.einsum("bqd,bkd->bqk", x_norm, y_norm)
    return inner / (denom + eps)


def _cap_graph_k(k: int, n_nodes: int) -> int:
    return max(1, min(int(k), int(n_nodes)))


def _k_nearest_neighbor(corr_matrix: torch.Tensor, k: int) -> torch.Tensor:
    batch_size, ts_d, _ = corr_matrix.shape
    k = _cap_graph_k(k, ts_d)
    edges = torch.topk(corr_matrix, k, dim=-1)[1]
    knn_adj = torch.zeros(batch_size, ts_d, ts_d, device=corr_matrix.device)
    knn_adj.scatter_(-1, edges, 1.0)
    return knn_adj.permute(0, 2, 1)


def _graph_construct(encoded_patch: torch.Tensor, k: int) -> torch.Tensor:
    # encoded_patch: [B, D, past_patch_num, d_model]
    ts_d = encoded_patch.shape[1]
    k = _cap_graph_k(k, ts_d)
    channel_encode = encoded_patch.max(dim=-2).values
    corr = _batch_cosine_similarity(channel_encode, channel_encode)
    knn_adj = _k_nearest_neighbor(corr, k)
    flat_k = min(k * ts_d, ts_d * ts_d)
    top_k_threshold = torch.topk(corr.reshape(encoded_patch.shape[0], -1), flat_k, dim=-1)[
        0
    ][:, -1]
    top_k_adj = (corr >= top_k_threshold[:, None, None]).float()
    return knn_adj * top_k_adj


class _ChannelGraphAttention(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: [BS, D, dm], adj: [BS, D, D]
        scores = torch.matmul(x, x.transpose(-2, -1)) / math.sqrt(x.shape[-1])
        scores = scores.masked_fill(adj == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = F.dropout(attn, self.dropout, training=self.training)
        return torch.matmul(attn, x)


class _TemporalChannelLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.temporal_layer = TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout, batch_first=True
        )
        self.channel_attn = _ChannelGraphAttention(d_model, dropout)
        self.channel_ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm_t = nn.LayerNorm(d_model)
        self.norm_c1 = nn.LayerNorm(d_model)
        self.norm_c2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, graph_adj: torch.Tensor) -> torch.Tensor:
        # x: [B, D, S, dm], graph_adj: [B, D, D]
        batch_size, ts_d, seq_len, d_model = x.shape
        temporal_in = x.reshape(batch_size * ts_d, seq_len, d_model)
        temporal_out = self.temporal_layer(self.norm_t(temporal_in))
        temporal_out = temporal_out.reshape(batch_size, ts_d, seq_len, d_model)

        channel_in = rearrange(
            temporal_out, "batch_size ts_d seq_len d_model -> (batch_size seq_len) ts_d d_model"
        )
        graph_adj_expand = graph_adj[:, None, :, :].expand(-1, seq_len, -1, -1)
        graph_adj_expand = rearrange(
            graph_adj_expand, "batch_size seq_len ts_d1 ts_d2 -> (batch_size seq_len) ts_d1 ts_d2"
        )
        channel_mid = self.channel_attn(self.norm_c1(channel_in), graph_adj_expand)
        channel_out = channel_in + self.dropout(channel_mid)
        channel_out = channel_out + self.dropout(self.channel_ff(self.norm_c2(channel_out)))

        return rearrange(
            channel_out,
            "(batch_size seq_len) ts_d d_model -> batch_size ts_d seq_len d_model",
            batch_size=batch_size,
        )


class MaskAETransformer(nn.Module):
    """UP2ME MaskAE backbone returning future patch tokens for MMPD."""

    def __init__(self, configs):
        super().__init__()
        self.in_len = configs.in_len
        self.out_len = configs.out_len
        self.patch_size = configs.patch_size
        self.data_dim = configs.data_dim
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.n_heads = configs.n_heads
        self.e_layers = configs.e_layers
        self.dropout = configs.dropout
        self.finetune_layers = int(
            getattr(configs, "finetune_layers", 0) or configs.d_layers
        )
        raw_neighbors = int(getattr(configs, "neighbor_num", 0) or 0)
        default_k = min(10, configs.data_dim)
        self.neighbor_num = _cap_graph_k(
            raw_neighbors if raw_neighbors > 0 else default_k,
            configs.data_dim,
        )

        self.patch_embedding = nn.Linear(self.patch_size, self.d_model, bias=False)
        self.position_embedding = _LearnablePositionEmbedding(self.d_model)
        self.channel_embedding = nn.Embedding(self.data_dim, self.d_model)
        self.encoder = _MAEEncoder(
            self.e_layers, self.d_model, self.n_heads, self.d_ff, self.dropout
        )
        self.enc_2_dec = nn.Linear(self.d_model, self.d_model)
        self.learnable_patch = nn.Parameter(torch.randn(self.d_model))
        self.tc_layers = nn.ModuleList(
            [
                _TemporalChannelLayer(
                    self.d_model, self.n_heads, self.d_ff, self.dropout
                )
                for _ in range(self.finetune_layers)
            ]
        )

    def _pad_to_patch(self, x_seq: torch.Tensor) -> tuple[torch.Tensor, int]:
        seq_len = x_seq.shape[-1]
        point_to_pad = (self.patch_size - (seq_len % self.patch_size)) % self.patch_size
        if point_to_pad > 0:
            x_seq = torch.cat([x_seq[:, :, :1].expand(-1, -1, point_to_pad), x_seq], dim=-1)
        return x_seq, point_to_pad

    def _encode_past(self, x_seq: torch.Tensor) -> tuple[torch.Tensor, int]:
        batch_size, data_dim, _ = x_seq.shape
        x_seq, _ = self._pad_to_patch(x_seq)
        seq_len = x_seq.shape[-1]
        past_patch_num = seq_len // self.patch_size

        x_flat = rearrange(x_seq, "b d l -> (b d) l")
        patch_seq = rearrange(x_flat, "b (n p) -> b n p", p=self.patch_size)
        patch_embed = self.patch_embedding(patch_seq)

        in_idxs = torch.arange(past_patch_num, device=x_seq.device)[None, :].expand(
            batch_size * data_dim, -1
        )
        pos_embed = self.position_embedding(in_idxs)
        channel_idx = (
            torch.arange(data_dim, device=x_seq.device)[None, :]
            .expand(batch_size, -1)
            .reshape(-1)
        )
        channel_embed = self.channel_embedding(channel_idx)[:, None, :]
        tokens = patch_embed + pos_embed + channel_embed
        encoded = self.enc_2_dec(self.encoder(tokens))
        encoded = rearrange(
            encoded,
            "(batch_size data_dim) patch_num d_model -> batch_size data_dim patch_num d_model",
            batch_size=batch_size,
        )
        return encoded, past_patch_num

    def forward(self, x_seq: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Return future latent tokens [batch_size, data_dim, out_patch_num, d_model]."""
        batch_size, data_dim, _ = x_seq.shape
        output_patch_num = math.ceil(self.out_len / self.patch_size)

        encoded_past, past_patch_num = self._encode_past(x_seq)
        graph_adj = _graph_construct(encoded_past, self.neighbor_num)

        channel_idx = torch.arange(data_dim, device=x_seq.device)
        channel_embed = self.channel_embedding(channel_idx)
        channel_embed_future = channel_embed[None, :, None, :].expand(
            batch_size, -1, output_patch_num, -1
        )
        patch_embed_future = self.learnable_patch[None, None, None, :].expand(
            batch_size, data_dim, output_patch_num, -1
        )
        future_patch_idx = torch.arange(
            past_patch_num, past_patch_num + output_patch_num, device=x_seq.device
        )[None, :].expand(batch_size, -1)
        position_embed_future = self.position_embedding(future_patch_idx)
        position_embed_future = position_embed_future[:, None, :, :].expand(
            -1, data_dim, -1, -1
        )
        patches_future = (
            patch_embed_future + position_embed_future + channel_embed_future
        )
        patches_full = torch.cat((encoded_past, patches_future), dim=2)

        for layer in self.tc_layers:
            patches_full = layer(patches_full, graph_adj)

        return patches_full[:, :, past_patch_num:, :]
