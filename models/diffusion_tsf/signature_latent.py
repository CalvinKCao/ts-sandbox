"""Log-signature latent encode/decode, fusion, and overlap-add reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.diffusion_tsf.signature_mse_loss import (
    extract_overlapping_patches,
    logsignature_dim,
    prepare_signature_path,
    signature_dim,
    truncated_logsignature,
    truncated_signature,
)


@dataclass(frozen=True)
class LatentConfig:
    depth: int = 3
    use_cumsum: bool = False
    logsig_mode: str = "words"
    latent_rep: str = "logsignature"
    normalize_logsig: bool = True
    logsig_eps: float = 1e-6
    patch_size: int = 24
    patch_stride: int = 12


def num_patches(sequence_length: int, patch_size: int, stride: int) -> int:
    if sequence_length < patch_size:
        return 0
    return (sequence_length - patch_size) // stride + 1


def select_channels(y: torch.Tensor, subset: Sequence[int]) -> torch.Tensor:
    """``y``: [B, T, C] -> [B, T, |S|]."""
    if not subset:
        return y
    idx = list(subset)
    return y[..., idx]


def encode_patch_logsig(
    patches: torch.Tensor,
    cfg: LatentConfig,
) -> torch.Tensor:
    """``patches`` [B, P, W, C] -> log-signature or signature [B, P, D]."""
    batch, n_patches, width, _ = patches.shape
    flat = patches.reshape(batch * n_patches, width, -1)
    path = prepare_signature_path(flat, use_cumsum=cfg.use_cumsum)

    if cfg.latent_rep == "signature":
        lat = truncated_signature(path, cfg.depth)
    else:
        lat = truncated_logsignature(path, cfg.depth, mode=cfg.logsig_mode)

    lat = lat.reshape(batch, n_patches, -1)
    if cfg.normalize_logsig:
        denom = torch.linalg.vector_norm(lat, dim=-1, keepdim=True).clamp_min(cfg.logsig_eps)
        lat = lat / denom
    return lat


def encode_series_logsig(
    y: torch.Tensor,
    cfg: LatentConfig,
    *,
    subset: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``y`` [B, T, C] -> (latent [B, P, D], patches [B, P, W, |S|])."""
    y_s = select_channels(y, subset or ())
    patches = extract_overlapping_patches(
        y_s, patch_size=cfg.patch_size, stride=cfg.patch_stride
    )
    return encode_patch_logsig(patches, cfg), patches


def latent_dim(cfg: LatentConfig, n_channels: int) -> int:
    path_ch = n_channels + 1
    if cfg.latent_rep == "signature":
        return signature_dim(path_ch, cfg.depth)
    return logsignature_dim(path_ch, cfg.depth)


class LogSigPatchDecoder(nn.Module):
    def __init__(self, latent_dim: int, patch_size: int, n_channels: int, hidden: int = 256):
        super().__init__()
        self.patch_size = patch_size
        self.n_channels = n_channels
        out_dim = patch_size * n_channels
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """``latent`` [B, P, D] -> patches [B, P, W, C]."""
        out = self.net(latent)
        batch, n_patches, _ = latent.shape
        return out.reshape(batch, n_patches, self.patch_size, self.n_channels)


def overlap_add_patches(
    patches: torch.Tensor,
    *,
    horizon: int,
    patch_size: int,
    stride: int,
) -> torch.Tensor:
    """Stitch patch predictions into ``[B, T, C]`` with Hann-weighted overlap-add."""
    batch, n_patches, width, n_channels = patches.shape
    device, dtype = patches.device, patches.dtype
    acc = torch.zeros(batch, horizon, n_channels, device=device, dtype=dtype)
    weight = torch.zeros(batch, horizon, 1, device=device, dtype=dtype)

    window = torch.hann_window(width, device=device, dtype=dtype).view(1, width, 1)

    for p in range(n_patches):
        start = p * stride
        end = start + width
        if end > horizon:
            break
        w = window
        acc[:, start:end, :] += patches[:, p, :, :] * w
        weight[:, start:end, :] += w

    return acc / weight.clamp_min(1e-8)


def fuse_logsig_precision(
    latents: List[torch.Tensor],
    weights: Optional[List[torch.Tensor]] = None,
) -> torch.Tensor:
    """Precision-weighted average of ``[B, P, D]`` tensors (same D)."""
    if len(latents) == 1:
        return latents[0]
    stack = torch.stack(latents, dim=0)
    if weights is None:
        return stack.mean(dim=0)
    w = torch.stack(weights, dim=0)
    w = w / w.sum(dim=0, keepdim=True).clamp_min(1e-8)
    return (stack * w).sum(dim=0)


def fuse_point_channels(
    branch_preds: List[Tuple[Sequence[int], torch.Tensor]],
    *,
    n_channels: int,
    weights: Optional[List[float]] = None,
) -> torch.Tensor:
    """Fuse ``[B, T, |S|]`` branch forecasts into ``[B, T, C]`` on overlapping channels."""
    if not branch_preds:
        raise ValueError("branch_preds empty")
    ref = branch_preds[0][1]
    batch, horizon, _ = ref.shape
    device, dtype = ref.device, ref.dtype
    acc = torch.zeros(batch, horizon, n_channels, device=device, dtype=dtype)
    wsum = torch.zeros(batch, horizon, n_channels, device=device, dtype=dtype)

    for bi, (subset, pred) in enumerate(branch_preds):
        w = 1.0 if weights is None else float(weights[bi])
        for local_i, ch in enumerate(subset):
            acc[:, :, ch] += pred[:, :, local_i] * w
            wsum[:, :, ch] += w

    return acc / wsum.clamp_min(1e-8)


def logsig_consistency_loss(
    decoded_patches: torch.Tensor,
    latent_hat: torch.Tensor,
    cfg: LatentConfig,
) -> torch.Tensor:
    target = encode_patch_logsig(decoded_patches, cfg)
    return F.l1_loss(target, latent_hat)
