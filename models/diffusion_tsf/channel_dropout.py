"""Train-only random channel dropout for mixer / DiT x-attn (fixed M, absolute IDs)."""

from __future__ import annotations

from typing import Optional

import torch


def n_keep_channels(n_variates: int, drop_frac: float) -> int:
    if drop_frac < 0.0 or drop_frac >= 1.0:
        raise ValueError(f"channel_dropout_drop_frac must be in [0, 1), got {drop_frac}")
    if n_variates < 1:
        raise ValueError(f"n_variates must be >= 1, got {n_variates}")
    keep_frac = 1.0 - float(drop_frac)
    return max(1, min(n_variates, int(round(n_variates * keep_frac))))


def sample_kept_channel_mask(
    n_windows: int,
    n_variates: int,
    drop_frac: float,
    *,
    training: bool,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """(n_windows, V) bool True=keep. None if dropout is off."""
    if not training or drop_frac <= 0.0:
        return None
    n_keep = n_keep_channels(n_variates, drop_frac)
    if n_keep >= n_variates:
        return None
    scores = torch.rand(n_windows, n_variates, device=device)
    _, idx = scores.topk(n_keep, dim=1)
    keep = torch.zeros(n_windows, n_variates, device=device, dtype=torch.bool)
    keep.scatter_(1, idx, True)
    return keep


def token_drop_mask_from_channel_keep(
    keep: torch.Tensor,
    token_variate_ids: torch.Tensor,
) -> torch.Tensor:
    """keep (B, V) True=keep -> dropped tokens (B, M) True=drop."""
    if keep.dim() != 2:
        raise ValueError(f"keep must be (B, V), got {tuple(keep.shape)}")
    if token_variate_ids.dim() != 1:
        raise ValueError(
            f"token_variate_ids must be (M,), got {tuple(token_variate_ids.shape)}"
        )
    if int(token_variate_ids.min()) < 0 or int(token_variate_ids.max()) >= keep.shape[1]:
        raise ValueError(
            f"token_variate_ids out of range for V={keep.shape[1]}: "
            f"[{int(token_variate_ids.min())}, {int(token_variate_ids.max())}]"
        )
    dropped_channels = ~keep
    return dropped_channels[:, token_variate_ids]
