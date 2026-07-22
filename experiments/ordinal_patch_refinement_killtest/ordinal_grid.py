"""Shared ordinal-grid canonicalization for discriminator kill tests."""

from __future__ import annotations

from typing import Any

import torch


def rank_to_bin_centres(
    ranks: torch.Tensor,
    rank_max: torch.Tensor,
    bins: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map ordinal ranks to bounded bin centres normalized to [0, 1]."""
    safe_max = rank_max.to(device=ranks.device, dtype=ranks.dtype).clamp_min(1.0)
    safe_max = safe_max.view(1, -1, 1)
    bin_ids = torch.floor(ranks / safe_max * bins).long().clamp_(0, bins - 1)
    centres = (bin_ids.to(ranks.dtype) + 0.5) / float(bins)
    return centres, bin_ids


def bin_centres_to_ranks(
    bin_ids: torch.Tensor,
    rank_max: torch.Tensor,
    bins: int,
) -> torch.Tensor:
    """Decode normalized bin centres into continuous ordinal-rank units."""
    safe_max = rank_max.to(device=bin_ids.device, dtype=torch.float32).clamp_min(1.0)
    safe_max = safe_max.view(1, -1, 1)
    return (bin_ids.to(torch.float32) + 0.5) / float(bins) * safe_max


def snap_ranks_to_ladder(ranks: torch.Tensor, ladder: Any) -> torch.Tensor:
    """Round continuous ranks onto valid integer positions of the global ladder."""
    snapped = ranks.clone()
    for variate in range(int(ranks.shape[1])):
        n_unique = int(ladder.n_unique[0, variate].item())
        snapped[:, variate] = ranks[:, variate].round().clamp_(0, max(0, n_unique - 1))
    return snapped


def canonicalize_ranks(
    ranks: torch.Tensor,
    rank_max: torch.Tensor,
    ladder: Any,
    bins: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rank -> bin centre -> global-ladder snap and return bin IDs."""
    _centres, bin_ids = rank_to_bin_centres(ranks, rank_max, bins)
    decoded = bin_centres_to_ranks(bin_ids, rank_max, bins)
    return snap_ranks_to_ladder(decoded, ladder), bin_ids
