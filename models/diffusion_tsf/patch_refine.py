"""Boundary-patch CDF refinement stage helpers."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .patch_refine_geometry import (
    PatchLocation,
    blend_patch_bins,
    coarse_edges_from_cdf,
    extract_patch_batch,
    select_patch_locations,
)
from .preprocessing import TimeSeriesTo2D


def naive_upscale_coarse_cdf(coarse_cdf: torch.Tensor, canvas_height: int) -> torch.Tensor:
    """Nearest-neighbor vertical upscale of coarse CDF to ``canvas_height``."""
    if coarse_cdf.ndim != 4:
        raise ValueError(f"coarse_cdf must be (B,V,H,W), got {tuple(coarse_cdf.shape)}")
    if coarse_cdf.shape[2] == canvas_height:
        return coarse_cdf
    flat = coarse_cdf.reshape(-1, 1, coarse_cdf.shape[2], coarse_cdf.shape[3])
    up = F.interpolate(flat, size=(canvas_height, coarse_cdf.shape[3]), mode="nearest")
    return up.reshape(coarse_cdf.shape[0], coarse_cdf.shape[1], canvas_height, coarse_cdf.shape[3])


def stack_past_coarse_fine(
    past_coarse: torch.Tensor,
    past_fine: torch.Tensor,
) -> torch.Tensor:
    """Lossless stack to ``(B,V,Hc+Hf,W)`` then flatten to ``(BV,1,H,W)``."""
    if past_coarse.shape[:2] != past_fine.shape[:2] or past_coarse.shape[-1] != past_fine.shape[-1]:
        raise ValueError(
            f"past coarse/fine shape mismatch: {tuple(past_coarse.shape)} vs {tuple(past_fine.shape)}"
        )
    stacked = torch.cat([past_coarse, past_fine], dim=2)
    b, v, h, w = stacked.shape
    return stacked.reshape(b * v, 1, h, w)


def encode_absolute_hir_cdf(
    values: torch.Tensor,
    *,
    canvas_height: int,
    max_scale: float,
    ordinal_rank_max: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Encode absolute hi-res CDF ``(B,V,H_hi,W)`` from 1D values."""
    if values.dim() == 2:
        values = values.unsqueeze(1)
    if ordinal_rank_max is not None:
        vmax = ordinal_rank_max.to(device=values.device, dtype=values.dtype).reshape(-1)
        maps = []
        for vi in range(values.shape[1]):
            span = float(vmax[vi].item())
            xi = values[:, vi : vi + 1]
            if span <= 0.0:
                bins = torch.zeros_like(xi, dtype=torch.long)
            else:
                pos = (xi.clamp(0.0, span) / span) * canvas_height
                bins = pos.long().clamp(0, canvas_height - 1)
            rows = torch.arange(canvas_height, device=values.device).view(1, 1, canvas_height, 1)
            maps.append((rows <= bins.unsqueeze(2)).to(values.dtype))
        return torch.cat(maps, dim=1)

    x_clipped = values.clamp(-max_scale, max_scale)
    pos = (x_clipped + max_scale) / (2 * max_scale) * canvas_height
    bins = pos.long().clamp(0, canvas_height - 1)
    rows = torch.arange(canvas_height, device=values.device).view(1, 1, canvas_height, 1)
    return (rows <= bins.unsqueeze(2)).to(values.dtype)


def decode_absolute_hir_cdf(
    hir_cdf: torch.Tensor,
    *,
    max_scale: float,
    ordinal_rank_max: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mid-bin decode of absolute hi-res CDF to 1D values ``(B,V,W)``."""
    bins = TimeSeriesTo2D.bin_indices_from_cdf(hir_cdf)
    height = float(hir_cdf.shape[-2])
    mid = (bins + 0.5) / height
    if ordinal_rank_max is not None:
        vmax = ordinal_rank_max.to(device=hir_cdf.device, dtype=hir_cdf.dtype).reshape(1, -1, 1)
        return mid * vmax
    return mid * (2 * max_scale) - max_scale


def build_patch_aux_channels(
    naive_canvas: torch.Tensor,
    coarse_edges: torch.Tensor,
    locations: Sequence[PatchLocation],
    *,
    patch_height: int,
    patch_width: int,
    canvas_height: int,
    coarse_height: int,
    horizon_width: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(aux, patch_coarse_bin, patch_time0)``.

    ``aux`` is ``(N,3,ph,pw)`` = naive crop, coarse-cell id map, absolute-time map.
    """
    naive_patches = extract_patch_batch(
        naive_canvas, locations, patch_height=patch_height, patch_width=patch_width,
    )
    n = len(locations)
    device = naive_canvas.device
    dtype = naive_canvas.dtype
    coarse_cell = torch.zeros(n, 1, patch_height, patch_width, device=device, dtype=dtype)
    time_map = torch.zeros_like(coarse_cell)
    patch_coarse_bin = torch.zeros(n, device=device, dtype=torch.long)
    patch_time0 = torch.zeros(n, device=device, dtype=torch.long)
    denom_bin = max(1, coarse_height - 1)
    denom_t = max(1, horizon_width - 1)

    for i, loc in enumerate(locations):
        cols = torch.arange(loc.col0, loc.col0 + patch_width, device=device)
        edges = coarse_edges[loc.batch_index, loc.variate_index, cols]
        # Invert NN-upscale edge formula: edge = (bin+1)*scale - 1.
        scale = canvas_height // coarse_height
        bins = ((edges + 1) // scale - 1).clamp(0, coarse_height - 1)
        cell = (bins.float() / float(denom_bin)).view(1, 1, 1, patch_width)
        coarse_cell[i] = cell.expand(1, 1, patch_height, patch_width)
        tnorm = (cols.float() / float(denom_t)).view(1, 1, 1, patch_width)
        time_map[i] = tnorm.expand(1, 1, patch_height, patch_width)
        mid = patch_width // 2
        patch_coarse_bin[i] = bins[mid]
        patch_time0[i] = loc.col0

    aux = torch.cat([naive_patches, coarse_cell, time_map], dim=1)
    return aux, patch_coarse_bin, patch_time0


def expand_lookback_cond_for_patches(
    lookback_cond: torch.Tensor,
    locations: Sequence[PatchLocation],
) -> torch.Tensor:
    """``(BV,1,H,Lb)`` → ``(N,1,H,Lb)`` by indexing each crop's flat variate row."""
    return torch.stack([lookback_cond[loc.flat_index] for loc in locations], dim=0)


def expand_ctx_for_patches(
    ctx_flat: Optional[torch.Tensor],
    locations: Sequence[PatchLocation],
) -> Optional[torch.Tensor]:
    if ctx_flat is None:
        return None
    return torch.stack([ctx_flat[loc.flat_index] for loc in locations], dim=0)


def expand_variate_indices_for_patches(
    locations: Sequence[PatchLocation],
    device: torch.device,
) -> torch.Tensor:
    return torch.tensor(
        [loc.variate_index for loc in locations],
        device=device,
        dtype=torch.long,
    )
