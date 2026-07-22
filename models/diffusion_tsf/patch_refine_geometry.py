"""Boundary-centered crop geometry for high-resolution CDF refinement."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from .preprocessing import TimeSeriesTo2D

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PatchLocation:
    """One crop in a flattened ``B * V`` future canvas."""

    flat_index: int
    batch_index: int
    variate_index: int
    row0: int
    col0: int


def coarse_edges_from_cdf(
    coarse_cdf: torch.Tensor,
    *,
    canvas_height: int,
) -> torch.Tensor:
    """Return nearest-neighbour-upscaled coarse boundary rows, shape ``(B,V,W)``."""
    if coarse_cdf.ndim != 4:
        raise ValueError(f"coarse_cdf must be (B,V,H,W), got {tuple(coarse_cdf.shape)}")
    coarse_height = int(coarse_cdf.shape[-2])
    if canvas_height % coarse_height:
        raise ValueError(
            f"canvas_height={canvas_height} must be divisible by coarse height={coarse_height}"
        )
    scale = canvas_height // coarse_height
    bins = TimeSeriesTo2D.bin_indices_from_cdf(coarse_cdf).round().long()
    return ((bins + 1) * scale - 1).clamp(0, canvas_height - 1)


def select_patch_locations(
    coarse_edges: torch.Tensor,
    *,
    canvas_height: int,
    patch_height: int,
    patch_width: int,
    col_stride: int,
    max_patches_per_variate: Optional[int] = None,
) -> list[PatchLocation]:
    """Place stride crops, then add crops until every timestep boundary is covered."""
    if coarse_edges.ndim != 3:
        raise ValueError(f"coarse_edges must be (B,V,W), got {tuple(coarse_edges.shape)}")
    if patch_height > canvas_height:
        raise ValueError("patch height exceeds canvas height")
    width = int(coarse_edges.shape[-1])
    if patch_width > width:
        raise ValueError(f"patch width {patch_width} exceeds future width {width}")
    if col_stride <= 0:
        raise ValueError("col_stride must be positive")

    batch_size, n_variates, _ = coarse_edges.shape
    max_row0 = canvas_height - patch_height
    max_col0 = width - patch_width
    primary_starts = list(range(0, max_col0 + 1, col_stride))
    # Primary stride crops + at most one fill-in per uncovered timestep.
    soft_cap = len(primary_starts) + width
    hard_cap = int(max_patches_per_variate) if max_patches_per_variate is not None else soft_cap
    if hard_cap < len(primary_starts):
        raise ValueError(
            f"max_patches_per_variate={hard_cap} < primary crop count {len(primary_starts)}"
        )
    locations: list[PatchLocation] = []
    n_primary_total = 0
    n_fallback_total = 0

    for bi in range(batch_size):
        for vi in range(n_variates):
            flat_index = bi * n_variates + vi
            edges = coarse_edges[bi, vi]
            covered = torch.zeros(width, device=edges.device, dtype=torch.bool)
            seen: set[tuple[int, int]] = set()
            n_before = len(locations)

            def add_crop(col0: int, anchor_t: int) -> None:
                row0 = max(
                    0,
                    min(int(edges[anchor_t].item()) - patch_height // 2, max_row0),
                )
                key = (row0, col0)
                if key not in seen:
                    if len(seen) >= hard_cap:
                        raise RuntimeError(
                            f"patch_refine crop cap exceeded at B={bi} V={vi}: "
                            f"cap={hard_cap} (primary={len(primary_starts)}, width={width})"
                        )
                    seen.add(key)
                    locations.append(
                        PatchLocation(
                            flat_index=flat_index,
                            batch_index=bi,
                            variate_index=vi,
                            row0=row0,
                            col0=col0,
                        )
                    )
                cols = torch.arange(
                    col0,
                    col0 + patch_width,
                    device=edges.device,
                )
                in_rows = (edges[cols] >= row0) & (edges[cols] < row0 + patch_height)
                covered[cols[in_rows]] = True

            for col0 in primary_starts:
                add_crop(col0, col0 + patch_width // 2)
            n_primary = len(locations) - n_before

            while not bool(covered.all()):
                timestep = int((~covered).nonzero(as_tuple=False)[0].item())
                col0 = max(0, min(timestep - patch_width // 2, max_col0))
                was_covered = bool(covered[timestep])
                add_crop(col0, timestep)
                if not bool(covered[timestep]) and not was_covered:
                    raise RuntimeError(
                        f"failed to cover boundary at B={bi} V={vi} t={timestep}"
                    )
            n_fallback = (len(locations) - n_before) - n_primary
            n_primary_total += n_primary
            n_fallback_total += n_fallback

    n_total = len(locations)
    logger.info(
        "patch_refine locations: n=%d primary=%d fallback=%d "
        "(B=%d V=%d W=%d stride=%d primary_per_var=%d)",
        n_total,
        n_primary_total,
        n_fallback_total,
        batch_size,
        n_variates,
        width,
        col_stride,
        len(primary_starts),
    )
    if n_fallback_total > n_primary_total:
        logger.warning(
            "patch_refine fallback crops (%d) exceed primary (%d); "
            "noisy coarse edges or tight patch height may be inflating the batch",
            n_fallback_total,
            n_primary_total,
        )
    return locations


def extract_patch_batch(
    canvas: torch.Tensor,
    locations: Sequence[PatchLocation],
    *,
    patch_height: int,
    patch_width: int,
) -> torch.Tensor:
    """Extract ``(N,1,patch_height,patch_width)`` crops from ``(B,V,H,W)``."""
    if not locations:
        raise ValueError("cannot extract an empty patch batch")
    patches = [
        canvas[
            loc.batch_index,
            loc.variate_index,
            loc.row0 : loc.row0 + patch_height,
            loc.col0 : loc.col0 + patch_width,
        ]
        for loc in locations
    ]
    return torch.stack(patches, dim=0).unsqueeze(1)


def blend_patch_bins(
    patch_cdf: torch.Tensor,
    locations: Sequence[PatchLocation],
    coarse_edges: torch.Tensor,
    *,
    canvas_height: int,
    patch_height: int,
    patch_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average absolute predicted bins per covered timestep and rebuild a hard CDF."""
    if patch_cdf.shape != (len(locations), 1, patch_height, patch_width):
        raise ValueError(
            "patch_cdf shape mismatch: "
            f"got {tuple(patch_cdf.shape)}, expected "
            f"({len(locations)},1,{patch_height},{patch_width})"
        )
    batch_size, n_variates, width = coarse_edges.shape
    sums = torch.zeros(
        batch_size,
        n_variates,
        width,
        device=patch_cdf.device,
        dtype=torch.float32,
    )
    counts = torch.zeros_like(sums)
    local_bins = TimeSeriesTo2D.bin_indices_from_cdf(patch_cdf[:, 0])

    for pi, loc in enumerate(locations):
        for local_col in range(patch_width):
            col = loc.col0 + local_col
            edge = int(coarse_edges[loc.batch_index, loc.variate_index, col].item())
            if loc.row0 <= edge < loc.row0 + patch_height:
                absolute_bin = float(loc.row0) + local_bins[pi, local_col]
                sums[loc.batch_index, loc.variate_index, col] += absolute_bin
                counts[loc.batch_index, loc.variate_index, col] += 1.0

    if bool((counts == 0).any()):
        missing = (counts == 0).nonzero(as_tuple=False)[0].tolist()
        raise RuntimeError(f"patch coverage invariant failed at B,V,t={missing}")
    bins = (sums / counts).round().clamp(0, canvas_height - 1).long()
    rows = torch.arange(
        canvas_height,
        device=patch_cdf.device,
    ).view(1, 1, canvas_height, 1)
    hard_cdf = (rows <= bins.unsqueeze(-2)).to(patch_cdf.dtype)
    return hard_cdf, counts
