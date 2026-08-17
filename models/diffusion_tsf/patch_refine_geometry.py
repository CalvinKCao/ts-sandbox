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


@dataclass(frozen=True)
class PatchLayout:
    """Device-resident, ragged patch layout used by the training hot path."""

    flat_index: torch.Tensor
    batch_index: torch.Tensor
    variate_index: torch.Tensor
    row0: torch.Tensor
    col0: torch.Tensor

    @property
    def n_patches(self) -> int:
        return int(self.flat_index.shape[0])

    def to_locations(self) -> list[PatchLocation]:
        """CPU list copy for diagnostics/viz that still walk PatchLocation."""
        return [
            PatchLocation(
                flat_index=int(flat_index),
                batch_index=int(batch_index),
                variate_index=int(variate_index),
                row0=int(row0),
                col0=int(col0),
            )
            for flat_index, batch_index, variate_index, row0, col0 in zip(
                self.flat_index.tolist(),
                self.batch_index.tolist(),
                self.variate_index.tolist(),
                self.row0.tolist(),
                self.col0.tolist(),
            )
        ]

    @classmethod
    def cat(cls, layouts: Sequence["PatchLayout"]) -> "PatchLayout":
        if not layouts:
            raise ValueError("cannot concatenate an empty layout list")
        return cls(
            flat_index=torch.cat([layout.flat_index for layout in layouts]),
            batch_index=torch.cat([layout.batch_index for layout in layouts]),
            variate_index=torch.cat([layout.variate_index for layout in layouts]),
            row0=torch.cat([layout.row0 for layout in layouts]),
            col0=torch.cat([layout.col0 for layout in layouts]),
        )

    @classmethod
    def from_locations(
        cls,
        locations: Sequence[PatchLocation],
        *,
        device: torch.device,
    ) -> "PatchLayout":
        """Compatibility bridge for inference-only/list geometry callers."""
        if not locations:
            raise ValueError("cannot build an empty patch layout")
        fields = (
            "flat_index",
            "batch_index",
            "variate_index",
            "row0",
            "col0",
        )
        return cls(
            **{
                name: torch.tensor(
                    [getattr(location, name) for location in locations],
                    device=device,
                    dtype=torch.long,
                )
                for name in fields
            }
        )


def patch_layout_for_fixed_col0(
    coarse_edges: torch.Tensor,
    col0: torch.Tensor,
    *,
    canvas_height: int,
    patch_height: int,
    patch_width: int,
    hir_canvas: Optional[torch.Tensor] = None,
) -> PatchLayout:
    """Vectorized equivalent of :func:`locations_for_fixed_col0`.

    The sampled ``col0`` remains per parent window, preserving unique-segment
    training stochasticity while keeping location values on the active device.
    """
    if coarse_edges.ndim != 3:
        raise ValueError(f"coarse_edges must be (B,V,W), got {tuple(coarse_edges.shape)}")
    batch_size, n_variates, width = coarse_edges.shape
    if col0.shape != (batch_size,):
        raise ValueError(f"col0 must be ({batch_size},), got {tuple(col0.shape)}")
    if patch_height > canvas_height or patch_width > width:
        raise ValueError("patch geometry exceeds its canvas")
    if hir_canvas is not None and hir_canvas.shape[:2] != (batch_size, n_variates):
        raise ValueError("hir_canvas batch/variate shape does not match coarse_edges")
    device = coarse_edges.device
    col0 = col0.to(device=device, dtype=torch.long)
    max_col0 = width - patch_width
    # Values come from the sampler or the dataloader. Keep range checking out
    # of the per-step path; invalid indices fail naturally at the gather below.
    col0 = col0.clamp(0, max_col0)
    batch_index = torch.arange(batch_size, device=device).repeat_interleave(n_variates)
    variate_index = torch.arange(n_variates, device=device).repeat(batch_size)
    flat_index = batch_index * n_variates + variate_index
    patch_col0 = col0.index_select(0, batch_index)
    anchor = patch_col0 + patch_width // 2
    edge_rows = coarse_edges.reshape(-1, width)[flat_index, anchor]
    max_row0 = canvas_height - patch_height
    row0 = (edge_rows - patch_height // 2).clamp(0, max_row0)
    if hir_canvas is not None:
        cols = patch_col0[:, None] + torch.arange(patch_width, device=device)
        rows = torch.arange(canvas_height, device=device)[None, :, None]
        local_cdf = hir_canvas.reshape(-1, canvas_height, width)[
            flat_index[:, None, None], rows, cols[:, None, :]
        ]
        hir_edges = local_cdf.sum(dim=1).long() - 1
        in_view = (hir_edges >= row0[:, None]) & (hir_edges < row0[:, None] + patch_height)
        fallback_edge = hir_edges[:, patch_width // 2].clamp(0, canvas_height - 1)
        fallback_row0 = (fallback_edge - patch_height // 2).clamp(0, max_row0)
        row0 = torch.where(in_view.any(dim=1), row0, fallback_row0)
    return PatchLayout(flat_index, batch_index, variate_index, row0, patch_col0)


def primary_stride_col0s(width: int, patch_width: int, col_stride: int) -> list[int]:
    max_col0 = int(width) - int(patch_width)
    if max_col0 < 0:
        raise ValueError(f"patch width {patch_width} exceeds future width {width}")
    if col_stride <= 0:
        raise ValueError("col_stride must be positive")
    return list(range(0, max_col0 + 1, int(col_stride)))


def coverage_mask_for_layout(
    coarse_edges: torch.Tensor,
    layout: PatchLayout,
    *,
    patch_height: int,
    patch_width: int,
) -> torch.Tensor:
    """Bool ``(B,V,W)`` — True where a layout crop sees the coarse boundary."""
    if coarse_edges.ndim != 3:
        raise ValueError(f"coarse_edges must be (B,V,W), got {tuple(coarse_edges.shape)}")
    batch_size, n_variates, width = coarse_edges.shape
    device = coarse_edges.device
    cols = layout.col0[:, None] + torch.arange(patch_width, device=device)
    edges = coarse_edges.reshape(-1, width)[layout.flat_index[:, None], cols]
    in_rows = (edges >= layout.row0[:, None]) & (edges < layout.row0[:, None] + patch_height)
    covered = torch.zeros(
        batch_size * n_variates, width, device=device, dtype=torch.bool,
    )
    flat = layout.flat_index[:, None].expand_as(cols)
    covered[flat[in_rows], cols[in_rows]] = True
    return covered.view(batch_size, n_variates, width)


def extract_patch_batch_layout(
    canvas: torch.Tensor,
    layout: PatchLayout,
    *,
    patch_height: int,
    patch_width: int,
) -> torch.Tensor:
    """Vectorized crop gather returning ``(N,1,patch_height,patch_width)``."""
    if canvas.ndim != 4:
        raise ValueError(f"canvas must be (B,V,H,W), got {tuple(canvas.shape)}")
    _, _, height, width = canvas.shape
    rows = layout.row0[:, None, None] + torch.arange(patch_height, device=canvas.device)[None, :, None]
    cols = layout.col0[:, None, None] + torch.arange(patch_width, device=canvas.device)[None, None, :]
    if layout.n_patches == 0:
        raise ValueError("cannot extract an empty patch layout")
    flat = canvas.reshape(-1, height, width)
    return flat[layout.flat_index[:, None, None], rows, cols].unsqueeze(1)


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


def blend_patch_bins_layout(
    patch_cdf: torch.Tensor,
    layout: PatchLayout,
    coarse_edges: torch.Tensor,
    *,
    canvas_height: int,
    patch_height: int,
    patch_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized blend of visible patch-bin votes into a hard CDF."""
    n_patches = layout.n_patches
    if patch_cdf.shape != (n_patches, 1, patch_height, patch_width):
        raise ValueError(
            "patch_cdf shape mismatch: "
            f"got {tuple(patch_cdf.shape)}, expected "
            f"({n_patches},1,{patch_height},{patch_width})"
        )
    batch_size, n_variates, width = coarse_edges.shape
    device = patch_cdf.device
    local_cdf = patch_cdf[:, 0]
    local_bins = TimeSeriesTo2D.bin_indices_from_cdf(local_cdf)
    occupancy = local_cdf.sum(dim=-2)
    visible = (occupancy > 0) & (occupancy < patch_height)
    cols = layout.col0[:, None] + torch.arange(patch_width, device=device)
    edges = coarse_edges.reshape(-1, width)[layout.flat_index[:, None], cols]
    in_rows = (edges >= layout.row0[:, None]) & (edges < layout.row0[:, None] + patch_height)
    vote = visible & in_rows
    absolute_bin = layout.row0[:, None].to(dtype=torch.float32) + local_bins.to(dtype=torch.float32)
    sums_flat = torch.zeros(
        batch_size * n_variates, width, device=device, dtype=torch.float32,
    )
    counts_flat = torch.zeros_like(sums_flat)
    flat = layout.flat_index[:, None].expand_as(cols)
    sums_flat.index_put_((flat[vote], cols[vote]), absolute_bin[vote], accumulate=True)
    counts_flat.index_put_(
        (flat[vote], cols[vote]),
        torch.ones_like(absolute_bin[vote]),
        accumulate=True,
    )
    sums = sums_flat.view(batch_size, n_variates, width)
    counts = counts_flat.view(batch_size, n_variates, width)
    has_vote = counts > 0
    averaged_bins = torch.zeros_like(sums)
    averaged_bins[has_vote] = sums[has_vote] / counts[has_vote]
    bins = torch.where(has_vote, averaged_bins.round(), coarse_edges.float())
    bins = bins.clamp(0, canvas_height - 1).long()
    rows = torch.arange(canvas_height, device=device).view(1, 1, canvas_height, 1)
    hard_cdf = (rows <= bins.unsqueeze(-2)).to(patch_cdf.dtype)
    return hard_cdf, counts


def blend_patch_bins(
    patch_cdf: torch.Tensor,
    locations: Sequence[PatchLocation],
    coarse_edges: torch.Tensor,
    *,
    canvas_height: int,
    patch_height: int,
    patch_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """List-API wrapper around :func:`blend_patch_bins_layout`."""
    layout = PatchLayout.from_locations(locations, device=patch_cdf.device)
    return blend_patch_bins_layout(
        patch_cdf,
        layout,
        coarse_edges,
        canvas_height=canvas_height,
        patch_height=patch_height,
        patch_width=patch_width,
    )
