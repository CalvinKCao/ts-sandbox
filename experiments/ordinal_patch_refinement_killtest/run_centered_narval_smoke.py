"""Boundary-centered Narval entry point for the oracle-coarse smoke test."""

from __future__ import annotations

import torch

from experiments.ordinal_patch_refinement_killtest import run_narval_smoke as _narval_adapter
from experiments.ordinal_patch_refinement_killtest import smoke


def _boundary_centered_patch_batch(
    upscaled: torch.Tensor,
    target: torch.Tensor,
    past_canvas: torch.Tensor,
    coarse_bins: torch.Tensor,
    patch_size: int,
):
    """Place the input-derived coarse CDF transition at local patch row 8."""
    inputs, conds, targets, coords = [], [], [], []
    horizon = int(coarse_bins.shape[-1])
    high = int(upscaled.shape[-2])
    center = patch_size // 2
    rows = torch.arange(patch_size, device=upscaled.device).view(1, patch_size, 1)
    for t in range(horizon):
        boundary_row = (int(coarse_bins[t].item()) + 1) * patch_size
        row0 = boundary_row - center
        col0 = t * patch_size
        if row0 < 0 or row0 + patch_size > high:
            raise RuntimeError(
                f"boundary-centered patch t={t} requires vertical padding: "
                f"boundary={boundary_row}, row0={row0}, canvas_height={high}"
            )
        inp, valid = smoke._extract_block(upscaled, row0, col0, patch_size)
        tgt, _ = smoke._extract_block(target, row0, col0, patch_size)
        hist, _ = smoke._extract_block(past_canvas, row0, 0, patch_size)
        boundary = (rows == center).to(upscaled.dtype).expand(1, patch_size, patch_size)
        time_pos = torch.full_like(inp, float(t) / max(1, horizon - 1))
        vertical_pos = torch.linspace(
            row0 / high,
            (row0 + patch_size - 1) / high,
            patch_size,
            device=upscaled.device,
        ).view(1, patch_size, 1).expand_as(inp)
        inputs.append(torch.cat([inp, boundary, valid, time_pos, vertical_pos], dim=0))
        conds.append(hist)
        targets.append(tgt)
        coords.append((row0, col0))
    return torch.stack(inputs), torch.stack(conds), torch.stack(targets), coords


smoke._patch_batch = _boundary_centered_patch_batch


if __name__ == "__main__":
    smoke.main()
