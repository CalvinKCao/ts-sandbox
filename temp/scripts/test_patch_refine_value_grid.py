#!/usr/bin/env python3
"""Synthetic contract test for non-ordinal patch-refine value-grid snapping."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.patch_refine_value_grid import (
    assert_on_patch_refine_grid,
    grid_coordinates,
    normalized_grid_step,
    snap_to_unbounded_patch_refine_grid,
    window_normalization_stats,
)


def main() -> None:
    config = SimpleNamespace(
        use_ordinal_window_norm=False,
        use_window_normalization=True,
        window_norm_center="mean",
        window_norm_std_floor=0.1,
        window_norm_low_var_threshold=0.0,
        patch_refine_canvas_height=256,
        max_scale=5.2,
    )
    # Different windows/variates make the contract explicitly local.
    past = torch.tensor(
        [
            [[1.0, 1.2, 1.4, 1.6], [10.0, 10.4, 10.8, 11.2]],
            [[-3.0, -2.8, -2.6, -2.4], [3.0, 3.3, 3.6, 3.9]],
        ],
        dtype=torch.float32,
    )
    center, std = window_normalization_stats(past, config)
    step = normalized_grid_step(config)
    # Includes values well beyond the finite rows 0..255.  They must round to
    # unbounded midpoint rows, not be clipped to the endpoints.
    targets = torch.cat(
        [
            center - 7.3 * std,
            center - 5.1 * std,
            center + 0.17 * std,
            center + 8.7 * std,
        ],
        dim=-1,
    )
    snapped = snap_to_unbounded_patch_refine_grid(targets, past, config)
    assert_on_patch_refine_grid(snapped, past, config)

    rows = grid_coordinates(snapped, past, config)
    finite_min = -config.max_scale + 0.5 * step
    finite_max = config.max_scale - 0.5 * step
    normalized = (snapped - center) / std
    assert bool((normalized < finite_min).any())
    assert bool((normalized > finite_max).any())
    assert torch.allclose(rows, rows.round(), atol=2e-4, rtol=0.0)

    # A decoded binary canvas has finite row ids 0..255.  This mimics those
    # midpoint decodes in several windows/variates and checks the same grid.
    binary_rows = torch.tensor([0.0, 1.0, 128.0, 255.0]).view(1, 1, -1)
    binary_normalized = -config.max_scale + (binary_rows + 0.5) * step
    binary_values = binary_normalized * std + center
    assert_on_patch_refine_grid(binary_values, past, config)
    binary_coords = grid_coordinates(binary_values, past, config)
    assert float(binary_coords.min()) >= -2e-4
    assert float(binary_coords.max()) <= 255.0002
    print(
        "[ok] binary midpoint decodes and GT snapping share each window's mean/std grid; "
        f"step_norm={step:.8f}, GT rows=[{int(rows.min())},{int(rows.max())}]"
    )


if __name__ == "__main__":
    main()
