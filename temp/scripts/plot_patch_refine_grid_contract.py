#!/usr/bin/env python3
"""Plot the h96 patch-refine value-grid and unblended-patch contract."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/patch_refine_h96_smoke/patch_refine_h96_smoke"),
    )
    parser.add_argument("--window", type=int, default=0)
    parser.add_argument("--variate", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.raw) as data:
        past = data["past"]
        gt = data["y_true"]
        binary = data["samples"][:, :, 0, :]
        grid = data["grid_values"]
        gt_rows = data["gt_rows"]
        patch_parent = data["unblended_nonoverlap_patch_parent"]
        patch_start = data["unblended_nonoverlap_patch_start"]
        patch_variate = data["unblended_nonoverlap_patch_variate"]
        gt_error = float(data["gt_grid_max_row_error"])
        binary_error = float(data["binary_grid_max_row_error"])

    w, v = int(args.window), int(args.variate)
    if not (0 <= w < gt.shape[0] and 0 <= v < gt.shape[1]):
        raise ValueError(f"window/variate out of range: ({w}, {v}) for {gt.shape[:2]}")
    grid_w = grid[w, v]
    step = float(np.median(np.diff(grid_w)))
    grid_coord_gt = (gt[w, v] - grid_w[0]) / step
    grid_coord_binary = (binary[w, v] - grid_w[0]) / step
    x_future = np.arange(gt.shape[-1])
    x_past = np.arange(-min(96, past.shape[-1]), 0)
    chosen_starts = np.sort(patch_start[(patch_parent == w) & (patch_variate == v)])
    if np.any(chosen_starts[1:] < chosen_starts[:-1] + 8):
        raise AssertionError("saved unblended patches overlap")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), constrained_layout=True)

    ax = axes[0]
    for value in grid_w:
        ax.axhline(value, color="0.55", alpha=0.10, linewidth=0.35, zorder=0)
    ax.plot(x_past, past[w, v, -len(x_past):], color="0.25", linewidth=1.2, label="lookback")
    ax.plot(x_future, gt[w, v], color="#1f77b4", linewidth=1.8, label="GT snapped")
    ax.plot(x_future, binary[w, v], color="#d62728", linewidth=1.3, label="binary patch-refine")
    ax.axvline(-0.5, color="0.2", linewidth=0.8)
    ax.set_title(
        f"Window {w}, variate {v}: every faint line is one of this window's 256 decoded values"
    )
    ax.set_xlabel("forecast timestep")
    ax.set_ylabel("binary dataset-z")
    ax.legend(loc="best")

    ax = axes[1]
    ax.axhline(0.0, color="0.2", linewidth=0.8)
    ax.plot(x_future, grid_coord_gt - np.rint(grid_coord_gt), ".", color="#1f77b4", label="GT row residual")
    ax.plot(x_future, grid_coord_binary - np.rint(grid_coord_binary), ".", color="#d62728", label="binary row residual")
    ax.set_ylim(-3e-4, 3e-4)
    ax.set_title(
        f"Exact-grid sanity check: max row error GT={gt_error:.2e}, binary={binary_error:.2e}"
    )
    ax.set_xlabel("forecast timestep")
    ax.set_ylabel("row coordinate minus nearest integer")
    ax.legend(loc="best")

    ax = axes[2]
    for row, start in enumerate(chosen_starts):
        ax.broken_barh([(int(start), 8)], (row - 0.35, 0.7), facecolors="#2ca02c")
        ax.text(int(start) + 4, row, f"{int(start)}–{int(start) + 7}", ha="center", va="center", fontsize=8)
    ax.set_xlim(-1, gt.shape[-1])
    ax.set_ylim(-1, max(1, len(chosen_starts)))
    ax.set_yticks(range(len(chosen_starts)))
    ax.set_title("Separate discriminator metric: selected raw 8-step patches, pairwise non-overlapping")
    ax.set_xlabel("forecast timestep")
    ax.set_ylabel("unblended patch example")

    output = args.output_dir / f"window{w}_variate{v}_grid_contract.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
