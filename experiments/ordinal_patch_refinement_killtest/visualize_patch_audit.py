"""Render coarse, direct high-resolution GT, refined, and error patch panels."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arrays", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--count", type=int, default=16)
    args = parser.parse_args()
    data = np.load(args.arrays)
    out = args.output or args.arrays.parent / "patch_audit.png"
    n = min(args.count, len(data["target_patches"]))
    patch = int(data["target_patches"].shape[-1])
    fig, axs = plt.subplots(n, 4, figsize=(12, max(3, n * 2.1)), squeeze=False)
    for i in range(n):
        coarse = data["input_patches"][i, 0]
        gt = data["target_patches"][i, 0]
        refined = data["refined_patches"][i, 0]
        panels = [
            (coarse, "naive coarse input", "viridis"),
            (gt, "direct hi-res GT target", "viridis"),
            (refined, "refined prediction", "viridis"),
            (refined - gt, "refined − GT", "coolwarm"),
        ]
        for ax, (img, title, cmap) in zip(axs[i], panels):
            ax.imshow(img, origin="lower", aspect="auto", cmap=cmap)
            if title != "refined − GT":
                ax.axhline(patch // 2, color="white", linewidth=0.7, linestyle="--")
            ax.set_title(f"t={i}: {title}")
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(out)


if __name__ == "__main__":
    main()
