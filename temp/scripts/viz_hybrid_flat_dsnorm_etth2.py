#!/usr/bin/env python3
"""One-panel viz: ETTh2 hybrid flat vs window-norm treatment (LULL vs HUFL)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from utils.hybrid_flat_dataset_norm import build_hybrid_affine_scales


def main() -> None:
    df = pd.read_csv(REPO / "datasets/ETT-small/ETTh2.csv")
    cols = [c for c in df.columns if c != "date"]
    data = df[cols].to_numpy(dtype=np.float64)
    train_end = 12 * 30 * 24
    lookback = 336
    max_scale = 5.2
    train = data[:train_end]
    hybrid = build_hybrid_affine_scales(
        train, lookback=lookback, max_scale=max_scale, frac_threshold=0.5, oob_coverage=0.99
    )
    mean = hybrid["mean"][0]
    scale = hybrid["std"][0]
    flat = hybrid["flat_mask"]

    # Short raw + affine snippet for flat (LULL=5) vs non-flat (HUFL=0)
    t0 = 2000
    span = 800
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex="col")
    for col, v, title in (
        (0, 0, "HUFL non-flat → window-norm at train"),
        (1, 5, "LULL flat → dataset affine only"),
    ):
        raw = train[t0 : t0 + span, v]
        aff = (raw - mean[v]) / scale[v]
        axes[0, col].plot(raw, lw=0.8, color="#1f4e79" if not flat[v] else "#8b1e1e")
        axes[0, col].set_title(f"{cols[v]} raw train  flat={bool(flat[v])} frac={hybrid['flat_frac'][v]:.2f}")
        axes[0, col].set_ylabel("raw")
        axes[1, col].plot(aff, lw=0.8, color="#1f4e79" if not flat[v] else "#8b1e1e")
        axes[1, col].axhline(max_scale, ls="--", color="gray", lw=0.7)
        axes[1, col].axhline(-max_scale, ls="--", color="gray", lw=0.7)
        axes[1, col].set_title(
            f"affine scale={scale[v]:.3g} (emp={hybrid['emp_std'][0, v]:.3g})"
        )
        axes[1, col].set_ylabel("affine space")
        axes[1, col].set_xlabel(f"t (offset {t0})")
    fig.suptitle(
        "ETTh2 hybrid flat dsnorm: flat vars skip instance norm; "
        f"scale chosen for ≥99% lb{lookback} in [-{max_scale},{max_scale}]"
    )
    fig.tight_layout()
    out = REPO / "temp" / "hybrid_flat_dsnorm_etth2_panel.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")
    print("flat_mask", flat.tolist())
    print("flat_frac", [round(float(x), 4) for x in hybrid["flat_frac"]])
    print("scales", [round(float(x), 4) for x in scale])
    print("details", hybrid["flat_details"])


if __name__ == "__main__":
    main()
