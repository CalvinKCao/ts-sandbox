#!/usr/bin/env python3
"""One-before/after panel per train-window augmentation (ETTh1)."""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

from models.diffusion_tsf.ordinal_window_norm import build_global_ladder_from_training
from models.diffusion_tsf.train_multivariate_pipeline import load_dataset
from models.diffusion_tsf.train_window_aug import (
    ALL_AUGS,
    apply_stacked_augs,
    estimate_variate_periodicity,
    _join_past_future,
)


def main() -> None:
    out_dir = os.path.join(REPO, "temp", "train_window_aug_viz")
    os.makedirs(out_dir, exist_ok=True)

    # Force ordinal on for ladder / bounds.
    import models.diffusion_tsf.train_multivariate_pipeline as m

    m.USE_ORDINAL_WINDOW_NORM = True
    m.LOOKBACK_LENGTH = 336
    m.FORECAST_LENGTH = 720
    m.LOOKBACK_OVERLAP = 8
    m.WINDOW_STRIDE = 60

    train_ds, _, _, stats = load_dataset(
        "ETTh1",
        lookback=336,
        horizon=720,
        stride=60,
        lookback_overlap=8,
        use_ordinal_window_norm=True,
    )
    ladder = stats["ordinal_ladder"]
    data = train_ds.data.numpy()
    flags = estimate_variate_periodicity(data)
    print("periodic flags:", flags.tolist())

    # Pick a non-flat window on variate 0.
    idx = min(40, len(train_ds) - 2)
    past_t, future_t = train_ds[idx]
    # raw from data (train_ds may yield ranks)
    start = idx * train_ds.stride
    past = data[start : start + train_ds.lookback].T.astype(np.float32)
    target_start = start + train_ds.lookback - train_ds.lookback_overlap
    target_end = start + train_ds.lookback + train_ds.horizon
    future = data[target_start:target_end].T.astype(np.float32)
    donor_past = data[
        (idx + 1) * train_ds.stride : (idx + 1) * train_ds.stride + train_ds.lookback
    ].T.astype(np.float32)

    overlap = train_ds.lookback_overlap
    # Temporarily disable periodicity so heavy augs still visualize.
    force_flags = np.zeros_like(flags)

    for name in ALL_AUGS:
        rng = np.random.default_rng(abs(hash(name)) % (2**31))
        past_a, fut_a, applied = apply_stacked_augs(
            past,
            future,
            overlap=overlap,
            ladder=ladder,
            periodic_flags=force_flags,
            rng=rng,
            donor_past=donor_past,
            force_names=[name],
        )
        full0 = _join_past_future(past, future, overlap)[0]
        full1 = _join_past_future(past_a, fut_a, overlap)[0]
        lb = train_ds.lookback

        fig, ax = plt.subplots(figsize=(12, 3.2))
        ax.plot(full0, color="#888", lw=1.2, label="original")
        ax.plot(full1, color="#c0392b", lw=1.4, label=name)
        ax.axvline(lb - 0.5, color="k", ls="--", lw=0.8, alpha=0.5)
        ax.set_title(f"ETTh1 v0  {name}  applied={applied}")
        ax.legend(loc="upper right")
        ax.set_xlabel("t (lookback | horizon)")
        fig.tight_layout()
        path = os.path.join(out_dir, f"{name}.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print("wrote", path)


if __name__ == "__main__":
    main()
