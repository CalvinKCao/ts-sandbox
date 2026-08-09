#!/usr/bin/env python3
"""Visualize unique-segment patch-refine conditioning (parents + stuffed prev)."""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

from models.diffusion_tsf.patch_refine import build_patch_aux_channels, naive_upscale_coarse_cdf
from models.diffusion_tsf.patch_refine_geometry import coarse_edges_from_cdf
from models.diffusion_tsf.patch_refine_segments import (
    UniquePatchSegmentDataset,
    compress_prev_refine_32_to_16,
    extract_prev_refine_crops,
    iter_unique_segment_starts,
    locations_for_fixed_col0,
    sample_parent_start,
)


OUT_DIR = os.path.join(REPO, "temp", "viz_patch_refine_unique_segments")


def _make_series(T: int = 800, V: int = 2, seed: int = 0) -> torch.Tensor:
    rng = np.random.RandomState(seed)
    t = np.arange(T, dtype=np.float32)
    series = []
    for v in range(V):
        series.append(
            0.4 * np.sin(2 * np.pi * t / (24 + 5 * v))
            + 0.2 * np.sin(2 * np.pi * t / (168 + 10 * v))
            + 0.05 * rng.randn(T).astype(np.float32)
        )
    return torch.from_numpy(np.stack(series, axis=1))


def _encode_hir(values: torch.Tensor, canvas_h: int = 256, max_scale: float = 3.5) -> torch.Tensor:
    # values: (B,V,W)
    x = values.clamp(-max_scale, max_scale)
    pos = (x + max_scale) / (2 * max_scale) * canvas_h
    bins = pos.long().clamp(0, canvas_h - 1)
    rows = torch.arange(canvas_h).view(1, 1, canvas_h, 1)
    return (rows <= bins.unsqueeze(2)).float()


def _encode_coarse(values: torch.Tensor, coarse_h: int = 16, max_scale: float = 3.5) -> torch.Tensor:
    x = values.clamp(-max_scale, max_scale)
    pos = (x + max_scale) / (2 * max_scale) * coarse_h
    bins = pos.long().clamp(0, coarse_h - 1)
    rows = torch.arange(coarse_h).view(1, 1, coarse_h, 1)
    return (rows <= bins.unsqueeze(2)).float()


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    lookback, horizon, overlap = 336, 96, 8
    patch_w, patch_h, col_stride = 8, 32, 6
    canvas_h, coarse_h = 256, 16
    data = _make_series()
    starts = iter_unique_segment_starts(
        data.shape[0],
        lookback=lookback,
        horizon=horizon,
        overlap=overlap,
        patch_width=patch_w,
        segment_stride=1,
    )
    assert starts, "expected unique segment starts"
    # Pick a mid-series segment so prev [t-6,t+2) exists.
    t = starts[len(starts) // 2]
    assert t - 6 >= lookback - overlap

    # Panel 1: same t, three epoch parents
    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    for ep, ax in enumerate(axes):
        S = sample_parent_start(
            t,
            epoch=ep,
            series_id=0,
            lookback=lookback,
            horizon=horizon,
            overlap=overlap,
            patch_width=patch_w,
            series_len=int(data.shape[0]),
        )
        fut_start = S + lookback - overlap
        col0 = t - fut_start
        series_1d = data[:, 0].numpy()
        ax.plot(series_1d, color="0.7", lw=0.8)
        ax.axvspan(S, S + lookback, color="C0", alpha=0.15, label="lookback")
        ax.axvspan(fut_start, S + lookback + horizon, color="C1", alpha=0.15, label="horizon")
        ax.axvspan(t, t + patch_w, color="C3", alpha=0.45, label="refine 8")
        ax.axvspan(t - 6, t + 2, color="C2", alpha=0.35, label="prev [t-6,t+2)")
        ax.set_xlim(S - 20, S + lookback + horizon + 20)
        ax.set_title(f"epoch={ep} parent_S={S} col0={col0} patch_time0={col0}")
        if ep == 0:
            ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(f"Unique segment t={t}: random parents across epochs")
    fig.tight_layout()
    parent_path = os.path.join(OUT_DIR, "parents_across_epochs.png")
    fig.savefig(parent_path, dpi=120)
    plt.close(fig)

    # Build one sample canvas and show stuffed aux
    ds = UniquePatchSegmentDataset(
        data,
        lookback=lookback,
        horizon=horizon,
        overlap=overlap,
        patch_width=patch_w,
        segment_stride=1,
    )
    ds.set_epoch(0)
    # Find index of chosen t
    idx = starts.index(t)
    past, future, col0_t = ds[idx]
    past_b = past.unsqueeze(0)
    future_b = future.unsqueeze(0)
    col0 = col0_t.view(1)

    coarse = _encode_coarse(future_b)
    hir = _encode_hir(future_b, canvas_h=canvas_h)
    naive = naive_upscale_coarse_cdf(coarse, canvas_h)
    edges = coarse_edges_from_cdf(coarse, canvas_height=canvas_h)
    locations = locations_for_fixed_col0(
        edges, col0, canvas_height=canvas_h, patch_height=patch_h, patch_width=patch_w,
    )
    prev_32 = extract_prev_refine_crops(
        hir,
        locations,
        patch_height=patch_h,
        patch_width=patch_w,
        col_stride=col_stride,
        coarse_edges=edges,
        canvas_height=canvas_h,
    )
    prev_16 = compress_prev_refine_32_to_16(prev_32)
    assert prev_16.abs().sum() > 0, "prev refine should be nonempty for mid-series t"

    aux, patch_coarse_bin, patch_time0 = build_patch_aux_channels(
        naive,
        edges,
        locations,
        patch_height=patch_h,
        patch_width=patch_w,
        canvas_height=canvas_h,
        coarse_height=coarse_h,
        horizon_width=int(hir.shape[-1]),
        prev_refine_16=prev_16,
    )
    # Canvas = xt placeholder | coord | aux3  — show aux + prev for clarity
    coord = torch.linspace(1.0, -1.0, patch_h).view(1, 1, patch_h, 1).expand(1, 1, patch_h, patch_w)
    xt = torch.zeros(1, 1, patch_h, patch_w)
    # Use first variate location only for viz
    loc0 = [locations[0]]
    aux0 = aux[0:1]
    canvas = torch.cat([xt, coord, aux0], dim=1)  # (1,5,32,8)

    names = ["xt (zeros)", "coord", "aux0 naive", "aux1 coarse_cell+prev", "aux2 time_map"]
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.2))
    for i, ax in enumerate(axes):
        im = ax.imshow(canvas[0, i].numpy(), aspect="auto", cmap="gray", vmin=0, vmax=1)
        ax.set_title(names[i], fontsize=9)
        ax.set_xlabel("w")
        if i == 0:
            ax.set_ylabel("h")
        if i == 3:
            ax.axhline(15.5, color="red", lw=1.5)
            ax.text(0.1, 2, "prev 16x8", color="red", fontsize=8)
            ax.text(0.1, 24, "coarse_cell", color="cyan", fontsize=8)
    fig.suptitle(
        f"Stuffed prev into aux1 top-16 (col0={int(col0.item())} "
        f"patch_time0={int(patch_time0[0].item())})"
    )
    fig.tight_layout()
    canvas_path = os.path.join(OUT_DIR, "canvas_channels_with_prev.png")
    fig.savefig(canvas_path, dpi=120)
    plt.close(fig)

    # Dropout-blank example
    aux_blank, _, _ = build_patch_aux_channels(
        naive,
        edges,
        locations,
        patch_height=patch_h,
        patch_width=patch_w,
        canvas_height=canvas_h,
        coarse_height=coarse_h,
        horizon_width=int(hir.shape[-1]),
        prev_refine_16=torch.zeros_like(prev_16),
    )
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(aux[0, 1].numpy(), aspect="auto", cmap="gray", vmin=0, vmax=1)
    axes[0].axhline(15.5, color="red", lw=1.2)
    axes[0].set_title("aux1 with prev")
    axes[1].imshow(aux_blank[0, 1].numpy(), aspect="auto", cmap="gray", vmin=0, vmax=1)
    axes[1].axhline(15.5, color="red", lw=1.2)
    axes[1].set_title("aux1 prev blanked (dropout)")
    fig.tight_layout()
    drop_path = os.path.join(OUT_DIR, "prev_cond_dropout.png")
    fig.savefig(drop_path, dpi=120)
    plt.close(fig)

    # Shape / semantics asserts
    assert aux.shape == (len(locations), 3, 32, 8)
    top = aux[0, 1, :16]
    bot = aux[0, 1, 16:]
    assert not torch.allclose(top, bot[:16] * 0 + bot.mean()), "sanity"
    assert not torch.allclose(top, torch.zeros_like(top))
    # Bottom half should be nearly H-constant per column
    assert torch.allclose(bot[0:1], bot, atol=1e-5)

    # Parent col0 tracking across epochs
    cols = []
    for ep in range(3):
        S = sample_parent_start(
            t, epoch=ep, series_id=0, lookback=lookback, horizon=horizon,
            overlap=overlap, patch_width=patch_w, series_len=int(data.shape[0]),
        )
        cols.append(t - (S + lookback - overlap))
    assert len(set(cols)) >= 1
    print(f"OK: wrote {parent_path}")
    print(f"OK: wrote {canvas_path}")
    print(f"OK: wrote {drop_path}")
    print(f"OK: segment starts={len(starts)} t={t} epoch_col0s={cols}")


if __name__ == "__main__":
    main()
