#!/usr/bin/env python3
"""Fast exact-coordinate smoke test for ordinal h96 patch-refine discrimination."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from models.diffusion_tsf.ordinal_window_norm import build_global_ladder_from_training
from models.diffusion_tsf.patch_refine_geometry import PatchLocation, blend_patch_bins, coarse_edges_from_cdf
from models.diffusion_tsf.preprocessing import TimeSeriesTo2D
from temp.eval_univariate_patch_refine_ordinal_vs_mmpd import _unblended_nonoverlap_patch_batch
from utils.dual_scale_bin_filter import align_mmpd_to_binary_dataset_norm
from utils.patch_refine_ordinal_ladder import (
    assert_on_patch_refine_levels,
    assert_support_is_causal,
    legal_patch_refine_levels_dataset_z,
    snap_to_patch_refine_levels,
)


def _load_mmpd_normalization():
    path = REPO / "temp" / "MMPD" / "exp" / "normalization.py"
    if not path.is_file():
        raise FileNotFoundError(f"missing MMPD normalization helper: {path}")
    spec = importlib.util.spec_from_file_location("mmpd_normalization", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_statistics, module.normalize, module.denormalize


def main() -> None:
    rng = np.random.default_rng(123)
    device = torch.device("cpu")
    # Non-uniform data-z ladder catches accidental uniform-z snapping.
    train = np.stack(
        [
            np.linspace(-2.5, 2.4, 4096, dtype=np.float32) ** 3 / 4.0,
            np.tanh(np.linspace(-4.0, 4.0, 4096, dtype=np.float32)) * 2.0,
        ],
        axis=1,
    )
    ladder = build_global_ladder_from_training(train, tie_atol=1e-7)
    past = np.stack(
        [
            0.7 * np.sin(np.linspace(-8.0, 0.0, 336, dtype=np.float32)),
            0.9 * np.cos(np.linspace(-7.0, 0.0, 336, dtype=np.float32)),
        ],
        axis=0,
    )[None]
    legal = legal_patch_refine_levels_dataset_z(past, ladder=ladder, device=device)
    if legal.shape != (1, 2, 256):
        raise AssertionError(f"expected (1,2,256) legal support, got {legal.shape}")
    if min(np.unique(legal[0, vi]).size for vi in range(2)) != 256:
        raise AssertionError("fixture must retain all 256 distinct decoded rows")

    raw_gt = np.stack(
        [
            1.8 * np.sin(np.linspace(0.0, 5.0, 96, dtype=np.float32)),
            1.7 * np.cos(np.linspace(0.0, 4.0, 96, dtype=np.float32)),
        ],
        axis=0,
    )[None]
    binary_rows = rng.integers(0, 256, size=(1, 2, 96))
    binary = np.take_along_axis(legal[:, :, None, :], binary_rows[..., None], axis=-1)[..., 0]

    get_statistics, normalize, denormalize = _load_mmpd_normalization()
    raw_mmpd = raw_gt + rng.normal(0.0, 0.11, size=raw_gt.shape).astype(np.float32)
    raw_mmpd_t = torch.from_numpy(raw_mmpd)
    mean, std = get_statistics(torch.from_numpy(past))
    instance = normalize(raw_mmpd_t, mean, std)
    restored = denormalize(instance, mean, std).numpy()
    if not np.allclose(restored, raw_mmpd, atol=1e-5):
        raise AssertionError("MMPD instance norm did not round-trip")

    # Exercise the exact discriminator feed-in scaler conversion with different
    # source/destination training scalers while preserving paired GT windows.
    mmpd_mean = np.array([10.0, -2.0], dtype=np.float64)
    mmpd_std = np.array([3.0, 2.0], dtype=np.float64)
    binary_mean = np.array([11.0, -3.5], dtype=np.float64)
    binary_std = np.array([2.0, 4.0], dtype=np.float64)
    raw_gt_physical = raw_gt * binary_std[None, :, None] + binary_mean[None, :, None]
    binary_gt = raw_gt.astype(np.float32)
    mmpd_gt = ((raw_gt_physical - mmpd_mean[None, :, None]) / mmpd_std[None, :, None]).astype(np.float32)
    mmpd_fake = ((raw_mmpd * binary_std[None, :, None] + binary_mean[None, :, None] - mmpd_mean[None, :, None]) / mmpd_std[None, :, None]).astype(np.float32)
    aligned_mmpd, _stats = align_mmpd_to_binary_dataset_norm(
        binary_y_true=binary_gt,
        mmpd_y_true=mmpd_gt,
        mmpd_fakes=mmpd_fake,
        mmpd_mean=mmpd_mean,
        mmpd_std=mmpd_std,
        binary_mean=binary_mean,
        binary_std=binary_std,
    )
    gt, gt_stats = snap_to_patch_refine_levels(binary_gt, legal)
    mmpd, mmpd_stats = snap_to_patch_refine_levels(aligned_mmpd, legal)
    assert_on_patch_refine_levels(binary, legal)
    assert_on_patch_refine_levels(gt, legal)
    assert_on_patch_refine_levels(mmpd, legal)
    assert_support_is_causal(
        past, raw_gt, raw_gt + 100.0, ladder=ladder, canvas_height=256, device=device,
    )

    # Three stride-6 crops cover 0:8, 6:14, and 12:20.  The raw coherent
    # metric must retain the first and third only: each retained forecast is
    # decoded from one CDF crop, never a blend of the overlapping votes.
    def _cdf(rows: torch.Tensor, height: int) -> torch.Tensor:
        grid = torch.arange(height, dtype=torch.long).view(1, 1, height, 1)
        return (grid <= rows.unsqueeze(-2)).to(torch.float32)

    coarse_rows = torch.full((1, 2, 96), 127, dtype=torch.long)
    coarse_cdf = _cdf(coarse_rows // 16, 16)
    patch_rows = torch.full((3, 8), 15, dtype=torch.long)
    patch_cdf = _cdf(patch_rows.unsqueeze(1), 32)
    raw_result = {
        "future_2d_coarse": coarse_cdf,
        "patch_cdf_unblended": patch_cdf,
        "patch_locations": [
            PatchLocation(flat_index=0, batch_index=0, variate_index=0, row0=112, col0=0),
            PatchLocation(flat_index=0, batch_index=0, variate_index=0, row0=112, col0=6),
            PatchLocation(flat_index=0, batch_index=0, variate_index=0, row0=112, col0=12),
        ],
    }
    raw_pred, raw_gt_snap, raw_past, parents, starts, variates, raw_info = _unblended_nonoverlap_patch_batch(
        result=raw_result,
        target=torch.from_numpy(raw_gt),
        past=torch.from_numpy(past),
        legal_levels=legal,
        canvas_height=256,
        patch_height=32,
        patch_width=8,
    )
    assert raw_pred.shape == raw_gt_snap.shape == (2, 1, 8)
    assert raw_past.shape == (2, 1, 336)
    assert parents.tolist() == [0, 0] and variates.tolist() == [0, 0]
    assert starts.tolist() == [0, 12]
    assert raw_info["selected"] == 2
    raw_support = np.repeat(legal[:, :1], repeats=2, axis=0)
    assert_on_patch_refine_levels(raw_pred, raw_support)
    assert_on_patch_refine_levels(raw_gt_snap, raw_support)
    blended, votes = blend_patch_bins(
        patch_cdf,
        raw_result["patch_locations"],
        coarse_edges_from_cdf(coarse_cdf, canvas_height=256),
        canvas_height=256,
        patch_height=32,
        patch_width=8,
    )
    blended_rows = TimeSeriesTo2D.bin_indices_from_cdf(blended)
    if not torch.equal(blended_rows, blended_rows.round().to(torch.long)):
        raise AssertionError("overlap blending did not round to integer absolute rows")
    if not bool((votes[0, 0, 6:8] == 2).all()):
        raise AssertionError("fixture did not exercise the two-timestep stride-6 overlap")

    out = REPO / "reports" / "h96_ordinal_patch_refine_disc_contract"
    out.mkdir(parents=True, exist_ok=True)
    fig, (ax, rows_ax) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    x_past = np.arange(-336, 0)
    x = np.arange(96)
    for level in legal[0, 0]:
        ax.axhline(level, color="0.55", lw=0.25, alpha=0.13, zorder=0)
    ax.plot(x_past, past[0, 0], color="0.45", lw=1.2, label="lookback")
    ax.plot(x, gt[0, 0], color="black", lw=1.25, label="GT snapped")
    ax.plot(x, binary[0, 0], color="#1f77b4", lw=1.0, label="binary")
    ax.plot(x, mmpd[0, 0], color="#ff7f0e", lw=1.0, label="MMPD snapped")
    ax.set_title("All discriminator inputs lie on binary's 256-row ordinal ladder")
    ax.legend(ncol=4, loc="upper right")
    ax.grid(alpha=0.15)
    rows_ax.plot(x, np.argmin(np.abs(binary[0, 0, :, None] - legal[0, 0]), axis=-1), label="binary row")
    rows_ax.plot(x, np.argmin(np.abs(gt[0, 0, :, None] - legal[0, 0]), axis=-1), label="GT row")
    rows_ax.plot(x, np.argmin(np.abs(mmpd[0, 0, :, None] - legal[0, 0]), axis=-1), label="MMPD row")
    rows_ax.set_ylim(-2, 257)
    rows_ax.set_ylabel("absolute ordinal row")
    rows_ax.set_xlabel("forecast timestep")
    rows_ax.legend(ncol=3)
    rows_ax.grid(alpha=0.2)
    fig.tight_layout()
    plot = out / "synthetic_mmpd_binary_256_ladder.png"
    fig.savefig(plot, dpi=180)
    print(
        f"PASS: 256 rows, GT max snap={gt_stats['max_abs_snap_delta']:.4g}, "
        f"MMPD max snap={mmpd_stats['max_abs_snap_delta']:.4g}; wrote {plot}"
    )


if __name__ == "__main__":
    main()
