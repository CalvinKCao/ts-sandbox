#!/usr/bin/env python3
"""Compute the established probabilistic-path diagnostics for iTransformer forecasts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from compare_fixed_control_prob_path_stats_vs_mmpd import (
    batched_dtw_l1_band3,
    windowed_dtw_mean,
)
from compute_fixed_control_prob_path_stats import variogram_cloud


REPO = Path(__file__).resolve().parents[2]
RESULT = REPO / "temp/iTransformer/results/ETTh1_336_96_lr0.001_L2_D256_iTransformer_ETTh1_M_ft336_sl48_ll96_pl256_dm8_nh2_el1_dl256_df1_fctimeF_ebTrue_dtcanvas128_subset_projection_0"
REPORT = REPO / "reports/itransformer_canvas128_subset_metrics.md"
DETAILS = REPORT.with_suffix("")


def main() -> None:
    y_true = np.load(RESULT / "true.npy")
    pred = np.load(RESULT / "pred.npy")
    if y_true.shape != pred.shape:
        raise ValueError(f"prediction shape mismatch: {pred.shape} vs {y_true.shape}")
    if y_true.ndim != 3:
        raise ValueError(f"expected (windows, horizon, variates), got {y_true.shape}")

    # Baseline output layout is (N, H, V); diagnostic helpers use (N, V, H).
    y_true = np.transpose(y_true, (0, 2, 1))
    pred = np.transpose(pred, (0, 2, 1))
    g8, _ = variogram_cloud(y_true, pred, 8)
    g16, _ = variogram_cloud(y_true, pred, 16)
    metrics = {
        "dataset": "ETTh1",
        "model": "iTransformer",
        "source": str(RESULT.relative_to(REPO)),
        "n_windows": int(y_true.shape[0]),
        "n_variates": int(y_true.shape[1]),
        "horizon": int(y_true.shape[2]),
        "mse": float(np.mean((pred - y_true) ** 2)),
        "variogram_cloud_g8": g8,
        "variogram_cloud_g16": g16,
        "dtw_demeaned_band3_l8": windowed_dtw_mean(y_true, pred, 8),
        "dtw_demeaned_band3_l16": windowed_dtw_mean(y_true, pred, 16),
        "protocol": {
            "variogram": "mean_h mean((error[t+h]-error[t])^2)",
            "dtw": "mean unnormalized L1 DTW over every de-meaned sliding L=8/L=16 subwindow; Sakoe-Chiba radius=3",
        },
    }
    DETAILS.mkdir(parents=True, exist_ok=True)
    (DETAILS / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    REPORT.write_text(
        "# iTransformer canvas128-subset diagnostics\n\n"
        "| dataset | windows | variates | MSE | variogram g8 | variogram g16 | DTW L8 | DTW L16 |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|\n"
        f"| ETTh1 | {metrics['n_windows']} | {metrics['n_variates']} | {metrics['mse']:.6f} | "
        f"{g8:.6f} | {g16:.6f} | {metrics['dtw_demeaned_band3_l8']:.6f} | "
        f"{metrics['dtw_demeaned_band3_l16']:.6f} |\n\n"
        "DTW uses all sliding horizon subwindows after separately de-meaning GT and prediction; "
        "the Sakoe-Chiba radius is 3.\n"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
