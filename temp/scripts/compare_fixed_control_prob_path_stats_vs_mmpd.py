#!/usr/bin/env python3
"""Compare fixed-control probabilistic paths with matching MMPD saved paths."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from compute_fixed_control_prob_path_stats import (
    PACKS,
    SEED,
    select_one_path_per_window,
    variogram_cloud,
)


REPO = Path(__file__).resolve().parents[2]
REPORT = REPO / "reports" / "fixed_control_prob_path_stats_vs_mmpd.md"
DETAILS = REPORT.with_suffix("")
SPECS = {
    "traffic": {
        "fixed": PACKS / "staged_dpmpp_samples_traffic.npz",
        "mmpd": REPO / "results/datasets/07-29-0151-h96-ordinal-binary-nonordinal-mmpd-mmpd/raw/mmpd_traffic.npz",
    },
    "exchange_rate": {
        "fixed": PACKS / "staged_dpmpp_samples_exchange_rate.npz",
        "mmpd": REPO / "results/datasets/07-29-0151-h96-ordinal-binary-nonordinal-mmpd-mmpd/raw/mmpd_exchange_rate.npz",
    },
    "PeMS": {
        "fixed": PACKS / "staged_dpmpp_samples_PeMS.npz",
        "mmpd": REPO / "results/datasets/08-06-mmpd-decoder-paper-lb336-hz96-matched-c128-four/raw/mmpd_PeMS.npz",
    },
}


def batched_dtw_l1_band3(target: np.ndarray, pred: np.ndarray, radius: int = 3) -> np.ndarray:
    """Unnormalized scalar L1 DTW within a Sakoe-Chiba band, batched over paths."""
    n_paths, horizon = target.shape
    prev = np.full((n_paths, horizon + 1), np.inf, dtype=np.float64)
    prev[:, 0] = 0.0
    for i in range(1, horizon + 1):
        curr = np.full_like(prev, np.inf)
        for j in range(max(1, i - radius), min(horizon, i + radius) + 1):
            curr[:, j] = np.abs(target[:, i - 1] - pred[:, j - 1]) + np.minimum(
                prev[:, j - 1], np.minimum(prev[:, j], curr[:, j - 1]),
            )
        prev = curr
    return prev[:, -1]


def windowed_dtw_mean(y_true: np.ndarray, pred: np.ndarray, length: int) -> float:
    """Mean band-3 DTW over every de-meaned sliding subwindow of each path."""
    target = np.lib.stride_tricks.sliding_window_view(y_true, length, axis=-1)
    forecast = np.lib.stride_tricks.sliding_window_view(pred, length, axis=-1)
    target = target - target.mean(axis=-1, keepdims=True)
    forecast = forecast - forecast.mean(axis=-1, keepdims=True)
    target = target.reshape(-1, length)
    forecast = forecast.reshape(-1, length)
    total = 0.0
    count = 0
    for start in range(0, len(target), 20_000):
        stop = min(start + 20_000, len(target))
        values = batched_dtw_l1_band3(target[start:stop], forecast[start:stop])
        total += float(values.sum())
        count += len(values)
    return total / count


def summarize(path: Path, dataset: str) -> dict:
    with np.load(path, allow_pickle=False) as pack:
        y_true = np.asarray(pack["y_true"])
        samples = np.asarray(pack["samples"])
    pred, choices = select_one_path_per_window(samples, dataset)
    if y_true.shape != pred.shape:
        raise ValueError(f"{path}: {y_true.shape} != {pred.shape}")
    g8, _ = variogram_cloud(y_true, pred, 8)
    g16, _ = variogram_cloud(y_true, pred, 16)
    return {
        "source": str(path.relative_to(REPO)),
        "n_windows": int(y_true.shape[0]),
        "n_variates": int(y_true.shape[1]),
        "n_saved_samples": int(samples.shape[2]),
        "sample_choices": choices.tolist(),
        "variogram_cloud_g8": g8,
        "variogram_cloud_g16": g16,
        "dtw_demeaned_band3_l8": windowed_dtw_mean(y_true, pred, 8),
        "dtw_demeaned_band3_l16": windowed_dtw_mean(y_true, pred, 16),
    }


def main() -> None:
    DETAILS.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset, paths in SPECS.items():
        fixed = summarize(paths["fixed"], dataset)
        mmpd = summarize(paths["mmpd"], dataset)
        rows.append({
            "dataset": dataset,
            "fixed_control": fixed,
            "mmpd": mmpd,
            "delta_mmpd_minus_fixed": {
                key: float(mmpd[key] - fixed[key])
                for key in (
                    "variogram_cloud_g8",
                    "variogram_cloud_g16",
                    "dtw_demeaned_band3_l8",
                    "dtw_demeaned_band3_l16",
                )
            },
        })
    output = {
        "fixed_condition": "current fixed HPs with dataset-policy default max_scale",
        "sampling": "One seeded whole-horizon saved trajectory per window, shared across variates.",
        "seed": SEED,
        "pool_note": "Fixed-control and MMPD runs retain their own saved evaluation pools; values are not window-paired.",
        "variogram_metric": "mean_h mean((error[t+h]-error[t])^2), lower is better",
        "dtw_metric": "mean unnormalized L1 DTW over every de-meaned sliding L=8/L=16 subwindow, Sakoe-Chiba radius=3; lower is better",
        "rows": rows,
    }
    (DETAILS / "comparison.json").write_text(json.dumps(output, indent=2) + "\n")
    lines = [
        "# Fixed HP + default max-scale vs MMPD: probabilistic path statistics",
        "",
        output["sampling"],
        "",
        output["pool_note"],
        "",
        "| dataset | fixed windows | MMPD windows | fixed vario g8 | MMPD vario g8 | fixed vario g16 | MMPD vario g16 | fixed DTW L8 | MMPD DTW L8 | fixed DTW L16 | MMPD DTW L16 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        fixed, mmpd = row["fixed_control"], row["mmpd"]
        lines.append(
            f"| {row['dataset']} | {fixed['n_windows']} | {mmpd['n_windows']} | "
            f"{fixed['variogram_cloud_g8']:.6f} | {mmpd['variogram_cloud_g8']:.6f} | "
            f"{fixed['variogram_cloud_g16']:.6f} | {mmpd['variogram_cloud_g16']:.6f} | "
            f"{fixed['dtw_demeaned_band3_l8']:.6f} | {mmpd['dtw_demeaned_band3_l8']:.6f} | "
            f"{fixed['dtw_demeaned_band3_l16']:.6f} | {mmpd['dtw_demeaned_band3_l16']:.6f} |"
        )
    REPORT.write_text("\n".join(lines) + "\n")
    for row in rows:
        print(row["dataset"], row["delta_mmpd_minus_fixed"])


if __name__ == "__main__":
    main()
