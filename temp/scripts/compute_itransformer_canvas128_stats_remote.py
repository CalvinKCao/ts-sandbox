#!/usr/bin/env python3
"""Compute canvas128 DTW and variogram diagnostics for completed iTransformer runs."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
SUMMARY_ROOT = ROOT / "temp/baselines_canvas128_subset/results/itransformer"
RESULTS_ROOT = ROOT / "temp/iTransformer/results"


def batched_dtw_l1_band3(target: np.ndarray, pred: np.ndarray, radius: int = 3) -> np.ndarray:
    """Unnormalized L1 DTW within a Sakoe-Chiba radius, batched over paths."""
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
    """Mean band-3 DTW over every separately de-meaned sliding subwindow."""
    total = 0.0
    count = 0
    # Avoid materializing every sliding window for the long, stride-1 test packs.
    for start in range(0, y_true.shape[0], 512):
        stop = min(start + 512, y_true.shape[0])
        target = np.lib.stride_tricks.sliding_window_view(y_true[start:stop], length, axis=-1)
        forecast = np.lib.stride_tricks.sliding_window_view(pred[start:stop], length, axis=-1)
        target = target - target.mean(axis=-1, keepdims=True)
        forecast = forecast - forecast.mean(axis=-1, keepdims=True)
        target = target.reshape(-1, length)
        forecast = forecast.reshape(-1, length)
        for path_start in range(0, len(target), 20_000):
            path_stop = min(path_start + 20_000, len(target))
            values = batched_dtw_l1_band3(target[path_start:path_stop], forecast[path_start:path_stop])
            total += float(values.sum())
            count += len(values)
    return total / count


def variogram_cloud(y_true: np.ndarray, pred: np.ndarray, max_gap: int) -> float:
    error = pred - y_true
    values = []
    for gap in range(1, max_gap + 1):
        values.append(float(np.mean((error[..., gap:] - error[..., :-gap]) ** 2)))
    return float(np.mean(values))


def select_result(dataset: str, tag: str) -> Path:
    hits = sorted(RESULTS_ROOT.glob(f"{dataset}_336_96_{tag}_iTransformer_*"))
    hits = [p for p in hits if (p / "pred.npy").is_file() and (p / "true.npy").is_file()]
    if len(hits) != 1:
        raise RuntimeError(f"{dataset}: expected one selected pack for {tag}, found {hits}")
    return hits[0]


def main() -> None:
    selected = set(sys.argv[2:])
    max_windows = int(os.environ.get("MAX_WINDOWS", "0"))
    rows = []
    for summary_path in sorted(SUMMARY_ROOT.glob("*/itransformer_summary.json")):
        summary = json.loads(summary_path.read_text())
        dataset = str(summary["dataset"])
        if selected and dataset not in selected:
            continue
        tag = str(summary["best"]["tag"])
        result_dir = select_result(dataset, tag)
        y_true = np.load(result_dir / "true.npy")
        pred = np.load(result_dir / "pred.npy")
        if y_true.shape != pred.shape or y_true.ndim != 3:
            raise RuntimeError(f"{dataset}: invalid pack shapes pred={pred.shape}, true={y_true.shape}")
        # iTransformer saves (window, horizon, variate); protocol uses (window, variate, horizon).
        y_true = y_true.transpose(0, 2, 1)
        pred = pred.transpose(0, 2, 1)
        n_total_windows = int(y_true.shape[0])
        if max_windows and n_total_windows > max_windows:
            rng = np.random.default_rng(20260812 + sum(map(ord, dataset)))
            indices = np.sort(rng.choice(n_total_windows, size=max_windows, replace=False))
            y_true = y_true[indices]
            pred = pred[indices]
        rows.append({
            "dataset": dataset,
            "source": str(result_dir.relative_to(ROOT)),
            "n_windows": int(y_true.shape[0]),
            "source_n_windows": n_total_windows,
            "n_variates": int(y_true.shape[1]),
            "horizon": int(y_true.shape[2]),
            "mse": float(np.mean((pred - y_true) ** 2)),
            "variogram_cloud_g8": variogram_cloud(y_true, pred, 8),
            "variogram_cloud_g16": variogram_cloud(y_true, pred, 16),
            "dtw_demeaned_band3_l8": windowed_dtw_mean(y_true, pred, 8),
            "dtw_demeaned_band3_l16": windowed_dtw_mean(y_true, pred, 16),
        })
    print(json.dumps({
        "protocol": {
            "variogram": "mean over gaps h=1..G of mean((error[t+h]-error[t])^2)",
        "dtw": "mean unnormalized L1 DTW across all separately de-meaned sliding L=8/L=16 subwindows, Sakoe-Chiba radius=3",
        },
        "sampling": (
            f"seeded random {max_windows}-window subset per dataset" if max_windows
            else "full saved evaluation pack"
        ),
        "rows": rows,
    }, indent=2))


if __name__ == "__main__":
    main()
