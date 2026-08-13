#!/usr/bin/env python3
"""Paired-origin Quad-T vs MMPD path DTW and variogram diagnostics.

Uses the exact binary origins and one seeded, whole-horizon saved trajectory
per origin from both models.  No resampling, interpolation, snapping, or
post-hoc variance scaling is applied.  DTW de-means each native subwindow.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compute_shape_descriptor_stats_paired_native import (
    BINARY_REL,
    COMMON_CHANNELS,
    DATASETS,
    MMPD_REL,
    as_nvh,
    load_probabilistic,
    nearest_alignment,
    select_sample,
)


SEED = 20260812
EPS = 1e-12
MODELS = ("binary_quad_t", "mmpd_probabilistic")


def variogram_cloud(y_true: np.ndarray, prediction: np.ndarray, max_gap: int) -> tuple[float, list[float]]:
    error = prediction.astype(np.float64) - y_true.astype(np.float64)
    values = [float(np.mean((error[..., gap:] - error[..., :-gap]) ** 2)) for gap in range(1, max_gap + 1)]
    return float(np.mean(values)), values


def batched_dtw_l1_band3(target: np.ndarray, pred: np.ndarray, radius: int = 3) -> np.ndarray:
    """Unnormalized L1 DTW for rows of equal-length native curves."""
    n_paths, length = target.shape
    prev = np.full((n_paths, length + 1), np.inf, dtype=np.float64)
    prev[:, 0] = 0.0
    for i in range(1, length + 1):
        current = np.full_like(prev, np.inf)
        for j in range(max(1, i - radius), min(length, i + radius) + 1):
            current[:, j] = np.abs(target[:, i - 1] - pred[:, j - 1]) + np.minimum(
                prev[:, j - 1], np.minimum(prev[:, j], current[:, j - 1])
            )
        prev = current
    return prev[:, -1]


def demeaned_dtw(y_true: np.ndarray, pred: np.ndarray, length: int) -> dict[str, float | int]:
    target = np.lib.stride_tricks.sliding_window_view(y_true, length, axis=-1)
    forecast = np.lib.stride_tricks.sliding_window_view(pred, length, axis=-1)
    target = target - target.mean(axis=-1, keepdims=True)
    forecast = forecast - forecast.mean(axis=-1, keepdims=True)
    target = target.reshape(-1, length)
    forecast = forecast.reshape(-1, length)
    totals: list[np.ndarray] = []
    for start in range(0, len(target), 20_000):
        stop = min(start + 20_000, len(target))
        totals.append(batched_dtw_l1_band3(target[start:stop], forecast[start:stop]))
    values = np.concatenate(totals)
    return {
        "mean": float(values.mean()), "median": float(np.median(values)),
        "p90": float(np.quantile(values, .9)), "n_subwindows": int(len(values)),
    }


def build_row(dataset: str, model: str, y_true: np.ndarray, pred: np.ndarray, source: Path, sample_ids: np.ndarray | None) -> tuple[dict[str, Any], dict[str, Any]]:
    g8, per_gap_8 = variogram_cloud(y_true, pred, 8)
    g16, per_gap_16 = variogram_cloud(y_true, pred, 16)
    l8, l16 = demeaned_dtw(y_true, pred, 8), demeaned_dtw(y_true, pred, 16)
    compact = {
        "dataset": dataset, "model": model, "n_origins": int(y_true.shape[0]), "n_variates": int(y_true.shape[1]),
        "variogram_g8": g8, "variogram_g16": g16,
        "dtw_demeaned_band3_l8": l8["mean"], "dtw_demeaned_band3_l16": l16["mean"],
        "dtw_l8_n_subwindows": l8["n_subwindows"], "dtw_l16_n_subwindows": l16["n_subwindows"],
    }
    detail = {**compact, "source": str(source), "sample_ids": None if sample_ids is None else sample_ids.tolist(),
              "variogram_per_gap_1_8": per_gap_8, "variogram_per_gap_1_16": per_gap_16,
              "dtw_l8": l8, "dtw_l16": l16}
    return compact, detail


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    compact_rows: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    for dataset in args.datasets:
        binary_path = args.binary_root / "results/datasets" / BINARY_REL[dataset]
        mmpd_path = args.reference_root / "results/datasets" / MMPD_REL[dataset]
        binary_y, binary_samples = load_probabilistic(binary_path)
        mmpd_y, mmpd_samples = load_probabilistic(mmpd_path)
        channels = COMMON_CHANNELS.get(dataset)
        if channels is not None:
            binary_y, binary_samples = binary_y[:, channels["binary_quad_t"]], binary_samples[:, channels["binary_quad_t"]]
            mmpd_y, mmpd_samples = mmpd_y[:, channels["mmpd"]], mmpd_samples[:, channels["mmpd"]]
        mmpd_rows, alignment = nearest_alignment(binary_y, mmpd_y, f"{dataset}/MMPD")
        indices = np.arange(len(binary_y))
        binary_pred, binary_ids = select_sample(binary_samples, indices, dataset, "binary_quad_t")
        mmpd_pred, mmpd_ids = select_sample(mmpd_samples, mmpd_rows, dataset, "mmpd")
        binary_compact, binary_detail = build_row(dataset, "binary_quad_t", binary_y, binary_pred, binary_path, binary_ids)
        mmpd_compact, mmpd_detail = build_row(dataset, "mmpd_probabilistic", mmpd_y[mmpd_rows], mmpd_pred, mmpd_path, mmpd_ids)
        compact_rows.extend((binary_compact, mmpd_compact))
        entries.append({"dataset": dataset, "channel_selection": channels, "alignment": alignment,
                        "binary": binary_detail, "mmpd": mmpd_detail,
                        "mmpd_minus_binary": {key: float(mmpd_compact[key] - binary_compact[key]) for key in ("variogram_g8", "variogram_g16", "dtw_demeaned_band3_l8", "dtw_demeaned_band3_l16")}})
        print(f"done {dataset}: {len(binary_y)} paired origins", flush=True)
    write_csv(args.output_dir / "per_dataset_metrics.csv", compact_rows)
    payload = {"protocol": {"sources": "same saved paths as paired native texture descriptors", "alignment": "each binary test GT horizon is matched to a unique MMPD GT horizon by native 96-step MSE <= 1e-8", "sampling": "one deterministic seeded full-horizon saved trajectory per paired origin; binary=Quad-T, MMPD=probabilistic", "variogram": "raw-scale mean over h=1..G of mean((error[t+h]-error[t])^2), lower is better", "dtw": "raw-scale, each native subwindow independently de-meaned; unnormalized L1 DTW, Sakoe-Chiba radius=3; lower is better", "no_interpolation": True, "no_variance_scaling": True}, "seed": SEED, "datasets": entries}
    (args.output_dir / "details.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    by_dataset = {(row["dataset"], row["model"]): row for row in compact_rows}
    lines = ["# Paired probabilistic path diagnostics", "", "Same saved Quad-T/MMPD sample paths and identical matched test origins as the native texture analysis. Lower is better. Variogram is raw-scale; DTW de-means each native subwindow but does not variance-normalize.", "", "| dataset | origins | variates | binary vario g8 | MMPD vario g8 | binary vario g16 | MMPD vario g16 | binary DTW L8 | MMPD DTW L8 | binary DTW L16 | MMPD DTW L16 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for dataset in args.datasets:
        binary = by_dataset[(dataset, "binary_quad_t")]
        mmpd = by_dataset[(dataset, "mmpd_probabilistic")]
        lines.append(f"| {dataset} | {binary['n_origins']} | {binary['n_variates']} | {binary['variogram_g8']:.6f} | {mmpd['variogram_g8']:.6f} | {binary['variogram_g16']:.6f} | {mmpd['variogram_g16']:.6f} | {binary['dtw_demeaned_band3_l8']:.6f} | {mmpd['dtw_demeaned_band3_l8']:.6f} | {binary['dtw_demeaned_band3_l16']:.6f} | {mmpd['dtw_demeaned_band3_l16']:.6f} |")
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
