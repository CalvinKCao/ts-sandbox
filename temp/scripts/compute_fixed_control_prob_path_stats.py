#!/usr/bin/env python3
"""Probabilistic path metrics for the fixed-HP/default-max-scale controls."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
PACKS = REPO / "temp" / "fixed_control_stats_packs"
OUTPUT = REPO / "temp" / "fixed_control_prob_path_stats"
SEED = 20260810
SPECS = {
    "traffic": "staged_dpmpp_samples_traffic.npz",
    "exchange_rate": "staged_dpmpp_samples_exchange_rate.npz",
    "PeMS": "staged_dpmpp_samples_PeMS.npz",
}


def select_one_path_per_window(samples: np.ndarray, dataset: str) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(np.random.SeedSequence([SEED, sum(map(ord, dataset))]))
    choice = rng.integers(samples.shape[2], size=samples.shape[0], endpoint=False)
    return samples[np.arange(samples.shape[0]), :, choice, :], choice


def variogram_cloud(y_true: np.ndarray, pred: np.ndarray, max_gap: int) -> tuple[float, list[float]]:
    error = pred.astype(np.float64) - y_true.astype(np.float64)
    per_gap = [
        float(np.mean((error[..., gap:] - error[..., :-gap]) ** 2))
        for gap in range(1, max_gap + 1)
    ]
    return float(np.mean(per_gap)), per_gap


def dtw_l1_per_step(target: np.ndarray, pred: np.ndarray) -> tuple[float, int]:
    """Classic monotonic DTW with L1 local cost, normalized by warp-path length."""
    n, m = len(target), len(pred)
    cost = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    length = np.zeros((n + 1, m + 1), dtype=np.int32)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            prev_costs = (cost[i - 1, j - 1], cost[i - 1, j], cost[i, j - 1])
            best = int(np.argmin(prev_costs))
            if best == 0:
                pi, pj = i - 1, j - 1
            elif best == 1:
                pi, pj = i - 1, j
            else:
                pi, pj = i, j - 1
            cost[i, j] = abs(float(target[i - 1]) - float(pred[j - 1])) + cost[pi, pj]
            length[i, j] = length[pi, pj] + 1
    return float(cost[n, m] / length[n, m]), int(length[n, m])


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset, filename in SPECS.items():
        with np.load(PACKS / filename, allow_pickle=False) as pack:
            y_true = np.asarray(pack["y_true"])
            samples = np.asarray(pack["samples"])
        pred, choice = select_one_path_per_window(samples, dataset)
        if y_true.shape != pred.shape:
            raise ValueError(f"{dataset}: {y_true.shape} vs {pred.shape}")
        g8, per_gap8 = variogram_cloud(y_true, pred, max_gap=8)
        g16, per_gap16 = variogram_cloud(y_true, pred, max_gap=16)
        dtw = []
        lengths = []
        for target, forecast in zip(y_true.reshape(-1, y_true.shape[-1]), pred.reshape(-1, pred.shape[-1])):
            value, length = dtw_l1_per_step(target, forecast)
            dtw.append(value)
            lengths.append(length)
        rows.append({
            "dataset": dataset,
            "source": str((PACKS / filename).relative_to(REPO)),
            "n_windows": int(y_true.shape[0]),
            "n_variates": int(y_true.shape[1]),
            "n_saved_samples": int(samples.shape[2]),
            "sample_choices": choice.tolist(),
            "variogram_cloud_g8": {"value": g8, "per_gap": per_gap8},
            "variogram_cloud_g16": {"value": g16, "per_gap": per_gap16},
            "dtw_l1_per_warp_step": {
                "mean": float(np.mean(dtw)),
                "median": float(np.median(dtw)),
                "p90": float(np.quantile(dtw, 0.9)),
                "mean_warp_path_length": float(np.mean(lengths)),
            },
        })
    payload = {
        "condition": "fixed current HPs with dataset-policy default max_scale",
        "sampling": "One seeded whole-horizon quad_t trajectory per window, shared across variates.",
        "seed": SEED,
        "variogram_metric": "mean_h mean((error[t+h]-error[t])^2), lower is better",
        "dtw_metric": "classic monotonic DTW with L1 local cost, normalized per warp-path step; lower is better",
        "rows": rows,
    }
    (OUTPUT / "probabilistic_path_stats.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# Fixed HP + default max-scale probabilistic path statistics",
        "",
        payload["sampling"],
        "",
        "| dataset | variogram g8 | variogram g16 | DTW L1 / warp step | DTW median | DTW p90 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        dtw = row["dtw_l1_per_warp_step"]
        lines.append(
            f"| {row['dataset']} | {row['variogram_cloud_g8']['value']:.6f} | "
            f"{row['variogram_cloud_g16']['value']:.6f} | {dtw['mean']:.6f} | "
            f"{dtw['median']:.6f} | {dtw['p90']:.6f} |"
        )
    (OUTPUT / "probabilistic_path_stats.md").write_text("\n".join(lines) + "\n")
    for row in rows:
        print(row["dataset"], row["variogram_cloud_g8"]["value"], row["variogram_cloud_g16"]["value"], row["dtw_l1_per_warp_step"]["mean"])


if __name__ == "__main__":
    main()
