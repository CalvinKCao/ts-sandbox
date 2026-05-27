#!/usr/bin/env python3
"""MMPD matrix eval plots: several test windows × 5 stochastic futures + lookback rows."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.eval_mmpd_gaussian_anchor import (  # noqa: E402
    REPO_ROOT as _,
    load_tsf_pipeline,
)


def denorm(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    m = mean.squeeze().unsqueeze(-1)
    s = std.squeeze().unsqueeze(-1)
    return x * s + m


def load_variate_indices(matrix_dir: Path, dataset: str) -> List[int]:
    manifest_path = matrix_dir / "run_manifest.json"
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as f:
            manifest = json.load(f)
        gauss = manifest.get("anchor_runs", {}).get("gaussian", {}).get(dataset, {})
        if gauss:
            return list(gauss["metadata"]["variate_indices"])
    # Fallback: all channels for standard ETT
    return list(range(7))


def choose_extra_indices(
    n_test: int,
    n_extra: int,
    rng: random.Random,
    exclude: Sequence[int],
) -> List[int]:
    pool = [i for i in range(n_test) if i not in exclude]
    if not pool or n_extra <= 0:
        return []
    return rng.sample(pool, min(n_extra, len(pool)))


def plot_mmpd_comparison(
    matrix_dir: Path,
    dataset: str,
    output_dir: Path,
    num_forecast_windows: int = 3,
    num_lookback_windows: int = 2,
    num_futures: int = 5,
    variables_to_plot: int = 3,
    lookback: int = 96,
    horizon: int = 96,
    seed: int = 2026,
) -> Path:
    npz_path = matrix_dir / "raw" / f"mmpd_{dataset}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing {npz_path}")

    pack = np.load(npz_path)
    y_true = pack["y_true"]
    det = pack["deterministic"]
    samples = pack["samples"]
    eval_indices = [int(i) for i in pack["indices"].tolist()]
    n_pack = y_true.shape[0]
    n_samples_avail = samples.shape[2]
    n_futures = min(num_futures, n_samples_avail)
    variate_indices = load_variate_indices(matrix_dir, dataset)

    pipeline = load_tsf_pipeline()
    _, _, test_ds, norm_stats = pipeline.load_dataset(
        dataset,
        variate_indices,
        lookback=lookback,
        horizon=horizon,
        stride=1,
    )
    mean = torch.tensor(norm_stats["mean"], dtype=torch.float32)
    std = torch.tensor(norm_stats["std"], dtype=torch.float32)

    rng = random.Random(seed)
    row_positions = np.linspace(0, n_pack - 1, min(num_forecast_windows, n_pack), dtype=int)
    forecast_rows = [int(i) for i in row_positions]
    forecast_test_indices = [eval_indices[r] for r in forecast_rows]
    extra_indices = choose_extra_indices(
        len(test_ds), num_lookback_windows, rng, exclude=eval_indices
    )

    n_vars_plot = min(variables_to_plot, y_true.shape[1])
    n_rows = len(forecast_rows) + len(extra_indices)
    context_len = min(horizon * 2, lookback)
    t_past = np.arange(-context_len, 0)
    t_future = np.arange(0, horizon)

    fig, axes = plt.subplots(
        n_rows,
        n_vars_plot,
        figsize=(5.5 * n_vars_plot, 3.2 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    for row, pack_row in enumerate(forecast_rows):
        test_idx = eval_indices[pack_row]
        past, future = test_ds[test_idx]
        past_dn = denorm(past, mean, std)
        future_dn = denorm(future[:, -horizon:], mean, std)

        gt = y_true[pack_row]
        point = det[pack_row]
        future_samples = samples[pack_row, :, :n_futures, :]

        for col in range(n_vars_plot):
            ax = axes[row, col]
            ax.plot(
                t_past,
                past_dn[col, -context_len:].numpy(),
                color="#9E9E9E",
                alpha=0.5,
                linewidth=0.8,
            )
            ax.plot(t_future, gt[col], color="#2196F3", linewidth=1.6, label="Ground truth" if row == 0 and col == 0 else "")
            ax.plot(
                t_future,
                point[col],
                color="#4CAF50",
                linewidth=1.4,
                linestyle="--",
                label="MMPD point" if row == 0 and col == 0 else "",
            )
            for s_idx in range(n_futures):
                ax.plot(
                    t_future,
                    future_samples[col, s_idx],
                    color="#E91E63",
                    linewidth=0.9,
                    alpha=0.45,
                    label=f"MMPD futures (n={n_futures})" if row == 0 and col == 0 and s_idx == 0 else "",
                )
            ax.axvline(x=0, color="black", linestyle=":", alpha=0.25)
            ax.grid(True, alpha=0.2)

            mae_point = float(np.mean(np.abs(point[col] - gt[col])))
            mae_mean = float(np.mean(np.abs(future_samples[col].mean(axis=0) - gt[col])))
            ax.text(
                0.97,
                0.97,
                f"point MAE {mae_point:.3f}\nmean MAE {mae_mean:.3f}",
                transform=ax.transAxes,
                fontsize=7,
                va="top",
                ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
            )
            if row == 0:
                ax.set_title(f"Var {col}", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"Test idx {test_idx}", fontsize=9)

    for row_off, test_idx in enumerate(extra_indices, start=len(forecast_rows)):
        past, _ = test_ds[test_idx]
        past_dn = denorm(past, mean, std)
        for col in range(n_vars_plot):
            ax = axes[row_off, col]
            ax.plot(
                t_past,
                past_dn[col, -context_len:].numpy(),
                color="#546E7A",
                linewidth=1.1,
            )
            ax.axvline(x=0, color="black", linestyle=":", alpha=0.25)
            ax.grid(True, alpha=0.2)
            if col == 0:
                ax.set_ylabel(f"Lookback {test_idx}", fontsize=9)
            if row_off == len(forecast_rows):
                ax.set_title(f"Var {col} (ctx)", fontsize=9)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle(
        f"{dataset} • MMPD • {n_futures} stochastic futures • {len(forecast_rows)} forecast + {len(extra_indices)} lookback windows",
        fontsize=12,
        fontweight="bold",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"mmpd_{dataset}_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix-dir",
        type=Path,
        required=True,
        help="Matrix eval output (raw/mmpd_<dataset>.npz)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "viz" / "mmpd_matrix",
    )
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--num-forecast-windows", type=int, default=3)
    parser.add_argument("--num-lookback-windows", type=int, default=2)
    parser.add_argument("--num-futures", type=int, default=5)
    parser.add_argument("--vars", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    matrix_dir = args.matrix_dir.resolve()
    datasets = args.datasets
    if datasets is None:
        datasets = sorted(
            p.name.replace("mmpd_", "").replace(".npz", "")
            for p in (matrix_dir / "raw").glob("mmpd_*.npz")
        )

    for dataset in datasets:
        plot_mmpd_comparison(
            matrix_dir,
            dataset,
            args.output_dir.resolve(),
            num_forecast_windows=args.num_forecast_windows,
            num_lookback_windows=args.num_lookback_windows,
            num_futures=args.num_futures,
            variables_to_plot=args.vars,
            seed=args.seed + sum(ord(c) for c in dataset),
        )


if __name__ == "__main__":
    main()
