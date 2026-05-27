#!/usr/bin/env python3
"""Overlay MMPD + binary-anchor forecasts on shared test windows (matrix eval npz)."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.eval_mmpd_gaussian_anchor import load_tsf_pipeline  # noqa: E402


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


def load_aligned_pack(matrix_dir: Path, dataset: str) -> Tuple[dict, dict, List[int]]:
    mmpd_path = matrix_dir / "raw" / f"mmpd_{dataset}.npz"
    bin_path = matrix_dir / "raw" / f"binary_anchor_{dataset}.npz"
    if not mmpd_path.exists():
        raise FileNotFoundError(f"Missing {mmpd_path}")
    if not bin_path.exists():
        raise FileNotFoundError(f"Missing {bin_path}")

    mmpd = np.load(mmpd_path)
    binary = np.load(bin_path)
    mmpd_idx = [int(i) for i in mmpd["indices"].tolist()]
    bin_idx = [int(i) for i in binary["indices"].tolist()]
    if mmpd_idx != bin_idx:
        raise ValueError(
            f"Index mismatch for {dataset}: MMPD and binary_anchor used different test subsets."
        )
    if mmpd["y_true"].shape != binary["y_true"].shape:
        raise ValueError(f"Shape mismatch for {dataset} between MMPD and binary packs.")

    return (
        {k: mmpd[k] for k in mmpd.files},
        {k: binary[k] for k in binary.files},
        mmpd_idx,
    )


def plot_dataset(
    matrix_dir: Path,
    dataset: str,
    output_dir: Path,
    num_forecast_windows: int = 2,
    num_lookback_windows: int = 2,
    num_futures: int = 5,
    variables_to_plot: int = 3,
    lookback: int = 96,
    horizon: int = 96,
    seed: int = 2026,
) -> Path:
    mmpd, binary, eval_indices = load_aligned_pack(matrix_dir, dataset)
    y_true = mmpd["y_true"]
    mmpd_det = mmpd["deterministic"]
    mmpd_samples = mmpd["samples"]
    bin_det = binary["deterministic"]
    bin_samples = binary["samples"]

    n_pack = y_true.shape[0]
    n_futures = min(num_futures, mmpd_samples.shape[2], bin_samples.shape[2])
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
        past, _ = test_ds[test_idx]
        past_dn = denorm(past, mean, std)

        gt = y_true[pack_row]
        m_point = mmpd_det[pack_row]
        b_point = bin_det[pack_row]
        m_futs = mmpd_samples[pack_row, :, :n_futures, :]
        b_futs = bin_samples[pack_row, :, :n_futures, :]

        for col in range(n_vars_plot):
            ax = axes[row, col]
            ax.plot(
                t_past,
                past_dn[col, -context_len:].numpy(),
                color="#9E9E9E",
                alpha=0.5,
                linewidth=0.8,
                label="Context" if row == 0 and col == 0 else "",
            )
            ax.plot(
                t_future,
                gt[col],
                color="#2196F3",
                linewidth=1.8,
                label="Ground truth" if row == 0 and col == 0 else "",
            )
            ax.plot(
                t_future,
                m_point[col],
                color="#4CAF50",
                linewidth=1.5,
                linestyle="--",
                label="MMPD point" if row == 0 and col == 0 else "",
            )
            ax.plot(
                t_future,
                b_point[col],
                color="#FF9800",
                linewidth=1.5,
                linestyle="-.",
                label="Binary anchor" if row == 0 and col == 0 else "",
            )
            for s_idx in range(n_futures):
                ax.plot(
                    t_future,
                    m_futs[col, s_idx],
                    color="#E91E63",
                    linewidth=0.85,
                    alpha=0.38,
                    label=f"MMPD futures (n={n_futures})" if row == 0 and col == 0 and s_idx == 0 else "",
                )
                ax.plot(
                    t_future,
                    b_futs[col, s_idx],
                    color="#7B1FA2",
                    linewidth=0.85,
                    alpha=0.38,
                    label=f"Binary futures (n={n_futures})" if row == 0 and col == 0 and s_idx == 0 else "",
                )

            ax.axvline(x=0, color="black", linestyle=":", alpha=0.25)
            ax.grid(True, alpha=0.2)

            m_mae = float(np.mean(np.abs(m_point[col] - gt[col])))
            b_mae = float(np.mean(np.abs(b_point[col] - gt[col])))
            ax.text(
                0.97,
                0.97,
                f"MMPD {m_mae:.3f}\nBinary {b_mae:.3f}",
                transform=ax.transAxes,
                fontsize=7,
                va="top",
                ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75),
            )
            if row == 0:
                ax.set_title(f"Var {col}", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"idx {test_idx}", fontsize=9)

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
        fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, 1.03))

    fig.suptitle(
        f"{dataset} • MMPD vs binary anchor • {len(forecast_rows)} test windows + {len(extra_indices)} lookback",
        fontsize=12,
        fontweight="bold",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"mmpd_binary_{dataset}_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "viz" / "mmpd_binary_matrix",
    )
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--num-forecast-windows", type=int, default=2)
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
        plot_dataset(
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
