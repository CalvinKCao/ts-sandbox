#!/usr/bin/env python3
"""Ground-truth instance-norm horizon trend on train / test windows.

Same metric as forecast_horizon_trend: norm(last) - norm(first) on the future
horizon, with per-window z-score from the lookback (binary window norm, std
floor 0.1). Use to compare model forecast distributions against the data.

Example:
  python utils/analyze_gt_horizon_trend_baseline.py --datasets exchange_rate,weather
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.diffusion_tsf.train_multivariate_pipeline import LOOKBACK_OVERLAP, load_dataset
from utils.analyze_horizon_trend_distribution import (
    BINARY_STD_FLOOR,
    DEFAULT_OUTPUT,
    FLAT_THRESH,
    TREND_BINS,
    binary_instance_norm,
    horizon_trend,
    trend_summary,
)
from utils.eval_mmpd_gaussian_anchor import (
    _load_data_subset_policy,
    resolve_subset_meta_for_dataset,
)
from utils.visualize_fair_mmpd_vs_binary_delta import EVAL_TEST_STRIDE, TRAIN_STRIDE


def _collate(batch: Sequence[tuple]) -> tuple[np.ndarray, np.ndarray]:
    past = torch.stack([b[0] for b in batch], dim=0).numpy()
    future = torch.stack([b[1] for b in batch], dim=0).numpy()
    return past, future


def gt_trends_for_loader(loader: DataLoader, *, overlap: int) -> np.ndarray:
    chunks: List[np.ndarray] = []
    for past, future in loader:
        if overlap > 0:
            future = future[..., overlap:]
        norm_future = binary_instance_norm(past, future, BINARY_STD_FLOOR)
        chunks.append(horizon_trend(norm_future))
    return np.concatenate(chunks, axis=0)


def plot_gt_histogram(
    *,
    dataset: str,
    split: str,
    trends: np.ndarray,
    summary: Mapping[str, float],
    output_path: Path,
) -> None:
    flat = trends.ravel()
    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    ax.hist(
        flat,
        bins=TREND_BINS,
        color="#43A047",
        alpha=0.85,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.axvline(0.0, color="#212121", lw=1.0, ls="--", alpha=0.7)
    ax.axvline(summary["mean"], color="#D84315", lw=1.4, label=f"mean={summary['mean']:.3f}")
    ax.axvline(summary["median"], color="#00897B", lw=1.2, ls=":", label=f"median={summary['median']:.3f}")
    ax.set_xlabel("instance-norm horizon trend  (norm[last] − norm[first])")
    ax.set_ylabel("count (per window × variate)")
    ax.set_title(
        f"{dataset} · GT {split}\n"
        f"flat(|Δ|<{FLAT_THRESH})={summary['pct_flat']:.1f}%  "
        f"up={summary['pct_up']:.1f}%  down={summary['pct_down']:.1f}%  "
        f"super high(>1.5)={summary['pct_super_high']:.1f}%"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def process_dataset(dataset: str, output_dir: Path, batch_size: int) -> Dict[str, object]:
    policy = _load_data_subset_policy(REPO_ROOT / "configs" / "binary_anchor_stationary_flat_subsets.yaml")
    subset = resolve_subset_meta_for_dataset(dataset, policy, seed=2026)
    variate_indices = [int(i) for i in subset["variate_indices"]]

    train_ds, val_ds, test_ds, _ = load_dataset(
        dataset,
        variate_indices,
        stride=TRAIN_STRIDE,
        test_stride=EVAL_TEST_STRIDE,
    )
    overlap = int(LOOKBACK_OVERLAP)
    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=_collate,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=_collate,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=_collate,
        ),
    }

    ds_out = output_dir / dataset
    ds_meta: Dict[str, object] = {"dataset": dataset, "splits": {}}
    for split, loader in loaders.items():
        trends = gt_trends_for_loader(loader, overlap=overlap)
        summary = trend_summary(trends)
        hist_path = ds_out / f"gt_{split}_horizon_trend_histogram.png"
        plot_gt_histogram(
            dataset=dataset,
            split=split,
            trends=trends,
            summary=summary,
            output_path=hist_path,
        )
        ds_meta["splits"][split] = {
            "n_windows": int(trends.shape[0]),
            "n_variates": int(trends.shape[1]),
            "per_window_variate": summary,
            "histogram": str(hist_path.relative_to(REPO_ROOT)),
        }
        print(
            f"[{dataset}/{split}] windows={trends.shape[0]} "
            f"mean={summary['mean']:+.4f} std={summary['std']:.4f} "
            f"flat={summary['pct_flat']:.1f}% -> {hist_path.relative_to(REPO_ROOT)}"
        )
    return ds_meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="exchange_rate,weather")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_meta: List[Dict[str, object]] = []
    for dataset in datasets:
        all_meta.append(process_dataset(dataset, args.output_dir, args.batch_size))

    out_path = args.output_dir / "gt_baseline_summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"datasets": all_meta}, f, indent=2)
    print(f"Wrote {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
