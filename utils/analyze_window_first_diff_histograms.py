#!/usr/bin/env python3
"""Histogram first-order diffs within instance-normalized 96-step windows.

For each dataset: slide length-96 windows with stride 48, z-score each window
per variate (mean 0, std 1), pool np.diff values across windows, plot histograms
for up to 10 randomly chosen variates.

Example:
  python utils/analyze_window_first_diff_histograms.py
  python utils/analyze_window_first_diff_histograms.py --datasets ETTh1,weather
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.diffusion_tsf.train_multivariate_pipeline import (
    DATASET_REGISTRY,
    _load_dataset_array,
    _resolve_registry_path,
)

DEFAULT_DATASETS = [k for k in DATASET_REGISTRY if k != "dalia"]

WINDOW_LEN = 96
WINDOW_STRIDE = 48
MAX_VARIATES = 10
STD_FLOOR = 1e-8
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "window_first_diff_histograms"
DEFAULT_SEED = 2026


def _instance_norm_window(window: np.ndarray, std_floor: float = STD_FLOOR) -> np.ndarray:
    """window: (T, C) -> same shape, per-column mean 0 std 1."""
    mean = window.mean(axis=0, keepdims=True)
    std = np.maximum(window.std(axis=0, keepdims=True), std_floor)
    return (window - mean) / std


def _pool_window_diffs(windows: np.ndarray) -> np.ndarray:
    """windows: (N, T, C). Returns pooled diffs (total_diffs, C)."""
    if windows.size == 0:
        n_vars = int(windows.shape[2]) if windows.ndim == 3 else 0
        return np.empty((0, n_vars), dtype=np.float32)
    parts = [np.diff(_instance_norm_window(w), axis=0) for w in windows]
    return np.concatenate(parts, axis=0)


def _sliding_windows(data: np.ndarray, window_len: int, window_stride: int) -> np.ndarray:
    """data: (T, C) -> (N, window_len, C)."""
    n_steps = data.shape[0]
    if n_steps < window_len:
        return np.empty((0, window_len, data.shape[1]), dtype=np.float32)
    starts = np.arange(0, n_steps - window_len + 1, window_stride)
    return np.stack([data[s : s + window_len] for s in starts], axis=0)


def _variate_names(path: str, date_col: Optional[str], n_cols: int) -> List[str]:
    if path.endswith(".npz"):
        return [f"var_{i}" for i in range(n_cols)]
    try:
        df = pd.read_csv(path, nrows=1)
        if date_col and date_col in df.columns:
            return [c for c in df.columns if c != date_col]
        return list(df.columns)
    except Exception:
        return [f"var_{i}" for i in range(n_cols)]


def _pick_variates(n_cols: int, seed: int, max_vars: int) -> List[int]:
    if n_cols <= max_vars:
        return list(range(n_cols))
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(n_cols, size=max_vars, replace=False).tolist())


def process_dataset(
    dataset: str,
    *,
    window_len: int,
    window_stride: int,
    max_variates: int,
    seed: int,
    output_dir: Path,
) -> Dict[str, object]:
    path, date_col = _resolve_registry_path(dataset)
    data = _load_dataset_array(path, date_col)
    names = _variate_names(path, date_col, data.shape[1])
    windows = _sliding_windows(data, window_len, window_stride)
    diffs = _pool_window_diffs(windows)

    n_vars = diffs.shape[1]
    picked = _pick_variates(n_vars, seed + hash(dataset) % 10_000, max_variates)
    picked_names = [names[i] if i < len(names) else f"var_{i}" for i in picked]

    stats: Dict[str, object] = {
        "dataset": dataset,
        "n_windows_used": int(windows.shape[0]),
        "window_len": window_len,
        "window_stride": window_stride,
        "n_variates_total": n_vars,
        "variate_indices": picked,
        "variate_names": picked_names,
        "per_variate": {},
    }

    n_plot = len(picked)
    ncols = min(3, n_plot)
    nrows = int(np.ceil(n_plot / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False)
    fig.suptitle(
        f"{dataset}: first-order diffs in instance-norm windows "
        f"(len={window_len}, stride={window_stride})",
        fontsize=11,
    )

    for plot_idx, (var_idx, var_name) in enumerate(zip(picked, picked_names)):
        vals = diffs[:, var_idx]
        ax = axes[plot_idx // ncols][plot_idx % ncols]
        ax.hist(vals, bins=100, density=True, color="#4C72B0", alpha=0.85, edgecolor="none")
        ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_title(str(var_name), fontsize=9)
        ax.set_xlabel("diff")
        ax.set_ylabel("density")
        stats["per_variate"][str(var_name)] = {
            "index": var_idx,
            "n_diffs": int(vals.size),
            "mean": float(vals.mean()) if vals.size else float("nan"),
            "std": float(vals.std()) if vals.size else float("nan"),
            "median": float(np.median(vals)) if vals.size else float("nan"),
            "p05": float(np.percentile(vals, 5)) if vals.size else float("nan"),
            "p95": float(np.percentile(vals, 95)) if vals.size else float("nan"),
        }

    for j in range(n_plot, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.tight_layout()
    out_png = output_dir / f"{dataset}.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    stats["plot"] = str(out_png)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated dataset names",
    )
    parser.add_argument("--window-len", type=int, default=WINDOW_LEN)
    parser.add_argument("--window-stride", type=int, default=WINDOW_STRIDE)
    parser.add_argument("--max-variates", type=int, default=MAX_VARIATES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_stats: List[Dict[str, object]] = []
    for dataset in datasets:
        if dataset not in DATASET_REGISTRY:
            raise ValueError(f"Unknown dataset {dataset!r}; known: {sorted(DATASET_REGISTRY)}")
        print(f"[{dataset}] loading windows...", flush=True)
        meta = process_dataset(
            dataset,
            window_len=args.window_len,
            window_stride=args.window_stride,
            max_variates=args.max_variates,
            seed=args.seed,
            output_dir=args.output_dir,
        )
        all_stats.append(meta)
        print(f"[{dataset}] {meta['n_windows_used']} windows -> {meta['plot']}", flush=True)

    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
