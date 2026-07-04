#!/usr/bin/env python3
"""Overview + window-range histograms for selected variates (global z-score norm)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from models.diffusion_tsf.train_multivariate_pipeline import (  # noqa: E402
    LOOKBACK_LENGTH,
    FORECAST_LENGTH,
    _load_dataset_array,
    _resolve_registry_path,
    _paper_split_borders,
)
from utils.eval_mmpd_gaussian_anchor import (  # noqa: E402
    _load_data_subset_policy,
    resolve_subset_meta_for_dataset,
)

DEFAULT_CONFIG = REPO / "configs" / "binary_anchor_stationary_flat_subsets.yaml"
WINDOW_LEN = LOOKBACK_LENGTH + FORECAST_LENGTH  # 192


def _normalize_train_zscore(data: np.ndarray, dataset: str, lookback: int) -> np.ndarray:
    _, border2s = _paper_split_borders(dataset, len(data), lookback)
    train = data[: border2s[0]]
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True) + 1e-8
    return ((data - mean) / std).astype(np.float64)


def _window_ranges(series: np.ndarray, window_len: int, window_stride: int) -> np.ndarray:
    n = len(series)
    if n < window_len:
        return np.array([], dtype=np.float64)
    starts = np.arange(0, n - window_len + 1, window_stride)
    out = np.empty(len(starts), dtype=np.float64)
    for i, s in enumerate(starts):
        w = series[s : s + window_len]
        out[i] = float(w.max() - w.min())
    return out


def _plot_overview(
    series: np.ndarray,
    *,
    title: str,
    out_path: Path,
    n_points: int,
    plot_stride: int,
) -> None:
    seg = series[:n_points]
    idx = np.arange(0, len(seg), plot_stride)
    fig, ax = plt.subplots(figsize=(14, 3.5), constrained_layout=True)
    ax.plot(idx, seg[idx], linewidth=0.7, color="#1565C0")
    ax.set_title(title)
    ax.set_xlabel(f"time index (first {n_points:,} steps, plotted every {plot_stride})")
    ax.set_ylabel("global z-score")
    ax.grid(True, alpha=0.25)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_histogram(
    ranges: np.ndarray,
    *,
    title: str,
    out_path: Path,
    bins: int = 60,
) -> dict:
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    counts, edges, _ = ax.hist(ranges, bins=bins, color="#43A047", edgecolor="white", linewidth=0.4)
    ax.axvline(float(np.mean(ranges)), color="#D84315", lw=1.4, label=f"mean={np.mean(ranges):.3f}")
    ax.axvline(float(np.median(ranges)), color="#00897B", lw=1.2, ls=":", label=f"median={np.median(ranges):.3f}")
    ax.set_xlabel(f"window range (max − min) over {WINDOW_LEN} steps, global z-score")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return {
        "n_windows": int(len(ranges)),
        "mean": float(np.mean(ranges)),
        "median": float(np.median(ranges)),
        "std": float(np.std(ranges)),
        "p05": float(np.percentile(ranges, 5)),
        "p95": float(np.percentile(ranges, 95)),
        "max": float(np.max(ranges)),
    }


def analyze_series(
    dataset: str,
    variate_idx: int,
    *,
    out_dir: Path,
    n_points: int,
    plot_stride: int,
    window_stride: int,
    seed: int,
    raw_column: bool = False,
) -> dict:
    path, date_col = _resolve_registry_path(dataset)
    raw = _load_dataset_array(path, date_col)
    policy = _load_data_subset_policy(DEFAULT_CONFIG)
    subset = resolve_subset_meta_for_dataset(dataset, policy, seed)
    col_names = None
    if path.endswith(".csv"):
        import pandas as pd

        df_head = pd.read_csv(path, nrows=1)
        if date_col and date_col in df_head.columns:
            col_names = [c for c in df_head.columns if c != date_col]
        else:
            col_names = list(df_head.columns)

    mapped_idx = variate_idx if raw_column else int(subset["variate_indices"][variate_idx])
    norm = _normalize_train_zscore(raw, dataset, LOOKBACK_LENGTH)
    series = norm[:, mapped_idx]

    label = f"{dataset} var{mapped_idx}"
    if col_names and mapped_idx < len(col_names):
        label += f" ({col_names[mapped_idx]})"

    seg_len = min(n_points, len(series))
    ranges = _window_ranges(series[:seg_len], WINDOW_LEN, window_stride)

    stem = f"{dataset}_var{mapped_idx}"
    overview_path = out_dir / f"{stem}_overview_20k_s{plot_stride}.png"
    hist_path = out_dir / f"{stem}_window_range_hist_w{WINDOW_LEN}_s{window_stride}.png"

    _plot_overview(
        series,
        title=f"{label} — normalized overview",
        out_path=overview_path,
        n_points=seg_len,
        plot_stride=plot_stride,
    )
    stats = _plot_histogram(
        ranges,
        title=f"{label} — {WINDOW_LEN}-step window range (n={len(ranges):,}, window stride {window_stride})",
        out_path=hist_path,
    )
    return {
        "dataset": dataset,
        "variate_index": mapped_idx,
        "variate_label": label,
        "n_timesteps_used": seg_len,
        "plot_stride": plot_stride,
        "window_len": WINDOW_LEN,
        "window_stride": window_stride,
        "overview_plot": str(overview_path),
        "histogram_plot": str(hist_path),
        "range_stats": stats,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=REPO / "reports" / "dynamic_exchange_window_range_analysis")
    p.add_argument("--n-points", type=int, default=20_000)
    p.add_argument("--plot-stride", type=int, default=4)
    p.add_argument("--window-stride", type=int, default=4)
    p.add_argument("--exchange-var", type=int, default=None, help="raw variate index; default random from subset")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.exchange_var is None:
        policy = _load_data_subset_policy(DEFAULT_CONFIG)
        sub = resolve_subset_meta_for_dataset("exchange_rate", policy, args.seed)
        rng = np.random.default_rng(args.seed)
        pick = int(rng.choice(sub["variate_indices"]))
        exchange_var = pick
    else:
        exchange_var = args.exchange_var

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "dynamic_var2": analyze_series(
            "dynamic",
            2,
            out_dir=out_dir,
            n_points=args.n_points,
            plot_stride=args.plot_stride,
            window_stride=args.window_stride,
            seed=args.seed,
            raw_column=True,
        ),
        "exchange_random": analyze_series(
            "exchange_rate",
            exchange_var,
            out_dir=out_dir,
            n_points=args.n_points,
            plot_stride=args.plot_stride,
            window_stride=args.window_stride,
            seed=args.seed,
            raw_column=True,
        ),
    }
    results["exchange_random"]["picked_raw_index"] = exchange_var

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    md = out_dir / "dynamic_exchange_window_range_analysis.md"
    d = results["dynamic_var2"]
    e = results["exchange_random"]
    md.write_text(
        f"# Dynamic var2 & exchange-rate variate window-range analysis\n\n"
        f"Global z-score (train-only). Overview: first {args.n_points:,} steps, plot stride {args.plot_stride}. "
        f"Histogram: max−min over {WINDOW_LEN}-step windows, window stride {args.window_stride}.\n\n"
        f"## Dynamic — {d['variate_label']}\n"
        f"- Range mean/median: {d['range_stats']['mean']:.4f} / {d['range_stats']['median']:.4f}\n"
        f"- ![dynamic overview]({Path(d['overview_plot']).name})\n"
        f"- ![dynamic hist]({Path(d['histogram_plot']).name})\n\n"
        f"## Exchange — {e['variate_label']}\n"
        f"- Range mean/median: {e['range_stats']['mean']:.4f} / {e['range_stats']['median']:.4f}\n"
        f"- ![exchange overview]({Path(e['overview_plot']).name})\n"
        f"- ![exchange hist]({Path(e['histogram_plot']).name})\n",
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2))
    print(f"Wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
