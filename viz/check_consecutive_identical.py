#!/usr/bin/env python3
"""Find variates with 4+ consecutive identical values and plot first two runs."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASETS_DIR = os.path.join(REPO_ROOT, "datasets")
DEFAULT_OUT = os.path.join(REPO_ROOT, "reports", "check_consecutive_identical")

DATASET_REGISTRY = {
    "ETTh2": ("ETT-small/ETTh2.csv", "date"),
    "ETTm1": ("ETT-small/ETTm1.csv", "date"),
    "ETTm2": ("ETT-small/ETTm2.csv", "date"),
    "exchange_rate": ("exchange_rate/exchange_rate.csv", "date"),
    "weather": ("weather/weather.csv", "date"),
    "PeMS": ("PeMS/PEMS04.npz", None),
}

PEMS_CANDIDATES = (
    "PeMS/PEMS04.npz",
    "PeMS/PEMS08.npz",
    "PeMS/PEMS03.npz",
    "PeMS/PEMS07.npz",
)

# (label, lo inclusive, hi inclusive; hi=None means open-ended)
RUN_LENGTH_BUCKETS: List[Tuple[str, int, Optional[int]]] = [
    ("4", 4, 4),
    ("5-10", 5, 10),
    ("11-20", 11, 20),
    ("21-50", 21, 50),
    ("51-100", 51, 100),
    ("101-200", 101, 200),
    ("201-500", 201, 500),
    ("501+", 501, None),
]


@dataclass
class FlatRun:
    variate: int
    variate_name: str
    start: int
    length: int
    value: float


def _load_pems_npz(path: str) -> np.ndarray:
    raw = np.load(path, allow_pickle=True)
    data = raw["data"]
    if data.ndim == 3:
        data = data[:, :, 0]
    return np.asarray(data, dtype=np.float32)


def _resolve_path(name: str) -> Tuple[str, Optional[str]]:
    if name == "PeMS":
        for rel in PEMS_CANDIDATES:
            path = os.path.join(DATASETS_DIR, rel)
            if os.path.isfile(path):
                return path, None
        raise FileNotFoundError(f"No PeMS file under {DATASETS_DIR}/PeMS/")
    rel, date_col = DATASET_REGISTRY[name]
    path = os.path.join(DATASETS_DIR, rel)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return path, date_col


def _load_array(path: str, date_col: Optional[str]) -> np.ndarray:
    if path.endswith(".npz"):
        return _load_pems_npz(path)
    df = pd.read_csv(path)
    if date_col and date_col in df.columns:
        cols = [c for c in df.columns if c != date_col]
    else:
        cols = list(df.columns)
    return df[cols].values.astype(np.float32)


def _variate_names(path: str, date_col: Optional[str], n_cols: int) -> List[str]:
    if path.endswith(".npz"):
        return [f"var_{i}" for i in range(n_cols)]
    df = pd.read_csv(path, nrows=1)
    if date_col and date_col in df.columns:
        return [c for c in df.columns if c != date_col]
    return list(df.columns)


def find_flat_runs(series: np.ndarray, min_len: int = 4) -> List[Tuple[int, int, float]]:
    runs: List[Tuple[int, int, float]] = []
    if series.size == 0:
        return runs
    start = 0
    for i in range(1, series.size):
        if series[i] != series[start]:
            run_len = i - start
            if run_len >= min_len:
                runs.append((start, run_len, float(series[start])))
            start = i
    run_len = series.size - start
    if run_len >= min_len:
        runs.append((start, run_len, float(series[start])))
    return runs


def scan_dataset(name: str, min_len: int = 4) -> Tuple[np.ndarray, List[str], List[FlatRun]]:
    path, date_col = _resolve_path(name)
    data = _load_array(path, date_col)
    names = _variate_names(path, date_col, data.shape[1])
    all_runs: List[FlatRun] = []
    for v in range(data.shape[1]):
        for start, length, value in find_flat_runs(data[:, v], min_len=min_len):
            all_runs.append(FlatRun(v, names[v], start, length, value))
    all_runs.sort(key=lambda r: (r.start, r.variate))
    return data, names, all_runs


def bucket_label(length: int) -> str:
    for label, lo, hi in RUN_LENGTH_BUCKETS:
        if hi is None:
            if length >= lo:
                return label
        elif lo <= length <= hi:
            return label
    return "other"


def count_runs_by_bucket(runs: Sequence[FlatRun]) -> Dict[str, int]:
    counts: Dict[str, int] = {label: 0 for label, _, _ in RUN_LENGTH_BUCKETS}
    for run in runs:
        counts[bucket_label(run.length)] += 1
    return counts


def print_bucket_table(name: str, counts: Dict[str, int], total: int) -> None:
    print(f"  run length histogram ({total} runs):")
    for label, _, _ in RUN_LENGTH_BUCKETS:
        n = counts[label]
        if n == 0:
            continue
        pct = 100.0 * n / total if total else 0.0
        print(f"    {label:>8}: {n:6d}  ({pct:5.1f}%)")


def save_histogram_csv(rows: List[dict], out_dir: str) -> str:
    path = os.path.join(out_dir, "run_length_histogram.csv")
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path


def print_variate_bucket_tables(
    name: str,
    names: List[str],
    runs: Sequence[FlatRun],
    max_variates: int,
) -> None:
    n_vars = min(max_variates, len(names))
    print(f"  per-variate histogram (first {n_vars} variates):")
    for v in range(n_vars):
        vruns = [r for r in runs if r.variate == v]
        if not vruns:
            print(f"    {names[v]}: no runs")
            continue
        counts = count_runs_by_bucket(vruns)
        parts = [f"{label}={counts[label]}" for label, _, _ in RUN_LENGTH_BUCKETS if counts[label]]
        print(f"    {names[v]} ({len(vruns)} runs): {', '.join(parts)}")


def save_variate_histogram_csv(rows: List[dict], out_dir: str) -> str:
    path = os.path.join(out_dir, "run_length_histogram_by_variate.csv")
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path


def plot_variate_histogram(
    dataset: str,
    names: List[str],
    runs: Sequence[FlatRun],
    out_dir: str,
    max_variates: int,
) -> None:
    n_vars = min(max_variates, len(names))
    if n_vars == 0:
        return
    bucket_labels = [label for label, _, _ in RUN_LENGTH_BUCKETS]
    matrix = np.zeros((n_vars, len(bucket_labels)), dtype=int)
    for v in range(n_vars):
        counts = count_runs_by_bucket([r for r in runs if r.variate == v])
        matrix[v] = [counts[label] for label in bucket_labels]

    if matrix.sum() == 0:
        return

    fig, ax = plt.subplots(figsize=(10, max(3, 0.45 * n_vars + 1.5)))
    im = ax.imshow(matrix, aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(bucket_labels)))
    ax.set_xticklabels(bucket_labels, rotation=45, ha="right")
    ax.set_yticks(range(n_vars))
    ax.set_yticklabels([names[v] for v in range(n_vars)])
    ax.set_xlabel("run length bucket")
    ax.set_ylabel("variate")
    ax.set_title(f"{dataset}: flat-run counts by variate (first {n_vars})")
    for i in range(n_vars):
        for j in range(len(bucket_labels)):
            val = matrix[i, j]
            if val:
                ax.text(j, i, str(val), ha="center", va="center", fontsize=7, color="black")
    fig.colorbar(im, ax=ax, label="# runs")
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{dataset}_run_length_by_variate.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_histogram(dataset: str, counts: Dict[str, int], out_dir: str) -> None:
    labels = [label for label, _, _ in RUN_LENGTH_BUCKETS if counts[label] > 0]
    if not labels:
        return
    values = [counts[label] for label in labels]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values, color="C0", edgecolor="white")
    ax.set_xlabel("run length (consecutive identical steps)")
    ax.set_ylabel("# runs")
    ax.set_title(f"{dataset}: flat-run length distribution")
    ax.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(values):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{dataset}_run_length_hist.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def _size_from_lengths(lengths: np.ndarray, min_pt: float = 8.0, max_pt: float = 280.0) -> np.ndarray:
    if lengths.size == 0:
        return lengths
    lo, hi = float(lengths.min()), float(lengths.max())
    if hi <= lo:
        return np.full(lengths.shape, (min_pt + max_pt) / 2)
    t = (np.sqrt(lengths.astype(float) - lo + 1) - 0) / (np.sqrt(hi - lo + 1) - 0)
    return min_pt + t * (max_pt - min_pt)


def plot_run_timeline_dots(
    dataset: str,
    names: List[str],
    runs: Sequence[FlatRun],
    timesteps: int,
    out_dir: str,
    max_variates: Optional[int] = None,
) -> None:
    """Stacked timeline: one row per variate, dot x=run center, size~sqrt(length)."""
    os.makedirs(out_dir, exist_ok=True)
    n_vars = len(names) if max_variates is None else min(max_variates, len(names))
    if n_vars == 0:
        return

    row_h = 0.22 if n_vars > 40 else (0.35 if n_vars > 15 else 0.55)
    fig_h = max(3.5, min(48.0, row_h * n_vars + 1.2))
    fig_w = 14 if timesteps > 20000 else 11
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    all_lengths: List[int] = []
    for v in range(n_vars):
        all_lengths.extend(r.length for r in runs if r.variate == v)
    ref_lengths = np.array(all_lengths, dtype=float) if all_lengths else np.array([4.0])

    for v in range(n_vars):
        vruns = [r for r in runs if r.variate == v]
        if not vruns:
            continue
        lengths = np.array([r.length for r in vruns], dtype=float)
        xs = np.array([r.start + 0.5 * (r.length - 1) for r in vruns])
        sizes = _size_from_lengths(lengths, min_pt=10, max_pt=260)
        ax.scatter(
            xs, np.full(len(vruns), v),
            s=sizes, c="C0", alpha=0.55, edgecolors="white", linewidths=0.3,
        )

    ax.set_xlim(0, timesteps)
    ax.set_ylim(-0.6, n_vars - 0.4)
    ax.set_yticks(range(n_vars))
    label_fs = 5 if n_vars > 80 else (6 if n_vars > 30 else 8)
    ax.set_yticklabels(names[:n_vars], fontsize=label_fs)
    ax.set_xlabel("time index")
    ax.set_ylabel("variate")
    ax.set_title(f"{dataset}: flat runs (dot size ~ run length, n={len(all_lengths)})")
    ax.grid(True, axis="x", alpha=0.25)
    for length, label in [(4, "len 4"), (24, "len 24"), (97, "len 97")]:
        if length <= ref_lengths.max():
            ax.scatter([], [], s=_size_from_lengths(np.array([float(length)]))[0],
                       c="C0", alpha=0.7, label=label)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper right", fontsize=7, framealpha=0.9)

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{dataset}_flat_run_timeline.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_random_run_samples(
    dataset: str,
    data: np.ndarray,
    names: List[str],
    runs: Sequence[FlatRun],
    out_dir: str,
    variates: Sequence[int],
    n_samples: int,
    context: int = 50,
    seed: int = 0,
) -> None:
    if not runs or n_samples <= 0:
        return
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    n_cols = len(variates)
    fig, axes = plt.subplots(n_samples, n_cols, figsize=(4.2 * n_cols, 2.4 * n_samples), squeeze=False)

    for col, v in enumerate(variates):
        vruns = [r for r in runs if r.variate == v]
        if not vruns:
            for row in range(n_samples):
                axes[row, col].axis("off")
                if row == 0:
                    axes[row, col].set_title(f"{names[v]} (no runs)")
            continue

        pick_n = min(n_samples, len(vruns))
        idx = rng.choice(len(vruns), size=pick_n, replace=False)
        chosen = [vruns[i] for i in idx]

        for row in range(n_samples):
            ax = axes[row, col]
            if row >= pick_n:
                ax.axis("off")
                continue
            run = chosen[row]
            lo = max(0, run.start - context)
            hi = min(data.shape[0], run.start + run.length + context)
            t = np.arange(lo, hi)
            y = data[lo:hi, run.variate]
            ax.plot(t, y, color="C0", lw=1.1)
            ax.axvspan(run.start, run.start + run.length - 1, color="C3", alpha=0.3)
            ax.axvline(run.start, color="C3", ls="--", lw=0.8, alpha=0.7)
            ax.axvline(run.start + run.length - 1, color="C3", ls="--", lw=0.8, alpha=0.7)
            if row == 0:
                ax.set_title(names[v])
            ax.set_ylabel("value", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.25)
            ax.text(
                0.02, 0.95,
                f"t={run.start}..{run.start + run.length - 1}\nlen={run.length}, v={run.value:g}",
                transform=ax.transAxes, va="top", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
            )

    for col in range(n_cols):
        axes[-1, col].set_xlabel("time index", fontsize=8)

    fig.suptitle(
        f"{dataset}: random flat runs (first {n_cols} variates, ±{context} steps context)",
        y=1.01,
    )
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{dataset}_random_flat_runs.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_occurrences(
    dataset: str,
    data: np.ndarray,
    runs: Sequence[FlatRun],
    out_dir: str,
    context: int = 30,
    max_plots: int = 2,
) -> None:
    if not runs:
        return
    os.makedirs(out_dir, exist_ok=True)
    to_plot = runs[:max_plots]
    n = len(to_plot)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), squeeze=False)
    for ax, run in zip(axes[:, 0], to_plot):
        lo = max(0, run.start - context)
        hi = min(data.shape[0], run.start + run.length + context)
        t = np.arange(lo, hi)
        y = data[lo:hi, run.variate]
        ax.plot(t, y, color="C0", lw=1.2)
        ax.axvspan(run.start, run.start + run.length - 1, color="C3", alpha=0.25)
        ax.set_title(
            f"{dataset} | {run.variate_name} (v{run.variate}) | "
            f"t={run.start}..{run.start + run.length - 1} ({run.length} steps @ {run.value:g})"
        )
        ax.set_xlabel("time index")
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{dataset}_flat_runs.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_REGISTRY.keys()),
        choices=list(DATASET_REGISTRY.keys()),
    )
    parser.add_argument("--min-len", type=int, default=4)
    parser.add_argument("--max-plots", type=int, default=2)
    parser.add_argument("--context", type=int, default=30)
    parser.add_argument("--max-variates", type=int, default=7)
    parser.add_argument("--random-samples", type=int, default=0,
                        help="If >0, plot this many random flat runs per variate")
    parser.add_argument("--random-variates", type=int, default=3,
                        help="Number of leading variates for random-sample plots")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeline-dots", action="store_true",
                        help="Stacked dot timeline (one PNG per dataset, all variates)")
    parser.add_argument("--timeline-only", action="store_true",
                        help="Only generate timeline dot plots (skip histograms)")
    parser.add_argument("--timeline-max-variates", type=int, default=None,
                        help="Cap variates on timeline plot (default: all)")
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    summary_lines = []
    hist_rows: List[dict] = []
    variate_hist_rows: List[dict] = []

    for name in args.datasets:
        print(f"\n=== {name} ===")
        data, names, runs = scan_dataset(name, min_len=args.min_len)
        if args.timeline_dots:
            plot_run_timeline_dots(
                name, names, runs, data.shape[0], args.out_dir,
                max_variates=args.timeline_max_variates,
            )
        if args.timeline_only:
            continue

        n_vars = min(args.max_variates, len(names))
        runs_sub = [r for r in runs if r.variate < n_vars]
        affected = sorted({r.variate for r in runs_sub})
        print(f"  shape T={data.shape[0]} V={data.shape[1]} (reporting first {n_vars})")
        print(f"  flat runs (>={args.min_len}): {len(runs_sub)} across {len(affected)} variates")

        bucket_counts = count_runs_by_bucket(runs_sub)
        print_bucket_table(name, bucket_counts, len(runs_sub))
        for label, _, _ in RUN_LENGTH_BUCKETS:
            hist_rows.append({"dataset": name, "bucket": label, "count": bucket_counts[label]})
        plot_histogram(name, bucket_counts, args.out_dir)

        print_variate_bucket_tables(name, names, runs_sub, n_vars)
        for v in range(n_vars):
            vruns = [r for r in runs_sub if r.variate == v]
            counts = count_runs_by_bucket(vruns)
            for label, _, _ in RUN_LENGTH_BUCKETS:
                variate_hist_rows.append({
                    "dataset": name,
                    "variate_idx": v,
                    "variate_name": names[v],
                    "bucket": label,
                    "count": counts[label],
                })
        plot_variate_histogram(name, names, runs_sub, args.out_dir, n_vars)

        if affected:
            for v in affected:
                vruns = [r for r in runs_sub if r.variate == v]
                print(f"    {names[v]}: {len(vruns)} runs (first at t={vruns[0].start}, len={vruns[0].length})")
            plot_occurrences(name, data, runs_sub, args.out_dir, args.context, args.max_plots)
            if args.random_samples > 0:
                n_rand_v = min(args.random_variates, len(names))
                plot_random_run_samples(
                    name, data, names, runs,
                    args.out_dir,
                    variates=list(range(n_rand_v)),
                    n_samples=args.random_samples,
                    context=args.context,
                    seed=args.seed,
                )
        else:
            print("  none found")
        summary_lines.append(f"{name}: {len(runs_sub)} runs, {len(affected)} variates (first {n_vars})")

    if args.timeline_only:
        return 0

    summary_path = os.path.join(args.out_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    hist_path = save_histogram_csv(hist_rows, args.out_dir)
    variate_hist_path = save_variate_histogram_csv(variate_hist_rows, args.out_dir)
    print(f"\nWrote {summary_path}")
    print(f"Wrote {hist_path}")
    print(f"Wrote {variate_hist_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
