#!/usr/bin/env python3
"""Compare tail-compressing transforms on 20k-step global z-score segments.

One figure per dataset (subset variate 0): stacked time-series panels for
linear, hard clip, asinh, tanh, signed-log, soft-saturate, and segment rank.

Example:
  python utils/visualize_series_transform_comparison.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from models.diffusion_tsf.pipeline.config import load_experiment_config  # noqa: E402
from models.diffusion_tsf.train_multivariate_pipeline import (  # noqa: E402
    _load_dataset_array,
    _paper_split_borders,
    _resolve_registry_path,
)
from utils.eval_mmpd_gaussian_anchor import (  # noqa: E402
    _load_data_subset_policy,
    resolve_subset_meta_for_dataset,
)

DEFAULT_CONFIG = REPO / "configs/binary_anchor_ar_patch_decoder_ctx_lb336_hz720_fixed.yaml"
DEFAULT_OUT = REPO / "reports/series_transform_comparison"
DEFAULT_DATASETS = [
    "ETTh1", "ETTh2", "ETTm1", "ETTm2", "exchange_rate", "weather",
    "traffic", "electricity", "illness", "PeMS", "solar_Alabama", "dynamic",
]
DYNAMIC_NAMES = ["aimp", "amud", "arnd", "asin1", "asin2", "adbr", "adfl"]


def _column_names(path: str, date_col: Optional[str]) -> Optional[List[str]]:
    if not path.endswith(".csv"):
        return None
    import pandas as pd

    df_head = pd.read_csv(path, nrows=1)
    if date_col and date_col in df_head.columns:
        return [c for c in df_head.columns if c != date_col]
    return list(df_head.columns)


def _variate_label(dataset: str, raw_idx: int, col_names: Optional[List[str]]) -> str:
    if dataset == "dynamic" and raw_idx < len(DYNAMIC_NAMES):
        return DYNAMIC_NAMES[raw_idx]
    if col_names and raw_idx < len(col_names):
        return str(col_names[raw_idx])
    return f"col{raw_idx}"


def _normalize_train_zscore(data: np.ndarray, dataset: str, lookback: int) -> np.ndarray:
    _, border2s = _paper_split_borders(dataset, len(data), lookback)
    train = data[: border2s[0]]
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True) + 1e-8
    return ((data - mean) / std).astype(np.float64)


def _resolve_max_scale(exp: dict, dataset: str) -> float:
    by_ds = exp.get("max_scale_by_dataset") or {}
    if dataset in by_ds:
        return float(by_ds[dataset])
    return float(exp.get("max_scale", 3.5))


def _segment_rank(z: np.ndarray) -> np.ndarray:
    """Empirical CDF rank within segment → [0, 1]."""
    order = np.argsort(z, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(z), dtype=np.float64)
    return ranks / max(len(z) - 1, 1)


def _build_transforms(max_scale: float, sigma: float = 1.0) -> List[Tuple[str, Callable[[np.ndarray], np.ndarray]]]:
    ms = max_scale
    s = sigma
    return [
        ("linear (z-score)", lambda x: x),
        (f"hard clip ±{ms:g}", lambda x: np.clip(x, -ms, ms)),
        (f"asinh(z/{s:g})", lambda x: np.arcsinh(x / s)),
        (f"tanh(z/{s:g})", lambda x: np.tanh(x / s)),
        (f"sign·log1p(|z|/{s:g})", lambda x: np.sign(x) * np.log1p(np.abs(x) / s)),
        (f"soft saturate ms·tanh(z/{ms:g})", lambda x: ms * np.tanh(x / ms)),
        ("segment rank ∈ [0,1]", _segment_rank),
    ]


def _plot_transfer_curves(out_path: Path, max_scale: float, sigma: float = 1.0) -> None:
    x = np.linspace(-8, 8, 800)
    transforms = _build_transforms(max_scale, sigma)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(transforms)))
    for (label, fn), color in zip(transforms, colors):
        if "rank" in label:
            continue
        y = fn(x)
        ax.plot(x, y, lw=1.4, label=label, color=color)
    ax.axvline(-max_scale, color="0.5", ls=":", lw=0.8)
    ax.axvline(max_scale, color="0.5", ls=":", lw=0.8)
    ax.axhline(-max_scale, color="0.5", ls=":", lw=0.8, alpha=0.5)
    ax.axhline(max_scale, color="0.5", ls=":", lw=0.8, alpha=0.5)
    ax.set_xlabel("input (global z-score)")
    ax.set_ylabel("transformed value")
    ax.set_title(f"Transfer functions (max_scale={max_scale:g}, σ={sigma:g})")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_dataset(
    z: np.ndarray,
    *,
    dataset: str,
    variate_label: str,
    max_scale: float,
    n_points: int,
    plot_stride: int,
    out_path: Path,
    sigma: float = 1.0,
) -> dict:
    seg_len = min(n_points, len(z))
    seg = z[:seg_len]
    idx = np.arange(0, seg_len, plot_stride)
    t = idx

    transforms = _build_transforms(max_scale, sigma)
    n_panels = len(transforms)
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(14, 1.55 * n_panels), sharex=True, constrained_layout=True,
    )
    if n_panels == 1:
        axes = [axes]

    stats = {}
    for ax, (label, fn) in zip(axes, transforms):
        y = fn(seg)
        y_plot = y[idx]
        ax.plot(t, y_plot, lw=0.65, color="#1565C0")
        p01, p50, p99 = np.percentile(y, [1, 50, 99])
        stats[label] = {
            "min": float(y.min()),
            "max": float(y.max()),
            "p01": float(p01),
            "p50": float(p50),
            "p99": float(p99),
        }
        ax.set_ylabel(label, fontsize=8, rotation=0, ha="right", labelpad=42)
        ax.grid(True, alpha=0.2)
        ax.text(
            0.995, 0.92,
            f"min={y.min():.2f}  p99={p99:.2f}  max={y.max():.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7, color="0.35",
        )

    axes[0].set_title(
        f"{dataset} — {variate_label} | first {seg_len:,} steps (stride {plot_stride}) | "
        f"max_scale={max_scale:g}",
        fontsize=10,
    )
    axes[-1].set_xlabel("time index")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return {"n_timesteps": seg_len, "transform_stats": stats}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS))
    p.add_argument("--n-points", type=int, default=20_000)
    p.add_argument("--plot-stride", type=int, default=4)
    p.add_argument("--sigma", type=float, default=1.0, help="scale for asinh/tanh/log")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    cfg = load_experiment_config(str(args.config.resolve()))
    exp = cfg.get("experiment", {})
    lookback = int(exp.get("lookback_length", 336))
    policy = _load_data_subset_policy(args.config)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    _plot_transfer_curves(out_dir / "transfer_functions.png", max_scale=5.0, sigma=args.sigma)

    results = []
    for dataset in datasets:
        print(f"Plotting {dataset}...", flush=True)
        max_scale = _resolve_max_scale(exp, dataset)
        subset = resolve_subset_meta_for_dataset(dataset, policy, args.seed)
        raw_idx = int(subset["variate_indices"][0])
        path, date_col = _resolve_registry_path(dataset)
        col_names = _column_names(path, date_col)
        raw = _load_dataset_array(path, date_col)
        norm = _normalize_train_zscore(raw, dataset, lookback)
        label = _variate_label(dataset, raw_idx, col_names)
        stem = f"{dataset}_{label}_transforms_20k_s{args.plot_stride}"
        row = _plot_dataset(
            norm[:, raw_idx],
            dataset=dataset,
            variate_label=label,
            max_scale=max_scale,
            n_points=args.n_points,
            plot_stride=args.plot_stride,
            out_path=out_dir / f"{stem}.png",
            sigma=args.sigma,
        )
        row.update({
            "dataset": dataset,
            "subset_id": subset.get("subset_id"),
            "variate": label,
            "raw_index": raw_idx,
            "max_scale": max_scale,
            "plot": str(out_dir / f"{stem}.png"),
        })
        results.append(row)

    summary = {
        "config": str(args.config.resolve()),
        "n_points": args.n_points,
        "plot_stride": args.plot_stride,
        "sigma": args.sigma,
        "datasets": results,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "# Series transform comparison (20k segments)",
        "",
        f"Config: `{args.config}`. Subset variate 0 per dataset. "
        f"Input: global train z-score. Plot stride {args.plot_stride}. "
        f"Nonlinear transforms use σ={args.sigma}.",
        "",
        "![transfer functions](transfer_functions.png)",
        "",
    ]
    for r in results:
        fname = Path(r["plot"]).name
        lines.extend([
            f"## {r['dataset']} — {r['variate']} (max_scale={r['max_scale']:g})",
            "",
            f"![{r['dataset']}]({fname})",
            "",
        ])
    (out_dir / "series_transform_comparison.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({"n_datasets": len(results), "out_dir": str(out_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
