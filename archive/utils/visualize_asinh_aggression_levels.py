#!/usr/bin/env python3
"""Visualize asinh(z/σ) at several aggression levels (σ sweeps).

Larger σ → more tail compression (spikes map lower). Smaller σ → steeper near 0.

Example:
  python utils/visualize_asinh_aggression_levels.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

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
DEFAULT_OUT = REPO / "reports/asinh_aggression_levels"
DEFAULT_DATASETS = [
    "ETTh1", "ETTh2", "ETTm1", "ETTm2", "exchange_rate", "weather",
    "traffic", "electricity", "illness", "PeMS", "solar_Alabama", "dynamic",
]
DYNAMIC_NAMES = ["aimp", "amud", "arnd", "asin1", "asin2", "adbr", "adfl"]

# σ small → linear-ish, tails expand; σ large → strong tail squash
DEFAULT_SIGMAS = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]


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


def _asinh_raw(z: np.ndarray, sigma: float) -> np.ndarray:
    return np.arcsinh(z / sigma)


def _asinh_norm(z: np.ndarray, sigma: float, max_scale: float) -> np.ndarray:
    """Map z=±max_scale to ±max_scale in warped space (bounded)."""
    denom = np.arcsinh(max_scale / sigma)
    if denom <= 0:
        return z
    return np.arcsinh(z / sigma) / denom * max_scale


def _plot_curves(out_path: Path, sigmas: List[float], max_scale: float) -> None:
    x = np.linspace(-10, 10, 1000)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    ax = axes[0]
    for i, s in enumerate(sigmas):
        y = _asinh_raw(x, s)
        ax.plot(x, y, lw=1.5, label=f"σ={s:g}")
    ax.plot(x, x, "k--", lw=0.8, alpha=0.4, label="linear")
    ax.set_title("asinh(z/σ) — unbounded")
    ax.set_xlabel("z")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1]
    for s in sigmas:
        y = _asinh_norm(x, s, max_scale)
        ax.plot(x, y, lw=1.5, label=f"σ={s:g}")
    ax.axhline(max_scale, color="0.45", ls=":", lw=0.9)
    ax.axhline(-max_scale, color="0.45", ls=":", lw=0.9)
    ax.axvline(max_scale, color="0.45", ls=":", lw=0.9, alpha=0.5)
    ax.axvline(-max_scale, color="0.45", ls=":", lw=0.9, alpha=0.5)
    ax.set_title(f"asinh(z/σ)/asinh(ms/σ)·ms  (ms={max_scale:g})")
    ax.set_xlabel("z")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_dataset(
    z: np.ndarray,
    *,
    dataset: str,
    variate_label: str,
    max_scale: float,
    sigmas: List[float],
    n_points: int,
    plot_stride: int,
    out_path: Path,
) -> dict:
    seg_len = min(n_points, len(z))
    seg = z[:seg_len]
    idx = np.arange(0, seg_len, plot_stride)
    t = idx

    n_sigma = len(sigmas)
    fig, axes = plt.subplots(
        n_sigma + 2, 1,
        figsize=(14, 1.35 * (n_sigma + 2)),
        sharex=True,
        constrained_layout=True,
    )

    panels = [("linear (z-score)", seg)] + [
        (f"asinh(z/σ), σ={s:g}", _asinh_raw(seg, s)) for s in sigmas
    ] + [
        (f"asinh norm, σ={sigmas[-1]:g}", _asinh_norm(seg, sigmas[-1], max_scale)),
    ]

    stats = {}
    for ax, (label, y) in zip(axes, panels):
        ax.plot(t, y[idx], lw=0.65, color="#1565C0")
        p99 = float(np.percentile(y, 99))
        p999 = float(np.percentile(y, 99.9))
        stats[label] = {"max": float(y.max()), "p99": p99, "p99.9": p999}
        ax.set_ylabel(label, fontsize=7.5, rotation=0, ha="right", labelpad=52)
        ax.grid(True, alpha=0.2)
        ax.text(
            0.995, 0.9,
            f"max={y.max():.2f}  p99={p99:.2f}  p99.9={p999:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7, color="0.35",
        )

    axes[0].set_title(
        f"{dataset} — {variate_label} | {seg_len:,} steps stride {plot_stride} | "
        f"larger σ = more tail squash",
        fontsize=10,
    )
    axes[-1].set_xlabel("time index")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return {"n_timesteps": seg_len, "stats": stats}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS))
    p.add_argument("--sigmas", type=str, default=",".join(str(s) for s in DEFAULT_SIGMAS))
    p.add_argument("--n-points", type=int, default=20_000)
    p.add_argument("--plot-stride", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    sigmas = [float(s.strip()) for s in args.sigmas.split(",") if s.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    cfg = load_experiment_config(str(args.config.resolve()))
    exp = cfg.get("experiment", {})
    lookback = int(exp.get("lookback_length", 336))
    policy = _load_data_subset_policy(args.config)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    _plot_curves(out_dir / "asinh_sigma_curves.png", sigmas, max_scale=6.0)

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
        fname = f"{dataset}_{label}_asinh_sigma_sweep.png"
        row = _plot_dataset(
            norm[:, raw_idx],
            dataset=dataset,
            variate_label=label,
            max_scale=max_scale,
            sigmas=sigmas,
            n_points=args.n_points,
            plot_stride=args.plot_stride,
            out_path=out_dir / fname,
        )
        row.update({
            "dataset": dataset,
            "variate": label,
            "max_scale": max_scale,
            "sigmas": sigmas,
            "plot": str(out_dir / fname),
        })
        results.append(row)

    summary = {"sigmas": sigmas, "datasets": results}
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    md = [
        "# asinh aggression levels (σ sweep)",
        "",
        "Transform: `asinh(z/σ)`. **Larger σ** compresses tails more (spikes map lower).",
        f"σ values: {sigmas}. Last panel: bounded `asinh(z/σ)/asinh(ms/σ)·ms`.",
        "",
        "![curves](asinh_sigma_curves.png)",
        "",
    ]
    for r in results:
        md.extend([
            f"## {r['dataset']} — {r['variate']}",
            "",
            f"![{r['dataset']}]({Path(r['plot']).name})",
            "",
        ])
    (out_dir / "asinh_aggression_levels.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "n": len(results)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
