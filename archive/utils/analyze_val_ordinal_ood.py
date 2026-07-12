#!/usr/bin/env python3
"""Measure how often full-dataset val splits exceed train value/bin envelopes.

Two notions of OOD (ordinal-relevant):
  1. Global z-score: val timesteps outside [train_min, train_max] per variate.
  2. Ordinal bins: val horizon timesteps whose window-normalized coarse bin falls
     outside the per-variate min/max bin seen on any train window horizon.

Writes reports/val_ordinal_ood_analysis/ with summary.json, summary.md, and
full train+val series plots for the worst variates per dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
_ARCHIVE_UTILS = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(_ARCHIVE_UTILS) not in sys.path:
    sys.path.insert(0, str(_ARCHIVE_UTILS))

from models.diffusion_tsf.train_multivariate_pipeline import (  # noqa: E402
    DATASET_REGISTRY,
    _load_dataset_array,
    _paper_split_borders,
    _resolve_registry_path,
)
from visualize_val_ordinal_ood import (  # noqa: E402
    NormBinCfg,
    _bin_indices,
    _load_cfg,
    _max_scale_for,
    _train_envelope,
    _variate_names,
    _window_norm,
)

DEFAULT_CONFIG = REPO / "configs" / "base" / "binary_staged.yaml"
DEFAULT_OUT = REPO / "reports" / "val_ordinal_ood_analysis"
DEFAULT_DATASETS = [k for k in DATASET_REGISTRY if k != "dalia"]
ZSCORE_EPS = 1e-8


def _zscore_train_only(raw: np.ndarray, train_end: int) -> np.ndarray:
    train = raw[:train_end]
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True) + ZSCORE_EPS
    return ((raw - mean) / std).astype(np.float64)


def _global_zscore_ood(
    z: np.ndarray,
    train_slice: Tuple[int, int],
    val_slice: Tuple[int, int],
) -> Dict[str, np.ndarray]:
    """Per-variate global z-score OOD on val timesteps vs train min/max."""
    t0, t1 = train_slice
    v0, v1 = val_slice
    train = z[t0:t1]
    val = z[v0:v1]
    tmin = train.min(axis=0)
    tmax = train.max(axis=0)
    below = val < tmin
    above = val > tmax
    ood = below | above
    return {
        "train_min": tmin,
        "train_max": tmax,
        "frac_val_ood": ood.mean(axis=0),
        "any_ood": ood.any(axis=0),
        "n_val_ood": ood.sum(axis=0),
        "val_len": np.full(z.shape[1], val.shape[0], dtype=np.int64),
    }


def _ordinal_bin_ood_stats(
    z: np.ndarray,
    train_slice: Tuple[int, int],
    val_slice: Tuple[int, int],
    cfg: NormBinCfg,
    ms: float,
) -> Dict[str, Any]:
    """Per-variate ordinal bin envelope OOD on val windows (stride=1 scan)."""
    from numpy.lib.stride_tricks import sliding_window_view

    t0, t1 = train_slice
    v0, v1 = val_slice
    n_vars = z.shape[1]
    tmin, tmax = _train_envelope(z, t0, t1, cfg, ms, n_vars)

    total = cfg.lookback + cfg.horizon
    val_sl = z[v0:v1]
    if len(val_sl) < total:
        return {
            "frac_val_windows_ood": np.zeros(n_vars),
            "frac_val_horizon_ood": np.zeros(n_vars),
            "any_ood": np.zeros(n_vars, dtype=bool),
            "train_min_bin": tmin,
            "train_max_bin": tmax,
            "n_val_windows": 0,
        }

    all_w = sliding_window_view(val_sl, total, axis=0)
    n_windows = all_w.shape[0]
    win_ood = np.zeros((n_windows, n_vars), dtype=bool)
    hz_ood_count = np.zeros(n_vars, dtype=np.int64)
    hz_total = n_windows * cfg.horizon

    batch = 256
    for b0 in range(0, n_windows, batch):
        picked = all_w[b0 : b0 + batch]
        past = picked[..., : cfg.lookback].astype(np.float32)
        fut = picked[..., cfg.lookback :].astype(np.float32)
        seg = np.concatenate([past, fut], axis=-1)
        wn = _window_norm(seg, past, cfg)
        fut_wn = wn[..., cfg.lookback :]
        fb = _bin_indices(fut_wn, ms, cfg.height)
        ood = (fb < tmin.reshape(1, -1, 1)) | (fb > tmax.reshape(1, -1, 1))
        win_ood[b0 : b0 + picked.shape[0]] = ood.any(axis=-1)
        hz_ood_count += ood.sum(axis=(0, 2))

    return {
        "frac_val_windows_ood": win_ood.mean(axis=0),
        "frac_val_horizon_ood": hz_ood_count / max(hz_total, 1),
        "any_ood": win_ood.any(axis=0),
        "train_min_bin": tmin,
        "train_max_bin": tmax,
        "n_val_windows": int(n_windows),
    }


def _plot_full_train_val(
    *,
    z: np.ndarray,
    variate_idx: int,
    variate_label: str,
    dataset: str,
    train_slice: Tuple[int, int],
    val_slice: Tuple[int, int],
    train_min: float,
    train_max: float,
    ordinal_meta: Dict[str, Any],
    output_path: Path,
    plot_stride: int = 1,
) -> None:
    t0, t1 = train_slice
    v0, v1 = val_slice
    train = z[t0:t1, variate_idx]
    val = z[v0:v1, variate_idx]
    series = np.concatenate([train, val])
    n_train = len(train)
    idx = np.arange(0, len(series), plot_stride)
    ood_val = (val < train_min) | (val > train_max)
    val_idx_global = np.arange(n_train, n_train + len(val))

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]})

    ax0 = axes[0]
    ax0.axhspan(train_min, train_max, color="#C8E6C9", alpha=0.5, label="train [min,max]")
    ax0.plot(idx, series[idx], color="#1565C0", lw=0.7, label="global z-score")
    ax0.axvline(n_train - 0.5, color="#333", ls="--", lw=1.0, label="train|val")
    if ood_val.any():
        ood_pts = val_idx_global[ood_val]
        ax0.scatter(
            ood_pts,
            val[ood_val],
            s=8,
            color="#D62728",
            alpha=0.85,
            zorder=5,
            label=f"val OOD ({ood_val.mean():.1%} of val)",
        )
    ax0.set_ylabel("global z-score")
    ax0.set_title(
        f"{dataset} | {variate_label} (var {variate_idx}) — full train + val\n"
        f"train z∈[{train_min:.2f},{train_max:.2f}] | "
        f"ordinal train bins [{ordinal_meta['train_min_bin']},{ordinal_meta['train_max_bin']}] | "
        f"val horizon bin OOD {ordinal_meta['frac_val_horizon_ood']:.1%}",
        fontsize=10,
    )
    ax0.grid(True, alpha=0.25)
    ax0.legend(loc="upper right", fontsize=8)

    ax1 = axes[1]
    ax1.bar(["global z OOD\n(val timesteps)", "ordinal bin OOD\n(val horizon steps)"],
            [float(ood_val.mean()), float(ordinal_meta["frac_val_horizon_ood"])],
            color=["#EF5350", "#FF7043"], alpha=0.85)
    ax1.set_ylim(0, max(0.05, float(ood_val.mean()), float(ordinal_meta["frac_val_horizon_ood"])) * 1.25)
    ax1.set_ylabel("fraction OOD")
    ax1.grid(True, axis="y", alpha=0.25)

    ax0.set_xlabel(f"time index (train {n_train:,} + val {len(val):,} steps)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def analyze_dataset(
    dataset: str,
    *,
    config_path: Path,
    output_dir: Path,
    max_plots: int,
) -> Dict[str, Any]:
    cfg = _load_cfg(config_path)
    ms = _max_scale_for(dataset, cfg, config_path)
    path, date_col = _resolve_registry_path(dataset)
    raw = _load_dataset_array(path, date_col).astype(np.float64)
    names = _variate_names(path, date_col, raw.shape[1])

    n = len(raw)
    b1, b2 = _paper_split_borders(dataset, n, cfg.lookback)
    train_slice = (b1[0], b2[0])
    val_slice = (b1[1], b2[1])
    z = _zscore_train_only(raw, b2[0])

    gz = _global_zscore_ood(z, train_slice, val_slice)
    ob = _ordinal_bin_ood_stats(z, train_slice, val_slice, cfg, ms)

    n_vars = raw.shape[1]
    n_gz_ood_vars = int(gz["any_ood"].sum())
    n_ob_ood_vars = int(ob["any_ood"].sum())

    # Rank: ordinal OOD first, then severity
    ordinal_any = ob["any_ood"]
    severity = (
        ob["frac_val_horizon_ood"] * 2.0
        + gz["frac_val_ood"]
        + ob["frac_val_windows_ood"] * 0.5
    )
    severity = np.where(ordinal_any, severity + 10.0, severity)
    ranked = np.argsort(-severity)
    plot_dir = output_dir / dataset
    plot_dir.mkdir(parents=True, exist_ok=True)
    plots: List[Dict[str, Any]] = []

    plot_stride = max(1, (b2[0] + (b2[1] - b1[1])) // 20_000)
    for rank, vi in enumerate(ranked[:max_plots]):
        if severity[vi] <= 0:
            break
        label = names[vi] if vi < len(names) else f"var_{vi}"
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(label))[:40]
        out_path = plot_dir / f"{dataset}_v{vi}_{safe}_full_train_val.png"
        om = {
            "train_min_bin": int(ob["train_min_bin"][vi]),
            "train_max_bin": int(ob["train_max_bin"][vi]),
            "frac_val_horizon_ood": float(ob["frac_val_horizon_ood"][vi]),
        }
        _plot_full_train_val(
            z=z,
            variate_idx=int(vi),
            variate_label=str(label),
            dataset=dataset,
            train_slice=train_slice,
            val_slice=val_slice,
            train_min=float(gz["train_min"][vi]),
            train_max=float(gz["train_max"][vi]),
            ordinal_meta=om,
            output_path=out_path,
            plot_stride=plot_stride,
        )
        plots.append(
            {
                "rank": rank + 1,
                "variate_idx": int(vi),
                "variate_label": str(label),
                "global_z_frac_val_ood": float(gz["frac_val_ood"][vi]),
                "ordinal_frac_val_horizon_ood": float(ob["frac_val_horizon_ood"][vi]),
                "ordinal_frac_val_windows_ood": float(ob["frac_val_windows_ood"][vi]),
                "train_z_min": float(gz["train_min"][vi]),
                "train_z_max": float(gz["train_max"][vi]),
                "train_bin_min": int(ob["train_min_bin"][vi]),
                "train_bin_max": int(ob["train_max_bin"][vi]),
                "plot": str(out_path.relative_to(REPO)),
            }
        )

    meta = {
        "dataset": dataset,
        "n_variates": int(n_vars),
        "n_train_steps": int(b2[0] - b1[0]),
        "n_val_steps": int(b2[1] - b1[1]),
        "max_scale": ms,
        "lookback": cfg.lookback,
        "horizon": cfg.horizon,
        "n_val_windows_scanned": ob["n_val_windows"],
        "global_zscore": {
            "pct_variates_any_ood": 100.0 * n_gz_ood_vars / n_vars,
            "pct_val_timesteps_ood_pooled": 100.0 * float(
                (gz["n_val_ood"].sum()) / max(gz["val_len"].sum(), 1)
            ),
            "n_variates_any_ood": n_gz_ood_vars,
        },
        "ordinal_bins": {
            "pct_variates_any_ood": 100.0 * n_ob_ood_vars / n_vars,
            "pct_val_horizon_timesteps_ood_pooled": 100.0 * float(
                ob["frac_val_horizon_ood"].sum() * ob["n_val_windows"] * cfg.horizon
                / max(n_vars * ob["n_val_windows"] * cfg.horizon, 1)
            ),
            "mean_frac_val_horizon_ood_per_variate": 100.0 * float(ob["frac_val_horizon_ood"].mean()),
            "n_variates_any_ood": n_ob_ood_vars,
        },
        "plots": plots,
    }
    print(
        f"[{dataset}] vars={n_vars} | "
        f"global-z OOD vars {n_gz_ood_vars}/{n_vars} ({meta['global_zscore']['pct_variates_any_ood']:.1f}%) | "
        f"ordinal OOD vars {n_ob_ood_vars}/{n_vars} ({meta['ordinal_bins']['pct_variates_any_ood']:.1f}%) | "
        f"pooled val horizon OOD {meta['ordinal_bins']['pct_val_horizon_timesteps_ood_pooled']:.3f}%"
    )
    return meta


def _write_summary_md(results: List[Dict[str, Any]], output_dir: Path) -> None:
    lines = [
        "# Val vs train range OOD (full datasets, no subsetting)",
        "",
        "Global z-score uses train-only mean/std. **Ordinal bin OOD** is the production-relevant",
        "metric: window-normalized horizon coarse bins on val windows outside the per-variate",
        "min/max bin seen on any train-window horizon (lookback=96, horizon=96, stride=1 val scan).",
        "",
        "For ordinal D3PM, bin OOD means val targets occupy bins the model may rarely or never",
        "see during training on that variate — values are clipped to `max_scale` before binning,",
        "but shifts in window-local scale can still push horizons into unseen bins.",
        "",
        "## Summary table",
        "",
        "| dataset | variates | global-z OOD vars (%) | ordinal OOD vars (%) | pooled val horizon OOD (%) |",
        "|---------|----------|------------------------|----------------------|----------------------------|",
    ]
    for r in results:
        gz = r["global_zscore"]
        ob = r["ordinal_bins"]
        lines.append(
            f"| {r['dataset']} | {r['n_variates']} | "
            f"{gz['n_variates_any_ood']} ({gz['pct_variates_any_ood']:.1f}%) | "
            f"{ob['n_variates_any_ood']} ({ob['pct_variates_any_ood']:.1f}%) | "
            f"{ob['pct_val_horizon_timesteps_ood_pooled']:.3f} |"
        )

    lines.extend(["", "## Example plots (worst variates per dataset)", ""])
    for r in results:
        lines.append(f"### {r['dataset']}")
        for p in r.get("plots") or []:
            rel = p["plot"]
            lines.append(
                f"- **{p['variate_label']}** (var {p['variate_idx']}): "
                f"global z OOD {p['global_z_frac_val_ood']:.1%}, "
                f"ordinal horizon OOD {p['ordinal_frac_val_horizon_ood']:.1%} "
                f"![plot]({rel})"
            )
        lines.append("")

    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-plots", type=int, default=3, help="Full train+val plots per dataset")
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []
    for ds in datasets:
        try:
            results.append(
                analyze_dataset(
                    ds,
                    config_path=args.config,
                    output_dir=args.output_dir,
                    max_plots=args.max_plots,
                )
            )
        except Exception as exc:
            print(f"[fail] {ds}: {exc}", file=sys.stderr)

    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    _write_summary_md(results, args.output_dir)
    print(f"Wrote {summary_path.relative_to(REPO)} and summary.md")


if __name__ == "__main__":
    main()
