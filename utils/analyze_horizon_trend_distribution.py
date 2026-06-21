#!/usr/bin/env python3
"""Instance-normalized horizon trend: norm(last step) - norm(first step) per forecast.

Uses saved eval npz + test past windows to re-apply each model's instance norm
(binary window z-score; MMPD get_statistics) before differencing. Saved npz
values are typically in raw space even when training used window norm.

Example:
  python utils/analyze_horizon_trend_distribution.py \\
    --datasets exchange_rate,weather

  # MMPD eval only (cluster, after ckpts exist):
  python utils/eval_mmpd_gaussian_anchor.py --phase mmpd --datasets exchange_rate \\
    --output-dir results/datasets/06-16-mmpd-maskae-fair-13d \\
    --skip-mmpd-train --force-mmpd-eval --mmpd-backbone MaskAE
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.diffusion_tsf.train_multivariate_pipeline import load_dataset
from utils.eval_mmpd_gaussian_anchor import (
    _load_data_subset_policy,
    resolve_subset_meta_for_dataset,
)
from utils.visualize_fair_mmpd_vs_binary_delta import (
    BINARY_CONFIG,
    DEFAULT_MMPD_RUN,
    EVAL_TEST_STRIDE,
    TRAIN_STRIDE,
    BinaryRun,
    discover_binary_run,
    load_mmpd_pack,
)


def load_binary_deterministic(binary_run: Path, dataset: str) -> np.ndarray:
    anchor_path = binary_run / "raw" / f"staged_anchor_{dataset}.npz"
    if not anchor_path.is_file():
        raise FileNotFoundError(f"Missing {anchor_path}")
    with np.load(anchor_path) as data:
        if "deterministic" in data.files:
            return data["deterministic"]
        if "anchor" in data.files:
            return data["anchor"]
    raise KeyError(f"{anchor_path}: expected deterministic or anchor")

DEFAULT_OUTPUT = REPO_ROOT / "reports" / "forecast_horizon_trend"
BINARY_STD_FLOOR = 0.1
TREND_BINS = np.linspace(-3.0, 3.0, 61)
FLAT_THRESH = 0.15


def _load_past_batch(dataset: str, window_indices: Sequence[int]) -> np.ndarray:
    policy = _load_data_subset_policy(REPO_ROOT / "configs" / "binary_anchor_stationary_flat_subsets.yaml")
    subset = resolve_subset_meta_for_dataset(dataset, policy, seed=2026)
    variate_indices = [int(i) for i in subset["variate_indices"]]
    _, _, test_ds, _ = load_dataset(
        dataset,
        variate_indices,
        stride=TRAIN_STRIDE,
        test_stride=EVAL_TEST_STRIDE,
    )
    past_list = []
    for idx in window_indices:
        past, _ = test_ds[int(idx)]
        past_list.append(past.numpy())
    return np.stack(past_list, axis=0)


def binary_instance_norm(
    past: np.ndarray,
    pred: np.ndarray,
    std_floor: float,
    *,
    center: str = "mean",
) -> np.ndarray:
    if center == "last":
        ref = past[..., -1:]
    elif center == "mean":
        ref = past.mean(axis=-1, keepdims=True)
    else:
        raise ValueError(f"window_norm center must be 'mean' or 'last', got {center!r}")
    std = past.std(axis=-1, keepdims=True)
    std = np.maximum(std, std_floor)
    return (pred - ref) / std


def mmpd_instance_norm(past: np.ndarray, pred: np.ndarray) -> np.ndarray:
    mean = past.mean(axis=-1)
    var = past.var(axis=-1, ddof=0)
    stdev = np.sqrt(var)
    eps = np.finfo(np.float64).eps
    stdev = np.where(stdev < 10.0 * eps, 1.0, stdev)
    while pred.ndim > mean.ndim:
        mean = mean[..., None]
        stdev = stdev[..., None]
    return (pred - mean) / stdev


def horizon_trend(norm_pred: np.ndarray) -> np.ndarray:
    """Per (window, variate) trend in instance-normalized space."""
    return norm_pred[..., -1] - norm_pred[..., 0]


def trend_summary(values: np.ndarray) -> Dict[str, float]:
    flat = values.ravel().astype(np.float64)
    abs_v = np.abs(flat)
    return {
        "count": int(flat.size),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "median": float(np.median(flat)),
        "p05": float(np.quantile(flat, 0.05)),
        "p25": float(np.quantile(flat, 0.25)),
        "p75": float(np.quantile(flat, 0.75)),
        "p95": float(np.quantile(flat, 0.95)),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "pct_flat": float((abs_v < FLAT_THRESH).mean() * 100.0),
        "pct_up": float((flat > FLAT_THRESH).mean() * 100.0),
        "pct_down": float((flat < -FLAT_THRESH).mean() * 100.0),
        "pct_super_high": float((flat > 1.5).mean() * 100.0),
        "pct_super_low": float((flat < -1.5).mean() * 100.0),
    }


def plot_histogram(
    *,
    dataset: str,
    model_label: str,
    trends: np.ndarray,
    summary: Mapping[str, float],
    output_path: Path,
) -> None:
    flat = trends.ravel()
    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    ax.hist(flat, bins=TREND_BINS, color="#5C6BC0" if "MMPD" in model_label else "#7E57C2", alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.axvline(0.0, color="#212121", lw=1.0, ls="--", alpha=0.7)
    ax.axvline(summary["mean"], color="#D84315", lw=1.4, label=f"mean={summary['mean']:.3f}")
    ax.axvline(summary["median"], color="#00897B", lw=1.2, ls=":", label=f"median={summary['median']:.3f}")
    ax.set_xlabel("instance-norm horizon trend  (norm[last] − norm[first])")
    ax.set_ylabel("count (per window × variate)")
    ax.set_title(
        f"{dataset} · {model_label}\n"
        f"flat(|Δ|<{FLAT_THRESH})={summary['pct_flat']:.1f}%  "
        f"up={summary['pct_up']:.1f}%  down={summary['pct_down']:.1f}%  "
        f"super high(>1.5)={summary['pct_super_high']:.1f}%"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_overlay(
    dataset: str,
    binary_trends: np.ndarray,
    mmpd_trends: np.ndarray,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    ax.hist(
        binary_trends.ravel(),
        bins=TREND_BINS,
        alpha=0.55,
        label="binary grad_accum_150_lr_lo",
        color="#7E57C2",
        edgecolor="white",
        linewidth=0.3,
    )
    ax.hist(
        mmpd_trends.ravel(),
        bins=TREND_BINS,
        alpha=0.55,
        label="MMPD MaskAE fair-13d",
        color="#26A69A",
        edgecolor="white",
        linewidth=0.3,
    )
    ax.axvline(0.0, color="#212121", lw=1.0, ls="--", alpha=0.7)
    ax.set_xlabel("instance-norm horizon trend  (norm[last] − norm[first])")
    ax.set_ylabel("count")
    ax.set_title(f"{dataset} · binary vs MMPD trend distribution")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def analyze_model_trends(
    *,
    dataset: str,
    model: str,
    pred: np.ndarray,
    past: np.ndarray,
    output_dir: Path,
    window_norm_center: str = "mean",
) -> Tuple[Dict[str, object], np.ndarray]:
    if pred.shape[0] != past.shape[0]:
        raise RuntimeError(f"{dataset}/{model}: pred batch {pred.shape[0]} != past {past.shape[0]}")
    if model == "binary":
        norm_pred = binary_instance_norm(
            past, pred, BINARY_STD_FLOOR, center=window_norm_center,
        )
        label = "binary grad_accum_150_lr_lo"
    elif model == "mmpd":
        norm_pred = mmpd_instance_norm(past, pred)
        label = "MMPD MaskAE fair-13d"
    else:
        raise ValueError(model)

    trends = horizon_trend(norm_pred)
    summary = trend_summary(trends)
    ds_out = output_dir / dataset
    hist_path = ds_out / f"{model}_horizon_trend_histogram.png"
    plot_histogram(dataset=dataset, model_label=label, trends=trends, summary=summary, output_path=hist_path)

    per_window = trends.mean(axis=1)
    window_summary = trend_summary(per_window)
    meta = {
        "dataset": dataset,
        "model": model,
        "label": label,
        "n_windows": int(pred.shape[0]),
        "n_variates": int(pred.shape[1]),
        "horizon": int(pred.shape[2]),
        "per_window_variate": summary,
        "per_window_mean_variate": window_summary,
        "window_norm_center": window_norm_center,
        "histogram": str(hist_path.relative_to(REPO_ROOT)),
    }
    with (ds_out / f"{model}_horizon_trend_summary.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(
        f"[{dataset}/{model}] windows={pred.shape[0]} variates={pred.shape[1]} "
        f"mean={summary['mean']:+.4f} std={summary['std']:.4f} "
        f"flat={summary['pct_flat']:.1f}% up={summary['pct_up']:.1f}% "
        f"down={summary['pct_down']:.1f}% super_high={summary['pct_super_high']:.1f}% "
        f"-> {hist_path.relative_to(REPO_ROOT)}"
    )
    return meta, trends


def process_dataset(
    dataset: str,
    *,
    mmpd_run: Path,
    datasets_root: Path,
    ckpt_root: Path,
    output_dir: Path,
    allow_fallback_binary: bool,
    binary_only: bool,
    binary_results_dir: Optional[Path],
    window_norm_center: str = "mean",
) -> Optional[Dict[str, object]]:
    ds_meta: Dict[str, object] = {"dataset": dataset, "models": {}}
    binary_trends: Optional[np.ndarray] = None
    mmpd_trends: Optional[np.ndarray] = None
    past_m: Optional[np.ndarray] = None
    binary_run: Optional[BinaryRun] = None

    if not binary_only:
        try:
            mmpd = load_mmpd_pack(mmpd_run, dataset)
        except FileNotFoundError as exc:
            print(f"[skip mmpd] {dataset}: {exc}")
            mmpd = None
    else:
        mmpd = None

    try:
        if binary_results_dir is not None:
            partial = binary_results_dir / "partials" / f"{dataset}_staged_anchor.json"
            anchor_npz = binary_results_dir / "raw" / f"staged_anchor_{dataset}.npz"
            if not anchor_npz.is_file():
                raise FileNotFoundError(f"Missing binary anchor npz under {binary_results_dir}")
            metrics = json.loads(partial.read_text(encoding="utf-8")) if partial.is_file() else {}
            binary_run = BinaryRun(
                results_dir=binary_results_dir,
                ckpt_dir=binary_results_dir,
                metrics=metrics,
                config_suffix=BINARY_CONFIG,
            )
        else:
            binary_run = discover_binary_run(
                datasets_root, ckpt_root, dataset, allow_fallback=allow_fallback_binary
            )
        binary_det = load_binary_deterministic(binary_run.results_dir, dataset)
        binary = {"deterministic": binary_det}
    except FileNotFoundError as exc:
        print(f"[skip binary] {dataset}: {exc}")
        binary = None

    if mmpd is not None and binary is not None:
        mmpd_indices = np.asarray(mmpd["indices"], dtype=np.int64)
        mmpd_det = mmpd["deterministic"]
        n_bin = binary["deterministic"].shape[0]
        if int(mmpd_indices.max(initial=0)) >= n_bin:
            raise RuntimeError(
                f"{dataset}: MMPD index {int(mmpd_indices.max())} out of binary range {n_bin}"
            )
        binary_det = binary["deterministic"][mmpd_indices]
    elif mmpd is not None:
        mmpd_indices = np.asarray(mmpd["indices"], dtype=np.int64)
        mmpd_det = mmpd["deterministic"]
        binary_det = None
    elif binary is not None:
        mmpd_indices = np.arange(binary["deterministic"].shape[0], dtype=np.int64)
        mmpd_det = None
        binary_det = binary["deterministic"]
    else:
        return None

    if mmpd_det is not None:
        past_m = _load_past_batch(dataset, mmpd_indices.tolist())
        meta_m, mmpd_trends = analyze_model_trends(
            dataset=dataset,
            model="mmpd",
            pred=mmpd_det,
            past=past_m,
            output_dir=output_dir,
            window_norm_center=window_norm_center,
        )
        ds_meta["models"]["mmpd"] = meta_m

    if binary_det is not None:
        bin_indices = mmpd_indices
        past_b = past_m if mmpd_det is not None else _load_past_batch(dataset, bin_indices.tolist())
        meta_b, binary_trends = analyze_model_trends(
            dataset=dataset,
            model="binary",
            pred=binary_det,
            past=past_b,
            output_dir=output_dir,
            window_norm_center=window_norm_center,
        )
        ds_meta["models"]["binary"] = meta_b
        ds_meta["binary_results_dir"] = str(binary_run.results_dir) if binary_run else None

    if binary_trends is not None and mmpd_trends is not None:
        overlay_path = output_dir / dataset / "binary_vs_mmpd_horizon_trend_overlay.png"
        plot_overlay(dataset, binary_trends, mmpd_trends, overlay_path)
        ds_meta["overlay_histogram"] = str(overlay_path.relative_to(REPO_ROOT))

    return ds_meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="exchange_rate,weather")
    parser.add_argument("--mmpd-run", type=Path, default=DEFAULT_MMPD_RUN)
    parser.add_argument("--datasets-root", type=Path, default=REPO_ROOT / "results" / "datasets")
    parser.add_argument("--ckpt-root", type=Path, default=REPO_ROOT / "results" / "ckpts")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-fallback-binary", action="store_true")
    parser.add_argument(
        "--binary-only",
        action="store_true",
        help="Skip MMPD (useful when only binary staged npz exists locally).",
    )
    parser.add_argument(
        "--binary-results-dir",
        type=Path,
        default=None,
        help="Override binary results root (must have raw/staged_anchor_<ds>.npz).",
    )
    parser.add_argument(
        "--window-norm-center",
        choices=("mean", "last"),
        default="mean",
        help="Re-normalize binary preds for trend metric (match training center).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_meta: List[Dict[str, object]] = []
    for dataset in datasets:
        meta = process_dataset(
            dataset,
            mmpd_run=args.mmpd_run,
            datasets_root=args.datasets_root,
            ckpt_root=args.ckpt_root,
            output_dir=args.output_dir,
            allow_fallback_binary=args.allow_fallback_binary,
            binary_only=args.binary_only,
            binary_results_dir=args.binary_results_dir,
            window_norm_center=args.window_norm_center,
        )
        if meta is not None:
            all_meta.append(meta)

    summary_path = args.output_dir / "horizon_trend_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({"datasets": all_meta}, f, indent=2)

    if not all_meta:
        print("No datasets processed.", file=sys.stderr)
        sys.exit(1)
    print(f"Wrote {len(all_meta)} dataset(s) -> {args.output_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
