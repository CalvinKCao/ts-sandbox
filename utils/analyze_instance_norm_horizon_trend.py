#!/usr/bin/env python3
"""Instance-normalized horizon trend: norm(last step) − norm(first step) per forecast.

Uses MMPD-style per-window instance norm (mean/std over lookback, per variate) on
**raw-scale** series — not global z-score and not raw-level deltas.

Loads saved staged-eval npz when present; optional --rerun-binary / --rerun-mmpd.

Example:
  python utils/analyze_instance_norm_horizon_trend.py
  python utils/analyze_instance_norm_horizon_trend.py --datasets exchange_rate,weather --rerun-mmpd
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

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
    align_packs,
    discover_binary_run,
    load_binary_pack,
    load_mmpd_pack,
)

DEFAULT_DATASETS = ("exchange_rate", "weather")
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "horizon_trend_instance_norm"
BINARY_CONFIG_PATH = REPO_ROOT / "configs" / f"{BINARY_CONFIG}.yaml"
LOOKBACK_OVERLAP = 8

ForecastField = Literal["deterministic", "sample_mean"]


@dataclass
class TrendStats:
    dataset: str
    model: str
    n_windows: int
    n_variates: int
    n_points: int
    mean: float
    std: float
    median: float
    p05: float
    p95: float
    buckets: Dict[str, int]
    bucket_fracs: Dict[str, float]
    values: np.ndarray


def _read_json(path: Path) -> Dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def mmpd_instance_stats(past_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """past_raw: [N, C, L] -> mean, stdev [N, C] (MMPD get_statistics)."""
    mean = past_raw.mean(axis=-1)
    stdev = past_raw.std(axis=-1, ddof=0)
    stdev = np.maximum(stdev, 1e-6)
    return mean, stdev


def instance_normalize(raw: np.ndarray, past_raw: np.ndarray) -> np.ndarray:
    """raw: [N, C, T]; past_raw: [N, C, L] -> instance-normalized raw."""
    mean, stdev = mmpd_instance_stats(past_raw)
    return (raw - mean[..., None]) / stdev[..., None]


def horizon_trend_delta(norm_seq: np.ndarray) -> np.ndarray:
    """norm_seq [N, C, H] -> per-window per-variate last − first."""
    if norm_seq.ndim != 3:
        raise ValueError(f"expected [N,C,H], got {norm_seq.shape}")
    return norm_seq[..., -1] - norm_seq[..., 0]


def global_norm_to_raw(arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """arr [N,C,T] in pipeline global z-score -> raw."""
    m = mean.reshape(1, -1, 1)
    s = std.reshape(1, -1, 1)
    return arr * s + m


def load_raw_past_for_indices(
    dataset: str,
    window_indices: Sequence[int],
) -> np.ndarray:
    policy = _load_data_subset_policy(REPO_ROOT / "configs" / "binary_anchor_stationary_flat_subsets.yaml")
    subset = resolve_subset_meta_for_dataset(dataset, policy, seed=2026)
    variate_indices = [int(i) for i in subset["variate_indices"]]
    _, _, test_ds, norm_stats = load_dataset(
        dataset,
        variate_indices,
        stride=TRAIN_STRIDE,
        test_stride=EVAL_TEST_STRIDE,
        lookback_overlap=LOOKBACK_OVERLAP,
    )
    mean = norm_stats["mean"].astype(np.float64)
    std = norm_stats["std"].astype(np.float64)
    past_rows: List[np.ndarray] = []
    for idx in window_indices:
        past_gn, _ = test_ds[int(idx)]
        past_raw = past_gn.numpy() * std.T + mean.T
        past_rows.append(past_raw)
    return np.stack(past_rows, axis=0)


def select_forecast(pack: Dict[str, np.ndarray], field: ForecastField) -> np.ndarray:
    if field == "deterministic":
        return pack["deterministic"]
    if field == "sample_mean":
        return pack["samples"].mean(axis=2)
    raise ValueError(field)


def raw_forecast_from_pack(
    pack: Dict[str, np.ndarray],
    *,
    dataset: str,
    window_indices: Sequence[int],
    source: Literal["binary", "mmpd"],
    field: ForecastField,
) -> np.ndarray:
    pred = select_forecast(pack, field)
    if source == "mmpd":
        return pred.astype(np.float64)
    policy = _load_data_subset_policy(REPO_ROOT / "configs" / "binary_anchor_stationary_flat_subsets.yaml")
    subset = resolve_subset_meta_for_dataset(dataset, policy, seed=2026)
    variate_indices = [int(i) for i in subset["variate_indices"]]
    _, _, _, norm_stats = load_dataset(
        dataset,
        variate_indices,
        stride=TRAIN_STRIDE,
        test_stride=EVAL_TEST_STRIDE,
        lookback_overlap=LOOKBACK_OVERLAP,
    )
    return global_norm_to_raw(pred, norm_stats["mean"], norm_stats["std"])


def classify_trends(
    deltas: np.ndarray,
    *,
    flat_eps: float,
    strong_eps: float,
) -> Dict[str, int]:
    flat = int(np.sum(np.abs(deltas) < flat_eps))
    mild_up = int(np.sum((deltas >= flat_eps) & (deltas < strong_eps)))
    mild_down = int(np.sum((deltas <= -flat_eps) & (deltas > -strong_eps)))
    strong_up = int(np.sum(deltas >= strong_eps))
    strong_down = int(np.sum(deltas <= -strong_eps))
    return {
        "flat": flat,
        "mild_up": mild_up,
        "mild_down": mild_down,
        "strong_up": strong_up,
        "strong_down": strong_down,
    }


def summarize_trends(
    deltas: np.ndarray,
    *,
    dataset: str,
    model: str,
    n_windows: int,
    n_variates: int,
    flat_eps: float,
    strong_eps: float,
) -> TrendStats:
    flat = deltas.reshape(-1)
    buckets = classify_trends(flat, flat_eps=flat_eps, strong_eps=strong_eps)
    n = flat.size
    fracs = {k: (v / n if n else 0.0) for k, v in buckets.items()}
    return TrendStats(
        dataset=dataset,
        model=model,
        n_windows=n_windows,
        n_variates=n_variates,
        n_points=n,
        mean=float(flat.mean()) if n else 0.0,
        std=float(flat.std(ddof=0)) if n else 0.0,
        median=float(np.median(flat)) if n else 0.0,
        p05=float(np.percentile(flat, 5)) if n else 0.0,
        p95=float(np.percentile(flat, 95)) if n else 0.0,
        buckets=buckets,
        bucket_fracs=fracs,
        values=flat,
    )


def compute_model_trends(
    *,
    dataset: str,
    model: Literal["binary", "mmpd"],
    pack: Dict[str, np.ndarray],
    window_indices: Sequence[int],
    field: ForecastField,
    flat_eps: float,
    strong_eps: float,
) -> TrendStats:
    past_raw = load_raw_past_for_indices(dataset, window_indices)
    pred_raw = raw_forecast_from_pack(
        pack,
        dataset=dataset,
        window_indices=window_indices,
        source=model,
        field=field,
    )
    if pred_raw.shape != past_raw.shape[:2] + (pred_raw.shape[-1],):
        raise RuntimeError(
            f"{dataset}/{model}: pred {pred_raw.shape} vs past batch {past_raw.shape}"
        )
    norm_pred = instance_normalize(pred_raw, past_raw)
    deltas = horizon_trend_delta(norm_pred)
    n_vars = deltas.shape[1] if deltas.ndim == 2 else 1
    return summarize_trends(
        deltas,
        dataset=dataset,
        model=model,
        n_windows=len(window_indices),
        n_variates=n_vars,
        flat_eps=flat_eps,
        strong_eps=strong_eps,
    )


def plot_histogram(
    stats: TrendStats,
    *,
    output_path: Path,
    flat_eps: float,
    strong_eps: float,
    bins: int,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    vals = stats.values
    ax.hist(vals, bins=bins, color="#4C72B0", alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.axvline(0.0, color="black", linewidth=1.0, linestyle="-", alpha=0.6)
    ax.axvline(flat_eps, color="#888", linewidth=0.9, linestyle="--", alpha=0.7)
    ax.axvline(-flat_eps, color="#888", linewidth=0.9, linestyle="--", alpha=0.7)
    ax.axvline(strong_eps, color="#C44E52", linewidth=0.9, linestyle=":", alpha=0.8)
    ax.axvline(-strong_eps, color="#C44E52", linewidth=0.9, linestyle=":", alpha=0.8)
    ax.set_title(f"{stats.dataset} — {stats.model}\ninstance-norm Δ(last−first)")
    ax.set_xlabel("Δ horizon (σ units from lookback stats)")
    ax.set_ylabel("count (windows × variates)")
    txt = (
        f"n={stats.n_points}  μ={stats.mean:+.3f}  med={stats.median:+.3f}\n"
        f"flat {stats.bucket_fracs['flat']:.0%} | "
        f"↑mild {stats.bucket_fracs['mild_up']:.0%} "
        f"↓mild {stats.bucket_fracs['mild_down']:.0%} | "
        f"↑strong {stats.bucket_fracs['strong_up']:.0%} "
        f"↓strong {stats.bucket_fracs['strong_down']:.0%}"
    )
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", fontsize=8.5)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_dataset_overlay(
    binary_stats: TrendStats,
    mmpd_stats: TrendStats,
    *,
    output_path: Path,
    bins: int,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(
        binary_stats.values,
        bins=bins,
        alpha=0.55,
        label="binary",
        color="#E45756",
        density=True,
        edgecolor="white",
        linewidth=0.3,
    )
    ax.hist(
        mmpd_stats.values,
        bins=bins,
        alpha=0.55,
        label="mmpd",
        color="#4C78A8",
        density=True,
        edgecolor="white",
        linewidth=0.3,
    )
    ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.5)
    ax.set_title(f"{binary_stats.dataset} — binary vs MMPD trend distribution")
    ax.set_xlabel("instance-norm Δ(last−first)")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def write_markdown_report(
    path: Path,
    all_stats: Sequence[TrendStats],
    *,
    flat_eps: float,
    strong_eps: float,
) -> None:
    lines = [
        "# Instance-normalized horizon trend",
        "",
        "Per forecast window×variate: `(norm[last] − norm[first])` where norm uses "
        "lookback mean/std on **raw** data (MMPD `get_statistics`).",
        "",
        f"Buckets: |Δ| < {flat_eps} flat; mild in [{flat_eps}, {strong_eps}); "
        f"strong ≥ {strong_eps} (or ≤ −{strong_eps}).",
        "",
        "| dataset | model | windows | variates | mean Δ | median Δ | std | flat | mild↑ | mild↓ | strong↑ | strong↓ |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in all_stats:
        b = s.bucket_fracs
        lines.append(
            f"| {s.dataset} | {s.model} | {s.n_windows} | {s.n_variates} | "
            f"{s.mean:+.4f} | {s.median:+.4f} | {s.std:.4f} | "
            f"{b['flat']:.1%} | {b['mild_up']:.1%} | {b['mild_down']:.1%} | "
            f"{b['strong_up']:.1%} | {b['strong_down']:.1%} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rerun_binary_eval(dataset: str, *, datasets_root: Path, ckpt_root: Path) -> None:
    run = discover_binary_run(datasets_root, ckpt_root, dataset, allow_fallback=False)
    results_dir = run.results_dir
    cmd = [
        sys.executable,
        str(REPO_ROOT / "models" / "diffusion_tsf" / "train_multivariate_pipeline.py"),
        "--config",
        str(BINARY_CONFIG_PATH),
        "--dataset",
        dataset,
        "--resume",
        "--results-dir",
        str(results_dir),
        "--checkpoint-dir",
        str(run.ckpt_dir),
    ]
    print(f"[rerun-binary] {dataset}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def rerun_mmpd_eval(dataset: str, *, mmpd_run: Path) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "utils" / "eval_mmpd_gaussian_anchor.py"),
        "--datasets",
        dataset,
        "--output-dir",
        str(mmpd_run),
        "--skip-mmpd-train",
        "--phase",
        "mmpd",
        "--force-mmpd-eval",
        "--anchor-config",
        BINARY_CONFIG,
        "--mmpd-backbone",
        "MaskAE",
        "--subset-config",
        str(REPO_ROOT / "configs" / "binary_anchor_stationary_flat_subsets.yaml"),
        "--eval-test-stride",
        str(EVAL_TEST_STRIDE),
    ]
    print(f"[rerun-mmpd] {dataset}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def _npz_ready(path: Path) -> bool:
    return path.is_file()


def process_dataset(
    dataset: str,
    args: argparse.Namespace,
) -> Dict[str, TrendStats]:
    if args.rerun_binary:
        rerun_binary_eval(dataset, datasets_root=args.datasets_root, ckpt_root=args.ckpt_root)
    if args.rerun_mmpd:
        rerun_mmpd_eval(dataset, mmpd_run=args.mmpd_run)

    binary_run = discover_binary_run(
        args.datasets_root,
        args.ckpt_root,
        dataset,
        allow_fallback=args.allow_fallback_binary,
    )
    mmpd_npz = args.mmpd_run / "raw" / f"mmpd_{dataset}.npz"
    bin_anchor = binary_run.results_dir / "raw" / f"staged_anchor_{dataset}.npz"
    bin_samples = binary_run.results_dir / "raw" / f"staged_dpmpp_samples_{dataset}.npz"

    if not (_npz_ready(bin_anchor) and _npz_ready(bin_samples)):
        raise FileNotFoundError(
            f"{dataset}: missing binary raw npz under {binary_run.results_dir}/raw "
            f"(use --rerun-binary after training ckpts exist)"
        )
    if not _npz_ready(mmpd_npz):
        raise FileNotFoundError(
            f"{dataset}: missing {mmpd_npz} (use --rerun-mmpd or pull from cluster)"
        )

    mmpd_pack = load_mmpd_pack(args.mmpd_run, dataset)
    binary_pack = load_binary_pack(binary_run, dataset)
    aligned = align_packs(mmpd_pack, binary_pack, dataset)
    mmpd_indices = [int(i) for i in aligned.indices]
    out: Dict[str, TrendStats] = {}
    ds_out = args.output_dir / dataset
    ds_out.mkdir(parents=True, exist_ok=True)

    n_bin = binary_pack["y_true"].shape[0]
    bin_all = list(range(n_bin))
    for model, pack, indices in (
        ("binary", binary_pack, bin_all),
        ("mmpd", mmpd_pack, mmpd_indices),
    ):
        stats = compute_model_trends(
            dataset=dataset,
            model=model,
            pack=pack,
            window_indices=indices,
            field=args.forecast_field,
            flat_eps=args.flat_eps,
            strong_eps=args.strong_eps,
        )
        out[model] = stats
        plot_histogram(
            stats,
            output_path=ds_out / f"histogram_{model}.png",
            flat_eps=args.flat_eps,
            strong_eps=args.strong_eps,
            bins=args.bins,
        )
        print(
            f"[{dataset}/{model}] windows={stats.n_windows} "
            f"meanΔ={stats.mean:+.4f} med={stats.median:+.4f} "
            f"flat={stats.bucket_fracs['flat']:.1%} "
            f"strong±={stats.bucket_fracs['strong_up'] + stats.bucket_fracs['strong_down']:.1%}"
        )

    bin_aligned = compute_model_trends(
        dataset=dataset,
        model="binary",
        pack=binary_pack,
        window_indices=mmpd_indices,
        field=args.forecast_field,
        flat_eps=args.flat_eps,
        strong_eps=args.strong_eps,
    )
    plot_dataset_overlay(
        bin_aligned,
        out["mmpd"],
        output_path=ds_out / "histogram_binary_vs_mmpd_overlay.png",
        bins=args.bins,
    )

    meta = {
        "dataset": dataset,
        "binary_results_dir": str(binary_run.results_dir),
        "mmpd_run": str(args.mmpd_run),
        "forecast_field": args.forecast_field,
        "flat_eps": args.flat_eps,
        "strong_eps": args.strong_eps,
        "binary": {
            "n_windows": out["binary"].n_windows,
            "mean": out["binary"].mean,
            "median": out["binary"].median,
            "buckets": out["binary"].buckets,
        },
        "mmpd": {
            "n_windows": out["mmpd"].n_windows,
            "mean": out["mmpd"].mean,
            "median": out["mmpd"].median,
            "buckets": out["mmpd"].buckets,
            "aligned_indices": mmpd_indices,
        },
    }
    with (ds_out / "trend_stats.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated datasets (default: exchange_rate,weather).",
    )
    parser.add_argument("--mmpd-run", type=Path, default=DEFAULT_MMPD_RUN)
    parser.add_argument("--datasets-root", type=Path, default=REPO_ROOT / "results" / "datasets")
    parser.add_argument("--ckpt-root", type=Path, default=REPO_ROOT / "results" / "ckpts")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--forecast-field",
        choices=["deterministic", "sample_mean"],
        default="deterministic",
    )
    parser.add_argument("--flat-eps", type=float, default=0.1, help="|Δ| below = flat.")
    parser.add_argument("--strong-eps", type=float, default=0.5, help="|Δ| at/above = strong trend.")
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--rerun-binary", action="store_true")
    parser.add_argument("--rerun-mmpd", action="store_true")
    parser.add_argument("--allow-fallback-binary", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_stats: List[TrendStats] = []
    for dataset in datasets:
        per_ds = process_dataset(dataset, args)
        all_stats.extend(per_ds.values())

    write_markdown_report(
        args.output_dir / "horizon_trend_report.md",
        all_stats,
        flat_eps=args.flat_eps,
        strong_eps=args.strong_eps,
    )
    serial = {
        f"{s.dataset}/{s.model}": {
            "mean": s.mean,
            "median": s.median,
            "bucket_fracs": s.bucket_fracs,
            "n_windows": s.n_windows,
        }
        for s in all_stats
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(serial, f, indent=2)
    print(f"Wrote report to {args.output_dir}")


if __name__ == "__main__":
    main()
