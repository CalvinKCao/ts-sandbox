#!/usr/bin/env python3
"""Per-dataset normalization and binning diagnostics for diffusion windows.

For each dataset (except dalia):
  1. Full-series raw summary stats (min, max, quartiles, mean, std) per variate.
  2. Sliding windows: lookback=96, horizon=96, stride=48 over the entire series.
  3. Per (window, variate): raw, dataset-z-scored, and window-normalized stats on
     lookback+horizon; bin usage on window-normalized horizon under production
     max_scale / image_height=16 coarse binning.

Outputs under reports/window_norm_binning_diagnostics/:
  summary_table.csv, summary.json, per-dataset JSON + histogram PNG.

Example:
  python utils/analyze_window_norm_binning_diagnostics.py
  python utils/analyze_window_norm_binning_diagnostics.py --datasets ETTh1,weather
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.diffusion_tsf.train_multivariate_pipeline import (  # noqa: E402
    DATASET_REGISTRY,
    LOOKBACK_LENGTH,
    FORECAST_LENGTH,
    _load_dataset_array,
    _paper_split_borders,
    _resolve_registry_path,
)

DEFAULT_DATASETS = [k for k in DATASET_REGISTRY if k != "dalia"]
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "window_norm_binning_diagnostics"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "base" / "binary_staged.yaml"
LOOKBACK = LOOKBACK_LENGTH
HORIZON = FORECAST_LENGTH
WINDOW_STRIDE = 48
IMAGE_HEIGHT = 16
ZSCORE_EPS = 1e-8


@dataclass(frozen=True)
class NormBinConfig:
    max_scale: float
    window_norm_std_floor: float
    window_norm_center: str
    image_height: int


def _load_norm_bin_config(config_path: Path, dataset: str) -> NormBinConfig:
    with config_path.open(encoding="utf-8") as f:
        exp = yaml.safe_load(f)["experiment"]
    ms_map = dict(exp.get("max_scale_by_dataset") or {})
    max_scale = float(ms_map.get(dataset, exp.get("max_scale", 3.5)))
    return NormBinConfig(
        max_scale=max_scale,
        window_norm_std_floor=float(exp.get("window_norm_std_floor", 1e-8)),
        window_norm_center=str(exp.get("window_norm_center", "mean")),
        image_height=int(exp.get("image_height", IMAGE_HEIGHT)),
    )


def _distribution_summary(x: np.ndarray) -> Dict[str, float]:
    flat = np.asarray(x, dtype=np.float64).ravel()
    if flat.size == 0:
        nan = float("nan")
        return {
            "count": 0,
            "min": nan,
            "p25": nan,
            "p50": nan,
            "p75": nan,
            "p95": nan,
            "max": nan,
            "mean": nan,
            "std": nan,
        }
    return {
        "count": int(flat.size),
        "min": float(np.min(flat)),
        "p25": float(np.quantile(flat, 0.25)),
        "p50": float(np.quantile(flat, 0.50)),
        "p75": float(np.quantile(flat, 0.75)),
        "p95": float(np.quantile(flat, 0.95)),
        "max": float(np.max(flat)),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
    }


def _summarize_across(values: np.ndarray, prefix: str) -> Dict[str, float]:
    """Summarize a 1D array of per-window×variate scalars."""
    base = _distribution_summary(values)
    out = {f"{prefix}_{k}": v for k, v in base.items() if k != "count"}
    out[f"{prefix}_count"] = base["count"]
    return out


def _window_norm_center(past_z: np.ndarray, center_mode: str) -> np.ndarray:
    """past_z: (N, V, L) -> center (N, V, 1)."""
    if center_mode == "last":
        return past_z[..., -1:]
    if center_mode == "mean":
        return past_z.mean(axis=-1, keepdims=True)
    raise ValueError(f"unknown window_norm_center {center_mode!r}")


def _window_normalize(
    segment_z: np.ndarray,
    past_z: np.ndarray,
    *,
    std_floor: float,
    center_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """segment_z, past_z: (N, V, T). Returns norm segment, center, std."""
    center = _window_norm_center(past_z, center_mode)
    std = np.maximum(past_z.std(axis=-1, keepdims=True), std_floor)
    return (segment_z - center) / std, center, std


def _coarse_bin_indices(x_norm: np.ndarray, max_scale: float, height: int) -> np.ndarray:
    """x_norm (..., T) -> integer bin indices in [0, height-1], matching TimeSeriesTo2D."""
    clipped = np.clip(x_norm, -max_scale, max_scale)
    pos = (clipped + max_scale) / (2.0 * max_scale) * height
    return np.clip(pos.astype(np.int64), 0, height - 1)


def _bin_usage_stats(
    x_norm: np.ndarray,
    *,
    max_scale: float,
    height: int,
) -> Dict[str, np.ndarray]:
    """x_norm: (N, V, T) -> per (N, V) arrays."""
    clipped = np.clip(x_norm, -max_scale, max_scale)
    clipped_mask = np.abs(x_norm) > max_scale
    bins = _coarse_bin_indices(x_norm, max_scale, height)

    # unique bin count via occupancy bitmask (height is small)
    used = np.zeros(bins.shape[:-1] + (height,), dtype=bool)
    for b in range(height):
        used[..., b] = (bins == b).any(axis=-1)

    return {
        "n_bins_used": used.sum(axis=-1).astype(np.int32),
        "min_bin": bins.min(axis=-1).astype(np.int32),
        "max_bin": bins.max(axis=-1).astype(np.int32),
        "bin_span": (bins.max(axis=-1) - bins.min(axis=-1)).astype(np.int32),
        "frac_clipped": clipped_mask.mean(axis=-1).astype(np.float32),
        "any_clipped": clipped_mask.any(axis=-1),
        "max_abs_norm": np.abs(x_norm).max(axis=-1).astype(np.float32),
    }


def _extract_windows(
    data: np.ndarray,
    *,
    lookback: int,
    horizon: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return window starts, past (N,V,L), future horizon (N,V,H) from (T,V) data."""
    total = lookback + horizon
    n_steps, n_vars = data.shape
    if n_steps < total:
        empty = np.empty((0, n_vars, 0), dtype=np.float32)
        return np.empty(0, dtype=np.int64), empty, empty

    all_windows = sliding_window_view(data, total, axis=0)  # (N_all, V, total)
    starts = np.arange(0, n_steps - total + 1, stride, dtype=np.int64)
    picked = all_windows[starts]  # (N, V, total)
    past = picked[..., :lookback].astype(np.float32)
    future = picked[..., lookback:].astype(np.float32)
    return starts, past, future


def _segment_stats_along_time(x: np.ndarray) -> Dict[str, np.ndarray]:
    """x: (N, V, T) -> per (N, V) stats."""
    return {
        "min": x.min(axis=-1),
        "max": x.max(axis=-1),
        "mean": x.mean(axis=-1),
        "std": x.std(axis=-1),
        "p25": np.quantile(x, 0.25, axis=-1),
        "p50": np.quantile(x, 0.50, axis=-1),
        "p75": np.quantile(x, 0.75, axis=-1),
    }


def _full_dataset_variate_table(raw: np.ndarray, variate_names: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for j in range(raw.shape[1]):
        name = variate_names[j] if j < len(variate_names) else f"var_{j}"
        stats = _distribution_summary(raw[:, j])
        rows.append({"variate": name, "variate_index": j, **stats})
    return rows


def _plot_histogram_grid(
    dataset: str,
    series: Mapping[str, np.ndarray],
    *,
    output_path: Path,
    bins: int = 50,
) -> None:
    keys = list(series.keys())
    n = len(keys)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.4 * nrows), squeeze=False)
    fig.suptitle(
        f"{dataset}: window×variate distributions (lookback={LOOKBACK}, horizon={HORIZON}, stride={WINDOW_STRIDE})",
        fontsize=11,
    )
    for idx, key in enumerate(keys):
        ax = axes[idx // ncols][idx % ncols]
        vals = np.asarray(series[key], dtype=np.float64).ravel()
        vals = vals[np.isfinite(vals)]
        if key == "any_clipped":
            # boolean -> {0,1} counts
            ax.hist(vals, bins=[-0.5, 0.5, 1.5], density=True, color="#C44E52", alpha=0.85, edgecolor="white")
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["no", "yes"])
        else:
            ax.hist(vals, bins=bins, density=True, color="#4C72B0", alpha=0.85, edgecolor="none")
        ax.set_title(key.replace("_", " "), fontsize=9)
        ax.set_ylabel("density")
        if vals.size:
            ax.axvline(float(np.median(vals)), color="#D62728", lw=1.0, ls="--", alpha=0.8)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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


def process_dataset(
    dataset: str,
    *,
    config_path: Path,
    lookback: int,
    horizon: int,
    stride: int,
    output_dir: Path,
) -> Dict[str, Any]:
    nb_cfg = _load_norm_bin_config(config_path, dataset)
    path, date_col = _resolve_registry_path(dataset)
    raw = _load_dataset_array(path, date_col).astype(np.float64)
    names = _variate_names(path, date_col, raw.shape[1])

    n_steps = raw.shape[0]
    border1s, border2s = _paper_split_borders(dataset, n_steps, lookback)
    train_end = border2s[0]
    train_slice = raw[:train_end]
    z_mean = train_slice.mean(axis=0, keepdims=True)
    z_std = train_slice.std(axis=0, keepdims=True) + ZSCORE_EPS
    zscored = (raw - z_mean) / z_std

    full_raw_summary = {
        "pooled": _distribution_summary(raw),
        "per_variate": _full_dataset_variate_table(raw, names),
    }

    starts, past_raw, future_raw = _extract_windows(
        raw, lookback=lookback, horizon=horizon, stride=stride,
    )
    _, past_z, future_z = _extract_windows(
        zscored, lookback=lookback, horizon=horizon, stride=stride,
    )
    if past_raw.shape[0] == 0:
        raise ValueError(f"{dataset}: no windows (T={n_steps} < {lookback + horizon})")

    segment_raw = np.concatenate([past_raw, future_raw], axis=-1)
    segment_z = np.concatenate([past_z, future_z], axis=-1)
    segment_wn, wn_center, wn_std = _window_normalize(
        segment_z,
        past_z,
        std_floor=nb_cfg.window_norm_std_floor,
        center_mode=nb_cfg.window_norm_center,
    )
    past_wn = segment_wn[..., :lookback]
    future_wn = segment_wn[..., lookback:]

    raw_past_stats = _segment_stats_along_time(past_raw)
    raw_future_stats = _segment_stats_along_time(future_raw)
    wn_past_stats = _segment_stats_along_time(past_wn)
    wn_future_stats = _segment_stats_along_time(future_wn)
    raw_stats = _segment_stats_along_time(segment_raw)
    z_stats = _segment_stats_along_time(segment_z)
    wn_stats = _segment_stats_along_time(segment_wn)
    horizon_bin = _bin_usage_stats(
        future_wn,
        max_scale=nb_cfg.max_scale,
        height=nb_cfg.image_height,
    )
    past_bin = _bin_usage_stats(
        past_wn,
        max_scale=nb_cfg.max_scale,
        height=nb_cfg.image_height,
    )

    def flat(arr: np.ndarray) -> np.ndarray:
        return arr.reshape(-1)

    hist_series = {
        "raw_segment_std": flat(raw_stats["std"]),
        "z_segment_std": flat(z_stats["std"]),
        "wn_segment_std": flat(wn_stats["std"]),
        "wn_horizon_max_abs": flat(horizon_bin["max_abs_norm"]),
        "horizon_n_bins_used": flat(horizon_bin["n_bins_used"]),
        "horizon_bin_span": flat(horizon_bin["bin_span"]),
        "horizon_frac_clipped": flat(horizon_bin["frac_clipped"]),
        "horizon_min_bin": flat(horizon_bin["min_bin"]),
        "horizon_max_bin": flat(horizon_bin["max_bin"]),
        "past_n_bins_used": flat(past_bin["n_bins_used"]),
        "any_clipped": flat(horizon_bin["any_clipped"].astype(np.float32)),
    }

    ds_out = output_dir / dataset
    ds_out.mkdir(parents=True, exist_ok=True)
    hist_path = ds_out / "window_distributional_histograms.png"
    _plot_histogram_grid(dataset, hist_series, output_path=hist_path)

    window_summary: Dict[str, float] = {}
    window_summary.update(_summarize_across(flat(raw_past_stats["std"]), "raw_lookback_std"))
    window_summary.update(_summarize_across(flat(raw_future_stats["std"]), "raw_horizon_std"))
    window_summary.update(_summarize_across(flat(wn_past_stats["std"]), "wn_lookback_std"))
    window_summary.update(_summarize_across(flat(wn_future_stats["std"]), "wn_horizon_std"))
    window_summary.update(_summarize_across(flat(raw_stats["std"]), "raw_segment_std"))
    window_summary.update(_summarize_across(flat(z_stats["std"]), "z_segment_std"))
    window_summary.update(_summarize_across(flat(wn_stats["std"]), "wn_segment_std"))
    window_summary.update(_summarize_across(flat(horizon_bin["max_abs_norm"]), "wn_horizon_max_abs"))
    window_summary.update(_summarize_across(flat(horizon_bin["n_bins_used"]), "horizon_n_bins_used"))
    window_summary.update(_summarize_across(flat(horizon_bin["bin_span"]), "horizon_bin_span"))
    window_summary.update(_summarize_across(flat(horizon_bin["frac_clipped"]), "horizon_frac_clipped"))
    window_summary.update(_summarize_across(flat(horizon_bin["min_bin"]), "horizon_min_bin"))
    window_summary.update(_summarize_across(flat(horizon_bin["max_bin"]), "horizon_max_bin"))
    window_summary.update(_summarize_across(flat(past_bin["n_bins_used"]), "past_n_bins_used"))
    window_summary["horizon_any_clipped_frac"] = float(horizon_bin["any_clipped"].mean())
    window_summary["horizon_all_16_bins_frac"] = float((horizon_bin["n_bins_used"] == nb_cfg.image_height).mean())

    result: Dict[str, Any] = {
        "dataset": dataset,
        "n_timesteps": int(n_steps),
        "n_variates": int(raw.shape[1]),
        "n_windows": int(past_raw.shape[0]),
        "lookback": lookback,
        "horizon": horizon,
        "stride": stride,
        "train_timesteps_for_zscore": int(train_end),
        "norm_bin_config": asdict(nb_cfg),
        "full_dataset_raw": full_raw_summary,
        "window_variate_summary": window_summary,
        "histogram": str(hist_path.relative_to(REPO_ROOT)),
    }

    json_path = ds_out / "diagnostics.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(
        f"[{dataset}] T={n_steps} windows={past_raw.shape[0]} vars={raw.shape[1]} "
        f"max_scale={nb_cfg.max_scale} "
        f"horizon_clip_any={window_summary['horizon_any_clipped_frac']:.1%} "
        f"bins_used median={window_summary['horizon_n_bins_used_p50']:.0f} "
        f"-> {hist_path.relative_to(REPO_ROOT)}"
    )
    return result


def _build_summary_table(results: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for r in results:
        raw_pooled = r["full_dataset_raw"]["pooled"]
        ws = r["window_variate_summary"]
        rows.append(
            {
                "dataset": r["dataset"],
                "n_timesteps": r["n_timesteps"],
                "n_variates": r["n_variates"],
                "n_windows": r["n_windows"],
                "max_scale": r["norm_bin_config"]["max_scale"],
                "window_norm_std_floor": r["norm_bin_config"]["window_norm_std_floor"],
                # full raw series (pooled over all timesteps × variates)
                "full_raw_min": raw_pooled["min"],
                "full_raw_p25": raw_pooled["p25"],
                "full_raw_p50": raw_pooled["p50"],
                "full_raw_p75": raw_pooled["p75"],
                "full_raw_max": raw_pooled["max"],
                "full_raw_mean": raw_pooled["mean"],
                "full_raw_std": raw_pooled["std"],
                # window×variate distributional summaries (median of per-window stats)
                "wn_horizon_max_abs_p50": ws.get("wn_horizon_max_abs_p50"),
                "wn_horizon_max_abs_p95": ws.get("wn_horizon_max_abs_p95"),
                "wn_lookback_std_p50": ws.get("wn_lookback_std_p50"),
                "wn_horizon_std_p50": ws.get("wn_horizon_std_p50"),
                "horizon_n_bins_used_p50": ws.get("horizon_n_bins_used_p50"),
                "horizon_n_bins_used_p95": ws.get("horizon_n_bins_used_p95"),
                "horizon_bin_span_p50": ws.get("horizon_bin_span_p50"),
                "horizon_min_bin_p50": ws.get("horizon_min_bin_p50"),
                "horizon_max_bin_p50": ws.get("horizon_max_bin_p50"),
                "horizon_frac_clipped_p50": ws.get("horizon_frac_clipped_p50"),
                "horizon_frac_clipped_p95": ws.get("horizon_frac_clipped_p95"),
                "horizon_any_clipped_frac": ws.get("horizon_any_clipped_frac"),
                "horizon_all_16_bins_frac": ws.get("horizon_all_16_bins_frac"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated dataset names (dalia excluded by default)",
    )
    parser.add_argument("--lookback", type=int, default=LOOKBACK)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--stride", type=int, default=WINDOW_STRIDE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip() and d.strip() != "dalia"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for ds in datasets:
        if ds not in DATASET_REGISTRY:
            print(f"[skip] unknown dataset {ds!r}")
            skipped.append(ds)
            continue
        try:
            results.append(
                process_dataset(
                    ds,
                    config_path=args.config,
                    lookback=args.lookback,
                    horizon=args.horizon,
                    stride=args.stride,
                    output_dir=args.output_dir,
                )
            )
        except FileNotFoundError as exc:
            print(f"[skip] {ds}: {exc}")
            skipped.append(ds)
        except Exception as exc:
            print(f"[error] {ds}: {exc}")
            skipped.append(ds)

    if not results:
        raise SystemExit("No datasets processed successfully.")

    table = _build_summary_table(results)
    csv_path = args.output_dir / "summary_table.csv"
    table.to_csv(csv_path, index=False)

    meta = {
        "lookback": args.lookback,
        "horizon": args.horizon,
        "stride": args.stride,
        "config": str(args.config.relative_to(REPO_ROOT)),
        "datasets_processed": [r["dataset"] for r in results],
        "datasets_skipped": skipped,
        "summary_table": str(csv_path.relative_to(REPO_ROOT)),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nWrote {csv_path.relative_to(REPO_ROOT)} ({len(results)} datasets)")
    if skipped:
        print(f"Skipped: {', '.join(skipped)}")


if __name__ == "__main__":
    main()
