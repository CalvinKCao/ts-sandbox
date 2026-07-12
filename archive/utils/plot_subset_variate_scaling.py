#!/usr/bin/env python3
"""Plot subset variates with calibrated low-var divisors (all datasets).

For each training-subset variate: 20k-step global z-score overview (plot stride 4)
alongside a to-scale window showing the representable ±max_scale band in z-score units.

Example:
  python utils/plot_subset_variate_scaling.py
  python utils/plot_subset_variate_scaling.py --datasets dynamic,electricity
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
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

DEFAULT_CONFIG = REPO / "configs" / "binary_anchor_ar_patch_decoder_ctx_healthy_norm_reduced_hp.yaml"
DEFAULT_OUT = REPO / "reports" / "subset_variate_scaling"
DEFAULT_DATASETS = [
    "ETTh1",
    "ETTh2",
    "ETTm1",
    "ETTm2",
    "exchange_rate",
    "weather",
    "traffic",
    "electricity",
    "illness",
    "PeMS",
    "solar_Alabama",
    "dynamic",
]
DYNAMIC_NAMES = ["aimp", "amud", "arnd", "asin1", "asin2", "adbr", "adfl"]
LOW_VAR_TH = 0.3
STD_FLOOR = 0.1


def _column_names(path: str, date_col: Optional[str]) -> Optional[List[str]]:
    if not path.endswith(".csv"):
        return None
    import pandas as pd

    df_head = pd.read_csv(path, nrows=1)
    if date_col and date_col in df_head.columns:
        return [c for c in df_head.columns if c != date_col]
    return list(df_head.columns)


def _variate_label(
    dataset: str,
    subset_idx: int,
    raw_idx: int,
    col_names: Optional[List[str]],
) -> str:
    if dataset == "dynamic" and subset_idx < len(DYNAMIC_NAMES):
        return DYNAMIC_NAMES[subset_idx]
    if col_names and raw_idx < len(col_names):
        return str(col_names[raw_idx])
    return f"var{subset_idx}"


def _normalize_train_zscore(
    data: np.ndarray, dataset: str, lookback: int,
) -> np.ndarray:
    _, border2s = _paper_split_borders(dataset, len(data), lookback)
    train = data[: border2s[0]]
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True) + 1e-8
    return ((data - mean) / std).astype(np.float64)


def _resolve_unit_stds(
    exp: dict,
    dataset: str,
    n_variates: int,
) -> List[float]:
    by_var = (exp.get("window_norm_low_var_unit_std_by_variate") or {}).get(dataset)
    if by_var is not None:
        if len(by_var) != n_variates:
            raise ValueError(
                f"{dataset}: window_norm_low_var_unit_std_by_variate length "
                f"{len(by_var)} != {n_variates} subset variates"
            )
        return [float(x) for x in by_var]
    default = float(exp.get("window_norm_low_var_unit_std", 1.0))
    by_ds = (exp.get("window_norm_low_var_unit_std_by_dataset") or {}).get(dataset)
    if by_ds is not None:
        return [float(by_ds)] * n_variates
    return [default] * n_variates


def _resolve_max_scale(exp: dict, dataset: str) -> float:
    by_ds = exp.get("max_scale_by_dataset") or {}
    if dataset in by_ds:
        return float(by_ds[dataset])
    return float(exp.get("max_scale", 3.5))


def _effective_std(past_std: float, unit_std: float, low_var_th: float) -> float:
    if past_std <= STD_FLOOR or past_std < low_var_th:
        return unit_std
    return max(past_std, STD_FLOOR)


def _effective_window(
    seg_len: int, lookback: int, horizon: int,
) -> tuple[int, int, int]:
    """Return (lookback, horizon, window_len) capped to available segment length."""
    lb = min(lookback, seg_len)
    hz = min(horizon, max(1, seg_len - lb))
    return lb, hz, lb + hz


def _pick_exemplar_start(
    series: np.ndarray,
    *,
    lookback: int,
    horizon: int,
    unit_std: float,
    max_scale: float,
    seg_len: int,
    low_var_th: float,
    prefer_low_var: bool = True,
) -> int:
    lookback, horizon, window_len = _effective_window(seg_len, lookback, horizon)
    if seg_len < window_len:
        return 0
    best_start = 0
    best_score = -1.0
    best_any_start = 0
    best_any_score = -1.0
    for start in range(0, seg_len - window_len + 1, 4):
        past = series[start : start + lookback]
        future = series[start + lookback : start + window_len]
        past_std = float(past.std())
        center = float(past.mean())
        std_eff = _effective_std(past_std, unit_std, low_var_th)
        wn_horizon = (future - center) / std_eff
        score = float(np.abs(wn_horizon).max())
        if score > best_any_score:
            best_any_score = score
            best_any_start = start
        if prefer_low_var and past_std >= low_var_th and past_std > STD_FLOOR:
            continue
        if score > best_score:
            best_score = score
            best_start = start
    return best_start if best_score >= 0 else best_any_start


def _plot_variate(
    series: np.ndarray,
    *,
    dataset: str,
    name: str,
    subset_idx: int,
    raw_idx: int,
    unit_std: float,
    max_scale: float,
    lookback: int,
    horizon: int,
    low_var_th: float,
    n_points: int,
    plot_stride: int,
    out_path: Path,
) -> dict:
    seg_len = min(n_points, len(series))
    seg = series[:seg_len]
    cfg_lookback, cfg_horizon = lookback, horizon
    lookback, horizon, window_len = _effective_window(seg_len, lookback, horizon)
    window_truncated = (lookback, horizon) != (cfg_lookback, cfg_horizon)

    exemplar_start = _pick_exemplar_start(
        seg,
        lookback=cfg_lookback,
        horizon=cfg_horizon,
        unit_std=unit_std,
        max_scale=max_scale,
        seg_len=seg_len,
        low_var_th=low_var_th,
    )
    past = seg[exemplar_start : exemplar_start + lookback]
    future = seg[exemplar_start + lookback : exemplar_start + window_len]
    window_seg = np.concatenate([past, future])
    past_std = float(past.std())
    center = float(past.mean())
    std_eff = _effective_std(past_std, unit_std, low_var_th)
    rep_half_height = max_scale * std_eff
    wn_horizon = (future - center) / std_eff
    n_clipped = int((np.abs(wn_horizon) > max_scale).sum())

    fig = plt.figure(figsize=(15, 3.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 1.0], wspace=0.08)
    ax_over = fig.add_subplot(gs[0, 0])
    ax_win = fig.add_subplot(gs[0, 1], sharey=ax_over)

    idx = np.arange(0, seg_len, plot_stride)
    ax_over.plot(idx, seg[idx], linewidth=0.7, color="#1565C0")
    ax_over.axvspan(
        exemplar_start,
        exemplar_start + window_len,
        color="#FFB74D",
        alpha=0.18,
        label="exemplar window",
    )
    ax_over.set_title(
        f"{dataset} var{subset_idx} ({name}, raw col {raw_idx}) — "
        f"unit_std={unit_std:g}, max_scale={max_scale:g}"
    )
    ax_over.set_xlabel(f"time index (first {seg_len:,} steps, stride {plot_stride})")
    ax_over.set_ylabel("global z-score")
    ax_over.grid(True, alpha=0.25)
    ax_over.legend(loc="upper right", fontsize=8)

    t_win = np.arange(window_len)
    ax_win.plot(t_win, window_seg, color="#1565C0", lw=1.1)
    ax_win.axvline(lookback - 0.5, color="0.45", ls=":", lw=1)
    ax_win.axhline(center, color="0.35", ls="--", lw=0.9, label="window mean")
    ax_win.fill_between(
        t_win,
        center - rep_half_height,
        center + rep_half_height,
        color="#66BB6A",
        alpha=0.35,
        label=f"±{max_scale:g}σ → ±{rep_half_height:.3f} z",
    )
    if n_clipped:
        clip_mask = np.abs(wn_horizon) > max_scale
        clip_t = t_win[lookback:][clip_mask]
        clip_y = future[clip_mask]
        ax_win.scatter(clip_t, clip_y, s=14, c="#E91E63", zorder=5, label="clipped horizon")
    ax_win.set_xlim(-2, window_len + 1)
    ax_win.set_title(
        f"{window_len}-step window @ t={exemplar_start}\n"
        f"past_std={past_std:.3f}, σ_eff={std_eff:g}"
        + (" (truncated to series length)" if window_truncated else ""),
        fontsize=9,
    )
    ax_win.set_xlabel("window time")
    ax_win.grid(True, alpha=0.25)
    ax_win.legend(loc="upper right", fontsize=7)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

    return {
        "subset_index": subset_idx,
        "raw_index": raw_idx,
        "name": name,
        "unit_std": unit_std,
        "max_scale": max_scale,
        "rep_half_height_z": rep_half_height,
        "exemplar_start": exemplar_start,
        "past_std": past_std,
        "std_eff": std_eff,
        "n_clipped_horizon": n_clipped,
        "plot": str(out_path),
        "window_truncated": window_truncated,
        "effective_lookback": lookback,
        "effective_horizon": horizon,
    }


def plot_dataset(
    dataset: str,
    *,
    exp: dict,
    policy: dict,
    seed: int,
    out_dir: Path,
    n_points: int,
    plot_stride: int,
) -> dict:
    lookback = int(exp.get("lookback_length", 96))
    horizon = int(exp.get("forecast_length", 96))
    window_len = lookback + horizon
    low_var_th = float(exp.get("window_norm_low_var_threshold", LOW_VAR_TH))
    max_scale = _resolve_max_scale(exp, dataset)

    subset = resolve_subset_meta_for_dataset(dataset, policy, seed)
    raw_indices = list(subset["variate_indices"])
    unit_stds = _resolve_unit_stds(exp, dataset, len(raw_indices))

    path, date_col = _resolve_registry_path(dataset)
    col_names = _column_names(path, date_col)
    raw = _load_dataset_array(path, date_col)
    norm = _normalize_train_zscore(raw, dataset, lookback)

    ds_out = out_dir / dataset
    ds_out.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, (raw_idx, unit_std) in enumerate(zip(raw_indices, unit_stds)):
        name = _variate_label(dataset, i, raw_idx, col_names)
        stem = f"{dataset}_v{i}_{name}_unit{unit_std:g}"
        row = _plot_variate(
            norm[:, raw_idx],
            dataset=dataset,
            name=name,
            subset_idx=i,
            raw_idx=raw_idx,
            unit_std=unit_std,
            max_scale=max_scale,
            lookback=lookback,
            horizon=horizon,
            low_var_th=low_var_th,
            n_points=n_points,
            plot_stride=plot_stride,
            out_path=ds_out / f"{stem}_overview_rep_window.png",
        )
        rows.append(row)

    return {
        "dataset": dataset,
        "subset_id": subset.get("subset_id"),
        "n_variates": len(rows),
        "lookback": lookback,
        "forecast": horizon,
        "window_len": window_len,
        "max_scale": max_scale,
        "low_var_threshold": low_var_th,
        "n_timesteps_used": min(n_points, len(norm)),
        "variates": rows,
    }


def _write_markdown(
    all_results: List[dict],
    out_dir: Path,
    *,
    n_points: int,
    plot_stride: int,
    summary_config: str = "",
) -> None:
    lines = [
        "# Subset variate scaling — all datasets",
        "",
        f"Config: `{summary_config}`." if summary_config else "",
        f"Global train z-score. Overview: first up to {n_points:,} steps, plot stride {plot_stride}. "
        "Right panel: exemplar low-var window (when available) with green band "
        "±max_scale·σ_eff in z-score space (to-scale with left panel).",
        "",
    ]
    for result in all_results:
        ds = result["dataset"]
        lines.extend(
            [
                f"## {ds} (`{result['subset_id']}`, {result['n_variates']} vars, "
                f"window={result['window_len']}, max_scale={result['max_scale']:g})",
                "",
                "| subset | name | unit_std | σ_eff | ±band (z) | clipped hzn |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for r in result["variates"]:
            rel = Path(r["plot"]).relative_to(out_dir).as_posix()
            lines.append(
                f"| {r['subset_index']} | {r['name']} | {r['unit_std']:g} | "
                f"{r['std_eff']:g} | ±{r['rep_half_height_z']:.3f} | {r['n_clipped_horizon']} | "
                f"[plot]({rel}) |"
            )
        lines.append("")
        for r in result["variates"]:
            rel = Path(r["plot"]).relative_to(out_dir).as_posix()
            lines.extend(
                [
                    f"### {ds} — {r['name']} (unit_std={r['unit_std']:g})",
                    "",
                    f"![{ds} {r['name']}]({rel})",
                    "",
                ]
            )
    (out_dir / "subset_variate_scaling.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--datasets",
        type=str,
        default=",".join(DEFAULT_DATASETS),
        help="comma-separated dataset names",
    )
    p.add_argument("--n-points", type=int, default=20_000)
    p.add_argument("--plot-stride", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    cfg = load_experiment_config(str(args.config.resolve()))
    exp = cfg.get("experiment", {})
    policy = _load_data_subset_policy(args.config)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for dataset in datasets:
        print(f"Plotting {dataset}...", flush=True)
        all_results.append(
            plot_dataset(
                dataset,
                exp=exp,
                policy=policy,
                seed=args.seed,
                out_dir=out_dir,
                n_points=args.n_points,
                plot_stride=args.plot_stride,
            )
        )

    summary = {
        "config": str(args.config.resolve()),
        "n_points": args.n_points,
        "plot_stride": args.plot_stride,
        "datasets": all_results,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    _write_markdown(
        all_results,
        out_dir,
        n_points=args.n_points,
        plot_stride=args.plot_stride,
        summary_config=str(args.config.resolve()),
    )

    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
