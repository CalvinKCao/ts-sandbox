#!/usr/bin/env python3
"""Plot val OOD windows alongside representative train windows for the same variate."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from numpy.lib.stride_tricks import sliding_window_view

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from models.diffusion_tsf.train_multivariate_pipeline import (  # noqa: E402
    _load_dataset_array,
    _paper_split_borders,
    _resolve_registry_path,
)

DEFAULT_CONFIG = REPO / "configs" / "base" / "binary_staged.yaml"
DEFAULT_OUT = REPO / "reports" / "val_ordinal_ood_examples"
DEFAULT_DATASETS = ["traffic", "electricity", "PeMS", "ETTh1", "weather"]
ZSCORE_EPS = 1e-8


@dataclass(frozen=True)
class NormBinCfg:
    max_scale: float
    std_floor: float
    center: str
    height: int
    lookback: int
    horizon: int


def _load_cfg(config_path: Path) -> NormBinCfg:
    with config_path.open(encoding="utf-8") as f:
        exp = yaml.safe_load(f)["experiment"]
    return NormBinCfg(
        max_scale=float(exp.get("max_scale", 3.5)),
        std_floor=float(exp.get("window_norm_std_floor", 1e-8)),
        center=str(exp.get("window_norm_center", "mean")),
        height=int(exp.get("image_height", 16)),
        lookback=int(exp.get("lookback_length", 96)),
        horizon=int(exp.get("forecast_length", 96)),
    )


def _max_scale_for(dataset: str, cfg: NormBinCfg, config_path: Path) -> float:
    with config_path.open(encoding="utf-8") as f:
        exp = yaml.safe_load(f)["experiment"]
    ms_map = dict(exp.get("max_scale_by_dataset") or {})
    return float(ms_map.get(dataset, cfg.max_scale))


def _variate_names(path: str, date_col: Optional[str], n_cols: int) -> List[str]:
    if path.endswith(".csv"):
        try:
            df = pd.read_csv(path, nrows=1)
            if date_col and date_col in df.columns:
                return [c for c in df.columns if c != date_col]
            return list(df.columns)
        except Exception:
            pass
    return [f"var_{i}" for i in range(n_cols)]


def _window_norm(segment_z: np.ndarray, past_z: np.ndarray, cfg: NormBinCfg) -> np.ndarray:
    if cfg.center == "last":
        center = past_z[..., -1:]
    else:
        center = past_z.mean(axis=-1, keepdims=True)
    std = np.maximum(past_z.std(axis=-1, keepdims=True), cfg.std_floor)
    return (segment_z - center) / std


def _bin_indices(x_norm: np.ndarray, max_scale: float, height: int) -> np.ndarray:
    clipped = np.clip(x_norm, -max_scale, max_scale)
    pos = (clipped + max_scale) / (2.0 * max_scale) * height
    return np.clip(pos.astype(np.int64), 0, height - 1)


def _bin_to_value_edges(bin_idx: int, max_scale: float, height: int) -> Tuple[float, float]:
    lo = (2.0 * bin_idx / height - 1.0) * max_scale
    hi = (2.0 * (bin_idx + 1) / height - 1.0) * max_scale
    return lo, hi


def _train_envelope(
    z: np.ndarray,
    i0: int,
    i1: int,
    cfg: NormBinCfg,
    ms: float,
    n_vars: int,
) -> Tuple[np.ndarray, np.ndarray]:
    total = cfg.lookback + cfg.horizon
    sl = z[i0:i1]
    if len(sl) < total:
        raise ValueError("train split too short")
    all_w = sliding_window_view(sl, total, axis=0)
    tmin = np.full(n_vars, cfg.height - 1, dtype=np.int64)
    tmax = np.zeros(n_vars, dtype=np.int64)
    batch = 512
    for b0 in range(0, all_w.shape[0], batch):
        picked = all_w[b0 : b0 + batch]
        past = picked[..., : cfg.lookback].astype(np.float32)
        fut = picked[..., cfg.lookback :].astype(np.float32)
        seg = np.concatenate([past, fut], axis=-1)
        wn = _window_norm(seg, past, cfg)[..., cfg.lookback :]
        fb = _bin_indices(wn, ms, cfg.height)
        flat = fb.transpose(0, 2, 1).reshape(-1, n_vars)
        tmin = np.minimum(tmin, flat.min(axis=0))
        tmax = np.maximum(tmax, flat.max(axis=0))
    return tmin, tmax


def _extract_window_payload(
    all_w: np.ndarray,
    window_idx: int,
    variate_idx: int,
    cfg: NormBinCfg,
    ms: float,
) -> Dict:
    picked = all_w[window_idx]
    past = picked[..., : cfg.lookback].astype(np.float32)
    fut = picked[..., cfg.lookback :].astype(np.float32)
    seg = np.concatenate([past, fut], axis=-1)
    wn = _window_norm(seg, past, cfg)
    fut_wn = wn[..., cfg.lookback :]
    fb = _bin_indices(fut_wn, ms, cfg.height)
    return {
        "window_idx": int(window_idx),
        "variate_idx": int(variate_idx),
        "past_wn": past[variate_idx].copy(),
        "fut_wn": fut_wn[variate_idx].copy(),
        "fut_bins": fb[variate_idx].copy(),
        "ood_mask": np.zeros(cfg.horizon, dtype=bool),
    }


def _find_representative_train_window(
    z: np.ndarray,
    i0: int,
    i1: int,
    cfg: NormBinCfg,
    ms: float,
    variate_idx: int,
    train_min_bin: int,
    train_max_bin: int,
) -> Dict:
    """Train window for same variate that best spans the per-variate train bin envelope."""
    total = cfg.lookback + cfg.horizon
    sl = z[i0:i1]
    all_w = sliding_window_view(sl, total, axis=0)
    best: Optional[Dict] = None
    best_score = -1
    batch = 512
    for b0 in range(0, all_w.shape[0], batch):
        picked = all_w[b0 : b0 + batch]
        past = picked[..., : cfg.lookback].astype(np.float32)
        fut = picked[..., cfg.lookback :].astype(np.float32)
        seg = np.concatenate([past, fut], axis=-1)
        wn = _window_norm(seg, past, cfg)
        fut_wn = wn[..., cfg.lookback :]
        fb = _bin_indices(fut_wn, ms, cfg.height)
        vi_fb = fb[:, variate_idx, :]
        for wi in range(picked.shape[0]):
            bins = vi_fb[wi]
            bmin, bmax = int(bins.min()), int(bins.max())
            span = bmax - bmin
            hits_lo = bmin <= train_min_bin
            hits_hi = bmax >= train_max_bin
            score = span * 10 + int(hits_lo) * 100 + int(hits_hi) * 100
            if bmin == train_min_bin:
                score += 50
            if bmax == train_max_bin:
                score += 50
            if score > best_score:
                best_score = score
                best = _extract_window_payload(all_w, b0 + wi, variate_idx, cfg, ms)
    if best is None:
        raise ValueError("no train windows")
    best["train_min_bin"] = train_min_bin
    best["train_max_bin"] = train_max_bin
    return best


def _find_ood_examples(
    z: np.ndarray,
    i0: int,
    i1: int,
    cfg: NormBinCfg,
    ms: float,
    tmin: np.ndarray,
    tmax: np.ndarray,
    max_per_variate: int,
    max_variates: int,
) -> List[Dict]:
    total = cfg.lookback + cfg.horizon
    sl = z[i0:i1]
    all_w = sliding_window_view(sl, total, axis=0)
    n_vars = z.shape[1]
    per_variate: Dict[int, List[Dict]] = defaultdict(list)

    batch = 256
    for b0 in range(0, all_w.shape[0], batch):
        picked = all_w[b0 : b0 + batch]
        past = picked[..., : cfg.lookback].astype(np.float32)
        fut = picked[..., cfg.lookback :].astype(np.float32)
        seg = np.concatenate([past, fut], axis=-1)
        wn = _window_norm(seg, past, cfg)
        fut_wn = wn[..., cfg.lookback :]
        fb = _bin_indices(fut_wn, ms, cfg.height)
        ood = (fb < tmin.reshape(1, -1, 1)) | (fb > tmax.reshape(1, -1, 1))
        for wi in range(picked.shape[0]):
            win_idx = b0 + wi
            for vi in range(n_vars):
                if not ood[wi, vi].any():
                    continue
                if len(per_variate[vi]) >= max_per_variate:
                    continue
                n_ood = int(ood[wi, vi].sum())
                per_variate[vi].append(
                    {
                        "window_idx": int(win_idx),
                        "variate_idx": int(vi),
                        "n_ood_timesteps": n_ood,
                        "past_wn": past[wi, vi].copy(),
                        "fut_wn": fut_wn[wi, vi].copy(),
                        "fut_bins": fb[wi, vi].copy(),
                        "ood_mask": ood[wi, vi].copy(),
                        "train_min_bin": int(tmin[vi]),
                        "train_max_bin": int(tmax[vi]),
                    }
                )

    # Rank variates by number of example windows, take top max_variates
    ranked = sorted(
        per_variate.items(),
        key=lambda kv: (-len(kv[1]), -sum(e["n_ood_timesteps"] for e in kv[1])),
    )[:max_variates]
    out: List[Dict] = []
    for vi, examples in ranked:
        examples.sort(key=lambda e: -e["n_ood_timesteps"])
        out.extend(examples[:max_per_variate])
    return out


def _draw_window_column(
    axes_val_bin: tuple,
    *,
    panel: Dict,
    cfg: NormBinCfg,
    ms: float,
    title: str,
    is_val: bool,
) -> None:
    ax0, ax1 = axes_val_bin
    lb, hz = cfg.lookback, cfg.horizon
    series = np.concatenate([panel["past_wn"], panel["fut_wn"]])
    t_axis = np.arange(len(series))
    tmin_b, tmax_b = panel["train_min_bin"], panel["train_max_bin"]
    env_lo, _ = _bin_to_value_edges(tmin_b, ms, cfg.height)
    _, env_hi = _bin_to_value_edges(tmax_b, ms, cfg.height)
    hz_idx = np.arange(lb, lb + hz)

    ax0.axvspan(lb - 0.5, len(series) - 0.5, color="#FFF3CD", alpha=0.35)
    ax0.axhspan(env_lo, env_hi, color="#C8E6C9", alpha=0.45)
    ax0.axhline(ms, color="#888", ls=":", lw=0.8)
    ax0.axhline(-ms, color="#888", ls=":", lw=0.8)
    color = "#D62728" if is_val else "#1f77b4"
    ax0.plot(t_axis, series, color=color, lw=1.2)
    ax0.axvline(lb - 0.5, color="#666", ls="--", lw=0.9)

    if is_val and panel["ood_mask"].any():
        ood_pts = hz_idx[panel["ood_mask"]]
        ax0.scatter(
            ood_pts,
            panel["fut_wn"][panel["ood_mask"]],
            color="#D62728",
            s=24,
            zorder=5,
            edgecolors="white",
            linewidths=0.4,
        )

    ax0.set_title(title, fontsize=9)
    ax0.set_ylabel("window-norm")
    ax0.grid(True, alpha=0.25)

    bar_color = "#D62728" if is_val else "#4C72B0"
    ax1.bar(hz_idx, panel["fut_bins"], width=0.9, color=bar_color, alpha=0.85)
    ax1.axhspan(tmin_b - 0.45, tmax_b + 0.45, color="#C8E6C9", alpha=0.45)
    if is_val and panel["ood_mask"].any():
        ax1.scatter(
            ood_pts,
            panel["fut_bins"][panel["ood_mask"]],
            color="#B71C1C",
            s=22,
            zorder=5,
        )
    ax1.set_ylim(-0.5, cfg.height - 0.5)
    ax1.set_xlabel("timestep (lookback | horizon)")
    ax1.set_ylabel("coarse bin")
    ax1.grid(True, axis="y", alpha=0.25)


def _plot_train_val_pair(
    *,
    dataset: str,
    variate_label: str,
    train_panel: Dict,
    val_panel: Dict,
    cfg: NormBinCfg,
    ms: float,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(
        2, 2, figsize=(13.5, 6.4), sharex="col", sharey="row",
        gridspec_kw={"width_ratios": [1, 1], "height_ratios": [2.2, 1.0], "wspace": 0.12},
    )
    tmin_b, tmax_b = val_panel["train_min_bin"], val_panel["train_max_bin"]
    fig.suptitle(
        f"{dataset} | {variate_label} | train envelope bins [{tmin_b},{tmax_b}]",
        fontsize=11,
        y=1.02,
    )

    _draw_window_column(
        (axes[0, 0], axes[1, 0]),
        panel=train_panel,
        cfg=cfg,
        ms=ms,
        title=(
            f"train window {train_panel['window_idx']} | "
            f"horizon bins {train_panel['fut_bins'].min()}–{train_panel['fut_bins'].max()}"
        ),
        is_val=False,
    )
    _draw_window_column(
        (axes[0, 1], axes[1, 1]),
        panel=val_panel,
        cfg=cfg,
        ms=ms,
        title=(
            f"val window {val_panel['window_idx']} (OOD) | "
            f"horizon bins {val_panel['fut_bins'].min()}–{val_panel['fut_bins'].max()} | "
            f"{int(val_panel['ood_mask'].sum())} OOD steps"
        ),
        is_val=True,
    )

    axes[0, 0].legend(
        handles=[
            plt.Line2D([0], [0], color="#1f77b4", lw=1.2, label="train window-norm"),
            plt.Line2D([0], [0], color="#D62728", lw=1.2, label="val window-norm"),
            plt.Rectangle((0, 0), 1, 1, fc="#C8E6C9", alpha=0.45, label=f"train bin envelope"),
            plt.Rectangle((0, 0), 1, 1, fc="#FFF3CD", alpha=0.35, label="horizon"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        ncol=4,
        fontsize=8,
    )

    fig.subplots_adjust(top=0.88, hspace=0.28, wspace=0.12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def process_dataset(
    dataset: str,
    *,
    config_path: Path,
    output_dir: Path,
    max_variates: int,
    max_examples_per_variate: int,
) -> Dict:
    cfg = _load_cfg(config_path)
    ms = _max_scale_for(dataset, cfg, config_path)
    path, date_col = _resolve_registry_path(dataset)
    raw = _load_dataset_array(path, date_col).astype(np.float64)
    names = _variate_names(path, date_col, raw.shape[1])

    n = len(raw)
    b1, b2 = _paper_split_borders(dataset, n, cfg.lookback)
    train_end = b2[0]
    z = (raw - raw[:train_end].mean(0, keepdims=True)) / (
        raw[:train_end].std(0, keepdims=True) + ZSCORE_EPS
    )
    tmin, tmax = _train_envelope(z, b1[0], b2[0], cfg, ms, raw.shape[1])
    examples = _find_ood_examples(
        z,
        b1[1],
        b2[1],
        cfg,
        ms,
        tmin,
        tmax,
        max_per_variate=max_examples_per_variate,
        max_variates=max_variates,
    )

    ds_out = output_dir / dataset
    ds_out.mkdir(parents=True, exist_ok=True)
    saved: List[Dict] = []
    seen_variates: set[int] = set()
    for ex in examples:
        vi = ex["variate_idx"]
        if vi not in seen_variates:
            seen_variates.add(vi)
        label = names[vi] if vi < len(names) else f"var_{vi}"
        safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(label))[:48]
        n_for_var = sum(1 for s in saved if s["variate_idx"] == vi)
        fname = f"{dataset}_v{vi}_{safe_label}_win{ex['window_idx']}.png"
        out_path = ds_out / fname
        train_panel = _find_representative_train_window(
            z, b1[0], b2[0], cfg, ms, vi, ex["train_min_bin"], ex["train_max_bin"],
        )
        _plot_train_val_pair(
            dataset=dataset,
            variate_label=label,
            train_panel=train_panel,
            val_panel=ex,
            cfg=cfg,
            ms=ms,
            output_path=out_path,
        )
        saved.append(
            {
                "variate_idx": vi,
                "variate_label": label,
                "val_window_idx": ex["window_idx"],
                "train_window_idx": train_panel["window_idx"],
                "n_ood_timesteps": ex["n_ood_timesteps"],
                "train_min_bin": ex["train_min_bin"],
                "train_max_bin": ex["train_max_bin"],
                "train_horizon_bin_min": int(train_panel["fut_bins"].min()),
                "train_horizon_bin_max": int(train_panel["fut_bins"].max()),
                "val_horizon_bin_min": int(ex["fut_bins"].min()),
                "val_horizon_bin_max": int(ex["fut_bins"].max()),
                "plot": str(out_path.relative_to(REPO)),
            }
        )
        if len(seen_variates) >= max_variates and n_for_var + 1 >= max_examples_per_variate:
            pass

    meta = {
        "dataset": dataset,
        "n_variates": int(raw.shape[1]),
        "max_scale": ms,
        "train_bin_range_pooled": [int(tmin.min()), int(tmax.max())],
        "n_plots": len(saved),
        "examples": saved,
    }
    with (ds_out / "examples.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[{dataset}] saved {len(saved)} plots -> {ds_out.relative_to(REPO)}")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-variates", type=int, default=3, help="Max distinct OOD variates per dataset")
    parser.add_argument("--max-examples-per-variate", type=int, default=1)
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_meta = []
    for ds in datasets:
        try:
            all_meta.append(
                process_dataset(
                    ds,
                    config_path=args.config,
                    output_dir=args.output_dir,
                    max_variates=args.max_variates,
                    max_examples_per_variate=args.max_examples_per_variate,
                )
            )
        except Exception as exc:
            print(f"[fail] {ds}: {exc}", file=sys.stderr)

    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(all_meta, f, indent=2)
    print(f"Wrote {summary_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
