#!/usr/bin/env python3
"""Trend / ACF recoverability under binary bit-flip noise vs sequence length.

At the same beta_t, longer CDF maps (336/720) can retain more *recoverable*
temporal structure than short ones (96/96) because autocorrelation is
redundant across time. This script measures that.

For each geometry in {96/96, 336/720_uncompressed} and t in
{0, T/8, ..., T}:
  - bit-flip coarse CDF maps at beta_t
  - decode to 1D
  - compare linear-trend fit and ACF(lags) vs clean (t=0)
  - average over validation windows

Outputs trend-R^2 and ACF-lag1 curves + noise-floor crossing % of T.

Example:
  python utils/diagnose_binary_noise_trend_recoverability.py --datasets ETTh1
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.diffusion_tsf.diffusion import BinaryDiffusionScheduler
from models.diffusion_tsf.pipeline.config import load_experiment_config
from models.diffusion_tsf.pipeline.globals_bridge import patch_globals
from models.diffusion_tsf.pipeline.phases.staged_diffusion_pretrain import patch_stage_globals
from models.diffusion_tsf.pipeline.state import PipelineState
from models.diffusion_tsf.pipeline.visualize_utils import save_figure_jpg
from models.diffusion_tsf.train_multivariate_pipeline import (
    create_diffusion_model,
    create_patch_guidance_stack,
    load_dataset,
    resolve_pipeline_data_subset,
    wrap_patch_guidance,
)
import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod

GEOMETRY_CONFIGS = {
    "96/96": "configs/binary_anchor_ar_patch_decoder_ctx.yaml",
    "336/720_uncompressed": (
        "configs/binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_uncompressed.yaml"
    ),
}

DEFAULT_DATASETS = "ETTh1,weather,electricity,exchange_rate,traffic"
DEFAULT_FRACTIONS = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)
DEFAULT_ACF_LAGS = (1, 5, 10)


def _build_state(config_path: str, dataset: str) -> PipelineState:
    cfg = load_experiment_config(config_path, cli_overrides={"dataset": dataset})
    state = PipelineState.from_config(cfg)
    state.dataset = dataset
    resolve_pipeline_data_subset(state)
    state.subset_id = state.subset_id or dataset
    return state


def _load_val_windows(
    state: PipelineState,
    *,
    n_samples: int,
    seed: int,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], Dict[str, Any]]:
    meta = state.data_subset_resolved or {}
    _, val_ds, _, norm_stats = load_dataset(
        state.dataset,
        list(state.variate_indices),
        lookback=int(state.lookback_length),
        horizon=int(state.forecast_length),
        stride=int(meta.get("train_stride", state.window_stride)),
        test_stride=1,
        ordinal_tie_atol=float(state.ordinal_tie_atol),
        use_ordinal_window_norm=state.use_ordinal_window_norm,
    )
    if norm_stats.get("ordinal_ladder") is not None:
        state.extra["global_ordinal_ladder"] = norm_stats["ordinal_ladder"]
        pipeline_mod.GLOBAL_ORDINAL_LADDER = norm_stats["ordinal_ladder"]
    rng = np.random.default_rng(seed)
    n = len(val_ds)
    if n == 0:
        raise RuntimeError(f"{state.dataset}: empty validation split")
    idxs = rng.choice(n, size=min(n_samples, n), replace=False)
    return [val_ds[int(i)] for i in idxs], norm_stats


def _make_model(state: PipelineState, stage: str = "coarse", ordinal_ladder=None):
    patch_globals(pipeline_mod, state, honor_dataset_windows=True)
    patch_stage_globals(pipeline_mod, state, stage, honor_dataset_windows=True)
    if ordinal_ladder is not None:
        pipeline_mod.GLOBAL_ORDINAL_LADDER = ordinal_ladder
    n_vars = len(state.variate_indices or [])
    lookback = int(state.lookback_length)
    horizon = int(state.forecast_length)
    stack = create_patch_guidance_stack(n_vars, in_len=lookback, out_len=horizon)
    guidance = wrap_patch_guidance(stack)
    model = create_diffusion_model(
        guidance_model=guidance,
        n_variates=n_vars,
        lookback=lookback,
        horizon=horizon,
        diffusion_stage=stage,
        ordinal_ladder=ordinal_ladder,
    )
    model.eval()
    return model, n_vars


def _scheduler_from_state(state: PipelineState) -> BinaryDiffusionScheduler:
    return BinaryDiffusionScheduler(
        num_steps=int(state.binary_num_steps),
        beta_start=float(state.binary_beta_start),
        beta_end=float(state.binary_beta_end),
        schedule_type=str(state.binary_noise_schedule),
        device="cpu",
    )


def _noise_at_t(sched: BinaryDiffusionScheduler, x0: torch.Tensor, t_idx: int) -> torch.Tensor:
    t_idx = int(min(max(0, t_idx), sched.num_steps - 1))
    beta = float(sched.betas[t_idx].item())
    zt = torch.bernoulli(torch.full_like(x0, beta))
    return (x0.bool() ^ zt.bool()).float()


@torch.no_grad()
def _encode_coarse_maps(model, past: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
    """Coarse binary CDF for future: (V, H, W)."""
    past_b = past.unsqueeze(0)
    future_b = future.unsqueeze(0)
    _past_norm, future_norm, _ = model._normalize_sequence(past_b, future_b)
    maps = model._encode_staged_maps(future_norm)
    return maps["coarse"][0].detach().cpu().float().clamp(0, 1)


@torch.no_grad()
def _decode_coarse_1d(model, coarse_map_vhw: torch.Tensor) -> np.ndarray:
    """(V, H, W) -> (V, W) decoded 1D values."""
    x = coarse_map_vhw.unsqueeze(0)  # (1, V, H, W)
    y = model._decode_coarse_1d_from_map(x, cdf_decoder="mean")
    # y: (1, V, W) or (V, W)
    if y.dim() == 3:
        y = y[0]
    return y.detach().cpu().numpy()


def _linear_trend(y: np.ndarray) -> np.ndarray:
    """Per-variate linear regression fit; y is (V, T) or (T,)."""
    if y.ndim == 1:
        t = np.arange(y.shape[0], dtype=np.float64)
        coef = np.polyfit(t, y.astype(np.float64), 1)
        return coef[0] * t + coef[1]
    out = np.empty_like(y, dtype=np.float64)
    t = np.arange(y.shape[1], dtype=np.float64)
    for v in range(y.shape[0]):
        coef = np.polyfit(t, y[v].astype(np.float64), 1)
        out[v] = coef[0] * t + coef[1]
    return out


def _moving_average(y: np.ndarray, frac: float = 0.08) -> np.ndarray:
    """Centered MA; window ~frac of length (odd, >=3)."""
    if y.ndim == 1:
        T = y.shape[0]
        w = max(3, int(round(frac * T)) | 1)
        pad = w // 2
        yp = np.pad(y.astype(np.float64), (pad, pad), mode="edge")
        kernel = np.ones(w, dtype=np.float64) / w
        return np.convolve(yp, kernel, mode="valid")
    return np.stack([_moving_average(y[v], frac=frac) for v in range(y.shape[0])], axis=0)


def _pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _r2_between(a: np.ndarray, b: np.ndarray) -> float:
    """R^2 of predicting a from b (same shape), after flattening."""
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    ss_res = float(np.sum((a - b) ** 2))
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    if ss_tot < 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def _acf_at_lags(y: np.ndarray, lags: Sequence[int]) -> Dict[int, float]:
    """Mean ACF across variates for each lag. y: (V, T) or (T,)."""
    if y.ndim == 1:
        y = y[None, :]
    out: Dict[int, float] = {}
    for lag in lags:
        vals = []
        for v in range(y.shape[0]):
            s = y[v].astype(np.float64)
            if lag >= s.shape[0] - 1:
                vals.append(0.0)
                continue
            a = s[:-lag] - s[:-lag].mean()
            b = s[lag:] - s[lag:].mean()
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            vals.append(0.0 if denom < 1e-12 else float(np.dot(a, b) / denom))
        out[int(lag)] = float(np.mean(vals))
    return out


def _crossing_pct(
    fractions: Sequence[float],
    values: Sequence[float],
    threshold: float,
) -> Optional[float]:
    """First fraction where value drops below threshold; None if never."""
    for f, v in zip(fractions, values):
        if v < threshold:
            return float(f)
    return None


def _noise_floor_metrics(
    model,
    map_shape: Tuple[int, ...],
    *,
    n_draws: int,
    acf_lags: Sequence[int],
    seed: int,
) -> Dict[str, float]:
    """Empirical floors from independent Bernoulli(0.5) maps."""
    rng = np.random.default_rng(seed)
    trend_r2s = []
    ma_rs = []
    acf_vals: Dict[int, List[float]] = {int(l): [] for l in acf_lags}
    for _ in range(n_draws):
        a = torch.from_numpy(rng.binomial(1, 0.5, size=map_shape).astype(np.float32))
        b = torch.from_numpy(rng.binomial(1, 0.5, size=map_shape).astype(np.float32))
        ya = _decode_coarse_1d(model, a)
        yb = _decode_coarse_1d(model, b)
        ta = _linear_trend(ya)
        tb = _linear_trend(yb)
        # Near-flat trends make R² unstable; clamp using Pearson as well.
        r2 = max(0.0, _r2_between(ta, tb))
        if float(np.std(ta)) < 1e-6 or float(np.std(tb)) < 1e-6:
            r2 = 0.0
        trend_r2s.append(r2)
        ma_rs.append(abs(_pearson_r(_moving_average(ya), _moving_average(yb))))
        for lag, val in _acf_at_lags(ya, acf_lags).items():
            acf_vals[lag].append(abs(val))
    out = {
        "trend_r2_floor": float(np.mean(trend_r2s)),
        "ma_r_floor": float(np.mean(ma_rs)),
    }
    for lag, vals in acf_vals.items():
        out[f"acf_lag{lag}_floor"] = float(np.mean(vals))
    return out


def _crossing_threshold(floor: float, *, absolute_cap: float = 0.25) -> float:
    """1.5× noise floor, but never above absolute_cap (avoids 1.5×floor > 1)."""
    return float(min(absolute_cap, max(0.05, 1.5 * max(0.0, floor))))


def evaluate_geometry(
    *,
    geometry: str,
    dataset: str,
    n_samples: int,
    seed: int,
    stage: str,
    fractions: Sequence[float],
    acf_lags: Sequence[int],
) -> Dict[str, Any]:
    config_path = GEOMETRY_CONFIGS[geometry]
    state = _build_state(config_path, dataset)
    windows, norm_stats = _load_val_windows(state, n_samples=n_samples, seed=seed)
    ladder = norm_stats.get("ordinal_ladder")
    model, n_vars = _make_model(state, stage=stage, ordinal_ladder=ladder)
    sched = _scheduler_from_state(state)
    maps = [_encode_coarse_maps(model, past, future) for past, future in windows]
    clean_1d = [_decode_coarse_1d(model, m) for m in maps]
    clean_trends = [_linear_trend(y) for y in clean_1d]
    clean_acfs = [_acf_at_lags(y, acf_lags) for y in clean_1d]

    T = sched.num_steps
    t_idxs = [int(round(f * (T - 1))) for f in fractions]
    trend_r2_mean = []
    trend_r_mean = []
    ma_r_mean = []
    acf_corrupted: Dict[int, List[float]] = {int(l): [] for l in acf_lags}
    acf_clean_ref: Dict[int, float] = {
        int(l): float(np.mean([c[int(l)] for c in clean_acfs])) for l in acf_lags
    }
    clean_mas = [_moving_average(y) for y in clean_1d]

    for t_idx in t_idxs:
        r2s = []
        rs = []
        ma_rs = []
        acf_batch: Dict[int, List[float]] = {int(l): [] for l in acf_lags}
        for m, ct, cma in zip(maps, clean_trends, clean_mas):
            xt = m if t_idx == 0 else _noise_at_t(sched, m, t_idx)
            y = _decode_coarse_1d(model, xt)
            tr = _linear_trend(y)
            r2 = max(0.0, _r2_between(ct, tr))
            if float(np.std(ct)) < 1e-6:
                r2 = 0.0
            r2s.append(r2)
            rs.append(_pearson_r(ct, tr))
            ma_rs.append(_pearson_r(cma, _moving_average(y)))
            for lag, val in _acf_at_lags(y, acf_lags).items():
                acf_batch[lag].append(val)
        trend_r2_mean.append(float(np.mean(r2s)))
        trend_r_mean.append(float(np.mean(rs)))
        ma_r_mean.append(float(np.mean(ma_rs)))
        for lag in acf_lags:
            acf_corrupted[int(lag)].append(float(np.mean(acf_batch[int(lag)])))

    floors = _noise_floor_metrics(
        model,
        tuple(maps[0].shape),
        n_draws=max(8, min(20, n_samples)),
        acf_lags=acf_lags,
        seed=seed + 17,
    )

    return {
        "geometry": geometry,
        "dataset": dataset,
        "subset_id": str(state.subset_id),
        "n_variates": n_vars,
        "lookback": int(state.lookback_length),
        "horizon": int(state.forecast_length),
        "map_shape": list(maps[0].shape),
        "n_samples": len(maps),
        "schedule": sched.schedule_type,
        "num_steps": T,
        "fractions": list(fractions),
        "t_idxs": t_idxs,
        "betas": [float(sched.betas[t].item()) for t in t_idxs],
        "trend_r2": trend_r2_mean,
        "trend_r": trend_r_mean,
        "ma_r": ma_r_mean,
        "acf_corrupted": {str(k): v for k, v in acf_corrupted.items()},
        "acf_clean": {str(k): v for k, v in acf_clean_ref.items()},
        "floors": floors,
    }


def plot_curves(
    results: Dict[str, Dict[str, Any]],
    *,
    dataset: str,
    out_dir: Path,
    jpeg_dpi: int,
    primary_acf_lag: int = 1,
) -> Tuple[Path, Path, Path]:
    # Plot 1: linear-trend R²
    fig, ax = plt.subplots(figsize=(8, 4.5))
    floors = []
    for geometry, res in results.items():
        fr = np.asarray(res["fractions"], dtype=np.float64)
        ax.plot(fr, res["trend_r2"], marker="o", label=f"{geometry} (W={res['map_shape'][-1]})")
        floors.append(res["floors"]["trend_r2_floor"])
    floor = float(np.mean(floors))
    thr = _crossing_threshold(floor)
    ax.axhline(floor, color="gray", linestyle="--", label=f"noise floor R²={floor:.3f}")
    ax.axhline(thr, color="gray", linestyle=":", alpha=0.7, label=f"cross thr={thr:.3f}")
    ax.set_xlabel("t / (T-1)")
    ax.set_ylabel("trend R² (clean fit vs corrupted fit)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"{dataset}: linear-trend recoverability vs bit-flip t")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p1 = out_dir / f"trend_r2_{dataset}.jpg"
    save_figure_jpg(fig, str(p1), dpi=jpeg_dpi)
    plt.close(fig)

    # Plot 1b: MA Pearson r (more stable shape recoverability)
    fig_ma, ax_ma = plt.subplots(figsize=(8, 4.5))
    floors_ma = []
    for geometry, res in results.items():
        fr = np.asarray(res["fractions"], dtype=np.float64)
        ax_ma.plot(fr, res["ma_r"], marker="o", label=f"{geometry} (W={res['map_shape'][-1]})")
        floors_ma.append(res["floors"]["ma_r_floor"])
    floor_ma = float(np.mean(floors_ma))
    thr_ma = _crossing_threshold(floor_ma, absolute_cap=0.35)
    ax_ma.axhline(floor_ma, color="gray", linestyle="--", label=f"noise floor |r|={floor_ma:.3f}")
    ax_ma.axhline(thr_ma, color="gray", linestyle=":", alpha=0.7, label=f"cross thr={thr_ma:.3f}")
    ax_ma.set_xlabel("t / (T-1)")
    ax_ma.set_ylabel("Pearson r (MA clean vs MA corrupted)")
    ax_ma.set_ylim(-0.05, 1.05)
    ax_ma.set_title(f"{dataset}: moving-average recoverability vs bit-flip t")
    ax_ma.grid(True, alpha=0.3)
    ax_ma.legend(fontsize=8)
    fig_ma.tight_layout()
    p_ma = out_dir / f"ma_r_{dataset}.jpg"
    save_figure_jpg(fig_ma, str(p_ma), dpi=jpeg_dpi)
    plt.close(fig_ma)

    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    floors2 = []
    for geometry, res in results.items():
        fr = np.asarray(res["fractions"], dtype=np.float64)
        series = res["acf_corrupted"][str(primary_acf_lag)]
        ax2.plot(fr, series, marker="o", label=f"{geometry} ACF(lag={primary_acf_lag})")
        floors2.append(res["floors"][f"acf_lag{primary_acf_lag}_floor"])
        ax2.axhline(
            res["acf_clean"][str(primary_acf_lag)],
            linestyle="--",
            alpha=0.35,
            label=f"{geometry} clean ACF",
        )
    floor2 = float(np.mean(floors2))
    thr2 = _crossing_threshold(floor2, absolute_cap=0.35)
    ax2.axhline(floor2, color="gray", linestyle="--", label=f"noise |ACF| floor={floor2:.3f}")
    ax2.axhline(thr2, color="gray", linestyle=":", alpha=0.7, label=f"cross thr={thr2:.3f}")
    ax2.set_xlabel("t / (T-1)")
    ax2.set_ylabel(f"ACF lag={primary_acf_lag}")
    ax2.set_title(f"{dataset}: ACF recoverability vs bit-flip t")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=7)
    fig2.tight_layout()
    p2 = out_dir / f"acf_lag{primary_acf_lag}_{dataset}.jpg"
    save_figure_jpg(fig2, str(p2), dpi=jpeg_dpi)
    plt.close(fig2)
    return p1, p_ma, p2


def print_crossings(results: Dict[str, Dict[str, Any]], *, primary_acf_lag: int = 1) -> List[Dict[str, Any]]:
    rows = []
    print("\n=== effective full-corruption points (first t where metric < cross thr) ===")
    for geometry, res in results.items():
        fr = res["fractions"]
        tr_thr = _crossing_threshold(res["floors"]["trend_r2_floor"])
        ma_thr = _crossing_threshold(res["floors"]["ma_r_floor"], absolute_cap=0.35)
        ac_thr = _crossing_threshold(res["floors"][f"acf_lag{primary_acf_lag}_floor"], absolute_cap=0.35)
        tr_cross = _crossing_pct(fr, res["trend_r2"], tr_thr)
        ma_cross = _crossing_pct(fr, res["ma_r"], ma_thr)
        ac_cross = _crossing_pct(fr, res["acf_corrupted"][str(primary_acf_lag)], ac_thr)
        def _fmt(x):
            return "never" if x is None else f"{100 * x:.1f}% of T"
        print(
            f"  {geometry:24s}  trend_R² @ {_fmt(tr_cross):14s}  "
            f"MA_r @ {_fmt(ma_cross):14s}  "
            f"ACF(lag={primary_acf_lag}) @ {_fmt(ac_cross)}  "
            f"(thr trend={tr_thr:.3f}, ma={ma_thr:.3f}, acf={ac_thr:.3f})"
        )
        rows.append({
            "geometry": geometry,
            "subset_id": res["subset_id"],
            "trend_r2_floor": res["floors"]["trend_r2_floor"],
            "trend_cross_frac": tr_cross if tr_cross is not None else "",
            "ma_r_floor": res["floors"]["ma_r_floor"],
            "ma_cross_frac": ma_cross if ma_cross is not None else "",
            "acf_lag": primary_acf_lag,
            "acf_floor": res["floors"][f"acf_lag{primary_acf_lag}_floor"],
            "acf_cross_frac": ac_cross if ac_cross is not None else "",
            "map_W": res["map_shape"][-1],
        })
    geos = list(results.keys())
    if len(geos) == 2:
        a, b = geos[0], geos[1]
        ca = _crossing_pct(
            results[a]["fractions"],
            results[a]["ma_r"],
            _crossing_threshold(results[a]["floors"]["ma_r_floor"], absolute_cap=0.35),
        )
        cb = _crossing_pct(
            results[b]["fractions"],
            results[b]["ma_r"],
            _crossing_threshold(results[b]["floors"]["ma_r_floor"], absolute_cap=0.35),
        )
        if ca is not None and cb is not None:
            if abs(ca - cb) <= 0.05:
                print("[flag] MA recoverability crosses at similar %T — schedule looks length-invariant.")
            else:
                later = a if ca > cb else b
                print(
                    f"[flag] {later} retains recoverable MA structure longer "
                    f"({100 * max(ca, cb):.1f}% vs {100 * min(ca, cb):.1f}% of T) — "
                    "consider length-dependent beta shift."
                )
        else:
            print("[flag] at least one geometry never crossed MA threshold within the grid.")
    return rows


def run_dataset(
    *,
    dataset: str,
    geometries: Sequence[str],
    n_samples: int,
    seed: int,
    stage: str,
    out_dir: Path,
    jpeg_dpi: int,
    acf_lags: Sequence[int],
    primary_acf_lag: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict[str, Any]] = {}
    for geometry in geometries:
        print(f"[eval] {dataset} {geometry} n={n_samples}", flush=True)
        results[geometry] = evaluate_geometry(
            geometry=geometry,
            dataset=dataset,
            n_samples=n_samples,
            seed=seed,
            stage=stage,
            fractions=DEFAULT_FRACTIONS,
            acf_lags=acf_lags,
        )
        res = results[geometry]
        print(
            f"  map={res['map_shape']} β_grid={[round(b, 3) for b in res['betas']]} "
            f"trend_r2={[round(v, 3) for v in res['trend_r2']]}",
            flush=True,
        )

    p1, p_ma, p2 = plot_curves(
        results,
        dataset=dataset,
        out_dir=out_dir,
        jpeg_dpi=jpeg_dpi,
        primary_acf_lag=primary_acf_lag,
    )
    print(f"[plot] {p1}")
    print(f"[plot] {p_ma}")
    print(f"[plot] {p2}")
    cross_rows = print_crossings(results, primary_acf_lag=primary_acf_lag)

    # per-t CSV
    csv_path = out_dir / f"metrics_{dataset}.csv"
    rows = []
    for geometry, res in results.items():
        for i, f in enumerate(res["fractions"]):
            row = {
                "dataset": dataset,
                "geometry": geometry,
                "subset_id": res["subset_id"],
                "frac": f,
                "t": res["t_idxs"][i],
                "beta": res["betas"][i],
                "trend_r2": res["trend_r2"][i],
                "trend_r": res["trend_r"][i],
                "ma_r": res["ma_r"][i],
            }
            for lag in acf_lags:
                row[f"acf_lag{lag}"] = res["acf_corrupted"][str(lag)][i]
                row[f"acf_lag{lag}_clean"] = res["acf_clean"][str(lag)]
            row.update({f"floor_{k}": v for k, v in res["floors"].items()})
            rows.append(row)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    cross_path = out_dir / f"crossings_{dataset}.csv"
    with cross_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cross_rows[0].keys()))
        w.writeheader()
        w.writerows(cross_rows)
    print(f"[csv] {csv_path}")
    print(f"[csv] {cross_path}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", default=DEFAULT_DATASETS)
    p.add_argument("--geometries", default="96/96,336/720_uncompressed")
    p.add_argument("--n-samples", type=int, default=24)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--stage", default="coarse", choices=("coarse", "fine"))
    p.add_argument("--acf-lags", default="1,5,10")
    p.add_argument("--primary-acf-lag", type=int, default=1)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "reports" / "noise_trend_recoverability",
    )
    p.add_argument("--jpeg-dpi", type=int, default=110)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    geometries = [g.strip() for g in args.geometries.split(",") if g.strip()]
    for g in geometries:
        if g not in GEOMETRY_CONFIGS:
            raise ValueError(f"unknown geometry {g!r}")
    acf_lags = tuple(int(x) for x in args.acf_lags.split(",") if x.strip())
    out_root = args.output_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        ds_dir = out_root / dataset
        print(f"==== {dataset} -> {ds_dir} ====", flush=True)
        run_dataset(
            dataset=dataset,
            geometries=geometries,
            n_samples=int(args.n_samples),
            seed=int(args.seed),
            stage=str(args.stage),
            out_dir=ds_dir,
            jpeg_dpi=int(args.jpeg_dpi),
            acf_lags=acf_lags,
            primary_acf_lag=int(args.primary_acf_lag),
        )
    print(f"[done] {out_root}", flush=True)


if __name__ == "__main__":
    main()
