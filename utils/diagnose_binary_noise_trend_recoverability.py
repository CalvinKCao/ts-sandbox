#!/usr/bin/env python3
"""Trend / ACF recoverability under binary bit-flip noise vs sequence length.

At the same beta_t, longer CDF maps (336/720) can retain more *recoverable*
temporal structure than short ones (96/96) because MA/ACF average over more
independent flips. This script measures that and optionally remaps β_t with a
length-dependent schedule so both geometries hit the noise floor at similar
fractions of T.

Noise floor (fixed): clean decoded series vs Bernoulli(0.5) maps — NOT
noise-vs-noise (two long MAs both collapse to the same mean → spurious r≈1).

Example:
  python utils/diagnose_binary_noise_trend_recoverability.py --datasets traffic \\
      --length-mode power --g-cal 1.5
  python utils/diagnose_binary_noise_trend_recoverability.py \\
      --recompute-crossings-from reports/noise_trend_recoverability_4146642
"""

from __future__ import annotations

import argparse
import csv
import json
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

from models.diffusion_tsf.diffusion import (
    BinaryDiffusionScheduler,
    length_schedule_g,
    length_schedule_scale,
)
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
# Coarse map widths used as calibration anchors for g(L) / scale(L).
L_REF = 104.0   # 96/96 map W
L_CAL = 728.0   # 336/720_uncompressed map W


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


def _resolve_length_params(
    map_w: int,
    *,
    length_mode: str,
    g_cal: float,
    scale_cal: float,
    g_override: Optional[float],
    scale_override: Optional[float],
) -> Tuple[str, float, float]:
    mode = (length_mode or "none").lower()
    if mode == "none":
        return "none", 1.0, 1.0
    if mode == "power":
        g = float(g_override) if g_override is not None else length_schedule_g(
            map_w, l_ref=L_REF, g_ref=1.0, l_cal=L_CAL, g_cal=g_cal
        )
        return "power", g, 1.0
    if mode == "scale":
        sc = float(scale_override) if scale_override is not None else length_schedule_scale(
            map_w, l_ref=L_REF, scale_ref=1.0, l_cal=L_CAL, scale_cal=scale_cal
        )
        return "scale", 1.0, sc
    raise ValueError(f"unknown length_mode {length_mode!r}")


def _scheduler_from_state(
    state: PipelineState,
    *,
    length_mode: str = "none",
    length_g: float = 1.0,
    length_scale: float = 1.0,
) -> BinaryDiffusionScheduler:
    return BinaryDiffusionScheduler(
        num_steps=int(state.binary_num_steps),
        beta_start=float(state.binary_beta_start),
        beta_end=float(state.binary_beta_end),
        schedule_type=str(state.binary_noise_schedule),
        device="cpu",
        length_mode=length_mode,
        length_g=length_g,
        length_scale=length_scale,
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


def _ma_width(T: int, *, ma_window: str, ma_frac: float = 0.08) -> int:
    """frac: window ~frac*T; fixed_ref: same odd width as at L_REF (length-fair)."""
    if ma_window == "fixed_ref":
        return max(3, int(round(ma_frac * L_REF)) | 1)
    return max(3, int(round(ma_frac * T)) | 1)


def _moving_average(
    y: np.ndarray,
    frac: float = 0.08,
    *,
    ma_window: str = "frac",
) -> np.ndarray:
    """Centered MA. ma_window=fixed_ref uses L_REF-based width for all lengths."""
    if y.ndim == 1:
        T = y.shape[0]
        w = _ma_width(T, ma_window=ma_window, ma_frac=frac)
        pad = w // 2
        yp = np.pad(y.astype(np.float64), (pad, pad), mode="edge")
        kernel = np.ones(w, dtype=np.float64) / w
        return np.convolve(yp, kernel, mode="valid")
    return np.stack(
        [_moving_average(y[v], frac=frac, ma_window=ma_window) for v in range(y.shape[0])],
        axis=0,
    )


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
    clean_1d: Sequence[np.ndarray],
    map_shape: Tuple[int, ...],
    *,
    n_draws: int,
    acf_lags: Sequence[int],
    seed: int,
    ma_window: str,
) -> Dict[str, float]:
    """Empirical floors: clean decoded series vs Bernoulli(0.5) maps.

    Noise-vs-noise is wrong for long MA windows: two independent long MAs both
    collapse toward the same constant mean, so Pearson r stays near 1.
    """
    rng = np.random.default_rng(seed)
    trend_r2s = []
    ma_rs = []
    acf_vals: Dict[int, List[float]] = {int(l): [] for l in acf_lags}
    clean_trends = [_linear_trend(y) for y in clean_1d]
    clean_mas = [_moving_average(y, ma_window=ma_window) for y in clean_1d]
    n_clean = len(clean_1d)
    for i in range(n_draws):
        noise = torch.from_numpy(rng.binomial(1, 0.5, size=map_shape).astype(np.float32))
        yn = _decode_coarse_1d(model, noise)
        ct = clean_trends[i % n_clean]
        cma = clean_mas[i % n_clean]
        tn = _linear_trend(yn)
        r2 = max(0.0, _r2_between(ct, tn))
        if float(np.std(ct)) < 1e-6 or float(np.std(tn)) < 1e-6:
            r2 = 0.0
        trend_r2s.append(r2)
        ma_rs.append(abs(_pearson_r(cma, _moving_average(yn, ma_window=ma_window))))
        for lag, val in _acf_at_lags(yn, acf_lags).items():
            acf_vals[lag].append(abs(val))
    out = {
        "trend_r2_floor": float(np.mean(trend_r2s)),
        "ma_r_floor": float(np.mean(ma_rs)),
    }
    for lag, vals in acf_vals.items():
        out[f"acf_lag{lag}_floor"] = float(np.mean(vals))
    return out


def _crossing_threshold(floor: float, *, absolute_cap: float = 0.25, margin: float = 0.05) -> float:
    """Threshold for 'reached noise floor'.

    Prefer 1.5×floor when that stays below absolute_cap (short maps). When the
    empirical floor itself is high (residual length structure), use floor+margin
    so a curve that lands on the floor still counts as crossed.
    """
    floor = float(max(0.0, floor))
    scaled = 1.5 * floor
    if scaled <= absolute_cap:
        return float(max(0.05, scaled))
    return float(floor + margin)


def _prepare_geometry_cache(
    *,
    geometry: str,
    dataset: str,
    n_samples: int,
    seed: int,
    stage: str,
    acf_lags: Sequence[int],
    ma_window: str,
) -> Dict[str, Any]:
    """Encode/decode once; reuse across g grid candidates."""
    config_path = GEOMETRY_CONFIGS[geometry]
    state = _build_state(config_path, dataset)
    windows, norm_stats = _load_val_windows(state, n_samples=n_samples, seed=seed)
    ladder = norm_stats.get("ordinal_ladder")
    model, n_vars = _make_model(state, stage=stage, ordinal_ladder=ladder)
    maps = [_encode_coarse_maps(model, past, future) for past, future in windows]
    clean_1d = [_decode_coarse_1d(model, m) for m in maps]
    floors = _noise_floor_metrics(
        model,
        clean_1d,
        tuple(maps[0].shape),
        n_draws=max(8, min(20, n_samples)),
        acf_lags=acf_lags,
        seed=seed + 17,
        ma_window=ma_window,
    )
    return {
        "geometry": geometry,
        "dataset": dataset,
        "state": state,
        "model": model,
        "n_vars": n_vars,
        "maps": maps,
        "clean_1d": clean_1d,
        "floors": floors,
        "ma_window": ma_window,
    }


def _eval_from_cache(
    cache: Dict[str, Any],
    *,
    fractions: Sequence[float],
    acf_lags: Sequence[int],
    length_mode: str,
    g_cal: float,
    scale_cal: float,
    g_override: Optional[float],
    scale_override: Optional[float],
) -> Dict[str, Any]:
    geometry = cache["geometry"]
    state = cache["state"]
    model = cache["model"]
    maps = cache["maps"]
    clean_1d = cache["clean_1d"]
    ma_window = cache["ma_window"]
    n_vars = cache["n_vars"]

    clean_trends = [_linear_trend(y) for y in clean_1d]
    clean_acfs = [_acf_at_lags(y, acf_lags) for y in clean_1d]
    clean_mas = [_moving_average(y, ma_window=ma_window) for y in clean_1d]
    map_w = int(maps[0].shape[-1])
    ma_w = _ma_width(map_w, ma_window=ma_window)

    apply_mode = length_mode
    if length_mode != "none" and geometry == "96/96" and g_override is None and scale_override is None:
        apply_mode = "none"
    mode, length_g, length_scale = _resolve_length_params(
        map_w,
        length_mode=apply_mode,
        g_cal=g_cal,
        scale_cal=scale_cal,
        g_override=g_override if apply_mode != "none" else None,
        scale_override=scale_override if apply_mode != "none" else None,
    )
    sched = _scheduler_from_state(
        state, length_mode=mode, length_g=length_g, length_scale=length_scale
    )
    print(
        f"  [sched] {geometry} W={map_w} ma_w={ma_w} ({ma_window}) mode={mode} "
        f"g={length_g:.4f} scale={length_scale:.4f} β_end={float(sched.betas[-1]):.4f}",
        flush=True,
    )

    T = sched.num_steps
    t_idxs = [int(round(f * (T - 1))) for f in fractions]
    trend_r2_mean = []
    trend_r_mean = []
    ma_r_mean = []
    acf_corrupted: Dict[int, List[float]] = {int(l): [] for l in acf_lags}
    acf_clean_ref: Dict[int, float] = {
        int(l): float(np.mean([c[int(l)] for c in clean_acfs])) for l in acf_lags
    }

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
            ma_rs.append(_pearson_r(cma, _moving_average(y, ma_window=ma_window)))
            for lag, val in _acf_at_lags(y, acf_lags).items():
                acf_batch[lag].append(val)
        trend_r2_mean.append(float(np.mean(r2s)))
        trend_r_mean.append(float(np.mean(rs)))
        ma_r_mean.append(float(np.mean(ma_rs)))
        for lag in acf_lags:
            acf_corrupted[int(lag)].append(float(np.mean(acf_batch[int(lag)])))

    return {
        "geometry": geometry,
        "dataset": cache["dataset"],
        "subset_id": str(state.subset_id),
        "n_variates": n_vars,
        "lookback": int(state.lookback_length),
        "horizon": int(state.forecast_length),
        "map_shape": list(maps[0].shape),
        "ma_window": ma_window,
        "ma_width": ma_w,
        "n_samples": len(maps),
        "schedule": sched.schedule_type,
        "length_mode": mode,
        "length_g": length_g,
        "length_scale": length_scale,
        "num_steps": T,
        "fractions": list(fractions),
        "t_idxs": t_idxs,
        "betas": [float(sched.betas[t].item()) for t in t_idxs],
        "trend_r2": trend_r2_mean,
        "trend_r": trend_r_mean,
        "ma_r": ma_r_mean,
        "acf_corrupted": {str(k): v for k, v in acf_corrupted.items()},
        "acf_clean": {str(k): v for k, v in acf_clean_ref.items()},
        "floors": cache["floors"],
    }


def evaluate_geometry(
    *,
    geometry: str,
    dataset: str,
    n_samples: int,
    seed: int,
    stage: str,
    fractions: Sequence[float],
    acf_lags: Sequence[int],
    length_mode: str,
    g_cal: float,
    scale_cal: float,
    g_override: Optional[float],
    scale_override: Optional[float],
    ma_window: str,
    cache: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if cache is None:
        cache = _prepare_geometry_cache(
            geometry=geometry,
            dataset=dataset,
            n_samples=n_samples,
            seed=seed,
            stage=stage,
            acf_lags=acf_lags,
            ma_window=ma_window,
        )
    return _eval_from_cache(
        cache,
        fractions=fractions,
        acf_lags=acf_lags,
        length_mode=length_mode,
        g_cal=g_cal,
        scale_cal=scale_cal,
        g_override=g_override,
        scale_override=scale_override,
    )


def plot_curves(
    results: Dict[str, Dict[str, Any]],
    *,
    dataset: str,
    out_dir: Path,
    jpeg_dpi: int,
    primary_acf_lag: int = 1,
) -> Tuple[Path, Path, Path]:
    # Per-geometry floors (do NOT average — long-map Bern(0.5) floor can be >> short-map)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for geometry, res in results.items():
        fr = np.asarray(res["fractions"], dtype=np.float64)
        label = f"{geometry} (W={res['map_shape'][-1]}, mode={res.get('length_mode', 'none')})"
        ax.plot(fr, res["trend_r2"], marker="o", label=label)
        floor = float(res["floors"]["trend_r2_floor"])
        thr = _crossing_threshold(floor)
        ax.axhline(floor, linestyle="--", alpha=0.55, label=f"{geometry} floor R²={floor:.3f}")
        ax.axhline(thr, linestyle=":", alpha=0.35, label=f"{geometry} thr={thr:.3f}")
    ax.set_xlabel("t / (T-1)")
    ax.set_ylabel("trend R² (clean fit vs corrupted fit)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"{dataset}: linear-trend recoverability vs bit-flip t")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    p1 = out_dir / f"trend_r2_{dataset}.jpg"
    save_figure_jpg(fig, str(p1), dpi=jpeg_dpi)
    plt.close(fig)

    fig_ma, ax_ma = plt.subplots(figsize=(8, 4.5))
    for geometry, res in results.items():
        fr = np.asarray(res["fractions"], dtype=np.float64)
        label = f"{geometry} (W={res['map_shape'][-1]}, mode={res.get('length_mode', 'none')})"
        ax_ma.plot(fr, res["ma_r"], marker="o", label=label)
        floor_ma = float(res["floors"]["ma_r_floor"])
        thr_ma = _crossing_threshold(floor_ma, absolute_cap=0.35)
        ax_ma.axhline(
            floor_ma, linestyle="--", alpha=0.55, label=f"{geometry} floor |r|={floor_ma:.3f}"
        )
        ax_ma.axhline(thr_ma, linestyle=":", alpha=0.35, label=f"{geometry} thr={thr_ma:.3f}")
    ax_ma.set_xlabel("t / (T-1)")
    ax_ma.set_ylabel("Pearson r (MA clean vs MA corrupted)")
    ax_ma.set_ylim(-0.05, 1.05)
    ax_ma.set_title(f"{dataset}: moving-average recoverability vs bit-flip t")
    ax_ma.grid(True, alpha=0.3)
    ax_ma.legend(fontsize=7)
    fig_ma.tight_layout()
    p_ma = out_dir / f"ma_r_{dataset}.jpg"
    save_figure_jpg(fig_ma, str(p_ma), dpi=jpeg_dpi)
    plt.close(fig_ma)

    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    for geometry, res in results.items():
        fr = np.asarray(res["fractions"], dtype=np.float64)
        series = res["acf_corrupted"][str(primary_acf_lag)]
        ax2.plot(fr, series, marker="o", label=f"{geometry} ACF(lag={primary_acf_lag})")
        floor2 = float(res["floors"][f"acf_lag{primary_acf_lag}_floor"])
        thr2 = _crossing_threshold(floor2, absolute_cap=0.35)
        ax2.axhline(
            res["acf_clean"][str(primary_acf_lag)],
            linestyle="--",
            alpha=0.35,
            label=f"{geometry} clean ACF",
        )
        ax2.axhline(floor2, linestyle="--", alpha=0.55, label=f"{geometry} |ACF| floor={floor2:.3f}")
        ax2.axhline(thr2, linestyle=":", alpha=0.35, label=f"{geometry} thr={thr2:.3f}")
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
    print("  note: ma_r = Pearson(MA_clean, MA_corrupted); acf_lag* = ACF of corrupted series")
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
            f"(thr trend={tr_thr:.3f}, ma={ma_thr:.3f}, acf={ac_thr:.3f}; "
            f"floors trend={res['floors']['trend_r2_floor']:.3f}, "
            f"ma={res['floors']['ma_r_floor']:.3f})"
        )
        # Endpoint check: did MA/ACF actually reach near floor by t=T?
        ma_end = float(res["ma_r"][-1])
        ac_end = float(res["acf_corrupted"][str(primary_acf_lag)][-1])
        print(
            f"    endpoint frac=1.0: ma_r={ma_end:.3f} (floor={res['floors']['ma_r_floor']:.3f})  "
            f"acf={ac_end:.3f} (floor={res['floors'][f'acf_lag{primary_acf_lag}_floor']:.3f})  "
            f"mode={res.get('length_mode')} g={res.get('length_g')} scale={res.get('length_scale')}"
        )
        rows.append({
            "geometry": geometry,
            "subset_id": res["subset_id"],
            "length_mode": res.get("length_mode", "none"),
            "length_g": res.get("length_g", 1.0),
            "length_scale": res.get("length_scale", 1.0),
            "trend_r2_floor": res["floors"]["trend_r2_floor"],
            "trend_cross_frac": tr_cross if tr_cross is not None else "",
            "ma_r_floor": res["floors"]["ma_r_floor"],
            "ma_cross_frac": ma_cross if ma_cross is not None else "",
            "ma_endpoint": ma_end,
            "acf_lag": primary_acf_lag,
            "acf_floor": res["floors"][f"acf_lag{primary_acf_lag}_floor"],
            "acf_cross_frac": ac_cross if ac_cross is not None else "",
            "acf_endpoint": ac_end,
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
            if abs(ca - cb) <= 0.1:
                print("[flag] MA recoverability crosses at similar %T — length remap looks ok.")
            else:
                later = a if ca > cb else b
                print(
                    f"[flag] {later} retains recoverable MA structure longer "
                    f"({100 * max(ca, cb):.1f}% vs {100 * min(ca, cb):.1f}% of T)."
                )
        else:
            print("[flag] at least one geometry never crossed MA threshold within the grid.")
    return rows


def _crossings_from_metrics_rows(
    rows: List[Dict[str, str]],
    *,
    primary_acf_lag: int,
) -> List[Dict[str, Any]]:
    """Recompute crossings from an existing metrics_*.csv (no re-corruption)."""
    by_geo: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        by_geo.setdefault(r["geometry"], []).append(r)
    out = []
    for geometry, grows in by_geo.items():
        grows = sorted(grows, key=lambda r: float(r["frac"]))
        fr = [float(r["frac"]) for r in grows]
        floors = {
            "trend_r2_floor": float(grows[0]["floor_trend_r2_floor"]),
            "ma_r_floor": float(grows[0]["floor_ma_r_floor"]),
            f"acf_lag{primary_acf_lag}_floor": float(
                grows[0][f"floor_acf_lag{primary_acf_lag}_floor"]
            ),
        }
        tr_thr = _crossing_threshold(floors["trend_r2_floor"])
        ma_thr = _crossing_threshold(floors["ma_r_floor"], absolute_cap=0.35)
        ac_thr = _crossing_threshold(floors[f"acf_lag{primary_acf_lag}_floor"], absolute_cap=0.35)
        tr_cross = _crossing_pct(fr, [float(r["trend_r2"]) for r in grows], tr_thr)
        ma_cross = _crossing_pct(fr, [float(r["ma_r"]) for r in grows], ma_thr)
        ac_cross = _crossing_pct(fr, [float(r[f"acf_lag{primary_acf_lag}"]) for r in grows], ac_thr)
        out.append({
            "geometry": geometry,
            "subset_id": grows[0].get("subset_id", ""),
            "trend_r2_floor": floors["trend_r2_floor"],
            "trend_cross_frac": tr_cross if tr_cross is not None else "",
            "ma_r_floor": floors["ma_r_floor"],
            "ma_cross_frac": ma_cross if ma_cross is not None else "",
            "acf_lag": primary_acf_lag,
            "acf_floor": floors[f"acf_lag{primary_acf_lag}_floor"],
            "acf_cross_frac": ac_cross if ac_cross is not None else "",
            "map_W": grows[0].get("map_W", ""),
            "note": (
                "floors from CSV as-is; if ma_r_floor≈1 on long maps, "
                "re-run eval (noise-vs-noise floor bug)"
            ),
        })
        print(
            f"  {geometry}: ma_floor={floors['ma_r_floor']:.4f} thr={ma_thr:.4f} "
            f"ma_cross={ma_cross}  "
            f"curve ma_r={[round(float(r['ma_r']), 3) for r in grows]}"
        )
    return out


def recompute_crossings_from_dir(root: Path, *, primary_acf_lag: int) -> None:
    root = root.resolve()
    print(f"[recompute] scanning {root}")
    for metrics_path in sorted(root.rglob("metrics_*.csv")):
        ds_dir = metrics_path.parent
        dataset = metrics_path.stem.replace("metrics_", "", 1)
        rows = list(csv.DictReader(metrics_path.open(encoding="utf-8")))
        print(f"\n=== {dataset} ({metrics_path}) ===")
        # Clarify common misread: acf_lag1 at frac=0.5 is NOT ma_r
        for geo in sorted({r["geometry"] for r in rows}):
            g = [r for r in rows if r["geometry"] == geo]
            r05 = next((r for r in g if abs(float(r["frac"]) - 0.5) < 1e-9), None)
            if r05 is not None:
                print(
                    f"  [readout] {geo} frac=0.5: ma_r={float(r05['ma_r']):.4f}  "
                    f"acf_lag1={float(r05['acf_lag1']):.4f}  "
                    f"(ma_cross uses ma_r column, not acf)"
                )
        cross_rows = _crossings_from_metrics_rows(rows, primary_acf_lag=primary_acf_lag)
        out = ds_dir / f"crossings_{dataset}_recomputed.csv"
        with out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(cross_rows[0].keys()))
            w.writeheader()
            w.writerows(cross_rows)
        print(f"[csv] {out}")


def compare_old_new_crossings(
    *,
    old_root: Optional[Path],
    new_rows: List[Dict[str, Any]],
    dataset: str,
) -> None:
    if old_root is None:
        return
    old_path = old_root / dataset / f"crossings_{dataset}.csv"
    if not old_path.is_file():
        print(f"[compare] no old crossings at {old_path}")
        return
    old = {r["geometry"]: r for r in csv.DictReader(old_path.open(encoding="utf-8"))}
    print(f"\n=== OLD vs NEW crossings ({dataset}) ===")
    for row in new_rows:
        geo = row["geometry"]
        o = old.get(geo, {})

        def _f(x):
            if x is None or x == "":
                return "never"
            return f"{float(x):.3f}"

        print(
            f"  {geo}:\n"
            f"    trend  old={_f(o.get('trend_cross_frac'))}  new={_f(row.get('trend_cross_frac'))}\n"
            f"    ma_r   old={_f(o.get('ma_cross_frac'))}  new={_f(row.get('ma_cross_frac'))}  "
            f"endpoint={row.get('ma_endpoint')}\n"
            f"    acf    old={_f(o.get('acf_cross_frac'))}  new={_f(row.get('acf_cross_frac'))}  "
            f"endpoint={row.get('acf_endpoint')}"
        )


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
    length_mode: str,
    g_cal: float,
    scale_cal: float,
    g_override: Optional[float],
    scale_override: Optional[float],
    ma_window: str,
    compare_old_dir: Optional[Path],
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
            length_mode=length_mode,
            g_cal=g_cal,
            scale_cal=scale_cal,
            g_override=g_override,
            scale_override=scale_override,
            ma_window=ma_window,
        )
        res = results[geometry]
        print(
            f"  map={res['map_shape']} β_grid={[round(b, 3) for b in res['betas']]} "
            f"trend_r2={[round(v, 3) for v in res['trend_r2']]} "
            f"ma_r={[round(v, 3) for v in res['ma_r']]}",
            flush=True,
        )

    meta = {
        "length_mode": length_mode,
        "g_cal": g_cal,
        "scale_cal": scale_cal,
        "g_override": g_override,
        "scale_override": scale_override,
        "ma_window": ma_window,
        "l_ref": L_REF,
        "l_cal": L_CAL,
        "per_geometry": {
            g: {
                "length_mode": results[g]["length_mode"],
                "length_g": results[g]["length_g"],
                "length_scale": results[g]["length_scale"],
                "ma_width": results[g]["ma_width"],
                "betas": results[g]["betas"],
                "floors": results[g]["floors"],
            }
            for g in results
        },
    }
    (out_dir / "schedule_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

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
    compare_old_new_crossings(old_root=compare_old_dir, new_rows=cross_rows, dataset=dataset)

    csv_path = out_dir / f"metrics_{dataset}.csv"
    rows = []
    for geometry, res in results.items():
        for i, f in enumerate(res["fractions"]):
            row = {
                "dataset": dataset,
                "geometry": geometry,
                "subset_id": res["subset_id"],
                "length_mode": res["length_mode"],
                "length_g": res["length_g"],
                "length_scale": res["length_scale"],
                "ma_window": res["ma_window"],
                "ma_width": res["ma_width"],
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
    p.add_argument(
        "--length-mode",
        default="none",
        choices=("none", "power", "scale"),
        help="none=reference linear; power=β∝u^(1/g); scale=clip(β*scale, 0.5)",
    )
    p.add_argument(
        "--g-cal",
        type=float,
        default=1.5,
        help="g(L_cal) for power mode (g>1 front-loads high β). Calibrate on 336/720.",
    )
    p.add_argument(
        "--scale-cal",
        type=float,
        default=1.5,
        help="scale(L_cal) for scale mode (scale>1 hits β=0.5 earlier).",
    )
    p.add_argument("--g-override", type=float, default=None, help="Force g for all remapped geos")
    p.add_argument("--scale-override", type=float, default=None, help="Force scale for remapped geos")
    p.add_argument(
        "--ma-window",
        default="fixed_ref",
        choices=("frac", "fixed_ref"),
        help="frac=0.08*T (old, length-biased); fixed_ref=same width as 96/96 (length-fair)",
    )
    p.add_argument(
        "--compare-old-dir",
        type=Path,
        default=None,
        help="Prior report dir (e.g. reports/noise_trend_recoverability_4146642) for OLD vs NEW printout",
    )
    p.add_argument(
        "--recompute-crossings-from",
        type=Path,
        default=None,
        help="Only recompute crossings_*.csv from existing metrics_*.csv under this dir",
    )
    p.add_argument(
        "--calibrate-g-grid",
        default="",
        help="Comma list of g values to try on 336/720 only (prints schedule β grids; no full eval)",
    )
    return p.parse_args(argv)


def _print_g_grid(g_values: Sequence[float]) -> None:
    from models.diffusion_tsf.diffusion import _build_transition_schedule

    print("=== β schedule grid at frac points (linear base, β_end=0.5) ===")
    fracs = DEFAULT_FRACTIONS
    T = 1000
    t_idxs = [int(round(f * (T - 1))) for f in fracs]
    for g in g_values:
        betas = _build_transition_schedule(
            T, 1e-5, 0.5, "linear", "cpu", length_mode="power", length_g=g
        )
        vals = [float(betas[t]) for t in t_idxs]
        print(f"  g={g:.3f}: β={[round(v, 3) for v in vals]}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.recompute_crossings_from is not None:
        recompute_crossings_from_dir(
            args.recompute_crossings_from, primary_acf_lag=int(args.primary_acf_lag)
        )
        return
    if args.calibrate_g_grid.strip():
        gs = [float(x) for x in args.calibrate_g_grid.split(",") if x.strip()]
        _print_g_grid(gs)
        return

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    geometries = [g.strip() for g in args.geometries.split(",") if g.strip()]
    for g in geometries:
        if g not in GEOMETRY_CONFIGS:
            raise ValueError(f"unknown geometry {g!r}")
    acf_lags = tuple(int(x) for x in args.acf_lags.split(",") if x.strip())
    out_root = args.output_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    compare_old = args.compare_old_dir.resolve() if args.compare_old_dir else None

    print(
        f"[cfg] length_mode={args.length_mode} g_cal={args.g_cal} scale_cal={args.scale_cal} "
        f"ma_window={args.ma_window} g_override={args.g_override} "
        f"scale_override={args.scale_override}",
        flush=True,
    )
    if args.length_mode == "power":
        print(
            f"[cfg] g(W={int(L_REF)})={length_schedule_g(int(L_REF), g_cal=args.g_cal):.4f}  "
            f"g(W={int(L_CAL)})={length_schedule_g(int(L_CAL), g_cal=args.g_cal):.4f}",
            flush=True,
        )
    elif args.length_mode == "scale":
        print(
            f"[cfg] scale(W={int(L_REF)})="
            f"{length_schedule_scale(int(L_REF), scale_cal=args.scale_cal):.4f}  "
            f"scale(W={int(L_CAL)})="
            f"{length_schedule_scale(int(L_CAL), scale_cal=args.scale_cal):.4f}",
            flush=True,
        )

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
            length_mode=str(args.length_mode),
            g_cal=float(args.g_cal),
            scale_cal=float(args.scale_cal),
            g_override=args.g_override,
            scale_override=args.scale_override,
            ma_window=str(args.ma_window),
            compare_old_dir=compare_old,
        )
    print(f"[done] {out_root}", flush=True)


if __name__ == "__main__":
    main()
