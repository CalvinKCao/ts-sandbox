#!/usr/bin/env python3
"""Boundary / lookback-overlap confound check for noise recoverability metrics.

Audit + ablation for the K=lookback_overlap prefix on the future canvas
(8 steps: past-copy used for inpainting consistency).

Step 1 (audit): corruption is applied uniformly to the full coarse map
(including the overlap prefix). Metrics in the original trend-recoverability
diag were also computed on the FULL decoded length (including overlap).

Step 2–3: re-run identity schedule (g=1 / length_mode=none) with metrics on
  - full sequence (boundary included)  — baseline
  - fill-in only (drop first K timesteps) — boundary excluded
and compare whether the 96/96 vs 336/720 MA/trend/ACF gap shrinks.

Corruption is already uniform; there is no boundary exemption to remove.
The "uniform corruption + full metrics" arm is therefore identical to the
existing baseline and is reported as such.

Example:
  python utils/diagnose_boundary_confound_noise_metrics.py \\
      --datasets exchange_rate,electricity --n-samples 24
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.diffusion_tsf.pipeline.visualize_utils import save_figure_jpg
from utils.diagnose_binary_noise_trend_recoverability import (
    DEFAULT_FRACTIONS,
    GEOMETRY_CONFIGS,
    LONG_GEO,
    REF_GEO,
    _crossing_or_nan,
    _crossing_threshold,
    _eval_from_cache,
    _noise_floor_metrics,
    _prepare_geometry_cache,
    _decode_coarse_1d,
    _acf_at_lags,
    _linear_trend,
    _moving_average,
    _pearson_r,
    _r2_between,
)

DEFAULT_DATASETS = "exchange_rate,electricity"


def _audit_boundary(cache: Dict[str, Any]) -> Dict[str, Any]:
    """Print/log how overlap interacts with corruption + metrics."""
    state = cache["state"]
    maps = cache["maps"]
    K = int(getattr(state, "lookback_overlap", 0) or 0)
    W = int(maps[0].shape[-1])
    # Corruption in the parent diag: _noise_at_t flips every bit of the map
    # with β_t — no mask / exemption for the first K columns.
    audit = {
        "geometry": cache["geometry"],
        "map_W": W,
        "lookback_overlap_K": K,
        "overlap_frac_of_W": float(K) / float(W) if W else 0.0,
        "corruption_applied_to_boundary": True,
        "corruption_note": (
            "BinaryDiffusionScheduler / _noise_at_t bit-flips the full (V,H,W) "
            "coarse map uniformly; no lookback_overlap mask."
        ),
        "metrics_default_region": "full_sequence",
        "metrics_note": (
            "Original diagnose_binary_noise_trend_recoverability.py computes "
            "trend/MA/ACF on the full decoded 1D length (includes overlap prefix)."
        ),
        "overlap_is_past_copy": True,
        "overlap_note": (
            "future_norm[..., :K] == past_norm[..., -K:] (exact); overlap is the "
            "inpainting-consistency past tail embedded in the future canvas."
        ),
    }
    print(
        f"  [audit] {cache['geometry']}: W={W} K={K} ({100 * audit['overlap_frac_of_W']:.1f}% of W)\n"
        f"          corruption applied to boundary: YES (uniform bit-flip)\n"
        f"          metrics computed over (default): FULL sequence\n"
        f"          overlap content: past-copy (exact)",
        flush=True,
    )
    return audit


def _slice_region(y: np.ndarray, k: int, region: str) -> np.ndarray:
    if region == "full":
        return y
    if region == "fill_only":
        if k <= 0:
            return y
        if y.ndim == 1:
            return y[k:]
        return y[..., k:]
    raise ValueError(f"unknown metric region {region!r}")


def _floors_for_region(
    cache: Dict[str, Any],
    *,
    region: str,
    acf_lags: Sequence[int],
    seed: int,
    n_draws: int,
) -> Dict[str, float]:
    """Recompute Bern(0.5) floors on the same metric region as the curves."""
    model = cache["model"]
    maps = cache["maps"]
    clean_1d = cache["clean_1d"]
    ma_window = cache["ma_window"]
    state = cache["state"]
    k = int(getattr(state, "lookback_overlap", 0) or 0)
    clean_sliced = [_slice_region(y, k, region) for y in clean_1d]
    # Floor draws: decode full random maps, then slice the same way.
    rng = np.random.default_rng(seed)
    map_shape = tuple(maps[0].shape)
    trend_r2s, ma_rs = [], []
    acf_vals: Dict[int, List[float]] = {int(l): [] for l in acf_lags}
    clean_trends = [_linear_trend(y) for y in clean_sliced]
    clean_mas = [_moving_average(y, ma_window=ma_window) for y in clean_sliced]
    n_clean = len(clean_sliced)
    for i in range(n_draws):
        noise = __import__("torch").from_numpy(
            rng.binomial(1, 0.5, size=map_shape).astype(np.float32)
        )
        yn_full = _decode_coarse_1d(model, noise)
        yn = _slice_region(yn_full, k, region)
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


def _eval_region(
    cache: Dict[str, Any],
    *,
    region: str,
    fractions: Sequence[float],
    acf_lags: Sequence[int],
    seed: int,
) -> Dict[str, Any]:
    """Identity schedule eval with metrics on full or fill_only region.

    Corruption always hits the full map (uniform). Only the metric window changes.
    """
    state = cache["state"]
    k = int(getattr(state, "lookback_overlap", 0) or 0)
    # Temporarily slice clean_1d in a shallow copy of cache for floor reuse path,
    # then run a custom eval loop mirroring _eval_from_cache.
    from utils.diagnose_binary_noise_trend_recoverability import (
        _noise_at_t,
        _scheduler_from_state,
        _ma_width,
    )

    model = cache["model"]
    maps = cache["maps"]
    clean_1d_full = cache["clean_1d"]
    ma_window = cache["ma_window"]
    n_vars = cache["n_vars"]

    clean_1d = [_slice_region(y, k, region) for y in clean_1d_full]
    clean_trends = [_linear_trend(y) for y in clean_1d]
    clean_acfs = [_acf_at_lags(y, acf_lags) for y in clean_1d]
    clean_mas = [_moving_average(y, ma_window=ma_window) for y in clean_1d]
    map_w = int(maps[0].shape[-1])
    ma_w = _ma_width(map_w if region == "full" else max(1, map_w - k), ma_window=ma_window)

    sched = _scheduler_from_state(state, length_mode="none", length_g=1.0, length_scale=1.0)
    T = sched.num_steps
    t_idxs = [int(round(f * (T - 1))) for f in fractions]
    trend_r2_mean, trend_r_mean, ma_r_mean = [], [], []
    acf_corrupted: Dict[int, List[float]] = {int(l): [] for l in acf_lags}
    acf_clean_ref = {
        int(l): float(np.mean([c[int(l)] for c in clean_acfs])) for l in acf_lags
    }

    for t_idx in t_idxs:
        r2s, rs, ma_rs = [], [], []
        acf_batch: Dict[int, List[float]] = {int(l): [] for l in acf_lags}
        for m, ct, cma in zip(maps, clean_trends, clean_mas):
            xt = m if t_idx == 0 else _noise_at_t(sched, m, t_idx)
            y_full = _decode_coarse_1d(model, xt)
            y = _slice_region(y_full, k, region)
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

    floors = _floors_for_region(
        cache,
        region=region,
        acf_lags=acf_lags,
        seed=seed + 17,
        n_draws=max(8, min(20, len(maps))),
    )
    metric_W = map_w if region == "full" else max(0, map_w - k)
    print(
        f"  [eval] {cache['geometry']} region={region} metric_W={metric_W} "
        f"(dropped K={k}) ma_end={ma_r_mean[-1]:.3f} floor={floors['ma_r_floor']:.3f}",
        flush=True,
    )
    return {
        "geometry": cache["geometry"],
        "dataset": cache["dataset"],
        "subset_id": str(state.subset_id),
        "n_variates": n_vars,
        "lookback": int(state.lookback_length),
        "horizon": int(state.forecast_length),
        "lookback_overlap": k,
        "metric_region": region,
        "map_shape": list(maps[0].shape),
        "metric_W": metric_W,
        "ma_window": ma_window,
        "ma_width": ma_w,
        "n_samples": len(maps),
        "schedule": sched.schedule_type,
        "length_mode": "none",
        "length_g": 1.0,
        "length_scale": 1.0,
        "num_steps": T,
        "fractions": list(fractions),
        "t_idxs": t_idxs,
        "betas": [float(sched.betas[t].item()) for t in t_idxs],
        "trend_r2": trend_r2_mean,
        "trend_r": trend_r_mean,
        "ma_r": ma_r_mean,
        "acf_corrupted": {str(kk): v for kk, v in acf_corrupted.items()},
        "acf_clean": {str(kk): v for kk, v in acf_clean_ref.items()},
        "floors": floors,
        "corruption_applied_to_boundary": True,
    }


def _plot_overlay(
    results_by_region: Dict[str, Dict[str, Dict[str, Any]]],
    *,
    dataset: str,
    out_dir: Path,
    jpeg_dpi: int,
    primary_acf_lag: int,
) -> Tuple[Path, Path, Path]:
    """Overlay full vs fill_only per geometry (solid=full, dashed=fill_only)."""
    styles = {"full": "-", "fill_only": "--"}
    markers = {"full": "o", "fill_only": "x"}

    def _one(metric_key: str, ylabel: str, title: str, fname: str, floor_key: str) -> Path:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        for region, geos in results_by_region.items():
            for geometry, res in geos.items():
                fr = np.asarray(res["fractions"], dtype=np.float64)
                if metric_key == "acf":
                    series = res["acf_corrupted"][str(primary_acf_lag)]
                else:
                    series = res[metric_key]
                ax.plot(
                    fr,
                    series,
                    linestyle=styles[region],
                    marker=markers[region],
                    label=f"{geometry} [{region}] W_m={res['metric_W']}",
                )
                floor = float(res["floors"][floor_key])
                ax.axhline(floor, linestyle=":", alpha=0.35)
        ax.set_xlabel("t / (T-1)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        path = out_dir / fname
        save_figure_jpg(fig, str(path), dpi=jpeg_dpi)
        plt.close(fig)
        return path

    p1 = _one(
        "trend_r2",
        "trend R²",
        f"{dataset}: trend R² — full vs fill_only (identity schedule)",
        f"trend_r2_boundary_ablation_{dataset}.jpg",
        "trend_r2_floor",
    )
    p2 = _one(
        "ma_r",
        "MA Pearson r",
        f"{dataset}: MA-r — full vs fill_only (identity schedule)",
        f"ma_r_boundary_ablation_{dataset}.jpg",
        "ma_r_floor",
    )
    p3 = _one(
        "acf",
        f"ACF lag={primary_acf_lag}",
        f"{dataset}: ACF — full vs fill_only (identity schedule)",
        f"acf_lag{primary_acf_lag}_boundary_ablation_{dataset}.jpg",
        f"acf_lag{primary_acf_lag}_floor",
    )
    return p1, p2, p3


def _summary_row(
    dataset: str,
    region: str,
    ref: Dict[str, Any],
    long: Dict[str, Any],
    *,
    primary_acf_lag: int,
) -> Dict[str, Any]:
    ma_end_long = float(long["ma_r"][-1])
    ma_floor_long = float(long["floors"]["ma_r_floor"])
    ma_end_ref = float(ref["ma_r"][-1])
    ma_floor_ref = float(ref["floors"]["ma_r_floor"])
    return {
        "dataset": dataset,
        "metric_region": region,
        "overlap_K": long["lookback_overlap"],
        "W_96": ref["metric_W"],
        "W_336": long["metric_W"],
        "overlap_frac_96": float(long["lookback_overlap"]) / float(ref["map_shape"][-1]),
        "overlap_frac_336": float(long["lookback_overlap"]) / float(long["map_shape"][-1]),
        "ma_end_96": ma_end_ref,
        "ma_floor_96": ma_floor_ref,
        "ma_end_336": ma_end_long,
        "ma_floor_336": ma_floor_long,
        "ma_gap_336_to_floor": abs(ma_end_long - ma_floor_long),
        "ma_end_diff_336_minus_96": ma_end_long - ma_end_ref,
        "trend_cross_96": _crossing_or_nan(ref, "trend_r2", primary_acf_lag=primary_acf_lag) or "",
        "trend_cross_336": _crossing_or_nan(long, "trend_r2", primary_acf_lag=primary_acf_lag) or "",
        "ma_cross_96": _crossing_or_nan(ref, "ma_r", primary_acf_lag=primary_acf_lag) or "",
        "ma_cross_336": _crossing_or_nan(long, "ma_r", primary_acf_lag=primary_acf_lag) or "",
        "acf_cross_96": _crossing_or_nan(ref, "acf", primary_acf_lag=primary_acf_lag) or "",
        "acf_cross_336": _crossing_or_nan(long, "acf", primary_acf_lag=primary_acf_lag) or "",
        "corruption_boundary": "uniform_yes",
    }


def run_dataset(
    *,
    dataset: str,
    n_samples: int,
    seed: int,
    stage: str,
    out_dir: Path,
    jpeg_dpi: int,
    acf_lags: Sequence[int],
    primary_acf_lag: int,
    ma_window: str,
) -> List[Dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"==== boundary ablation: {dataset} ====", flush=True)

    caches = {}
    audits = {}
    for geo in (REF_GEO, LONG_GEO):
        print(f"[cache] {dataset} {geo}", flush=True)
        caches[geo] = _prepare_geometry_cache(
            geometry=geo,
            dataset=dataset,
            n_samples=n_samples,
            seed=seed,
            stage=stage,
            acf_lags=acf_lags,
            ma_window=ma_window,
        )
        audits[geo] = _audit_boundary(caches[geo])

    (out_dir / "boundary_audit.json").write_text(
        json.dumps({"dataset": dataset, "per_geometry": audits}, indent=2),
        encoding="utf-8",
    )

    results_by_region: Dict[str, Dict[str, Dict[str, Any]]] = {}
    summary_rows: List[Dict[str, Any]] = []
    for region in ("full", "fill_only"):
        print(f"[region] {region}", flush=True)
        geos = {}
        for geo in (REF_GEO, LONG_GEO):
            geos[geo] = _eval_region(
                caches[geo],
                region=region,
                fractions=DEFAULT_FRACTIONS,
                acf_lags=acf_lags,
                seed=seed,
            )
        results_by_region[region] = geos
        summary_rows.append(
            _summary_row(
                dataset, region, geos[REF_GEO], geos[LONG_GEO], primary_acf_lag=primary_acf_lag
            )
        )

        # per-region metrics CSV
        rows = []
        for geometry, res in geos.items():
            for i, f in enumerate(res["fractions"]):
                row = {
                    "dataset": dataset,
                    "geometry": geometry,
                    "metric_region": region,
                    "lookback_overlap": res["lookback_overlap"],
                    "metric_W": res["metric_W"],
                    "frac": f,
                    "t": res["t_idxs"][i],
                    "beta": res["betas"][i],
                    "trend_r2": res["trend_r2"][i],
                    "ma_r": res["ma_r"][i],
                    f"acf_lag{primary_acf_lag}": res["acf_corrupted"][str(primary_acf_lag)][i],
                }
                row.update({f"floor_{k}": v for k, v in res["floors"].items()})
                rows.append(row)
        csv_path = out_dir / f"metrics_{region}_{dataset}.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[csv] {csv_path}")

    p1, p2, p3 = _plot_overlay(
        results_by_region,
        dataset=dataset,
        out_dir=out_dir,
        jpeg_dpi=jpeg_dpi,
        primary_acf_lag=primary_acf_lag,
    )
    print(f"[plot] {p1}\n[plot] {p2}\n[plot] {p3}")

    # Compare full vs fill_only gap shrinkage
    full = next(r for r in summary_rows if r["metric_region"] == "full")
    fill = next(r for r in summary_rows if r["metric_region"] == "fill_only")
    gap_full = float(full["ma_gap_336_to_floor"])
    gap_fill = float(fill["ma_gap_336_to_floor"])
    end_diff_full = float(full["ma_end_diff_336_minus_96"])
    end_diff_fill = float(fill["ma_end_diff_336_minus_96"])
    # Primary discrepancy signal: |MA_end_336 - MA_end_96| (both should be near their floors)
    disc_full = abs(end_diff_full)
    disc_fill = abs(end_diff_fill)
    shrink = None
    if disc_full > 1e-6:
        shrink = 1.0 - (disc_fill / disc_full)
    flag = "UNCHANGED"
    if shrink is not None and shrink > 0.5:
        flag = "GAP_SHRANK_>50%"
    elif shrink is not None and shrink > 0.2:
        flag = "GAP_SHRANK_PARTIAL"
    elif shrink is not None and shrink < -0.1:
        flag = "GAP_GREW"

    print("\n=== boundary confound verdict ===", flush=True)
    print(
        f"  {dataset}: MA_end 336−96  full={end_diff_full:+.3f}  fill_only={end_diff_fill:+.3f}  "
        f"shrink={shrink if shrink is not None else 'n/a'}",
        flush=True,
    )
    print(
        f"  {dataset}: |MA_end−floor|_336  full={gap_full:.4f}  fill_only={gap_fill:.4f}",
        flush=True,
    )
    print(
        f"  {dataset}: trend_cross 96/336  full={full['trend_cross_96']}/{full['trend_cross_336']}  "
        f"fill={fill['trend_cross_96']}/{fill['trend_cross_336']}",
        flush=True,
    )
    print(
        f"  {dataset}: acf_cross 96/336  full={full['acf_cross_96']}/{full['acf_cross_336']}  "
        f"fill={fill['acf_cross_96']}/{fill['acf_cross_336']}",
        flush=True,
    )
    print(f"  [flag] {flag}", flush=True)
    print(
        "  note: corruption already uniform on boundary — no separate "
        "'remove exemption' arm; full-region arm IS the uniform-corruption baseline.",
        flush=True,
    )

    for r in summary_rows:
        r["ma_end_diff_shrink_frac"] = shrink if shrink is not None else ""
        r["verdict_flag"] = flag

    return summary_rows


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", default=DEFAULT_DATASETS)
    p.add_argument("--n-samples", type=int, default=24)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--stage", default="coarse", choices=("coarse", "fine"))
    p.add_argument("--acf-lags", default="1,5,10")
    p.add_argument("--primary-acf-lag", type=int, default=1)
    p.add_argument("--ma-window", default="fixed_ref", choices=("frac", "fixed_ref"))
    p.add_argument("--jpeg-dpi", type=int, default=110)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "reports" / "noise_boundary_confound",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    acf_lags = tuple(int(x) for x in args.acf_lags.split(",") if x.strip())
    out_root = args.output_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print(
        "[cfg] identity schedule (length_mode=none); corruption=uniform full map; "
        "metrics=full vs fill_only (drop lookback_overlap prefix)",
        flush=True,
    )
    all_rows: List[Dict[str, Any]] = []
    for dataset in datasets:
        rows = run_dataset(
            dataset=dataset,
            n_samples=int(args.n_samples),
            seed=int(args.seed),
            stage=str(args.stage),
            out_dir=out_root / dataset,
            jpeg_dpi=int(args.jpeg_dpi),
            acf_lags=acf_lags,
            primary_acf_lag=int(args.primary_acf_lag),
            ma_window=str(args.ma_window),
        )
        all_rows.extend(rows)

    summary_path = out_root / "boundary_confound_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n[csv] {summary_path}")
    print("=== summary table ===")
    for r in all_rows:
        print(
            f"  {r['dataset']:16s} {r['metric_region']:10s}  "
            f"MA_end 96/336={float(r['ma_end_96']):.3f}/{float(r['ma_end_336']):.3f}  "
            f"floor336={float(r['ma_floor_336']):.3f}  "
            f"Δend={float(r['ma_end_diff_336_minus_96']):+.3f}  "
            f"tr={r['trend_cross_96']}/{r['trend_cross_336']}  "
            f"acf={r['acf_cross_96']}/{r['acf_cross_336']}  "
            f"flag={r['verdict_flag']}"
        )
    print(f"[done] {out_root}", flush=True)


if __name__ == "__main__":
    main()
