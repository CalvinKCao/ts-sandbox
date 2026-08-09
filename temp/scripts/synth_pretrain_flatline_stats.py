#!/usr/bin/env python3
"""Flatline rate in shared RealTS synth pretrain pool (canvas128 / ETTh geometry).

Corpus: ``synth_data/synth_pool_v{V}_L{L}.npy`` — shared by n_variates, not
dataset-specific. ETTh1 and ETTh2 both use V=7 → same pool
``synth_pool_v7_L192.npy`` (L = lookback 96 + forecast 96).

Encode (matches staged synth pretrain window-norm + coarse ladder):
  past = seq[:, :lb], future = seq[:, lb-K :]
  z = (future - mean(past)) / max(std(past, unbiased), std_floor)
  coarse bin = floor(clip(z, ±max_scale) mapped onto [0, H_c))

Flat rule (same as etth2_coarse_flat_run_stats.py):
  run = ≥ min_run consecutive identical coarse bins on the future window
  flat iff max(z)-min(z) over the run ≤ flat_eps_frac × (2·max_scale/H_c)

Examples:
  source .venv/bin/activate
  python temp/scripts/synth_pretrain_flatline_stats.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_POOL = REPO_ROOT / "synth_data" / "synth_pool_v7_L192.npy"
DEFAULT_OUT = REPO_ROOT / "temp" / "lean_disc_c128_results" / "synth_flatline_stats"
DEFAULT_CFG = "configs/binary_window_norm_patch_refine_canvas128_p64x6.yaml"
FLAT_EPS_FRAC = 0.25
MIN_RUN = 3
# Pool geometry for V7 L192 (base binary_staged lb/hz; RealTS total_length).
POOL_LOOKBACK = 96
POOL_FORECAST = 96
POOL_OVERLAP = 8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--config", type=str, default=DEFAULT_CFG)
    p.add_argument("--dataset", type=str, default="ETTh1", help="Only for max_scale / WN knobs")
    p.add_argument("--max-scale", type=float, default=None)
    p.add_argument("--coarse-h", type=int, default=None)
    p.add_argument("--std-floor", type=float, default=None)
    p.add_argument("--lookback", type=int, default=POOL_LOOKBACK)
    p.add_argument("--forecast", type=int, default=POOL_FORECAST)
    p.add_argument("--overlap", type=int, default=POOL_OVERLAP)
    p.add_argument("--flat-eps-frac", type=float, default=FLAT_EPS_FRAC)
    p.add_argument("--min-run", type=int, default=MIN_RUN)
    p.add_argument("--max-samples", type=int, default=None, help="Cap pool rows (debug)")
    args = p.parse_args()
    args.pool = args.pool.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    return args


def _find_runs(bins_1d: np.ndarray, min_run: int) -> List[Tuple[int, int, int]]:
    n = int(bins_1d.shape[0])
    out: List[Tuple[int, int, int]] = []
    i = 0
    while i < n:
        j = i + 1
        while j < n and int(bins_1d[j]) == int(bins_1d[i]):
            j += 1
        if j - i >= min_run:
            out.append((i, j, int(bins_1d[i])))
        i = j
    return out


def _resolve_lattice(args: argparse.Namespace) -> Dict[str, Any]:
    from models.diffusion_tsf.pipeline.config import load_experiment_config
    from models.diffusion_tsf.pipeline.state import PipelineState

    cfg = load_experiment_config(args.config, cli_overrides={"dataset": args.dataset})
    state = PipelineState.from_config(cfg)
    max_scale = float(
        args.max_scale
        if args.max_scale is not None
        else state.max_scale_by_dataset.get(args.dataset, state.max_scale)
    )
    coarse_h = int(args.coarse_h or state.coarse_image_height)
    std_floor = float(
        args.std_floor if args.std_floor is not None else state.window_norm_std_floor
    )
    if max_scale <= 0 or coarse_h <= 0 or std_floor <= 0:
        raise RuntimeError(
            f"bad lattice: max_scale={max_scale} coarse_h={coarse_h} std_floor={std_floor}"
        )
    if not bool(state.use_window_normalization):
        raise RuntimeError(f"{args.config}: expected use_window_normalization=True")
    if str(state.window_norm_center) != "mean":
        raise RuntimeError(f"expected window_norm_center=mean, got {state.window_norm_center!r}")
    if bool(state.use_ordinal_window_norm):
        raise RuntimeError(f"{args.config}: expected use_ordinal_window_norm=False for synth encode")
    return {
        "max_scale": max_scale,
        "coarse_h": coarse_h,
        "std_floor": std_floor,
        "window_norm_center": str(state.window_norm_center),
        "use_window_normalization": True,
        "patch_refine_canvas_height": int(state.patch_refine_canvas_height),
        "state_lookback": int(state.lookback_length),
        "state_forecast": int(state.forecast_length),
        "state_overlap": int(state.lookback_overlap),
    }


def _encode_futures(
    pool: np.ndarray,
    *,
    lookback: int,
    forecast: int,
    overlap: int,
    max_scale: float,
    coarse_h: int,
    std_floor: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (bins, z) for future windows: shapes (N, V, Tf)."""
    if pool.ndim != 3:
        raise ValueError(f"expected pool (N,V,L), got {pool.shape}")
    n, v, L = pool.shape
    need = lookback + forecast
    if L != need:
        raise RuntimeError(
            f"pool L={L} != lookback+forecast={need} "
            f"(lb={lookback} hz={forecast}); refuse silent truncate"
        )
    if overlap < 0 or overlap >= lookback:
        raise RuntimeError(f"bad overlap={overlap} for lookback={lookback}")

    past = pool[:, :, :lookback].astype(np.float64, copy=False)
    future = pool[:, :, lookback - overlap :].astype(np.float64, copy=False)
    # torch.std default = unbiased (ddof=1)
    center = past.mean(axis=-1, keepdims=True)
    if lookback > 1:
        std = past.std(axis=-1, keepdims=True, ddof=1)
    else:
        std = past.std(axis=-1, keepdims=True, ddof=0)
    std = np.maximum(std, std_floor)
    z = (future - center) / std
    z_clip = np.clip(z, -max_scale, max_scale)
    pos = (z_clip + max_scale) / (2.0 * max_scale) * coarse_h
    bins = np.floor(pos).astype(np.int64)
    bins = np.clip(bins, 0, coarse_h - 1)
    return bins, z


def _analyze(
    bins: np.ndarray,
    z: np.ndarray,
    *,
    flat_eps: float,
    min_run: int,
    t_slice: slice,
    label: str,
) -> Dict[str, Any]:
    """bins/z: (N, V, Tf). Analyze each (sample, variate) series over t_slice."""
    n, n_vars, _tf = bins.shape
    per_var: List[Dict[str, Any]] = []
    overall_runs = 0
    overall_flat = 0
    win_with_run = 0
    win_with_flat = 0
    win_with_wobbly_only = 0
    total_windows = n * n_vars

    for v in range(n_vars):
        n_runs = 0
        n_flat = 0
        lengths: List[int] = []
        z_ranges_flat: List[float] = []
        z_ranges_wobbly: List[float] = []
        v_win_run = 0
        v_win_flat = 0
        for i in range(n):
            bins_v = bins[i, v, t_slice]
            z_v = z[i, v, t_slice]
            runs = _find_runs(bins_v, min_run)
            if runs:
                v_win_run += 1
                win_with_run += 1
            any_flat = False
            any_wobbly = False
            for a, b, _bid in runs:
                seg = z_v[a:b]
                z_range = float(seg.max() - seg.min())
                lengths.append(b - a)
                n_runs += 1
                if z_range <= flat_eps:
                    n_flat += 1
                    any_flat = True
                    z_ranges_flat.append(z_range)
                else:
                    any_wobbly = True
                    z_ranges_wobbly.append(z_range)
            if any_flat:
                v_win_flat += 1
                win_with_flat += 1
            elif any_wobbly:
                win_with_wobbly_only += 1
        n_wobbly = n_runs - n_flat
        overall_runs += n_runs
        overall_flat += n_flat
        per_var.append(
            {
                "variate": v,
                "n_windows": n,
                "n_runs_ge3": n_runs,
                "n_flat": n_flat,
                "n_wobbly": n_wobbly,
                "pct_flat_of_runs": (100.0 * n_flat / n_runs) if n_runs else float("nan"),
                "pct_wobbly_of_runs": (100.0 * n_wobbly / n_runs) if n_runs else float("nan"),
                "windows_with_ge3_run": v_win_run,
                "windows_with_flat_run": v_win_flat,
                "pct_windows_with_flat": 100.0 * v_win_flat / n if n else float("nan"),
                "mean_run_len": float(np.mean(lengths)) if lengths else float("nan"),
                "max_run_len": int(max(lengths)) if lengths else 0,
                "mean_z_range_flat": float(np.mean(z_ranges_flat)) if z_ranges_flat else float("nan"),
                "mean_z_range_wobbly": float(np.mean(z_ranges_wobbly))
                if z_ranges_wobbly
                else float("nan"),
            }
        )

    n_wobbly_all = overall_runs - overall_flat
    return {
        "region": label,
        "n_samples": n,
        "n_variates": n_vars,
        "analyzed_T_per_window": int(bins[..., t_slice].shape[-1]),
        "total_windows": total_windows,
        "overall": {
            "n_runs_ge3": overall_runs,
            "n_flat": overall_flat,
            "n_wobbly": n_wobbly_all,
            "pct_flat_of_runs": (100.0 * overall_flat / overall_runs)
            if overall_runs
            else float("nan"),
            "pct_wobbly_of_runs": (100.0 * n_wobbly_all / overall_runs)
            if overall_runs
            else float("nan"),
            "windows_with_ge3_run": win_with_run,
            "windows_with_flat_run": win_with_flat,
            "windows_with_wobbly_only": win_with_wobbly_only,
            "pct_windows_with_flat": 100.0 * win_with_flat / total_windows
            if total_windows
            else float("nan"),
            "pct_windows_with_ge3_run": 100.0 * win_with_run / total_windows
            if total_windows
            else float("nan"),
        },
        "per_variate": per_var,
    }


def _fmt_pct(x: float) -> str:
    if x != x:
        return "n/a"
    return f"{x:.1f}%"


def _markdown(payload: Dict[str, Any]) -> str:
    meta = payload["meta"]
    lines: List[str] = [
        "# Synth pretrain flatline stats (RealTS pool)",
        "",
        f"- pool: `{meta['pool']}` shape `{meta['pool_shape']}`",
        f"- shared corpus: **not dataset-specific** — keyed by "
        f"`n_variates={meta['n_variates']}` + `L={meta['pool_L']}` "
        f"(ETTh1 and ETTh2 both use this V=7 pool)",
        f"- leaf knobs from: `{meta['config']}` dataset=`{meta['dataset']}` "
        f"(max_scale / coarse_h / std_floor only; lookback for encode follows pool)",
        f"- pool geometry: lookback={meta['lookback']}, forecast={meta['forecast']}, "
        f"overlap={meta['overlap']} → future T={meta['future_T']}",
        f"- note: canvas128 finetune leaf uses lookback={meta['state_lookback']}; "
        f"existing disk pool is L={meta['pool_L']} (=96+96). Stats below are for "
        f"the **actual synth pool on disk** (what RealTS serves when total_length matches).",
        f"- encode: window-norm (past mean / torch-unbiased std, "
        f"std_floor={meta['std_floor']}) → coarse H={meta['coarse_h']}, "
        f"max_scale={meta['max_scale']}",
        f"- coarse bin width: `2·max_scale/H = {meta['coarse_bin_width']:.6g}`",
        f"- flat rule: `max(z)−min(z) ≤ {meta['flat_eps_frac']} × bin_width` "
        f"= `{meta['flat_eps_abs']:.6g}`",
        f"- min run: ≥{meta['min_run']} identical coarse bins",
        "",
    ]
    for region in payload["regions"]:
        o = region["overall"]
        lines.extend(
            [
                f"## Region: {region['region']} (T={region['analyzed_T_per_window']})",
                "",
                f"- samples × variates = {region['n_samples']} × {region['n_variates']} "
                f"= {region['total_windows']} windows",
                f"- windows with ≥1 run (≥{meta['min_run']} same bin): "
                f"{o['windows_with_ge3_run']} ({_fmt_pct(o['pct_windows_with_ge3_run'])})",
                f"- windows with ≥1 **true flat** run: "
                f"{o['windows_with_flat_run']} ({_fmt_pct(o['pct_windows_with_flat'])})",
                f"- windows with ≥1 run but only wobbly: {o['windows_with_wobbly_only']}",
                "",
                "| metric | value |",
                "|---|---:|",
                f"| runs (≥{meta['min_run']} same bin) | {o['n_runs_ge3']} |",
                f"| flat runs | {o['n_flat']} |",
                f"| wobbly runs | {o['n_wobbly']} |",
                f"| % flat of runs | {_fmt_pct(o['pct_flat_of_runs'])} |",
                f"| % wobbly of runs | {_fmt_pct(o['pct_wobbly_of_runs'])} |",
                "",
                "### Per variate",
                "",
                "| variate | runs | flat | wobbly | % flat | % win w/ flat | mean len | max len |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in region["per_variate"]:
            lines.append(
                f"| {row['variate']} | {row['n_runs_ge3']} | {row['n_flat']} | "
                f"{row['n_wobbly']} | {_fmt_pct(row['pct_flat_of_runs'])} | "
                f"{_fmt_pct(row['pct_windows_with_flat'])} | "
                f"{row['mean_run_len']:.2f} | {row['max_run_len']} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    if not args.pool.is_file():
        raise FileNotFoundError(args.pool)

    lattice = _resolve_lattice(args)
    max_scale = float(lattice["max_scale"])
    coarse_h = int(lattice["coarse_h"])
    std_floor = float(lattice["std_floor"])
    bin_width = 2.0 * max_scale / float(coarse_h)
    flat_eps = float(args.flat_eps_frac) * bin_width
    if flat_eps <= 0:
        raise RuntimeError(f"flat_eps must be > 0, got {flat_eps}")

    pool = np.load(args.pool, mmap_mode="r")
    if args.max_samples is not None:
        pool = np.array(pool[: int(args.max_samples)])
    else:
        pool = np.array(pool)  # materialize once for vectorized encode
    n, n_vars, L = pool.shape
    print(
        f"[synth_flat] pool={args.pool.name} shape=({n},{n_vars},{L}) "
        f"lb={args.lookback} hz={args.forecast} K={args.overlap} "
        f"max_scale={max_scale} Hc={coarse_h} bin_w={bin_width:.6g} "
        f"flat_eps={flat_eps:.6g}",
        flush=True,
    )

    bins, z = _encode_futures(
        pool,
        lookback=int(args.lookback),
        forecast=int(args.forecast),
        overlap=int(args.overlap),
        max_scale=max_scale,
        coarse_h=coarse_h,
        std_floor=std_floor,
    )
    tf = bins.shape[-1]
    # Full future incl. overlap (what RealTS returns / model trains on).
    full = _analyze(
        bins, z, flat_eps=flat_eps, min_run=int(args.min_run), t_slice=slice(None), label="full_future"
    )
    # Exclusive forecast after overlap prefix.
    excl = _analyze(
        bins,
        z,
        flat_eps=flat_eps,
        min_run=int(args.min_run),
        t_slice=slice(int(args.overlap), None),
        label="forecast_exclusive",
    )

    payload = {
        "meta": {
            "pool": str(args.pool),
            "pool_shape": [n, n_vars, L],
            "pool_L": L,
            "n_variates": n_vars,
            "config": args.config,
            "dataset": args.dataset,
            "lookback": int(args.lookback),
            "forecast": int(args.forecast),
            "overlap": int(args.overlap),
            "future_T": tf,
            "state_lookback": lattice["state_lookback"],
            "state_forecast": lattice["state_forecast"],
            "state_overlap": lattice["state_overlap"],
            "max_scale": max_scale,
            "coarse_h": coarse_h,
            "std_floor": std_floor,
            "coarse_bin_width": bin_width,
            "flat_eps_frac": float(args.flat_eps_frac),
            "flat_eps_abs": flat_eps,
            "min_run": int(args.min_run),
            "patch_refine_canvas_height": lattice["patch_refine_canvas_height"],
            "shared_note": (
                "RealTS pool is shared by (n_variates, L); ETTh1/ETTh2 both V=7 → same file"
            ),
        },
        "regions": [full, excl],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "stats.json"
    md_path = args.output_dir / "stats.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    md_path.write_text(_markdown(payload))
    print(f"[synth_flat] wrote {json_path}", flush=True)
    print(f"[synth_flat] wrote {md_path}", flush=True)
    o = full["overall"]
    print(
        f"[synth_flat] FULL future: runs={o['n_runs_ge3']} flat={o['n_flat']} "
        f"({o['pct_flat_of_runs']:.1f}%) | "
        f"windows_with_flat={o['windows_with_flat_run']}/"
        f"{full['total_windows']} ({o['pct_windows_with_flat']:.1f}%)",
        flush=True,
    )


if __name__ == "__main__":
    main()
