#!/usr/bin/env python3
"""ETTh2 canvas128: coarse same-bin run stats (flat vs wobbly-thresholded).

Uses the live binary canvas128 leaf encode path:
  configs/binary_window_norm_patch_refine_canvas128_p64x6_etth2.yaml
  ckpt 08-04-4601319-... (max_scale from coarse/patch_refine metadata)

Coarse GT = staged dual coarse ladder after past-mean window-norm
(H_coarse=16, NOT canvas128 HIR rows). Fail-fast if heights / max_scale unclear.

Per paper split × variate, build a **contiguous** coarse-bin series over the
split array (causal lookback ending at t−1, same affine as training window-norm
for the point at t), then count maximal runs of ≥3 identical coarse bins.

Flat rule (window-norm z at each t, same affine as that timestep's encode):
  max(z)−min(z) ≤ ε · coarse_bin_width
Default ε = 0.25. coarse_bin_width = 2·max_scale / H_coarse.

Examples:
  source .venv/bin/activate
  python temp/scripts/etth2_coarse_flat_run_stats.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.diffusion_tsf.pipeline.config import load_experiment_config  # noqa: E402
from models.diffusion_tsf.pipeline.globals_bridge import patch_globals  # noqa: E402
from models.diffusion_tsf.pipeline.phases.staged_diffusion_pretrain import (  # noqa: E402
    patch_stage_globals,
)
from models.diffusion_tsf.pipeline.state import PipelineState  # noqa: E402
from models.diffusion_tsf.train_multivariate_pipeline import load_dataset  # noqa: E402
import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod  # noqa: E402
from temp.scripts.eval_ablation_disc_l8_l16 import (  # noqa: E402
    _max_scale_from_ckpt_metadata,
)

DATASET = "ETTh2"
DEFAULT_CKPT = (
    REPO_ROOT
    / "results/ckpts/08-04-4601319-ETTh2-binary_window_norm_patch_refine_canvas128_p64x6_etth2"
)
DEFAULT_CFG = "configs/binary_window_norm_patch_refine_canvas128_p64x6_etth2.yaml"
DEFAULT_OUT = REPO_ROOT / "temp/lean_disc_c128_results/etth2_coarse_flat_run_stats"
FLAT_EPS_FRAC = 0.25
MIN_RUN = 3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--config", type=str, default=DEFAULT_CFG)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--lookback", type=int, default=None)
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--flat-eps-frac", type=float, default=FLAT_EPS_FRAC)
    p.add_argument("--min-run", type=int, default=MIN_RUN)
    p.add_argument("--batch-size", type=int, default=512)
    args = p.parse_args()
    args.ckpt = args.ckpt.expanduser().resolve()
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


def _causal_window_norm_bins(
    data_tv: np.ndarray,
    *,
    lookback: int,
    max_scale: float,
    coarse_h: int,
    std_floor: float,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Causal encode matching training window-norm + coarse bin formula.

    For each t in [lookback, T):
      past = data[t-lookback:t]          # ends at t-1 (exclusive of t)
      center, std = mean(past), max(std(past), std_floor)
      z[t] = (data[t] - center) / std
      bin[t] = floor( clip(z, ±MS) mapped onto [0, H) )

    Returns (bins[Tanal, V], z[Tanal, V]) for absolute times lookback..T-1.
    """
    if data_tv.ndim != 2:
        raise ValueError(f"expected (T,V), got {data_tv.shape}")
    t_len, n_vars = data_tv.shape
    if t_len <= lookback:
        raise RuntimeError(f"split_T={t_len} ≤ lookback={lookback}")
    if max_scale <= 0.0 or coarse_h <= 0:
        raise RuntimeError(f"bad lattice: max_scale={max_scale} coarse_h={coarse_h}")
    if std_floor <= 0.0:
        raise RuntimeError(f"std_floor must be > 0, got {std_floor}")

    n_out = t_len - lookback
    bins = np.empty((n_out, n_vars), dtype=np.int64)
    z_out = np.empty((n_out, n_vars), dtype=np.float64)
    x = data_tv.astype(np.float64, copy=False)

    for t0 in range(0, n_out, batch_size):
        t1 = min(n_out, t0 + batch_size)
        # past windows for absolute times [lookback+t0, lookback+t1)
        # past[i] = x[lookback+t0+i-lookback : lookback+t0+i] = x[t0+i : lookback+t0+i]
        b = t1 - t0
        # Build (b, lookback, V) past cubes
        past = np.stack(
            [x[t0 + i : t0 + i + lookback] for i in range(b)],
            axis=0,
        )  # (b, L, V)
        center = past.mean(axis=1)  # (b, V)
        std = past.std(axis=1)  # (b, V) population std (ddof=0) — torch.std default is unbiased!
        # Training uses torch.std which is unbiased (ddof=1) for L>1.
        # Match torch: ddof=1 when lookback > 1.
        if lookback > 1:
            std = past.std(axis=1, ddof=1)
        std = np.maximum(std, std_floor)
        cur = x[lookback + t0 : lookback + t1]  # (b, V)
        z = (cur - center) / std
        z_clip = np.clip(z, -max_scale, max_scale)
        # Same formula as TimeSeriesTo2D.encode_dual_heights coarse_pos.long()
        pos = (z_clip + max_scale) / (2.0 * max_scale) * coarse_h
        bin_idx = np.floor(pos).astype(np.int64)
        bin_idx = np.clip(bin_idx, 0, coarse_h - 1)
        bins[t0:t1] = bin_idx
        z_out[t0:t1] = z
    return bins, z_out


def _analyze_split(
    *,
    name: str,
    data_tv: np.ndarray,
    lookback: int,
    max_scale: float,
    coarse_h: int,
    std_floor: float,
    flat_eps: float,
    bin_width: float,
    min_run: int,
    batch_size: int,
    exclusive_lo: int,
    exclusive_hi: int,
) -> Dict[str, Any]:
    """Analyze one split.

    ``data_tv`` is the TimeSeriesDataset contiguous array (val/test include
    lookback prefix from the previous split). ``exclusive_lo/hi`` are indices
    into that array marking the split's own timesteps (exclude borrowed
    lookback history from run counting).
    """
    bins, z = _causal_window_norm_bins(
        data_tv,
        lookback=lookback,
        max_scale=max_scale,
        coarse_h=coarse_h,
        std_floor=std_floor,
        batch_size=batch_size,
    )
    # bins[i] corresponds to absolute index abs_t = lookback + i in data_tv
    n_vars = int(data_tv.shape[1])
    per_var: List[Dict[str, Any]] = []
    overall_runs = 0
    overall_flat = 0

    for v in range(n_vars):
        # Restrict to exclusive split region in absolute data_tv indices.
        # Mapped series index i covers abs_t = lookback + i.
        i_lo = max(0, exclusive_lo - lookback)
        i_hi = max(i_lo, exclusive_hi - lookback)
        bins_v = bins[i_lo:i_hi, v]
        z_v = z[i_lo:i_hi, v]
        runs = _find_runs(bins_v, min_run)
        n_runs = len(runs)
        n_flat = 0
        lengths = []
        z_ranges_flat = []
        z_ranges_wobbly = []
        for a, b, _bin_id in runs:
            seg = z_v[a:b]
            z_range = float(seg.max() - seg.min())
            lengths.append(b - a)
            if z_range <= flat_eps:
                n_flat += 1
                z_ranges_flat.append(z_range)
            else:
                z_ranges_wobbly.append(z_range)
        n_wobbly = n_runs - n_flat
        overall_runs += n_runs
        overall_flat += n_flat
        per_var.append(
            {
                "variate": v,
                "n_runs_ge3": n_runs,
                "n_flat": n_flat,
                "n_wobbly": n_wobbly,
                "pct_flat": (100.0 * n_flat / n_runs) if n_runs else float("nan"),
                "pct_wobbly": (100.0 * n_wobbly / n_runs) if n_runs else float("nan"),
                "mean_run_len": float(np.mean(lengths)) if lengths else float("nan"),
                "max_run_len": int(max(lengths)) if lengths else 0,
                "mean_z_range_flat": float(np.mean(z_ranges_flat)) if z_ranges_flat else float("nan"),
                "mean_z_range_wobbly": float(np.mean(z_ranges_wobbly))
                if z_ranges_wobbly
                else float("nan"),
                "n_timesteps_analyzed": int(i_hi - i_lo),
            }
        )

    n_wobbly_all = overall_runs - overall_flat
    return {
        "split": name,
        "split_T_array": int(data_tv.shape[0]),
        "exclusive_lo": int(exclusive_lo),
        "exclusive_hi": int(exclusive_hi),
        "exclusive_T": int(max(0, exclusive_hi - exclusive_lo)),
        "analyzed_T": int(max(0, (exclusive_hi - lookback) - max(0, exclusive_lo - lookback))),
        "lookback": lookback,
        "n_variates": n_vars,
        "bin_width_window_norm": bin_width,
        "flat_eps_abs": flat_eps,
        "overall": {
            "n_runs_ge3": overall_runs,
            "n_flat": overall_flat,
            "n_wobbly": n_wobbly_all,
            "pct_flat": (100.0 * overall_flat / overall_runs) if overall_runs else float("nan"),
            "pct_wobbly": (100.0 * n_wobbly_all / overall_runs) if overall_runs else float("nan"),
        },
        "per_variate": per_var,
    }


def _fmt_pct(x: float) -> str:
    if x != x:
        return "n/a"
    return f"{x:.1f}%"


def _markdown_tables(payload: Dict[str, Any]) -> str:
    meta = payload["meta"]
    lines: List[str] = [
        "# ETTh2 coarse same-bin run stats (canvas128 leaf)",
        "",
        f"- config: `{meta['config']}`",
        f"- ckpt: `{meta['ckpt']}`",
        f"- encode: causal window-norm (past mean / torch-unbiased std, "
        f"std_floor={meta['window_norm_std_floor']}) → coarse bin "
        f"(`coarse_image_height={meta['coarse_image_height']}`)",
        f"- canvas128 HIR height (patch_refine only): `{meta['patch_refine_canvas_height']}` "
        f"({meta['fine_bins_per_coarse']} fine rows / coarse bin)",
        f"- max_scale: **{meta['max_scale']}** (from ckpt metadata)",
        f"- coarse bin width (window-norm z): `2·max_scale/H = {meta['coarse_bin_width']:.6g}`",
        f"- flat rule: `max(z)−min(z) ≤ {meta['flat_eps_frac']} × bin_width` "
        f"= `{meta['flat_eps_abs']:.6g}` in window-norm z",
        f"- min run length: ≥{meta['min_run']} consecutive timesteps (same coarse bin)",
        f"- series: contiguous per-split exclusive timesteps (val/test borrow lookback "
        f"history for encode only; runs counted on exclusive region)",
        f"- paper splits: ETTh 12/4/4 months; lookback={meta['lookback']}, "
        f"horizon={meta['horizon']} (horizon unused for this series scan)",
        "",
        "## Overall per split",
        "",
        "| split | analyzed T | runs (≥3 same bin) | flat | wobbly | % flat | % wobbly |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for sp in payload["splits"]:
        o = sp["overall"]
        lines.append(
            f"| {sp['split']} | {sp['analyzed_T']} | {o['n_runs_ge3']} | {o['n_flat']} | "
            f"{o['n_wobbly']} | {_fmt_pct(o['pct_flat'])} | {_fmt_pct(o['pct_wobbly'])} |"
        )

    lines.extend(["", "## Per split × variate", ""])
    for sp in payload["splits"]:
        lines.extend(
            [
                f"### {sp['split']} (analyzed_T={sp['analyzed_T']}, "
                f"exclusive_T={sp['exclusive_T']}, array_T={sp['split_T_array']})",
                "",
                "| variate | runs | flat | wobbly | % flat | % wobbly | mean len | max len |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in sp["per_variate"]:
            lines.append(
                f"| {row['variate']} | {row['n_runs_ge3']} | {row['n_flat']} | "
                f"{row['n_wobbly']} | {_fmt_pct(row['pct_flat'])} | "
                f"{_fmt_pct(row['pct_wobbly'])} | "
                f"{row['mean_run_len']:.2f} | {row['max_run_len']} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    if not args.ckpt.is_dir():
        raise FileNotFoundError(args.ckpt)

    max_scale = float(_max_scale_from_ckpt_metadata(args.ckpt, DATASET))
    if max_scale <= 0.0:
        raise RuntimeError(f"invalid max_scale={max_scale}")

    cfg = load_experiment_config(args.config, cli_overrides={"dataset": DATASET})
    state = PipelineState.from_config(cfg)
    state.dataset = DATASET
    state.subset_id = DATASET
    patch_globals(pipeline_mod, state, honor_dataset_windows=True)
    patch_stage_globals(pipeline_mod, state, "coarse", honor_dataset_windows=True)

    if bool(getattr(pipeline_mod, "USE_ORDINAL_WINDOW_NORM", False)):
        raise RuntimeError(f"{args.config}: expected use_ordinal_window_norm=False")
    if not bool(getattr(pipeline_mod, "USE_WINDOW_NORMALIZATION", False)):
        raise RuntimeError(f"{args.config}: expected use_window_normalization=True")
    if str(getattr(pipeline_mod, "WINDOW_NORM_CENTER", "")) != "mean":
        raise RuntimeError(
            f"expected window_norm_center=mean, got {pipeline_mod.WINDOW_NORM_CENTER!r}"
        )

    coarse_h = int(getattr(pipeline_mod, "COARSE_IMAGE_HEIGHT", 0) or state.coarse_image_height)
    canvas_h = int(getattr(state, "patch_refine_canvas_height", 0) or 0)
    if coarse_h <= 0:
        raise RuntimeError(f"coarse_image_height unclear: {coarse_h}")
    if canvas_h <= 0:
        raise RuntimeError(f"patch_refine_canvas_height unclear: {canvas_h}")
    if canvas_h % coarse_h != 0:
        raise RuntimeError(f"canvas_h={canvas_h} not divisible by coarse_h={coarse_h}")

    lookback = int(args.lookback or pipeline_mod.LOOKBACK_LENGTH)
    horizon = int(args.horizon or pipeline_mod.FORECAST_LENGTH)
    std_floor = float(getattr(pipeline_mod, "WINDOW_NORM_STD_FLOOR", 0.1))
    bin_width = 2.0 * max_scale / float(coarse_h)
    flat_eps = float(args.flat_eps_frac) * bin_width
    if flat_eps <= 0.0:
        raise RuntimeError(f"flat_eps must be > 0, got {flat_eps}")

    print(
        f"[etth2_coarse_flat] dataset={DATASET} coarse_h={coarse_h} canvas_h={canvas_h} "
        f"max_scale={max_scale} bin_width={bin_width:.6g} flat_eps={flat_eps:.6g} "
        f"lb={lookback} hz={horizon}",
        flush=True,
    )

    train_ds, val_ds, test_ds, stats = load_dataset(
        DATASET,
        lookback=lookback,
        horizon=horizon,
        use_ordinal_window_norm=False,
        max_scale=max_scale,
    )

    # Paper exclusive borders inside each split array:
    # train array = data[0:train_end] → exclusive [0, train_end)
    # val array   = data[train_end-lb : val_end] → exclusive [lb, len)
    # test array  = data[val_end-lb : test_end] → exclusive [lb, len)
    split_specs = [
        ("train", train_ds, 0, int(train_ds.data.shape[0])),
        ("val", val_ds, lookback, int(val_ds.data.shape[0])),
        ("test", test_ds, lookback, int(test_ds.data.shape[0])),
    ]

    splits_out = []
    for name, ds, excl_lo, excl_hi in split_specs:
        data_np = ds.data.detach().cpu().numpy()
        print(
            f"[etth2_coarse_flat] {name}: array_T={data_np.shape[0]} "
            f"exclusive=[{excl_lo},{excl_hi})",
            flush=True,
        )
        splits_out.append(
            _analyze_split(
                name=name,
                data_tv=data_np,
                lookback=lookback,
                max_scale=max_scale,
                coarse_h=coarse_h,
                std_floor=std_floor,
                flat_eps=flat_eps,
                bin_width=bin_width,
                min_run=int(args.min_run),
                batch_size=int(args.batch_size),
                exclusive_lo=excl_lo,
                exclusive_hi=excl_hi,
            )
        )

    # Verify torch.std matches our ddof=1 on a smoke slice.
    _smoke = torch.randn(lookback)
    torch_std = float(_smoke.std(dim=0).item())
    np_std = float(_smoke.numpy().std(ddof=1))
    if abs(torch_std - np_std) > 1e-5:
        raise RuntimeError(f"std convention mismatch torch={torch_std} np={np_std}")

    payload: Dict[str, Any] = {
        "meta": {
            "dataset": DATASET,
            "config": args.config,
            "ckpt": str(args.ckpt),
            "max_scale": max_scale,
            "coarse_image_height": coarse_h,
            "patch_refine_canvas_height": canvas_h,
            "fine_bins_per_coarse": canvas_h // coarse_h,
            "lookback": lookback,
            "horizon": horizon,
            "window_norm_center": "mean",
            "window_norm_std_floor": std_floor,
            "use_window_normalization": True,
            "use_ordinal_window_norm": False,
            "coarse_bin_width": bin_width,
            "flat_eps_frac": float(args.flat_eps_frac),
            "flat_eps_abs": flat_eps,
            "min_run": int(args.min_run),
            "encode_mode": "causal_window_norm_coarse_bin",
            "norm_mean": np.asarray(stats["mean"]).reshape(-1).tolist(),
            "norm_std": np.asarray(stats["std"]).reshape(-1).tolist(),
        },
        "splits": splits_out,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "stats.json"
    md_path = args.output_dir / "stats.md"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    md = _markdown_tables(payload)
    md_path.write_text(md, encoding="utf-8")
    print(md)
    print(f"[etth2_coarse_flat] wrote {json_path}")
    print(f"[etth2_coarse_flat] wrote {md_path}")


if __name__ == "__main__":
    main()
