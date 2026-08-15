#!/usr/bin/env python3
"""ETTh2 canvas128: flat/wiggle prediction accuracy on test forecast windows.

Reuses flat-run definitions from etth2_coarse_flat_run_stats.py, but scores
**binary patch_refine preds** vs GT on the same horizon spans.

Data: ablation/disc pack
  results/datasets/08-04-2009-ablation-disc-l8-l16-ETTh2-c128-valtest80-byvar
  raw/binary_window_norm_c128_ETTh2_val-test.npz
  (ckpt 08-04-4601319, kind=patch_refine, sample0)

Encode (fail-fast if ambiguous):
  Per window, lookback past mean / torch-unbiased std (std_floor) → window-z
  for both GT horizon and sample0 pred (same affine). Coarse bins use
  coarse_image_height=16 + max_scale from ckpt — NOT canvas128 fine rows.
  Pred source = pack samples[:,:,0,:] (final patch_refine / sample0), not
  a separate coarse-stage tensor (none stored in this pack).

Unit: maximal ≥3 same-bin GT runs inside each test-window H=96 horizon.
  flat   = z-range ≤ 0.25 × coarse_bin_width
  wobbly = same-bin run but not flat
Pred label on the same [a,b) span:
  pred_flatline = all pred bins identical AND pred z-range ≤ ε
  pred_wiggle   = not pred_flatline

Examples:
  source .venv/bin/activate
  python temp/scripts/etth2_coarse_flat_pred_acc.py
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
from models.diffusion_tsf.pipeline.phases.staged_diffusion_pretrain import (  # noqa: E402
    stage_state,
)
from models.diffusion_tsf.pipeline.state import PipelineState  # noqa: E402
import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod  # noqa: E402
from temp.scripts.eval_ablation_disc_l8_l16 import (  # noqa: E402
    _max_scale_from_ckpt_metadata,
)
from utils.forecast_pack_reduce import reduce_pack_forecast  # noqa: E402

DATASET = "ETTh2"
DEFAULT_PACK = (
    REPO_ROOT
    / "results/datasets/08-04-2009-ablation-disc-l8-l16-ETTh2-c128-valtest80-byvar"
    / "raw"
    / "binary_window_norm_c128_ETTh2_val-test.npz"
)
DEFAULT_CKPT = (
    REPO_ROOT
    / "results/ckpts/08-04-4601319-ETTh2-binary_window_norm_patch_refine_canvas128_p64x6_etth2"
)
DEFAULT_CFG = "configs/binary_window_norm_patch_refine_canvas128_p64x6_etth2.yaml"
DEFAULT_OUT = REPO_ROOT / "temp/lean_disc_c128_results/etth2_coarse_flat_pred_acc"
FLAT_EPS_FRAC = 0.25
MIN_RUN = 3
# ETTh paper: 12/4/4 months × 30×24
TRAIN_END = 12 * 30 * 24  # 8640
VAL_END = TRAIN_END + 4 * 30 * 24  # 11520


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--config", type=str, default=DEFAULT_CFG)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--flat-eps-frac", type=float, default=FLAT_EPS_FRAC)
    p.add_argument("--min-run", type=int, default=MIN_RUN)
    p.add_argument("--fake-agg", choices=["sample0", "prob_mean"], default="sample0")
    p.add_argument("--lookback", type=int, default=None)
    p.add_argument("--horizon", type=int, default=None)
    args = p.parse_args()
    args.pack = args.pack.expanduser().resolve()
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


def _window_norm_z_and_bins(
    past: np.ndarray,
    future: np.ndarray,
    *,
    max_scale: float,
    coarse_h: int,
    std_floor: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """past (N,V,L), future (N,V,H) → z (N,V,H), bins (N,V,H)."""
    if past.ndim != 3 or future.ndim != 3:
        raise ValueError(f"expected (N,V,*), got past={past.shape} future={future.shape}")
    if past.shape[:2] != future.shape[:2]:
        raise ValueError(f"N/V mismatch past={past.shape} future={future.shape}")
    if max_scale <= 0.0 or coarse_h <= 0:
        raise RuntimeError(f"bad lattice max_scale={max_scale} coarse_h={coarse_h}")
    if std_floor <= 0.0:
        raise RuntimeError(f"std_floor must be > 0, got {std_floor}")

    past_t = torch.from_numpy(np.asarray(past, dtype=np.float32))
    fut_t = torch.from_numpy(np.asarray(future, dtype=np.float32))
    center = past_t.mean(dim=-1)  # (N,V)
    # Match training: torch.std default is unbiased (ddof=1).
    std = past_t.std(dim=-1).clamp_min(std_floor)
    z = (fut_t - center.unsqueeze(-1)) / std.unsqueeze(-1)
    z_np = z.detach().cpu().numpy().astype(np.float64)
    z_clip = np.clip(z_np, -max_scale, max_scale)
    pos = (z_clip + max_scale) / (2.0 * max_scale) * coarse_h
    bins = np.floor(pos).astype(np.int64)
    bins = np.clip(bins, 0, coarse_h - 1)
    return z_np, bins


def _pred_is_flatline(
    pred_bins_1d: np.ndarray,
    pred_z_1d: np.ndarray,
    a: int,
    b: int,
    flat_eps: float,
) -> bool:
    seg_b = pred_bins_1d[a:b]
    seg_z = pred_z_1d[a:b]
    if seg_b.size < 1:
        raise RuntimeError("empty pred segment")
    same_bin = bool(np.all(seg_b == seg_b[0]))
    z_flat = float(seg_z.max() - seg_z.min()) <= flat_eps
    return same_bin and z_flat


def _analyze(
    *,
    past: np.ndarray,
    y_true: np.ndarray,
    pred: np.ndarray,
    max_scale: float,
    coarse_h: int,
    std_floor: float,
    flat_eps: float,
    min_run: int,
) -> Dict[str, Any]:
    z_gt, bins_gt = _window_norm_z_and_bins(
        past, y_true, max_scale=max_scale, coarse_h=coarse_h, std_floor=std_floor
    )
    z_pr, bins_pr = _window_norm_z_and_bins(
        past, pred, max_scale=max_scale, coarse_h=coarse_h, std_floor=std_floor
    )
    n_win, n_vars, _h = bins_gt.shape
    per_var: List[Dict[str, Any]] = []
    tot_flat = tot_flat_hit = 0
    tot_wob = tot_wob_hit = 0

    for v in range(n_vars):
        n_flat = n_flat_hit = 0
        n_wob = n_wob_hit = 0
        for n in range(n_win):
            runs = _find_runs(bins_gt[n, v], min_run)
            for a, b, _bin_id in runs:
                gt_range = float(z_gt[n, v, a:b].max() - z_gt[n, v, a:b].min())
                pred_flat = _pred_is_flatline(bins_pr[n, v], z_pr[n, v], a, b, flat_eps)
                if gt_range <= flat_eps:
                    n_flat += 1
                    if pred_flat:
                        n_flat_hit += 1
                else:
                    n_wob += 1
                    if not pred_flat:
                        n_wob_hit += 1
        tot_flat += n_flat
        tot_flat_hit += n_flat_hit
        tot_wob += n_wob
        tot_wob_hit += n_wob_hit
        per_var.append(
            {
                "variate": v,
                "n_gt_flat": n_flat,
                "n_pred_flat_given_gt_flat": n_flat_hit,
                "pct_pred_flat_given_gt_flat": (
                    100.0 * n_flat_hit / n_flat if n_flat else float("nan")
                ),
                "pct_pred_not_flat_given_gt_flat": (
                    100.0 * (n_flat - n_flat_hit) / n_flat if n_flat else float("nan")
                ),
                "n_gt_wobbly": n_wob,
                "n_pred_wobbly_given_gt_wobbly": n_wob_hit,
                "pct_pred_wobbly_given_gt_wobbly": (
                    100.0 * n_wob_hit / n_wob if n_wob else float("nan")
                ),
                "pct_pred_flat_given_gt_wobbly": (
                    100.0 * (n_wob - n_wob_hit) / n_wob if n_wob else float("nan")
                ),
            }
        )

    overall = {
        "n_gt_flat": tot_flat,
        "n_pred_flat_given_gt_flat": tot_flat_hit,
        "pct_pred_flat_given_gt_flat": (
            100.0 * tot_flat_hit / tot_flat if tot_flat else float("nan")
        ),
        "pct_pred_not_flat_given_gt_flat": (
            100.0 * (tot_flat - tot_flat_hit) / tot_flat if tot_flat else float("nan")
        ),
        "n_gt_wobbly": tot_wob,
        "n_pred_wobbly_given_gt_wobbly": tot_wob_hit,
        "pct_pred_wobbly_given_gt_wobbly": (
            100.0 * tot_wob_hit / tot_wob if tot_wob else float("nan")
        ),
        "pct_pred_flat_given_gt_wobbly": (
            100.0 * (tot_wob - tot_wob_hit) / tot_wob if tot_wob else float("nan")
        ),
    }
    return {
        "n_windows": n_win,
        "n_variates": n_vars,
        "horizon": int(bins_gt.shape[-1]),
        "overall": overall,
        "per_variate": per_var,
    }


def _fmt_pct(x: float) -> str:
    if x != x:
        return "n/a"
    return f"{x:.1f}%"


def _markdown(payload: Dict[str, Any]) -> str:
    m = payload["meta"]
    o = payload["test"]["overall"]
    lines = [
        "# ETTh2 coarse flat/wiggle prediction accuracy (test windows)",
        "",
        f"- pack: `{m['pack']}`",
        f"- ckpt: `{m['ckpt']}`",
        f"- config: `{m['config']}`",
        f"- pred source: `{m['pred_source']}`",
        f"- encode: `{m['encode_mode']}` → coarse H={m['coarse_image_height']} "
        f"(canvas128 HIR={m['patch_refine_canvas_height']} unused for this coarse)",
        f"- max_scale={m['max_scale']}, bin_width={m['coarse_bin_width']:.6g}, "
        f"flat_eps={m['flat_eps_frac']}×bin_width={m['flat_eps_abs']:.6g}",
        f"- min run ≥{m['min_run']}; lookback={m['lookback']}, horizon={m['horizon']}",
        f"- test filter: series_starts ≥ val_end−lb = {m['test_series_start_min']} "
        f"({m['n_test_windows']} windows, stride≈{m['stride']})",
        f"- also on Killarney: `{m['killarney_note']}`",
        "",
        "## Overall (test)",
        "",
        (
            "| n_gt_flat | % pred flat given gt flat | % pred NOT flat given gt flat "
            "| n_gt_wobbly | % pred wobbly given gt wobbly | % pred flat given gt wobbly |"
        ),
        "|---:|---:|---:|---:|---:|---:|",
        (
            f"| {o['n_gt_flat']} | {_fmt_pct(o['pct_pred_flat_given_gt_flat'])} | "
            f"{_fmt_pct(o['pct_pred_not_flat_given_gt_flat'])} | {o['n_gt_wobbly']} | "
            f"{_fmt_pct(o['pct_pred_wobbly_given_gt_wobbly'])} | "
            f"{_fmt_pct(o['pct_pred_flat_given_gt_wobbly'])} |"
        ),
        "",
        "## Per variate (test)",
        "",
        (
            "| variate | n_gt_flat | % pred flat given gt flat | % miss flat "
            "| n_gt_wobbly | % pred wobbly given gt wobbly | % false flat |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["test"]["per_variate"]:
        lines.append(
            f"| {row['variate']} | {row['n_gt_flat']} | "
            f"{_fmt_pct(row['pct_pred_flat_given_gt_flat'])} | "
            f"{_fmt_pct(row['pct_pred_not_flat_given_gt_flat'])} | "
            f"{row['n_gt_wobbly']} | "
            f"{_fmt_pct(row['pct_pred_wobbly_given_gt_wobbly'])} | "
            f"{_fmt_pct(row['pct_pred_flat_given_gt_wobbly'])} |"
        )
    lines.append("")
    lines.extend(
        [
            "## Complementary error rates (overall)",
            "",
            f"- Among GT flatlines: **{_fmt_pct(o['pct_pred_not_flat_given_gt_flat'])}** "
            f"missed (pred not flat).",
            f"- Among GT wiggles: **{_fmt_pct(o['pct_pred_flat_given_gt_wobbly'])}** "
            f"false flat (pred flat).",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not args.pack.is_file():
        raise FileNotFoundError(args.pack)
    if not args.ckpt.is_dir():
        raise FileNotFoundError(args.ckpt)

    max_scale = float(_max_scale_from_ckpt_metadata(args.ckpt, DATASET))
    if max_scale <= 0.0:
        raise RuntimeError(f"invalid max_scale={max_scale}")

    cfg = load_experiment_config(args.config, cli_overrides={"dataset": DATASET})
    state = PipelineState.from_config(cfg)
    state.dataset = DATASET
    state.subset_id = DATASET
    state = stage_state(state, "coarse", honor_dataset_windows=True)

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
    if abs(bin_width - 0.65) > 1e-9:
        raise RuntimeError(
            f"expected coarse bin width 0.65 (2*5.2/16), got {bin_width} "
            f"(max_scale={max_scale}, coarse_h={coarse_h})"
        )

    pack = dict(np.load(args.pack, allow_pickle=True))
    required = ("past", "y_true", "samples", "series_starts", "indices", "kind")
    missing = [k for k in required if k not in pack]
    if missing:
        raise KeyError(f"pack missing keys {missing}; have {sorted(pack)}")
    kind = str(np.asarray(pack["kind"]).reshape(-1)[0])
    if kind != "patch_refine":
        raise RuntimeError(f"expected kind=patch_refine, got {kind!r}")
    canvas_pack = int(np.asarray(pack.get("canvas_height", [-1])).reshape(-1)[0])
    if canvas_pack != canvas_h:
        raise RuntimeError(f"pack canvas_height={canvas_pack} != config {canvas_h}")

    past = np.asarray(pack["past"], dtype=np.float32)
    y_true = np.asarray(pack["y_true"], dtype=np.float32)
    pred = reduce_pack_forecast(pack, agg=args.fake_agg)
    series_starts = np.asarray(pack["series_starts"], dtype=np.int64)
    if past.shape[1:] != (7, lookback):
        raise RuntimeError(f"unexpected past shape {past.shape} (want N,7,{lookback})")
    if y_true.shape[-1] != horizon or pred.shape[-1] != horizon:
        raise RuntimeError(
            f"horizon mismatch y_true={y_true.shape} pred={pred.shape} want H={horizon}"
        )
    if past.shape[0] != y_true.shape[0] or pred.shape != y_true.shape:
        raise RuntimeError(
            f"N/V mismatch past={past.shape} y_true={y_true.shape} pred={pred.shape}"
        )

    test_start_min = VAL_END - lookback
    test_mask = series_starts >= test_start_min
    if not bool(np.any(test_mask)):
        raise RuntimeError(f"no test windows (series_starts >= {test_start_min})")
    # Fail if val windows somehow include test_start_min boundary incorrectly.
    if bool(np.any(series_starts[test_mask] < test_start_min)):
        raise RuntimeError("test mask inconsistent")

    past_t = past[test_mask]
    y_t = y_true[test_mask]
    pred_t = pred[test_mask]
    ss_t = series_starts[test_mask]
    stride_u = np.unique(np.diff(ss_t)) if len(ss_t) > 1 else np.array([0])
    stride = int(stride_u[0]) if len(stride_u) == 1 else -1

    print(
        f"[etth2_flat_pred] test windows={past_t.shape[0]} "
        f"ss=[{int(ss_t.min())},{int(ss_t.max())}] stride={stride} "
        f"coarse_h={coarse_h} max_scale={max_scale} flat_eps={flat_eps:.6g} "
        f"agg={args.fake_agg} kind={kind}",
        flush=True,
    )

    test_stats = _analyze(
        past=past_t,
        y_true=y_t,
        pred=pred_t,
        max_scale=max_scale,
        coarse_h=coarse_h,
        std_floor=std_floor,
        flat_eps=flat_eps,
        min_run=int(args.min_run),
    )

    payload: Dict[str, Any] = {
        "meta": {
            "dataset": DATASET,
            "pack": str(args.pack),
            "ckpt": str(args.ckpt),
            "config": args.config,
            "pred_source": (
                f"pack['samples'] reduced with agg={args.fake_agg} "
                f"(kind={kind}, final patch_refine traj in dataset-z; "
                f"coarse bins derived via lookback window-norm, not stored coarse stage)"
            ),
            "encode_mode": "lookback_window_norm_mean_std_then_coarse_bin",
            "max_scale": max_scale,
            "coarse_image_height": coarse_h,
            "patch_refine_canvas_height": canvas_h,
            "lookback": lookback,
            "horizon": horizon,
            "window_norm_center": "mean",
            "window_norm_std_floor": std_floor,
            "coarse_bin_width": bin_width,
            "flat_eps_frac": float(args.flat_eps_frac),
            "flat_eps_abs": flat_eps,
            "min_run": int(args.min_run),
            "train_end": TRAIN_END,
            "val_end": VAL_END,
            "test_series_start_min": test_start_min,
            "n_test_windows": int(past_t.shape[0]),
            "n_pack_windows": int(past.shape[0]),
            "stride": stride,
            "fake_agg": args.fake_agg,
            "killarney_note": (
                "same pack under "
                "$SCRATCH/ts-sandbox-corrupt-20260729-013232/results/datasets/"
                "08-04-2009-ablation-disc-l8-l16-ETTh2-c128-valtest80-byvar and "
                "disc_forecast_cache/...4601319...npz; ckpt also on "
                "$SCRATCH/ts-sandbox-ordinal-fine/results/ckpts/08-04-4601319-..."
            ),
        },
        "test": test_stats,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "stats.json"
    md_path = args.output_dir / "stats.md"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    md = _markdown(payload)
    md_path.write_text(md, encoding="utf-8")
    print(md)
    print(f"[etth2_flat_pred] wrote {json_path}")
    print(f"[etth2_flat_pred] wrote {md_path}")


if __name__ == "__main__":
    main()
