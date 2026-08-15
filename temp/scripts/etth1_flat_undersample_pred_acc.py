#!/usr/bin/env python3
"""ETTh1 canvas128 flat-undersample: crop-level flat/wiggle prediction recall.

Unit = 6-wide refine crop × active variate (same predicate as
``flatline_windows.classify_unique_segment_flatline_crops`` / train undersample):
  true-flat crop ⇔ ≥ min_run identical coarse bins AND continuous z-range
  ≤ flat_eps_frac × coarse_bin_width somewhere inside that crop.

Scores (test set):
  P(pred=flat | GT=flat) and P(pred=wiggle | GT=wiggle)
for final patch_refine decode (dataset-z → lookback window-norm → coarse bins).
Optionally also scores coarse-stage decode the same way.

Examples:
  source .venv/bin/activate
  # After Killarney materialize (or with --pack):
  python temp/scripts/etth1_flat_undersample_pred_acc.py \\
    --ckpt results/ckpts/08-09-4678498-ETTh1-binary_window_norm_patch_refine_canvas128_p64x6_etth1_flat_undersample \\
    --pack temp/lean_disc_c128_results/etth1_flat_undersample_pred_acc/raw/binary_pack.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.diffusion_tsf.flatline_windows import (  # noqa: E402
    DEFAULT_EPS_FRAC,
    DEFAULT_MIN_RUN,
    variate_has_true_flatline,
)
from models.diffusion_tsf.pipeline.config import load_experiment_config  # noqa: E402
from models.diffusion_tsf.pipeline.phases.staged_diffusion_pretrain import (  # noqa: E402
    stage_state,
)
from models.diffusion_tsf.pipeline.state import PipelineState  # noqa: E402
from models.diffusion_tsf.pipeline.visualize_utils import (  # noqa: E402
    decode_staged_anchor_components,
)
import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod  # noqa: E402
from temp.scripts.eval_ablation_disc_l8_l16 import (  # noqa: E402
    _load_models,
    _max_scale_from_ckpt_metadata,
    _write_pack,
)
from utils.eval_mmpd_gaussian_anchor import (  # noqa: E402
    load_tsf_pack_pool,
    parse_pack_splits,
    run_variate_indices,
)
from utils.staged_binary_forecast import generate_staged_forecast  # noqa: E402

DATASET = "ETTh1"
DEFAULT_CFG = (
    "configs/binary_window_norm_patch_refine_canvas128_p64x6_etth1_flat_undersample.yaml"
)
DEFAULT_CKPT = (
    REPO_ROOT
    / "results/ckpts/08-09-4678498-ETTh1-binary_window_norm_patch_refine_canvas128_p64x6_etth1_flat_undersample"
)
DEFAULT_OUT = REPO_ROOT / "temp/lean_disc_c128_results/etth1_flat_undersample_pred_acc"
# ETTh paper: 12/4/4 months × 30×24
TRAIN_END = 12 * 30 * 24  # 8640
VAL_END = TRAIN_END + 4 * 30 * 24  # 11520


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pack", type=Path, default=None, help="Existing npz with past/y_true/samples")
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--config", type=str, default=DEFAULT_CFG)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--pack-splits", default="test")
    p.add_argument("--pack-test-stride", type=int, default=4)
    p.add_argument("--lookback", type=int, default=336)
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--patch-width", type=int, default=6)
    p.add_argument(
        "--crop-stride",
        type=int,
        default=None,
        help="Crop step inside H (default: config patch_refine_col_stride, else 5)",
    )
    p.add_argument("--flat-eps-frac", type=float, default=DEFAULT_EPS_FRAC)
    p.add_argument("--min-run", type=int, default=DEFAULT_MIN_RUN)
    p.add_argument("--num-sampling-steps", type=int, default=20)
    p.add_argument("--probabilistic-sampler", default="quad_t")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument(
        "--force-materialize",
        action="store_true",
        help="Ignore existing pack / raw cache and regenerate",
    )
    p.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Optional smoke truncate of pack pool indices",
    )
    p.add_argument(
        "--skip-coarse",
        action="store_true",
        help="Do not decode/score coarse-stage prediction",
    )
    args = p.parse_args()
    args.ckpt = args.ckpt.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.pack is not None:
        args.pack = args.pack.expanduser().resolve()
    return args


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
    center = past_t.mean(dim=-1)
    std = past_t.std(dim=-1).clamp_min(std_floor)
    z = (fut_t - center.unsqueeze(-1)) / std.unsqueeze(-1)
    z_np = z.detach().cpu().numpy().astype(np.float64)
    z_clip = np.clip(z_np, -max_scale, max_scale)
    pos = (z_clip + max_scale) / (2.0 * max_scale) * coarse_h
    bins = np.floor(pos).astype(np.int64)
    bins = np.clip(bins, 0, coarse_h - 1)
    return z_np, bins


def _crop_offsets(horizon: int, patch_width: int, crop_stride: int) -> List[int]:
    if patch_width <= 0 or crop_stride <= 0:
        raise ValueError(f"bad pw={patch_width} stride={crop_stride}")
    if patch_width > horizon:
        raise ValueError(f"patch_width={patch_width} > horizon={horizon}")
    return list(range(0, horizon - patch_width + 1, crop_stride))


def _analyze_crops(
    *,
    past: np.ndarray,
    y_true: np.ndarray,
    pred: np.ndarray,
    max_scale: float,
    coarse_h: int,
    std_floor: float,
    flat_eps: float,
    min_run: int,
    patch_width: int,
    crop_stride: int,
) -> Dict[str, Any]:
    z_gt, bins_gt = _window_norm_z_and_bins(
        past, y_true, max_scale=max_scale, coarse_h=coarse_h, std_floor=std_floor
    )
    z_pr, bins_pr = _window_norm_z_and_bins(
        past, pred, max_scale=max_scale, coarse_h=coarse_h, std_floor=std_floor
    )
    n_win, n_vars, horizon = bins_gt.shape
    if pred.shape != y_true.shape:
        raise RuntimeError(f"pred shape {pred.shape} != y_true {y_true.shape}")
    offsets = _crop_offsets(horizon, patch_width, crop_stride)
    if not offsets:
        raise RuntimeError("zero crop offsets")

    per_var: List[Dict[str, Any]] = []
    tot_flat = tot_flat_hit = 0
    tot_wig = tot_wig_hit = 0
    pw = int(patch_width)

    for v in range(n_vars):
        n_flat = n_flat_hit = 0
        n_wig = n_wig_hit = 0
        for n in range(n_win):
            for o in offsets:
                gt_flat = variate_has_true_flatline(
                    z_gt[n, v, o : o + pw],
                    bins_gt[n, v, o : o + pw],
                    flat_eps=flat_eps,
                    min_run=min_run,
                )
                pred_flat = variate_has_true_flatline(
                    z_pr[n, v, o : o + pw],
                    bins_pr[n, v, o : o + pw],
                    flat_eps=flat_eps,
                    min_run=min_run,
                )
                if gt_flat:
                    n_flat += 1
                    if pred_flat:
                        n_flat_hit += 1
                else:
                    n_wig += 1
                    if not pred_flat:
                        n_wig_hit += 1
        tot_flat += n_flat
        tot_flat_hit += n_flat_hit
        tot_wig += n_wig
        tot_wig_hit += n_wig_hit
        per_var.append(
            {
                "variate": v,
                "n_gt_flat": n_flat,
                "n_pred_flat_given_gt_flat": n_flat_hit,
                "pct_pred_flat_given_gt_flat": (
                    100.0 * n_flat_hit / n_flat if n_flat else float("nan")
                ),
                "n_gt_wiggle": n_wig,
                "n_pred_wiggle_given_gt_wiggle": n_wig_hit,
                "pct_pred_wiggle_given_gt_wiggle": (
                    100.0 * n_wig_hit / n_wig if n_wig else float("nan")
                ),
                "pct_pred_flat_given_gt_wiggle": (
                    100.0 * (n_wig - n_wig_hit) / n_wig if n_wig else float("nan")
                ),
            }
        )

    overall = {
        "n_gt_flat": tot_flat,
        "n_pred_flat_given_gt_flat": tot_flat_hit,
        "pct_pred_flat_given_gt_flat": (
            100.0 * tot_flat_hit / tot_flat if tot_flat else float("nan")
        ),
        "n_gt_wiggle": tot_wig,
        "n_pred_wiggle_given_gt_wiggle": tot_wig_hit,
        "pct_pred_wiggle_given_gt_wiggle": (
            100.0 * tot_wig_hit / tot_wig if tot_wig else float("nan")
        ),
        "pct_pred_flat_given_gt_wiggle": (
            100.0 * (tot_wig - tot_wig_hit) / tot_wig if tot_wig else float("nan")
        ),
        "n_crops_total": tot_flat + tot_wig,
        "n_crop_offsets_per_window": len(offsets),
    }
    return {
        "n_windows": n_win,
        "n_variates": n_vars,
        "horizon": int(horizon),
        "crop_offsets": offsets,
        "overall": overall,
        "per_variate": per_var,
    }


def _fmt_pct(x: float) -> str:
    if x != x:
        return "n/a"
    return f"{x:.1f}%"


def _markdown(payload: Dict[str, Any]) -> str:
    m = payload["meta"]
    lines = [
        "# ETTh1 flat-undersample crop-level flat/wiggle prediction accuracy",
        "",
        f"- job/ckpt: `{m['ckpt']}`",
        f"- config: `{m['config']}`",
        f"- pack: `{m.get('pack', '')}`",
        f"- unit: **{m['patch_width']}-wide refine crop** × active variate "
        f"(crop_stride={m['crop_stride']} inside H={m['horizon']})",
        f"- GT / pred flat predicate: same as `flatline_windows` "
        f"(≥{m['min_run']} identical coarse bins AND z-range ≤ "
        f"{m['flat_eps_frac']}×bin_width={m['flat_eps_abs']:.6g} inside crop)",
        f"- encode: lookback window-norm (mean/std, std_floor={m['window_norm_std_floor']}) "
        f"→ coarse H={m['coarse_image_height']} bins; max_scale={m['max_scale']}, "
        f"bin_width={m['coarse_bin_width']:.6g}",
        f"- test: pack_splits={m['pack_splits']}, stride={m['pack_test_stride']}, "
        f"n_windows={m['n_test_windows']}",
        "",
    ]

    for key, title in (
        ("refine", "Final patch_refine decode (sample0 / prediction_global_norm)"),
        ("coarse", "Coarse-stage decode only"),
    ):
        block = payload.get(key)
        if block is None:
            continue
        o = block["overall"]
        lines.extend(
            [
                f"## {title}",
                "",
                f"- pred source: `{m['pred_sources'][key]}`",
                "",
                (
                    "| n_gt_flat | GT flat→pred flat % | n_gt_wiggle | "
                    "GT wiggle→pred wiggle % | GT wiggle→pred flat % |"
                ),
                "|---:|---:|---:|---:|---:|",
                (
                    f"| {o['n_gt_flat']} | {_fmt_pct(o['pct_pred_flat_given_gt_flat'])} | "
                    f"{o['n_gt_wiggle']} | {_fmt_pct(o['pct_pred_wiggle_given_gt_wiggle'])} | "
                    f"{_fmt_pct(o['pct_pred_flat_given_gt_wiggle'])} |"
                ),
                "",
                "### Per variate",
                "",
                (
                    "| variate | n_gt_flat | flat→flat % | n_gt_wiggle | "
                    "wiggle→wiggle % | false flat % |"
                ),
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in block["per_variate"]:
            lines.append(
                f"| {row['variate']} | {row['n_gt_flat']} | "
                f"{_fmt_pct(row['pct_pred_flat_given_gt_flat'])} | "
                f"{row['n_gt_wiggle']} | "
                f"{_fmt_pct(row['pct_pred_wiggle_given_gt_wiggle'])} | "
                f"{_fmt_pct(row['pct_pred_flat_given_gt_wiggle'])} |"
            )
        lines.append("")

    return "\n".join(lines)


def _materialize_pack(
    args: argparse.Namespace,
    *,
    device: torch.device,
    want_coarse: bool,
) -> Dict[str, np.ndarray]:
    pack_splits = parse_pack_splits(args.pack_splits)
    run, coarse, refine, _ladder, kind, canvas_height = _load_models(
        dataset=DATASET,
        ckpt_root=args.ckpt,
        config_path=args.config,
        lookback=args.lookback,
        horizon=args.horizon,
        device=device,
    )
    var_idx = list(run_variate_indices(run))
    pool, starts, splits, _, _ = load_tsf_pack_pool(
        DATASET,
        var_idx,
        lookback=args.lookback,
        horizon=args.horizon,
        train_stride=int(args.pack_test_stride),
        test_stride=int(args.pack_test_stride),
        pack_splits=pack_splits,
        use_ordinal_window_norm=False,
    )
    indices = list(range(len(pool)))
    if args.max_windows is not None:
        indices = indices[: int(args.max_windows)]
    if not indices:
        raise RuntimeError("empty pack pool")

    loader = DataLoader(
        Subset(pool, indices),
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    past_all: List[np.ndarray] = []
    y_true_all: List[np.ndarray] = []
    refine_all: List[np.ndarray] = []
    coarse_all: List[np.ndarray] = []
    n_batches = len(loader)
    print(
        f"[etth1_flat_pred] materialize windows={len(indices)} batches={n_batches} "
        f"steps={args.num_sampling_steps} sampler={args.probabilistic_sampler} "
        f"want_coarse={want_coarse} kind={kind} canvas={canvas_height}",
        flush=True,
    )
    t0 = time.time()
    with torch.no_grad():
        for batch_idx, (past, future) in enumerate(loader):
            past = past.to(device)
            future = future.to(device)
            overlap = int(getattr(refine.config, "lookback_overlap", 0) or 0)
            target = future[..., overlap:] if overlap else future
            torch.manual_seed(int(args.seed) + batch_idx * 1009)
            if want_coarse:
                # Explicit coarse→refine so we can decode coarse_anchor too.
                coarse_out = coarse.generate(
                    past,
                    sampler=args.probabilistic_sampler,
                    num_inference_steps=int(args.num_sampling_steps),
                    emit_guidance_prediction=False,
                )
                refine_out = refine.generate(
                    past,
                    future_coarse_2d=coarse_out["future_2d_coarse"],
                    sampler=args.probabilistic_sampler,
                    num_inference_steps=int(args.num_sampling_steps),
                    emit_guidance_prediction=True,
                )
                coarse_np, _fine_np, _final_np = decode_staged_anchor_components(
                    refine, coarse_out, refine_out
                )
                pred = refine_out["prediction_global_norm"]
            else:
                result = generate_staged_forecast(
                    coarse,
                    refine,
                    past,
                    vertical_dual=False,
                    sampler=args.probabilistic_sampler,
                    num_inference_steps=int(args.num_sampling_steps),
                )
                pred = result["prediction_global_norm"]
                coarse_np = None

            if pred.shape != target.shape:
                raise RuntimeError(
                    f"pred/target mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}"
                )
            past_all.append(past.detach().cpu().numpy().astype(np.float32))
            y_true_all.append(target.detach().cpu().numpy().astype(np.float32))
            refine_all.append(
                pred.detach().cpu().numpy().astype(np.float32)[:, :, None, :]
            )
            if coarse_np is not None:
                c = np.asarray(coarse_np, dtype=np.float32)
                if c.shape != target.shape:
                    raise RuntimeError(
                        f"coarse/target mismatch: {c.shape} vs {tuple(target.shape)}"
                    )
                coarse_all.append(c[:, :, None, :])
            if (batch_idx + 1) == n_batches or (batch_idx + 1) % max(1, n_batches // 5) == 0:
                print(
                    f"[etth1_flat_pred] generate {batch_idx + 1}/{n_batches}",
                    flush=True,
                )

    pack: Dict[str, np.ndarray] = {
        "past": np.concatenate(past_all, axis=0).astype(np.float32),
        "y_true": np.concatenate(y_true_all, axis=0).astype(np.float32),
        "samples": np.concatenate(refine_all, axis=0).astype(np.float32),
        "indices": np.asarray(indices, dtype=np.int64),
        "series_starts": np.asarray(starts, dtype=np.int64)[
            np.asarray(indices, dtype=np.int64)
        ],
        "pack_splits": np.asarray(list(pack_splits), dtype=object),
        "kind": np.asarray([kind]),
        "canvas_height": np.asarray([int(canvas_height)], dtype=np.int64),
        "run": np.asarray([run.dataset if hasattr(run, "dataset") else DATASET]),
    }
    if coarse_all:
        pack["coarse_samples"] = np.concatenate(coarse_all, axis=0).astype(np.float32)
    print(f"[etth1_flat_pred] materialize done in {time.time() - t0:.1f}s", flush=True)
    return pack


def _load_or_materialize(args: argparse.Namespace) -> Dict[str, np.ndarray]:
    raw_dir = args.output_dir / "raw"
    cache = raw_dir / (
        f"binary_{DATASET}_{args.pack_splits.replace(',', '-')}_"
        f"stride{args.pack_test_stride}_s{args.num_sampling_steps}.npz"
    )
    if args.pack is not None:
        if not args.pack.is_file():
            raise FileNotFoundError(args.pack)
        print(f"[etth1_flat_pred] loading --pack {args.pack}", flush=True)
        return dict(np.load(args.pack, allow_pickle=True))
    if cache.is_file() and not args.force_materialize:
        print(f"[etth1_flat_pred] loading cache {cache}", flush=True)
        return dict(np.load(cache, allow_pickle=True))

    if not args.ckpt.is_dir():
        raise FileNotFoundError(args.ckpt)
    if args.cpu:
        device = torch.device("cpu")
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required (pass --cpu to force CPU)")
        device = torch.device(f"cuda:{int(args.gpu)}")
    pack = _materialize_pack(args, device=device, want_coarse=not args.skip_coarse)
    raw_dir.mkdir(parents=True, exist_ok=True)
    _write_pack(cache, pack, label="etth1_flat_pred")
    return pack


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_experiment_config(args.config, cli_overrides={"dataset": DATASET})
    state = PipelineState.from_config(cfg)
    state.dataset = DATASET
    state.subset_id = DATASET
    state = stage_state(state, "coarse", honor_dataset_windows=True)

    if bool(getattr(pipeline_mod, "USE_ORDINAL_WINDOW_NORM", False)):
        raise RuntimeError(f"{args.config}: expected use_ordinal_window_norm=False")
    if not bool(getattr(pipeline_mod, "USE_WINDOW_NORMALIZATION", False)):
        raise RuntimeError(f"{args.config}: expected use_window_normalization=True")

    coarse_h = int(getattr(pipeline_mod, "COARSE_IMAGE_HEIGHT", 0) or state.coarse_image_height)
    canvas_h = int(getattr(state, "patch_refine_canvas_height", 0) or 0)
    col_stride = int(getattr(state, "patch_refine_col_stride", 5) or 5)
    crop_stride = int(args.crop_stride if args.crop_stride is not None else col_stride)
    patch_width = int(args.patch_width or state.patch_refine_patch_width)
    std_floor = float(getattr(pipeline_mod, "WINDOW_NORM_STD_FLOOR", 0.1))
    if coarse_h <= 0 or canvas_h <= 0:
        raise RuntimeError(f"bad coarse_h={coarse_h} canvas_h={canvas_h}")
    if int(args.min_run) > patch_width:
        raise RuntimeError(f"min_run={args.min_run} > patch_width={patch_width}")

    max_scale = float(_max_scale_from_ckpt_metadata(args.ckpt, DATASET))
    bin_width = 2.0 * max_scale / float(coarse_h)
    flat_eps = float(args.flat_eps_frac) * bin_width
    if abs(bin_width - 0.65) > 1e-9:
        raise RuntimeError(
            f"expected coarse bin width 0.65 (2*5.2/16), got {bin_width} "
            f"(max_scale={max_scale}, coarse_h={coarse_h})"
        )

    pack = _load_or_materialize(args)
    required = ("past", "y_true", "samples", "series_starts")
    missing = [k for k in required if k not in pack]
    if missing:
        raise KeyError(f"pack missing {missing}; have {sorted(pack)}")

    past = np.asarray(pack["past"], dtype=np.float32)
    y_true = np.asarray(pack["y_true"], dtype=np.float32)
    samples = np.asarray(pack["samples"], dtype=np.float32)
    if samples.ndim != 4 or samples.shape[2] < 1:
        raise RuntimeError(f"samples must be (N,V,S,H), got {samples.shape}")
    pred_refine = samples[:, :, 0, :].astype(np.float32, copy=False)
    series_starts = np.asarray(pack["series_starts"], dtype=np.int64)

    if past.shape[0] != y_true.shape[0] or pred_refine.shape != y_true.shape:
        raise RuntimeError(
            f"N/V mismatch past={past.shape} y_true={y_true.shape} pred={pred_refine.shape}"
        )
    if y_true.shape[-1] != int(args.horizon):
        raise RuntimeError(f"horizon mismatch y_true={y_true.shape} want H={args.horizon}")

    # Prefer absolute series_starts test filter when pack may mix splits.
    test_start_min = VAL_END - int(args.lookback)
    if str(args.pack_splits).strip().lower() == "test":
        test_mask = np.ones(len(series_starts), dtype=bool)
    else:
        test_mask = series_starts >= test_start_min
    if not bool(np.any(test_mask)):
        raise RuntimeError(f"no test windows (series_starts >= {test_start_min})")

    past_t = past[test_mask]
    y_t = y_true[test_mask]
    pred_t = pred_refine[test_mask]
    ss_t = series_starts[test_mask]

    print(
        f"[etth1_flat_pred] analyze refine: n={past_t.shape[0]} "
        f"ss=[{int(ss_t.min())},{int(ss_t.max())}] pw={patch_width} "
        f"crop_stride={crop_stride} flat_eps={flat_eps:.6g}",
        flush=True,
    )
    refine_stats = _analyze_crops(
        past=past_t,
        y_true=y_t,
        pred=pred_t,
        max_scale=max_scale,
        coarse_h=coarse_h,
        std_floor=std_floor,
        flat_eps=flat_eps,
        min_run=int(args.min_run),
        patch_width=patch_width,
        crop_stride=crop_stride,
    )

    coarse_stats: Optional[Dict[str, Any]] = None
    pred_sources = {
        "refine": (
            "pack['samples'][:,:,0,:] = patch_refine prediction_global_norm "
            "(dataset-z sample0); coarse bins via lookback window-norm"
        ),
    }
    if (not args.skip_coarse) and ("coarse_samples" in pack):
        coarse_s = np.asarray(pack["coarse_samples"], dtype=np.float32)
        if coarse_s.ndim != 4 or coarse_s.shape[2] < 1:
            raise RuntimeError(f"coarse_samples bad shape {coarse_s.shape}")
        pred_c = coarse_s[test_mask][:, :, 0, :]
        print(
            f"[etth1_flat_pred] analyze coarse: n={pred_c.shape[0]}",
            flush=True,
        )
        coarse_stats = _analyze_crops(
            past=past_t,
            y_true=y_t,
            pred=pred_c,
            max_scale=max_scale,
            coarse_h=coarse_h,
            std_floor=std_floor,
            flat_eps=flat_eps,
            min_run=int(args.min_run),
            patch_width=patch_width,
            crop_stride=crop_stride,
        )
        pred_sources["coarse"] = (
            "decode_staged_anchor_components coarse_1d from future_2d_coarse "
            "(cdf mean); same window-norm→coarse bins as refine"
        )

    payload: Dict[str, Any] = {
        "meta": {
            "dataset": DATASET,
            "ckpt": str(args.ckpt),
            "config": args.config,
            "pack": str(args.pack) if args.pack else "",
            "pred_sources": pred_sources,
            "unit": "refine_crop",
            "patch_width": patch_width,
            "crop_stride": crop_stride,
            "config_col_stride": col_stride,
            "max_scale": max_scale,
            "coarse_image_height": coarse_h,
            "patch_refine_canvas_height": canvas_h,
            "lookback": int(args.lookback),
            "horizon": int(args.horizon),
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
            "pack_splits": args.pack_splits,
            "pack_test_stride": int(args.pack_test_stride),
            "num_sampling_steps": int(args.num_sampling_steps),
            "probabilistic_sampler": args.probabilistic_sampler,
            "job_note": "4678498 ETTh1 canvas128 flat-undersample refine",
        },
        "refine": refine_stats,
    }
    if coarse_stats is not None:
        payload["coarse"] = coarse_stats

    json_path = args.output_dir / "stats.json"
    md_path = args.output_dir / "stats.md"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    md = _markdown(payload)
    md_path.write_text(md, encoding="utf-8")
    print(md)
    print(f"[etth1_flat_pred] wrote {json_path}")
    print(f"[etth1_flat_pred] wrote {md_path}")


if __name__ == "__main__":
    main()
