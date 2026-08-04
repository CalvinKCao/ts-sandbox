#!/usr/bin/env python3
"""L8/L16 candidate-only disc for patch_refine / residual-fine ablation ckpts.

Shared fair protocol (matches eval_univariate_disc_two_ablations_vs_gt):
  - generate final forecasts (sample0)
  - snap GT / binary / MMPD onto the absolute ordinal patch-refine ladder
    with ``canvas_height`` from the run config (256 legacy / 128 coarser leaf).
    Window-norm ckpts still use this absolute z-ladder for fair disc inputs;
    do not swap in per-window ``snap_to_unbounded_patch_refine_grid``.
  - train candidate-only L=8 and L=16 discriminators vs GT with bin-center shift

Supports both checkpoint layouts:
  - coarse + patch_refine  (window-norm / canvas128 leaves)
  - coarse + fine          (ordinal residual)

``--viz-only``: skip disc train; write zoomed L8/L16 disc-input panels so the
ladder snap is visually checkable before a Killarney submit.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.diffusion_tsf.pipeline.globals_bridge import patch_globals
from models.diffusion_tsf.train_multivariate_pipeline import (
    load_dataset,
    load_wrapped_guidance,
    resolve_pipeline_data_subset,
)
from temp.eval_univariate_patch_refine_vs_gt import load_patch_refine_run
from utils.disc_bin_center_shift import bin_center_shift, nearest_bin_indices
from utils.dual_scale_bin_filter import align_mmpd_to_binary_dataset_norm
from utils.eval_discriminator_binary_vs_mmpd_univariate import train_classifier
from utils.eval_discriminator_texture_staged_vs_mmpd import (
    binary_mmpd_train_scaler_map,
    split_windows,
    write_json,
)
from utils.eval_mmpd_gaussian_anchor import (
    DEFAULT_MMPD_DATA,
    AnchorRun,
    load_tsf_pack_pool,
    parse_pack_splits,
    run_subset_id,
    run_test_stride,
    run_train_stride,
    run_variate_indices,
)
from utils.forecast_pack_reduce import reduce_pack_forecast
from utils.patch_refine_ordinal_ladder import (
    assert_on_patch_refine_levels,
    legal_patch_refine_levels_dataset_z,
    snap_to_patch_refine_levels,
)
from utils.staged_binary_forecast import generate_staged_forecast
from utils.visualize_staged_eval_2d_preds import (
    _build_state,
    _load_stage_model,
    _resolve_guidance_ckpt,
)
from utils.visualize_staged_forecast import _load_staged_bundle


DEFAULT_MMPD = (
    REPO_ROOT / "results/datasets/07-29-0151-h96-ordinal-binary-nonordinal-mmpd-mmpd"
)
# Default: canvas128 coarser ladder leaf (override --runs after train / for legacy 256 ckpts).
DEFAULT_RUNS = (
    "window_norm_c128:results/ckpts/PLACEHOLDER-ETTh1-binary_window_norm_patch_refine_canvas128_p64x6:"
    "configs/binary_window_norm_patch_refine_canvas128_p64x6.yaml",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="ETTh1")
    p.add_argument(
        "--runs",
        nargs="+",
        default=list(DEFAULT_RUNS),
        help="name:ckpt_root:config triples",
    )
    p.add_argument("--mmpd-output-root", type=Path, default=DEFAULT_MMPD)
    p.add_argument("--mmpd-data-dir", type=Path, default=DEFAULT_MMPD_DATA)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--lookback", type=int, default=336)
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--pack-test-stride", type=int, default=4)
    p.add_argument("--pack-splits", default="test")
    p.add_argument("--fake-agg", choices=["sample0", "prob_mean"], default="sample0")
    p.add_argument("--slice-lengths", type=int, nargs="+", default=[8, 16])
    p.add_argument("--num-sampling-steps", type=int, default=20)
    p.add_argument("--probabilistic-sampler", default="quad_t")
    p.add_argument("--raw-binary-batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force-raw-eval", action="store_true")
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--test-fraction", type=float, default=1.0)
    p.add_argument("--disc-index-stride", type=int, default=1)
    p.add_argument("--candidate-only", action="store_true", default=True)
    p.add_argument("--no-candidate-only", action="store_false", dest="candidate_only")
    p.add_argument("--disc-bin-center-shift", action="store_true", default=True)
    p.add_argument("--no-disc-bin-center-shift", action="store_false", dest="disc_bin_center_shift")
    p.add_argument("--disc-bin-center-reduce", default="per_variate")
    p.add_argument("--nonoverlapping-patches", action="store_true", default=False)
    p.add_argument("--no-offset-embedding", action="store_true", default=False)
    p.add_argument("--offset-stride", type=int, default=1)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--d-ff", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--max-batches-per-epoch", type=int, default=None)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--train-fraction", type=float, default=0.7)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--max-train-examples", type=int, default=None)
    p.add_argument("--max-eval-examples", type=int, default=None)
    p.add_argument("--force-train", action="store_true", default=True)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--canvas-height",
        type=int,
        default=None,
        help="Absolute patch-refine ladder rows. Default: read from each run's config "
        "(patch_refine_canvas_height). Fail if missing.",
    )
    p.add_argument(
        "--viz-only",
        action="store_true",
        help="Generate zoomed L8/L16 disc-input lattice panels; skip disc training.",
    )
    p.add_argument("--viz-n-windows", type=int, default=2)
    p.add_argument("--viz-variate", type=int, default=0)
    p.add_argument("--viz-zoom-steps", type=int, default=12)
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def apply_smoke(args: argparse.Namespace) -> None:
    if not args.smoke_test:
        return
    args.max_windows = min(int(args.max_windows or 4), 4)
    args.num_sampling_steps = min(int(args.num_sampling_steps), 2)
    args.epochs = min(int(args.epochs), 2)
    args.viz_n_windows = min(int(args.viz_n_windows), 2)
    args.raw_binary_batch_size = 1


def _parse_run_specs(specs: Sequence[str]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for spec in specs:
        parts = str(spec).split(":")
        if len(parts) != 3:
            raise ValueError(f"bad --runs entry (want name:ckpt:config): {spec}")
        out.append({"name": parts[0], "ckpt": parts[1], "config": parts[2]})
    return out


def _load_fine_run(dataset: str, checkpoint_dir: Path) -> Tuple[AnchorRun, Dict[str, Path]]:
    bundle = _load_staged_bundle(checkpoint_dir, dataset)
    if str(bundle.get("stage")) == "vertical_dual":
        raise ValueError(f"{checkpoint_dir}: vertical_dual not supported here")
    meta = dict(bundle["fine_metadata"])
    meta["dataset_name"] = dataset
    meta["dataset"] = dataset
    run = AnchorRun(
        variant="binary_coarse_fine",
        dataset=dataset,
        root=checkpoint_dir,
        subset_dir=Path(bundle["coarse_pt"]).parent.parent,
        best_pt=Path(bundle["fine_pt"]),
        itrans_pt=None,
        metadata=meta,
    )
    return run, {
        "coarse_pt": Path(bundle["coarse_pt"]),
        "refine_pt": Path(bundle["fine_pt"]),
        "stage": "fine",
    }


def load_ablation_run(
    dataset: str,
    checkpoint_dir: Path,
) -> Tuple[AnchorRun, Dict[str, Path], str]:
    """Return (run, stages, kind) where kind is patch_refine|fine."""
    try:
        run, stages = load_patch_refine_run(dataset, checkpoint_dir, test_stride=None)
        stages = dict(stages)
        stages["stage"] = "patch_refine"
        return run, stages, "patch_refine"
    except FileNotFoundError:
        run, stages = _load_fine_run(dataset, checkpoint_dir)
        return run, stages, "fine"


def _mmpd_pack(root: Path, dataset: str) -> Dict[str, np.ndarray]:
    path = root / "raw" / f"mmpd_{dataset}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"missing MMPD pack: {path}")
    with np.load(path) as data:
        pack = {key: data[key] for key in data.files}
    for key in ("y_true", "samples", "indices"):
        if key not in pack:
            raise KeyError(f"{path} missing {key}")
    return pack


def _subset_aligned(
    indices: Sequence[int],
    pack: Mapping[str, np.ndarray],
    pick: np.ndarray,
) -> Tuple[List[int], Dict[str, np.ndarray]]:
    pick = np.asarray(pick, dtype=np.int64)
    thinned_indices = [int(indices[int(i)]) for i in pick.tolist()]
    n_full = len(indices)
    thinned = {
        key: (
            value[pick]
            if isinstance(value, np.ndarray) and value.shape[:1] == (n_full,)
            else value
        )
        for key, value in pack.items()
    }
    return thinned_indices, thinned


def _thin_windows(
    indices: Sequence[int],
    pack: Mapping[str, np.ndarray],
    *,
    max_windows: Optional[int],
    seed: int,
) -> Tuple[List[int], Dict[str, np.ndarray]]:
    n = len(indices)
    if max_windows is None or max_windows >= n:
        return list(indices), dict(pack)
    rng = np.random.default_rng(int(seed))
    pick = np.sort(rng.choice(n, size=int(max_windows), replace=False))
    return _subset_aligned(indices, pack, pick)


def _binary_lattice_atol(legal_levels: np.ndarray) -> float:
    gaps = np.diff(np.sort(np.asarray(legal_levels, dtype=np.float64), axis=-1), axis=-1)
    positive = gaps[gaps > 0]
    if positive.size == 0:
        return 1e-4
    return float(max(1e-4, 0.25 * float(np.min(positive))))


def _ladder_only(
    *,
    dataset: str,
    run: AnchorRun,
    lookback: int,
    horizon: int,
) -> Any:
    _, _, _, norm_stats = load_dataset(
        dataset,
        run_variate_indices(run),
        stride=run_train_stride(run),
        test_stride=run_test_stride(run),
        lookback=lookback,
        horizon=horizon,
        ordinal_tie_atol=1e-6,
        use_ordinal_window_norm=True,
    )
    ladder = norm_stats.get("ordinal_ladder")
    if ladder is None:
        raise RuntimeError(f"{dataset}: ordinal ladder missing (needed for patch-refine snap)")
    return ladder


def _canvas_height_from_state(state: Any, override: Optional[int]) -> int:
    if override is not None and int(override) > 0:
        return int(override)
    h = int(getattr(state, "patch_refine_canvas_height", 0) or 0)
    if h <= 0:
        raise RuntimeError(
            "patch_refine_canvas_height missing/invalid; pass --canvas-height or set it in YAML"
        )
    return h


def _load_models(
    *,
    dataset: str,
    ckpt_root: Path,
    config_path: str,
    lookback: int,
    horizon: int,
    device: torch.device,
) -> Tuple[AnchorRun, Any, Any, Any, str, int]:
    run, stages, kind = load_ablation_run(dataset, ckpt_root)
    state = _build_state(ckpt_root, dataset, run_subset_id(run), config_path)
    resolve_pipeline_data_subset(state)
    import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod

    ladder = _ladder_only(
        dataset=dataset, run=run, lookback=lookback, horizon=horizon,
    )
    canvas_height = _canvas_height_from_state(state, None)

    # Model-side ladder only when the checkpoint itself was ordinal-trained.
    if bool(state.use_ordinal_window_norm):
        state.extra["global_ordinal_ladder"] = ladder
        pipeline_mod.GLOBAL_ORDINAL_LADDER = ladder
    else:
        state.extra.pop("global_ordinal_ladder", None)
        pipeline_mod.GLOBAL_ORDINAL_LADDER = None
    patch_globals(pipeline_mod, state, honor_dataset_windows=True)

    guidance = None
    if bool(state.use_guidance_channel) or not bool(state.disable_cross_attention):
        path, guidance_type = _resolve_guidance_ckpt(ckpt_root, run_subset_id(run), "auto")
        guidance = load_wrapped_guidance(
            str(path),
            len(run_variate_indices(run)),
            device,
            guidance_type=guidance_type,
            dataset_lookback=lookback,
            dataset_horizon=horizon,
        )
        if hasattr(guidance, "ordinal_ladder") and bool(state.use_ordinal_window_norm):
            guidance.ordinal_ladder = ladder

    refine_stage = "patch_refine" if kind == "patch_refine" else "fine"
    coarse = _load_stage_model(
        state, "coarse", stages["coarse_pt"], guidance, len(run_variate_indices(run)), device,
        strict_non_guidance_shapes=True,
    )
    refine = _load_stage_model(
        state, refine_stage, stages["refine_pt"], guidance, len(run_variate_indices(run)), device,
        strict_non_guidance_shapes=True,
    )
    for model in (coarse, refine):
        if bool(state.use_ordinal_window_norm):
            model._ordinal_input_is_ranked = False
            model._ordinal_apply_ood_shift = True
    return run, coarse, refine, ladder, kind, canvas_height


def materialize_binary_pack(
    args: argparse.Namespace,
    *,
    dataset: str,
    run_name: str,
    ckpt_root: Path,
    config_path: str,
    indices: Sequence[int],
    device: torch.device,
) -> Tuple[Dict[str, np.ndarray], AnchorRun, Any, str, int]:
    cache = args.output_dir / "raw" / f"binary_{run_name}_{dataset}.npz"
    if cache.is_file() and not args.force_raw_eval:
        with np.load(cache, allow_pickle=True) as data:
            pack = {key: data[key] for key in data.files}
        if np.array_equal(pack.get("indices"), np.asarray(indices, dtype=np.int64)):
            run, _stages, kind = load_ablation_run(dataset, ckpt_root)
            kind = str(pack.get("kind", [kind])[0]) if "kind" in pack else kind
            ladder = _ladder_only(
                dataset=dataset,
                run=run,
                lookback=args.lookback,
                horizon=args.horizon,
            )
            state = _build_state(ckpt_root, dataset, run_subset_id(run), config_path)
            if "canvas_height" in pack:
                cached_h = int(np.asarray(pack["canvas_height"]).reshape(-1)[0])
            else:
                cached_h = 0
            canvas_height = _canvas_height_from_state(
                state, getattr(args, "canvas_height", None) or (cached_h or None),
            )
            return pack, run, ladder, kind, canvas_height

    run, coarse, refine, ladder, kind, canvas_height = _load_models(
        dataset=dataset,
        ckpt_root=ckpt_root,
        config_path=config_path,
        lookback=args.lookback,
        horizon=args.horizon,
        device=device,
    )
    if getattr(args, "canvas_height", None) is not None:
        canvas_height = int(args.canvas_height)
        if canvas_height <= 0:
            raise RuntimeError(f"bad --canvas-height {canvas_height}")
    pool, starts, splits, _, _ = load_tsf_pack_pool(
        dataset,
        run_variate_indices(run),
        lookback=args.lookback,
        horizon=args.horizon,
        train_stride=run_train_stride(run),
        test_stride=int(args.pack_test_stride),
        pack_splits=parse_pack_splits(args.pack_splits),
        use_ordinal_window_norm=False,
    )
    if not indices or min(indices) < 0 or max(indices) >= len(pool):
        raise ValueError(
            f"{dataset}/{run_name}: indices outside pack pool "
            f"(n={len(indices)}, pool_len={len(pool)}, stride={args.pack_test_stride})"
        )
    loader = DataLoader(
        Subset(pool, list(indices)),
        batch_size=max(1, int(args.raw_binary_batch_size)),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
    )
    past_all: List[np.ndarray] = []
    y_true_all: List[np.ndarray] = []
    samples_all: List[np.ndarray] = []
    n_batches = len(loader)
    print(
        f"[{run_name}/{dataset}] materializing: windows={len(indices)} "
        f"batches={n_batches} steps={args.num_sampling_steps} sampler={args.probabilistic_sampler} "
        f"canvas_height={canvas_height}",
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
            result = generate_staged_forecast(
                coarse,
                refine,
                past,
                vertical_dual=False,
                sampler=args.probabilistic_sampler,
                num_inference_steps=int(args.num_sampling_steps),
            )
            pred = result["prediction_global_norm"]
            if pred.shape != target.shape:
                raise RuntimeError(
                    f"pred/target mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}"
                )
            past_all.append(past.detach().cpu().numpy().astype(np.float32))
            y_true_all.append(target.detach().cpu().numpy().astype(np.float32))
            samples_all.append(pred.detach().cpu().numpy().astype(np.float32)[:, :, None, :])
            if (batch_idx + 1) == n_batches or (batch_idx + 1) % max(1, n_batches // 5) == 0:
                print(
                    f"[{run_name}/{dataset}] generate {batch_idx + 1}/{n_batches}",
                    flush=True,
                )
    pack = {
        "past": np.concatenate(past_all, axis=0).astype(np.float32),
        "y_true": np.concatenate(y_true_all, axis=0).astype(np.float32),
        "samples": np.concatenate(samples_all, axis=0).astype(np.float32),
        "indices": np.asarray(indices, dtype=np.int64),
        "series_starts": np.asarray(starts, dtype=np.int64)[np.asarray(indices, dtype=np.int64)],
        "pack_splits": np.asarray(list(splits) if not isinstance(splits, dict) else list(splits.keys()), dtype=object),
        "kind": np.asarray([kind]),
        "canvas_height": np.asarray([int(canvas_height)], dtype=np.int64),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **pack)
    print(f"[{run_name}/{dataset}] wrote {cache} in {time.time() - t0:.1f}s", flush=True)
    return pack, run, ladder, kind, canvas_height


def _snap_bundle(
    *,
    binary_pack: Mapping[str, np.ndarray],
    mmpd_pack: Mapping[str, np.ndarray],
    run: AnchorRun,
    ladder: Any,
    args: argparse.Namespace,
    device: torch.device,
    canvas_height: int,
) -> Dict[str, np.ndarray]:
    binary_gt = np.asarray(binary_pack["y_true"], dtype=np.float32)
    binary_pred = reduce_pack_forecast(binary_pack, agg=args.fake_agg)
    mmpd_gt = np.asarray(mmpd_pack["y_true"], dtype=np.float32)
    mmpd_pred = reduce_pack_forecast(mmpd_pack, agg=args.fake_agg)
    if not np.array_equal(binary_pack["indices"], mmpd_pack["indices"]):
        raise RuntimeError("binary/MMPD indices differ after thinning")
    scalers = binary_mmpd_train_scaler_map(args, run)
    mmpd_binary_z, align = align_mmpd_to_binary_dataset_norm(
        binary_y_true=binary_gt,
        mmpd_y_true=mmpd_gt,
        mmpd_fakes=mmpd_pred,
        **scalers,
    )
    past = np.asarray(binary_pack["past"], dtype=np.float32)
    h = int(canvas_height)
    if h <= 0:
        raise RuntimeError(f"canvas_height must be positive, got {h}")
    print(f"  snap ladder canvas_height={h}", flush=True)
    legal_levels = legal_patch_refine_levels_dataset_z(
        past, ladder=ladder, canvas_height=h, device=device,
    )
    gt, gt_snap = snap_to_patch_refine_levels(binary_gt, legal_levels)
    mmpd, mmpd_snap = snap_to_patch_refine_levels(mmpd_binary_z, legal_levels)
    atol = _binary_lattice_atol(legal_levels)
    binary_raw = np.asarray(binary_pred, dtype=np.float32)
    binary, binary_snap = snap_to_patch_refine_levels(binary_raw, legal_levels)
    raw_err = float(np.abs(binary_raw - binary).max(initial=0.0))
    if raw_err > atol:
        print(
            f"  binary off lattice max_error={raw_err:.6g} atol={atol:.6g}; "
            f"snapping (mean_abs_delta={binary_snap['mean_abs_snap_delta']:.6g})",
            flush=True,
        )
    lattice = {
        "gt": assert_on_patch_refine_levels(gt, legal_levels),
        "binary": assert_on_patch_refine_levels(binary, legal_levels),
        "mmpd": assert_on_patch_refine_levels(mmpd, legal_levels),
        "gt_snap": gt_snap,
        "binary_snap": binary_snap,
        "mmpd_snap": mmpd_snap,
        "mmpd_align": align,
        "raw_binary_max_error": raw_err,
        "support_atol": atol,
        "canvas_height": h,
    }
    return {
        "gt": gt,
        "binary": binary,
        "mmpd": mmpd,
        "past": past,
        "legal_levels": np.asarray(legal_levels, dtype=np.float32),
        "indices": np.asarray(binary_pack["indices"], dtype=np.int64),
        "series_starts": np.asarray(binary_pack["series_starts"], dtype=np.int64),
        "lattice": lattice,
        "canvas_height": h,
    }


def _snap_residual(values_1d: np.ndarray, levels_1d: np.ndarray) -> float:
    vals = np.asarray(values_1d, dtype=np.float32)
    lev = np.asarray(levels_1d, dtype=np.float32)
    return float(np.abs(vals[:, None] - lev[None, :]).min(axis=1).max(initial=0.0))


def _plot_snap_proof_panel(
    *,
    out_path: Path,
    title: str,
    levels_1d: np.ndarray,
    series: Mapping[str, np.ndarray],
    colors: Mapping[str, str],
    t0: int = 0,
) -> Dict[str, float]:
    """Marker + occupied-rung proof that values sit on the absolute ladder.

    Drawing *all* rungs on a dense canvas looks continuous, and steps-post
    verticals cross between rungs — both make a true snap look wrong. Here we
    only draw occupied legal levels and plot markers (no step verticals), plus
    an integer bin-index panel that cannot lie.
    """
    names = list(series.keys())
    y_stack = np.concatenate([np.asarray(series[n], dtype=np.float64) for n in names])
    x = np.arange(t0, t0 + int(y_stack.size // len(names)))
    # per-series length check
    length = int(next(iter(series.values())).shape[0])
    x = np.arange(t0, t0 + length)
    n_rows = int(np.asarray(levels_1d).shape[0])

    residuals = {n: _snap_residual(series[n], levels_1d) for n in names}
    max_err = float(max(residuals.values()))
    if max_err > 1e-5:
        raise RuntimeError(f"{title}: snap residual {max_err:.3e} — refusing to plot")

    occupied = np.unique(
        np.concatenate([np.asarray(series[n], dtype=np.float64) for n in names])
    )
    bins = {
        n: nearest_bin_indices(
            np.asarray(series[n], dtype=np.float32)[None, None, :],
            np.asarray(levels_1d, dtype=np.float32)[None, None, :],
        )[0, 0]
        for n in names
    }

    fig, (ax_y, ax_b) = plt.subplots(
        2, 1, figsize=(max(9.0, 0.55 * length + 3.5), 7.0),
        gridspec_kw={"height_ratios": [2.2, 1.4]}, sharex=True,
    )
    ax_y.set_facecolor("white")
    # Occupied rungs only — exact membership of the plotted points.
    for y in occupied:
        ax_y.axhline(float(y), color="0.55", lw=0.9, alpha=0.85, zorder=0)
    for n in names:
        y = np.asarray(series[n], dtype=np.float64)
        # Faint polyline (no steps) so eye can track series; markers carry the snap proof.
        ax_y.plot(x, y, color=colors[n], lw=1.0, alpha=0.35, zorder=1)
        ax_y.plot(
            x, y, linestyle="none", marker="o", markersize=7.5,
            markerfacecolor=colors[n], markeredgecolor="white", markeredgewidth=0.6,
            label=f"{n} (max|Δ|={residuals[n]:.1e})", zorder=3,
        )
    ax_y.set_ylabel("dataset-z (snapped)")
    ax_y.set_title(
        f"{title}\noccupied rungs only ({occupied.size}/{n_rows}); "
        f"all markers on ladder (max residual {max_err:.1e})",
        fontsize=10,
    )
    ax_y.legend(loc="best", fontsize=8, framealpha=0.9)
    ax_y.grid(alpha=0.15)

    for n in names:
        ax_b.plot(
            x, bins[n], color=colors[n], lw=1.0, alpha=0.35, zorder=1,
        )
        ax_b.plot(
            x, bins[n], linestyle="none", marker="s", markersize=6.5,
            markerfacecolor=colors[n], markeredgecolor="white", markeredgewidth=0.5,
            label=n, zorder=3,
        )
    ax_b.set_ylabel(f"{n_rows}-row bin index")
    ax_b.set_xlabel("horizon step t")
    ax_b.set_title("integer ladder row (discrete; same alphabet for GT / binary / MMPD)", fontsize=9)
    ax_b.legend(loc="best", fontsize=8, framealpha=0.9, ncol=3)
    ax_b.grid(alpha=0.15)
    ax_b.set_yticks(sorted({int(v) for b in bins.values() for v in b.tolist()}))

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {"max_snap_residual": max_err, "n_occupied_rungs": float(occupied.size), **{
        f"residual_{n}": residuals[n] for n in names
    }}


def _write_zoom_viz(
    *,
    out_dir: Path,
    run_name: str,
    dataset: str,
    snapped: Mapping[str, np.ndarray],
    n_windows: int,
    variate: int,
    slice_lengths: Sequence[int],
    zoom_steps: int,
    seed: int,
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    gt = snapped["gt"]
    binary = snapped["binary"]
    mmpd = snapped["mmpd"]
    levels = snapped["legal_levels"]
    indices = snapped["indices"]
    n = int(gt.shape[0])
    rng = np.random.default_rng(int(seed) + 17)
    picks = np.sort(rng.choice(n, size=min(int(n_windows), n), replace=False))
    colors = {"GT": "black", "binary": "#1f77b4", "MMPD": "#d62728"}
    paths: List[Path] = []
    for local in picks.tolist():
        pool_i = int(indices[local])
        levels_v = levels[local, variate]
        for L in slice_lengths:
            L = int(L)
            if L > int(gt.shape[-1]):
                continue
            offset = max(0, (int(gt.shape[-1]) - L) // 2)
            # Disc sees bin-center-shifted L-slice; show that exact input.
            series_raw = {
                "GT": gt[local, variate, offset : offset + L],
                "binary": binary[local, variate, offset : offset + L],
                "MMPD": mmpd[local, variate, offset : offset + L],
            }
            series_disc: Dict[str, np.ndarray] = {}
            for name, seg in series_raw.items():
                shifted, _ = bin_center_shift(
                    seg[None, None, :],
                    levels[local : local + 1, variate : variate + 1, :],
                    reduce="per_variate",
                )
                series_disc[name] = shifted[0, 0]
            # Zoom crop inside the L-slice for readability.
            z_steps = min(int(zoom_steps), L)
            z0 = max(0, (L - z_steps) // 2)
            z1 = z0 + z_steps
            path = out_dir / (
                f"{run_name}_{dataset}_v{variate}_local{local}_pool{pool_i}_"
                f"L{L}_off{offset}_snapproof.png"
            )
            _plot_snap_proof_panel(
                out_path=path,
                title=(
                    f"{run_name}/{dataset} pool={pool_i} local={local} v={variate} | "
                    f"disc L={L} off={offset} t=[{z0},{z1}) AFTER bin_center_shift"
                ),
                levels_1d=levels_v,
                series={k: v[z0:z1] for k, v in series_disc.items()},
                colors=colors,
                t0=offset + z0,
            )
            paths.append(path)

        # Early-horizon snap proof (pre bin-center; post lattice snap).
        z1 = min(16, int(gt.shape[-1]))
        path = out_dir / (
            f"{run_name}_{dataset}_v{variate}_local{local}_pool{pool_i}_t0-{z1}_snapproof.png"
        )
        _plot_snap_proof_panel(
            out_path=path,
            title=(
                f"{run_name}/{dataset} pool={pool_i} local={local} v={variate} | "
                f"post-snap (pre bin_center) t=0..{z1 - 1}"
            ),
            levels_1d=levels_v,
            series={
                "GT": gt[local, variate, :z1],
                "binary": binary[local, variate, :z1],
                "MMPD": mmpd[local, variate, :z1],
            },
            colors=colors,
            t0=0,
        )
        paths.append(path)
    return paths


def run_one(
    args: argparse.Namespace,
    *,
    run_name: str,
    ckpt_root: Path,
    config_path: str,
    device: torch.device,
) -> Dict[str, Any]:
    dataset = str(args.dataset)
    print(f"\n=== {run_name} ({ckpt_root.name}) ===", flush=True)
    mmpd_full = _mmpd_pack(args.mmpd_output_root, dataset)
    indices = [int(x) for x in np.asarray(mmpd_full["indices"], dtype=np.int64).tolist()]
    indices, mmpd_pack = _thin_windows(
        indices, mmpd_full, max_windows=args.max_windows, seed=args.seed,
    )
    print(f"[{run_name}] windows={len(indices)} (MMPD-aligned)", flush=True)

    binary_pack, run, ladder, kind, canvas_height = materialize_binary_pack(
        args,
        dataset=dataset,
        run_name=run_name,
        ckpt_root=ckpt_root,
        config_path=config_path,
        indices=indices,
        device=device,
    )
    print(f"[{run_name}] stage_kind={kind} canvas_height={canvas_height}", flush=True)
    snapped = _snap_bundle(
        binary_pack=binary_pack,
        mmpd_pack=mmpd_pack,
        run=run,
        ladder=ladder,
        args=args,
        device=device,
        canvas_height=canvas_height,
    )
    write_json(
        args.output_dir / "partials" / f"lattice_{run_name}_{dataset}.json",
        {
            "kind": kind,
            "canvas_height": canvas_height,
            "raw_binary_max_error": snapped["lattice"]["raw_binary_max_error"],
            "support_atol": snapped["lattice"]["support_atol"],
            "gt": snapped["lattice"]["gt"],
            "binary": snapped["lattice"]["binary"],
            "mmpd": snapped["lattice"]["mmpd"],
            "gt_snap": snapped["lattice"]["gt_snap"],
            "binary_snap": snapped["lattice"]["binary_snap"],
            "mmpd_snap": snapped["lattice"]["mmpd_snap"],
        },
    )

    viz_dir = args.output_dir / "viz" / run_name
    viz_paths = _write_zoom_viz(
        out_dir=viz_dir,
        run_name=run_name,
        dataset=dataset,
        snapped=snapped,
        n_windows=int(args.viz_n_windows),
        variate=int(args.viz_variate),
        slice_lengths=args.slice_lengths,
        zoom_steps=int(args.viz_zoom_steps),
        seed=int(args.seed),
    )
    print(f"[{run_name}] wrote {len(viz_paths)} viz panels under {viz_dir}", flush=True)
    if args.viz_only:
        return {"kind": kind, "viz": [str(p) for p in viz_paths], "metrics": {}}

    bundle = SimpleNamespace(
        fakes={"binary_staged": snapped["binary"], "mmpd": snapped["mmpd"]},
        y_true_by_source={
            "binary_staged": snapped["gt"],
            "mmpd": snapped["gt"].copy(),
        },
        past=snapped["past"],
        legal_levels=snapped["legal_levels"],
        indices=snapped["indices"],
        series_starts=snapped["series_starts"],
        run=run,
        pack_splits=[str(x) for x in binary_pack["pack_splits"].tolist()],
    )
    splits = split_windows(
        len(snapped["gt"]),
        args,
        dataset,
        indices=bundle.indices,
        lookback=args.lookback,
        horizon=args.horizon,
        test_stride=int(args.pack_test_stride),
        series_starts=bundle.series_starts,
    )
    metrics: Dict[str, Any] = {"kind": kind}
    for source in ("binary_staged", "mmpd"):
        per_len: Dict[str, Any] = {}
        for length in args.slice_lengths:
            if int(length) <= args.horizon:
                per_len[str(int(length))] = train_classifier(
                    args, dataset, source, int(length), bundle, splits, device,
                )
        write_json(args.output_dir / "partials" / f"{run_name}__{dataset}__{source}.json", per_len)
        metrics[source] = per_len
    return {"kind": kind, "viz": [str(p) for p in viz_paths], "metrics": metrics}


def main() -> None:
    args = parse_args()
    apply_smoke(args)
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "partials").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "raw").mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"device={device} viz_only={args.viz_only} smoke={args.smoke_test}", flush=True)

    summary: Dict[str, Any] = {}
    for spec in _parse_run_specs(args.runs):
        ckpt = Path(spec["ckpt"])
        if not ckpt.is_absolute():
            ckpt = REPO_ROOT / ckpt
        summary[spec["name"]] = run_one(
            args,
            run_name=spec["name"],
            ckpt_root=ckpt,
            config_path=spec["config"],
            device=device,
        )
    write_json(args.output_dir / "summary.json", summary)
    # Flat AUROC table for the two ablations vs GT.
    rows = []
    for name, payload in summary.items():
        for source, per_len in (payload.get("metrics") or {}).items():
            if not isinstance(per_len, dict):
                continue
            for L, mets in per_len.items():
                if isinstance(mets, dict) and "disc_auroc" in mets:
                    rows.append(
                        {
                            "run": name,
                            "kind": payload.get("kind"),
                            "source": source,
                            "L": int(L),
                            "disc_auroc": float(mets["disc_auroc"]),
                            "disc_acc": float(mets.get("disc_acc", float("nan"))),
                        }
                    )
    if rows:
        write_json(args.output_dir / "auroc_table.json", rows)
        print("\nAUROC table:", flush=True)
        for row in rows:
            print(
                f"  {row['run']:16s} {row['source']:14s} L{row['L']:<3d} "
                f"auroc={row['disc_auroc']:.4f} acc={row['disc_acc']:.4f}",
                flush=True,
            )
    print(f"\ndone → {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
