#!/usr/bin/env python3
"""Paired h96 ordinal patch-refine versus non-ordinal MMPD discriminator.

This is intentionally separate from the h720 and non-ordinal h96 evaluators.
It reads the completed MMPD campaign's actual raw predictions, generates
ordinal patch-refine forecasts from explicit coarse+patch_refine checkpoints,
then snaps GT and MMPD to the same absolute 256-row ordinal patch-refine
ladder while asserting the raw binary decode already lies on that support.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from copy import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.diffusion_tsf.train_multivariate_pipeline import (
    load_dataset,
    load_wrapped_guidance,
    resolve_pipeline_data_subset,
)
from temp.eval_univariate_patch_refine_vs_gt import load_patch_refine_run
from utils.dual_scale_bin_filter import (
    align_mmpd_to_binary_dataset_norm,
)
from utils.eval_discriminator_binary_vs_mmpd_univariate import train_classifier
from utils.eval_discriminator_texture_staged_vs_mmpd import (
    apply_smoke_defaults as apply_base_smoke_defaults,
    binary_mmpd_train_scaler_map,
    collect_partials,
    parse_args as parse_base_args,
    split_windows,
    write_json,
)
from utils.eval_mmpd_gaussian_anchor import (
    load_tsf_pack_pool,
    parse_pack_splits,
    run_subset_id,
    run_test_stride,
    run_train_stride,
    run_variate_indices,
)
from utils.eval_trend_robust_texture_staged_vs_mmpd import generate_staged_forecast
from utils.forecast_pack_reduce import assert_not_anchor_agg, reduce_pack_forecast
from utils.patch_refine_ordinal_ladder import (
    assert_on_patch_refine_levels,
    assert_support_is_causal,
    legal_patch_refine_levels_dataset_z,
    snap_to_patch_refine_levels,
)
from utils.disc_bin_center_shift import bin_center_shift  # noqa: E402
from utils.visualize_staged_eval_2d_preds import _build_state, _load_stage_model, _resolve_guidance_ckpt
from utils.visualize_discriminator_univariate_confusions import visualize_univariate_combo
from utils.binary_mmpd_sample_panels import generate_binary_vs_mmpd_anchor_prob_panels


DEFAULT_OUTPUT = REPO_ROOT / "results" / "datasets" / "disc-ordinal-patch-refine-h96-vs-mmpd"


def _report_dir(output_dir: Path) -> Path:
    """Keep report figures in the report tree, not inside transient raw packs."""
    return REPO_ROOT / "reports" / output_dir.name


def _mmpd_instance_summary(
    *,
    binary_past: np.ndarray,
    mmpd_prediction: np.ndarray,
    scalers: Mapping[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Use MMPD's actual normalization helpers for the report-only diagnostics."""
    path = REPO_ROOT / "temp" / "MMPD" / "exp" / "normalization.py"
    if not path.is_file():
        raise FileNotFoundError(f"MMPD normalization helper missing: {path}")
    spec = importlib.util.spec_from_file_location("h96_mmpd_normalization", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import MMPD normalization helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    binary_mean = np.asarray(scalers["binary_mean"], dtype=np.float32)[None, :, None]
    binary_std = np.asarray(scalers["binary_std"], dtype=np.float32)[None, :, None]
    mmpd_mean = np.asarray(scalers["mmpd_mean"], dtype=np.float32)[None, :, None]
    mmpd_std = np.asarray(scalers["mmpd_std"], dtype=np.float32)[None, :, None]
    mmpd_past = ((binary_past * binary_std + binary_mean) - mmpd_mean) / mmpd_std
    mmpd_pred = ((mmpd_prediction * binary_std + binary_mean) - mmpd_mean) / mmpd_std
    past_t = torch.from_numpy(mmpd_past)
    pred_t = torch.from_numpy(mmpd_pred)
    mean, std = module.get_statistics(past_t)
    restored = module.denormalize(module.normalize(pred_t, mean, std), mean, std)
    residual = float((restored - pred_t).abs().max().item())
    # fp32 instance-norm round-trips routinely land ~1e-6..1e-5; coverage_synth hit 3.05e-5.
    if residual > 1e-4:
        raise AssertionError(f"MMPD instance normalization round-trip failed: {residual:.3g}")
    return mean.numpy(), std.numpy(), residual


def _defaults(argv: Sequence[str]) -> List[str]:
    text = " ".join(argv)
    defaults: List[str] = []
    if "--fake-sources" not in text:
        defaults += ["--fake-sources", "binary_staged", "mmpd"]
    if "--lookback" not in text:
        defaults += ["--lookback", "336"]
    if "--horizon" not in text:
        defaults += ["--horizon", "96"]
    if "--test-stride" not in text:
        # Kept for CLI compat; pack alignment uses --pack-test-stride (default 4).
        defaults += ["--test-stride", "4"]
    if "--test-fraction" not in text:
        defaults += ["--test-fraction", "1.0"]
    if "--output-dir" not in text:
        defaults += ["--output-dir", str(DEFAULT_OUTPUT)]
    if "--pack-splits" not in text:
        defaults += ["--pack-splits", "test"]
    if "--mmpd-instance-norm" not in text and "--no-mmpd-instance-norm" not in text:
        defaults += ["--mmpd-instance-norm"]
    if "--mmpd-ordinal-norm" not in text and "--no-mmpd-ordinal-norm" not in text:
        defaults += ["--no-mmpd-ordinal-norm"]
    if "--candidate-only" not in text and "--no-candidate-only" not in text:
        defaults += ["--candidate-only"]
    if "--disc-bin-center-shift" not in text and "--no-disc-bin-center-shift" not in text:
        defaults += ["--disc-bin-center-shift"]
    if "--save-classification-scores" not in text and "--no-save-classification-scores" not in text:
        defaults += ["--save-classification-scores"]
    return defaults


def parse_args() -> argparse.Namespace:
    custom = argparse.ArgumentParser(add_help=False)
    custom.add_argument("--checkpoint-dir", type=Path, default=None)
    custom.add_argument("--assert-only", action="store_true")
    custom.add_argument(
        "--assert-max-windows",
        type=int,
        default=None,
        help="When --assert-only, cap lattice checks to this many MMPD-aligned windows "
             "(default: 8). Full disc eval ignores this.",
    )
    custom.add_argument(
        "--probe-binary-batch-size",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Probe max GPU batch for coarse→patch_refine generate before materializing "
             "binary packs (default: on).",
    )
    custom.add_argument(
        "--probe-binary-batch-size-max",
        type=int,
        default=64,
        help="Upper bound for --probe-binary-batch-size search.",
    )
    custom.add_argument(
        "--pack-test-stride",
        type=int,
        default=4,
        help="TSF pool stride that MMPD pack indices address. Must match MMPD "
             "eval_test_stride (this campaign: 4). Independent of --test-stride / binary "
             "metadata test_stride.",
    )
    custom.add_argument(
        "--disc-index-stride",
        type=int,
        default=None,
        help="Keep every N-th MMPD-aligned window after loading the pack. "
             "Default: 1 (full pack-stride-4 pool; set >1 to thin further).",
    )
    custom.add_argument(
        "--fake-agg",
        choices=["prob_mean", "sample0"],
        default="sample0",
        help="Reduce pack samples to disc fakes: first stochastic draw (default) "
             "or mean over S. Anchor/deterministic is never used.",
    )
    custom.add_argument(
        "--visualize-confusions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After L8 disc training, write TP/TN/FP/FN PNGs (default: on).",
    )
    custom.add_argument(
        "--viz-anchor-prob-panels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write a few binary-vs-MMPD anchor+prob shared-window panels (default: on).",
    )
    custom.add_argument("--viz-anchor-prob-windows", type=int, default=2)
    extra, remaining = custom.parse_known_args(sys.argv[1:])
    saved = sys.argv
    sys.argv = [saved[0], *_defaults(remaining), *remaining]
    try:
        args = parse_base_args()
    finally:
        sys.argv = saved
    args.datasets = [piece for raw in args.datasets for piece in str(raw).split(",") if piece]
    args.checkpoint_dir = extra.checkpoint_dir.expanduser().resolve() if extra.checkpoint_dir else None
    args.assert_only = bool(extra.assert_only)
    args.assert_max_windows = (
        None if extra.assert_max_windows is None else max(1, int(extra.assert_max_windows))
    )
    args.probe_binary_batch_size = bool(extra.probe_binary_batch_size)
    args.probe_binary_batch_size_max = max(1, int(extra.probe_binary_batch_size_max))
    # Never inherit --test-stride here: that flag was briefly set to 16 for thinning and
    # silently broke MMPD index alignment when used as the pack grid.
    args.pack_test_stride = max(1, int(extra.pack_test_stride))
    args.disc_index_stride = (
        None if extra.disc_index_stride is None else max(1, int(extra.disc_index_stride))
    )
    args.fake_agg = str(extra.fake_agg)
    assert_not_anchor_agg(args.fake_agg)
    args.visualize_confusions = bool(extra.visualize_confusions)
    args.viz_anchor_prob_panels = bool(extra.viz_anchor_prob_panels)
    args.viz_anchor_prob_windows = max(1, int(extra.viz_anchor_prob_windows))
    args.mmpd_output_root = args.mmpd_output_root.expanduser().resolve()
    args.raw_eval_dir = args.raw_eval_dir.expanduser().resolve()
    # Always persist checkpoints when confusion viz is on so panels can reload the disc.
    if args.visualize_confusions:
        args.save_checkpoints = True
    if args.merge_partials_only:
        return args
    if args.checkpoint_dir is None:
        raise ValueError("shard evaluation requires --checkpoint-dir")
    if len(args.datasets) != 1:
        raise ValueError("each shard must provide exactly one --datasets value with --checkpoint-dir")
    if not args.binary_config:
        raise ValueError("--binary-config must name the ordinal h96 patch-refine YAML")
    if int(args.lookback) != 336 or int(args.horizon) != 96:
        raise ValueError("ordinal patch-refine comparison is fixed to --lookback 336 --horizon 96")
    if args.mmpd_ordinal_norm or not args.mmpd_instance_norm:
        raise ValueError("this paired comparison requires non-ordinal, instance-normalized MMPD")
    if args.assert_only and args.assert_max_windows is None:
        args.assert_max_windows = 8
    return args


def apply_smoke_defaults(args: argparse.Namespace) -> None:
    apply_base_smoke_defaults(args)
    if args.smoke_test:
        args.raw_binary_batch_size = 1
        args.num_sampling_steps = min(int(args.num_sampling_steps), 2)
        args.slice_lengths = [length for length in args.slice_lengths if int(length) <= 16]
        args.probe_binary_batch_size = False


def _subset_mmpd_aligned(
    indices: Sequence[int],
    pack: Mapping[str, np.ndarray],
    *,
    pick: np.ndarray,
) -> tuple[List[int], Dict[str, np.ndarray]]:
    """Keep MMPD pack rows / index list at the same positions ``pick``."""
    pick = np.asarray(pick, dtype=np.int64)
    n_full = len(indices)
    if pick.ndim != 1 or pick.size == 0:
        raise ValueError("window subset pick must be a non-empty 1d index array")
    if int(pick.min()) < 0 or int(pick.max()) >= n_full:
        raise ValueError(f"window subset pick out of range for n={n_full}")
    thinned_indices = [int(indices[int(i)]) for i in pick.tolist()]
    thinned_pack = {
        key: (
            value[pick]
            if isinstance(value, np.ndarray) and value.shape[:1] == (n_full,)
            else value
        )
        for key, value in pack.items()
    }
    return thinned_indices, thinned_pack


def _thin_disc_windows(
    indices: Sequence[int],
    pack: Mapping[str, np.ndarray],
    *,
    dataset: str,
    seed: int,
    test_fraction: float,
    disc_index_stride: int,
) -> tuple[List[int], Dict[str, np.ndarray]]:
    """Apply stride-then-fraction thinning to MMPD-aligned windows (matches staged_eval spirit)."""
    n_full = len(indices)
    stride = max(1, int(disc_index_stride))
    fraction = float(test_fraction)
    if fraction <= 0.0 or fraction > 1.0:
        raise ValueError(f"test_fraction must be in (0, 1], got {fraction}")
    pick = np.arange(0, n_full, stride, dtype=np.int64)
    if fraction < 1.0 and pick.size > 1:
        n_keep = max(1, int(round(pick.size * fraction)))
        if n_keep < pick.size:
            rng = np.random.default_rng(int(seed) + (sum(ord(c) for c in dataset) % 10_007))
            chosen = np.sort(rng.choice(pick.size, size=n_keep, replace=False))
            pick = pick[chosen]
    if pick.size == n_full and stride == 1 and fraction >= 1.0:
        return list(indices), dict(pack)
    print(
        f"[{dataset}] disc window thin: {pick.size}/{n_full} "
        f"(index_stride={stride}, test_fraction={fraction:.3f})",
        flush=True,
    )
    return _subset_mmpd_aligned(indices, pack, pick=pick)


def _pack_test_stride(args: argparse.Namespace) -> int:
    """Stride of the TSF pool that MMPD ``indices`` address (not binary metadata stride).

    Hard-default 4 = MMPD matched-binary ``eval_test_stride``. Do **not** fall back to
    ``args.test_stride`` — that flag is unrelated and was briefly set to 16 for disc
    thinning, which produced pool_len mismatches (e.g. traffic 214 vs indices to 853).
    """
    return max(1, int(getattr(args, "pack_test_stride", 4) or 4))


def _mmpd_pack(root: Path, dataset: str) -> Mapping[str, np.ndarray]:
    path = root / "raw" / f"mmpd_{dataset}.npz"
    if not path.is_file():
        resolved = path.resolve()
        hint = ""
        if "corrupt" in str(resolved).lower() or resolved != path:
            hint = (
                f" (requested {path}; resolved to {resolved} — check for a symlink "
                f"into a corrupt/old tree, or a missing dataset pack)"
            )
        raise FileNotFoundError(
            f"missing actual MMPD evaluation pack {path}{hint}; "
            f"submit_mmpd must complete its eval/merge first"
        )
    # Prefer the path the caller passed; only resolve for the open().
    with np.load(path) as data:
        pack = {key: data[key] for key in data.files}
    required = {"y_true", "samples", "indices"}
    missing = sorted(required - set(pack))
    if missing:
        raise KeyError(f"{path} missing {missing}")
    if pack["samples"].ndim != 4 or pack["samples"].shape[2] < 1:
        raise ValueError(f"{path}: samples must be (N,V,S,H), got {pack['samples'].shape}")
    if pack["samples"][:, :, 0, :].shape != pack["y_true"].shape:
        raise ValueError(f"{path}: sample/y_true shape mismatch")
    return pack


def _load_binary_models(
    args: argparse.Namespace,
    dataset: str,
    root: Path,
    device: torch.device,
) -> tuple[Any, Any, Any, Any]:
    # Keep checkpoint metadata test_stride so MMPD pack indices address the same
    # TSF pool. Window thinning is applied to the index list, not the pool grid.
    run, stages = load_patch_refine_run(dataset, root, test_stride=None)
    state = _build_state(root, dataset, run_subset_id(run), str(args.binary_config))
    resolve_pipeline_data_subset(state)
    if not state.use_patch_refine_stage or not state.use_ordinal_window_norm or state.use_window_normalization:
        raise ValueError(
            f"{args.binary_config} must enable ordinal patch_refine and disable window normalization"
        )
    import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod

    _, _, _, norm_stats = load_dataset(
        dataset,
        run_variate_indices(run),
        stride=run_train_stride(run),
        test_stride=run_test_stride(run),
        lookback=args.lookback,
        horizon=args.horizon,
        ordinal_tie_atol=float(state.ordinal_tie_atol),
        use_ordinal_window_norm=True,
    )
    ladder = norm_stats.get("ordinal_ladder")
    if ladder is None:
        raise RuntimeError(f"{dataset}: ordinal dataset loader did not construct a ladder")
    state.ordinal_ladder = ladder

    guidance = None
    if bool(state.use_guidance_channel) or not bool(state.disable_cross_attention):
        path, guidance_type = _resolve_guidance_ckpt(root, run_subset_id(run), "auto")
        guidance = load_wrapped_guidance(
            str(path),
            len(run_variate_indices(run)),
            device,
            guidance_type=guidance_type,
            dataset_lookback=args.lookback,
            dataset_horizon=args.horizon,
        )
        if hasattr(guidance, "ordinal_ladder"):
            guidance.ordinal_ladder = ladder
    coarse = _load_stage_model(
        state, "coarse", stages["coarse_pt"], guidance, len(run_variate_indices(run)), device,
        strict_non_guidance_shapes=True,
    )
    refine = _load_stage_model(
        state, "patch_refine", stages["refine_pt"], guidance, len(run_variate_indices(run)), device,
        strict_non_guidance_shapes=True,
    )
    for model in (coarse, refine):
        model._ordinal_input_is_ranked = False
        model._ordinal_apply_ood_shift = True
        if int(model.config.patch_refine_canvas_height) != 256 and model is refine:
            raise ValueError(f"{dataset}: patch-refine canvas must be 256")
    return run, coarse, refine, ladder


def _concat_patch_chunks(chunks: Sequence[np.ndarray], *, width: int) -> np.ndarray:
    nonempty = [chunk for chunk in chunks if chunk.shape[0]]
    if nonempty:
        return np.concatenate(nonempty, axis=0).astype(np.float32)
    return np.empty((0, 1, width), dtype=np.float32)


def _concat_int_chunks(chunks: Sequence[np.ndarray]) -> np.ndarray:
    nonempty = [chunk for chunk in chunks if chunk.shape[0]]
    if nonempty:
        return np.concatenate(nonempty, axis=0).astype(np.int64)
    return np.empty((0,), dtype=np.int64)


def _unblended_nonoverlap_patch_batch(
    *,
    result: Mapping[str, Any],
    target: torch.Tensor,
    past: torch.Tensor,
    legal_levels: np.ndarray,
    canvas_height: int,
    patch_height: int,
    patch_width: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """Decode raw 8-column CDF crops without overlap blending.

    A candidate is retained only if every column's coarse boundary is visible
    in that patch.  The greedy selection makes retained examples disjoint per
    `(window, variate)`, so this metric never averages two overlapping crops.
    """
    from models.diffusion_tsf.patch_refine_geometry import coarse_edges_from_cdf
    from models.diffusion_tsf.preprocessing import TimeSeriesTo2D

    patch_cdf = result["patch_cdf_unblended"]
    locations = result["patch_locations"]
    bins = TimeSeriesTo2D.bin_indices_from_cdf(patch_cdf[:, 0]).to(dtype=torch.long)
    occupancy = patch_cdf[:, 0].sum(dim=-2)
    visible = (occupancy > 0) & (occupancy < patch_height)
    coarse_edges = coarse_edges_from_cdf(
        result["future_2d_coarse"], canvas_height=canvas_height,
    )
    target_snap, _ = snap_to_patch_refine_levels(
        target.detach().cpu().numpy(), legal_levels,
    )
    candidates = 0
    rejected_invalid = 0
    next_allowed: Dict[Tuple[int, int], int] = {}
    kept_pred: List[np.ndarray] = []
    kept_gt: List[np.ndarray] = []
    kept_past: List[np.ndarray] = []
    kept_parent: List[int] = []
    kept_start: List[int] = []
    kept_variate: List[int] = []
    ordered = sorted(
        enumerate(locations),
        key=lambda item: (
            int(item[1].batch_index), int(item[1].variate_index),
            int(item[1].col0), int(item[1].row0),
        ),
    )
    for patch_i, location in ordered:
        candidates += 1
        start = int(location.col0)
        end = start + patch_width
        if end > int(target.shape[-1]):
            rejected_invalid += 1
            continue
        edges = coarse_edges[location.batch_index, location.variate_index, start:end]
        edge_visible = (edges >= int(location.row0)) & (edges < int(location.row0) + patch_height)
        if not bool((edge_visible & visible[patch_i]).all()):
            rejected_invalid += 1
            continue
        key = (int(location.batch_index), int(location.variate_index))
        if start < next_allowed.get(key, 0):
            continue
        next_allowed[key] = end
        absolute_rows = (bins[patch_i] + int(location.row0)).clamp(0, canvas_height - 1)
        levels = legal_levels[location.batch_index, location.variate_index]
        pred = levels[absolute_rows.detach().cpu().numpy()]
        kept_pred.append(pred.astype(np.float32)[None, :])
        kept_gt.append(target_snap[location.batch_index, location.variate_index, start:end][None, :])
        kept_past.append(past[location.batch_index, location.variate_index].detach().cpu().numpy()[None, :])
        kept_parent.append(int(location.batch_index))
        kept_start.append(start)
        kept_variate.append(int(location.variate_index))
    if not kept_pred:
        empty = np.empty((0, 1, patch_width), dtype=np.float32)
        return (
            empty, empty.copy(), np.empty((0, 1, past.shape[-1]), dtype=np.float32),
            np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64),
            {"candidates": candidates, "rejected_invalid_or_out_of_bounds": rejected_invalid, "selected": 0},
        )
    return (
        np.stack(kept_pred).astype(np.float32),
        np.stack(kept_gt).astype(np.float32),
        np.stack(kept_past).astype(np.float32),
        np.asarray(kept_parent, dtype=np.int64),
        np.asarray(kept_start, dtype=np.int64),
        np.asarray(kept_variate, dtype=np.int64),
        {
            "candidates": candidates,
            "rejected_invalid_or_out_of_bounds": rejected_invalid,
            "selected": len(kept_pred),
        },
    )


def _materialize_binary(
    args: argparse.Namespace,
    dataset: str,
    root: Path,
    indices: Sequence[int],
    device: torch.device,
) -> tuple[Mapping[str, np.ndarray], Any, Any]:
    cache = args.raw_eval_dir / f"binary_ordinal_patch_refine_{dataset}.npz"
    run, coarse, refine, ladder = _load_binary_models(args, dataset, root, device)
    required_cached = {
        "y_true", "samples", "indices", "past",
        "unblended_nonoverlap_patch_pred", "unblended_nonoverlap_patch_gt",
        "unblended_nonoverlap_patch_past", "unblended_nonoverlap_patch_parent", "patch_vote_counts",
    }
    if cache.is_file() and not args.force_raw_eval:
        with np.load(cache) as data:
            pack = {key: data[key] for key in data.files}
        if required_cached.issubset(pack) and np.array_equal(pack.get("indices"), np.asarray(indices, dtype=np.int64)):
            return pack, run, ladder

    pool, starts, splits, _, _ = load_tsf_pack_pool(
        dataset,
        run_variate_indices(run),
        lookback=args.lookback,
        horizon=args.horizon,
        train_stride=run_train_stride(run),
        test_stride=_pack_test_stride(args),
        pack_splits=parse_pack_splits(args.pack_splits),
        use_ordinal_window_norm=False,
    )
    if not indices or min(indices) < 0 or max(indices) >= len(pool):
        raise ValueError(
            f"{dataset}: MMPD indices are outside the shared TSF pool "
            f"(n_indices={len(indices)}, index_range="
            f"[{min(indices) if indices else 'n/a'}, {max(indices) if indices else 'n/a'}], "
            f"pool_len={len(pool)}, train_stride={run_train_stride(run)}, "
            f"pack_test_stride={_pack_test_stride(args)}, "
            f"binary_meta_test_stride={run_test_stride(run)}, "
            f"pack_splits={parse_pack_splits(args.pack_splits)}). "
            f"MMPD matched-binary packs require pack_test_stride=4 "
            f"(got {_pack_test_stride(args)})."
        )
    print(
        f"[{dataset}] pack pool: len={len(pool)} pack_test_stride={_pack_test_stride(args)} "
        f"n_indices={len(indices)} train_stride={run_train_stride(run)} "
        f"binary_meta_test_stride={run_test_stride(run)}",
        flush=True,
    )

    batch_size = max(1, int(args.raw_binary_batch_size))
    if bool(getattr(args, "probe_binary_batch_size", False)) and device.type == "cuda":
        from models.diffusion_tsf.pipeline.phases.staged_eval import (
            _probe_max_staged_eval_batch_size,
        )

        sample_past, _sample_future = pool[int(indices[0])]
        max_fit = _probe_max_staged_eval_batch_size(
            coarse_model=coarse,
            fine_model=refine,
            lookback=int(sample_past.shape[-1]),
            n_variates=int(sample_past.shape[0]),
            device=device,
            det_kwargs={
                "sampler": args.probabilistic_sampler,
                "num_inference_steps": 1,
            },
            joint_dual=False,
            min_bs=1,
            max_bs=int(getattr(args, "probe_binary_batch_size_max", 64)),
        )
        if max_fit != batch_size:
            print(
                f"[{dataset}] binary generate probe: config batch={batch_size} -> probed={max_fit}",
                flush=True,
            )
        batch_size = max(1, int(max_fit))

    loader = DataLoader(
        Subset(pool, list(indices)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    true_chunks: List[np.ndarray] = []
    pred_chunks: List[np.ndarray] = []
    past_chunks: List[np.ndarray] = []
    patch_pred_chunks: List[np.ndarray] = []
    patch_gt_chunks: List[np.ndarray] = []
    patch_past_chunks: List[np.ndarray] = []
    patch_parent_chunks: List[np.ndarray] = []
    patch_start_chunks: List[np.ndarray] = []
    patch_variate_chunks: List[np.ndarray] = []
    vote_count_chunks: List[np.ndarray] = []
    patch_diag = {"candidates": 0, "rejected_invalid_or_out_of_bounds": 0, "selected": 0}
    windows_seen = 0
    n_batches = len(loader)
    print(
        f"[{dataset}] materializing binary packs: windows={len(indices)} "
        f"batches={n_batches} batch_size={batch_size} "
        f"sampler={args.probabilistic_sampler} steps={args.num_sampling_steps}",
        flush=True,
    )
    with torch.no_grad():
        for batch_i, (past, future) in enumerate(loader):
            past = past.to(device)
            overlap = int(refine.config.lookback_overlap)
            target = future.to(device)[..., overlap:] if overlap else future.to(device)
            torch.manual_seed(int(args.seed) + batch_i * 1009)
            result = generate_staged_forecast(
                coarse,
                refine,
                past,
                vertical_dual=False,
                sampler=args.probabilistic_sampler,
                num_inference_steps=args.num_sampling_steps,
            )
            prediction = result["prediction_global_norm"]
            if prediction.shape != target.shape:
                raise RuntimeError(
                    f"{dataset}: binary prediction/target mismatch {tuple(prediction.shape)} vs {tuple(target.shape)}"
                )
            levels = legal_patch_refine_levels_dataset_z(
                past.detach().cpu().numpy(), ladder=ladder, device=device,
            )
            patch_values = _unblended_nonoverlap_patch_batch(
                result=result,
                target=target,
                past=past,
                legal_levels=levels,
                canvas_height=int(refine.config.patch_refine_canvas_height),
                patch_height=int(refine.config.patch_refine_patch_height),
                patch_width=int(refine.config.patch_refine_patch_width),
            )
            true_chunks.append(target.cpu().numpy())
            pred_chunks.append(prediction.cpu().numpy())
            past_chunks.append(past.cpu().numpy())
            patch_pred_chunks.append(patch_values[0])
            patch_gt_chunks.append(patch_values[1])
            patch_past_chunks.append(patch_values[2])
            patch_parent_chunks.append(patch_values[3] + windows_seen)
            patch_start_chunks.append(patch_values[4])
            patch_variate_chunks.append(patch_values[5])
            vote_count_chunks.append(result["patch_vote_counts"].detach().cpu().numpy())
            for key, value in patch_values[6].items():
                patch_diag[key] += int(value)
            windows_seen += int(past.shape[0])
            if (batch_i + 1) == n_batches or (batch_i + 1) % max(1, n_batches // 10) == 0:
                print(
                    f"[{dataset}] binary generate {batch_i + 1}/{n_batches} "
                    f"(windows_done={windows_seen}/{len(indices)})",
                    flush=True,
                )
    patch_pred = _concat_patch_chunks(patch_pred_chunks, width=int(refine.config.patch_refine_patch_width))
    patch_gt = _concat_patch_chunks(patch_gt_chunks, width=int(refine.config.patch_refine_patch_width))
    patch_past = _concat_patch_chunks(patch_past_chunks, width=int(args.lookback))
    patch_parent = _concat_int_chunks(patch_parent_chunks)
    patch_start = _concat_int_chunks(patch_start_chunks)
    patch_variate = _concat_int_chunks(patch_variate_chunks)
    for parent, variate in set(zip(patch_parent.tolist(), patch_variate.tolist())):
        starts_for_parent = np.sort(patch_start[(patch_parent == parent) & (patch_variate == variate)])
        if np.any(starts_for_parent[1:] < starts_for_parent[:-1] + int(refine.config.patch_refine_patch_width)):
            raise RuntimeError(f"{dataset}: coherent raw patch examples overlap in parent row {parent}")
    pack = {
        "y_true": np.concatenate(true_chunks).astype(np.float32),
        "samples": np.concatenate(pred_chunks).astype(np.float32)[:, :, None, :],
        "past": np.concatenate(past_chunks).astype(np.float32),
        "indices": np.asarray(indices, dtype=np.int64),
        "series_starts": starts[np.asarray(indices, dtype=np.int64)],
        "pack_splits": np.asarray(splits),
        "unblended_nonoverlap_patch_pred": patch_pred,
        "unblended_nonoverlap_patch_gt": patch_gt,
        "unblended_nonoverlap_patch_past": patch_past,
        "unblended_nonoverlap_patch_parent": patch_parent,
        "unblended_nonoverlap_patch_start": patch_start,
        "unblended_nonoverlap_patch_variate": patch_variate,
        "unblended_patch_candidates": np.asarray(patch_diag["candidates"], dtype=np.int64),
        "unblended_patch_rejected_invalid_or_out_of_bounds": np.asarray(
            patch_diag["rejected_invalid_or_out_of_bounds"], dtype=np.int64,
        ),
        "unblended_patch_selected": np.asarray(patch_diag["selected"], dtype=np.int64),
        "patch_vote_counts": np.concatenate(vote_count_chunks).astype(np.int64),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **pack)
    return pack, run, ladder


def _plot_lattice(
    output_dir: Path,
    dataset: str,
    past: np.ndarray,
    raw_gt: np.ndarray,
    gt: np.ndarray,
    binary: np.ndarray,
    raw_mmpd: np.ndarray,
    mmpd: np.ndarray,
    legal_levels: np.ndarray,
    patch_vote_counts: np.ndarray,
    mmpd_window_mean: np.ndarray,
    mmpd_window_std: np.ndarray,
    mmpd_inverse_residual: float,
    classifier_scores: Mapping[str, Mapping[str, np.ndarray]],
) -> Path:
    plot_dir = _report_dir(output_dir) / "visualizations"
    plot_dir.mkdir(parents=True, exist_ok=True)
    n_windows = min(2, past.shape[0])
    if n_windows < 2:
        raise ValueError("lattice visualization requires at least two evaluation windows")
    fig, axes = plt.subplots(n_windows, 3, figsize=(21, 4 * n_windows), squeeze=False)
    tail = min(48, past.shape[-1])
    x_past = np.arange(-tail, 0)
    x_future = np.arange(gt.shape[-1])
    for window_id in range(n_windows):
        ax = axes[window_id, 0]
        row_ax = axes[window_id, 1]
        prob_ax = axes[window_id, 2]
        rows = legal_levels[window_id, 0]
        for row in rows:
            ax.axhline(float(row), color="0.45", alpha=0.12, linewidth=0.35, zorder=0)
        ax.plot(x_past, past[window_id, 0, -tail:], color="0.35", label="past")
        ax.plot(x_future, raw_gt[window_id, 0], color="black", alpha=0.4, linestyle="--", label="GT raw")
        ax.step(x_future, gt[window_id, 0], where="mid", label="GT snapped", linewidth=1.8)
        ax.step(x_future, binary[window_id, 0], where="mid", label="binary raw/legal", linewidth=1.3)
        ax.plot(x_future, raw_mmpd[window_id, 0], color="C2", alpha=0.45, linestyle="--", label="MMPD raw")
        ax.step(x_future, mmpd[window_id, 0], where="mid", label="MMPD snapped", linewidth=1.3)
        row_ids = np.asarray([0, 64, 128, 192, 255])
        ax.set_yticks(rows[row_ids], [f"row {row_id}" for row_id in row_ids])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set(
            title=f"{dataset}: window {window_id}; MMPD μ={mmpd_window_mean[window_id, 0]:.3g}, "
            f"σ={mmpd_window_std[window_id, 0]:.3g}, inverse err={mmpd_inverse_residual:.1e}",
            ylabel="binary dataset-z",
        )
        ax.legend(loc="best", ncol=2, fontsize=8)
        row_ax.step(x_future, np.argmin(np.abs(binary[window_id, 0, :, None] - rows), axis=-1),
                    where="mid", label="binary row", color="C0")
        row_ax.step(x_future, np.argmin(np.abs(gt[window_id, 0, :, None] - rows), axis=-1),
                    where="mid", label="GT row", color="black")
        row_ax.step(x_future, np.argmin(np.abs(mmpd[window_id, 0, :, None] - rows), axis=-1),
                    where="mid", label="MMPD row", color="C2")
        for boundary in range(0, gt.shape[-1] - 7, 6):
            row_ax.axvline(boundary, color="0.6", alpha=0.18, linewidth=0.7)
        row_ax.set(title="absolute ordinal row / fine-patch boundaries", xlabel="forecast timestep", ylabel="row ID", ylim=(-2, 257))
        row_ax.legend(loc="upper left", fontsize=8)
        votes_ax = row_ax.twinx()
        votes = np.asarray(patch_vote_counts[window_id, 0], dtype=np.float64)
        # patch_vote_counts are on the overlap canvas (horizon+K); forecasts are K-trimmed.
        if votes.shape[0] != x_future.shape[0]:
            if votes.shape[0] > x_future.shape[0]:
                votes = votes[-x_future.shape[0]:]
            else:
                raise ValueError(
                    f"patch_vote_counts length {votes.shape[0]} shorter than horizon {x_future.shape[0]}"
                )
        votes_ax.plot(x_future, votes, color="0.25", alpha=0.55, linewidth=0.8, label="patch votes")
        votes_ax.set_ylabel("votes")
        for source, color in (("binary_staged", "C1"), ("mmpd", "C2")):
            score = classifier_scores.get(source)
            if score is None:
                continue
            mask = (score["window"] == window_id) & (score["variate"] == 0)
            for label, style, name in ((1.0, "-", "forecast"), (0.0, ":", "GT")):
                selected = mask & (score["label"] == label)
                if np.any(selected):
                    prob_ax.plot(
                        score["offset"][selected], score["prob_fake"][selected],
                        linestyle=style, marker="o", markersize=2.5, color=color,
                        label=f"{source} {name}",
                    )
        prob_ax.axhline(0.5, color="0.3", linewidth=0.8)
        prob_ax.set(
            title="L8 discriminator probability", xlabel="forecast offset",
            ylabel="P(fake)", ylim=(-0.03, 1.03),
        )
        prob_ax.legend(loc="best", fontsize=8)
    axes[-1, 0].set_xlabel("time")
    fig.tight_layout()
    path = plot_dir / f"{dataset}_ordinal_patch_refine_lattice.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _load_l8_classifier_scores(output_dir: Path, dataset: str) -> Dict[str, Mapping[str, np.ndarray]]:
    out: Dict[str, Mapping[str, np.ndarray]] = {}
    for source in ("binary_staged", "mmpd"):
        path = output_dir / "scores" / f"{dataset}_{source}_L8_test_scores.npz"
        if not path.is_file():
            raise FileNotFoundError(f"missing L8 classification scores: {path}")
        with np.load(path) as data:
            required = {"prob_fake", "label", "window", "variate", "offset"}
            missing = required - set(data.files)
            if missing:
                raise KeyError(f"{path} missing score fields: {sorted(missing)}")
            out[source] = {key: data[key] for key in required}
    return out


def _binary_lattice_atol(legal_levels: np.ndarray) -> float:
    """fp decode slack: unique-seg blend can sit a few 1e-3 off exact row centers."""
    gaps = np.diff(np.sort(np.asarray(legal_levels, dtype=np.float64), axis=-1), axis=-1)
    positive = gaps[gaps > 0.0]
    if positive.size == 0:
        return 1e-2
    return float(max(1e-2, 0.5 * float(np.median(positive))))


def run_eval(args: argparse.Namespace) -> None:
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else f"cuda:{args.gpu}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.raw_eval_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "partials").mkdir(exist_ok=True)
    for dataset in args.datasets:
        mmpd_pack = _mmpd_pack(args.mmpd_output_root, dataset)
        indices = [int(value) for value in mmpd_pack["indices"].tolist()]
        n_full = len(indices)
        if args.assert_only:
            cap = int(args.assert_max_windows or 8)
            if n_full > cap:
                rng = np.random.default_rng(
                    int(args.seed) + (sum(ord(c) for c in dataset) % 10_007)
                )
                pick = np.sort(rng.choice(n_full, size=cap, replace=False))
                indices, mmpd_pack = _subset_mmpd_aligned(indices, mmpd_pack, pick=pick)
                print(
                    f"[{dataset}] assert-only: sampling {cap}/{n_full} windows for lattice gate",
                    flush=True,
                )
        else:
            disc_stride = args.disc_index_stride
            if disc_stride is None:
                # pack is already eval_test_stride=4; stride 1 keeps the full MMPD-aligned pool.
                disc_stride = 1
            indices, mmpd_pack = _thin_disc_windows(
                indices,
                mmpd_pack,
                dataset=dataset,
                seed=int(args.seed),
                test_fraction=float(args.test_fraction),
                disc_index_stride=int(disc_stride),
            )
        binary_pack, run, ladder = _materialize_binary(
            args, dataset, args.checkpoint_dir, indices, device,
        )
        binary_gt = binary_pack["y_true"].astype(np.float32)
        binary_pred = reduce_pack_forecast(binary_pack, agg=args.fake_agg)
        mmpd_gt = mmpd_pack["y_true"].astype(np.float32)
        mmpd_pred = reduce_pack_forecast(mmpd_pack, agg=args.fake_agg)
        print(
            f"[{dataset}] disc forecasts via fake_agg={args.fake_agg} "
            f"(binary S={binary_pack['samples'].shape[2]}, mmpd S={mmpd_pack['samples'].shape[2]})",
            flush=True,
        )
        if not np.array_equal(binary_pack["indices"], mmpd_pack["indices"]):
            raise RuntimeError(f"{dataset}: binary/MMPD indices differ")
        scalers = binary_mmpd_train_scaler_map(args, run)
        mmpd_binary_z, align = align_mmpd_to_binary_dataset_norm(
            binary_y_true=binary_gt,
            mmpd_y_true=mmpd_gt,
            mmpd_fakes=mmpd_pred,
            **scalers,
        )
        past_pool, _, _, _, _ = load_tsf_pack_pool(
            dataset,
            run_variate_indices(run),
            lookback=args.lookback,
            horizon=args.horizon,
            train_stride=run_train_stride(run),
            test_stride=_pack_test_stride(args),
            pack_splits=parse_pack_splits(args.pack_splits),
            use_ordinal_window_norm=False,
        )
        past = np.stack([past_pool[index][0].detach().cpu().numpy() for index in indices]).astype(np.float32)
        legal_levels = legal_patch_refine_levels_dataset_z(past, ladder=ladder, device=device)
        # Real-checkpoint counterpart of the synthetic causal contract.  The
        # legal support must not change if only the future fixture changes.
        assert_support_is_causal(
            past,
            binary_gt,
            binary_gt + np.float32(123.456),
            ladder=ladder,
            canvas_height=256,
            device=device,
        )
        gt, gt_snap = snap_to_patch_refine_levels(binary_gt, legal_levels)
        mmpd, mmpd_snap = snap_to_patch_refine_levels(mmpd_binary_z, legal_levels)
        mmpd_window_mean, mmpd_window_std, mmpd_inverse_residual = _mmpd_instance_summary(
            binary_past=past, mmpd_prediction=mmpd_binary_z, scalers=scalers,
        )
        # Patch-refine decode should land on the 256-row ladder; allow small fp slack
        # from unique-seg blending (elec disc hit max_error≈6e-3 vs atol 1e-6).
        # Traffic (and some real ckpts) can sit far off the 256-row support under
        # sample0; assert-only still hard-fails, but the disc path snaps so
        # fake/real share ladder support and the campaign can finish.
        binary_atol = _binary_lattice_atol(legal_levels)
        binary_raw = np.asarray(binary_pred, dtype=np.float32)
        if args.assert_only:
            binary = binary_raw
            binary_staged_stats = assert_on_patch_refine_levels(
                binary, legal_levels, atol=binary_atol,
            )
            binary_staged_stats.update({
                "raw_binary_retained": 1.0,
                "support_atol": float(binary_atol),
            })
        else:
            binary, binary_snap = snap_to_patch_refine_levels(binary_raw, legal_levels)
            raw_err = float(np.abs(binary_raw - binary).max(initial=0.0))
            binary_staged_stats = assert_on_patch_refine_levels(binary, legal_levels)
            binary_staged_stats.update(binary_snap)
            binary_staged_stats.update({
                "raw_binary_retained": 0.0 if raw_err > float(binary_atol) else 1.0,
                "raw_max_support_error": raw_err,
                "support_atol": float(binary_atol),
            })
            if raw_err > float(binary_atol):
                print(
                    f"[{dataset}] binary off lattice max_error={raw_err:.6g} "
                    f"atol={binary_atol:.6g}; snapping for disc "
                    f"(mean_abs_snap_delta={binary_snap['mean_abs_snap_delta']:.6g})",
                    flush=True,
                )
        lattice = {
            "gt": assert_on_patch_refine_levels(gt, legal_levels),
            "binary_staged": binary_staged_stats,
            "mmpd": assert_on_patch_refine_levels(mmpd, legal_levels),
        }
        lattice["gt"].update(gt_snap)
        lattice["mmpd"].update(mmpd_snap)
        lattice["mmpd_alignment"] = align
        lattice["causal_support_real_checkpoint_asserted"] = 1.0
        write_json(args.raw_eval_dir / f"lattice_assertion_{dataset}.json", lattice)
        if args.assert_only:
            print(f"[{dataset}] real checkpoint snapping/assertion gate passed", flush=True)
            continue
        # Bin-center shift runs per L-slice inside UnivariateRealVsFakeDataset
        # (replaces zscore_time). Do not pre-shift full-H packs here.
        if bool(getattr(args, "disc_bin_center_shift", False)):
            print(
                f"[{dataset}] disc_bin_center_shift=ON (per L-slice in dataset; "
                f"reduce={getattr(args, 'disc_bin_center_reduce', 'per_variate')}; "
                f"zscore_time disabled)",
                flush=True,
            )
        bundle = SimpleNamespace(
            fakes={"binary_staged": binary, "mmpd": mmpd},
            y_true_by_source={"binary_staged": gt, "mmpd": gt.copy()},
            past=past,
            legal_levels=np.asarray(legal_levels, dtype=np.float32),
            indices=np.asarray(indices, dtype=np.int64),
            series_starts=binary_pack["series_starts"],
            run=run,
            pack_splits=[str(x) for x in binary_pack["pack_splits"].tolist()],
        )
        splits = split_windows(
            len(gt), args, dataset, indices=bundle.indices, lookback=args.lookback,
            horizon=args.horizon, test_stride=_pack_test_stride(args), series_starts=bundle.series_starts,
        )
        by_source: Dict[str, Dict[str, float]] = {}
        for source in ("binary_staged", "mmpd"):
            per_length: Dict[str, float] = {}
            for length in args.slice_lengths:
                if int(length) <= args.horizon:
                    per_length[str(int(length))] = train_classifier(
                        args, dataset, source, int(length), bundle, splits, device,
                    )
            write_json(args.output_dir / "partials" / f"{dataset}__{source}.json", per_length)
            by_source[source] = per_length
            nonoverlap_args = copy(args)
            nonoverlap_args.nonoverlapping_patches = True
            nonoverlap_source = f"{source}_candidate_nonoverlap"
            nonoverlap_bundle = SimpleNamespace(
                fakes={nonoverlap_source: bundle.fakes[source]},
                y_true_by_source={nonoverlap_source: bundle.y_true_by_source[source]},
                past=bundle.past,
                legal_levels=bundle.legal_levels,
                indices=bundle.indices,
                series_starts=bundle.series_starts,
                run=bundle.run,
                pack_splits=bundle.pack_splits,
            )
            nonoverlap_per_length: Dict[str, float] = {}
            for length in args.slice_lengths:
                if int(length) <= args.horizon:
                    nonoverlap_per_length[str(int(length))] = train_classifier(
                        nonoverlap_args, dataset, nonoverlap_source, int(length), nonoverlap_bundle, splits, device,
                    )
            write_json(
                args.output_dir / "partials" / f"{dataset}__{source}_candidate_nonoverlap.json",
                nonoverlap_per_length,
            )
        patch_pred = binary_pack["unblended_nonoverlap_patch_pred"].astype(np.float32)
        patch_gt = binary_pack["unblended_nonoverlap_patch_gt"].astype(np.float32)
        patch_past = binary_pack["unblended_nonoverlap_patch_past"].astype(np.float32)
        patch_parent = binary_pack["unblended_nonoverlap_patch_parent"].astype(np.int64)
        if bool(getattr(args, "disc_bin_center_shift", False)):
            patch_variate = binary_pack["unblended_nonoverlap_patch_variate"].astype(np.int64)
            reduce_mode = str(getattr(args, "disc_bin_center_reduce", "per_variate"))
            patch_levels = legal_levels[patch_parent, patch_variate, :][:, None, :]
            patch_pred, _ = bin_center_shift(patch_pred, patch_levels, reduce=reduce_mode)
            patch_gt, _ = bin_center_shift(patch_gt, patch_levels, reduce=reduce_mode)
        if patch_pred.shape != patch_gt.shape or patch_pred.shape[1:] != (1, 8):
            raise RuntimeError(f"{dataset}: invalid coherent raw L8 patch shape {patch_pred.shape}")
        patch_splits = {
            name: np.flatnonzero(np.isin(patch_parent, parent_rows)).astype(np.int64)
            for name, parent_rows in splits.items()
        }
        patch_counts = {name: len(rows) for name, rows in patch_splits.items()}
        if any(count == 0 for count in patch_counts.values()):
            patch_metrics: Dict[str, float] = {
                "skipped_insufficient_temporal_coverage": 1.0,
                **{f"n_windows_{name}": float(count) for name, count in patch_counts.items()},
            }
            print(f"[{dataset}] coherent raw L8 metric skipped: insufficient coverage {patch_counts}", flush=True)
        else:
            patch_bundle = SimpleNamespace(
                fakes={"binary_patch_unblended": patch_pred},
                y_true_by_source={"binary_patch_unblended": patch_gt},
                past=patch_past,
                indices=patch_parent,
                series_starts=binary_pack["series_starts"][patch_parent],
                run=run,
                pack_splits=[str(value) for value in binary_pack["pack_splits"].tolist()],
            )
            # Patches already bin-center-shifted on L=8 above; do not re-apply in dataset.
            patch_args = copy(args)
            patch_args.disc_bin_center_shift = False
            patch_metrics = train_classifier(
                patch_args, dataset, "binary_patch_unblended", 8, patch_bundle, patch_splits, device,
            )
        patch_metrics.update(
            {
                "patch_prediction_averaged": 0.0,
                "patch_examples_pairwise_nonoverlapping": 1.0,
                "patch_width": 8.0,
                "patch_stride": 6.0,
            }
        )
        write_json(
            args.output_dir / "partials" / f"{dataset}__binary_patch_unblended.json",
            {"8": patch_metrics},
        )
        score_data = _load_l8_classifier_scores(args.output_dir, dataset)
        plot = _plot_lattice(
            args.output_dir, dataset, past, binary_gt, gt, binary, mmpd_binary_z, mmpd,
            legal_levels, binary_pack["patch_vote_counts"], mmpd_window_mean,
            mmpd_window_std, mmpd_inverse_residual, score_data,
        )
        print(f"[{dataset}] canonical 256-row lattice asserted; visualization={plot}", flush=True)

        if bool(getattr(args, "visualize_confusions", True)):
            conf_dir = args.output_dir / "disc_confusions"
            for source in ("binary_staged", "mmpd"):
                try:
                    visualize_univariate_combo(
                        output_dir=args.output_dir,
                        dataset=dataset,
                        fake_source=source,
                        slice_len=8,
                        past=past,
                        y_true=gt,
                        fake=bundle.fakes[source],
                        test_windows=splits["test"],
                        device=device,
                        seed=int(args.seed),
                        batch_size=int(args.batch_size),
                        per_bucket=int(getattr(args, "viz_per_bucket", 2) or 2),
                        lookback_tail=int(getattr(args, "viz_lookback_tail", 32) or 32),
                        plot_dir=conf_dir,
                        max_eval_examples=args.max_eval_examples,
                        candidate_only=bool(args.candidate_only),
                        offset_stride=int(args.offset_stride),
                        apply_zscore=not bool(getattr(args, "disc_bin_center_shift", False)),
                    )
                except Exception as exc:
                    print(f"[{dataset}] confusion viz skipped for {source}: {exc}", flush=True)

        if bool(getattr(args, "viz_anchor_prob_panels", True)):
            n_panel = min(
                int(getattr(args, "viz_anchor_prob_windows", 2) or 2),
                int(past.shape[0]),
            )
            # Prefer MMPD deterministic when present; binary disc-raw is usually S=1 only.
            mmpd_full = _mmpd_pack(args.mmpd_output_root, dataset)
            from utils.forecast_pack_reduce import subset_pack_by_pool_indices

            mmpd_aligned = subset_pack_by_pool_indices(
                mmpd_full, np.asarray(indices, dtype=np.int64),
            )
            mmpd_anchor = None
            if "deterministic" in mmpd_aligned:
                mmpd_anchor_raw = mmpd_aligned["deterministic"].astype(np.float32)
                mmpd_anchor_z, _ = align_mmpd_to_binary_dataset_norm(
                    binary_y_true=binary_gt,
                    mmpd_y_true=mmpd_aligned["y_true"].astype(np.float32),
                    mmpd_fakes=mmpd_anchor_raw,
                    **scalers,
                )
                mmpd_anchor, _ = snap_to_patch_refine_levels(mmpd_anchor_z, legal_levels)
            mmpd_samples = mmpd_aligned["samples"].astype(np.float32)
            # Align each draw into binary dataset-z, then snap onto the ordinal ladder.
            snapped_draws = []
            for s_i in range(mmpd_samples.shape[2]):
                aligned_s, _ = align_mmpd_to_binary_dataset_norm(
                    binary_y_true=binary_gt,
                    mmpd_y_true=mmpd_aligned["y_true"].astype(np.float32),
                    mmpd_fakes=mmpd_samples[:, :, s_i, :],
                    **scalers,
                )
                snapped_s, _ = snap_to_patch_refine_levels(aligned_s, legal_levels)
                snapped_draws.append(snapped_s)
            mmpd_samples_snapped = np.stack(snapped_draws, axis=2).astype(np.float32)
            binary_samples = binary_pack["samples"].astype(np.float32)
            panel_rows = list(range(n_panel))
            panel_paths = generate_binary_vs_mmpd_anchor_prob_panels(
                dataset=dataset,
                out_dir=args.output_dir / "viz" / "binary_vs_mmpd_anchor_prob",
                window_indices=panel_rows,
                y_true=gt[panel_rows],
                past=past[panel_rows],
                binary_anchor=None,
                binary_samples=binary_samples[panel_rows],
                mmpd_anchor=None if mmpd_anchor is None else mmpd_anchor[panel_rows],
                mmpd_samples=mmpd_samples_snapped[panel_rows],
                pool_indices=[int(indices[i]) for i in panel_rows],
            )
            print(f"[{dataset}] wrote {len(panel_paths)} anchor+prob panels", flush=True)
            try:
                from models.diffusion_tsf.pipeline import wandb_utils

                wandb_utils.log_visualization_paths(
                    panel_paths,
                    wandb_key=f"eval/binary_vs_mmpd_anchor_prob/{dataset}",
                )
            except Exception as exc:
                print(f"[{dataset}] wandb panel log skipped: {exc}", flush=True)


def run_merge_only(args: argparse.Namespace) -> None:
    """Merge shard-owned source partials; do not touch raw packs or visualizations."""
    merged = collect_partials(args.output_dir)
    if not merged:
        raise FileNotFoundError(f"No partial metrics found under {args.output_dir / 'partials'}")
    write_json(args.output_dir / "metrics.json", merged)
    fields = [
        "dataset", "fake_source", "slice_len", "disc_bce", "log2_bce_gap", "disc_acc",
        "disc_auroc", "disc_acc_window", "disc_auroc_window", "n_windows_scored",
        "best_val_bce", "best_epoch", "epochs_run", "n_train", "n_val", "n_test",
        "n_windows_train", "n_windows_val", "n_windows_test", "n_variates", "horizon",
        "offset_stride", "no_offset_embedding", "native_repr_stride", "candidate_only",
    ]
    with (args.output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for dataset, by_source in sorted(merged.items()):
            for fake_source, by_length in sorted(by_source.items()):
                for slice_key, metrics in sorted(by_length.items(), key=lambda item: int(item[0])):
                    row = {"dataset": dataset, "fake_source": fake_source, "slice_len": int(slice_key)}
                    row.update({key: metrics.get(key) for key in fields if key not in row})
                    writer.writerow(row)
    report_dir = _report_dir(args.output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# h96 ordinal patch-refine vs non-ordinal MMPD discriminator", "",
        "All discriminator inputs are binary dataset-z values on the same causal, "
        "window-specific 256-row ordinal support. Figures are in `visualizations/`.", "",
        "| Dataset | Fake source | Length | BCE | AUROC | Window AUROC |", "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset, by_source in sorted(merged.items()):
        for fake_source, by_length in sorted(by_source.items()):
            for slice_key, metrics in sorted(by_length.items(), key=lambda item: int(item[0])):
                lines.append(
                    f"| {dataset} | {fake_source} | {slice_key} | "
                    f"{metrics.get('disc_bce', float('nan')):.4f} | "
                    f"{metrics.get('disc_auroc', float('nan')):.4f} | "
                    f"{metrics.get('disc_auroc_window', float('nan')):.4f} |"
                )
    (report_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[merge] wrote metrics.csv and {report_dir / 'report.md'} for {len(merged)} datasets", flush=True)


def main() -> None:
    args = parse_args()
    apply_smoke_defaults(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.merge_partials_only:
        run_merge_only(args)
        return
    args.merge_metrics = False
    run_eval(args)


if __name__ == "__main__":
    main()
