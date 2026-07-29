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
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Sequence, Tuple

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
from utils.patch_refine_ordinal_ladder import (
    assert_on_patch_refine_levels,
    legal_patch_refine_levels_dataset_z,
    snap_to_patch_refine_levels,
)
from utils.visualize_staged_eval_2d_preds import _build_state, _load_stage_model, _resolve_guidance_ckpt


DEFAULT_OUTPUT = REPO_ROOT / "results" / "datasets" / "disc-ordinal-patch-refine-h96-vs-mmpd"


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
        defaults += ["--test-stride", "4"]
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
    if "--save-classification-scores" not in text and "--no-save-classification-scores" not in text:
        defaults += ["--save-classification-scores"]
    return defaults


def parse_args() -> argparse.Namespace:
    custom = argparse.ArgumentParser(add_help=False)
    custom.add_argument("--checkpoint-dir", type=Path, default=None)
    extra, remaining = custom.parse_known_args(sys.argv[1:])
    saved = sys.argv
    sys.argv = [saved[0], *_defaults(remaining), *remaining]
    try:
        args = parse_base_args()
    finally:
        sys.argv = saved
    args.datasets = [piece for raw in args.datasets for piece in str(raw).split(",") if piece]
    args.checkpoint_dir = extra.checkpoint_dir.expanduser().resolve() if extra.checkpoint_dir else None
    args.mmpd_output_root = args.mmpd_output_root.expanduser().resolve()
    args.raw_eval_dir = args.raw_eval_dir.expanduser().resolve()
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
    return args


def apply_smoke_defaults(args: argparse.Namespace) -> None:
    apply_base_smoke_defaults(args)
    if args.smoke_test:
        args.raw_binary_batch_size = 1
        args.num_sampling_steps = min(int(args.num_sampling_steps), 2)
        args.slice_lengths = [length for length in args.slice_lengths if int(length) <= 16]


def _mmpd_pack(root: Path, dataset: str) -> Mapping[str, np.ndarray]:
    path = root / "raw" / f"mmpd_{dataset}.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing actual MMPD evaluation pack {path}; submit_mmpd must complete its eval/merge first"
        )
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
    run, stages = load_patch_refine_run(dataset, root, args.test_stride)
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
    state.extra["global_ordinal_ladder"] = ladder
    pipeline_mod.GLOBAL_ORDINAL_LADDER = ladder
    patch_globals(pipeline_mod, state, honor_dataset_windows=True)

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
        "unblended_nonoverlap_patch_past", "unblended_nonoverlap_patch_parent",
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
        test_stride=run_test_stride(run),
        pack_splits=parse_pack_splits(args.pack_splits),
        use_ordinal_window_norm=False,
    )
    if not indices or min(indices) < 0 or max(indices) >= len(pool):
        raise ValueError(f"{dataset}: MMPD indices are outside the shared TSF pool")
    loader = DataLoader(
        Subset(pool, list(indices)),
        batch_size=min(int(args.raw_binary_batch_size), 2),
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
    patch_diag = {"candidates": 0, "rejected_invalid_or_out_of_bounds": 0, "selected": 0}
    windows_seen = 0
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
            for key, value in patch_values[6].items():
                patch_diag[key] += int(value)
            windows_seen += int(past.shape[0])
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
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **pack)
    return pack, run, ladder


def _plot_lattice(
    output_dir: Path,
    dataset: str,
    past: np.ndarray,
    gt: np.ndarray,
    binary: np.ndarray,
    mmpd: np.ndarray,
    legal_levels: np.ndarray,
    classifier_scores: Mapping[str, Mapping[str, np.ndarray]],
) -> Path:
    plot_dir = output_dir / "visualizations"
    plot_dir.mkdir(parents=True, exist_ok=True)
    n_windows = min(2, past.shape[0])
    if n_windows < 2:
        raise ValueError("lattice visualization requires at least two evaluation windows")
    fig, axes = plt.subplots(n_windows, 2, figsize=(16, 4 * n_windows), squeeze=False)
    tail = min(48, past.shape[-1])
    x_past = np.arange(-tail, 0)
    x_future = np.arange(gt.shape[-1])
    for window_id in range(n_windows):
        ax = axes[window_id, 0]
        prob_ax = axes[window_id, 1]
        rows = legal_levels[window_id, 0]
        for row in rows:
            ax.axhline(float(row), color="0.45", alpha=0.12, linewidth=0.35, zorder=0)
        ax.plot(x_past, past[window_id, 0, -tail:], color="0.35", label="past")
        ax.step(x_future, gt[window_id, 0], where="mid", label="GT snapped", linewidth=1.8)
        ax.step(x_future, binary[window_id, 0], where="mid", label="binary raw/legal", linewidth=1.3)
        ax.step(x_future, mmpd[window_id, 0], where="mid", label="MMPD snapped", linewidth=1.3)
        row_ids = np.asarray([0, 64, 128, 192, 255])
        ax.set_yticks(rows[row_ids], [f"row {row_id}" for row_id in row_ids])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set(title=f"{dataset}: ordinal window {window_id} (256 legal rows)", ylabel="ordinal row ID")
        ax.legend(loc="best", ncol=2)
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


def run_eval(args: argparse.Namespace) -> None:
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else f"cuda:{args.gpu}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.raw_eval_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "partials").mkdir(exist_ok=True)
    for dataset in args.datasets:
        mmpd_pack = _mmpd_pack(args.mmpd_output_root, dataset)
        indices = [int(value) for value in mmpd_pack["indices"].tolist()]
        binary_pack, run, ladder = _materialize_binary(
            args, dataset, args.checkpoint_dir, indices, device,
        )
        binary_gt = binary_pack["y_true"].astype(np.float32)
        binary_pred = binary_pack["samples"][:, :, 0, :].astype(np.float32)
        mmpd_gt = mmpd_pack["y_true"].astype(np.float32)
        mmpd_pred = mmpd_pack["samples"][:, :, 0, :].astype(np.float32)
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
            test_stride=run_test_stride(run),
            pack_splits=parse_pack_splits(args.pack_splits),
            use_ordinal_window_norm=False,
        )
        past = np.stack([past_pool[index][0].detach().cpu().numpy() for index in indices]).astype(np.float32)
        legal_levels = legal_patch_refine_levels_dataset_z(past, ladder=ladder, device=device)
        gt, gt_snap = snap_to_patch_refine_levels(binary_gt, legal_levels)
        mmpd, mmpd_snap = snap_to_patch_refine_levels(mmpd_binary_z, legal_levels)
        # Patch-refine output must already be a legal absolute 256-row decode.
        # Keep it raw after this assertion; snapping it would conceal a bad decode.
        binary = binary_pred
        lattice = {
            "gt": assert_on_patch_refine_levels(gt, legal_levels),
            "binary_staged": assert_on_patch_refine_levels(binary, legal_levels),
            "mmpd": assert_on_patch_refine_levels(mmpd, legal_levels),
        }
        lattice["gt"].update(gt_snap)
        lattice["binary_staged"].update({"raw_binary_retained": 1.0})
        lattice["mmpd"].update(mmpd_snap)
        lattice["mmpd_alignment"] = align
        write_json(args.raw_eval_dir / f"lattice_assertion_{dataset}.json", lattice)
        bundle = SimpleNamespace(
            fakes={"binary_staged": binary, "mmpd": mmpd},
            y_true_by_source={"binary_staged": gt, "mmpd": gt.copy()},
            past=past,
            indices=np.asarray(indices, dtype=np.int64),
            series_starts=binary_pack["series_starts"],
            run=run,
            pack_splits=[str(x) for x in binary_pack["pack_splits"].tolist()],
        )
        splits = split_windows(
            len(gt), args, dataset, indices=bundle.indices, lookback=args.lookback,
            horizon=args.horizon, test_stride=run_test_stride(run), series_starts=bundle.series_starts,
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
        patch_pred = binary_pack["unblended_nonoverlap_patch_pred"].astype(np.float32)
        patch_gt = binary_pack["unblended_nonoverlap_patch_gt"].astype(np.float32)
        patch_past = binary_pack["unblended_nonoverlap_patch_past"].astype(np.float32)
        patch_parent = binary_pack["unblended_nonoverlap_patch_parent"].astype(np.int64)
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
            patch_metrics = train_classifier(
                args, dataset, "binary_patch_unblended", 8, patch_bundle, patch_splits, device,
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
            args.output_dir, dataset, past, gt, binary, mmpd, legal_levels, score_data,
        )
        print(f"[{dataset}] canonical 256-row lattice asserted; visualization={plot}", flush=True)


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
    print(f"[merge] wrote metrics.csv for {len(merged)} datasets", flush=True)


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
