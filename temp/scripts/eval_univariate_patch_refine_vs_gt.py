#!/usr/bin/env python3
"""Univariate discriminator for non-ordinal h96 patch-refine forecasts vs GT.

This intentionally does not call the h720 binary/MMPD evaluator.  Patch-refine
checkpoints have ``coarse`` + ``patch_refine`` stages, and their decoded 256-row
grid is local to each forecast window.  GT is snapped to that local midpoint
grid without clipping values beyond rows 0..255.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.diffusion_tsf.pipeline.globals_bridge import patch_globals
from models.diffusion_tsf.train_multivariate_pipeline import (
    load_dataset,
    load_wrapped_guidance,
    resolve_pipeline_data_subset,
)
from utils.eval_discriminator_binary_vs_mmpd_univariate import train_classifier
from utils.eval_discriminator_texture_staged_vs_mmpd import (
    DEFAULT_DISC_OUTPUT,
    apply_smoke_defaults as apply_base_smoke_defaults,
    parse_args as parse_base_args,
    split_windows,
    write_json,
)
from utils.eval_mmpd_gaussian_anchor import (
    AnchorRun,
    load_tsf_pack_pool,
    parse_pack_splits,
    run_subset_id,
    run_test_stride,
    run_train_stride,
    run_variate_indices,
)
from utils.eval_trend_robust_texture_staged_vs_mmpd import (
    EvalProgress,
    fmt_duration,
    generate_staged_forecast,
    make_indices,
)
from utils.patch_refine_value_grid import (
    assert_on_patch_refine_grid,
    grid_coordinates,
    normalized_grid_step,
    snap_to_unbounded_patch_refine_grid,
    window_normalization_stats,
)
from utils.visualize_staged_eval_2d_preds import (
    _build_state,
    _load_stage_model,
    _resolve_guidance_ckpt,
)


DEFAULT_ANCHOR_CONFIG = "binary_patch_refine_lb336_hz96_full"
DEFAULT_BINARY_CONFIG = REPO_ROOT / "configs" / f"{DEFAULT_ANCHOR_CONFIG}.yaml"
DEFAULT_OUTPUT = DEFAULT_DISC_OUTPUT.parent / "disc-univariate-patch-refine-lb336-hz96-vs-gt"


def _unblended_nonoverlap_patch_batch(
    result: Dict[str, Any],
    target: torch.Tensor,
    past: torch.Tensor,
    config: Any,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, int]]:
    """Return disjoint, raw 8-column patch forecasts without vote averaging.

    Crops are accepted only when all eight columns have a visible local CDF
    boundary inside that crop.  We then retain pairwise-disjoint examples per
    `(window,variate)`.  Each one is decoded from exactly one raw patch CDF,
    rather than from ``blend_patch_bins``.
    """
    from models.diffusion_tsf.patch_refine_geometry import coarse_edges_from_cdf
    from models.diffusion_tsf.preprocessing import TimeSeriesTo2D

    patch_cdf = result["patch_cdf_unblended"]
    locations = result["patch_locations"]
    patch_width = int(config.patch_refine_patch_width)
    patch_height = int(config.patch_refine_patch_height)
    horizon = int(target.shape[-1])
    bins = TimeSeriesTo2D.bin_indices_from_cdf(patch_cdf[:, 0]).to(dtype=past.dtype)
    occupancy = patch_cdf[:, 0].sum(dim=-2)
    visible = (occupancy > 0) & (occupancy < int(config.patch_refine_patch_height))
    coarse_edges = coarse_edges_from_cdf(
        result["future_2d_coarse"],
        canvas_height=int(config.patch_refine_canvas_height),
    )
    center, std = window_normalization_stats(past, config)
    step = normalized_grid_step(config)
    kept_pred: List[torch.Tensor] = []
    kept_gt: List[torch.Tensor] = []
    kept_past: List[torch.Tensor] = []
    kept_parent: List[int] = []
    kept_start: List[int] = []
    kept_variate: List[int] = []
    candidates = 0
    rejected_invalid = 0
    next_allowed: Dict[Tuple[int, int], int] = {}
    ordered_locations = sorted(
        enumerate(locations),
        key=lambda item: (
            int(item[1].batch_index), int(item[1].variate_index),
            int(item[1].col0), int(item[1].row0),
        ),
    )
    for patch_i, loc in ordered_locations:
        candidates += 1
        end = int(loc.col0) + patch_width
        if end > horizon:
            rejected_invalid += 1
            continue
        edge = coarse_edges[loc.batch_index, loc.variate_index, loc.col0:end]
        edge_visible = (edge >= int(loc.row0)) & (edge < int(loc.row0) + patch_height)
        if not bool((edge_visible & visible[patch_i]).all()):
            rejected_invalid += 1
            continue
        key = (int(loc.batch_index), int(loc.variate_index))
        if int(loc.col0) < next_allowed.get(key, 0):
            continue
        next_allowed[key] = end
        absolute_bins = bins[patch_i] + float(loc.row0)
        normalized = -float(config.max_scale) + (absolute_bins + 0.5) * step
        pred = normalized * std[loc.batch_index, loc.variate_index, 0] + center[loc.batch_index, loc.variate_index, 0]
        kept_pred.append(pred.unsqueeze(0))
        kept_gt.append(target[loc.batch_index, loc.variate_index, loc.col0:end].unsqueeze(0))
        kept_past.append(past[loc.batch_index, loc.variate_index].unsqueeze(0))
        kept_parent.append(int(loc.batch_index))
        kept_start.append(int(loc.col0))
        kept_variate.append(int(loc.variate_index))
    if not kept_pred:
        empty = past.new_empty((0, 1, patch_width))
        return (
            empty,
            empty.clone(),
            past.new_empty((0, 1, past.shape[-1])),
            torch.empty(0, device=past.device, dtype=torch.long),
            torch.empty(0, device=past.device, dtype=torch.long),
            torch.empty(0, device=past.device, dtype=torch.long),
            {
                "candidates": candidates,
                "rejected_invalid_or_out_of_bounds": rejected_invalid,
                "selected": 0,
            },
        )
    return (
        torch.stack(kept_pred),
        torch.stack(kept_gt),
        torch.stack(kept_past),
        torch.tensor(kept_parent, device=past.device, dtype=torch.long),
        torch.tensor(kept_start, device=past.device, dtype=torch.long),
        torch.tensor(kept_variate, device=past.device, dtype=torch.long),
        {
            "candidates": candidates,
            "rejected_invalid_or_out_of_bounds": rejected_invalid,
            "selected": len(kept_pred),
        },
    )


def _default_arg_tokens(argv: Sequence[str]) -> List[str]:
    joined = " ".join(argv)
    defaults: List[str] = []
    if "--datasets" not in joined:
        defaults += ["--datasets", "ETTh1", "traffic", "exchange_rate"]
    if "--anchor-config" not in joined:
        defaults += ["--anchor-config", DEFAULT_ANCHOR_CONFIG]
    if "--binary-config" not in joined:
        defaults += ["--binary-config", str(DEFAULT_BINARY_CONFIG)]
    if "--lookback" not in joined:
        defaults += ["--lookback", "336"]
    if "--horizon" not in joined:
        defaults += ["--horizon", "96"]
    if "--test-stride" not in joined:
        defaults += ["--test-stride", "4"]
    if "--fake-sources" not in joined:
        defaults += ["--fake-sources", "binary_staged"]
    if "--output-dir" not in joined:
        defaults += ["--output-dir", str(DEFAULT_OUTPUT)]
    return defaults


def parse_args() -> argparse.Namespace:
    saved = sys.argv
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Use this h96 patch-refine run root instead of resolving --anchor-config.",
    )
    extra_args, remaining = extra.parse_known_args(saved[1:])
    sys.argv = [saved[0], *_default_arg_tokens(remaining), *remaining]
    try:
        args = parse_base_args()
    finally:
        sys.argv = saved
    args.checkpoint_dir = extra_args.checkpoint_dir
    if extra_args.checkpoint_dir is not None and "--datasets" not in " ".join(remaining):
        # A manually supplied root names one dataset run, unlike --ckpt-base.
        args.datasets = [args.datasets[0]]
    if list(args.fake_sources) != ["binary_staged"]:
        raise ValueError("this patch-refine evaluator supports only --fake-sources binary_staged")
    if int(args.lookback) != 336 or int(args.horizon) != 96:
        raise ValueError("this evaluator is fixed to the h96 patch-refine geometry: --lookback 336 --horizon 96")
    return args


def apply_smoke_defaults(args: argparse.Namespace) -> None:
    apply_base_smoke_defaults(args)
    if args.smoke_test and args.output_dir == DEFAULT_OUTPUT:
        args.output_dir = DEFAULT_OUTPUT.parent / f"{DEFAULT_OUTPUT.name}-smoke"
    if args.smoke_test:
        args.raw_binary_batch_size = 1
        args.num_sampling_steps = min(int(args.num_sampling_steps), 2)


def load_patch_refine_run(
    dataset: str,
    checkpoint_dir: Path,
    test_stride: int | None,
) -> Tuple[AnchorRun, Dict[str, Path]]:
    candidates: List[Tuple[AnchorRun, Dict[str, Path]]] = []
    seen_roots: set[Path] = set()
    for subset_dir in sorted(checkpoint_dir.iterdir()):
        # Symlink aliases (e.g. ETTh2_7v_s1 -> ETTh2) must not double-count.
        try:
            resolved = subset_dir.resolve()
        except OSError:
            resolved = subset_dir
        if resolved in seen_roots:
            continue
        coarse_pt = subset_dir / "coarse" / "best.pt"
        refine_pt = subset_dir / "patch_refine" / "best.pt"
        metadata_path = subset_dir / "patch_refine" / "metadata.json"
        if not (coarse_pt.is_file() and refine_pt.is_file() and metadata_path.is_file()):
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("dataset_name") != dataset:
            continue
        seen_roots.add(resolved)
        subset_id = str(metadata["subset_id"])
        metadata = dict(metadata)
        metadata["dataset_name"] = dataset
        metadata["dataset"] = dataset
        data_subset = dict(metadata.get("data_subset") or {})
        # Only override when the caller intentionally remaps the window grid.
        # MMPD-aligned disc/assert must keep the campaign pool stride or indices
        # land outside ``load_tsf_pack_pool``.
        if test_stride is not None:
            data_subset["test_stride"] = int(test_stride)
        metadata["data_subset"] = data_subset
        run = AnchorRun(
            variant="binary_patch_refine",
            dataset=dataset,
            root=checkpoint_dir,
            subset_dir=subset_dir,
            best_pt=refine_pt,
            itrans_pt=None,
            metadata=metadata,
        )
        candidates.append((run, {"coarse_pt": coarse_pt, "refine_pt": refine_pt}))
    if not candidates:
        raise FileNotFoundError(
            f"No coarse + patch_refine checkpoint for {dataset} under {checkpoint_dir}"
        )
    if len(candidates) != 1:
        raise RuntimeError(f"ambiguous patch-refine subsets for {dataset} under {checkpoint_dir}")
    return candidates[0]


def resolve_complete_patch_refine_ckpt_dir(
    ckpt_base: Path,
    dataset: str,
    anchor_config: str,
) -> Path:
    """Choose the newest named run that actually has both h96 stages."""
    if not ckpt_base.is_dir():
        raise FileNotFoundError(f"missing --ckpt-base: {ckpt_base}")
    matches = []
    suffix = f"-{dataset}-{anchor_config}"
    for root in ckpt_base.iterdir():
        if not root.is_dir() or not root.name.endswith(suffix):
            continue
        complete = any(
            (subset / "coarse" / "best.pt").is_file()
            and (subset / "patch_refine" / "best.pt").is_file()
            for subset in root.iterdir()
            if subset.is_dir()
        )
        if complete:
            matches.append(root)
    if not matches:
        raise FileNotFoundError(
            f"no complete coarse + patch_refine run matching *{suffix} under {ckpt_base}"
        )
    return max(matches, key=lambda path: path.stat().st_mtime)


def _validate_patch_refine_state(state: Any, model: Any) -> None:
    if not bool(state.use_patch_refine_stage):
        raise ValueError("binary config does not enable patch refinement")
    if bool(state.use_ordinal_window_norm):
        raise ValueError("this evaluator is for non-ordinal patch refinement only")
    if int(model.config.patch_refine_canvas_height) != 256:
        raise ValueError("expected a 256-row patch-refine canvas")
    if bool(getattr(model.config, "lookback_overlap_center_shift", False)):
        raise ValueError("unsupported: patch-refine grid has overlap center-shift enabled")


def materialize_raw_pack(
    args: argparse.Namespace,
    dataset: str,
    run: AnchorRun,
    stages: Dict[str, Path],
    indices: Sequence[int],
    device: torch.device,
) -> Dict[str, np.ndarray]:
    raw_path = args.output_dir / "raw" / f"binary_patch_refine_{dataset}.npz"
    if raw_path.is_file() and not args.force_raw_eval:
        with np.load(raw_path) as data:
            return {key: data[key] for key in data.files}

    pool, series_starts, splits, _lengths, _stats = load_tsf_pack_pool(
        dataset,
        run_variate_indices(run),
        lookback=args.lookback,
        horizon=args.horizon,
        train_stride=run_train_stride(run),
        test_stride=run_test_stride(run),
        pack_splits=parse_pack_splits(args.pack_splits),
        use_ordinal_window_norm=False,
    )
    loader = DataLoader(
        Subset(pool, list(indices)),
        batch_size=min(int(args.raw_binary_batch_size), 2),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    subset_id = run_subset_id(run)
    state = _build_state(run.root, dataset, subset_id, str(args.binary_config))
    resolve_pipeline_data_subset(state)
    import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod

    pipeline_mod.GLOBAL_ORDINAL_LADDER = None
    patch_globals(pipeline_mod, state, honor_dataset_windows=True)
    # This verifies the config's data geometry before model construction.
    load_dataset(
        dataset,
        run_variate_indices(run),
        stride=run_train_stride(run),
        test_stride=run_test_stride(run),
        lookback=args.lookback,
        horizon=args.horizon,
        use_ordinal_window_norm=False,
    )
    guidance = None
    if bool(state.use_guidance_channel) or not bool(state.disable_cross_attention):
        guidance_path, guidance_type = _resolve_guidance_ckpt(run.root, subset_id, "auto")
        guidance = load_wrapped_guidance(
            str(guidance_path),
            len(run_variate_indices(run)),
            device,
            guidance_type=guidance_type,
            dataset_lookback=args.lookback,
            dataset_horizon=args.horizon,
        )
    coarse = _load_stage_model(
        state, "coarse", stages["coarse_pt"], guidance, len(run_variate_indices(run)), device,
        strict_non_guidance_shapes=True,
    )
    refine = _load_stage_model(
        state, "patch_refine", stages["refine_pt"], guidance, len(run_variate_indices(run)), device,
        strict_non_guidance_shapes=True,
    )
    _validate_patch_refine_state(state, refine)

    past_all: List[np.ndarray] = []
    target_all: List[np.ndarray] = []
    target_raw_all: List[np.ndarray] = []
    sample_all: List[np.ndarray] = []
    patch_pred_all: List[np.ndarray] = []
    patch_gt_all: List[np.ndarray] = []
    patch_past_all: List[np.ndarray] = []
    patch_parent_all: List[np.ndarray] = []
    patch_start_all: List[np.ndarray] = []
    patch_variate_all: List[np.ndarray] = []
    patch_diag = {"candidates": 0, "rejected_invalid_or_out_of_bounds": 0, "selected": 0}
    windows_seen = 0
    progress = EvalProgress(f"patch-refine/{dataset}", len(loader))
    t0 = time.time()
    with torch.no_grad():
        for batch_idx, (past, future) in enumerate(loader):
            batch_start = time.time()
            past = past.to(device)
            future = future.to(device)
            overlap = int(refine.config.lookback_overlap)
            target_raw = future[..., overlap:] if overlap else future
            target = snap_to_unbounded_patch_refine_grid(target_raw, past, refine.config)

            torch.manual_seed(int(args.seed) + batch_idx * 1009)
            result = generate_staged_forecast(
                coarse,
                refine,
                past,
                vertical_dual=False,
                sampler=args.probabilistic_sampler,
                num_inference_steps=args.num_sampling_steps,
            )
            pred = result["prediction_global_norm"]
            if pred.shape != target.shape:
                raise RuntimeError(f"prediction/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
            assert_on_patch_refine_grid(pred, past, refine.config)
            assert_on_patch_refine_grid(target, past, refine.config)
            patch_pred, patch_gt, patch_past, patch_parent, patch_start, patch_variate, patch_info = _unblended_nonoverlap_patch_batch(
                result, target, past, refine.config,
            )
            if patch_pred.shape[0]:
                assert_on_patch_refine_grid(patch_pred, patch_past, refine.config)
                assert_on_patch_refine_grid(patch_gt, patch_past, refine.config)

            past_all.append(past.cpu().numpy())
            target_all.append(target.cpu().numpy())
            target_raw_all.append(target_raw.cpu().numpy())
            sample_all.append(pred.cpu().numpy()[:, :, None, :])
            patch_pred_all.append(patch_pred.cpu().numpy())
            patch_gt_all.append(patch_gt.cpu().numpy())
            patch_past_all.append(patch_past.cpu().numpy())
            patch_parent_all.append((patch_parent + windows_seen).cpu().numpy())
            patch_start_all.append(patch_start.cpu().numpy())
            patch_variate_all.append(patch_variate.cpu().numpy())
            for key in patch_diag:
                patch_diag[key] += int(patch_info[key])
            windows_seen += int(past.shape[0])
            progress.maybe_log(
                batch_idx + 1,
                extra=f"last_batch={fmt_duration(time.time() - batch_start)} elapsed={fmt_duration(time.time() - t0)}",
            )
    progress.done(extra=f"writing {raw_path}")

    y_true = np.concatenate(target_all, axis=0).astype(np.float32)
    y_true_raw = np.concatenate(target_raw_all, axis=0).astype(np.float32)
    past_np = np.concatenate(past_all, axis=0).astype(np.float32)
    samples = np.concatenate(sample_all, axis=0).astype(np.float32)
    patch_pred = np.concatenate(patch_pred_all, axis=0).astype(np.float32)
    patch_gt = np.concatenate(patch_gt_all, axis=0).astype(np.float32)
    patch_past = np.concatenate(patch_past_all, axis=0).astype(np.float32)
    patch_parent = np.concatenate(patch_parent_all, axis=0).astype(np.int64)
    patch_start = np.concatenate(patch_start_all, axis=0).astype(np.int64)
    patch_variate = np.concatenate(patch_variate_all, axis=0).astype(np.int64)
    for parent, variate in set(zip(patch_parent.tolist(), patch_variate.tolist())):
        starts = np.sort(patch_start[(patch_parent == parent) & (patch_variate == variate)])
        if np.any(starts[1:] < starts[:-1] + 8):
            raise RuntimeError(f"{dataset}: unblended patch examples overlap in parent row {parent}")
    coords = grid_coordinates(torch.from_numpy(y_true), torch.from_numpy(past_np), refine.config)
    pred_coords = grid_coordinates(
        torch.from_numpy(samples[:, :, 0, :]), torch.from_numpy(past_np), refine.config,
    )
    grid_center, grid_std = window_normalization_stats(torch.from_numpy(past_np), refine.config)
    grid_rows = torch.arange(
        int(refine.config.patch_refine_canvas_height), dtype=grid_center.dtype,
    ).view(1, 1, -1)
    grid_values = (
        -float(refine.config.max_scale)
        + (grid_rows + 0.5) * normalized_grid_step(refine.config)
    ) * grid_std + grid_center
    finite_rows = int(refine.config.patch_refine_canvas_height)
    out_of_range = (coords < 0) | (coords >= finite_rows)
    pack = {
        "y_true": y_true,
        "y_true_raw": y_true_raw,
        "past": past_np,
        "samples": samples,
        "indices": np.asarray(indices, dtype=np.int64),
        "series_starts": series_starts[np.asarray(indices, dtype=np.int64)],
        "pack_splits": np.asarray(splits),
        "gt_rows": np.rint(coords.numpy()).astype(np.int64),
        "grid_values": grid_values.numpy().astype(np.float32),
        "gt_grid_max_row_error": np.asarray(
            float((coords - coords.round()).abs().max()), dtype=np.float32,
        ),
        "binary_grid_max_row_error": np.asarray(
            float((pred_coords - pred_coords.round()).abs().max()), dtype=np.float32,
        ),
        "gt_unbounded_row_fraction": np.asarray(float(out_of_range.float().mean().item()), dtype=np.float32),
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
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(raw_path, **pack)
    print(
        f"[{dataset}] snapped GT to local 256-row grid; "
        f"unbounded-row fraction={float(out_of_range.float().mean()):.4f}",
        flush=True,
    )
    return pack


def run_eval(args: argparse.Namespace) -> None:
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else f"cuda:{args.gpu}")
    print(f"[device] {device}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "partials").mkdir(parents=True, exist_ok=True)
    merged: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for dataset in args.datasets:
        ckpt_dir = args.checkpoint_dir or resolve_complete_patch_refine_ckpt_dir(
            args.ckpt_base, dataset, args.anchor_config,
        )
        run, stages = load_patch_refine_run(dataset, ckpt_dir, args.test_stride)
        indices = make_indices(args, run)
        pack = materialize_raw_pack(args, dataset, run, stages, indices, device)
        y_true = pack["y_true"].astype(np.float32)
        fake = pack["samples"][:, :, 0, :].astype(np.float32)
        past = pack["past"].astype(np.float32)
        if y_true.shape != fake.shape:
            raise RuntimeError(f"{dataset}: snapped GT/fake mismatch {y_true.shape} vs {fake.shape}")
        bundle = SimpleNamespace(
            fakes={"binary_staged": fake},
            y_true_by_source={"binary_staged": y_true},
            past=past,
            indices=pack["indices"],
            series_starts=pack["series_starts"],
            run=run,
            pack_splits=[str(x) for x in pack["pack_splits"].tolist()],
        )
        splits = split_windows(
            len(y_true), args, dataset, indices=pack["indices"], lookback=args.lookback,
            horizon=args.horizon, test_stride=run_test_stride(run), series_starts=pack["series_starts"],
        )
        blended_by_len: Dict[str, Dict[str, float]] = {}
        for length in args.slice_lengths:
            if int(length) > args.horizon:
                continue
            metrics = train_classifier(args, dataset, "binary_staged", int(length), bundle, splits, device)
            blended_by_len[str(int(length))] = metrics
            write_json(
                args.output_dir / "partials" / f"{dataset}__binary_staged.json",
                blended_by_len,
            )

        patch_pred = pack["unblended_nonoverlap_patch_pred"].astype(np.float32)
        patch_gt = pack["unblended_nonoverlap_patch_gt"].astype(np.float32)
        patch_past = pack["unblended_nonoverlap_patch_past"].astype(np.float32)
        patch_parent = pack["unblended_nonoverlap_patch_parent"].astype(np.int64)
        if patch_pred.shape != patch_gt.shape or patch_pred.shape[1:] != (1, 8):
            raise RuntimeError(f"{dataset}: invalid unblended patch shape {patch_pred.shape}")
        patch_splits = {
            name: np.flatnonzero(np.isin(patch_parent, parent_rows)).astype(np.int64)
            for name, parent_rows in splits.items()
        }
        patch_counts = {name: len(rows) for name, rows in patch_splits.items()}
        if any(count == 0 for count in patch_counts.values()):
            # The tiny smoke checkpoint can emit almost no fully-visible crops.
            # Preserve its valid-grid smoke result without manufacturing a split.
            patch_metrics: Dict[str, float] = {
                "skipped_insufficient_temporal_coverage": 1.0,
                **{f"n_windows_{name}": float(count) for name, count in patch_counts.items()},
            }
            print(
                f"[{dataset}] skipping unblended L8 classifier: insufficient valid temporal coverage "
                f"{patch_counts}",
                flush=True,
            )
        else:
            patch_bundle = SimpleNamespace(
                fakes={"binary_patch_unblended": patch_pred},
                y_true_by_source={"binary_patch_unblended": patch_gt},
                past=patch_past,
                indices=patch_parent,
                series_starts=pack["series_starts"][patch_parent],
                run=run,
                pack_splits=[str(x) for x in pack["pack_splits"].tolist()],
            )
            patch_metrics = train_classifier(
                args,
                dataset,
                "binary_patch_unblended",
                8,
                patch_bundle,
                patch_splits,
                device,
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
        merged[dataset] = {
            "overlap_blended": blended_by_len,
            "unblended_nonoverlap_patch": {"8": patch_metrics},
        }
    write_json(args.output_dir / "metrics.json", merged)
    write_json(
        args.output_dir / "run_manifest.json",
        {
            "task": "univariate_patch_refine_vs_locally_snapped_gt",
            "binary_config": str(args.binary_config),
            "coordinate_space": "binary_dataset_z_then_per_window_patch_refine_midpoint_grid",
            "gt_grid": "unbounded nearest midpoint row; no endpoint clipping",
            "unblended_patch_metric": (
                "raw 8-column patch CDFs, greedily selected as pairwise "
                "non-overlapping examples only after every column has a visible in-crop edge; "
                "no patch-vote averaging"
            ),
            "datasets": list(args.datasets),
        },
    )
    print(f"[done] wrote {args.output_dir / 'metrics.json'}", flush=True)


def main() -> None:
    args = parse_args()
    apply_smoke_defaults(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run_eval(args)


if __name__ == "__main__":
    main()
