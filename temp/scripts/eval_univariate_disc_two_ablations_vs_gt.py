#!/usr/bin/env python3
"""L8/L16 univariate disc for window-norm patch_refine vs ordinal residual-fine.

Fair shared protocol (campaign-matched):
  - candidate-only horizon slices L∈{8,16}
  - sample0 fake aggregation
  - snap GT / binary / MMPD onto the same absolute 256-row ordinal ladder
  - per-slice bin-center shift (no zscore)

Accepts either second stage:
  - coarse + patch_refine  (4524397 window-norm)
  - coarse + fine          (4525834 ordinal residual)

Before the full disc fit, writes zoomed L8/L16 disc-input panels so you can
verify discrete snaps land on the shared 256-row grid with GT and MMPD.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from temp.scripts.viz_disc_bin_center_shift import (  # noqa: E402
    _draw_ladder_hlines,
    _plot_slice_panel,
    _slice_offsets,
    _ylim_rung_pad,
    _zoom_t_window,
)
from utils.disc_bin_center_shift import center_bin_index  # noqa: E402
from utils.dual_scale_bin_filter import align_mmpd_to_binary_dataset_norm  # noqa: E402
from utils.disc_shared import write_json  # noqa: E402
from utils.eval_discriminator_binary_vs_mmpd_univariate import (  # noqa: E402
    collect_partials,
    train_classifier,
    write_metrics_csv,
)
from utils.disc_shared import (  # noqa: E402
    apply_smoke_defaults as apply_base_smoke_defaults,
    binary_mmpd_train_scaler_map,
    parse_args as parse_base_args,
    split_windows,
)
from utils.eval_mmpd_gaussian_anchor import (  # noqa: E402
    AnchorRun,
    load_tsf_pack_pool,
    parse_pack_splits,
    run_subset_id,
    run_test_stride,
    run_train_stride,
    run_variate_indices,
)
from utils.forecast_pack_reduce import (  # noqa: E402
    assert_not_anchor_agg,
    reduce_pack_forecast,
    subset_pack_by_pool_indices,
)
from utils.patch_refine_ordinal_ladder import (  # noqa: E402
    assert_on_patch_refine_levels,
    legal_patch_refine_levels_dataset_z,
    snap_to_patch_refine_levels,
)
from utils.staged_binary_forecast import generate_staged_forecast  # noqa: E402
from utils.visualize_staged_eval_2d_preds import (  # noqa: E402
    _build_state,
    _load_stage_model,
    _resolve_guidance_ckpt,
)


DEFAULT_OUTPUT = REPO_ROOT / "results" / "datasets" / "disc-ablation-window-norm-vs-ordinal-fine"
DEFAULT_MMPD = (
    REPO_ROOT / "results" / "datasets" / "07-29-0151-h96-ordinal-binary-nonordinal-mmpd-mmpd"
)
DEFAULT_MODELS = (
    "window_norm="
    "results/ckpts/08-01-4524397-ETTh1-binary_window_norm_patch_refine_earlyjuly_norm"
    ":configs/binary_window_norm_patch_refine_earlyjuly_norm.yaml",
    "ordinal_fine="
    "results/ckpts/08-02-4525834-ETTh1-binary_ordinal_fine_finer_earlyjuly_hps"
    ":configs/binary_ordinal_fine_finer_earlyjuly_hps.yaml",
)


def _report_dir(output_dir: Path) -> Path:
    return REPO_ROOT / "reports" / output_dir.name


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
    if "--test-fraction" not in text:
        defaults += ["--test-fraction", "1.0"]
    if "--output-dir" not in text:
        defaults += ["--output-dir", str(DEFAULT_OUTPUT)]
    if "--pack-splits" not in text:
        defaults += ["--pack-splits", "test"]
    if "--slice-lengths" not in text:
        defaults += ["--slice-lengths", "8", "16"]
    if "--candidate-only" not in text and "--no-candidate-only" not in text:
        defaults += ["--candidate-only"]
    if "--disc-bin-center-shift" not in text and "--no-disc-bin-center-shift" not in text:
        defaults += ["--disc-bin-center-shift"]
    if "--mmpd-output-root" not in text:
        defaults += ["--mmpd-output-root", str(DEFAULT_MMPD)]
    return defaults


def parse_args() -> argparse.Namespace:
    custom = argparse.ArgumentParser(add_help=False)
    custom.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="tag=ckpt_root:config_yaml entries (one or more).",
    )
    custom.add_argument("--dataset", type=str, default="ETTh1")
    custom.add_argument("--pack-test-stride", type=int, default=4)
    custom.add_argument("--disc-index-stride", type=int, default=1)
    custom.add_argument("--fake-agg", choices=["prob_mean", "sample0"], default="sample0")
    custom.add_argument("--raw-eval-dir", type=Path, default=None)
    custom.add_argument(
        "--viz-only",
        action="store_true",
        help="Materialize/snap packs and write zoomed L8/L16 panels; skip disc train.",
    )
    custom.add_argument("--viz-windows", type=int, default=2)
    custom.add_argument("--viz-variate", type=int, default=0)
    custom.add_argument("--viz-zoom-steps", type=int, default=8)
    custom.add_argument("--viz-y-rung-pad", type=int, default=3)
    custom.add_argument("--seed", type=int, default=2026)
    extra, remaining = custom.parse_known_args(sys.argv[1:])
    saved = sys.argv
    sys.argv = [saved[0], *_defaults(remaining), *remaining]
    try:
        args = parse_base_args()
    finally:
        sys.argv = saved

    args.dataset = str(extra.dataset)
    args.datasets = [args.dataset]
    args.models = _parse_models(extra.models)
    args.pack_test_stride = max(1, int(extra.pack_test_stride))
    args.disc_index_stride = max(1, int(extra.disc_index_stride))
    args.fake_agg = str(extra.fake_agg)
    assert_not_anchor_agg(args.fake_agg)
    args.raw_eval_dir = (
        extra.raw_eval_dir.expanduser().resolve()
        if extra.raw_eval_dir is not None
        else (args.output_dir.parent / f"{args.output_dir.name}-raw")
    )
    args.viz_only = bool(extra.viz_only)
    args.viz_windows = max(1, int(extra.viz_windows))
    args.viz_variate = int(extra.viz_variate)
    args.viz_zoom_steps = max(4, int(extra.viz_zoom_steps))
    args.viz_y_rung_pad = max(1, int(extra.viz_y_rung_pad))
    args.seed = int(extra.seed)
    if int(args.lookback) != 336 or int(args.horizon) != 96:
        raise ValueError("this adapter is fixed to lb336 / hz96")
    return args


def apply_smoke_defaults(args: argparse.Namespace) -> None:
    apply_base_smoke_defaults(args)
    if args.smoke_test:
        args.raw_binary_batch_size = 1
        args.num_sampling_steps = min(int(args.num_sampling_steps), 2)
        args.viz_windows = 1
        args.test_fraction = min(float(args.test_fraction), 0.02)
        args.test_max_items = min(int(getattr(args, "test_max_items", 4) or 4), 4)


def _parse_models(entries: Sequence[str]) -> List[Dict[str, Path]]:
    out: List[Dict[str, Path]] = []
    for raw in entries:
        for piece in str(raw).split(","):
            piece = piece.strip()
            if not piece:
                continue
            if "=" not in piece or ":" not in piece:
                raise ValueError(
                    f"bad --models entry {piece!r}; expected tag=ckpt_root:config.yaml"
                )
            tag, rest = piece.split("=", 1)
            ckpt_s, cfg_s = rest.rsplit(":", 1)
            ckpt = Path(ckpt_s).expanduser()
            cfg = Path(cfg_s).expanduser()
            if not ckpt.is_absolute():
                ckpt = (REPO_ROOT / ckpt).resolve()
            if not cfg.is_absolute():
                cfg = (REPO_ROOT / cfg).resolve()
            out.append({"tag": tag.strip(), "ckpt": ckpt, "config": cfg})
    if not out:
        raise ValueError("need at least one --models entry")
    return out


def load_second_stage_run(
    dataset: str,
    checkpoint_dir: Path,
) -> Tuple[AnchorRun, Dict[str, Path], str]:
    """Accept coarse+patch_refine or coarse+fine under one subset."""
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"missing checkpoint dir: {checkpoint_dir}")
    candidates: List[Tuple[AnchorRun, Dict[str, Path], str]] = []
    for subset_dir in sorted(checkpoint_dir.iterdir()):
        if not subset_dir.is_dir():
            continue
        coarse_pt = subset_dir / "coarse" / "best.pt"
        if not coarse_pt.is_file():
            continue
        for stage, meta_name in (("patch_refine", "patch_refine"), ("fine", "fine")):
            stage_pt = subset_dir / stage / "best.pt"
            meta_path = subset_dir / meta_name / "metadata.json"
            if not (stage_pt.is_file() and meta_path.is_file()):
                continue
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            if metadata.get("dataset_name") != dataset:
                continue
            metadata = dict(metadata)
            metadata["dataset_name"] = dataset
            metadata["dataset"] = dataset
            run = AnchorRun(
                variant=f"binary_{stage}",
                dataset=dataset,
                root=checkpoint_dir,
                subset_dir=subset_dir,
                best_pt=stage_pt,
                itrans_pt=None,
                metadata=metadata,
            )
            stages = {"coarse_pt": coarse_pt, "second_pt": stage_pt, "stage": stage}
            candidates.append((run, stages, stage))
    if not candidates:
        raise FileNotFoundError(
            f"No coarse+(patch_refine|fine) checkpoint for {dataset} under {checkpoint_dir}"
        )
    if len(candidates) != 1:
        stages_found = sorted({c[2] for c in candidates})
        raise RuntimeError(
            f"ambiguous second-stage subsets for {dataset} under {checkpoint_dir}: {stages_found}"
        )
    return candidates[0]


def _mmpd_pack(root: Path, dataset: str) -> Dict[str, np.ndarray]:
    path = root / "raw" / f"mmpd_{dataset}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"missing MMPD pack: {path}")
    with np.load(path) as data:
        pack = {key: data[key] for key in data.files}
    required = {"y_true", "samples", "indices"}
    missing = sorted(required - set(pack))
    if missing:
        raise KeyError(f"{path} missing {missing}")
    return pack


def _thin_indices(
    indices: np.ndarray,
    *,
    seed: int,
    test_fraction: float,
    disc_index_stride: int,
    test_max_items: Optional[int],
) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64)
    if disc_index_stride > 1:
        idx = idx[::disc_index_stride]
    if test_fraction < 1.0:
        rng = np.random.default_rng(int(seed))
        n = max(1, int(round(len(idx) * float(test_fraction))))
        pick = np.sort(rng.choice(len(idx), size=min(n, len(idx)), replace=False))
        idx = idx[pick]
    if test_max_items is not None and len(idx) > int(test_max_items):
        rng = np.random.default_rng(int(seed) + 17)
        pick = np.sort(rng.choice(len(idx), size=int(test_max_items), replace=False))
        idx = idx[pick]
    if idx.size == 0:
        raise ValueError("no windows left after thinning")
    return idx


def _build_disc_ladder(
    dataset: str,
    run: AnchorRun,
    lookback: int,
    horizon: int,
    device: torch.device,
) -> Any:
    """Always build the absolute 256-row ordinal support used for disc snaps."""
    _, _, _, norm_stats = load_dataset(
        dataset,
        run_variate_indices(run),
        stride=run_train_stride(run),
        test_stride=run_test_stride(run),
        lookback=lookback,
        horizon=horizon,
        use_ordinal_window_norm=True,
    )
    ladder = norm_stats.get("ordinal_ladder")
    if ladder is None:
        raise RuntimeError(f"{dataset}: ordinal ladder missing from dataset loader")
    return ladder


def _load_models(
    *,
    dataset: str,
    ckpt_root: Path,
    config_path: Path,
    device: torch.device,
) -> Tuple[AnchorRun, Any, Any, str, Any]:
    run, stages, stage = load_second_stage_run(dataset, ckpt_root)
    state = _build_state(ckpt_root, dataset, run_subset_id(run), str(config_path))
    resolve_pipeline_data_subset(state)
    import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod

    use_ordinal = bool(state.use_ordinal_window_norm)
    _, _, _, norm_stats = load_dataset(
        dataset,
        run_variate_indices(run),
        stride=run_train_stride(run),
        test_stride=run_test_stride(run),
        lookback=int(state.lookback_length),
        horizon=int(state.forecast_length),
        ordinal_tie_atol=float(state.ordinal_tie_atol),
        use_ordinal_window_norm=use_ordinal,
    )
    ladder_model = norm_stats.get("ordinal_ladder")
    if use_ordinal:
        if ladder_model is None:
            raise RuntimeError(f"{dataset}: ordinal model config but no ladder")
        state.extra["global_ordinal_ladder"] = ladder_model
        pipeline_mod.GLOBAL_ORDINAL_LADDER = ladder_model
    else:
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
            dataset_lookback=int(state.lookback_length),
            dataset_horizon=int(state.forecast_length),
        )
        if ladder_model is not None and hasattr(guidance, "ordinal_ladder"):
            guidance.ordinal_ladder = ladder_model

    n_vars = len(run_variate_indices(run))
    coarse = _load_stage_model(
        state, "coarse", stages["coarse_pt"], guidance, n_vars, device,
        strict_non_guidance_shapes=True,
    )
    second = _load_stage_model(
        state, stage, stages["second_pt"], guidance, n_vars, device,
        strict_non_guidance_shapes=True,
    )
    if use_ordinal:
        for model in (coarse, second):
            model._ordinal_input_is_ranked = False
            model._ordinal_apply_ood_shift = True
    disc_ladder = _build_disc_ladder(
        dataset, run, int(state.lookback_length), int(state.forecast_length), device,
    )
    return run, coarse, second, stage, disc_ladder


def _materialize_binary(
    args: argparse.Namespace,
    *,
    tag: str,
    dataset: str,
    ckpt_root: Path,
    config_path: Path,
    indices: Sequence[int],
    device: torch.device,
) -> Tuple[Dict[str, np.ndarray], AnchorRun, Any]:
    cache = args.raw_eval_dir / f"binary_{tag}_{dataset}.npz"
    required = {"y_true", "samples", "indices", "past"}
    if cache.is_file() and not args.force_raw_eval:
        with np.load(cache) as data:
            pack = {key: data[key] for key in data.files}
        if required.issubset(pack) and np.array_equal(
            pack["indices"], np.asarray(indices, dtype=np.int64)
        ):
            run, _stages, _stage = load_second_stage_run(dataset, ckpt_root)
            ladder = _build_disc_ladder(dataset, run, args.lookback, args.horizon, device)
            return pack, run, ladder

    run, coarse, second, stage, ladder = _load_models(
        dataset=dataset, ckpt_root=ckpt_root, config_path=config_path, device=device,
    )
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
    if min(indices) < 0 or max(indices) >= len(pool):
        raise ValueError(
            f"{tag}/{dataset}: indices outside pack pool "
            f"(pool={len(pool)} pack_test_stride={args.pack_test_stride})"
        )
    loader = DataLoader(
        Subset(pool, list(indices)),
        batch_size=max(1, int(args.raw_binary_batch_size)),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    true_chunks: List[np.ndarray] = []
    pred_chunks: List[np.ndarray] = []
    past_chunks: List[np.ndarray] = []
    n_batches = len(loader)
    print(
        f"[{tag}/{dataset}] generate stage={stage} windows={len(indices)} "
        f"batches={n_batches} sampler={args.probabilistic_sampler} "
        f"steps={args.num_sampling_steps}",
        flush=True,
    )
    with torch.no_grad():
        for batch_i, (past, future) in enumerate(loader):
            past = past.to(device)
            overlap = int(getattr(second.config, "lookback_overlap", 0) or 0)
            target = future.to(device)[..., overlap:] if overlap else future.to(device)
            torch.manual_seed(int(args.seed) + batch_i * 1009)
            result = generate_staged_forecast(
                coarse,
                second,
                past,
                vertical_dual=False,
                sampler=args.probabilistic_sampler,
                num_inference_steps=args.num_sampling_steps,
            )
            prediction = result["prediction_global_norm"]
            if prediction.shape != target.shape:
                raise RuntimeError(
                    f"{tag}: pred/target mismatch {tuple(prediction.shape)} vs {tuple(target.shape)}"
                )
            true_chunks.append(target.cpu().numpy())
            pred_chunks.append(prediction.cpu().numpy())
            past_chunks.append(past.cpu().numpy())
            if (batch_i + 1) == n_batches or (batch_i + 1) % max(1, n_batches // 10) == 0:
                print(
                    f"[{tag}/{dataset}] generate {batch_i + 1}/{n_batches}",
                    flush=True,
                )
    pack = {
        "y_true": np.concatenate(true_chunks).astype(np.float32),
        "samples": np.concatenate(pred_chunks).astype(np.float32)[:, :, None, :],
        "past": np.concatenate(past_chunks).astype(np.float32),
        "indices": np.asarray(indices, dtype=np.int64),
        "series_starts": starts[np.asarray(indices, dtype=np.int64)],
        "pack_splits": np.asarray(splits),
        "second_stage": np.asarray(stage),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **pack)
    return pack, run, ladder


def _snap_bundle(
    *,
    past: np.ndarray,
    gt_raw: np.ndarray,
    binary_raw: np.ndarray,
    mmpd_raw: np.ndarray,
    ladder: Any,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    legal = legal_patch_refine_levels_dataset_z(past, ladder=ladder, device=device)
    gt, gt_stats = snap_to_patch_refine_levels(gt_raw, legal)
    binary, bin_stats = snap_to_patch_refine_levels(binary_raw, legal)
    mmpd, mmpd_stats = snap_to_patch_refine_levels(mmpd_raw, legal)
    lattice = {
        "gt": {**gt_stats, **assert_on_patch_refine_levels(gt, legal)},
        "binary": {**bin_stats, **assert_on_patch_refine_levels(binary, legal)},
        "mmpd": {**mmpd_stats, **assert_on_patch_refine_levels(mmpd, legal)},
        "n_rows": float(legal.shape[-1]),
    }
    return gt, binary, mmpd, legal.astype(np.float32), lattice


def _write_zoomed_disc_input_viz(
    args: argparse.Namespace,
    *,
    tag: str,
    dataset: str,
    gt: np.ndarray,
    binary: np.ndarray,
    mmpd: np.ndarray,
    legal_levels: np.ndarray,
    indices: np.ndarray,
) -> List[Path]:
    """Zoomed L8/L16 panels of the exact snapped alphabet fed to the disc."""
    out_dir = _report_dir(args.output_dir) / "disc_input_lattice_zoom" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    v = int(args.viz_variate)
    if v < 0 or v >= gt.shape[1]:
        raise ValueError(f"viz_variate={v} out of range for V={gt.shape[1]}")
    n = min(int(args.viz_windows), gt.shape[0])
    # Prefer high-amplitude GT windows so ladder rungs are visible.
    amp = gt[: , v, :].max(axis=-1) - gt[:, v, :].min(axis=-1)
    order = np.argsort(-amp)
    locals_ = order[:n].astype(np.int64)
    written: List[Path] = []
    for local in locals_:
        pool_i = int(indices[local])
        for slice_len in args.slice_lengths:
            L = int(slice_len)
            if L > gt.shape[-1]:
                continue
            for offset in _slice_offsets(gt.shape[-1], L)[:1]:
                gt_sl = gt[local, v, offset : offset + L]
                bin_sl = binary[local, v, offset : offset + L]
                mmpd_sl = mmpd[local, v, offset : offset + L]
                levels = legal_levels[local, v]
                path = out_dir / (
                    f"{dataset}_v{v}_local{local}_pool{pool_i}_L{L}_off{offset}_zoom.png"
                )
                _plot_slice_panel(
                    out_path=path,
                    dataset=f"{dataset}/{tag}",
                    local=int(local),
                    pool_i=pool_i,
                    variate=v,
                    slice_len=L,
                    offset=int(offset),
                    gt_slice=gt_sl,
                    mmpd_slice=mmpd_sl,
                    binary_slice=bin_sl,
                    levels_1d=levels,
                    lw=1.35,
                    ms=7.0,
                    dpi=220,
                    zoom=True,
                    zoom_steps=min(int(args.viz_zoom_steps), L),
                    zoom_t0=None,
                    y_rung_pad=int(args.viz_y_rung_pad),
                )
                written.append(path)
                # Extra raw-snap panel (pre bin-center) with explicit rung markers.
                raw_path = out_dir / (
                    f"{dataset}_v{v}_local{local}_pool{pool_i}_L{L}_off{offset}_snap_grid.png"
                )
                _plot_raw_snap_zoom(
                    out_path=raw_path,
                    title=f"{dataset}/{tag} snapped disc alphabet L={L}",
                    gt_slice=gt_sl,
                    binary_slice=bin_sl,
                    mmpd_slice=mmpd_sl,
                    levels_1d=levels,
                    zoom_steps=min(int(args.viz_zoom_steps), L),
                    y_rung_pad=int(args.viz_y_rung_pad),
                )
                written.append(raw_path)
    manifest = {
        "tag": tag,
        "dataset": dataset,
        "n_panels": len(written),
        "paths": [str(p.relative_to(REPO_ROOT)) for p in written],
        "protocol": {
            "candidate_only": True,
            "slice_lengths": list(map(int, args.slice_lengths)),
            "canvas_rows": 256,
            "bin_center_shift": True,
        },
    }
    write_json(out_dir / "viz_manifest.json", manifest)
    print(f"[{tag}] wrote {len(written)} zoomed disc-input panels under {out_dir}", flush=True)
    return written


def _plot_raw_snap_zoom(
    *,
    out_path: Path,
    title: str,
    gt_slice: np.ndarray,
    binary_slice: np.ndarray,
    mmpd_slice: np.ndarray,
    levels_1d: np.ndarray,
    zoom_steps: int,
    y_rung_pad: int,
) -> None:
    t0, t1 = _zoom_t_window(len(gt_slice), zoom_steps, None)
    view = slice(t0, t1)
    x = np.arange(t0, t1)
    series = [
        ("GT snap", gt_slice[view], "black"),
        ("MMPD snap", mmpd_slice[view], "#d62728"),
        ("binary snap", binary_slice[view], "#1f77b4"),
    ]
    y_all = np.concatenate([s[1] for s in series])
    y_lo, y_hi = _ylim_rung_pad(levels_1d, y_all, rung_pad=y_rung_pad)
    center = int(center_bin_index(levels_1d[None, None, :])[0, 0])
    fig, ax = plt.subplots(1, 1, figsize=(max(8.5, 0.55 * (t1 - t0) + 3.0), 4.2))
    _draw_ladder_hlines(ax, levels_1d, y_lo, y_hi, lw=0.55)
    for label, y, color in series:
        ax.plot(
            x, y, color=color, lw=1.4, label=label,
            drawstyle="steps-post", marker="o", markersize=7.0,
            markeredgewidth=0.55, markerfacecolor=color,
        )
    ax.axhline(float(levels_1d[center]), color="0.35", ls="--", lw=0.9, label=f"center idx={center}")
    ax.set_ylim(y_lo, y_hi)
    ax.set_xticks(x)
    ax.set_xlabel(f"step inside L-slice zoom t=[{t0},{t1})")
    ax.set_ylabel("dataset-z (256-row ladder)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8, frameon=False)
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else f"cuda:{args.gpu}")
    print(f"[device] {device}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.raw_eval_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "partials").mkdir(parents=True, exist_ok=True)
    dataset = args.dataset

    mmpd_full = _mmpd_pack(Path(args.mmpd_output_root), dataset)
    indices = _thin_indices(
        mmpd_full["indices"],
        seed=int(args.seed),
        test_fraction=float(args.test_fraction),
        disc_index_stride=int(args.disc_index_stride),
        test_max_items=getattr(args, "test_max_items", None),
    )
    mmpd_pack = dict(subset_pack_by_pool_indices(mmpd_full, indices))
    print(
        f"[{dataset}] MMPD-aligned windows={len(indices)} "
        f"pack_test_stride={args.pack_test_stride} fake_agg={args.fake_agg}",
        flush=True,
    )

    summary_rows: List[Dict[str, Any]] = []
    mmpd_trained = False
    for spec in args.models:
        tag = spec["tag"]
        pack, run, ladder = _materialize_binary(
            args,
            tag=tag,
            dataset=dataset,
            ckpt_root=spec["ckpt"],
            config_path=spec["config"],
            indices=indices.tolist(),
            device=device,
        )
        if not np.array_equal(pack["indices"], indices):
            raise RuntimeError(f"{tag}: pack indices drifted from MMPD alignment")
        binary_gt = pack["y_true"].astype(np.float32)
        binary_pred = reduce_pack_forecast(pack, agg=args.fake_agg)
        past = pack["past"].astype(np.float32)
        mmpd_gt = mmpd_pack["y_true"].astype(np.float32)
        mmpd_pred = reduce_pack_forecast(mmpd_pack, agg=args.fake_agg)
        scalers = binary_mmpd_train_scaler_map(args, run)
        mmpd_binary_z, align = align_mmpd_to_binary_dataset_norm(
            binary_y_true=binary_gt,
            mmpd_y_true=mmpd_gt,
            mmpd_fakes=mmpd_pred,
            **scalers,
        )
        gt, binary, mmpd, legal, lattice = _snap_bundle(
            past=past,
            gt_raw=binary_gt,
            binary_raw=binary_pred,
            mmpd_raw=mmpd_binary_z,
            ladder=ladder,
            device=device,
        )
        lattice["mmpd_alignment"] = align
        lattice["second_stage"] = str(pack.get("second_stage", "?"))
        write_json(args.raw_eval_dir / f"lattice_{tag}_{dataset}.json", lattice)
        print(
            f"[{tag}] lattice snap max_abs "
            f"gt={lattice['gt']['max_abs_snap_delta']:.4g} "
            f"binary={lattice['binary']['max_abs_snap_delta']:.4g} "
            f"mmpd={lattice['mmpd']['max_abs_snap_delta']:.4g}",
            flush=True,
        )
        _write_zoomed_disc_input_viz(
            args,
            tag=tag,
            dataset=dataset,
            gt=gt,
            binary=binary,
            mmpd=mmpd,
            legal_levels=legal,
            indices=indices,
        )
        if args.viz_only:
            continue

        sources = [tag] if mmpd_trained else [tag, "mmpd"]
        bundle = SimpleNamespace(
            fakes={f"{tag}": binary, "mmpd": mmpd},
            y_true_by_source={f"{tag}": gt, "mmpd": gt.copy()},
            past=past,
            legal_levels=legal,
            indices=indices,
            series_starts=pack["series_starts"],
            run=run,
            pack_splits=[str(x) for x in pack["pack_splits"].tolist()],
        )
        splits = split_windows(
            len(gt),
            args,
            dataset,
            indices=indices,
            lookback=args.lookback,
            horizon=args.horizon,
            test_stride=int(args.pack_test_stride),
            series_starts=bundle.series_starts,
        )
        for source in sources:
            per_length: Dict[str, Dict[str, float]] = {}
            for length in args.slice_lengths:
                if int(length) > args.horizon:
                    continue
                metrics = train_classifier(
                    args, dataset, source, int(length), bundle, splits, device,
                )
                per_length[str(int(length))] = metrics
                summary_rows.append(
                    {
                        "tag": tag if source == tag else "mmpd",
                        "compared_under": tag,
                        "source": source,
                        "slice_len": int(length),
                        "auroc": float(metrics.get("disc_auroc", float("nan"))),
                        "acc": float(metrics.get("disc_acc", float("nan"))),
                    }
                )
            write_json(
                args.output_dir / "partials" / f"{dataset}__{source}.json",
                per_length,
            )
            if source == "mmpd":
                mmpd_trained = True

    if not args.viz_only:
        merged = collect_partials(args.output_dir)
        write_json(args.output_dir / "metrics.json", merged)
        write_metrics_csv(args.output_dir, merged)
        write_json(args.output_dir / "summary_rows.json", summary_rows)
        print(f"[done] metrics -> {args.output_dir / 'metrics.json'}", flush=True)
    else:
        print("[done] viz-only; skipped disc train", flush=True)


def main() -> None:
    args = parse_args()
    apply_smoke_defaults(args)
    run(args)


if __name__ == "__main__":
    main()
