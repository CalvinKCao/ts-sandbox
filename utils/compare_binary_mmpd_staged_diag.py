#!/usr/bin/env python3
"""Compare binary staged anchor vs MMPD per-window, plot worst gaps, cache eval.

Ranks test windows by anchor MSE gap (default: binary - mmpd, i.e. where binary is
worse). Picks top-K with minimum spacing on series start (default 360 steps). For
each pick, saves binary coarse/fine 2D maps + 1D reps vs GT with MMPD overlay.

Caches per-dataset npz under --output-dir/eval_cache/ so plots can be regenerated
without re-running eval.

Example:
  python utils/compare_binary_mmpd_staged_diag.py \\
    --mmpd-dir results/datasets/07-08-mmpd-decoder-ordinal-norm-lb336-hz720 \\
    --binary-config configs/binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm.yaml \\
    --mmpd-config configs/mmpd_decoder_flat_subsets_paper_lb336_hz720_ordinal_norm.yaml \\
    --datasets ETTh1,weather,electricity,exchange_rate,traffic \\
    --output-dir reports/binary_vs_mmpd_ordinal_lb336_hz720
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
import torch
import yaml
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.diffusion_tsf.pipeline.config import load_experiment_config
from models.diffusion_tsf.pipeline.haar_frequency_calibration import ensure_haar_frequency_calibration
from models.diffusion_tsf.pipeline.fourier_frequency_calibration import ensure_fourier_frequency_calibration
from models.diffusion_tsf.pipeline.globals_bridge import patch_globals
from models.diffusion_tsf.pipeline.phase_diagnostics import select_spaced_top_k
from models.diffusion_tsf.pipeline.phases.staged_eval import StagedEvalPhase
from models.diffusion_tsf.pipeline.state import PipelineState
from models.diffusion_tsf.pipeline.visualize_utils import per_window_anchor_mse, save_figure_jpg
from models.diffusion_tsf.train_multivariate_pipeline import (
    generate_dataset_job,
    load_dataset,
    load_wrapped_guidance,
    resolve_pipeline_data_subset,
)
from utils.eval_mmpd_gaussian_anchor import (
    build_anchor_runs_from_subset_config,
    eval_test_stride,
    get_or_create_indices,
    indices_path,
    indices_root,
    load_indices,
    make_eval_indices,
    run_mmpd_eval,
    stable_dataset_seed,
    subsample_eval_indices,
    summarize_anchor_prob_core_metrics,
)
from utils.visualize_staged_eval_2d_preds import (
    _anchor_maps,
    _build_state,
    _decode_staged_1d_from_maps,
    _load_stage_model,
    _load_staged_bundle,
    _mark_lookback_overlap_1d,
    _mark_lookback_overlap_2d,
    _resolve_guidance_ckpt,
)
from utils.visualize_staged_forecast import _window_lengths


DEFAULT_BINARY_CONFIG = "configs/binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm.yaml"
DEFAULT_MMPD_CONFIG = "configs/mmpd_decoder_flat_subsets_paper_lb336_hz720_ordinal_norm.yaml"
DEFAULT_MMPD_DIR = "results/datasets/07-08-mmpd-decoder-ordinal-norm-lb336-hz720"
DEFAULT_BINARY_CKPT_STEM = "binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm"


def _staged_eval_overrides(config_path: str, dataset: str) -> Dict[str, Any]:
    cfg = load_experiment_config(config_path, cli_overrides={"dataset": dataset})
    for phase in cfg.get("phases") or []:
        if phase.get("phase") == "staged_eval":
            return {k: v for k, v in phase.items() if k != "phase"}
    raise KeyError(f"No staged_eval phase in {config_path}")


def discover_binary_ckpt(ckpt_base: Path, dataset: str, config_stem: str) -> Path:
    suffix = f"-{dataset}-{config_stem}"
    matches = sorted(
        [p for p in ckpt_base.iterdir() if p.is_dir() and p.name.endswith(suffix)],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(f"No checkpoint dir *{suffix} under {ckpt_base}")
    return matches[0]


def build_mmpd_args(
    *,
    mmpd_dir: Path,
    mmpd_config: Path,
    repo: Path,
    force_mmpd_eval: bool,
    smoke_test: bool = False,
) -> argparse.Namespace:
    args = argparse.Namespace(
        datasets=[],
        output_dir=mmpd_dir.resolve(),
        ckpt_base=(repo / "results" / "ckpts").resolve(),
        mmpd_repo=(repo / "temp" / "MMPD").resolve(),
        mmpd_data_dir=(repo / "temp" / "mmpd_datasets").resolve(),
        mmpd_run_config=mmpd_config.resolve(),
        seed=2026,
        skip_mmpd_train=True,
        force_mmpd_eval=force_mmpd_eval,
        force_indices=False,
        # Always keep MMPD campaign index files at full coverage; diag fraction
        # is applied in-memory in run_or_load_dataset_eval.
        test_fraction=1.0,
        metrics_profile="anchor-compat",
        mmpd_instance_norm=False,
        no_update_mmpd=True,
        sample_num=20,
        num_sampling_steps=20,
        gmm_components=10,
        gmm_iterations=10,
        topk_max=3,
        mmpd_eval_batch_size=16,
        num_workers=0,
        gpu=0,
        cpu=False,
        indices_dir=None,
        mmpd_output_root=None,
        test_max_items=None,
        eval_test_stride=None,
        use_ordinal_window_norm=False,
        ordinal_tie_atol=1.0e-6,
        mmpd_backbone="Decoder",
        lookback=336,
        horizon=720,
        patch_size=None,
        subset_config=None,
    )
    with mmpd_config.open(encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f) or {}
    mmpd_block = full_cfg.get("mmpd")
    if not isinstance(mmpd_block, dict):
        raise ValueError(f"{mmpd_config} missing mmpd: block")
    from utils.mmpd_run_config import apply_mmpd_run_config

    apply_mmpd_run_config(args, mmpd_block)
    exp = full_cfg.get("experiment") or {}
    if exp.get("use_ordinal_window_norm"):
        args.use_ordinal_window_norm = True
        args.mmpd_instance_norm = False
    if "ordinal_tie_atol" in exp:
        args.ordinal_tie_atol = float(exp["ordinal_tie_atol"])
    if args.subset_config is None:
        raise ValueError(
            f"{mmpd_config}: mmpd.subset_config required (e.g. binary_anchor_ar.yaml)"
        )
    if smoke_test:
        args.test_max_items = 8
        args.sample_num = 2
        args.num_sampling_steps = 2
        args.mmpd_eval_batch_size = 4
        args.force_mmpd_eval = True
    return args


def cache_path(output_dir: Path, dataset: str) -> Path:
    return output_dir / "eval_cache" / f"{dataset}.npz"


def summary_path(cache_dir: Path, dataset: str) -> Path:
    return cache_dir / f"{dataset}_summary.json"


def load_eval_cache(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def save_eval_cache(
    path: Path,
    *,
    window_indices: np.ndarray,
    series_starts: np.ndarray,
    binary_mse: np.ndarray,
    mmpd_mse: np.ndarray,
    diff: np.ndarray,
    binary_metrics: Dict[str, float],
    mmpd_metrics: Dict[str, float],
    test_stride: int,
    binary_ckpt: str,
    mmpd_dir: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        window_indices=window_indices,
        series_starts=series_starts,
        binary_anchor_mse=binary_mse,
        mmpd_anchor_mse=mmpd_mse,
        error_diff=diff,
        test_stride=np.array([test_stride], dtype=np.int64),
    )
    summary = {
        "dataset": path.stem,
        "binary_ckpt": binary_ckpt,
        "mmpd_dir": mmpd_dir,
        "test_stride": test_stride,
        "n_windows": int(len(window_indices)),
        "binary_metrics": binary_metrics,
        "mmpd_metrics": mmpd_metrics,
        "mean_binary_anchor_mse": float(binary_mse.mean()),
        "mean_mmpd_anchor_mse": float(mmpd_mse.mean()),
        "mean_error_diff": float(diff.mean()),
    }
    with summary_path(path.parent, path.stem).open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


def align_mmpd_pack(pack: Dict[str, np.ndarray], indices: Sequence[int]) -> Dict[str, np.ndarray]:
    pack_indices = pack.get("indices")
    if pack_indices is None:
        if pack["y_true"].shape[0] != len(indices):
            raise ValueError("MMPD pack missing indices and row count mismatch")
        pack_indices = np.asarray(indices, dtype=np.int64)
    order = {int(wi): row for row, wi in enumerate(pack_indices)}
    rows = []
    for wi in indices:
        if int(wi) not in order:
            raise KeyError(f"window {wi} missing from MMPD eval pack")
        rows.append(order[int(wi)])
    idx = np.asarray(rows, dtype=np.int64)
    out = {
        "y_true": pack["y_true"][idx],
        "deterministic": pack["deterministic"][idx],
        "indices": np.asarray(indices, dtype=np.int64),
    }
    if "samples" in pack:
        out["samples"] = pack["samples"][idx]
    return out


@torch.no_grad()
def run_binary_staged_eval(
    *,
    checkpoint_dir: Path,
    dataset: str,
    config_path: str,
    window_indices: Sequence[int],
    test_stride: int,
    device: torch.device,
    prob_samples: Optional[int] = None,
    prob_steps: Optional[int] = None,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    overrides = _staged_eval_overrides(config_path, dataset)
    phase = StagedEvalPhase(**overrides)
    cfg = load_experiment_config(config_path, cli_overrides={"dataset": dataset})
    state = PipelineState.from_config(cfg)
    state.checkpoint_dir = str(checkpoint_dir.resolve())
    state.dataset = dataset
    resolve_pipeline_data_subset(state)
    subset_id = state.subset_id or dataset
    state.subset_id = subset_id

    ensure_haar_frequency_calibration(state)
    ensure_fourier_frequency_calibration(state)
    import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod

    patch_globals(pipeline_mod, state, honor_dataset_windows=True)

    variate_indices = state.variate_indices
    if variate_indices is None:
        variate_indices = generate_dataset_job(dataset)["variate_indices"]
    subset_meta = state.data_subset_resolved or {}
    train_stride = int(subset_meta.get("train_stride", state.window_stride))

    lookback, horizon = _window_lengths(dataset, state)
    _, _, test_ds, norm_stats = load_dataset(
        dataset,
        variate_indices,
        stride=train_stride,
        test_stride=test_stride,
        lookback=lookback,
        horizon=horizon,
        ordinal_tie_atol=float(state.ordinal_tie_atol),
        use_ordinal_window_norm=state.use_ordinal_window_norm,
    )
    if norm_stats.get("ordinal_ladder") is not None:
        state.extra["global_ordinal_ladder"] = norm_stats["ordinal_ladder"]

    guidance_path, guidance_type = _resolve_guidance_ckpt(checkpoint_dir, subset_id, "auto")
    guidance = load_wrapped_guidance(
        str(guidance_path),
        len(variate_indices),
        device,
        guidance_type=guidance_type,
        dataset_lookback=lookback,
        dataset_horizon=horizon,
    )
    coarse_model = phase._load_model(state, "coarse", guidance, len(variate_indices), device)
    fine_model = phase._load_model(state, "fine", guidance, len(variate_indices), device)
    finer_model = (
        phase._load_model(state, "finer", guidance, len(variate_indices), device)
        if state.use_triple_scale
        else None
    )

    subset = Subset(test_ds, list(window_indices))
    batch_size = int(overrides.get("batch_size", 8))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
    n_prob = int(prob_samples if prob_samples is not None else overrides.get("probabilistic_n_samples", 20))
    n_steps = int(prob_steps if prob_steps is not None else overrides.get("probabilistic_num_inference_steps", 20))
    metrics, pack = phase._run_eval(
        state=state,
        subset_id=subset_id,
        loader=loader,
        device=device,
        coarse_model=coarse_model,
        fine_model=fine_model,
        finer_model=finer_model,
        prob_sampler=str(overrides.get("probabilistic_sampler", "dpmpp")),
        prob_steps=n_steps,
        prob_samples=n_prob,
        gmm_components=int(overrides.get("gmm_components", 10)),
        topk_max=int(overrides.get("topk_max", 3)),
        window_indices=list(window_indices),
        test_stride=test_stride,
    )
    return metrics, pack


def run_or_load_dataset_eval(
    *,
    dataset: str,
    mmpd_args: argparse.Namespace,
    binary_ckpt: Path,
    binary_config: str,
    output_dir: Path,
    device: torch.device,
    force_eval: bool,
    test_fraction: float = 1.0,
) -> Dict[str, np.ndarray]:
    cache = cache_path(output_dir, dataset)
    if cache.is_file() and not force_eval:
        print(f"[cache] {dataset}: loading {cache}")
        return load_eval_cache(cache)

    if mmpd_args.subset_config is None:
        raise ValueError("mmpd.subset_config not resolved; check mmpd run YAML")
    subset_runs = build_anchor_runs_from_subset_config(
        Path(mmpd_args.subset_config),
        [dataset],
        int(mmpd_args.seed),
    )
    run = subset_runs[dataset]
    idx_root = indices_root(mmpd_args)
    indices_file = indices_path(idx_root, dataset)
    if indices_file.is_file() and not mmpd_args.force_indices:
        window_indices = load_indices(idx_root, dataset)
    else:
        window_indices = get_or_create_indices(mmpd_args, run)
    # Fraction subsample in-memory only — do not rewrite MMPD campaign index files.
    frac = float(test_fraction)
    if frac < 1.0:
        n_full = len(window_indices)
        keep = make_eval_indices(
            n_full,
            frac,
            stable_dataset_seed(int(mmpd_args.seed), dataset),
            None,
        )
        window_indices = [int(window_indices[i]) for i in keep]
        print(
            f"[subset] {dataset}: test_fraction={frac:g} -> {len(window_indices)}/{n_full} windows",
            flush=True,
        )
    if mmpd_args.test_max_items is not None:
        window_indices = subsample_eval_indices(
            window_indices,
            mmpd_args.test_max_items,
            seed=int(mmpd_args.seed),
            dataset=dataset,
        )
    test_stride = eval_test_stride(mmpd_args, run)

    smoke_prob = None
    smoke_steps = None
    if mmpd_args.test_max_items is not None and mmpd_args.sample_num <= 4:
        smoke_prob = int(mmpd_args.sample_num)
        smoke_steps = int(mmpd_args.num_sampling_steps)

    print(f"[eval] {dataset}: binary staged anchor ({len(window_indices)} windows, stride={test_stride})")
    binary_metrics, binary_pack = run_binary_staged_eval(
        checkpoint_dir=binary_ckpt,
        dataset=dataset,
        config_path=binary_config,
        window_indices=window_indices,
        test_stride=test_stride,
        device=device,
        prob_samples=smoke_prob,
        prob_steps=smoke_steps,
    )

    print(f"[eval] {dataset}: MMPD anchor")
    mmpd_pack_raw = run_mmpd_eval(mmpd_args, run, window_indices)
    mmpd_pack = align_mmpd_pack(mmpd_pack_raw, window_indices)
    mmpd_metrics = summarize_anchor_prob_core_metrics(mmpd_pack)

    y_true_bin = binary_pack["y_true"]
    y_true_mmpd = mmpd_pack["y_true"]
    if y_true_bin.shape != y_true_mmpd.shape:
        raise RuntimeError(
            f"{dataset}: shape mismatch binary {y_true_bin.shape} vs mmpd {y_true_mmpd.shape}"
        )
    if not np.allclose(y_true_bin, y_true_mmpd, rtol=1e-4, atol=1e-4):
        row = int(np.argmax(np.abs(y_true_bin - y_true_mmpd).reshape(len(y_true_bin), -1).mean(axis=1)))
        wi = int(window_indices[row]) if row < len(window_indices) else row
        raise RuntimeError(
            f"{dataset}: y_true mismatch at row {row} (window {wi}); "
            "check eval_test_stride / ordinal norm alignment"
        )
    y_true = y_true_mmpd

    binary_mse = per_window_anchor_mse(y_true, binary_pack["final_anchor"])
    mmpd_mse = per_window_anchor_mse(y_true, mmpd_pack["deterministic"])
    series_starts = np.asarray(window_indices, dtype=np.int64) * int(test_stride)
    diff = binary_mse - mmpd_mse

    save_eval_cache(
        cache,
        window_indices=np.asarray(window_indices, dtype=np.int64),
        series_starts=series_starts,
        binary_mse=binary_mse,
        mmpd_mse=mmpd_mse,
        diff=diff,
        binary_metrics=binary_metrics,
        mmpd_metrics=mmpd_metrics,
        test_stride=test_stride,
        binary_ckpt=str(binary_ckpt),
        mmpd_dir=str(mmpd_args.output_dir),
    )
    print(f"[cache] {dataset}: wrote {cache}")
    return load_eval_cache(cache)


def rank_scores(cache: Dict[str, np.ndarray], diff_mode: str) -> np.ndarray:
    if diff_mode == "mmpd_minus_binary":
        return cache["mmpd_anchor_mse"] - cache["binary_anchor_mse"]
    if diff_mode == "binary_minus_mmpd":
        return cache["binary_anchor_mse"] - cache["mmpd_anchor_mse"]
    raise ValueError(f"unknown diff_mode: {diff_mode}")


def select_top_windows(
    cache: Dict[str, np.ndarray],
    *,
    top_k: int,
    min_spacing: int,
    diff_mode: str,
) -> List[Dict[str, Any]]:
    scores = rank_scores(cache, diff_mode)
    picks = select_spaced_top_k(
        scores,
        cache["series_starts"],
        k=top_k,
        min_spacing=min_spacing,
    )
    manifest: List[Dict[str, Any]] = []
    for rank, row in enumerate(picks, start=1):
        wi = int(cache["window_indices"][row])
        manifest.append({
            "rank": rank,
            "row": int(row),
            "window_index": wi,
            "series_start": int(cache["series_starts"][row]),
            "binary_anchor_mse": float(cache["binary_anchor_mse"][row]),
            "mmpd_anchor_mse": float(cache["mmpd_anchor_mse"][row]),
            "error_diff_mmpd_minus_binary": float(cache["mmpd_anchor_mse"][row] - cache["binary_anchor_mse"][row]),
            "error_diff_binary_minus_mmpd": float(cache["binary_anchor_mse"][row] - cache["mmpd_anchor_mse"][row]),
            "rank_score": float(scores[row]),
        })
    return manifest


def _plot_compare_panel(
    *,
    maps: Dict[str, Any],
    fine_model: torch.nn.Module,
    mmpd_1d: np.ndarray,
    dataset: str,
    window_index: int,
    meta: Dict[str, Any],
    output_path: Path,
    variables_to_plot: int,
    jpeg_dpi: int,
) -> Path:
    gt_c = maps["gt_coarse"]
    gt_f = maps["gt_fine"]
    pr_c = maps["pred_coarse"]
    pr_f = maps["pred_fine"]
    n_vars = min(variables_to_plot, gt_c.shape[0])

    coarse_np, fine_np, final_np = _decode_staged_1d_from_maps(fine_model, pr_c, pr_f)
    past_norm = maps["past_norm"].numpy()
    future_norm = maps["future_norm"].numpy()
    k = int(getattr(fine_model.config, "lookback_overlap", 0) or 0)
    lookback = int(past_norm.shape[-1])
    fut_len = int(future_norm.shape[-1])
    w_past = lookback

    gt_1d = np.concatenate([past_norm, future_norm], axis=-1)
    common_len = min(gt_1d.shape[-1], coarse_np.shape[-1], fine_np.shape[-1], final_np.shape[-1])
    gt_1d = gt_1d[..., :common_len]
    coarse_np = coarse_np[..., :common_len]
    fine_np = fine_np[..., :common_len]
    final_np = final_np[..., :common_len]
    t_axis = np.arange(-lookback, common_len - lookback)
    t_future = np.arange(0, mmpd_1d.shape[-1])
    span_label = f"LB={lookback}, K={k} overlap, H={fut_len - k} horizon"

    fig = plt.figure(figsize=(4.2 * n_vars, 2.4 * 5), constrained_layout=True)
    gs = fig.add_gridspec(5, n_vars)
    row_pairs = (
        ("GT coarse 2D", gt_c, "Binary coarse 2D", pr_c),
        ("GT fine 2D", gt_f, "Binary fine 2D", pr_f),
    )
    for row_idx, (_l_gt, d_gt, _l_pr, d_pr) in enumerate(row_pairs):
        for col in range(n_vars):
            for sub_row, data, label in (
                (0, d_gt[col], row_pairs[row_idx][0]),
                (1, d_pr[col], row_pairs[row_idx][2]),
            ):
                ax = fig.add_subplot(gs[row_idx * 2 + sub_row, col])
                h, w = data.shape
                im = ax.imshow(
                    data,
                    aspect="auto",
                    origin="lower",
                    extent=[0, w, 0, h],
                    cmap="plasma",
                    vmin=0.0,
                    vmax=1.0,
                )
                _mark_lookback_overlap_2d(ax, w_past, k)
                ax.set_title(f"var {col} | {label} ({h}x{w}, {span_label})", fontsize=8)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for col in range(n_vars):
        ax = fig.add_subplot(gs[4, col])
        ax.plot(t_axis, gt_1d[col], color="#2196F3", linewidth=1.5, label="GT")
        ax.plot(t_axis, coarse_np[col], color="#FF9800", linewidth=1.0, label="Binary coarse")
        ax.plot(t_axis, fine_np[col], color="#4CAF50", linewidth=1.0, label="Binary fine")
        ax.plot(t_axis, final_np[col], color="#E91E63", linewidth=1.2, label="Binary final")
        ax.plot(t_future, mmpd_1d[col], color="#9C27B0", linewidth=1.2, linestyle="--", label="MMPD (future)")
        _mark_lookback_overlap_1d(ax, w_past, k)
        ax.grid(True, alpha=0.12)
        ax.set_title(f"var {col} 1D window-norm ({span_label})", fontsize=8)
        if col == 0:
            ax.legend(fontsize=6, loc="upper right")

    title = (
        f"{dataset} win {window_index} | rank {meta['rank']:02d} | "
        f"bin_mse={meta['binary_anchor_mse']:.4f} mmpd_mse={meta['mmpd_anchor_mse']:.4f} "
        f"(mmpd-bin)={meta['error_diff_mmpd_minus_binary']:.4f}"
    )
    fig.suptitle(title, fontsize=10, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure_jpg(fig, str(output_path), dpi=jpeg_dpi)
    return output_path


@torch.no_grad()
def plot_dataset_windows(
    *,
    dataset: str,
    binary_ckpt: Path,
    binary_config: str,
    mmpd_pack_path: Path,
    cache: Dict[str, np.ndarray],
    top_manifest: Sequence[Dict[str, Any]],
    output_dir: Path,
    test_stride: int,
    device: torch.device,
    variables_to_plot: int,
    jpeg_dpi: int,
) -> List[Path]:
    bundle = _load_staged_bundle(binary_ckpt, dataset)
    subset_id = bundle["subset_id"]
    variate_indices = bundle["variate_indices"]
    state = _build_state(binary_ckpt, dataset, subset_id, binary_config)
    lookback, horizon = _window_lengths(dataset, state)
    data_subset = bundle["fine_metadata"].get("data_subset") or {}
    _, _, test_ds, _ = load_dataset(
        dataset,
        variate_indices,
        stride=int(data_subset.get("train_stride", state.window_stride)),
        test_stride=int(test_stride),
        lookback=lookback,
        horizon=horizon,
        ordinal_tie_atol=float(state.ordinal_tie_atol),
        use_ordinal_window_norm=state.use_ordinal_window_norm,
    )

    guidance_path, guidance_type = _resolve_guidance_ckpt(binary_ckpt, subset_id, "auto")
    guidance_model = load_wrapped_guidance(
        str(guidance_path),
        len(variate_indices),
        device,
        guidance_type=guidance_type,
        dataset_lookback=lookback,
        dataset_horizon=horizon,
    )
    coarse_model = _load_stage_model(state, "coarse", bundle["coarse_pt"], guidance_model, len(variate_indices), device)
    fine_model = _load_stage_model(state, "fine", bundle["fine_pt"], guidance_model, len(variate_indices), device)

    with np.load(mmpd_pack_path) as mmpd_data:
        mmpd_det = mmpd_data["deterministic"]
        mmpd_indices = mmpd_data["indices"] if "indices" in mmpd_data.files else cache["window_indices"]

    plot_dir = output_dir / "plots" / dataset
    saved: List[Path] = []
    for entry in top_manifest:
        wi = int(entry["window_index"])
        row = int(entry["row"])
        past, future = test_ds[wi]
        past_b = past.unsqueeze(0).to(device)
        future_b = future.unsqueeze(0).to(device)
        maps = _anchor_maps(coarse_model, fine_model, past_b, future_b)
        mmpd_rows = np.where(mmpd_indices == wi)[0]
        if mmpd_rows.size == 0:
            mmpd_rows = np.array([row])
        mmpd_1d = mmpd_det[int(mmpd_rows[0])]
        out_path = plot_dir / f"rank{entry['rank']:02d}_win{wi}_mmpd_minus_bin{entry['error_diff_mmpd_minus_binary']:+.4f}.jpg"
        saved.append(
            _plot_compare_panel(
                maps=maps,
                fine_model=fine_model,
                mmpd_1d=mmpd_1d,
                dataset=dataset,
                window_index=wi,
                meta=entry,
                output_path=out_path,
                variables_to_plot=variables_to_plot,
                jpeg_dpi=jpeg_dpi,
            )
        )
        print(f"[plot] {out_path}")
    return saved


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mmpd-dir", type=Path, default=REPO_ROOT / DEFAULT_MMPD_DIR)
    p.add_argument("--mmpd-config", type=Path, default=REPO_ROOT / DEFAULT_MMPD_CONFIG)
    p.add_argument("--binary-config", default=DEFAULT_BINARY_CONFIG)
    p.add_argument("--binary-ckpt-base", type=Path, default=REPO_ROOT / "results" / "ckpts")
    p.add_argument("--binary-ckpt-stem", default=DEFAULT_BINARY_CKPT_STEM)
    p.add_argument("--datasets", default="ETTh1,weather,electricity,exchange_rate,traffic")
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "reports" / "binary_vs_mmpd_ordinal_lb336_hz720")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--min-spacing", type=int, default=360)
    p.add_argument(
        "--diff-mode",
        choices=("binary_minus_mmpd", "mmpd_minus_binary"),
        default="binary_minus_mmpd",
        help="Rank score: binary-mmpd highlights windows where binary is worse (default).",
    )
    p.add_argument("--force-eval", action="store_true")
    p.add_argument("--force-mmpd-eval", action="store_true")
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="8 windows, 2 prob samples, small top-k; writes under output-dir_smoke",
    )
    p.add_argument(
        "--test-fraction",
        type=float,
        default=0.125,
        help="Random fraction of test windows per dataset (default 1/8). Use 1.0 for full set.",
    )
    p.add_argument("--test-max-items", type=int, default=None, help="Cap eval windows per dataset")
    p.add_argument("--plots-only", action="store_true", help="Skip eval; require eval_cache/*.npz")
    p.add_argument("--skip-plots", action="store_true")
    p.add_argument("--variables-to-plot", type=int, default=3)
    p.add_argument("--jpeg-dpi", type=int, default=100)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    output_dir = args.output_dir.resolve()
    if args.smoke_test:
        if not str(output_dir).endswith("_smoke"):
            output_dir = output_dir.parent / f"{output_dir.name}_smoke"
        args.top_k = min(int(args.top_k), 3)
        args.min_spacing = min(int(args.min_spacing), 48)
        args.force_eval = True
        args.test_fraction = 1.0
    elif float(args.test_fraction) < 1.0:
        # Keep full-set caches separate from fraction runs.
        frac_tag = f"_f{args.test_fraction:g}".replace(".", "p")
        if frac_tag not in output_dir.name and not str(output_dir).endswith("_smoke"):
            output_dir = output_dir.parent / f"{output_dir.name}{frac_tag}"
        args.force_eval = True
        args.force_mmpd_eval = True
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    mmpd_campaign_dir = args.mmpd_dir.resolve()
    mmpd_args = build_mmpd_args(
        mmpd_dir=mmpd_campaign_dir,
        mmpd_config=args.mmpd_config.resolve(),
        repo=REPO_ROOT,
        force_mmpd_eval=args.force_mmpd_eval or args.smoke_test,
        smoke_test=args.smoke_test,
    )
    # Always read campaign index files; write fraction MMPD packs under the diag
    # output dir so we do not clobber the full-set campaign raw/*.npz.
    mmpd_args.indices_dir = mmpd_campaign_dir
    if float(args.test_fraction) < 1.0 or args.smoke_test:
        mmpd_args.output_dir = output_dir
        (output_dir / "raw").mkdir(parents=True, exist_ok=True)
    if args.test_max_items is not None:
        mmpd_args.test_max_items = int(args.test_max_items)

    all_top: Dict[str, List[Dict[str, Any]]] = {}
    summary_rows: List[Dict[str, Any]] = []

    for dataset in datasets:
        binary_ckpt = discover_binary_ckpt(args.binary_ckpt_base, dataset, args.binary_ckpt_stem)
        if args.plots_only:
            cache_file = cache_path(output_dir, dataset)
            if not cache_file.is_file():
                raise FileNotFoundError(f"--plots-only but missing {cache_file}")
            cache = load_eval_cache(cache_file)
        else:
            cache = run_or_load_dataset_eval(
                dataset=dataset,
                mmpd_args=mmpd_args,
                binary_ckpt=binary_ckpt,
                binary_config=args.binary_config,
                output_dir=output_dir,
                device=device,
                force_eval=args.force_eval,
                test_fraction=float(args.test_fraction),
            )

        top = select_top_windows(
            cache,
            top_k=args.top_k,
            min_spacing=args.min_spacing,
            diff_mode=args.diff_mode,
        )
        all_top[dataset] = top
        top_json = output_dir / "top_windows" / f"{dataset}.json"
        top_json.parent.mkdir(parents=True, exist_ok=True)
        with top_json.open("w", encoding="utf-8") as f:
            json.dump(top, f, indent=2)
        print(f"[rank] {dataset}: wrote {len(top)} picks -> {top_json}")

        summary_file = summary_path(output_dir / "eval_cache", dataset)
        if summary_file.is_file():
            with summary_file.open(encoding="utf-8") as f:
                sm = json.load(f)
            summary_rows.append({
                "dataset": dataset,
                "binary_ckpt": sm.get("binary_ckpt", str(binary_ckpt)),
                "mean_binary_anchor_mse": sm.get("mean_binary_anchor_mse"),
                "mean_mmpd_anchor_mse": sm.get("mean_mmpd_anchor_mse"),
                "mean_error_diff_binary_minus_mmpd": sm.get("mean_error_diff"),
                "n_windows": sm.get("n_windows"),
            })

        if args.skip_plots:
            continue

        mmpd_npz = Path(mmpd_args.output_dir) / "raw" / f"mmpd_{dataset}.npz"
        if not mmpd_npz.is_file():
            raise FileNotFoundError(f"Missing MMPD raw pack for plots: {mmpd_npz}")
        test_stride = int(cache["test_stride"][0]) if "test_stride" in cache else 1
        plot_dataset_windows(
            dataset=dataset,
            binary_ckpt=binary_ckpt,
            binary_config=args.binary_config,
            mmpd_pack_path=mmpd_npz,
            cache=cache,
            top_manifest=top,
            output_dir=output_dir,
            test_stride=test_stride,
            device=device,
            variables_to_plot=args.variables_to_plot,
            jpeg_dpi=args.jpeg_dpi,
        )

    if summary_rows:
        csv_path = output_dir / "dataset_summary.csv"
        fields = list(summary_rows[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"[summary] {csv_path}")


if __name__ == "__main__":
    main()
