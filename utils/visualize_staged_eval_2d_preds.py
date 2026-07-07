#!/usr/bin/env python3
"""GT vs anchor-pred 2D coarse/fine CDF maps + final 1D for staged checkpoints.

Random test windows plus worst-window picks from worst_windows.json when present.

Example:
  python utils/visualize_staged_eval_2d_preds.py \\
    --checkpoint-dir results/ckpts/07-02-4041709-weather-..._fourier_flatline_blur \\
    --dataset weather \\
    --results-dir results/datasets/07-02-4041709-weather-..._fourier_flatline_blur \\
    --n-random 2 --n-worst 3

  python utils/visualize_staged_eval_2d_preds.py \\
    --checkpoint-dir results/ckpts/07-04-4053057-dynamic-..._healthy_norm_retrain \\
    --dataset dynamic \\
    --config configs/binary_anchor_ar_patch_decoder_ctx_healthy_norm_retrain.yaml \\
    --results-dir results/datasets/07-04-4053057-dynamic-..._healthy_norm_retrain \\
    --guidance-type patch_decoder --n-random 2 --n-worst 3
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.diffusion_tsf.pipeline.config import load_experiment_config
from models.diffusion_tsf.pipeline.phases.staged_diffusion_pretrain import patch_stage_globals
from models.diffusion_tsf.pipeline.state import PipelineState
from models.diffusion_tsf.pipeline.visualize_utils import (
    pick_sample_indices,
    save_figure_jpg,
    _staged_fine_value_range,
)
from models.diffusion_tsf.train_multivariate_pipeline import (
    anchor_kwargs_from_params,
    create_diffusion_model,
    load_dataset,
    load_diffusion_state_keep_attached_guidance,
    load_wrapped_guidance,
)
from utils.visualize_staged_forecast import _load_staged_bundle, _window_lengths


DEFAULT_CONFIG = (
    "configs/binary_anchor_stationary_flat_subsets_grad_accum_150_fourier_flatline_blur.yaml"
)


def _eval_test_stride(config_path: str, dataset: str, fallback: int) -> int:
    cfg = load_experiment_config(config_path, cli_overrides={"dataset": dataset})
    for phase in cfg.get("phases") or []:
        if phase.get("phase") != "staged_eval":
            continue
        by_dataset = phase.get("eval_test_fraction_by_dataset") or {}
        if dataset in by_dataset or phase.get("eval_test_fraction") is not None:
            return int(phase.get("test_stride", fallback))
    return fallback


def _build_state(checkpoint_dir: Path, dataset: str, subset_id: str, config_path: str) -> PipelineState:
    cfg = load_experiment_config(config_path, cli_overrides={"dataset": dataset})
    state = PipelineState.from_config(cfg)
    state.checkpoint_dir = str(checkpoint_dir.resolve())
    state.dataset = dataset
    state.subset_id = subset_id
    return state


def _resolve_guidance_ckpt(
    checkpoint_dir: Path,
    subset_id: str,
    guidance_type: str,
) -> Tuple[Path, str]:
    patch_path = checkpoint_dir / f"{subset_id}_patch_guidance.pt"
    itrans_path = checkpoint_dir / f"{subset_id}_itransformer_finetuned.pt"
    if guidance_type == "patch_decoder":
        if not patch_path.is_file():
            raise FileNotFoundError(f"Missing patch guidance ckpt: {patch_path}")
        return patch_path, "patch_decoder"
    if guidance_type == "itransformer":
        if not itrans_path.is_file():
            raise FileNotFoundError(f"Missing iTrans guidance ckpt: {itrans_path}")
        return itrans_path, "itransformer"
    if patch_path.is_file():
        return patch_path, "patch_decoder"
    if itrans_path.is_file():
        return itrans_path, "itransformer"
    raise FileNotFoundError(
        f"Missing guidance ckpt under {checkpoint_dir} "
        f"(expected {patch_path.name} or {itrans_path.name})"
    )


def _load_stage_model(
    state: PipelineState,
    stage: str,
    ckpt_path: Path,
    guidance_model: Any,
    n_vars: int,
    device: torch.device,
) -> torch.nn.Module:
    import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod

    patch_stage_globals(pipeline_mod, state, stage, honor_dataset_windows=True)
    lookback, horizon = _window_lengths(state.dataset, state)
    meta_path = ckpt_path.parent / "metadata.json"
    tuned: Dict[str, Any] = {}
    if meta_path.is_file():
        with meta_path.open(encoding="utf-8") as f:
            tuned = json.load(f).get("tuned_params") or {}

    model = create_diffusion_model(
        n_variates=n_vars,
        lookback=lookback,
        horizon=horizon,
        guidance_model=guidance_model,
        diffusion_stage=stage,
        use_guidance_channel=state.use_guidance_channel,
        **anchor_kwargs_from_params(tuned),
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    load_diffusion_state_keep_attached_guidance(model, ckpt["model_state_dict"])
    model.eval()
    return model


def _load_worst_indices(results_dir: Optional[Path], subset_id: str, metrics: Sequence[str], top_k: int) -> List[Tuple[int, str, int, float]]:
    if results_dir is None:
        return []
    worst_path = results_dir / subset_id / "worst_windows.json"
    if not worst_path.is_file():
        return []
    with worst_path.open(encoding="utf-8") as f:
        entries = json.load(f)
    out: List[Tuple[int, str, int, float]] = []
    for metric in metrics:
        ranked = [e for e in entries if str(e.get("metric")) == metric]
        ranked.sort(key=lambda e: int(e["rank"]))
        for e in ranked[:top_k]:
            out.append((int(e["window_index"]), metric, int(e["rank"]), float(e["score"])))
    return out


def _pick_windows(
    n_test: int,
    *,
    seed: int,
    n_random: int,
    worst: Sequence[Tuple[int, str, int, float]],
) -> List[Tuple[int, str]]:
    chosen: List[Tuple[int, str]] = []
    used: Set[int] = set[int]()
    for wi, metric, rank, _score in worst:
        if wi in used or wi < 0 or wi >= n_test:
            continue
        used.add(wi)
        chosen.append((wi, f"worst_{metric}_rank{rank:02d}"))
    random.seed(seed)
    for wi in pick_sample_indices(n_test, n_random, seed=seed):
        if wi in used:
            continue
        used.add(wi)
        chosen.append((wi, "random"))
    return chosen


@torch.no_grad()
def _decode_staged_1d_from_maps(
    fine_model: torch.nn.Module,
    coarse_map: np.ndarray,
    fine_map: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode (V, H, W) coarse/fine maps to 1D window-norm series."""
    coarse_t = torch.from_numpy(coarse_map).unsqueeze(0)
    fine_t = torch.from_numpy(fine_map).unsqueeze(0)
    b, v = coarse_t.shape[:2]
    bv = b * v
    to_2d = fine_model.to_2d
    coarse_flat = coarse_t.reshape(bv, 1, coarse_t.shape[-2], coarse_t.shape[-1])
    fine_flat = fine_t.reshape(bv, 1, fine_t.shape[-2], fine_t.shape[-1])
    coarse_1d = to_2d._decode_occupancy_in_range(
        coarse_flat, value_range=to_2d.max_scale, cdf_decoder="mean",
    )
    fine_1d = to_2d._decode_occupancy_in_range(
        fine_flat, value_range=_staged_fine_value_range(fine_model), cdf_decoder="mean",
    )
    final_1d = fine_model.decode_dual_from_2d(coarse_t, fine_t, from_diffusion=False)
    return (
        coarse_1d.reshape(b, v, -1).detach().cpu().numpy()[0],
        fine_1d.reshape(b, v, -1).detach().cpu().numpy()[0],
        final_1d.detach().cpu().numpy()[0],
    )


@torch.no_grad()
def _anchor_maps(
    coarse_model: torch.nn.Module,
    fine_model: torch.nn.Module,
    past_b: torch.Tensor,
    future_b: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    coarse_out = coarse_model.generate(past_b, sampler="anchor", num_inference_steps=1)
    fine_out = fine_model.generate(
        past_b,
        sampler="anchor",
        num_inference_steps=1,
        future_coarse_2d=coarse_out["future_2d_coarse"],
    )
    past_norm, future_norm, _norm_stats = fine_model._normalize_sequence(past_b, future_b)
    past_maps_gt = fine_model._encode_staged_maps(past_norm)
    future_maps_gt = fine_model._encode_staged_maps(future_norm)
    past_c_gt = past_maps_gt["coarse"][0].cpu().numpy()
    past_f_gt = past_maps_gt["fine"][0].cpu().numpy()
    fut_c_gt = future_maps_gt["coarse"][0].cpu().numpy()
    fut_f_gt = future_maps_gt["fine"][0].cpu().numpy()
    past_c_pred = coarse_out["past_2d_coarse"][0].cpu().numpy()
    past_f_pred = coarse_out["past_2d_fine"][0].cpu().numpy()
    fut_c_pred = coarse_out["future_2d_coarse"][0].cpu().numpy()
    fut_f_pred = fine_out["future_2d_fine"][0].cpu().numpy()
    return {
        "gt_coarse": np.concatenate([past_c_gt, fut_c_gt], axis=-1),
        "gt_fine": np.concatenate([past_f_gt, fut_f_gt], axis=-1),
        "pred_coarse": np.concatenate([past_c_pred, fut_c_pred], axis=-1),
        "pred_fine": np.concatenate([past_f_pred, fut_f_pred], axis=-1),
        "past_norm": past_norm[0].cpu(),
        "future_norm": future_norm[0].cpu(),
        "coarse_out": coarse_out,
        "fine_out": fine_out,
    }


def _mark_lookback_overlap_2d(ax: plt.Axes, w_past: int, k: int) -> None:
    """Vertical guides: lookback | overlap | horizon on a past+future 2D canvas."""
    if k > 0:
        overlap_start = w_past - k
        ax.axvspan(overlap_start, w_past + k, color="white", alpha=0.14)
        ax.axvline(x=overlap_start, color="white", linestyle=":", linewidth=0.8, alpha=0.8)
        ax.axvline(x=w_past + k, color="white", linestyle="--", linewidth=0.9, alpha=0.85)
    ax.axvline(x=w_past, color="white", linestyle="-", linewidth=0.9, alpha=0.85)


def _mark_lookback_overlap_1d(ax: plt.Axes, w_past: int, k: int) -> None:
    """Guides on 1D axis with t=0 at lookback|future boundary."""
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8, alpha=0.35)
    if k <= 0:
        return
    ax.axvspan(-k, k, color="#FFC107", alpha=0.12)
    ax.axvline(x=-k, color="#F57C00", linestyle=":", linewidth=0.9, alpha=0.7)
    ax.axvline(x=k, color="black", linestyle=":", linewidth=0.8, alpha=0.35)


def _plot_panel(
    *,
    maps: Dict[str, Any],
    fine_model: torch.nn.Module,
    dataset: str,
    window_index: int,
    tag: str,
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
    span_label = f"LB={lookback}, K={k} overlap, H={fut_len - k} horizon"

    fig = plt.figure(figsize=(4.2 * n_vars, 2.4 * 5), constrained_layout=True)
    gs = fig.add_gridspec(5, n_vars)

    row_pairs = (
        ("GT coarse 2D", gt_c, "Pred coarse 2D", pr_c),
        ("GT fine 2D", gt_f, "Pred fine 2D", pr_f),
    )
    for row_idx, (_l_gt, d_gt, _l_pr, d_pr) in enumerate(row_pairs):
        for col in range(n_vars):
            for sub_row, data, label in ((0, d_gt[col], row_pairs[row_idx][0]), (1, d_pr[col], row_pairs[row_idx][2])):
                ax = fig.add_subplot(gs[row_idx * 2 + sub_row, col])
                h, w = data.shape
                im = ax.imshow(data, aspect="auto", origin="lower", extent=[0, w, 0, h], cmap="plasma", vmin=0.0, vmax=1.0)
                _mark_lookback_overlap_2d(ax, w_past, k)
                ax.set_title(f"var {col} | {label} ({h}x{w}, {span_label})", fontsize=8)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for col in range(n_vars): # bruh
        ax = fig.add_subplot(gs[4, col])
        ax.plot(t_axis, gt_1d[col], color="#2196F3", linewidth=1.5, label="GT")
        ax.plot(t_axis, coarse_np[col], color="#FF9800", linewidth=1.1, label="Coarse pred")
        ax.plot(t_axis, fine_np[col], color="#4CAF50", linewidth=1.0, label="Fine pred")
        ax.plot(t_axis, final_np[col], color="#E91E63", linewidth=1.2, label="Final pred")
        _mark_lookback_overlap_1d(ax, w_past, k)
        ax.grid(True, alpha=0.12)
        ax.set_title(f"var {col} 1D window-norm ({span_label})", fontsize=8)
        if col == 0:
            ax.legend(fontsize=6, loc="upper right")

    fig.suptitle(f"{dataset} win {window_index} | {tag}", fontsize=11, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure_jpg(fig, str(output_path), dpi=jpeg_dpi)
    return output_path


def run_viz(
    *,
    checkpoint_dir: Path,
    dataset: str,
    output_dir: Path,
    results_dir: Optional[Path],
    config_path: str,
    guidance_type: str,
    test_stride: Optional[int],
    n_random: int,
    n_worst: int,
    worst_metrics: Sequence[str],
    seed: int,
    variables_to_plot: int,
    jpeg_dpi: int,
    device: torch.device,
) -> List[Path]:
    bundle = _load_staged_bundle(checkpoint_dir, dataset)
    subset_id = bundle["subset_id"]
    variate_indices = bundle["variate_indices"]
    n_vars = len(variate_indices)
    state = _build_state(checkpoint_dir, dataset, subset_id, config_path)
    lookback, horizon = _window_lengths(dataset, state)

    data_subset = bundle["fine_metadata"].get("data_subset") or {}
    if test_stride is None:
        test_stride = _eval_test_stride(
            config_path,
            dataset,
            int(data_subset.get("test_stride", state.window_stride)),
        )
    _, _, test_ds, _norm_stats = load_dataset(
        dataset,
        variate_indices,
        stride=int(data_subset.get("train_stride", state.window_stride)),
        test_stride=int(test_stride),
        lookback=lookback,
        horizon=horizon,
    )
    if len(test_ds) == 0:
        raise ValueError(f"Empty test set for {dataset}")

    guidance_path, resolved_guidance_type = _resolve_guidance_ckpt(
        checkpoint_dir, subset_id, guidance_type,
    )
    guidance_model = load_wrapped_guidance(
        str(guidance_path),
        n_vars,
        device,
        guidance_type=resolved_guidance_type,
        dataset_lookback=lookback,
        dataset_horizon=horizon,
    )
    coarse_model = _load_stage_model(state, "coarse", bundle["coarse_pt"], guidance_model, n_vars, device)
    fine_model = _load_stage_model(state, "fine", bundle["fine_pt"], guidance_model, n_vars, device)

    worst = _load_worst_indices(results_dir, subset_id, worst_metrics, n_worst)
    windows = _pick_windows(len(test_ds), seed=seed, n_random=n_random, worst=worst)

    saved: List[Path] = []
    for window_index, tag in windows:
        past, future = test_ds[window_index]
        past_b = past.unsqueeze(0).to(device)
        future_b = future.unsqueeze(0).to(device)
        maps = _anchor_maps(coarse_model, fine_model, past_b, future_b)
        out_path = output_dir / f"{dataset}_win{window_index}_{tag}_2d_preds.jpg"
        saved.append(
            _plot_panel(
                maps=maps,
                fine_model=fine_model,
                dataset=dataset,
                window_index=window_index,
                tag=tag,
                output_path=out_path,
                variables_to_plot=variables_to_plot,
                jpeg_dpi=jpeg_dpi,
            )
        )
        print(f"wrote {out_path}")
    return saved


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--results-dir", type=Path, default=None, help="for worst_windows.json")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument(
        "--guidance-type",
        choices=("auto", "patch_decoder", "itransformer"),
        default="auto",
        help="Guidance ckpt to load (auto prefers patch_guidance.pt when present)",
    )
    p.add_argument(
        "--test-stride",
        type=int,
        default=None,
        help="Test dataloader stride (default: staged_eval.test_stride from config, else data_subset)",
    )
    p.add_argument("--n-random", type=int, default=2)
    p.add_argument("--n-worst", type=int, default=3)
    p.add_argument("--worst-metrics", default="anchor_mse,crps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--variables-to-plot", type=int, default=3)
    p.add_argument("--jpeg-dpi", type=int, default=100)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    ckpt_dir = args.checkpoint_dir.resolve()
    results_dir = args.results_dir.resolve() if args.results_dir else None
    output_dir = args.output_dir
    if output_dir is None:
        if results_dir is not None:
            output_dir = results_dir / "viz" / "eval_2d_preds"
        else:
            output_dir = REPO_ROOT / "reports" / "staged_eval_2d_preds" / ckpt_dir.name
    output_dir = output_dir.resolve()

    metrics = [m.strip() for m in args.worst_metrics.split(",") if m.strip()]
    device = torch.device(args.device)
    run_viz(
        checkpoint_dir=ckpt_dir,
        dataset=args.dataset,
        output_dir=output_dir,
        results_dir=results_dir,
        config_path=args.config,
        guidance_type=args.guidance_type,
        test_stride=args.test_stride,
        n_random=args.n_random,
        n_worst=args.n_worst,
        worst_metrics=metrics,
        seed=args.seed,
        variables_to_plot=args.variables_to_plot,
        jpeg_dpi=args.jpeg_dpi,
        device=device,
    )


if __name__ == "__main__":
    main()
