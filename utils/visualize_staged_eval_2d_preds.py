#!/usr/bin/env python3
"""GT vs anchor-pred 2D coarse/fine CDF maps + final 1D for staged checkpoints.

Random test windows plus worst-window picks from worst_windows.json when present.

Example:
  python utils/visualize_staged_eval_2d_preds.py \\
    --checkpoint-dir results/ckpts/07-02-4041709-weather-..._fourier_flatline_blur \\
    --dataset weather \\
    --results-dir results/datasets/07-02-4041709-weather-..._fourier_flatline_blur \\
    --n-random 2 --n-worst 3
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

from models.diffusion_tsf.guidance import iTransformerGuidance
from models.diffusion_tsf.pipeline.config import load_experiment_config
from models.diffusion_tsf.pipeline.phases.staged_diffusion_pretrain import patch_stage_globals
from models.diffusion_tsf.pipeline.state import PipelineState
from models.diffusion_tsf.pipeline.visualize_utils import (
    decode_staged_anchor_components,
    pick_sample_indices,
    save_figure_jpg,
)
from models.diffusion_tsf.train_multivariate_pipeline import (
    anchor_kwargs_from_params,
    create_diffusion_model,
    load_dataset,
    load_diffusion_state_keep_attached_guidance,
    load_itransformer_from_checkpoint,
)
from utils.visualize_staged_forecast import _load_staged_bundle, _window_lengths


DEFAULT_CONFIG = (
    "configs/binary_anchor_stationary_flat_subsets_grad_accum_150_fourier_flatline_blur.yaml"
)


def _build_state(checkpoint_dir: Path, dataset: str, subset_id: str, config_path: str) -> PipelineState:
    cfg = load_experiment_config(config_path, cli_overrides={"dataset": dataset})
    state = PipelineState.from_config(cfg)
    state.checkpoint_dir = str(checkpoint_dir.resolve())
    state.dataset = dataset
    state.subset_id = subset_id
    return state


def _load_stage_model(
    state: PipelineState,
    stage: str,
    ckpt_path: Path,
    itrans_guidance: iTransformerGuidance,
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
        guidance_model=itrans_guidance,
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
    used: Set[int] = set()
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
    gt_maps = fine_model._encode_staged_maps(future_norm)
    return {
        "gt_coarse": gt_maps["coarse"][0].cpu().numpy(),
        "gt_fine": gt_maps["fine"][0].cpu().numpy(),
        "pred_coarse": coarse_out["future_2d_coarse"][0].cpu().numpy(),
        "pred_fine": fine_out["future_2d_fine"][0].cpu().numpy(),
        "future_norm": future_norm[0].cpu(),
        "coarse_out": coarse_out,
        "fine_out": fine_out,
    }


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

    coarse_np, fine_np, final_np = decode_staged_anchor_components(
        fine_model, maps["coarse_out"], maps["fine_out"],
    )
    future_norm = maps["future_norm"].numpy()
    k = int(getattr(fine_model.config, "lookback_overlap", 0) or 0)
    if k > 0:
        future_norm = future_norm[..., k:]
        coarse_np = coarse_np[..., k:]
        fine_np = fine_np[..., k:]
        final_np = final_np[..., k:]

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
                ax.set_title(f"var {col} | {label} ({h}x{w})", fontsize=8)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    t_axis = np.arange(future_norm.shape[-1])
    for col in range(n_vars):
        ax = fig.add_subplot(gs[4, col])
        ax.plot(t_axis, future_norm[col], color="#2196F3", linewidth=1.5, label="GT")
        ax.plot(t_axis, coarse_np[0, col], color="#FF9800", linewidth=1.1, label="Coarse")
        ax.plot(t_axis, fine_np[0, col], color="#4CAF50", linewidth=1.0, label="Fine")
        ax.plot(t_axis, final_np[0, col], color="#E91E63", linewidth=1.2, label="Final")
        ax.grid(True, alpha=0.12)
        ax.set_title(f"var {col} forecast 1D (window-norm)", fontsize=8)
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
    test_stride = int(data_subset.get("test_stride", 1))
    _, _, test_ds, _norm_stats = load_dataset(
        dataset,
        variate_indices,
        stride=int(data_subset.get("train_stride", state.window_stride)),
        test_stride=test_stride,
        lookback=lookback,
        horizon=horizon,
    )
    if len(test_ds) == 0:
        raise ValueError(f"Empty test set for {dataset}")

    guidance_path = checkpoint_dir / f"{subset_id}_itransformer_finetuned.pt"
    if not guidance_path.is_file():
        raise FileNotFoundError(f"Missing guidance ckpt: {guidance_path}")

    guidance_model = load_itransformer_from_checkpoint(str(guidance_path), n_vars, device)
    itrans_guidance = iTransformerGuidance(guidance_model)
    coarse_model = _load_stage_model(state, "coarse", bundle["coarse_pt"], itrans_guidance, n_vars, device)
    fine_model = _load_stage_model(state, "fine", bundle["fine_pt"], itrans_guidance, n_vars, device)

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
