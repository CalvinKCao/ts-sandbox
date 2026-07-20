#!/usr/bin/env python3
"""Probabilistic coarse/fine/1D viz for the latest sequential ordinal exchange_rate run.

Default run (most recent sequential coarse→fine + ordinal_norm on exchange_rate):
  07-12-4213914-exchange_rate-…_ordinal_norm_g10p0
  CRPS≈0.609, phases: coarse → fine (not vertical_dual)

Cluster paths (submit_binary / slurm_worker layout):
  $SCRATCH/ts-sandbox/results/ckpts/<run>/exchange_rate/{coarse,fine}/best.pt
  $SCRATCH/ts-sandbox/results/ckpts/<run>/exchange_rate_patch_guidance.pt

Example (on Killarney login/interactive GPU, or after syncing ckpts locally):
  source .venv/bin/activate
  python temp/visualize_sequential_ordinal_exchange_prob_samples.py
  python temp/visualize_sequential_ordinal_exchange_prob_samples.py \\
    --n-windows 10 --n-prob-samples 10 --variables-to-plot 2
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from models.diffusion_tsf.pipeline.globals_bridge import patch_globals
from models.diffusion_tsf.pipeline.visualize_utils import save_figure_jpg
from models.diffusion_tsf.train_multivariate_pipeline import load_dataset
from utils.visualize_staged_eval_2d_preds import (
    _build_state,
    _load_stage_model,
    _resolve_guidance_ckpt,
)
from utils.visualize_staged_forecast import _load_staged_bundle, _window_lengths
from models.diffusion_tsf.train_multivariate_pipeline import load_wrapped_guidance

DEFAULT_RUN = (
    "07-12-4213914-exchange_rate-"
    "binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_g10p0"
)
DEFAULT_CONFIG = (
    "configs/binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_g10p0.yaml"
)
DEFAULT_DATASET = "exchange_rate"


def _repo_root_candidates() -> List[Path]:
    out: List[Path] = [REPO]
    scratch = os.environ.get("SCRATCH")
    if scratch:
        out.insert(0, Path(scratch) / "ts-sandbox")
    return out


def _resolve_checkpoint_dir(explicit: Optional[Path], run_name: str) -> Path:
    if explicit is not None:
        p = explicit.expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(f"--checkpoint-dir not found: {p}")
        return p
    for root in _repo_root_candidates():
        cand = root / "results" / "ckpts" / run_name
        coarse = cand / "exchange_rate" / "coarse" / "best.pt"
        fine = cand / "exchange_rate" / "fine" / "best.pt"
        if coarse.is_file() and fine.is_file():
            return cand
    tried = [str(r / "results" / "ckpts" / run_name) for r in _repo_root_candidates()]
    raise FileNotFoundError(
        "Could not find coarse/fine best.pt for run. Tried:\n  - "
        + "\n  - ".join(tried)
        + "\nSync from cluster or pass --checkpoint-dir explicitly."
    )


def _evenly_spaced_indices(n: int, k: int) -> List[int]:
    if n <= 0:
        return []
    k = min(k, n)
    if k == 1:
        return [0]
    return sorted({int(round(i)) for i in np.linspace(0, n - 1, k)})


@torch.no_grad()
def _gt_maps(
    model: torch.nn.Module,
    past_b: torch.Tensor,
    future_b: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return GT coarse/fine 2D (V,H,W_fut) and GT combined 1D in model space (V,W_fut)."""
    past_norm, future_norm, _ = model._normalize_sequence(past_b, future_b)
    fut_maps = model._encode_staged_maps(future_norm)
    coarse = fut_maps["coarse"][0].detach().cpu().numpy()
    fine = fut_maps["fine"][0].detach().cpu().numpy()
    gt_1d = future_norm[0].detach().cpu().numpy()
    return coarse, fine, gt_1d


@torch.no_grad()
def _prob_sample_chain(
    coarse_model: torch.nn.Module,
    fine_model: torch.nn.Module,
    past_b: torch.Tensor,
    *,
    sampler: str,
    num_steps: int,
) -> Dict[str, np.ndarray]:
    coarse_out = coarse_model.generate(
        past_b, sampler=sampler, num_inference_steps=num_steps,
    )
    fine_out = fine_model.generate(
        past_b,
        sampler=sampler,
        num_inference_steps=num_steps,
        future_coarse_2d=coarse_out["future_2d_coarse"],
    )
    # Prefer fine model's ordinal-aware dual decode of the sampled maps.
    combined = fine_model._decode_staged_combined_1d(
        fine_out["future_2d_coarse"],
        fine_out["future_2d_fine"],
    )[0].detach().cpu().numpy()
    return {
        "coarse": fine_out["future_2d_coarse"][0].detach().cpu().numpy(),
        "fine": fine_out["future_2d_fine"][0].detach().cpu().numpy(),
        "combined_1d": combined,
    }


def _plot_coarse_or_fine_2d(
    *,
    kind: str,
    gt: np.ndarray,
    samples: Sequence[np.ndarray],
    var_idx: int,
    window_index: int,
    out_path: Path,
    jpeg_dpi: int,
) -> Path:
    """GT | sample-mean | up to 8 sample maps for one variate."""
    s_stack = np.stack([s[var_idx] for s in samples], axis=0)  # (S,H,W)
    mean_map = s_stack.mean(axis=0)
    show = min(8, len(samples))
    n_cols = 2 + show
    fig, axes = plt.subplots(1, n_cols, figsize=(2.2 * n_cols, 3.2), constrained_layout=True)
    if n_cols == 1:
        axes = [axes]
    panels = [("GT", gt[var_idx]), ("sample mean", mean_map)]
    panels += [(f"s{i}", s_stack[i]) for i in range(show)]
    for ax, (title, data) in zip(axes, panels):
        h, w = data.shape
        im = ax.imshow(
            data, aspect="auto", origin="lower", extent=[0, w, 0, h],
            cmap="plasma", vmin=0.0, vmax=1.0,
        )
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("horizon t")
        ax.set_ylabel("bin")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"exchange_rate win={window_index} var={var_idx} | {kind} 2D CDF "
        f"(GT vs {len(samples)} prob samples)",
        fontsize=10,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure_jpg(fig, str(out_path), dpi=jpeg_dpi)
    return out_path


def _plot_combined_1d(
    *,
    gt_1d: np.ndarray,
    samples_1d: Sequence[np.ndarray],
    var_idx: int,
    window_index: int,
    k_overlap: int,
    out_path: Path,
    jpeg_dpi: int,
) -> Path:
    """Horizon 1D: GT + faint samples + mean (ordinal/model space)."""
    s = np.stack([x[var_idx] for x in samples_1d], axis=0)
    # Drop lookback-overlap prefix on the future canvas if present.
    if k_overlap > 0 and s.shape[-1] > k_overlap:
        s_core = s[..., k_overlap:]
        gt_core = gt_1d[var_idx, k_overlap:]
    else:
        s_core = s
        gt_core = gt_1d[var_idx]
    t = np.arange(gt_core.shape[-1])
    mean = s_core.mean(axis=0)
    lo, hi = np.quantile(s_core, [0.1, 0.9], axis=0)

    fig, ax = plt.subplots(figsize=(10, 3.2), constrained_layout=True)
    ax.fill_between(t, lo, hi, color="#90CAF9", alpha=0.35, label="p10–p90")
    for i in range(len(s_core)):
        ax.plot(t, s_core[i], color="#90CAF9", linewidth=0.6, alpha=0.35)
    ax.plot(t, mean, color="#1565C0", linewidth=1.6, label="sample mean")
    ax.plot(t, gt_core, color="#C62828", linewidth=1.8, label="GT")
    ax.set_xlabel("horizon step (after overlap)")
    ax.set_ylabel("ordinal / model space")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8, loc="best")
    ax.set_title(
        f"exchange_rate win={window_index} var={var_idx} | combined 1D "
        f"({len(samples_1d)} prob samples)",
        fontsize=10,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure_jpg(fig, str(out_path), dpi=jpeg_dpi)
    return out_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", default=DEFAULT_RUN)
    p.add_argument("--checkpoint-dir", type=Path, default=None)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--n-windows", type=int, default=10)
    p.add_argument("--n-prob-samples", type=int, default=10)
    p.add_argument("--sampler", default="quad_t")
    p.add_argument("--num-inference-steps", type=int, default=20)
    p.add_argument("--test-stride", type=int, default=4)
    p.add_argument("--variables-to-plot", type=int, default=2)
    p.add_argument("--jpeg-dpi", type=int, default=120)
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    ckpt_dir = _resolve_checkpoint_dir(args.checkpoint_dir, args.run_name)
    out_dir = args.output_dir or (
        REPO / "results" / "viz" / "sequential_ordinal_exchange_prob_samples" / args.run_name
    )
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"checkpoint_dir={ckpt_dir}")
    print(f"output_dir={out_dir}")
    print(f"device={device}")

    bundle = _load_staged_bundle(ckpt_dir, args.dataset)
    if bundle.get("stage") == "vertical_dual":
        raise RuntimeError(f"Run is vertical_dual, expected sequential coarse/fine: {ckpt_dir}")
    subset_id = bundle["subset_id"]
    variate_indices = bundle["variate_indices"]
    n_vars = len(variate_indices)
    state = _build_state(ckpt_dir, args.dataset, subset_id, args.config)
    lookback, horizon = _window_lengths(args.dataset, state)

    import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod

    patch_globals(pipeline_mod, state, honor_dataset_windows=True)
    data_subset = bundle["fine_metadata"].get("data_subset") or {}
    _, _, test_ds, norm_stats = load_dataset(
        args.dataset,
        variate_indices,
        stride=int(data_subset.get("train_stride", state.window_stride)),
        test_stride=int(args.test_stride),
        lookback=lookback,
        horizon=horizon,
        ordinal_tie_atol=float(state.ordinal_tie_atol),
        use_ordinal_window_norm=state.use_ordinal_window_norm,
    )
    if len(test_ds) == 0:
        raise RuntimeError("empty test set")
    if norm_stats.get("ordinal_ladder") is not None:
        state.extra["global_ordinal_ladder"] = norm_stats["ordinal_ladder"]
        pipeline_mod.GLOBAL_ORDINAL_LADDER = norm_stats["ordinal_ladder"]

    guidance_path, gtype = _resolve_guidance_ckpt(ckpt_dir, subset_id, "patch_decoder")
    guidance = load_wrapped_guidance(
        str(guidance_path),
        n_vars,
        device,
        guidance_type=gtype,
        dataset_lookback=lookback,
        dataset_horizon=horizon,
    )
    coarse_model = _load_stage_model(
        state, "coarse", bundle["coarse_pt"], guidance, n_vars, device,
    )
    fine_model = _load_stage_model(
        state, "fine", bundle["fine_pt"], guidance, n_vars, device,
    )
    ranked = bool(getattr(test_ds, "yields_ordinal_ranks", False))
    for m in (coarse_model, fine_model):
        m._ordinal_input_is_ranked = ranked
        m._ordinal_apply_ood_shift = bool(not ranked)
    if bool(state.use_ordinal_window_norm) and not fine_model._uses_global_ordinal_encoding():
        raise RuntimeError("ordinal_ladder not wired into fine model")

    k_overlap = int(getattr(fine_model.config, "lookback_overlap", 0) or 0)
    win_idxs = _evenly_spaced_indices(len(test_ds), args.n_windows)
    print(f"test_len={len(test_ds)} windows={win_idxs}")

    n_plot_vars = min(args.variables_to_plot, n_vars)
    saved: List[Path] = []
    for wi in win_idxs:
        past, future = test_ds[wi]
        past_b = past.unsqueeze(0).to(device)
        future_b = future.unsqueeze(0).to(device)
        gt_c, gt_f, gt_1d = _gt_maps(fine_model, past_b, future_b)

        samples: List[Dict[str, np.ndarray]] = []
        for s_i in range(args.n_prob_samples):
            samples.append(
                _prob_sample_chain(
                    coarse_model,
                    fine_model,
                    past_b,
                    sampler=args.sampler,
                    num_steps=args.num_inference_steps,
                )
            )
            print(f"  win={wi} sample {s_i+1}/{args.n_prob_samples}")

        for v in range(n_plot_vars):
            saved.append(
                _plot_coarse_or_fine_2d(
                    kind="coarse",
                    gt=gt_c,
                    samples=[s["coarse"] for s in samples],
                    var_idx=v,
                    window_index=wi,
                    out_path=out_dir / f"win{wi:04d}_var{v}_coarse_2d.jpg",
                    jpeg_dpi=args.jpeg_dpi,
                )
            )
            saved.append(
                _plot_coarse_or_fine_2d(
                    kind="fine",
                    gt=gt_f,
                    samples=[s["fine"] for s in samples],
                    var_idx=v,
                    window_index=wi,
                    out_path=out_dir / f"win{wi:04d}_var{v}_fine_2d.jpg",
                    jpeg_dpi=args.jpeg_dpi,
                )
            )
            saved.append(
                _plot_combined_1d(
                    gt_1d=gt_1d,
                    samples_1d=[s["combined_1d"] for s in samples],
                    var_idx=v,
                    window_index=wi,
                    k_overlap=k_overlap,
                    out_path=out_dir / f"win{wi:04d}_var{v}_combined_1d.jpg",
                    jpeg_dpi=args.jpeg_dpi,
                )
            )
        print(f"wrote panels for window {wi}")

    print(f"done: {len(saved)} images → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
