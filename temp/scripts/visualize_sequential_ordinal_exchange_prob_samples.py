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
  python temp/scripts/visualize_sequential_ordinal_exchange_prob_samples.py
  python temp/scripts/visualize_sequential_ordinal_exchange_prob_samples.py \\
    --n-windows 5 --n-prob-samples 10 --variables-to-plot 2
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

REPO = Path(__file__).resolve().parents[2]
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """GT coarse/fine 2D, dual-decoded GT 1D (repr W), and past_norm 1D (raw lookback)."""
    past_norm, future_norm, _ = model._normalize_sequence(past_b, future_b)
    fut_maps = model._encode_staged_maps(future_norm)
    coarse = fut_maps["coarse"][0].detach().cpu().numpy()
    fine = fut_maps["fine"][0].detach().cpu().numpy()
    gt_1d = model._decode_staged_combined_1d(
        fut_maps["coarse"], fut_maps["fine"],
    )[0].detach().cpu().numpy()
    past_1d = past_norm[0].detach().cpu().numpy()
    return coarse, fine, gt_1d, past_1d


@torch.no_grad()
def _prob_sample_chain(
    coarse_model: torch.nn.Module,
    fine_model: torch.nn.Module,
    past_b: torch.Tensor,
    *,
    sampler: str,
    num_steps: int,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    coarse_out = coarse_model.generate(
        past_b, sampler=sampler, num_inference_steps=num_steps,
    )
    fine_out = fine_model.generate(
        past_b,
        sampler=sampler,
        num_inference_steps=num_steps,
        future_coarse_2d=coarse_out["future_2d_coarse"],
    )
    combined = fine_out["prediction_norm"][0].detach().cpu().numpy()
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
    """Wide stacked rows: GT, sample-mean, then every prob sample (one lookback)."""
    s_stack = np.stack([s[var_idx] for s in samples], axis=0)  # (S,H,W)
    mean_map = s_stack.mean(axis=0)
    panels: List[Tuple[str, np.ndarray]] = [
        ("GT", gt[var_idx]),
        (f"sample mean (n={len(samples)})", mean_map),
    ]
    panels += [(f"prob s{i}", s_stack[i]) for i in range(len(samples))]

    h, w = panels[0][1].shape
    # Wide canvas: ~1" per ~28 horizon cols, floor at 12" so stride-2 hz≈356 isn't squashed.
    panel_w = max(12.0, w / 28.0)
    panel_h = 2.4
    n_rows = len(panels)
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(panel_w, panel_h * n_rows + 0.6),
        constrained_layout=True,
        sharex=True,
    )
    if n_rows == 1:
        axes = [axes]
    for ax, (title, data) in zip(axes, panels):
        im = ax.imshow(
            data,
            aspect="auto",
            origin="lower",
            extent=[0, w, 0, h],
            cmap="plasma",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        ax.set_ylabel(f"{title}\nbin", fontsize=8)
        ax.set_yticks([0, h // 2, h] if h >= 2 else [0])
        fig.colorbar(im, ax=ax, fraction=0.012, pad=0.01)
    axes[-1].set_xlabel("horizon t (repr cols)")
    fig.suptitle(
        f"exchange_rate win={window_index} var={var_idx} | {kind} 2D CDF "
        f"(GT + {len(samples)} prob samples, same lookback)",
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure_jpg(fig, str(out_path), dpi=jpeg_dpi)
    return out_path


def _plot_combined_1d(
    *,
    past_1d: np.ndarray,
    gt_1d: np.ndarray,
    samples_1d: Sequence[np.ndarray],
    var_idx: int,
    window_index: int,
    k_overlap: int,
    lookback: int,
    horizon: int,
    out_path: Path,
    jpeg_dpi: int,
) -> Path:
    """Lookback context + all prob samples + mean/band + GT on one wide plot."""
    s = np.stack([x[var_idx] for x in samples_1d], axis=0)
    gt_v = gt_1d[var_idx]
    if s.shape[-1] != gt_v.shape[-1]:
        raise ValueError(
            f"combined 1D length mismatch: samples T={s.shape[-1]} vs GT T={gt_v.shape[-1]} "
            "(GT must be dual-decoded from staged maps, not raw future_norm)"
        )
    if k_overlap > 0 and s.shape[-1] > k_overlap:
        s_core = s[..., k_overlap:]
        gt_core = gt_v[..., k_overlap:]
    else:
        s_core = s
        gt_core = gt_v

    context_len = min(int(horizon) * 2, int(lookback), int(past_1d.shape[-1]))
    past_v = past_1d[var_idx, -context_len:]
    t_past = np.arange(-context_len, 0)
    t_fut = np.arange(s_core.shape[-1])
    mean = s_core.mean(axis=0)
    n_s = len(s_core)
    if n_s >= 2:
        lo, hi = np.quantile(s_core, [0.1, 0.9], axis=0)
    else:
        lo = hi = mean

    fig_w = max(14.0, (context_len + s_core.shape[-1]) / 45.0)
    fig, ax = plt.subplots(figsize=(fig_w, 4.0), constrained_layout=True)
    ax.plot(t_past, past_v, color="#9E9E9E", linewidth=1.0, alpha=0.75, label="lookback")
    ax.axvline(0, color="k", linestyle=":", alpha=0.35)
    if n_s >= 2:
        ax.fill_between(t_fut, lo, hi, color="#90CAF9", alpha=0.28, label="p10–p90")
    cmap = plt.get_cmap("tab10")
    for i in range(n_s):
        ax.plot(
            t_fut,
            s_core[i],
            color=cmap(i % 10),
            linewidth=1.0,
            alpha=0.55,
            label=f"prob s{i}" if i < 10 else None,
        )
    ax.plot(t_fut, mean, color="#0D47A1", linewidth=2.0, label="sample mean")
    ax.plot(t_fut, gt_core, color="#C62828", linewidth=2.0, label="GT")
    ax.set_xlabel("t (lookback < 0 | horizon ≥ 0, after overlap)")
    ax.set_ylabel("ordinal / model space")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=7, loc="best", ncol=2, framealpha=0.9)
    ax.set_title(
        f"exchange_rate win={window_index} var={var_idx} | lookback+horizon 1D "
        f"({n_s} prob samples, same lookback)",
        fontsize=11,
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
    p.add_argument("--n-windows", type=int, default=5)
    p.add_argument("--n-prob-samples", type=int, default=10)
    p.add_argument("--sampler", default="quad_t")
    p.add_argument("--num-inference-steps", type=int, default=20)
    p.add_argument("--test-stride", type=int, default=4)
    p.add_argument("--variables-to-plot", type=int, default=2)
    p.add_argument("--jpeg-dpi", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
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
    print(f"n_windows={args.n_windows} n_prob_samples={args.n_prob_samples}")

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

    k_overlap = int(fine_model._overlap_repr_cols())
    win_idxs = _evenly_spaced_indices(len(test_ds), args.n_windows)
    print(f"test_len={len(test_ds)} windows={win_idxs}")

    n_plot_vars = min(args.variables_to_plot, n_vars)
    saved: List[Path] = []
    for wi in win_idxs:
        past, future = test_ds[wi]
        past_b = past.unsqueeze(0).to(device)
        future_b = future.unsqueeze(0).to(device)
        gt_c, gt_f, gt_1d, past_1d = _gt_maps(fine_model, past_b, future_b)

        samples: List[Dict[str, np.ndarray]] = []
        for s_i in range(args.n_prob_samples):
            samples.append(
                _prob_sample_chain(
                    coarse_model,
                    fine_model,
                    past_b,
                    sampler=args.sampler,
                    num_steps=args.num_inference_steps,
                    seed=int(args.seed) + 10_000 + int(wi) * 100 + int(s_i),
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
                    past_1d=past_1d,
                    gt_1d=gt_1d,
                    samples_1d=[s["combined_1d"] for s in samples],
                    var_idx=v,
                    window_index=wi,
                    k_overlap=k_overlap,
                    lookback=lookback,
                    horizon=horizon,
                    out_path=out_dir / f"win{wi:04d}_var{v}_combined_1d.jpg",
                    jpeg_dpi=args.jpeg_dpi,
                )
            )
        print(f"wrote panels for window {wi}")

    print(f"done: {len(saved)} images → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
