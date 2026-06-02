#!/usr/bin/env python3
"""Forecast panel for coarse→fine staged binary diffusion (2-stage grid).

One random test window: all variates with GT, iTrans guidance, full-dataset iTrans
baseline, staged anchor (coarse→fine), and N probabilistic dpmpp samples. Extra
lookback-only rows for context.

Example:
  python utils/visualize_staged_forecast.py \\
    --checkpoint-dir results/ckpts/06-02-3849018-ETTh1-binary_dual_scale_staged \\
    --dataset ETTh1 \\
    --output-dir reports/06-01_cfg_ablation_mmpd_matrix_combined/viz_2stage/ETTh1
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGED_CONFIG = REPO_ROOT / "configs" / "binary_dual_scale_staged.yaml"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.diffusion_tsf.guidance import iTransformerGuidance
from models.diffusion_tsf.pipeline.config import load_experiment_config
from models.diffusion_tsf.pipeline.phases.staged_diffusion_pretrain import patch_stage_globals
from models.diffusion_tsf.pipeline.state import PipelineState
from models.diffusion_tsf.train_multivariate_pipeline import (
    anchor_kwargs_from_params,
    create_diffusion_model,
    dataset_window_lengths,
    load_dataset,
    load_diffusion_state_keep_attached_guidance,
    load_itransformer_from_checkpoint,
)
from models.diffusion_tsf.visualize_comparison import choose_extra_indices, denorm
from utils.visualize_binary_dual_scale_forecast import _itrans_forward


def _load_staged_bundle(checkpoint_dir: Path, dataset: str) -> Dict[str, Any]:
    """Find subset dir under staged run root with coarse/fine best.pt for *dataset*."""
    candidates: List[Dict[str, Any]] = []
    for sub_dir in sorted(checkpoint_dir.iterdir()):
        if not sub_dir.is_dir():
            continue
        coarse_pt = sub_dir / "coarse" / "best.pt"
        fine_pt = sub_dir / "fine" / "best.pt"
        fine_meta_path = sub_dir / "fine" / "metadata.json"
        if not (coarse_pt.is_file() and fine_pt.is_file() and fine_meta_path.is_file()):
            continue
        with fine_meta_path.open(encoding="utf-8") as f:
            fine_meta = json.load(f)
        if fine_meta.get("dataset_name") != dataset:
            continue
        coarse_meta_path = sub_dir / "coarse" / "metadata.json"
        coarse_meta: Dict[str, Any] = {}
        if coarse_meta_path.is_file():
            with coarse_meta_path.open(encoding="utf-8") as f:
                coarse_meta = json.load(f)
        candidates.append(
            {
                "subset_id": fine_meta["subset_id"],
                "variate_indices": fine_meta["variate_indices"],
                "variate_names": fine_meta.get("variate_names", []),
                "coarse_pt": coarse_pt,
                "fine_pt": fine_pt,
                "fine_metadata": fine_meta,
                "coarse_metadata": coarse_meta,
                "root": checkpoint_dir,
            }
        )
    if not candidates:
        raise FileNotFoundError(
            f"No staged coarse/fine best.pt for dataset={dataset} under {checkpoint_dir}"
        )
    return candidates[0]


def _build_pipeline_state(checkpoint_dir: Path, dataset: str, subset_id: str) -> PipelineState:
    cfg = load_experiment_config(
        str(STAGED_CONFIG),
        cli_overrides={"dataset": dataset},
    )
    state = PipelineState.from_config(cfg)
    state.checkpoint_dir = str(checkpoint_dir.resolve())
    state.dataset = dataset
    state.subset_id = subset_id
    return state


def _load_staged_diffusion(
    state: PipelineState,
    stage: str,
    ckpt_path: Path,
    itrans_guidance: iTransformerGuidance,
    n_vars: int,
    device: torch.device,
) -> torch.nn.Module:
    import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod

    patch_stage_globals(pipeline_mod, state, stage, honor_dataset_windows=True)
    lookback, horizon = dataset_window_lengths(state.dataset)
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


def _resolve_itrans_paths(root: Path, subset_id: str) -> Tuple[Optional[Path], Optional[Path]]:
    guidance = root / f"{subset_id}_itransformer_finetuned.pt"
    full_baseline = root / f"{subset_id}_itrans_full_dataset.pt"
    return (
        guidance if guidance.is_file() else None,
        full_baseline if full_baseline.is_file() else None,
    )


def _staged_anchor_and_samples(
    coarse_model: torch.nn.Module,
    fine_model: torch.nn.Module,
    past: torch.Tensor,
    *,
    prob_samples: int,
    prob_sampler: str,
    prob_steps: int,
    seed: int,
    test_index: int,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Return fine-stage anchor (C, T) and list of prob forecasts in norm space."""
    with torch.no_grad():
        torch.manual_seed(seed + test_index)
        coarse_det = coarse_model.generate(past, sampler="anchor", num_inference_steps=prob_steps)
        fine_det = fine_model.generate(
            past,
            sampler="anchor",
            num_inference_steps=prob_steps,
            future_coarse_2d=coarse_det["future_2d_coarse"],
        )
        anchor = fine_det.get("prediction_global_norm", fine_det["prediction"]).cpu()[0]

        prob_preds: List[torch.Tensor] = []
        prob_kwargs = {"sampler": prob_sampler, "num_inference_steps": prob_steps}
        for k in range(prob_samples):
            sample_seed = seed + 10_000 + test_index * prob_samples + k
            torch.manual_seed(sample_seed)
            coarse_s = coarse_model.generate(past, **prob_kwargs)
            torch.manual_seed(sample_seed)
            fine_s = fine_model.generate(
                past,
                future_coarse_2d=coarse_s["future_2d_coarse"],
                **prob_kwargs,
            )
            prob_preds.append(
                fine_s.get("prediction_global_norm", fine_s["prediction"]).cpu()[0]
            )
    return anchor, prob_preds


def plot_staged_forecast_panel(
    checkpoint_dir: Path,
    dataset: str,
    output_dir: Path,
    test_index: Optional[int],
    prob_samples: int,
    num_extra_lookbacks: int,
    prob_sampler: str,
    prob_steps: int,
    seed: int,
    device: torch.device,
) -> Path:
    sub = _load_staged_bundle(checkpoint_dir, dataset)
    subset_id = sub["subset_id"]
    variate_indices = sub["variate_indices"]
    n_vars = len(variate_indices)
    lookback, horizon = dataset_window_lengths(dataset)

    state = _build_pipeline_state(checkpoint_dir, dataset, subset_id)

    _, _, test_ds, norm_stats = load_dataset(
        dataset,
        variate_indices,
        stride=1,
        test_stride=1,
        lookback=lookback,
        horizon=horizon,
    )
    n_test = len(test_ds)
    if n_test == 0:
        raise ValueError(f"Empty test set for {dataset}")

    rng = random.Random(seed)
    if test_index is None:
        test_index = rng.randrange(n_test)
    extra_indices = choose_extra_indices(
        n_test, num_extra_lookbacks, rng, exclude=[test_index]
    )

    mean = torch.tensor(norm_stats["mean"], dtype=torch.float32)
    std = torch.tensor(norm_stats["std"], dtype=torch.float32)

    guidance_path, full_path = _resolve_itrans_paths(checkpoint_dir, subset_id)
    if guidance_path is None:
        raise FileNotFoundError(
            f"Missing guidance checkpoint: {subset_id}_itransformer_finetuned.pt under {checkpoint_dir}"
        )
    guidance_model = load_itransformer_from_checkpoint(
        str(guidance_path), n_vars, device
    )
    full_model = None
    if full_path is not None:
        full_model = load_itransformer_from_checkpoint(str(full_path), n_vars, device)

    itrans_guidance = iTransformerGuidance(guidance_model)
    coarse_model = _load_staged_diffusion(
        state, "coarse", sub["coarse_pt"], itrans_guidance, n_vars, device
    )
    fine_model = _load_staged_diffusion(
        state, "fine", sub["fine_pt"], itrans_guidance, n_vars, device
    )

    past, future = test_ds[test_index]
    past_t = past.unsqueeze(0).to(device)

    with torch.no_grad():
        guidance_pred = _itrans_forward(guidance_model, past_t, horizon, device)
        full_pred = (
            _itrans_forward(full_model, past_t, horizon, device)
            if full_model is not None
            else None
        )

    anchor_pred, prob_preds = _staged_anchor_and_samples(
        coarse_model,
        fine_model,
        past_t,
        prob_samples=prob_samples,
        prob_sampler=prob_sampler,
        prob_steps=prob_steps,
        seed=seed,
        test_index=test_index,
    )

    K = getattr(coarse_model.config, "lookback_overlap", 0)
    context_len = min(horizon * 2, lookback)
    t_past = np.arange(-context_len, 0)
    t_future = np.arange(0, horizon)
    future_slice = future[:, -horizon:]
    if K > 0:
        future_slice = future_slice[..., K:]

    fig, axes = plt.subplots(
        n_vars + num_extra_lookbacks,
        1,
        figsize=(11, 2.2 * (n_vars + num_extra_lookbacks)),
        squeeze=False,
        constrained_layout=True,
    )

    names = sub["variate_names"] or [f"v{i}" for i in range(n_vars)]

    for v in range(n_vars):
        ax = axes[v, 0]
        past_dn = denorm(past, mean, std)[v].numpy()
        gt = denorm(future_slice, mean, std)[v].numpy()
        gdn = denorm(guidance_pred, mean, std)[v].numpy()
        adn = denorm(anchor_pred[:, -horizon:], mean, std)[v].numpy()

        ax.plot(t_past, past_dn[-context_len:], color="#9E9E9E", lw=0.9, alpha=0.6)
        ax.plot(t_future, gt, color="#2196F3", lw=1.8, label="Ground truth")
        ax.plot(
            t_future,
            gdn,
            color="#FF9800",
            lw=1.2,
            ls="--",
            label="iTrans guidance (finetuned)",
        )
        if full_pred is not None:
            fdn = denorm(full_pred[:, -horizon:], mean, std)[v].numpy()
            ax.plot(
                t_future,
                fdn,
                color="#4CAF50",
                lw=1.2,
                ls="-.",
                label="iTrans full baseline",
            )
        ax.plot(
            t_future,
            adn,
            color="#E91E63",
            lw=1.4,
            label="2-stage anchor (coarse→fine)",
        )
        for k, pp in enumerate(prob_preds):
            pdn = denorm(pp[:, -horizon:], mean, std)[v].numpy()
            ax.plot(
                t_future,
                pdn,
                color="#F48FB1",
                lw=0.9,
                alpha=0.55,
                label="Prob sample" if v == 0 and k == 0 else "",
            )
        ax.axvline(0, color="k", ls=":", alpha=0.25)
        ax.set_ylabel(names[v] if v < len(names) else f"var {v}", fontsize=9)
        if v == 0:
            ax.legend(loc="upper right", fontsize=7, ncol=2)
        ax.grid(alpha=0.2)

    for row_off, idx in enumerate(extra_indices, start=n_vars):
        ax = axes[row_off, 0]
        lb_past, _ = test_ds[idx]
        lb_dn = denorm(lb_past, mean, std)
        for v in range(n_vars):
            ax.plot(
                t_past,
                lb_dn[v, -context_len:].numpy(),
                color="#546E7A",
                lw=0.9,
                alpha=0.7 if v > 0 else 1.0,
            )
        ax.axvline(0, color="k", ls=":", alpha=0.25)
        ax.set_ylabel(f"lookback ctx {row_off - n_vars + 1}", fontsize=8)
        ax.grid(alpha=0.2)

    fig.suptitle(
        f"{dataset} / {subset_id} — test idx {test_index} | "
        f"2-stage anchor + {prob_sampler}×{prob_samples} (steps={prob_steps})",
        fontsize=11,
        fontweight="bold",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"forecast_panel_{dataset}_{subset_id}_idx{test_index}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def pick_staged_ckpt_dir(ckpt_root: Path, dataset: str) -> Path:
    """Newest staged run dir that has eval-ready coarse/fine checkpoints for *dataset*."""
    candidates: List[Path] = []
    for d in ckpt_root.iterdir():
        if not d.is_dir() or not d.name.endswith("binary_dual_scale_staged"):
            continue
        try:
            _load_staged_bundle(d, dataset)
            candidates.append(d)
        except FileNotFoundError:
            continue
    if not candidates:
        raise FileNotFoundError(
            f"No binary_dual_scale_staged ckpt for {dataset} under {ckpt_root}"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-index", type=int, default=None)
    parser.add_argument("--prob-samples", type=int, default=5)
    parser.add_argument("--num-extra-lookbacks", type=int, default=2)
    parser.add_argument("--prob-sampler", type=str, default="dpmpp")
    parser.add_argument("--prob-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = plot_staged_forecast_panel(
        args.checkpoint_dir.resolve(),
        args.dataset,
        args.output_dir.resolve(),
        args.test_index,
        args.prob_samples,
        args.num_extra_lookbacks,
        args.prob_sampler,
        args.prob_steps,
        args.seed,
        device,
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
