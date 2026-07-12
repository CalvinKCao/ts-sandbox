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

  # q99.5 norm-cal grid (auto-pick job 3852944–3852955 ckpt):
  python utils/visualize_staged_forecast.py --staged-norm --dataset ETTh1 \\
    --output-dir reports/06-01_cfg_ablation_mmpd_matrix_combined/viz_2stage_q995/ETTh1

  # All 12 datasets:
  python utils/visualize_report_staged_norm_grid.py
"""

from __future__ import annotations

import argparse
import json
import random
import re
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

# Jobs 3852944–3852955: q99.5 max_scale_by_dataset + window_norm_std_floor retrain
STAGED_NORM_JOB_RE = re.compile(
    r"06-02-(385294[4-9]|385295[0-5])-(.+)-binary_dual_scale_staged$"
)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.diffusion_tsf.guidance import iTransformerGuidance
from models.diffusion_tsf.pipeline.config import load_experiment_config
from models.diffusion_tsf.pipeline.phases.staged_diffusion_pretrain import patch_stage_globals
from models.diffusion_tsf.pipeline.state import PipelineState
from models.diffusion_tsf.train_multivariate_pipeline import (
    anchor_kwargs_from_params,
    create_diffusion_model,
    load_dataset,
    load_diffusion_state_keep_attached_guidance,
    load_itransformer_from_checkpoint,
)
from models.diffusion_tsf.visualize_comparison import choose_extra_indices, denorm


def _itrans_forward(
    model: torch.nn.Module,
    past: torch.Tensor,
    forecast_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Past (1, C, L) -> forecast (C, F) in normalized space."""
    B, C, L = past.shape
    x_enc = past.permute(0, 2, 1)
    seq_sl = getattr(model, "seq_len", L)
    if x_enc.shape[1] > seq_sl:
        x_enc = x_enc[:, -seq_sl:, :]
    x_dec = torch.zeros(B, forecast_length, C, device=device)
    out = model(x_enc, None, x_dec, None)
    if isinstance(out, tuple):
        out = out[0]
    return out.permute(0, 2, 1).cpu()[0]


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


def _window_lengths(dataset: str, state: PipelineState) -> Tuple[int, int]:
    """Per-dataset windows; do not use train_multivariate_pipeline globals (staged load mutates them)."""
    return state.lookback_length, state.forecast_length


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


def _resolve_itrans_paths(root: Path, subset_id: str) -> Tuple[Optional[Path], Optional[Path]]:
    guidance = root / f"{subset_id}_itransformer_finetuned.pt"
    full_baseline = root / f"{subset_id}_itrans_full_dataset.pt"
    return (
        guidance if guidance.is_file() else None,
        full_baseline if full_baseline.is_file() else None,
    )


def ensure_full_dataset_itrans_baseline(
    checkpoint_dir: Path,
    dataset: str,
    subset_id: str,
    variate_indices: List[int],
    device: torch.device,
    *,
    data_subset: Optional[Dict[str, Any]] = None,
    test_stride: int = 1,
) -> Path:
    """Train or reuse standalone iTrans baseline (not the diffusion guidance ckpt)."""
    ckpt_path = checkpoint_dir / f"{subset_id}_itrans_full_dataset.pt"
    if ckpt_path.is_file():
        return ckpt_path

    import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod
    from models.diffusion_tsf.train_multivariate_pipeline import (
        train_subset_itransformer_full_baseline,
    )

    old_ckpt_dir = pipeline_mod.CHECKPOINT_DIR
    pipeline_mod.CHECKPOINT_DIR = str(checkpoint_dir.resolve())
    try:
        train_subset_itransformer_full_baseline(
            dataset,
            variate_indices,
            subset_id,
            device,
            train_stride=(data_subset or {}).get("train_stride"),
            test_stride=test_stride,
            data_subset=data_subset,
        )
    finally:
        pipeline_mod.CHECKPOINT_DIR = old_ckpt_dir

    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Full-dataset iTrans baseline not created at {ckpt_path}"
        )
    return ckpt_path


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
    *,
    ensure_full_baseline: bool = False,
    model_label: str = "2-stage",
) -> Path:
    sub = _load_staged_bundle(checkpoint_dir, dataset)
    subset_id = sub["subset_id"]
    variate_indices = sub["variate_indices"]
    n_vars = len(variate_indices)
    state = _build_pipeline_state(checkpoint_dir, dataset, subset_id)
    lookback, horizon = _window_lengths(dataset, state)

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

    data_subset = sub["fine_metadata"].get("data_subset") or {}
    test_stride = int(data_subset.get("test_stride", 1))

    guidance_path, full_path = _resolve_itrans_paths(checkpoint_dir, subset_id)
    if guidance_path is None:
        raise FileNotFoundError(
            f"Missing guidance checkpoint: {subset_id}_itransformer_finetuned.pt under {checkpoint_dir}"
        )
    if ensure_full_baseline and full_path is None:
        full_path = ensure_full_dataset_itrans_baseline(
            checkpoint_dir,
            dataset,
            subset_id,
            variate_indices,
            device,
            data_subset=data_subset,
            test_stride=test_stride,
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

    K = int(getattr(coarse_model.config, "lookback_overlap", 0) or 0)
    context_len = min(horizon * 2, lookback)
    t_past = np.arange(-context_len, 0)
    future_slice = future[:, -horizon:]
    if K > 0:
        future_slice = future_slice[..., K:]
    t_fut_len = int(future_slice.shape[-1])
    t_future = np.arange(0, t_fut_len)

    def _forecast_tail(pred: torch.Tensor) -> torch.Tensor:
        tail = pred[:, -horizon:]
        return tail[..., K:] if K > 0 else tail

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
        g_pred = guidance_pred[0] if guidance_pred.dim() == 3 else guidance_pred
        gdn = denorm(_forecast_tail(g_pred), mean, std)[v].numpy()
        adn = denorm(_forecast_tail(anchor_pred), mean, std)[v].numpy()

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
            fdn = denorm(_forecast_tail(full_pred), mean, std)[v].numpy()
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
            label=f"{model_label} anchor (coarse→fine)",
        )
        for k, pp in enumerate(prob_preds):
            pdn = denorm(_forecast_tail(pp), mean, std)[v].numpy()
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
        f"{model_label} anchor + {prob_sampler}×{prob_samples} (steps={prob_steps})",
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


def pick_staged_norm_ckpt_dir(ckpt_root: Path, dataset: str) -> Path:
    """Newest q99.5 norm-cal staged run (jobs 3852944–3852955) for *dataset*."""
    candidates: List[Path] = []
    for d in ckpt_root.iterdir():
        if not d.is_dir():
            continue
        m = STAGED_NORM_JOB_RE.match(d.name)
        if not m or m.group(2) != dataset:
            continue
        try:
            _load_staged_bundle(d, dataset)
            candidates.append(d)
        except FileNotFoundError:
            continue
    if not candidates:
        raise FileNotFoundError(
            f"No q99.5 staged ckpt (3852944–3852955) for {dataset} under {ckpt_root}"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Staged run ckpt dir (default: auto-pick under --ckpt-root)",
    )
    parser.add_argument(
        "--ckpt-root",
        type=Path,
        default=REPO_ROOT / "results" / "ckpts",
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-index", type=int, default=None)
    parser.add_argument("--prob-samples", type=int, default=5)
    parser.add_argument("--num-extra-lookbacks", type=int, default=2)
    parser.add_argument("--prob-sampler", type=str, default="dpmpp")
    parser.add_argument("--prob-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ensure-full-baseline",
        action="store_true",
        help="Train full-dataset iTrans baseline if missing (slow first run)",
    )
    parser.add_argument(
        "--staged-norm",
        action="store_true",
        help="Pick q99.5 norm-cal ckpt (3852944–3852955) instead of newest staged dir",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.checkpoint_dir is not None:
        ckpt_dir = args.checkpoint_dir.resolve()
    elif args.staged_norm:
        ckpt_dir = pick_staged_norm_ckpt_dir(args.ckpt_root.resolve(), args.dataset)
    else:
        ckpt_dir = pick_staged_ckpt_dir(args.ckpt_root.resolve(), args.dataset)
    out = plot_staged_forecast_panel(
        ckpt_dir,
        args.dataset,
        args.output_dir.resolve(),
        args.test_index,
        args.prob_samples,
        args.num_extra_lookbacks,
        args.prob_sampler,
        args.prob_steps,
        args.seed,
        device,
        ensure_full_baseline=args.ensure_full_baseline,
        model_label="2-stage (q99.5 MS)" if args.staged_norm else "2-stage",
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
