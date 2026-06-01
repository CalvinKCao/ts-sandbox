#!/usr/bin/env python3
"""Plot one test window: GT, iTrans guidance, iTrans full baseline, anchor + probabilistic diffusion.

Uses the same checkpoint layout as the 05-31 binary_dual_scale grid:
  {checkpoint_dir}/{subset_id}/best.pt
  {checkpoint_dir}/{subset_id}_itransformer_finetuned.pt  (guidance)
  {checkpoint_dir}/{subset_id}_itrans_full_dataset.pt     (standalone baseline)

Example:
  python utils/visualize_binary_dual_scale_forecast.py \\
    --checkpoint-dir results/ckpts/05-31-3828089-ETTh1-binary_dual_scale \\
    --dataset ETTh1 \\
    --output-dir reports/3838179_mmpd_binary_aligned_retry_indices_fix
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.diffusion_tsf.guidance import iTransformerGuidance
from models.diffusion_tsf.train_multivariate_pipeline import (
    LOOKBACK_LENGTH,
    FORECAST_LENGTH,
    LOOKBACK_OVERLAP,
    create_diffusion_model,
    dataset_window_lengths,
    load_dataset,
    load_diffusion_state_keep_attached_guidance,
    load_itransformer_from_checkpoint,
)
from models.diffusion_tsf.visualize_comparison import (
    apply_checkpoint_architecture,
    denorm,
    infer_anchor_kwargs,
    infer_diffusion_type,
    infer_model_type,
    choose_extra_indices,
)


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


def _load_subset_bundle(checkpoint_dir: Path, dataset: str) -> dict:
    """Pick first subset dir under checkpoint root matching dataset."""
    candidates = []
    for d in sorted(checkpoint_dir.iterdir()):
        meta = d / "metadata.json"
        best = d / "best.pt"
        if not d.is_dir() or not meta.is_file() or not best.is_file():
            continue
        with meta.open(encoding="utf-8") as f:
            meta_j = json.load(f)
        if meta_j.get("dataset_name") == dataset:
            candidates.append(
                {
                    "subset_id": meta_j["subset_id"],
                    "variate_indices": meta_j["variate_indices"],
                    "variate_names": meta_j.get("variate_names", []),
                    "best_pt": best,
                    "metadata": meta_j,
                    "root": checkpoint_dir,
                }
            )
    if not candidates:
        raise FileNotFoundError(
            f"No subset with metadata+best.pt for dataset={dataset} under {checkpoint_dir}"
        )
    return candidates[0]


def _resolve_itrans_paths(root: Path, subset_id: str) -> Tuple[Optional[Path], Optional[Path]]:
    guidance = root / f"{subset_id}_itransformer_finetuned.pt"
    full_baseline = root / f"{subset_id}_itrans_full_dataset.pt"
    return (
        guidance if guidance.is_file() else None,
        full_baseline if full_baseline.is_file() else None,
    )


def plot_forecast_panel(
    checkpoint_dir: Path,
    dataset: str,
    output_dir: Path,
    test_index: Optional[int],
    prob_samples: int,
    num_extra_lookbacks: int,
    anchor_sampler: str,
    prob_sampler: str,
    prob_steps: int,
    seed: int,
    device: torch.device,
) -> Path:
    sub = _load_subset_bundle(checkpoint_dir, dataset)
    subset_id = sub["subset_id"]
    variate_indices = sub["variate_indices"]
    n_vars = len(variate_indices)
    lookback, horizon = dataset_window_lengths(dataset)

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
        raise FileNotFoundError(f"Missing guidance checkpoint: {subset_id}_itransformer_finetuned.pt")
    guidance_model = load_itransformer_from_checkpoint(
        str(guidance_path), n_vars, device
    )
    full_model = None
    if full_path is not None:
        full_model = load_itransformer_from_checkpoint(str(full_path), n_vars, device)

    ckpt = torch.load(sub["best_pt"], map_location=device, weights_only=False)
    diff_type = infer_diffusion_type(ckpt, None)
    backbone = infer_model_type(ckpt, None)
    apply_checkpoint_architecture(ckpt, diff_type, None)
    anchor_kwargs = infer_anchor_kwargs(ckpt, sub["metadata"])
    diff_model = create_diffusion_model(
        n_variates=n_vars,
        lookback=lookback,
        horizon=horizon,
        diffusion_type=diff_type,
        model_type=backbone,
        guidance_model=iTransformerGuidance(guidance_model),
        **anchor_kwargs,
    ).to(device)
    load_diffusion_state_keep_attached_guidance(diff_model, ckpt["model_state_dict"])
    diff_model.eval()

    past, future = test_ds[test_index]
    past_t = past.unsqueeze(0).to(device)

    with torch.no_grad():
        guidance_pred = _itrans_forward(guidance_model, past_t, horizon, device)
        full_pred = (
            _itrans_forward(full_model, past_t, horizon, device)
            if full_model is not None
            else None
        )

        torch.manual_seed(seed + test_index)
        anchor_out = diff_model.generate(
            past_t,
            sampler=anchor_sampler,
            num_inference_steps=prob_steps,
        )
        anchor_pred = anchor_out.get(
            "prediction_global_norm", anchor_out["prediction"]
        ).cpu()[0]

        prob_preds: List[torch.Tensor] = []
        for k in range(prob_samples):
            torch.manual_seed(seed + 10_000 + test_index * prob_samples + k)
            out = diff_model.generate(
                past_t,
                sampler=prob_sampler,
                num_inference_steps=prob_steps,
            )
            prob_preds.append(
                out.get("prediction_global_norm", out["prediction"]).cpu()[0]
            )

    context_len = min(horizon * 2, lookback)
    t_past = np.arange(-context_len, 0)
    t_future = np.arange(0, horizon)
    future_slice = future[:, -horizon:]

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
            label=f"Diffusion anchor ({anchor_sampler})",
        )
        for k, pp in enumerate(prob_preds):
            pdn = denorm(pp[:, -horizon:], mean, std)[v].numpy()
            ax.plot(
                t_future,
                pdn,
                color="#F48FB1",
                lw=0.9,
                alpha=0.55,
                label=f"Prob sample" if v == 0 and k == 0 else "",
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
        f"anchor={anchor_sampler}, prob={prob_sampler}×{prob_samples} (steps={prob_steps})",
        fontsize=11,
        fontweight="bold",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"forecast_panel_{dataset}_{subset_id}_idx{test_index}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-index", type=int, default=None)
    parser.add_argument("--prob-samples", type=int, default=5)
    parser.add_argument("--num-extra-lookbacks", type=int, default=2)
    parser.add_argument("--anchor-sampler", type=str, default="anchor")
    parser.add_argument("--prob-sampler", type=str, default="dpmpp")
    parser.add_argument("--prob-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = plot_forecast_panel(
        args.checkpoint_dir.resolve(),
        args.dataset,
        args.output_dir.resolve(),
        args.test_index,
        args.prob_samples,
        args.num_extra_lookbacks,
        args.anchor_sampler,
        args.prob_sampler,
        args.prob_steps,
        args.seed,
        device,
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
