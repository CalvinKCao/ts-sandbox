"""
Probabilistic forecast visualization script: samples stochastic futures.

Loads a fine-tuned diffusion model, its underlying iTransformer guidance checkpoint,
and a baseline full-dataset iTransformer model. Takes a random test set sample,
runs 5 independent stochastic generation paths, and plots them along with
the guidance prediction, baseline prediction, and ground truth. 
Also includes a few randomly sampled lookback windows to visualize data characteristics.

Usage:
    python viz/compare_samples.py \
        --checkpoint-dir results/ckpts/05-26-3037-gauss-anchor-etth1 \
        --output-dir results/ \
        --sampler dpmpp \
        --steps 20
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.diffusion_tsf.train_multivariate_pipeline import (
    LOOKBACK_LENGTH, FORECAST_LENGTH,
    create_diffusion_model, load_dataset,
    load_itransformer_from_checkpoint,
    load_diffusion_state_keep_attached_guidance,
)
from models.diffusion_tsf.guidance import iTransformerGuidance
from models.diffusion_tsf.visualize_comparison import (
    apply_checkpoint_architecture,
    denorm,
    infer_anchor_kwargs,
    infer_diffusion_type,
    infer_model_type,
)
import models.diffusion_tsf.train_multivariate_pipeline as train_pipeline


def find_run_bundle(ckpt_root: Path) -> Tuple[Path, Path, Path]:
    """Locate (subset_dir, metadata.json, best.pt) under a Slurm run stem."""
    if (ckpt_root / "metadata.json").exists() and (ckpt_root / "best.pt").exists():
        return ckpt_root, ckpt_root / "metadata.json", ckpt_root / "best.pt"

    candidates = []
    for d in ckpt_root.iterdir():
        if not d.is_dir():
            continue
        meta = d / "metadata.json"
        best = d / "best.pt"
        if meta.exists() and best.exists():
            candidates.append((d, meta, best))
    if not candidates:
        raise FileNotFoundError(
            f"No metadata.json + best.pt under {ckpt_root} "
            f"(expected subset subdir from pipeline finetune HP)."
        )
    candidates.sort(key=lambda t: t[0].name)
    return candidates[0]


def run_probabilistic_visualization(
    checkpoint_dir: str,
    output_dir: str,
    num_futures: int = 5,
    num_random_lookbacks: int = 3,
    lookback_length: int = LOOKBACK_LENGTH,
    forecast_length: int = FORECAST_LENGTH,
    diffusion_sampler: str = "ddim",
    num_inference_steps: int = 20,
    sample_index: Optional[int] = None,
    random_seed: int = 42,
    name_suffix: str = "",
    plot_all_variates: bool = True,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    ckpt_root = Path(checkpoint_dir)
    if not ckpt_root.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_root}")

    sub, meta_path, best_pt = find_run_bundle(ckpt_root)
    print(f"Using model subdirectory: {sub.name}")

    with open(meta_path) as f:
        meta = json.load(f)

    dataset_name = meta['dataset_name']
    subset_id = meta.get('subset_id', dataset_name)
    variate_indices = meta['variate_indices']
    var_names = meta.get('variate_names', [])
    n_vars = len(variate_indices)
    subset_meta = meta.get("data_subset") or {}
    train_stride = int(subset_meta.get("train_stride", meta.get("window_stride", 1)))
    test_stride = int(subset_meta.get("test_stride", 1))
    if meta.get("lookback_length"):
        lookback_length = int(meta["lookback_length"])
    if meta.get("forecast_length"):
        forecast_length = int(meta["forecast_length"])
    train_pipeline.LOOKBACK_LENGTH = lookback_length
    train_pipeline.FORECAST_LENGTH = forecast_length

    print(f"Dataset: {dataset_name} ({n_vars} variables, subset={subset_id})")

    _, _, test_ds, norm_stats = load_dataset(
        dataset_name, variate_indices,
        stride=train_stride, test_stride=test_stride,
        lookback=lookback_length, horizon=forecast_length,
    )

    n_test = len(test_ds)
    if n_test == 0:
        raise ValueError(f"No test samples available for {dataset_name}")

    if sample_index is None:
        random.seed(random_seed)
        sample_index = random.randint(0, n_test - 1)
    
    print(f"Selected test sample index: {sample_index} / {n_test}")

    mean = torch.tensor(norm_stats['mean'], dtype=torch.float32)
    std = torch.tensor(norm_stats['std'], dtype=torch.float32)

    # 1. Load guidance iTransformer
    ft_path = ckpt_root / f'{subset_id}_itransformer_finetuned.pt'
    if not ft_path.exists():
        ft_path = ckpt_root / f'{subset_id}_itrans_ft_hp_best.pt'
    if not ft_path.exists():
        raise FileNotFoundError(f"Could not find guidance iTransformer checkpoint at {ckpt_root}")
    print(f"Loading guidance iTransformer from: {ft_path}")
    itrans_guidance_model = load_itransformer_from_checkpoint(str(ft_path), n_vars, device)

    # 2. Load baseline iTransformer
    base_path = ckpt_root / f'{subset_id}_itrans_full_dataset.pt'
    itrans_baseline_model = None
    if base_path.exists():
        print(f"Loading baseline iTransformer from: {base_path}")
        itrans_baseline_model = load_itransformer_from_checkpoint(str(base_path), n_vars, device)
    else:
        print(f"Warning: Baseline iTransformer checkpoint not found at {base_path}. Proceeding without it.")

    # 3. Load diffusion model
    diff_ckpt = torch.load(best_pt, map_location=device, weights_only=False)
    diff_type = infer_diffusion_type(diff_ckpt, meta.get('diffusion_type'))
    backbone = infer_model_type(diff_ckpt)
    applied_h = apply_checkpoint_architecture(diff_ckpt, diff_type)
    if meta.get("use_dual_scale"):
        train_pipeline.USE_DUAL_SCALE = True
        train_pipeline.IMAGE_HEIGHT = int(meta.get("image_height", applied_h))
    if meta.get("disable_cross_attention") is not None:
        train_pipeline.DISABLE_CROSS_ATTENTION = bool(meta["disable_cross_attention"])
    if meta.get("use_window_normalization") is not None:
        train_pipeline.USE_WINDOW_NORMALIZATION = bool(meta["use_window_normalization"])
    anchor_kwargs = infer_anchor_kwargs(diff_ckpt, meta)
    print(f"Diffusion architecture: type={diff_type}, backbone={backbone}, image_height={applied_h}")

    itrans_guidance = iTransformerGuidance(itrans_guidance_model)
    diff_model = create_diffusion_model(
        n_variates=n_vars, diffusion_type=diff_type, model_type=backbone,
        guidance_model=itrans_guidance,
        **anchor_kwargs,
    ).to(device)
    load_diffusion_state_keep_attached_guidance(diff_model, diff_ckpt['model_state_dict'])
    diff_model.eval()

    past, future = test_ds[sample_index]
    past_t = past.unsqueeze(0).to(device)  # (1, C, L)

    # Generate random background lookbacks
    background_pasts = []
    if num_random_lookbacks > 0:
        bg_indices = random.sample(range(n_test), min(num_random_lookbacks, n_test))
        for bg_idx in bg_indices:
            if bg_idx != sample_index:
                bg_past, _ = test_ds[bg_idx]
                background_pasts.append(denorm(bg_past, mean, std))

    def run_itrans(model, past_tensor):
        with torch.no_grad():
            B, C, L = past_tensor.shape
            x_enc = past_tensor.permute(0, 2, 1)
            seq_sl = getattr(model, 'seq_len', L)
            if x_enc.shape[1] > seq_sl:
                x_enc = x_enc[:, -seq_sl:, :]
            x_dec = torch.zeros(B, forecast_length, C, device=device)
            out = model(x_enc, None, x_dec, None)
            if isinstance(out, tuple): out = out[0]
            return out.permute(0, 2, 1).cpu()[0]

    # Run models
    itrans_guidance_pred = run_itrans(itrans_guidance_model, past_t)
    itrans_baseline_pred = run_itrans(itrans_baseline_model, past_t) if itrans_baseline_model else None

    print(
        f"Sampling {num_futures} stochastic futures "
        f"({diffusion_sampler}, steps={num_inference_steps})..."
    )
    sampled_futures = []
    for f_idx in range(num_futures):
        with torch.no_grad():
            torch.manual_seed(random_seed + f_idx * 9973 + sample_index)
            res = diff_model.generate(
                past_t,
                sampler=diffusion_sampler,
                num_inference_steps=num_inference_steps,
            )
            # Use global norm prediction if available (denormalized at the window level, needs global denorm)
            pred = res.get('prediction_global_norm', res['prediction']).cpu()[0]
            sampled_futures.append(pred)

    # Denormalize
    past_dn = denorm(past, mean, std)
    future_dn = denorm(future[:, -forecast_length:], mean, std)
    guidance_dn = denorm(itrans_guidance_pred, mean, std)
    baseline_dn = denorm(itrans_baseline_pred, mean, std) if itrans_baseline_pred is not None else None
    
    diff_dns = []
    for f_pred in sampled_futures:
        f_sliced = f_pred[:, -forecast_length:] if f_pred.shape[-1] > forecast_length else f_pred
        diff_dns.append(denorm(f_sliced, mean, std))

    # Plot (all variates in the subset unless capped)
    n_plot = n_vars if plot_all_variates else min(4, n_vars)
    n_cols = min(4, n_plot)
    n_rows = (n_plot + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 3.5 * n_rows), constrained_layout=True)
    axes = axes.flatten() if n_vars > 1 else np.array([axes])

    context_len = min(forecast_length * 2, lookback_length)
    t_past = np.arange(-context_len, 0)
    t_future = np.arange(0, forecast_length)

    for col in range(n_plot):
        ax = axes[col]
        gt = future_dn[col].numpy()
        guide = guidance_dn[col].numpy()
        
        # Plot background lookbacks
        for i, bg in enumerate(background_pasts):
            ax.plot(t_past, bg[col, -context_len:].numpy(), color='#B0BEC5', alpha=0.3, linewidth=0.8,
                    label='Random Context' if (col == 0 and i == 0) else '')

        # Past context
        ax.plot(t_past, past_dn[col, -context_len:].numpy(), color='#424242', alpha=0.9, linewidth=1.5, 
                label='Context' if col == 0 else '')
        
        # Ground truth
        ax.plot(t_future, gt, color='#2196F3', linewidth=2.0, label='Ground Truth' if col == 0 else '')
        
        # Guidance & Baseline
        ax.plot(t_future, guide, color='#FF9800', linewidth=1.6, linestyle='--', label='iTrans Guidance' if col == 0 else '')
        if baseline_dn is not None:
            base = baseline_dn[col].numpy()
            ax.plot(t_future, base, color='#4CAF50', linewidth=1.6, linestyle='-.', label='iTrans Baseline' if col == 0 else '')

        # Diffusion futures
        for i, df_dn in enumerate(diff_dns):
            df = df_dn[col].numpy()
            ax.plot(t_future, df, color='#E91E63', linewidth=1.0, alpha=0.5,
                    label='Diffusion Future' if (col == 0 and i == 0) else '')

        ax.axvline(x=0, color='black', linestyle=':', alpha=0.3)
        ax.grid(True, alpha=0.2)
        vname = var_names[col] if col < len(var_names) else f'Var {col}'
        ax.set_title(vname, fontsize=11, fontweight='semibold')
        ax.tick_params(labelsize=8)

    for col in range(n_plot, len(axes)):
        fig.delaxes(axes[col])

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=10, bbox_to_anchor=(0.5, 1.05))

    fig.suptitle(f'{dataset_name} • Sample {sample_index} • Stochastic Futures (N={num_futures})',
                 fontsize=14, fontweight='bold', y=1.0 if n_rows > 1 else 1.0)

    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_{name_suffix}" if name_suffix else ""
    out_path = os.path.join(output_dir, f"compare_samples_{dataset_name}_{subset_id}_{diffusion_sampler}{suffix}.png")
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved probabilistic forecast plot to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize probabilistic diffusion futures vs baselines')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Run stem, e.g. results/ckpts/05-30-3819110-ETTm1-binary_dual_scale')
    parser.add_argument('--scan-ckpts-root', type=str, default=None,
                        help='Plot every run under this dir that has subset/best.pt (e.g. results/ckpts)')
    parser.add_argument('--run-glob', type=str, default='05-30-38216*-binary_anchor',
                        help='With --scan-ckpts-root, only matching run folder names')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: reports/<report_stem>/)')
    parser.add_argument('--report-stem', type=str, default='3821627_binary_anchor_grid',
                        help='Subfolder under reports/ for figures when --output-dir omitted')
    parser.add_argument('--require-results-json', action='store_true',
                        help='Skip runs without results/datasets/<stem>/*/results.json')
    parser.add_argument('--num-futures', type=int, default=5)
    parser.add_argument('--num-random-lookbacks', type=int, default=2)
    parser.add_argument('--sampler', type=str, default='ddim',
                        choices=['ddim', 'dpmpp', 'ddpm', 'anchor'],
                        help='Stochastic paths: prefer ddim/dpmpp (anchor is near-deterministic)')
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--index', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--name-suffix', type=str, default='')
    parser.add_argument('--max-vars', type=int, default=0,
                        help='Cap variate panels (0 = all variates in subset)')
    args = parser.parse_args()

    out_base = args.output_dir or os.path.join(project_root, 'reports', args.report_stem)

    if args.scan_ckpts_root:
        root = Path(args.scan_ckpts_root)
        runs = sorted(root.glob(args.run_glob))
        if not runs:
            raise SystemExit(f"No runs match {args.run_glob} under {root}")
        for run_dir in runs:
            try:
                find_run_bundle(run_dir)
            except FileNotFoundError as exc:
                print(f"Skip {run_dir.name}: {exc}")
                continue
            if args.require_results_json:
                rj = list(
                    (project_root / "results" / "datasets").glob(f"{run_dir.name}/*/results.json")
                )
                if not rj:
                    print(f"Skip {run_dir.name}: no results.json (incomplete eval)")
                    continue
            print(f"\n=== {run_dir.name} ===")
            run_probabilistic_visualization(
                checkpoint_dir=str(run_dir),
                output_dir=out_base,
                num_futures=args.num_futures,
                num_random_lookbacks=args.num_random_lookbacks,
                diffusion_sampler=args.sampler,
                num_inference_steps=args.steps,
                sample_index=args.index,
                random_seed=args.seed,
                name_suffix=args.name_suffix,
                plot_all_variates=(args.max_vars <= 0),
            )
        print(f"\nBatch done. Figures in {out_base}")
        return

    if not args.checkpoint_dir:
        parser.error('Provide --checkpoint-dir or --scan-ckpts-root')

    run_probabilistic_visualization(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=out_base,
        num_futures=args.num_futures,
        num_random_lookbacks=args.num_random_lookbacks,
        diffusion_sampler=args.sampler,
        num_inference_steps=args.steps,
        sample_index=args.index,
        random_seed=args.seed,
        name_suffix=args.name_suffix,
        plot_all_variates=(args.max_vars <= 0),
    )


if __name__ == '__main__':
    main()
