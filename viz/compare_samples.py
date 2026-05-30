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
from typing import Optional, List

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
)
from models.diffusion_tsf.guidance import iTransformerGuidance
from models.diffusion_tsf.visualize_comparison import (
    apply_checkpoint_architecture,
    denorm,
    infer_anchor_kwargs,
    infer_diffusion_type,
    infer_model_type,
    infer_prediction_mode,
)


def run_probabilistic_visualization(
    checkpoint_dir: str,
    output_dir: str,
    num_futures: int = 5,
    num_random_lookbacks: int = 3,
    lookback_length: int = LOOKBACK_LENGTH,
    forecast_length: int = FORECAST_LENGTH,
    diffusion_sampler: str = "dpmpp",
    num_inference_steps: int = 20,
    sample_index: Optional[int] = None,
    random_seed: int = 42,
    name_suffix: str = "",
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    ckpt_root = Path(checkpoint_dir)
    if not ckpt_root.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_root}")

    subfolders = [d for d in ckpt_root.iterdir() if d.is_dir() and (d / 'metadata.json').exists() and (d / 'best.pt').exists()]
    if not subfolders:
        raise FileNotFoundError(f"Could not find any subdirectory with metadata.json and best.pt under {ckpt_root}")
    
    sub = subfolders[0]
    print(f"Using model subdirectory: {sub.name}")
    
    with open(sub / 'metadata.json') as f:
        meta = json.load(f)

    dataset_name = meta['dataset_name']
    subset_id = meta.get('subset_id', dataset_name)
    variate_indices = meta['variate_indices']
    var_names = meta.get('variate_names', [])
    n_vars = len(variate_indices)

    print(f"Dataset: {dataset_name} ({n_vars} variables)")

    _, _, test_ds, norm_stats = load_dataset(
        dataset_name, variate_indices, stride=1,
        lookback=lookback_length, horizon=forecast_length
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
    best_pt = sub / 'best.pt'
    diff_ckpt = torch.load(best_pt, map_location=device, weights_only=False)
    diff_type = infer_diffusion_type(diff_ckpt, meta.get('diffusion_type'))
    backbone = infer_model_type(diff_ckpt)
    pred_mode = infer_prediction_mode(diff_ckpt)
    applied_h = apply_checkpoint_architecture(diff_ckpt, diff_type)
    anchor_kwargs = infer_anchor_kwargs(diff_ckpt, meta)
    print(f"Diffusion architecture: type={diff_type}, backbone={backbone}, prediction_mode={pred_mode}, image_height={applied_h}")

    diff_model = create_diffusion_model(
        n_variates=n_vars, diffusion_type=diff_type, model_type=backbone,
        prediction_mode=pred_mode, **anchor_kwargs,
    ).to(device)
    
    itrans_guidance = iTransformerGuidance(itrans_guidance_model)
    diff_model.set_guidance_model(itrans_guidance)
    diff_model.load_state_dict(diff_ckpt['model_state_dict'])
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

    print(f"Sampling {num_futures} futures using {diffusion_sampler} sampler...")
    sampled_futures = []
    for f_idx in range(num_futures):
        with torch.no_grad():
            res = diff_model.generate(past_t, sampler=diffusion_sampler, num_inference_steps=num_inference_steps)
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

    # Plot
    n_cols = min(4, n_vars)
    n_rows = (n_vars + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 3.5 * n_rows), constrained_layout=True)
    axes = axes.flatten() if n_vars > 1 else np.array([axes])

    context_len = min(forecast_length * 2, lookback_length)
    t_past = np.arange(-context_len, 0)
    t_future = np.arange(0, forecast_length)

    for col in range(n_vars):
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

    for col in range(n_vars, len(axes)):
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
    parser.add_argument('--checkpoint-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--num-futures', type=int, default=5)
    parser.add_argument('--num-random-lookbacks', type=int, default=3)
    parser.add_argument('--sampler', type=str, default='dpmpp')
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--index', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--name-suffix', type=str, default='')
    args = parser.parse_args()

    run_probabilistic_visualization(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        num_futures=args.num_futures,
        num_random_lookbacks=args.num_random_lookbacks,
        diffusion_sampler=args.sampler,
        num_inference_steps=args.steps,
        sample_index=args.index,
        random_seed=args.seed,
        name_suffix=args.name_suffix,
    )


if __name__ == '__main__':
    main()
