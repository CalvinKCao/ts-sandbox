"""
Probabilistic forecast visualization script: samples 5 stochastic futures.

Loads a fine-tuned diffusion model and its underlying iTransformer guidance
checkpoint, takes a random test set sample, runs 5 independent stochastic
generation paths (using dpmpp or ddim), and plots the 5 futures against
both the iTransformer baseline and the ground truth.

Usage:
    python -m models.diffusion_tsf.visualize_probabilistic \
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
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.diffusion_tsf.train_multivariate_pipeline import (
    RESULTS_DIR,
    LOOKBACK_LENGTH, FORECAST_LENGTH, LOOKBACK_OVERLAP,
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

    # Discover the subfolder containing metadata/best weights
    subfolders = [d for d in ckpt_root.iterdir() if d.is_dir() and (d / 'metadata.json').exists() and (d / 'best.pt').exists()]
    if not subfolders:
        raise FileNotFoundError(f"Could not find any subdirectory with metadata.json and best.pt under {ckpt_root}")
    
    sub = subfolders[0]
    print(f"Using model subdirectory: {sub.name}")
    
    with open(sub / 'metadata.json') as f:
        meta = json.load(f)

    dataset_name = meta['dataset_name']
    subset_id = meta['subset_id']
    variate_indices = meta['variate_indices']
    var_names = meta.get('variate_names', [])
    n_vars = len(variate_indices)

    print(f"Dataset: {dataset_name} ({n_vars} variables)")

    # Load dataset
    _, _, test_ds, norm_stats = load_dataset(
        dataset_name, variate_indices, stride=1,
        lookback=lookback_length, horizon=forecast_length
    )

    n_test = len(test_ds)
    if n_test == 0:
        raise ValueError(f"No test samples available for {dataset_name}")

    # Pick a random sample index or use the specified one
    if sample_index is None:
        random.seed(random_seed)
        sample_index = random.randint(0, n_test - 1)
    
    print(f"Selected test sample index: {sample_index} / {n_test}")

    mean = torch.tensor(norm_stats['mean'], dtype=torch.float32)
    std = torch.tensor(norm_stats['std'], dtype=torch.float32)

    # Load iTransformer baseline / guidance checkpoint
    ft_path = ckpt_root / f'{subset_id}_itransformer_finetuned.pt'
    if not ft_path.exists():
        # Fall back to hp best
        ft_path = ckpt_root / f'{subset_id}_itrans_ft_hp_best.pt'
    
    if not ft_path.exists():
        raise FileNotFoundError(f"Could not find finetuned iTransformer checkpoint at {ckpt_root}")

    print(f"Loading finetuned iTransformer baseline from: {ft_path}")
    itrans_model = load_itransformer_from_checkpoint(str(ft_path), n_vars, device)

    # Load diffusion model checkpoint (match train arch: binary + image_height from ckpt)
    best_pt = sub / 'best.pt'
    diff_ckpt = torch.load(best_pt, map_location=device, weights_only=False)
    meta_diffusion = meta.get('diffusion_type')
    diff_type = infer_diffusion_type(diff_ckpt, meta_diffusion)
    backbone = infer_model_type(diff_ckpt)
    pred_mode = infer_prediction_mode(diff_ckpt)
    applied_h = apply_checkpoint_architecture(diff_ckpt, diff_type)
    anchor_kwargs = infer_anchor_kwargs(diff_ckpt, meta)
    print(
        f"Diffusion architecture: type={diff_type}, backbone={backbone}, "
        f"prediction_mode={pred_mode}, image_height={applied_h}"
    )

    diff_model = create_diffusion_model(
        n_variates=n_vars,
        diffusion_type=diff_type,
        model_type=backbone,
        prediction_mode=pred_mode,
        **anchor_kwargs,
    ).to(device)
    
    itrans_guidance = iTransformerGuidance(itrans_model)
    diff_model.set_guidance_model(itrans_guidance)
    diff_model.load_state_dict(diff_ckpt['model_state_dict'])
    diff_model.eval()

    # Get sample data
    past, future = test_ds[sample_index]
    past_t = past.unsqueeze(0).to(device)  # (1, C, L)

    # 1. Run standalone baseline iTransformer
    with torch.no_grad():
        B, C, L = past_t.shape
        x_enc = past_t.permute(0, 2, 1)
        seq_sl = getattr(itrans_model, 'seq_len', L)
        if x_enc.shape[1] > seq_sl:
            x_enc = x_enc[:, -seq_sl:, :]
        x_dec = torch.zeros(B, forecast_length, C, device=device)
        itrans_out = itrans_model(x_enc, None, x_dec, None)
        if isinstance(itrans_out, tuple):
            itrans_out = itrans_out[0]
        itrans_pred = itrans_out.permute(0, 2, 1).cpu()[0]  # (C, F)

    # 2. Run diffusion model 5 times stochastically to sample futures
    print(f"Sampling {num_futures} futures using {diffusion_sampler} sampler...")
    sampled_futures = []
    for f_idx in range(num_futures):
        with torch.no_grad():
            res = diff_model.generate(
                past_t,
                sampler=diffusion_sampler,
                num_inference_steps=num_inference_steps,
            )
            sampled_futures.append(res['prediction'].cpu()[0])  # (C, F)

    # Denormalize all predictions
    past_dn = denorm(past, mean, std)
    future_sliced = future[:, -forecast_length:]
    future_dn = denorm(future_sliced, mean, std)
    itrans_dn = denorm(itrans_pred, mean, std)
    
    diff_dns = []
    for f_pred in sampled_futures:
        f_sliced = f_pred[:, -forecast_length:] if f_pred.shape[-1] > forecast_length else f_pred
        diff_dns.append(denorm(f_sliced, mean, std))

    # Grid layout calculation
    n_cols = min(4, n_vars)
    n_rows = (n_vars + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 3.2 * n_rows),
        constrained_layout=True,
    )
    # Ensure axes is a flat array
    axes = axes.flatten() if n_vars > 1 else np.array([axes])

    context_len = min(forecast_length * 2, lookback_length)
    t_past = np.arange(-context_len, 0)
    t_future = np.arange(0, forecast_length)

    for col in range(n_vars):
        ax = axes[col]
        gt = future_dn[col].numpy()
        it = itrans_dn[col].numpy()

        # 1. Plot past context
        ax.plot(t_past, past_dn[col, -context_len:].numpy(),
                color='#757575', alpha=0.6, linewidth=1.0, label='Context' if col == 0 else '')
        
        # 2. Plot ground truth future
        ax.plot(t_future, gt, color='#2196F3', linewidth=2.0,
                label='Ground Truth' if col == 0 else '')
        
        # 3. Plot baseline iTransformer prediction
        ax.plot(t_future, it, color='#FF9800', linewidth=1.6, linestyle='--',
                label='iTransformer' if col == 0 else '')

        # 4. Plot the 5 sampled diffusion futures
        for i, df_dn in enumerate(diff_dns):
            df = df_dn[col].numpy()
            ax.plot(t_future, df, color='#E91E63', linewidth=0.9, alpha=0.45,
                    label=f'Diffusion Futures' if (col == 0 and i == 0) else '')

        ax.axvline(x=0, color='black', linestyle=':', alpha=0.3)
        ax.grid(True, alpha=0.2)

        vname = var_names[col] if col < len(var_names) else f'Var {col}'
        ax.set_title(vname, fontsize=11, fontweight='semibold')
        ax.tick_params(labelsize=8)

    # Hide unused subplots in the grid
    for col in range(n_vars, len(axes)):
        fig.delaxes(axes[col])

    # Show single combined legend at the top
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=10,
                   bbox_to_anchor=(0.5, 1.03))

    fig.suptitle(
        f'{dataset_name} • Sample {sample_index} • Stochastic Futures (N={num_futures})',
        fontsize=13, fontweight='bold', y=0.98 if n_rows > 1 else 0.95
    )

    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_{name_suffix}" if name_suffix else ""
    out_path = os.path.join(
        output_dir,
        f"probabilistic_forecast_{dataset_name}_{subset_id}_{diffusion_sampler}{suffix}.png",
    )
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved probabilistic forecast plot to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize probabilistic diffusion futures')
    parser.add_argument('--checkpoint-dir', type=str, required=True,
                        help='Path to checkpoint directory (e.g. results/ckpts/05-26-3037-gauss-anchor-etth1)')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory to save the generated visualization')
    parser.add_argument('--num-futures', type=int, default=5,
                        help='Number of stochastic futures to generate')
    parser.add_argument('--sampler', type=str, default='dpmpp',
                        choices=['ddim', 'dpmpp', 'ddpm', 'anchor'],
                        help='Stochastic sampler to generate futures')
    parser.add_argument('--steps', type=int, default=20,
                        help='Number of inference steps')
    parser.add_argument('--index', type=int, default=None,
                        help='Specific test sample index (randomly selected if not provided)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument(
        '--name-suffix',
        type=str,
        default='',
        help='Extra token before .png (e.g. h128) to avoid overwriting other heights.',
    )
    args = parser.parse_args()

    run_probabilistic_visualization(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        num_futures=args.num_futures,
        diffusion_sampler=args.sampler,
        num_inference_steps=args.steps,
        sample_index=args.index,
        random_seed=args.seed,
        name_suffix=args.name_suffix,
    )


if __name__ == '__main__':
    main()
