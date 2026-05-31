"""
Dual-Scale CDF Visualization: plots coarse, fine (residual), and combined forecasts/lookbacks
side-by-side and stacked, both in 1D and 2D.

Usage:
    python viz/compare_dual_scales.py \
        --checkpoint-dir results/05-29-3812425-ETTh1-binary_dual_scale \
        --output-dir reports/3812425_binary_dual_scale_report \
        --vars 3
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


def run_dual_scale_visualization(
    checkpoint_dir: str,
    output_dir: str,
    lookback_length: int = LOOKBACK_LENGTH,
    forecast_length: int = FORECAST_LENGTH,
    diffusion_sampler: str = "anchor",
    num_inference_steps: int = 20,
    variables_to_plot: int = 3,
    sample_index: Optional[int] = None,
    random_seed: int = 42,
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

    # 2. Load diffusion model
    best_pt = sub / 'best.pt'
    diff_ckpt = torch.load(best_pt, map_location=device, weights_only=False)
    diff_type = infer_diffusion_type(diff_ckpt, meta.get('diffusion_type'))
    backbone = infer_model_type(diff_ckpt)
    applied_h = apply_checkpoint_architecture(diff_ckpt, diff_type)
    anchor_kwargs = infer_anchor_kwargs(diff_ckpt, meta)

    print(f"Diffusion architecture: type={diff_type}, backbone={backbone}, image_height={applied_h}")

    itrans_guidance = iTransformerGuidance(itrans_guidance_model)
    diff_model = create_diffusion_model(
        n_variates=n_vars,
        diffusion_type=diff_type,
        model_type=backbone,
        guidance_model=itrans_guidance,
        **anchor_kwargs,
    ).to(device)
    load_diffusion_state_keep_attached_guidance(diff_model, diff_ckpt['model_state_dict'])
    diff_model.eval()

    # Get sample data
    past, future = test_ds[sample_index]
    past_t = past.unsqueeze(0).to(device)  # (1, C, L)

    with torch.no_grad():
        # Generate with dual scale output
        res = diff_model.generate(
            past_t,
            sampler=diffusion_sampler,
            num_inference_steps=num_inference_steps,
        )

    # Extract 2D maps
    past_coarse = res['past_2d_coarse'].cpu()  # (1, V, H, W_past)
    past_fine = res['past_2d_fine'].cpu()      # (1, V, H, W_past)
    future_coarse = res['future_2d_coarse'].cpu()  # (1, V, H, W_fut)
    future_fine = res['future_2d_fine'].cpu()      # (1, V, H, W_fut)

    # Concatenate past and future 2D maps along time axis
    coarse_map_full = torch.cat([past_coarse, future_coarse], dim=-1)  # (1, V, H, W_past + W_fut)
    fine_map_full = torch.cat([past_fine, future_fine], dim=-1)        # (1, V, H, W_past + W_fut)

    # Decode 1D values from occupancy maps
    # Use the internal _decode_occupancy_in_range helper
    to_2d = diff_model.to_2d
    coarse_1d = to_2d._decode_occupancy_in_range(coarse_map_full, value_range=to_2d.max_scale, cdf_decoder="mean").cpu()
    fine_1d = to_2d._decode_occupancy_in_range(fine_map_full, value_range=to_2d.max_scale / to_2d.height, cdf_decoder="mean").cpu()
    combined_1d = coarse_1d + fine_1d

    # Slice to plot (whole sequence: past + future horizon)
    W_past = past_coarse.shape[-1]
    W_fut = future_coarse.shape[-1]
    total_len = W_past + W_fut
    t_axis = np.arange(-W_past, W_fut)

    # Ground truth full sequence
    gt_full_norm = torch.cat([past, future[:, -W_fut:]], dim=-1)  # (C, W_past + W_fut)

    # Denormalize 1D paths
    gt_full_dn = denorm(gt_full_norm, mean, std)
    coarse_dn = denorm(coarse_1d[0], mean, std)
    # Fine residual is zero-centered, scale by std without adding mean
    fine_dn = fine_1d[0] * std.view(-1, 1)
    combined_dn = denorm(combined_1d[0], mean, std)

    # Setup Plot
    n_vars_to_plot = min(variables_to_plot, n_vars)
    fig, axes = plt.subplots(
        4, n_vars_to_plot, 
        figsize=(5.5 * n_vars_to_plot, 10.0), 
        sharex='row', 
        constrained_layout=True
    )
    if n_vars_to_plot == 1:
        axes = axes.reshape(4, 1)

    for col in range(n_vars_to_plot):
        var_name = var_names[col] if col < len(var_names) else f"Var {col}"

        # ------------------ Row 1: 1D Reconstruction vs Coarse ------------------
        ax1 = axes[0, col]
        ax1.plot(t_axis, gt_full_dn[col].numpy(), color='#2196F3', linewidth=2.0, label='Ground Truth')
        ax1.plot(t_axis, coarse_dn[col].numpy(), color='#FF9800', linewidth=1.5, drawstyle='steps-mid', alpha=0.8, label='Coarse Decoded')
        ax1.plot(t_axis, combined_dn[col].numpy(), color='#E91E63', linewidth=1.5, label='Combined (Coarse+Fine)')
        ax1.axvline(x=0, color='black', linestyle=':', alpha=0.3)
        ax1.grid(True, alpha=0.15)
        ax1.set_title(f"{var_name}\n1D Comparison", fontsize=11, fontweight='semibold')
        if col == 0:
            ax1.set_ylabel("Original Value Scale", fontsize=9)
            ax1.legend(loc='lower left', fontsize=7)

        # ------------------ Row 2: 1D Residual Correction ------------------
        ax2 = axes[1, col]
        ax2.plot(t_axis, fine_dn[col].numpy(), color='#4CAF50', linewidth=1.5, label='Residual (Fine Scale)')
        ax2.axhline(y=0, color='grey', linestyle='--', alpha=0.5)
        ax2.axvline(x=0, color='black', linestyle=':', alpha=0.3)
        ax2.grid(True, alpha=0.15)
        ax2.set_title("1D Fine Residual", fontsize=10)
        if col == 0:
            ax2.set_ylabel("Residual Value Scale", fontsize=9)
            ax2.legend(loc='lower left', fontsize=7)

        # ------------------ Row 3: 2D Coarse Occupancy Map ------------------
        ax3 = axes[2, col]
        coarse_img = coarse_map_full[0, col].numpy()  # (H, Total_Len)
        im3 = ax3.imshow(coarse_img, aspect='auto', origin='lower', extent=[-W_past, W_fut, 0, applied_h], cmap='plasma')
        ax3.axvline(x=0, color='white', linestyle='--', alpha=0.6)
        ax3.set_title("2D Coarse CDF Occupancy Map", fontsize=10)
        if col == 0:
            ax3.set_ylabel("Coarse Bins (H)", fontsize=9)
        fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        # ------------------ Row 4: 2D Fine Occupancy Map ------------------
        ax4 = axes[3, col]
        fine_img = fine_map_full[0, col].numpy()      # (H, Total_Len)
        im4 = ax4.imshow(fine_img, aspect='auto', origin='lower', extent=[-W_past, W_fut, 0, applied_h], cmap='plasma')
        ax4.axvline(x=0, color='white', linestyle='--', alpha=0.6)
        ax4.set_title("2D Fine Residual CDF Occupancy Map", fontsize=10)
        ax4.set_xlabel("Time Index (t)", fontsize=9)
        if col == 0:
            ax4.set_ylabel("Fine Bins (H)", fontsize=9)
        fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    fig.suptitle(
        f'{dataset_name} • Sample {sample_index} • Dual-Scale CDF Forecast & Residual Decomposition',
        fontsize=14, fontweight='bold'
    )

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"compare_dual_scales_{dataset_name}_{subset_id}.png")
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved dual scale comparison plot to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize coarse/fine predictions and 2D occupancy maps')
    parser.add_argument('--checkpoint-dir', type=str, required=True, help='Path to checkpoint folder')
    parser.add_argument('--output-dir', type=str, required=True, help='Path to output report directory')
    parser.add_argument('--vars', type=int, default=3, help='Number of variables to plot')
    parser.add_argument('--sampler', type=str, default='anchor', help='Sampler type (e.g. anchor, dpmpp)')
    parser.add_argument('--steps', type=int, default=20, help='Inference steps for sampler')
    parser.add_argument('--index', type=int, default=None, help='Index of test sample to plot')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sample selection')
    args = parser.parse_args()

    run_dual_scale_visualization(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        variables_to_plot=args.vars,
        diffusion_sampler=args.sampler,
        num_inference_steps=args.steps,
        sample_index=args.index,
        random_seed=args.seed,
    )


if __name__ == '__main__':
    main()
