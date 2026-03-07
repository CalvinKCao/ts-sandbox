"""
Generate clean comparison plots: Ground Truth vs iTransformer vs Diffusion.

For each dataset, picks a few test windows and plots 1D overlays showing
how the diffusion model's forecast compares to iTransformer-only baseline.

Usage:
    python -m models.diffusion_tsf.visualize_comparison \
        --checkpoint-dir /path/to/checkpoints \
        --output-dir /path/to/output \
        --num-samples 3
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.diffusion_tsf.train_7var_pipeline import (
    CHECKPOINT_DIR, RESULTS_DIR, DATASETS_DIR, DATASET_REGISTRY,
    LOOKBACK_LENGTH, FORECAST_LENGTH, N_VARIATES,
    create_itransformer, create_diffusion_model, load_dataset,
    get_dim_for_dataset, pretrain_dir_for_dim,
)
from models.diffusion_tsf.guidance import iTransformerGuidance


def denorm(x, mean, std):
    """Denormalize (C, T) tensor using (1, C) stats."""
    m = mean.squeeze().unsqueeze(-1)   # (C, 1)
    s = std.squeeze().unsqueeze(-1)
    return x * s + m


def run_comparison(
    checkpoint_dir: str,
    output_dir: str,
    num_samples: int = 3,
    variables_to_plot: int = 3,
    diffusion_ensemble: int = 3,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ckpt_root = Path(checkpoint_dir)

    # Discover subsets and group by dataset (pick first subset per dataset)
    by_dataset = defaultdict(list)
    for d in sorted(ckpt_root.iterdir()):
        meta_path = d / 'metadata.json'
        best_path = d / 'best.pt'
        if not d.is_dir() or not meta_path.exists() or not best_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        by_dataset[meta['dataset_name']].append({
            'subset_id': meta['subset_id'],
            'variate_indices': meta['variate_indices'],
            'variate_names': meta.get('variate_names', []),
            'best_pt': str(best_path),
        })

    if not by_dataset:
        print(f"No subsets found in {checkpoint_dir}")
        return

    print(f"Found {len(by_dataset)} datasets: {', '.join(sorted(by_dataset))}")
    os.makedirs(output_dir, exist_ok=True)

    # Cache loaded iTransformer models per-dim
    _itrans_cache = {}

    def _get_itrans_model(dim):
        if dim in _itrans_cache:
            return _itrans_cache[dim]
        # Try new per-dim layout first, fall back to legacy flat layout
        dim_dir = pretrain_dir_for_dim(dim, base_dir=checkpoint_dir)
        candidates = [
            os.path.join(dim_dir, 'itransformer.pt'),
            os.path.join(checkpoint_dir, 'pretrained_itransformer.pt'),
        ]
        for p in candidates:
            if os.path.exists(p):
                model = create_itransformer(num_vars=dim).to(device)
                ckpt = torch.load(p, map_location=device, weights_only=False)
                model.load_state_dict(ckpt['model_state_dict'])
                model.eval()
                _itrans_cache[dim] = model
                print(f"  Loaded iTransformer (dim={dim}) from {p}")
                return model
        print(f"  WARNING: no iTransformer checkpoint for dim={dim}")
        return None

    for dataset_name, subsets in sorted(by_dataset.items()):
        sub = subsets[0]
        subset_id = sub['subset_id']
        variate_indices = sub['variate_indices']
        var_names = sub['variate_names']
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name} (subset: {subset_id})")
        print(f"{'='*60}")

        try:
            _, _, test_ds, norm_stats = load_dataset(
                dataset_name, variate_indices, stride=LOOKBACK_LENGTH
            )
        except ValueError as e:
            print(f"  Skipping {dataset_name}: {e}")
            continue

        n_test = len(test_ds)
        if n_test == 0:
            print(f"  No test samples for {dataset_name}")
            continue

        mean = torch.tensor(norm_stats['mean'], dtype=torch.float32)
        std = torch.tensor(norm_stats['std'], dtype=torch.float32)

        # Load per-dim iTransformer for this dataset
        n_vars = len(variate_indices)
        itrans_model = _get_itrans_model(n_vars)
        if itrans_model is None:
            print(f"  Skipping {dataset_name}: no iTransformer for dim={n_vars}")
            continue

        # Load fine-tuned diffusion with iTransformer guidance
        diff_model = create_diffusion_model(n_variates=n_vars, use_guidance=True).to(device)
        itrans_guidance = iTransformerGuidance(
            itrans_model, use_norm=True,
            seq_len=LOOKBACK_LENGTH, pred_len=FORECAST_LENGTH
        )
        diff_model.set_guidance_model(itrans_guidance)
        diff_ckpt = torch.load(sub['best_pt'], map_location=device, weights_only=False)
        diff_model.load_state_dict(diff_ckpt['model_state_dict'])
        diff_model.eval()

        sample_indices = np.linspace(0, n_test - 1, min(num_samples, n_test), dtype=int)
        n_vars_plot = min(variables_to_plot, len(variate_indices))
        n_rows = len(sample_indices)

        fig, axes = plt.subplots(
            n_rows, n_vars_plot,
            figsize=(5.5 * n_vars_plot, 3.2 * n_rows),
            squeeze=False,
        )

        for row, idx in enumerate(sample_indices):
            past, future = test_ds[idx]
            past_t = past.unsqueeze(0).to(device)   # (1, C, L)

            with torch.no_grad():
                # iTransformer standalone
                B, C, L = past_t.shape
                x_enc = past_t.permute(0, 2, 1)       # (1, L, C)
                x_dec = torch.zeros(B, FORECAST_LENGTH, C, device=device)
                itrans_out = itrans_model(x_enc, None, x_dec, None)
                if isinstance(itrans_out, tuple):
                    itrans_out = itrans_out[0]
                itrans_pred = itrans_out.permute(0, 2, 1).cpu()[0]   # (C, F)

                # Diffusion prediction (single or averaged)
                if diffusion_ensemble <= 1:
                    result = diff_model.generate(past_t)
                    diff_pred = result['prediction'].cpu()[0]  # (C, F)
                else:
                    diff_preds = []
                    for _ in range(diffusion_ensemble):
                        result = diff_model.generate(past_t)
                        diff_preds.append(result['prediction'].cpu())
                    diff_pred = torch.stack(diff_preds).mean(0)[0]  # (C, F)

            # Denormalize everything to original scale
            past_dn = denorm(past, mean, std)
            future_dn = denorm(future, mean, std)
            itrans_dn = denorm(itrans_pred, mean, std)
            diff_dn = denorm(diff_pred, mean, std)

            # Show last N steps of context for visual continuity
            context_len = min(FORECAST_LENGTH * 2, LOOKBACK_LENGTH)
            t_past = np.arange(-context_len, 0)
            t_future = np.arange(0, FORECAST_LENGTH)

            for col in range(n_vars_plot):
                ax = axes[row, col]

                gt = future_dn[col].numpy()
                it = itrans_dn[col].numpy()
                df = diff_dn[col].numpy()

                # Context
                ax.plot(t_past, past_dn[col, -context_len:].numpy(),
                        color='#9E9E9E', alpha=0.5, linewidth=0.8)
                # Ground truth
                ax.plot(t_future, gt, color='#2196F3', linewidth=1.6,
                        label='Ground Truth' if row == 0 and col == 0 else '')
                # iTransformer baseline
                ax.plot(t_future, it, color='#FF9800', linewidth=1.2,
                        linestyle='--', alpha=0.85,
                        label='iTransformer' if row == 0 and col == 0 else '')
                # Diffusion (iTransformer-guided)
                ax.plot(t_future, df, color='#E91E63', linewidth=1.2,
                        label='Diffusion' if row == 0 and col == 0 else '')

                ax.axvline(x=0, color='black', linestyle=':', alpha=0.25)

                # Per-cell MAE annotations
                it_mae = np.mean(np.abs(it - gt))
                df_mae = np.mean(np.abs(df - gt))
                ax.text(0.97, 0.97,
                        f'iTrans MAE: {it_mae:.3f}\nDiff MAE: {df_mae:.3f}',
                        transform=ax.transAxes, fontsize=7,
                        va='top', ha='right',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

                vname = var_names[col] if col < len(var_names) else f'Var {col}'
                if row == 0:
                    ax.set_title(vname, fontsize=10)
                if col == 0:
                    ax.set_ylabel(f'Sample {row + 1}', fontsize=10)
                ax.tick_params(labelsize=7)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=10,
                   bbox_to_anchor=(0.5, 1.01))
        mode_label = 'single sample' if diffusion_ensemble <= 1 else f'{diffusion_ensemble}-sample avg'
        fig.suptitle(f'{dataset_name}  ({mode_label})',
                     fontsize=14, fontweight='bold', y=1.04)

        plt.tight_layout()
        out_path = os.path.join(output_dir, f'comparison_{dataset_name}.png')
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {out_path}")

    print(f"\nDone! All comparison plots in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Diffusion vs iTransformer comparison plots')
    parser.add_argument('--checkpoint-dir', type=str, default=CHECKPOINT_DIR)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--num-samples', type=int, default=3, help='Samples per dataset')
    parser.add_argument('--vars', type=int, default=3, help='Variables to plot per sample')
    parser.add_argument('--ensemble', type=int, default=1,
                        help='Diffusion samples to average (1=single sample, 30=full avg)')
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(RESULTS_DIR, 'viz', 'comparison')
    run_comparison(args.checkpoint_dir, output_dir, args.num_samples, args.vars, args.ensemble)


if __name__ == '__main__':
    main()
