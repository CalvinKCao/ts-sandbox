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
import sys
from pathlib import Path

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.diffusion_tsf.pipeline.visualize_utils import generate_dual_scale_comparisons


def run_dual_scale_visualization(
    checkpoint_dir: str,
    output_dir: str,
    lookback_length: int = 96,
    forecast_length: int = 96,
    diffusion_sampler: str = "anchor",
    num_inference_steps: int = 20,
    variables_to_plot: int = 3,
    sample_index: int | None = None,
    n_samples: int = 1,
    random_seed: int = 42,
):
    import torch
    from models.diffusion_tsf.train_multivariate_pipeline import LOOKBACK_LENGTH, FORECAST_LENGTH

    lookback_length = lookback_length or LOOKBACK_LENGTH
    forecast_length = forecast_length or FORECAST_LENGTH
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    ckpt_root = Path(checkpoint_dir)
    if not ckpt_root.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_root}")

    subfolders = [
        d for d in ckpt_root.iterdir()
        if d.is_dir() and (d / 'metadata.json').exists() and (d / 'best.pt').exists()
    ]
    if not subfolders:
        raise FileNotFoundError(
            f"Could not find any subdirectory with metadata.json and best.pt under {ckpt_root}"
        )

    sub = subfolders[0]
    print(f"Using model subdirectory: {sub.name}")

    with open(sub / 'metadata.json') as f:
        meta = json.load(f)

    dataset_name = meta['dataset_name']
    subset_id = meta.get('subset_id', dataset_name)
    variate_indices = meta['variate_indices']

    ft_path = ckpt_root / f'{subset_id}_itransformer_finetuned.pt'
    if not ft_path.exists():
        ft_path = ckpt_root / f'{subset_id}_itrans_ft_hp_best.pt'
    if not ft_path.exists():
        raise FileNotFoundError(f"Could not find guidance iTransformer checkpoint at {ckpt_root}")

    sample_indices = [sample_index] if sample_index is not None else None
    paths = generate_dual_scale_comparisons(
        diff_ckpt_path=str(sub / 'best.pt'),
        itrans_ckpt_path=str(ft_path),
        dataset_name=dataset_name,
        variate_indices=variate_indices,
        output_dir=output_dir,
        device=device,
        tuned_params=meta.get('tuned_params'),
        lookback_length=lookback_length,
        forecast_length=forecast_length,
        diffusion_sampler=diffusion_sampler,
        num_inference_steps=num_inference_steps,
        variables_to_plot=variables_to_plot,
        sample_indices=sample_indices,
        n_samples=n_samples,
        random_seed=random_seed,
        jpeg_dpi=100,
        tag=f"compare_dual_scales_{dataset_name}_{subset_id}",
    )
    for p in paths:
        print(f"Saved dual scale comparison plot to: {p}")


def main():
    parser = argparse.ArgumentParser(description='Visualize coarse/fine predictions and 2D occupancy maps')
    parser.add_argument('--checkpoint-dir', type=str, required=True, help='Path to checkpoint folder')
    parser.add_argument('--output-dir', type=str, required=True, help='Path to output report directory')
    parser.add_argument('--vars', type=int, default=3, help='Number of variables to plot')
    parser.add_argument('--sampler', type=str, default='anchor', help='Sampler type (e.g. anchor, dpmpp)')
    parser.add_argument('--steps', type=int, default=20, help='Inference steps for sampler')
    parser.add_argument('--index', type=int, default=None, help='Index of test sample to plot')
    parser.add_argument('--n-samples', type=int, default=1, help='Number of random test samples')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sample selection')
    args = parser.parse_args()

    run_dual_scale_visualization(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        variables_to_plot=args.vars,
        diffusion_sampler=args.sampler,
        num_inference_steps=args.steps,
        sample_index=args.index,
        n_samples=args.n_samples,
        random_seed=args.seed,
    )


if __name__ == '__main__':
    main()
