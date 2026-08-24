"""
Multivariate comparison plots: ground truth vs iTransformer vs diffusion.

Walks checkpoint subdirs with metadata.json + best.pt, loads the **finetuned**
per-subset iTransformer (`{subset_id}_itransformer_finetuned.pt`) plus diffusion,
so curves match training/eval. Falls back to synthetic `pretrained_dim*/itransformer.pt`
only if finetuned weights are missing.

Usage:
    python -m models.diffusion_tsf.visualize_comparison \\
        --checkpoint-dir /path/to/checkpoints \\
        --output-dir /path/to/output \\
        --num-samples 3 --vars 3
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from collections import defaultdict
from typing import List, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.diffusion_tsf.storage_paths import RESULTS_DIR, resolve_checkpoint_dir
import models.diffusion_tsf.train_multivariate_pipeline as train_pipeline
from models.diffusion_tsf.train_multivariate_pipeline import (
    create_diffusion_model, load_dataset,
    pretrain_dir_for_dim,
    load_itransformer_from_checkpoint,
)
LOOKBACK_LENGTH = 96
FORECAST_LENGTH = 96
DEFAULT_IMAGE_HEIGHT = 16
from models.diffusion_tsf.guidance import iTransformerGuidance


def denorm(x, mean, std):
    """Denormalize (C, T) tensor using (1, C) stats."""
    m = mean.squeeze().unsqueeze(-1)   # (C, 1)
    s = std.squeeze().unsqueeze(-1)
    return x * s + m


def infer_diffusion_type(ckpt: dict, override: Optional[str] = None) -> str:
    if override:
        return override
    cfg = ckpt.get('config')
    if hasattr(cfg, 'diffusion_type'):
        return cfg.diffusion_type
    if isinstance(cfg, dict):
        return cfg.get('diffusion_type', 'binary')
    sd = ckpt.get('model_state_dict', {})
    head_bias = sd.get('noise_predictor.head.bias')
    if head_bias is not None:
        patch_area = int(train_pipeline.DIT_PATCH_SIZE[0] * train_pipeline.DIT_PATCH_SIZE[1])
        if int(head_bias.shape[0]) == 2 * patch_area:
            return 'binary'
    for key, value in sd.items():
        if key.endswith('noise_predictor.final_conv.weight') and value.shape[0] == 2:
            return 'binary'
        if key.endswith('noise_predictor.head.bias') and value.shape[0] == 2:
            return 'binary'
    return 'binary'


def infer_model_type(ckpt: dict, override: Optional[str] = None) -> str:
    if override:
        return override
    cfg = ckpt.get('config')
    if hasattr(cfg, 'model_type'):
        return cfg.model_type
    if isinstance(cfg, dict) and cfg.get('model_type'):
        return cfg['model_type']
    sd = ckpt.get('model_state_dict', {})
    for key in sd:
        if 'noise_predictor.blocks.' in key:
            return 'dit'
    return 'dit'


def infer_image_height(ckpt: dict, override: Optional[int] = None) -> int:
    if override is not None:
        return int(override)
    cfg = ckpt.get('config')
    if hasattr(cfg, 'image_height'):
        return int(cfg.image_height)
    if isinstance(cfg, dict) and cfg.get('image_height') is not None:
        return int(cfg['image_height'])
    bin_centers = ckpt.get('model_state_dict', {}).get('to_2d.bin_centers')
    if bin_centers is not None:
        return int(bin_centers.shape[0])
    return DEFAULT_IMAGE_HEIGHT


def apply_checkpoint_architecture(ckpt: dict, diffusion_type: str, image_height: Optional[int] = None) -> int:
    """Match train_multivariate_pipeline globals to checkpoint (DiT + binary height)."""
    height = infer_image_height(ckpt, image_height)
    train_pipeline.IMAGE_HEIGHT = height
    cfg = ckpt.get('config')
    if hasattr(cfg, 'disable_cross_attention'):
        train_pipeline.DISABLE_CROSS_ATTENTION = bool(cfg.disable_cross_attention)
    elif isinstance(cfg, dict) and 'disable_cross_attention' in cfg:
        train_pipeline.DISABLE_CROSS_ATTENTION = bool(cfg['disable_cross_attention'])
    if hasattr(cfg, 'cross_variate_context_bias'):
        train_pipeline.CROSS_VARIATE_CONTEXT_BIAS = float(cfg.cross_variate_context_bias)
    elif isinstance(cfg, dict) and 'cross_variate_context_bias' in cfg:
        train_pipeline.CROSS_VARIATE_CONTEXT_BIAS = float(cfg['cross_variate_context_bias'])
    if hasattr(cfg, 'use_window_normalization'):
        train_pipeline.USE_WINDOW_NORMALIZATION = bool(cfg.use_window_normalization)
    elif isinstance(cfg, dict) and 'use_window_normalization' in cfg:
        train_pipeline.USE_WINDOW_NORMALIZATION = bool(cfg['use_window_normalization'])
    if hasattr(cfg, 'window_norm_center'):
        train_pipeline.WINDOW_NORM_CENTER = str(cfg.window_norm_center)
    elif isinstance(cfg, dict) and 'window_norm_center' in cfg:
        train_pipeline.WINDOW_NORM_CENTER = str(cfg['window_norm_center'])
    if hasattr(cfg, 'staged_representation'):
        train_pipeline.STAGED_REPRESENTATION = str(cfg.staged_representation)
    elif isinstance(cfg, dict) and 'staged_representation' in cfg:
        train_pipeline.STAGED_REPRESENTATION = str(cfg['staged_representation'])
    state = ckpt.get('model_state_dict', {})
    # Prefer explicit config, then Conv2d kernel (supports non-square e.g. 16x4).
    # Do NOT infer square side from head out_features: 16*4 == 8*8 == 64.
    patch_from_cfg = None
    if hasattr(cfg, "dit_patch_size") and cfg.dit_patch_size is not None:
        patch_from_cfg = tuple(int(x) for x in cfg.dit_patch_size)
    elif isinstance(cfg, dict) and cfg.get("dit_patch_size") is not None:
        patch_from_cfg = tuple(int(x) for x in cfg["dit_patch_size"])
    x_embed = state.get("noise_predictor.x_embed.weight")
    patch_set = False
    if patch_from_cfg is not None and len(patch_from_cfg) == 2:
        train_pipeline.DIT_PATCH_SIZE = patch_from_cfg
        patch_set = True
    elif x_embed is not None and getattr(x_embed, "ndim", 0) == 4:
        train_pipeline.DIT_PATCH_SIZE = (int(x_embed.shape[2]), int(x_embed.shape[3]))
        patch_set = True
    head_weight = state.get('noise_predictor.head.weight')
    if head_weight is not None:
        out_features, embed_dim = map(int, head_weight.shape[:2])
        train_pipeline.DIT_EMBED_DIM = embed_dim
        if not patch_set:
            # Legacy square-only fallback when neither config nor x_embed is available.
            out_channels = 2 if diffusion_type == 'binary' else 1
            patch_area = out_features // out_channels
            patch_side = int(round(patch_area ** 0.5))
            if patch_side * patch_side == patch_area:
                train_pipeline.DIT_PATCH_SIZE = (patch_side, patch_side)
    return height


def infer_anchor_kwargs(ckpt: dict, metadata: Optional[dict] = None) -> dict:
    cfg = ckpt.get('config')
    meta_params = (metadata or {}).get('tuned_params', {})
    if cfg is None:
        if not meta_params:
            return {}
        return {
            'use_deterministic_anchor_loss': (
                'deterministic_anchor_lambda' in meta_params
                or 'deterministic_anchor_alpha' in meta_params
            ),
            'deterministic_anchor_lambda': meta_params.get('deterministic_anchor_lambda', 0.99),
            'deterministic_anchor_alpha': meta_params.get('deterministic_anchor_alpha', 0.5),
        }
    if isinstance(cfg, dict):
        params = cfg.get('tuned_params', cfg)
        has_anchor = (
            cfg.get('use_deterministic_anchor_loss', False)
            or 'deterministic_anchor_lambda' in params
            or 'deterministic_anchor_alpha' in params
            or 'deterministic_anchor_lambda' in meta_params
            or 'deterministic_anchor_alpha' in meta_params
        )
        return {
            'use_deterministic_anchor_loss': has_anchor,
            'deterministic_anchor_lambda': params.get(
                'deterministic_anchor_lambda',
                meta_params.get('deterministic_anchor_lambda', 0.99),
            ),
            'deterministic_anchor_alpha': params.get(
                'deterministic_anchor_alpha',
                meta_params.get('deterministic_anchor_alpha', 0.5),
            ),
        }
    return {
        'use_deterministic_anchor_loss': getattr(cfg, 'use_deterministic_anchor_loss', False),
        'deterministic_anchor_lambda': getattr(cfg, 'deterministic_anchor_lambda', 0.99),
        'deterministic_anchor_alpha': getattr(cfg, 'deterministic_anchor_alpha', 0.5),
    }


def choose_extra_indices(n_test: int, n_extra: int, rng: random.Random, exclude: List[int]) -> List[int]:
    pool = [i for i in range(n_test) if i not in exclude]
    if not pool or n_extra <= 0:
        return []
    return rng.sample(pool, min(n_extra, len(pool)))


def run_comparison(
    checkpoint_dir: Optional[str],
    output_dir: str,
    num_samples: int = 3,
    variables_to_plot: int = 3,
    diffusion_ensemble: int = 3,
    lookback_length: int = LOOKBACK_LENGTH,
    forecast_length: int = FORECAST_LENGTH,
    dataset_filter: Optional[str] = None,
    num_extra_windows: int = 2,
    diffusion_type: Optional[str] = None,
    model_type: Optional[str] = None,
    prediction_mode: Optional[str] = None,
    diffusion_sampler: str = "ddim",
    random_seed: int = 13,
    image_height: Optional[int] = None,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if checkpoint_dir is None:
        roots = [Path(resolve_checkpoint_dir(script_dir))]
        print(f"Checkpoint root: {roots[0]}")
    else:
        roots = [Path(checkpoint_dir)]

    # Discover subsets and group by dataset (pick first subset per dataset)
    by_dataset = defaultdict(list)
    for ckpt_root in roots:
        if not ckpt_root.is_dir():
            print(f"Skip missing checkpoint dir: {ckpt_root}")
            continue
        for d in sorted(ckpt_root.iterdir()):
            meta_path = d / 'metadata.json'
            best_path = d / 'best.pt'
            if not d.is_dir() or not meta_path.exists() or not best_path.exists():
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            if dataset_filter and meta['dataset_name'] != dataset_filter:
                continue
            by_dataset[meta['dataset_name']].append({
                'subset_id': meta['subset_id'],
                'variate_indices': meta['variate_indices'],
                'variate_names': meta.get('variate_names', []),
                'best_pt': str(best_path),
                'metadata': meta,
            })

    if not by_dataset:
        print(f"No subsets found under {roots}")
        return

    print(f"Found {len(by_dataset)} datasets: {', '.join(sorted(by_dataset))}")
    os.makedirs(output_dir, exist_ok=True)

    # Cache loaded iTransformer models per subset (finetuned weights differ per subset)
    _itrans_cache = {}

    def _get_itrans_model(subset_id: str, n_vars: int):
        key = subset_id
        if key in _itrans_cache:
            return _itrans_cache[key]
        for base in roots:
            base_s = str(base)
            ft_path = os.path.join(base_s, f'{subset_id}_itransformer_finetuned.pt')
            if os.path.exists(ft_path):
                model = load_itransformer_from_checkpoint(ft_path, n_vars, device)
                _itrans_cache[key] = model
                print(f"  Loaded finetuned iTransformer ({subset_id}) from {ft_path}")
                return model
        for base in roots:
            base_s = str(base)
            dim_dir = pretrain_dir_for_dim(n_vars, base_dir=base_s)
            candidates = [
                os.path.join(dim_dir, 'itransformer.pt'),
                os.path.join(base_s, 'pretrained_itransformer.pt'),
            ]
            for p in candidates:
                if os.path.exists(p):
                    model = load_itransformer_from_checkpoint(p, n_vars, device)
                    _itrans_cache[key] = model
                    print(
                        f"  WARNING: using synthetic-pretrained iTransformer from {p} "
                        f"(no {subset_id}_itransformer_finetuned.pt); orange curve may look random vs real data."
                    )
                    return model
        print(f"  WARNING: no iTransformer checkpoint for subset={subset_id} dim={n_vars}")
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
                dataset_name, variate_indices, stride=lookback_length,
                lookback=lookback_length, horizon=forecast_length
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

        # Load per-subset finetuned iTransformer (matches diffusion training guidance)
        n_vars = len(variate_indices)
        itrans_model = _get_itrans_model(subset_id, n_vars)
        if itrans_model is None:
            print(f"  Skipping {dataset_name}: no iTransformer for subset={subset_id}")
            continue

        diff_ckpt = torch.load(sub['best_pt'], map_location=device, weights_only=False)
        diff_type = infer_diffusion_type(diff_ckpt, diffusion_type)
        backbone = infer_model_type(diff_ckpt, model_type)
        applied_h = apply_checkpoint_architecture(diff_ckpt, diff_type, image_height)
        print(
            f"  diffusion_type={diff_type} model_type={backbone} image_height={applied_h}"
        )

        # Load fine-tuned diffusion with same guidance wrapper as training
        anchor_kwargs = infer_anchor_kwargs(diff_ckpt, sub.get('metadata'))
        itrans_guidance = iTransformerGuidance(itrans_model)
        diff_model = create_diffusion_model(
            n_variates=n_vars,
            diffusion_type=diff_type,
            model_type=backbone,
            guidance_model=itrans_guidance,
            **anchor_kwargs,
        ).to(device)
        diff_model.load_state_dict(diff_ckpt['model_state_dict'])
        diff_model.eval()

        rng = random.Random(random_seed)
        sample_indices = np.linspace(0, n_test - 1, min(num_samples, n_test), dtype=int).tolist()
        extra_indices = choose_extra_indices(n_test, num_extra_windows, rng, exclude=sample_indices)
        n_vars_plot = min(variables_to_plot, len(variate_indices))
        n_rows = len(sample_indices) + len(extra_indices)

        fig, axes = plt.subplots(
            n_rows, n_vars_plot,
            figsize=(5.5 * n_vars_plot, 3.2 * n_rows),
            squeeze=False,
            constrained_layout=True,
        )

        for row, idx in enumerate(sample_indices):
            past, future = test_ds[idx]
            past_t = past.unsqueeze(0).to(device)   # (1, C, L)

            with torch.no_grad():
                # iTransformer standalone (last seq_sl steps of context, same as training)
                B, C, L = past_t.shape
                x_enc = past_t.permute(0, 2, 1)
                seq_sl = getattr(itrans_model, 'seq_len', L)
                if x_enc.shape[1] > seq_sl:
                    x_enc = x_enc[:, -seq_sl:, :]
                x_dec = torch.zeros(B, forecast_length, C, device=device)
                itrans_out = itrans_model(x_enc, None, x_dec, None)
                if isinstance(itrans_out, tuple):
                    itrans_out = itrans_out[0]
                itrans_pred = itrans_out.permute(0, 2, 1).cpu()[0]   # (C, F)

                # Diffusion prediction (single or averaged)
                if diffusion_ensemble <= 1:
                    torch.manual_seed(42 + idx)
                    result = diff_model.generate(past_t, sampler=diffusion_sampler)
                    diff_pred = result.get(
                        'prediction_global_norm', result['prediction']
                    ).cpu()[0]  # (C, F)
                else:
                    diff_preds = []
                    for s_idx in range(diffusion_ensemble):
                        torch.manual_seed(1000 + s_idx * 17 + idx)
                        result = diff_model.generate(past_t, sampler=diffusion_sampler)
                        diff_preds.append(
                            result.get('prediction_global_norm', result['prediction']).cpu()
                        )
                    diff_pred = torch.stack(diff_preds).mean(0)[0]  # (C, F)

            # Denormalize everything to original scale
            past_dn = denorm(past, mean, std)
            # future includes lookback_overlap; slice it for plotting (last forecast_length steps)
            future_sliced = future[:, -forecast_length:]
            future_dn = denorm(future_sliced, mean, std)
            itrans_dn = denorm(itrans_pred, mean, std)
            diff_pred_sliced = diff_pred[:, -forecast_length:] if diff_pred.shape[-1] > forecast_length else diff_pred
            diff_dn = denorm(diff_pred_sliced, mean, std)

            # Show last N steps of context for visual continuity
            context_len = min(forecast_length * 2, lookback_length)
            t_past = np.arange(-context_len, 0)
            t_future = np.arange(0, forecast_length)

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
                vname = var_names[col] if col < len(var_names) else f'Var {col}'
                print(f"    - {vname:15}: iTrans MAE={it_mae:.4f}, Diff MAE={df_mae:.4f}")

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

        context_len = min(forecast_length * 2, lookback_length)
        t_past = np.arange(-context_len, 0)
        for row_off, idx in enumerate(extra_indices, start=len(sample_indices)):
            past, _future = test_ds[idx]
            past_dn = denorm(past, mean, std)
            for col in range(n_vars_plot):
                ax = axes[row_off, col]
                ax.plot(
                    t_past, past_dn[col, -context_len:].numpy(),
                    color='#546E7A', linewidth=1.1,
                )
                ax.axvline(x=0, color='black', linestyle=':', alpha=0.25)
                if col == 0:
                    ax.set_ylabel(f'Lookback {row_off - len(sample_indices) + 1}', fontsize=9)
                vname = var_names[col] if col < len(var_names) else f'Var {col}'
                if row_off == len(sample_indices):
                    ax.set_title(f'{vname} (ctx)', fontsize=9)
                ax.grid(alpha=0.2)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=10,
                       bbox_to_anchor=(0.5, 1.02))
        mode_label = 'single sample' if diffusion_ensemble <= 1 else f'{diffusion_ensemble}-sample avg'
        fig.suptitle(
            f'{dataset_name} • {subset_id} • {diff_type} diffusion ({mode_label})',
            fontsize=12, fontweight='bold',
        )

        out_path = os.path.join(output_dir, f'comparison_{dataset_name}_{subset_id}.png')
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {out_path}")

    print(f"\nDone! All comparison plots in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Diffusion vs iTransformer comparison plots')
    parser.add_argument(
        '--checkpoint-dir', type=str, default=None,
        help='Checkpoint root; default: checkpoints_multivariate under the package',
    )
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--num-samples', type=int, default=3, help='Samples per dataset')
    parser.add_argument('--vars', type=int, default=3, help='Variables to plot per sample')
    parser.add_argument('--ensemble', type=int, default=1,
                        help='Diffusion samples to average (1=single sample, 30=full avg)')
    parser.add_argument('--lookback-length', type=int, default=LOOKBACK_LENGTH)
    parser.add_argument('--forecast-length', type=int, default=FORECAST_LENGTH)
    parser.add_argument('--dataset', type=str, default=None,
                        help='Plot only this dataset name (e.g. ETTh2)')
    parser.add_argument('--num-extra-windows', type=int, default=2,
                        help='Extra lookback-only rows beneath forecast rows')
    parser.add_argument('--diffusion-type', type=str, default=None,
                        choices=['gaussian', 'binary'],
                        help='Override diffusion type inferred from checkpoint')
    parser.add_argument('--model-type', type=str, default=None, choices=['unet', 'dit'],
                        help='Override backbone inferred from checkpoint (required for DiT if config missing)')
    parser.add_argument('--prediction-mode', type=str, default=None,
                        choices=['epsilon', 'x0_cumsum'],
                        help='Override prediction mode inferred from checkpoint')
    parser.add_argument('--diffusion-sampler', type=str, default='ddim',
                        choices=['ddim', 'quad_t', 'ddim_quad', 'anchor', 'deterministic_anchor'],
                        help='Sampler for diffusion plots')
    parser.add_argument('--random-seed', type=int, default=13)
    parser.add_argument(
        '--image-height', type=int, default=None,
        help='Override 2D representation height (default: read from checkpoint)',
    )
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(project_root, RESULTS_DIR, 'viz', 'comparison')
    run_comparison(
        args.checkpoint_dir, output_dir, args.num_samples, args.vars, args.ensemble,
        args.lookback_length, args.forecast_length,
        dataset_filter=args.dataset,
        num_extra_windows=args.num_extra_windows,
        diffusion_type=args.diffusion_type,
        model_type=args.model_type,
        prediction_mode=args.prediction_mode,
        diffusion_sampler='anchor' if args.diffusion_sampler == 'deterministic_anchor' else args.diffusion_sampler,
        random_seed=args.random_seed,
        image_height=args.image_height,
    )


if __name__ == '__main__':
    main()
