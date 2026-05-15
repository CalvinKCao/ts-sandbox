"""
Multivariate comparison plots: ground truth vs iTransformer vs diffusion.

**Legacy** layout: checkpoint subdirs with ``metadata.json`` + ``best.pt``, plus
optional ``{subset_id}_itransformer_finetuned.pt`` beside those dirs.

**Joint e2e** layout: ``{subset_id}_joint_finetuned_gB.pt`` or ``_gC.pt`` in the
checkpoint root (same directory training writes to). The full ``DiffusionTSF``
state (including the jointly trained iTransformer head) is loaded from that file;
orange curves use ``guidance_model.get_forecast`` so they match finetuning.

Usage:
    python -m models.diffusion_tsf.visualize_comparison \\
        --checkpoint-dir /path/to/checkpoints \\
        --output-dir /path/to/output \\
        --num-samples 3 --vars 3
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

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

from models.diffusion_tsf.config import DiffusionTSFConfig
from models.diffusion_tsf.diffusion_model import DiffusionTSF
from models.diffusion_tsf.storage_paths import checkpoint_roots_ordered
from models.diffusion_tsf.train_multivariate_pipeline import (
    RESULTS_DIR,
    DATASET_REGISTRY,
    LOOKBACK_LENGTH,
    FORECAST_LENGTH,
    LOOKBACK_OVERLAP,
    create_diffusion_model,
    create_itransformer,
    load_dataset,
    pretrain_dir_for_dim,
    load_itransformer_from_checkpoint,
    generate_dataset_job,
)
from models.diffusion_tsf.guidance import iTransformerGuidance


def denorm(x, mean, std):
    """Denormalize (C, T) tensor using (1, C) stats."""
    m = mean.squeeze().unsqueeze(-1)   # (C, 1)
    s = std.squeeze().unsqueeze(-1)
    return x * s + m


def _diffusion_config_from_snapshot(cfg_d: dict) -> DiffusionTSFConfig:
    fields = set(DiffusionTSFConfig.__dataclass_fields__)
    kwargs = {k: v for k, v in cfg_d.items() if k in fields}
    return DiffusionTSFConfig(**kwargs)


def _pick_joint_ckpts(ckpt_root: Path) -> Dict[str, Path]:
    """Map subset_id -> checkpoint path, preferring _gB over _gC when both exist."""
    picked: Dict[str, Path] = {}
    for path in sorted(ckpt_root.glob('*_joint_finetuned_g*.pt')):
        name = path.name
        for suf in ('_joint_finetuned_gB.pt', '_joint_finetuned_gC.pt'):
            if name.endswith(suf):
                subset_id = name[: -len(suf)]
                prev = picked.get(subset_id)
                if prev is None:
                    picked[subset_id] = path
                elif prev.name.endswith('_joint_finetuned_gC.pt') and name.endswith(
                    '_joint_finetuned_gB.pt'
                ):
                    picked[subset_id] = path
                break
    return picked


def _resolve_joint_dataset_fields(subset_id: str, ckpt_root: Path) -> Tuple[str, List[int], List[str]]:
    """Infer dataset_name and variate columns for a joint finetune checkpoint."""
    job_dir = ckpt_root.parent
    for rel in (
        Path('eval_test') / subset_id / 'results.json',
        Path('datasets') / subset_id / 'results.json',
    ):
        p = job_dir / rel
        if p.is_file():
            with open(p) as f:
                meta = json.load(f)
            ds = meta.get('dataset') or meta.get('dataset_name')
            if not ds:
                continue
            vi = meta.get('variate_indices')
            if vi is None:
                continue
            vn = meta.get('variate_names') or []
            return str(ds), list(vi), list(vn)

    if subset_id in DATASET_REGISTRY:
        job = generate_dataset_job(subset_id)
        return subset_id, job['variate_indices'], job['variate_names']

    raise ValueError(
        f"Cannot resolve dataset for joint subset_id={subset_id!r} under {ckpt_root} "
        f"(no eval_test/.../results.json and not in DATASET_REGISTRY)."
    )


def _load_joint_diffusion_model(
    ckpt_path: str,
    device: torch.device,
) -> Tuple[DiffusionTSF, DiffusionTSFConfig]:
    """Rebuild DiffusionTSF + frozen iTransformer guidance from a joint finetune .pt file."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_d = ckpt.get('config') or {}
    if not cfg_d:
        raise RuntimeError(f"Joint checkpoint {ckpt_path} has no 'config' entry.")
    cfg = _diffusion_config_from_snapshot(cfg_d)
    state = ckpt['model_state_dict']
    w_enc = 'guidance_model.model.enc_embedding.value_embedding.weight'
    w_proj = 'guidance_model.model.projector.weight'
    if w_enc not in state or w_proj not in state:
        raise RuntimeError(
            f"Joint checkpoint {ckpt_path} missing guidance iTransformer keys "
            f"({w_enc!r} / {w_proj!r})."
        )
    seq_len = int(state[w_enc].shape[1])
    pred_len = int(state[w_proj].shape[0])

    itrans = create_itransformer(
        seq_len=seq_len,
        pred_len=pred_len,
        num_vars=cfg.num_variables,
        dropout=0.1,
    ).to(device)
    guidance = iTransformerGuidance(itrans, freeze=True)
    model = DiffusionTSF(cfg, guidance_model=guidance).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        bad = [k for k in missing if k and not k.startswith('guidance_model.')]
        if bad:
            raise RuntimeError(f"Joint load missing keys: {bad[:8]}")
    if unexpected:
        bad_u = [k for k in unexpected if not k.startswith('guidance_model.')]
        if bad_u:
            print(f"  WARNING: unexpected keys in joint ckpt (non-guidance): {bad_u[:8]}")
    model.eval()
    return model, cfg


def run_comparison(
    checkpoint_dir: Optional[str],
    output_dir: str,
    num_samples: int = 3,
    variables_to_plot: int = 3,
    diffusion_ensemble: int = 3,
    lookback_length: int = LOOKBACK_LENGTH,
    forecast_length: int = FORECAST_LENGTH,
    output_tag: Optional[str] = None,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if checkpoint_dir is None:
        roots = [Path(p) for p in checkpoint_roots_ordered(script_dir)]
        print(f"Checkpoint roots: {', '.join(str(r) for r in roots)}")
    else:
        roots = [Path(checkpoint_dir)]

    def _output_png_basename(dataset_name: str, *, joint: bool = False) -> str:
        parts = ['comparison']
        if output_tag:
            parts.append(str(output_tag).replace(os.sep, '_').replace('/', '_'))
        if joint:
            parts.append('joint')
        parts.append(dataset_name)
        return '_'.join(parts) + '.png'

    # Discover legacy subset dirs (metadata.json + best.pt) and flat joint finetune ckpts.
    legacy_by_dataset: Dict[str, List[dict]] = defaultdict(list)
    joint_jobs: List[dict] = []

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
            legacy_by_dataset[meta['dataset_name']].append({
                'subset_id': meta['subset_id'],
                'variate_indices': meta['variate_indices'],
                'variate_names': meta.get('variate_names', []),
                'best_pt': str(best_path),
            })

        for subset_id, jpath in _pick_joint_ckpts(ckpt_root).items():
            try:
                dn, vi, vn = _resolve_joint_dataset_fields(subset_id, ckpt_root)
            except ValueError as err:
                print(f"  Skip joint ckpt {jpath.name}: {err}")
                continue
            joint_jobs.append({
                'subset_id': subset_id,
                'dataset_name': dn,
                'variate_indices': vi,
                'variate_names': vn,
                'joint_ckpt': str(jpath),
            })

    if not legacy_by_dataset and not joint_jobs:
        print(f"No legacy subsets or joint finetune checkpoints found under {roots}")
        return

    n_leg = len(legacy_by_dataset)
    n_joint = len(joint_jobs)
    print(
        f"Found {n_leg} legacy dataset entries, {n_joint} joint finetune checkpoint(s): "
        f"{', '.join(sorted(legacy_by_dataset))}"
        + (f"; joint: {', '.join(j['subset_id'] for j in joint_jobs)}" if joint_jobs else '')
    )
    os.makedirs(output_dir, exist_ok=True)

    # Cache loaded iTransformer models per subset (finetuned weights differ per subset)
    _itrans_cache: Dict[str, object] = {}

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

    for dataset_name, subsets in sorted(legacy_by_dataset.items()):
        sub = subsets[0]
        subset_id = sub['subset_id']
        variate_indices = sub['variate_indices']
        var_names = sub['variate_names']
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name} (subset: {subset_id}) [legacy]")
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

        n_vars = len(variate_indices)
        itrans_model = _get_itrans_model(subset_id, n_vars)
        if itrans_model is None:
            print(f"  Skipping {dataset_name}: no iTransformer for subset={subset_id}")
            continue

        diff_model = create_diffusion_model(n_variates=n_vars).to(device)
        itrans_guidance = iTransformerGuidance(itrans_model)
        diff_model.set_guidance_model(itrans_guidance)
        diff_ckpt = torch.load(sub['best_pt'], map_location=device, weights_only=False)
        diff_model.load_state_dict(diff_ckpt['model_state_dict'])
        diff_model.eval()

        H_plot = forecast_length
        LB_plot = lookback_length

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

                if diffusion_ensemble <= 1:
                    result = diff_model.generate(past_t)
                    diff_pred = result['prediction'].cpu()[0]  # (C, F)
                else:
                    diff_preds = []
                    for _ in range(diffusion_ensemble):
                        result = diff_model.generate(past_t)
                        diff_preds.append(result['prediction'].cpu())
                    diff_pred = torch.stack(diff_preds).mean(0)[0]  # (C, F)

            past_dn = denorm(past, mean, std)
            future_sliced = future[:, -H_plot:]
            future_dn = denorm(future_sliced, mean, std)
            itrans_dn = denorm(itrans_pred, mean, std)
            diff_pred_sliced = diff_pred[:, -H_plot:] if diff_pred.shape[-1] > H_plot else diff_pred
            diff_dn = denorm(diff_pred_sliced, mean, std)

            context_len = min(H_plot * 2, LB_plot)
            t_past = np.arange(-context_len, 0)
            t_future = np.arange(0, H_plot)

            for col in range(n_vars_plot):
                ax = axes[row, col]

                gt = future_dn[col].numpy()
                it = itrans_dn[col].numpy()
                df = diff_dn[col].numpy()

                ax.plot(t_past, past_dn[col, -context_len:].numpy(),
                        color='#9E9E9E', alpha=0.5, linewidth=0.8)
                ax.plot(t_future, gt, color='#2196F3', linewidth=1.6,
                        label='Ground Truth' if row == 0 and col == 0 else '')
                ax.plot(t_future, it, color='#FF9800', linewidth=1.2,
                        linestyle='--', alpha=0.85,
                        label='iTransformer' if row == 0 and col == 0 else '')
                ax.plot(t_future, df, color='#E91E63', linewidth=1.2,
                        label='Diffusion' if row == 0 and col == 0 else '')

                ax.axvline(x=0, color='black', linestyle=':', alpha=0.25)

                it_mae = np.mean(np.abs(it - gt))
                df_mae = np.mean(np.abs(df - gt))
                vname = var_names[col] if col < len(var_names) else f'Var {col}'
                print(f"    - {vname:15}: iTrans MAE={it_mae:.4f}, Diff MAE={df_mae:.4f}")

                ax.text(0.97, 0.97,
                        f'iTrans MAE: {it_mae:.3f}\nDiff MAE: {df_mae:.3f}',
                        transform=ax.transAxes, fontsize=7,
                        va='top', ha='right',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

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
        out_path = os.path.join(output_dir, _output_png_basename(dataset_name, joint=False))
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {out_path}")

    for meta in sorted(joint_jobs, key=lambda x: (x['dataset_name'], x['subset_id'])):
        dataset_name = meta['dataset_name']
        subset_id = meta['subset_id']
        variate_indices = meta['variate_indices']
        var_names = meta['variate_names']
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name} (subset: {subset_id}) [joint e2e]")
        print(f"{'='*60}")

        try:
            diff_model, mcfg = _load_joint_diffusion_model(meta['joint_ckpt'], device)
        except Exception as e:
            print(f"  Skipping joint ckpt {meta['joint_ckpt']}: {e}")
            continue

        K = mcfg.lookback_overlap
        lb = mcfg.lookback_length
        H = int(mcfg.forecast_length - K)
        if H <= 0:
            print(f"  Skipping {dataset_name}: invalid horizon H={H} from saved config.")
            continue

        try:
            _, _, test_ds, norm_stats = load_dataset(
                dataset_name, variate_indices, stride=lb,
                lookback=lb, horizon=H, lookback_overlap=K,
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
        gm = diff_model.guidance_model
        if gm is None:
            print(f"  Skipping {dataset_name}: joint model has no guidance_model")
            continue

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
            past_t = past.unsqueeze(0).to(device)

            with torch.no_grad():
                itrans_pred = gm.get_forecast(past_t, H).cpu()[0]

                if diffusion_ensemble <= 1:
                    result = diff_model.generate(past_t)
                    diff_pred = result['prediction'].cpu()[0]
                else:
                    diff_preds = []
                    for _ in range(diffusion_ensemble):
                        result = diff_model.generate(past_t)
                        diff_preds.append(result['prediction'].cpu())
                    diff_pred = torch.stack(diff_preds).mean(0)[0]

            past_dn = denorm(past, mean, std)
            future_sliced = future[:, -H:]
            future_dn = denorm(future_sliced, mean, std)
            itrans_dn = denorm(itrans_pred, mean, std)
            diff_pred_sliced = diff_pred[:, -H:] if diff_pred.shape[-1] > H else diff_pred
            diff_dn = denorm(diff_pred_sliced, mean, std)

            context_len = min(H * 2, lb)
            t_past = np.arange(-context_len, 0)
            t_future = np.arange(0, H)

            for col in range(n_vars_plot):
                ax = axes[row, col]

                gt = future_dn[col].numpy()
                it = itrans_dn[col].numpy()
                df = diff_dn[col].numpy()

                ax.plot(t_past, past_dn[col, -context_len:].numpy(),
                        color='#9E9E9E', alpha=0.5, linewidth=0.8)
                ax.plot(t_future, gt, color='#2196F3', linewidth=1.6,
                        label='Ground Truth' if row == 0 and col == 0 else '')
                ax.plot(t_future, it, color='#FF9800', linewidth=1.2,
                        linestyle='--', alpha=0.85,
                        label='iTransformer' if row == 0 and col == 0 else '')
                ax.plot(t_future, df, color='#E91E63', linewidth=1.2,
                        label='Diffusion' if row == 0 and col == 0 else '')

                ax.axvline(x=0, color='black', linestyle=':', alpha=0.25)

                it_mae = np.mean(np.abs(it - gt))
                df_mae = np.mean(np.abs(df - gt))
                vname = var_names[col] if col < len(var_names) else f'Var {col}'
                print(f"    - {vname:15}: iTrans MAE={it_mae:.4f}, Diff MAE={df_mae:.4f}")

                ax.text(0.97, 0.97,
                        f'iTrans MAE: {it_mae:.3f}\nDiff MAE: {df_mae:.3f}',
                        transform=ax.transAxes, fontsize=7,
                        va='top', ha='right',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

                if row == 0:
                    ax.set_title(vname, fontsize=10)
                if col == 0:
                    ax.set_ylabel(f'Sample {row + 1}', fontsize=10)
                ax.tick_params(labelsize=7)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=10,
                   bbox_to_anchor=(0.5, 1.01))
        mode_label = 'single sample' if diffusion_ensemble <= 1 else f'{diffusion_ensemble}-sample avg'
        fig.suptitle(f'{dataset_name} (joint)  ({mode_label})',
                     fontsize=14, fontweight='bold', y=1.04)

        plt.tight_layout()
        out_path = os.path.join(output_dir, _output_png_basename(dataset_name, joint=True))
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {out_path}")

    print(f"\nDone! All comparison plots in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Diffusion vs iTransformer comparison plots')
    parser.add_argument(
        '--checkpoint-dir', type=str, default=None,
        help='Single checkpoint root; default: scan checkpoints_multivariate and legacy checkpoints_7var if present',
    )
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--num-samples', type=int, default=3, help='Samples per dataset')
    parser.add_argument('--vars', type=int, default=3, help='Variables to plot per sample')
    parser.add_argument('--ensemble', type=int, default=1,
                        help='Diffusion samples to average (1=single sample, 30=full avg)')
    parser.add_argument(
        '--output-tag', type=str, default=None,
        help='Optional tag embedded in output PNG names (avoids collisions across jobs).',
    )
    parser.add_argument('--lookback-length', type=int, default=LOOKBACK_LENGTH)
    parser.add_argument('--forecast-length', type=int, default=FORECAST_LENGTH)
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(RESULTS_DIR, 'viz', 'comparison')
    run_comparison(
        args.checkpoint_dir, output_dir, args.num_samples, args.vars, args.ensemble,
        args.lookback_length, args.forecast_length, args.output_tag,
    )


if __name__ == '__main__':
    main()
