"""Throwaway ablation: iTransformer guidance channel × hybrid cross-variate context.

Uses the same electricity 4-variate subset as ``throwaway_diffsteps_sweep``:
consumers 93, 292, 81, 84. One shared iTransformer pretrain (cached), then for
each ablation: short diffusion pretrain (max 10 epochs) → finetune (20 epochs)
→ eval. Reports MAE/MSE and wall times (pretrain / finetune / eval / total).

Default geometry matches a mid sweep point: ``T=1000``, ``H=128`` (overridable).

Configs
-------
- **both** — guidance ghost channel + hybrid 1D context (typical setup)
- **guidance_only** — guidance on, ``use_hybrid_condition=False``
- **hybrid_only** — no guidance channel; context from normalized past only
- **neither** — visual concat U-Net only (no ghost, no cross-attn context)

Usage (repo root)::

    python -m models.diffusion_tsf.throwaway_guidance_hybrid_ablation --smoke-test

Full path on local GPU in a few minutes (all four ablations, tiny T/H and sample counts)::

    python -m models.diffusion_tsf.throwaway_guidance_hybrid_ablation --smoke-pipeline --store /tmp/abl-smoke --amp

Cluster::

    sbatch slurm_throwaway_guidance_hybrid_ablation.sh
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Path setup — must come before any repo imports
# ---------------------------------------------------------------------------
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import models.diffusion_tsf.train_multivariate_pipeline as P
from models.diffusion_tsf.train_multivariate_pipeline import (
    create_itransformer,
    create_diffusion_model,
    evaluate_model,
    finetune_on_dataset,
    iTransformerGuidance,
    load_dataset,
    pretrain_diffusion,
    pretrain_itransformer,
    setup_logging,
    get_device,
)

# ---------------------------------------------------------------------------
# Fixed params (match throwaway_diffsteps_sweep)
# ---------------------------------------------------------------------------
DATASET = 'electricity'
VARIATE_INDICES = [93, 292, 81, 84]
VARIATE_NAMES = ['93', '292', '81', '84']
N_VARIATES = 4

FIXED_LR = 2.05e-5
FIXED_BATCH = 16
FIXED_DROPOUT = 0.1

MAX_DIFFUSION_PRETRAIN_EPOCHS = 10
DIFFUSION_PRETRAIN_PATIENCE = 5

ITRANS_PRETRAIN_SAMPLES = 10000
ITRANS_PRETRAIN_EPOCHS = 30
ITRANS_PRETRAIN_PATIENCE = 5

FINETUNE_EPOCHS = 20
FINETUNE_PATIENCE = 5

DEFAULT_NUM_DIFFUSION_STEPS = 1000
DEFAULT_IMAGE_HEIGHT = 128

SMOKE_ITRANS_SAMPLES = 64
SMOKE_DIFFUSION_PRETRAIN_SAMPLES = 64
SMOKE_PRETRAIN_EPOCHS = 1
SMOKE_FINETUNE_EPOCHS = 1
SMOKE_FINETUNE_BATCH = 8

# Full-pipeline smoke: still runs iTrans → 4×(diffusion pretrain → finetune → eval), but slashed budgets.
SMOKE_PIPELINE_ITRANS_SAMPLES = 16
SMOKE_PIPELINE_DIFFUSION_SAMPLES = 8
SMOKE_PIPELINE_FINETUNE_BATCH = 4

ABLATION_CONFIGS: List[Tuple[str, bool, bool]] = [
    ('both', True, True),
    ('guidance_only', True, False),
    ('hybrid_only', False, True),
    ('neither', False, False),
]


def _make_store(base_store: str) -> str:
    os.makedirs(base_store, exist_ok=True)
    return base_store


def _trial_dir(store: str, name: str, T: int, H: int) -> str:
    d = os.path.join(store, f'ablation_{name}_T{T}_H{H}')
    os.makedirs(d, exist_ok=True)
    return d


def _itrans_ckpt_path(store: str) -> str:
    return os.path.join(store, 'shared_itransformer.pt')


def _ensure_itransformer(store: str, smoke_test: bool, smoke_pipeline: bool) -> str:
    ckpt = _itrans_ckpt_path(store)
    if os.path.exists(ckpt):
        print(f"[ablation] Reusing cached iTransformer: {ckpt}")
        return ckpt

    print('[ablation] Pretraining iTransformer (shared, runs once) ...')
    P.N_VARIATES = N_VARIATES

    best_params = {'learning_rate': FIXED_LR, 'dropout': FIXED_DROPOUT}
    if smoke_pipeline:
        n_samples = SMOKE_PIPELINE_ITRANS_SAMPLES
        epochs = 1
        patience = 1
    elif smoke_test:
        n_samples = SMOKE_ITRANS_SAMPLES
        epochs = SMOKE_PRETRAIN_EPOCHS
        patience = 1
    else:
        n_samples = ITRANS_PRETRAIN_SAMPLES
        epochs = ITRANS_PRETRAIN_EPOCHS
        patience = ITRANS_PRETRAIN_PATIENCE

    pretrain_itransformer(
        best_params,
        n_samples=n_samples,
        epochs=epochs,
        patience=patience,
        checkpoint_dir=store,
        smoke_test=smoke_test,
    )
    default_path = os.path.join(store, 'pretrained_itransformer.pt')
    if os.path.exists(default_path) and default_path != ckpt:
        os.replace(default_path, ckpt)
    return ckpt


def run_ablation(
    name: str,
    use_guidance_channel: bool,
    use_hybrid_condition: bool,
    T: int,
    H: int,
    itrans_ckpt: str,
    store: str,
    smoke_test: bool,
    smoke_pipeline: bool,
    wandb_run: Any,
) -> Dict[str, Any]:
    """One ablation: diffusion pretrain → finetune → eval; return metrics + timings."""
    trial_dir = _trial_dir(store, name, T, H)
    result_path = os.path.join(trial_dir, 'result.json')
    if os.path.exists(result_path):
        with open(result_path) as f:
            cached = json.load(f)
        print(
            f"[ablation] {name}: cached  MAE={cached.get('eval_mae', float('nan')):.4f}  "
            f"total_s={cached.get('wall_total_s', 0):.0f}"
        )
        return cached

    device = get_device()
    P.N_VARIATES = N_VARIATES
    P.IMAGE_HEIGHT = H

    timings: Dict[str, float] = {}
    t_wall0 = time.time()

    diff_ckpt = os.path.join(trial_dir, 'pretrained_diffusion.pt')
    if not os.path.exists(diff_ckpt):
        print(f"[ablation] {name}: diffusion pretrain (guidance={use_guidance_channel}, hybrid={use_hybrid_condition}) ...")
        t0 = time.time()
        best_params = {'learning_rate': FIXED_LR}
        pretrain_diffusion(
            best_params,
            itrans_ckpt,
            n_samples=(
                SMOKE_PIPELINE_DIFFUSION_SAMPLES
                if smoke_pipeline
                else (SMOKE_DIFFUSION_PRETRAIN_SAMPLES if smoke_test else 5000)
            ),
            epochs=SMOKE_PRETRAIN_EPOCHS if smoke_test else MAX_DIFFUSION_PRETRAIN_EPOCHS,
            patience=1 if smoke_test else DIFFUSION_PRETRAIN_PATIENCE,
            checkpoint_dir=trial_dir,
            smoke_test=smoke_test,
            use_guidance_channel=use_guidance_channel,
            use_hybrid_condition=use_hybrid_condition,
            num_diffusion_steps=T,
            image_height=H,
        )
        timings['wall_diffusion_pretrain_s'] = time.time() - t0
    else:
        timings['wall_diffusion_pretrain_s'] = 0.0

    subset_info = {
        'subset_id': f'elec-4v-ablation-{name}-T{T}H{H}',
        'dataset': DATASET,
        'variate_indices': VARIATE_INDICES,
        'variate_names': VARIATE_NAMES,
    }
    ft_batch = (
        SMOKE_PIPELINE_FINETUNE_BATCH
        if smoke_pipeline
        else (SMOKE_FINETUNE_BATCH if smoke_test else FIXED_BATCH)
    )
    tuned_params = {'learning_rate': FIXED_LR, 'batch_size': ft_batch}

    print(f"[ablation] {name}: finetune ...")
    t0 = time.time()
    ft_ckpt, train_metrics = finetune_on_dataset(
        subset_info,
        diff_ckpt,
        itrans_ckpt,
        tuned_params,
        epochs=SMOKE_FINETUNE_EPOCHS if smoke_test else FINETUNE_EPOCHS,
        patience=1 if smoke_test else FINETUNE_PATIENCE,
        checkpoint_dir=trial_dir,
        smoke_test=smoke_test,
        use_guidance_channel=use_guidance_channel,
        use_hybrid_condition=use_hybrid_condition,
        num_diffusion_steps=T,
        image_height=H,
    )
    timings['wall_finetune_s'] = time.time() - t0

    print(f"[ablation] {name}: evaluate ...")
    t0 = time.time()
    model = create_diffusion_model(
        use_guidance=use_guidance_channel,
        use_hybrid_condition=use_hybrid_condition,
        diffusion_type=P.DIFFUSION_TYPE,
        num_diffusion_steps=T,
        image_height=H,
    ).to(device)
    if use_guidance_channel:
        itrans_model = create_itransformer().to(device)
        ckpt_data = torch.load(itrans_ckpt, map_location=device, weights_only=False)
        itrans_model.load_state_dict(ckpt_data['model_state_dict'])
        guidance = iTransformerGuidance(
            itrans_model,
            use_norm=True,
            seq_len=P.LOOKBACK_LENGTH,
            pred_len=P.FORECAST_LENGTH,
        )
        model.set_guidance_model(guidance)
    ckpt_data = torch.load(ft_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt_data['model_state_dict'])

    _, _, test_ds, _ = load_dataset(DATASET, VARIATE_INDICES, stride=P.LOOKBACK_LENGTH)
    if smoke_test:
        from torch.utils.data import Subset

        test_ds = Subset(test_ds, list(range(min(2, len(test_ds)))))
    from torch.utils.data import DataLoader

    test_loader = DataLoader(test_ds, batch_size=4 if not smoke_test else 2, shuffle=False)

    eval_results = evaluate_model(
        model,
        test_loader,
        device,
        n_samples=5 if smoke_test else 30,
        smoke_test=smoke_test,
    )
    timings['wall_eval_s'] = time.time() - t0
    timings['wall_total_s'] = time.time() - t_wall0

    result = {
        'name': name,
        'use_guidance_channel': use_guidance_channel,
        'use_hybrid_condition': use_hybrid_condition,
        'T': T,
        'H': H,
        'eval_mae': eval_results['averaged']['mae'],
        'eval_mse': eval_results['averaged']['mse'],
        'train_metrics': train_metrics,
        **timings,
    }

    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)

    if wandb_run is not None:
        try:
            import wandb

            wandb.log(
                {
                    f'ablation/{name}_mae': result['eval_mae'],
                    f'ablation/{name}_mse': result['eval_mse'],
                    f'ablation/{name}_wall_total_s': result['wall_total_s'],
                    f'ablation/{name}_wall_pretrain_s': result['wall_diffusion_pretrain_s'],
                    f'ablation/{name}_wall_finetune_s': result['wall_finetune_s'],
                    f'ablation/{name}_wall_eval_s': result['wall_eval_s'],
                }
            )
        except Exception:
            pass

    print(
        f"[ablation] {name}: MAE={result['eval_mae']:.4f}  MSE={result['eval_mse']:.4f}  "
        f"times: pretrain={result['wall_diffusion_pretrain_s']:.0f}s "
        f"ft={result['wall_finetune_s']:.0f}s eval={result['wall_eval_s']:.0f}s "
        f"total={result['wall_total_s']:.0f}s"
    )
    return result


def _print_results_table(results: List[Dict[str, Any]]) -> None:
    results = sorted(results, key=lambda r: r['eval_mae'])
    fastest = min(results, key=lambda r: r['wall_total_s'])
    best_mae = results[0]

    hdr = (
        f"{'config':<16} {'guid':>5} {'hybr':>5}  "
        f"{'MAE':>8} {'MSE':>10}  "
        f"{'pretr':>7} {'ft':>7} {'eval':>6} {'total':>7}  "
        f"{'vs_best_t':>9}"
    )
    print('\n' + '=' * len(hdr))
    print('GUIDANCE × HYBRID CONDITIONING ABLATION (ranked by MAE)')
    print('=' * len(hdr))
    print(hdr)
    print('-' * len(hdr))

    for r in results:
        tag = '← best MAE' if r is best_mae else ''
        if r is fastest and r is not best_mae:
            tag = '← fastest' if not tag else tag + ' | fastest'
        elif r is fastest:
            tag = '← best MAE, fastest' if r is best_mae else '← fastest'

        ratio = r['wall_total_s'] / max(fastest['wall_total_s'], 1e-6)
        print(
            f"{r['name']:<16} "
            f"{str(r['use_guidance_channel']):>5} {str(r['use_hybrid_condition']):>5}  "
            f"{r['eval_mae']:>8.4f} {r['eval_mse']:>10.4f}  "
            f"{r['wall_diffusion_pretrain_s']:>7.0f} {r['wall_finetune_s']:>7.0f} "
            f"{r['wall_eval_s']:>6.0f} {r['wall_total_s']:>7.0f}  "
            f"{ratio:>8.2f}x{('  ' + tag) if tag else ''}"
        )

    print('=' * len(hdr))
    print(
        f"Best MAE: {best_mae['name']}  (MAE={best_mae['eval_mae']:.4f}, "
        f"MSE={best_mae['eval_mse']:.4f})"
    )
    print(
        f"Fastest wall time: {fastest['name']}  ({fastest['wall_total_s']:.0f}s total)"
    )
    print('=' * len(hdr) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Ablation: iTransformer guidance × hybrid cross-variate context',
    )
    parser.add_argument('--store', type=str, default=None, help='Checkpoint / result root')
    parser.add_argument('--smoke-test', action='store_true', help='1 epoch, tiny data')
    parser.add_argument(
        '--smoke-pipeline',
        action='store_true',
        help=(
            'Full ablation path on minimal budgets: shared iTrans, then all four configs '
            '(diffusion pretrain → finetune → eval). Implies --smoke-test and defaults '
            'T=20, H=32 unless you pass --num-diffusion-steps / --image-height.'
        ),
    )
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb-project', type=str, default='diffusion-tsf')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--gradient-checkpointing', action='store_true')
    parser.add_argument(
        '--num-diffusion-steps',
        type=int,
        default=DEFAULT_NUM_DIFFUSION_STEPS,
        help=f'DDPM steps T (default {DEFAULT_NUM_DIFFUSION_STEPS})',
    )
    parser.add_argument(
        '--image-height',
        type=int,
        default=DEFAULT_IMAGE_HEIGHT,
        help=f'Occupancy image height H (default {DEFAULT_IMAGE_HEIGHT})',
    )
    args = parser.parse_args()

    smoke_pipeline = bool(args.smoke_pipeline)
    if smoke_pipeline:
        args.smoke_test = True
        if args.num_diffusion_steps == DEFAULT_NUM_DIFFUSION_STEPS:
            args.num_diffusion_steps = 20
        if args.image_height == DEFAULT_IMAGE_HEIGHT:
            args.image_height = 32

    setup_logging()

    if args.store is None:
        scratch = os.environ.get('SCRATCH', '')
        if scratch:
            args.store = os.path.join(scratch, 'diffusion-tsf-guidance-hybrid-ablation')
        else:
            args.store = os.path.join(_script_dir, 'checkpoints_guidance_hybrid_ablation')

    store = _make_store(args.store)
    print(f'[ablation] Store: {store}')
    print(
        f'[ablation] Geometry: T={args.num_diffusion_steps}, H={args.image_height}  '
        f'dataset={DATASET} variates={VARIATE_INDICES}'
        + ('  (smoke-pipeline)' if smoke_pipeline else '')
    )

    if args.amp:
        P.USE_AMP = True
    if args.gradient_checkpointing:
        P.USE_GRADIENT_CHECKPOINTING = True

    wandb_run = None
    if args.wandb:
        try:
            import wandb

            from models.diffusion_tsf.train_multivariate_pipeline import _require_wandb_api_key_or_exit

            _require_wandb_api_key_or_exit()
            date_tag = datetime.now().strftime('%m-%d')
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=f'{date_tag}-guidance-hybrid-ablation-T{args.num_diffusion_steps}H{args.image_height}',
                group=f'{date_tag}-guidance-hybrid-ablation',
                config={
                    'fixed_lr': FIXED_LR,
                    'fixed_batch': FIXED_BATCH,
                    'n_variates': N_VARIATES,
                    'dataset': DATASET,
                    'variate_indices': VARIATE_INDICES,
                    'T': args.num_diffusion_steps,
                    'H': args.image_height,
                    'ablations': ABLATION_CONFIGS,
                    'smoke_test': args.smoke_test,
                    'smoke_pipeline': smoke_pipeline,
                },
                tags=['throwaway', 'guidance-hybrid-ablation'],
            )
        except Exception as e:
            print(f'[ablation] wandb init failed: {e}. Continuing without wandb.')
            wandb_run = None

    itrans_ckpt = _ensure_itransformer(store, args.smoke_test, smoke_pipeline)

    results: List[Dict[str, Any]] = []
    for name, ug, uh in ABLATION_CONFIGS:
        try:
            r = run_ablation(
                name,
                ug,
                uh,
                args.num_diffusion_steps,
                args.image_height,
                itrans_ckpt,
                store,
                args.smoke_test,
                smoke_pipeline,
                wandb_run,
            )
            results.append(r)
        except Exception as e:
            print(f'[ablation] {name}: FAILED — {e}')
            import traceback

            traceback.print_exc()

    if not results:
        print('[ablation] All runs failed.')
        return

    _print_results_table(results)

    summary_path = os.path.join(store, 'ablation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(
            {
                'results': results,
                'T': args.num_diffusion_steps,
                'H': args.image_height,
                'smoke_pipeline': smoke_pipeline,
            },
            f,
            indent=2,
        )
    print(f'[ablation] Summary saved: {summary_path}')

    if wandb_run is not None:
        try:
            import wandb as _w

            best = min(results, key=lambda x: x['eval_mae'])
            _w.run.summary['best_ablation_name'] = best['name']
            _w.run.summary['best_mae'] = best['eval_mae']
            _w.finish()
        except Exception:
            pass


if __name__ == '__main__':
    main()
