"""Throwaway grid-search: num_diffusion_steps × image_height.

Pretrains iTransformer once (cached), then for each (T, H) combo
runs a short diffusion pretrain (max 10 epochs) followed by a quick finetune
on a 4-variate electricity subset (consumers 93, 292, 81, 84) and eval.  Reports the best combo at the end.

Fixed config: lr=2.05e-5, batch_size=16, n_variates=4, dataset=electricity.
Sweep space: T ∈ {200,500,1000,1500} × H ∈ {64,128}

Usage (login node, from repo root):
    python -m models.diffusion_tsf.throwaway_diffsteps_sweep \\
        --wandb --wandb-project diffusion-tsf \\
        --store /scratch/ccao87/diffusion-tsf-diffsteps-sweep

    # smoke test (1 epoch, tiny data):
    python -m models.diffusion_tsf.throwaway_diffsteps_sweep --smoke-test
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

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
    iTransformerGuidance,
    pretrain_itransformer,
    pretrain_diffusion,
    finetune_on_dataset,
    evaluate_model,
    evaluate_itransformer_baseline,
    load_dataset,
    save_checkpoint,
    get_device,
    get_synthetic_dataloader,
    EarlyStopping,
    logger as _pipeline_logger,
    setup_logging,
)

# ---------------------------------------------------------------------------
# Sweep config — fixed params
# ---------------------------------------------------------------------------
# Electricity dataset, 4 hand-picked variates (hourly consumption, 321 consumers):
#  93  — highly seasonal (acf@24h=0.985), flat trend; correlated with 292 (r=0.968)
# 292  — highly seasonal (0.981), correlated with 93; different noise profile
#  81  — random-walk-like trajectory (ar1=0.981, low mean reversion)
#  84  — noisy and less seasonal than most channels (seasonality=0.491)
# Criteria: (93,292) are closely correlated; 81 behaves like a random walk; 84 is noisy.
DATASET = 'electricity'
VARIATE_INDICES = [93, 292, 81, 84]
VARIATE_NAMES = ['93', '292', '81', '84']
N_VARIATES = 4

FIXED_LR = 2.05e-5
FIXED_BATCH = 16
FIXED_DROPOUT = 0.1

MAX_DIFFUSION_PRETRAIN_EPOCHS = 10
DIFFUSION_PRETRAIN_PATIENCE = 5

# iTransformer pretrain keeps full default epochs (pretrain once, reuse)
ITRANS_PRETRAIN_SAMPLES = 10000   # smaller than full 60k to keep it fast for a throwaway
ITRANS_PRETRAIN_EPOCHS = 30
ITRANS_PRETRAIN_PATIENCE = 5

# Finetune budget per trial (short — we only need enough signal to rank combos)
FINETUNE_EPOCHS = 20
FINETUNE_PATIENCE = 5

# Explicit combos: H=128 for all T values, plus H=64 specifically for T=1000
# (lower H at T=1000 lets us isolate whether image resolution vs step count
# is the bottleneck at the highest T setting)
SWEEP_COMBOS = [
    (200, 128),
    (500, 128),
    (1000, 128),
    (1000, 64),
    (1500, 128),
]

# Smoke-test overrides
SMOKE_ITRANS_SAMPLES = 64
SMOKE_PRETRAIN_EPOCHS = 1
SMOKE_FINETUNE_EPOCHS = 1


def _make_store(base_store: str) -> str:
    os.makedirs(base_store, exist_ok=True)
    return base_store


def _trial_dir(store: str, T: int, H: int) -> str:
    d = os.path.join(store, f'T{T}_H{H}')
    os.makedirs(d, exist_ok=True)
    return d


def _itrans_ckpt_path(store: str) -> str:
    return os.path.join(store, 'shared_itransformer.pt')


# ---------------------------------------------------------------------------
# Phase 1C-1: iTransformer pretrain (shared across all trials)
# ---------------------------------------------------------------------------
def _ensure_itransformer(store: str, smoke_test: bool) -> str:
    ckpt = _itrans_ckpt_path(store)
    if os.path.exists(ckpt):
        print(f"[sweep] Reusing cached iTransformer: {ckpt}")
        return ckpt

    print("[sweep] Pretraining iTransformer (shared, runs once) ...")

    # Temporarily override the pipeline globals this call depends on
    P.N_VARIATES = N_VARIATES

    best_params = {'learning_rate': FIXED_LR, 'dropout': FIXED_DROPOUT}
    n_samples = SMOKE_ITRANS_SAMPLES if smoke_test else ITRANS_PRETRAIN_SAMPLES
    epochs = SMOKE_PRETRAIN_EPOCHS if smoke_test else ITRANS_PRETRAIN_EPOCHS
    patience = 1 if smoke_test else ITRANS_PRETRAIN_PATIENCE

    result = pretrain_itransformer(
        best_params,
        n_samples=n_samples,
        epochs=epochs,
        patience=patience,
        checkpoint_dir=store,
        smoke_test=smoke_test,
    )
    # pretrain_itransformer writes to checkpoint_dir/pretrained_itransformer.pt;
    # move it to our shared name so it doesn't clash with per-trial dirs
    default_path = os.path.join(store, 'pretrained_itransformer.pt')
    if os.path.exists(default_path) and default_path != ckpt:
        os.replace(default_path, ckpt)
    return ckpt


# ---------------------------------------------------------------------------
# One (T, H) trial: diffusion pretrain → finetune → eval
# ---------------------------------------------------------------------------
def run_trial(T: int, H: int, itrans_ckpt: str, store: str,
              smoke_test: bool, wandb_run=None) -> dict:
    """Train a diffusion model with T steps and H image height, finetune on 4-variate electricity,
    and return eval metrics."""
    trial_dir = _trial_dir(store, T, H)
    result_path = os.path.join(trial_dir, 'result.json')
    if os.path.exists(result_path):
        with open(result_path) as f:
            cached = json.load(f)
        print(f"[sweep] T={T} H={H}: using cached result "
              f"(MAE={cached.get('eval_mae','?'):.4f})")
        return cached

    device = get_device()

    # Override pipeline globals for this trial's model geometry
    P.N_VARIATES = N_VARIATES
    P.IMAGE_HEIGHT = H

    # ------------------------------------------------------------------
    # Phase 1C-2: diffusion pretrain (max MAX_DIFFUSION_PRETRAIN_EPOCHS)
    # ------------------------------------------------------------------
    diff_ckpt = os.path.join(trial_dir, 'pretrained_diffusion.pt')
    if not os.path.exists(diff_ckpt):
        print(f"[sweep] T={T} H={H}: diffusion pretrain ...")
        best_params = {'learning_rate': FIXED_LR}
        diff_ckpt = pretrain_diffusion(
            best_params,
            itrans_ckpt,
            n_samples=SMOKE_ITRANS_SAMPLES if smoke_test else 5000,
            epochs=SMOKE_PRETRAIN_EPOCHS if smoke_test else MAX_DIFFUSION_PRETRAIN_EPOCHS,
            patience=1 if smoke_test else DIFFUSION_PRETRAIN_PATIENCE,
            checkpoint_dir=trial_dir,
            smoke_test=smoke_test,
        )

    # ------------------------------------------------------------------
    # Phase 2: finetune on 4-variate electricity subset
    # ------------------------------------------------------------------
    subset_info = {
        'subset_id': f'elec-4v-T{T}H{H}',
        'dataset': DATASET,
        'variate_indices': VARIATE_INDICES,
        'variate_names': VARIATE_NAMES,
    }
    tuned_params = {'learning_rate': FIXED_LR, 'batch_size': FIXED_BATCH}

    ft_ckpt, train_metrics = finetune_on_dataset(
        subset_info, diff_ckpt, itrans_ckpt, tuned_params,
        epochs=SMOKE_FINETUNE_EPOCHS if smoke_test else FINETUNE_EPOCHS,
        patience=1 if smoke_test else FINETUNE_PATIENCE,
        checkpoint_dir=trial_dir,
        smoke_test=smoke_test,
    )

    # ------------------------------------------------------------------
    # Eval
    # ------------------------------------------------------------------
    print(f"[sweep] T={T} H={H}: evaluating ...")
    itrans_model = create_itransformer().to(device)
    ckpt_data = torch.load(itrans_ckpt, map_location=device, weights_only=False)
    itrans_model.load_state_dict(ckpt_data['model_state_dict'])
    guidance = iTransformerGuidance(
        itrans_model, use_norm=True,
        seq_len=P.LOOKBACK_LENGTH, pred_len=P.FORECAST_LENGTH,
    )

    model = create_diffusion_model(
        use_guidance=True,
        diffusion_type=P.DIFFUSION_TYPE,
        num_diffusion_steps=T,
        image_height=H,
    ).to(device)
    model.set_guidance_model(guidance)
    ckpt_data = torch.load(ft_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt_data['model_state_dict'])

    _, _, test_ds, _ = load_dataset(DATASET, VARIATE_INDICES, stride=P.LOOKBACK_LENGTH)
    if smoke_test:
        from torch.utils.data import Subset
        test_ds = Subset(test_ds, list(range(min(2, len(test_ds)))))
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_ds, batch_size=4 if not smoke_test else 2, shuffle=False)

    eval_results = evaluate_model(model, test_loader, device,
                                  n_samples=5 if smoke_test else 30,
                                  smoke_test=smoke_test)

    result = {
        'T': T,
        'H': H,
        'eval_mae': eval_results['averaged']['mae'],
        'eval_mse': eval_results['averaged']['mse'],
        'train_metrics': train_metrics,
    }

    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)

    if wandb_run is not None:
        try:
            import wandb
            wandb.log({
                f'grid/T{T}_H{H}_mae': result['eval_mae'],
                f'grid/T{T}_H{H}_mse': result['eval_mse'],
                'T': T, 'H': H,
            })
            wandb.run.summary[f'T{T}_H{H}_mae'] = result['eval_mae']
        except Exception:
            pass

    print(f"[sweep] T={T} H={H}: MAE={result['eval_mae']:.4f}  MSE={result['eval_mse']:.4f}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Diffusion steps × image-height sweep')
    parser.add_argument('--store', type=str, default=None,
                        help='Base storage directory for checkpoints and results')
    parser.add_argument('--smoke-test', action='store_true',
                        help='Fast validation run (1 epoch, tiny data)')
    parser.add_argument('--wandb', action='store_true', help='Enable wandb')
    parser.add_argument('--wandb-project', type=str, default='diffusion-tsf')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--gradient-checkpointing', action='store_true')
    args = parser.parse_args()

    setup_logging()

    if args.store is None:
        scratch = os.environ.get('SCRATCH', '')
        if scratch:
            args.store = os.path.join(scratch, 'diffusion-tsf-diffsteps-sweep')
        else:
            args.store = os.path.join(_script_dir, 'checkpoints_diffsteps_sweep')

    store = _make_store(args.store)
    print(f"[sweep] Store: {store}")

    if args.amp:
        P.USE_AMP = True
    if args.gradient_checkpointing:
        P.USE_GRADIENT_CHECKPOINTING = True

    # wandb
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            from models.diffusion_tsf.train_multivariate_pipeline import _require_wandb_api_key_or_exit
            _require_wandb_api_key_or_exit()
            date_tag = datetime.now().strftime('%m-%d')
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=f'{date_tag}-diffsteps-img-sweep',
                group=f'{date_tag}-diffsteps-img-sweep',
                config={
                    'fixed_lr': FIXED_LR,
                    'fixed_batch': FIXED_BATCH,
                    'fixed_dropout': FIXED_DROPOUT,
                    'n_variates': N_VARIATES,
                    'dataset': DATASET,
                    'variate_indices': VARIATE_INDICES,
                    'max_diffusion_epochs': MAX_DIFFUSION_PRETRAIN_EPOCHS,
                    'combos': SWEEP_COMBOS,
                    'smoke_test': args.smoke_test,
                },
                tags=['throwaway', 'diffsteps-sweep'],
            )
        except Exception as e:
            print(f"[sweep] wandb init failed: {e}. Continuing without wandb.")
            wandb_run = None

    # Phase 1C-1: iTransformer pretrain (once, shared)
    itrans_ckpt = _ensure_itransformer(store, args.smoke_test)

    combos = SWEEP_COMBOS
    print(f"\n[sweep] Starting sweep: {len(combos)} combos: {combos}\n")

    results = []
    for T, H in combos:
        t0 = time.time()
        try:
            r = run_trial(T, H, itrans_ckpt, store, args.smoke_test, wandb_run)
            results.append(r)
        except Exception as e:
            print(f"[sweep] T={T} H={H}: FAILED — {e}")
            import traceback; traceback.print_exc()
        print(f"[sweep] T={T} H={H} done in {time.time()-t0:.0f}s\n")

    if not results:
        print("[sweep] All trials failed.")
        return

    # Sort by averaged MAE
    results.sort(key=lambda x: x['eval_mae'])
    best = results[0]

    print("\n" + "=" * 60)
    print("SWEEP RESULTS (ranked by MAE)")
    print("=" * 60)
    for r in results:
        marker = " ← best" if r is best else ""
        print(f"  T={r['T']:>4}  H={r['H']:>3}  "
              f"MAE={r['eval_mae']:.4f}  MSE={r['eval_mse']:.4f}{marker}")
    print("=" * 60)
    print(f"Best combo: T={best['T']}, H={best['H']}")
    print(f"  → set NUM_DIFFUSION_STEPS={best['T']} and IMAGE_HEIGHT={best['H']} "
          f"in train_multivariate_pipeline.py constants")
    print("=" * 60)

    summary_path = os.path.join(store, 'sweep_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({'results': results, 'best': best}, f, indent=2)
    print(f"\n[sweep] Summary saved: {summary_path}")

    if wandb_run is not None:
        try:
            import wandb as _w
            _w.run.summary['best_T'] = best['T']
            _w.run.summary['best_H'] = best['H']
            _w.run.summary['best_mae'] = best['eval_mae']
            _w.run.summary['best_mse'] = best['eval_mse']
            _w.finish()
        except Exception:
            pass


if __name__ == '__main__':
    main()
