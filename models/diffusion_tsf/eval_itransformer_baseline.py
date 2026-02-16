"""
Evaluate iTransformer guidance alone (baseline comparison).

Runs the pretrained iTransformer on the same test sets used for diffusion
evaluation, so you can compare diffusion vs guidance-only performance.

Usage:
    python -m models.diffusion_tsf.eval_itransformer_baseline \
        --checkpoint-dir /path/to/checkpoints \
        --results-dir /path/to/results
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.diffusion_tsf.train_7var_pipeline import (
    CHECKPOINT_DIR, RESULTS_DIR, DATASETS_DIR, DATASET_REGISTRY,
    LOOKBACK_LENGTH, FORECAST_LENGTH, N_VARIATES,
    TrainingManifest, TimeSeriesDataset,
    create_itransformer,
)


def evaluate_itransformer_on_subset(
    itrans_model: torch.nn.Module,
    dataset_name: str,
    variate_indices: list,
    device: torch.device,
) -> dict:
    """Evaluate iTransformer predictions on a dataset subset."""
    
    # Load data
    path = os.path.join(DATASETS_DIR, DATASET_REGISTRY[dataset_name][0])
    date_col = DATASET_REGISTRY[dataset_name][1]
    
    df = pd.read_csv(path)
    data_cols = [c for c in df.columns if c != date_col]
    data = df[data_cols].values.astype(np.float32)
    
    if variate_indices is not None:
        data = data[:, variate_indices]
    
    # Normalize
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True) + 1e-8
    data = (data - mean) / std
    
    # Test split (last 20%)
    n = len(data)
    val_end = int(n * 0.8)
    test_data = data[val_end:]
    
    test_ds = TimeSeriesDataset(test_data, LOOKBACK_LENGTH, FORECAST_LENGTH, stride=LOOKBACK_LENGTH)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)
    
    if len(test_ds) == 0:
        return None
    
    # Evaluate
    itrans_model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for past, future in test_loader:
            # past: (B, vars, seq_len) -> iTransformer expects (B, seq_len, vars)
            x_enc = past.permute(0, 2, 1).to(device)
            y_true = future  # Keep on CPU for metrics
            
            # iTransformer forward
            y_pred = itrans_model(x_enc, None, None, None)  # (B, pred_len, vars)
            y_pred = y_pred.cpu()
            
            # Convert to (B, vars, pred_len) to match diffusion format
            y_pred = y_pred.permute(0, 2, 1)
            
            all_preds.append(y_pred)
            all_targets.append(y_true)
    
    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    # Compute metrics
    mse = torch.nn.functional.mse_loss(preds, targets).item()
    mae = torch.nn.functional.l1_loss(preds, targets).item()
    
    # Trend accuracy
    pred_diff = preds[:, :, 1:] - preds[:, :, :-1]
    target_diff = targets[:, :, 1:] - targets[:, :, :-1]
    trend_acc = ((pred_diff > 0) == (target_diff > 0)).float().mean().item()
    
    return {'mse': mse, 'mae': mae, 'trend_accuracy': trend_acc}


def main():
    parser = argparse.ArgumentParser(description='Evaluate iTransformer baseline')
    parser.add_argument('--checkpoint-dir', type=str, default=CHECKPOINT_DIR)
    parser.add_argument('--results-dir', type=str, default=RESULTS_DIR)
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load iTransformer
    itrans_ckpt = os.path.join(args.checkpoint_dir, 'pretrained_itransformer.pt')
    if not os.path.exists(itrans_ckpt):
        print(f"No iTransformer checkpoint found at {itrans_ckpt}")
        return
    
    print(f"Loading iTransformer from {itrans_ckpt}")
    itrans_model = create_itransformer().to(device)
    ckpt = torch.load(itrans_ckpt, map_location=device, weights_only=False)
    itrans_model.load_state_dict(ckpt['model_state_dict'])
    itrans_model.eval()
    
    # Load manifest to find completed subsets
    manifest_path = os.path.join(args.checkpoint_dir, 'training_manifest.json')
    if not os.path.exists(manifest_path):
        print(f"No manifest found at {manifest_path}")
        return
    
    manifest = TrainingManifest.load(path=manifest_path)
    completed = {k: v for k, v in manifest.subsets.items() if v.get('status') == 'complete'}
    
    print(f"Evaluating iTransformer baseline on {len(completed)} subsets...")
    
    all_results = {}
    
    for subset_id, info in sorted(completed.items()):
        # Parse dataset name
        if '-' in subset_id and subset_id.split('-')[-1].isdigit():
            dataset_name = '-'.join(subset_id.split('-')[:-1])
        else:
            dataset_name = subset_id
        
        variate_indices = info.get('variate_indices')
        
        # Also try metadata.json
        if not variate_indices:
            meta_path = os.path.join(args.checkpoint_dir, subset_id, 'metadata.json')
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                variate_indices = meta.get('variate_indices')
        
        if dataset_name not in DATASET_REGISTRY:
            print(f"  {subset_id}: unknown dataset '{dataset_name}', skipping")
            continue
        
        try:
            metrics = evaluate_itransformer_on_subset(itrans_model, dataset_name, variate_indices, device)
            if metrics is None:
                print(f"  {subset_id}: no test data, skipping")
                continue
            
            all_results[subset_id] = {
                'dataset': dataset_name,
                'variate_indices': variate_indices,
                'itransformer_metrics': metrics,
            }
            
            print(f"  {subset_id}: MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}, "
                  f"Trend={metrics['trend_accuracy']:.3f}")
            
        except Exception as e:
            print(f"  {subset_id}: ERROR - {e}")
    
    # Save results
    os.makedirs(args.results_dir, exist_ok=True)
    output_path = os.path.join(args.results_dir, 'itransformer_baseline.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nSaved to {output_path}")
    
    # Print comparison summary if diffusion results exist
    print("\n" + "=" * 80)
    print("COMPARISON: Diffusion vs iTransformer Baseline")
    print("=" * 80)
    print(f"{'Subset':<20} {'Diff MSE':>10} {'iTrans MSE':>10} {'Δ MSE':>10} {'Diff MAE':>10} {'iTrans MAE':>10} {'Δ MAE':>10}")
    print("-" * 80)
    
    from collections import defaultdict
    by_dataset = defaultdict(list)
    
    for subset_id, itrans_data in sorted(all_results.items()):
        # Load corresponding diffusion results
        diff_path = os.path.join(args.results_dir, f'{subset_id}_results.json')
        if os.path.exists(diff_path):
            with open(diff_path) as f:
                diff_data = json.load(f)
            
            diff_mse = diff_data.get('eval_metrics', {}).get('averaged', {}).get('mse', None)
            diff_mae = diff_data.get('eval_metrics', {}).get('averaged', {}).get('mae', None)
            itrans_mse = itrans_data['itransformer_metrics']['mse']
            itrans_mae = itrans_data['itransformer_metrics']['mae']
            
            if diff_mse is not None:
                delta_mse = diff_mse - itrans_mse
                delta_mae = diff_mae - itrans_mae
                better = "✓" if delta_mse < 0 else "✗"
                
                print(f"{subset_id:<20} {diff_mse:>10.4f} {itrans_mse:>10.4f} {delta_mse:>+10.4f} "
                      f"{diff_mae:>10.4f} {itrans_mae:>10.4f} {delta_mae:>+10.4f} {better}")
                
                by_dataset[itrans_data['dataset']].append({
                    'diff_mse': diff_mse, 'itrans_mse': itrans_mse,
                    'diff_mae': diff_mae, 'itrans_mae': itrans_mae,
                })
    
    # Per-dataset summary
    print("\n" + "=" * 80)
    print("PER-DATASET AVERAGE")
    print("=" * 80)
    print(f"{'Dataset':<15} {'N':>4} {'Diff MSE':>10} {'iTrans MSE':>10} {'Δ MSE':>10} {'Winner':>10}")
    print("-" * 60)
    
    total_diff_wins = 0
    total_count = 0
    
    for dataset in sorted(by_dataset.keys()):
        items = by_dataset[dataset]
        n = len(items)
        avg_diff_mse = sum(i['diff_mse'] for i in items) / n
        avg_itrans_mse = sum(i['itrans_mse'] for i in items) / n
        delta = avg_diff_mse - avg_itrans_mse
        winner = "Diffusion" if delta < 0 else "iTransformer"
        
        if delta < 0:
            total_diff_wins += n
        total_count += n
        
        print(f"{dataset:<15} {n:>4} {avg_diff_mse:>10.4f} {avg_itrans_mse:>10.4f} {delta:>+10.4f} {winner:>10}")
    
    print(f"\nDiffusion wins: {total_diff_wins}/{total_count} subsets")


if __name__ == '__main__':
    main()

