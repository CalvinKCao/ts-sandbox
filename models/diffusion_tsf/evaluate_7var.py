"""
Evaluation Script for 7-Variate Models.

Evaluates trained models on test set with:
1. Single sample (random diffusion seed) - quick baseline
2. Averaged samples (~30 repeated samples) - robust metrics

Metrics:
- MSE, MAE (standard forecasting metrics)
- Shape metrics (from metrics.py)

Usage:
    # Evaluate all completed models
    python -m models.diffusion_tsf.evaluate_7var
    
    # Evaluate specific model
    python -m models.diffusion_tsf.evaluate_7var --subset traffic-5
    
    # Change number of samples for averaging
    python -m models.diffusion_tsf.evaluate_7var --n-samples 50
    
    # Smoke test
    python -m models.diffusion_tsf.evaluate_7var --smoke-test
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Setup path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.diffusion_tsf.config import DiffusionTSFConfig
from models.diffusion_tsf.diffusion_model import DiffusionTSF
from models.diffusion_tsf.metrics import compute_metrics

# Import from training script
from models.diffusion_tsf.train_7var_pipeline import (
    CHECKPOINT_DIR,
    MANIFEST_PATH,
    DATASETS_DIR,
    DATASET_REGISTRY,
    LOOKBACK_LENGTH,
    FORECAST_LENGTH,
    IMAGE_HEIGHT,
    N_VARIATES,
    TrainingManifest,
    TimeSeriesDataset,
    create_diffusion_model,
    load_dataset,
    evaluate_itransformer_baseline,
    _load_subset_results,
    _save_subset_results,
    update_summary_csv,
)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Results directory
RESULTS_DIR = os.path.join(script_dir, 'results_7var')


# ============================================================================
# Evaluation Functions
# ============================================================================

def generate_single_sample(
    model: DiffusionTSF,
    past: torch.Tensor,
    device: torch.device,
    seed: int = None,
) -> torch.Tensor:
    """Generate a single forecast sample from the diffusion model.
    
    Args:
        model: Trained DiffusionTSF model
        past: Past context (batch, variates, lookback)
        device: Device to run on
        seed: Optional random seed for reproducibility
        
    Returns:
        Forecast tensor (batch, variates, horizon)
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    model.eval()
    with torch.no_grad():
        past = past.to(device)
        # model.generate returns dict with 'prediction' key
        result = model.generate(past)
        forecast = result['prediction']
    
    return forecast


def generate_averaged_samples(
    model: DiffusionTSF,
    past: torch.Tensor,
    device: torch.device,
    n_samples: int = 30,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate multiple samples and return mean and std.
    
    Args:
        model: Trained DiffusionTSF model
        past: Past context (batch, variates, lookback)
        device: Device to run on
        n_samples: Number of samples to generate
        
    Returns:
        Tuple of (mean_forecast, std_forecast), each (batch, variates, horizon)
    """
    model.eval()
    samples = []
    
    with torch.no_grad():
        past = past.to(device)
        for i in range(n_samples):
            result = model.generate(past)
            forecast = result['prediction']
            samples.append(forecast.cpu())
    
    # Stack and compute statistics
    stacked = torch.stack(samples, dim=0)  # (n_samples, batch, variates, horizon)
    mean_forecast = stacked.mean(dim=0)
    std_forecast = stacked.std(dim=0)
    
    return mean_forecast, std_forecast


def compute_shape_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Compute shape-preservation metrics.
    
    Args:
        pred: Predicted forecast (batch, variates, horizon)
        target: Ground truth (batch, variates, horizon)
        
    Returns:
        Dict with shape metrics
    """
    # Flatten for metrics
    pred_flat = pred.reshape(-1, pred.shape[-1])
    target_flat = target.reshape(-1, target.shape[-1])
    
    metrics = {}
    
    # Trend accuracy (sign of first difference)
    pred_diff = pred_flat[:, 1:] - pred_flat[:, :-1]
    target_diff = target_flat[:, 1:] - target_flat[:, :-1]
    trend_match = ((pred_diff > 0) == (target_diff > 0)).float().mean()
    metrics['trend_accuracy'] = trend_match.item()
    
    # Correlation (per-sample, then averaged)
    correlations = []
    for i in range(pred_flat.shape[0]):
        if pred_flat[i].std() > 1e-6 and target_flat[i].std() > 1e-6:
            corr = torch.corrcoef(torch.stack([pred_flat[i], target_flat[i]]))[0, 1]
            if not torch.isnan(corr):
                correlations.append(corr.item())
    metrics['correlation'] = np.mean(correlations) if correlations else 0.0
    
    # Dynamic Time Warping approximation (simplified - just endpoint distance)
    endpoint_error = torch.abs(pred_flat[:, -1] - target_flat[:, -1]).mean()
    metrics['endpoint_mae'] = endpoint_error.item()
    
    return metrics


def evaluate_model(
    model: DiffusionTSF,
    test_loader: DataLoader,
    device: torch.device,
    n_samples: int = 30,
    smoke_test: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Evaluate model on test set.
    
    Returns dict with:
        - 'single': metrics from single random sample
        - 'averaged': metrics from averaged samples
    """
    model.eval()
    
    all_preds_single = []
    all_preds_avg = []
    all_targets = []
    
    n_batches = len(test_loader)
    if smoke_test:
        n_batches = min(2, n_batches)
    
    for batch_idx, (past, future) in enumerate(test_loader):
        if batch_idx >= n_batches:
            break
        
        logger.info(f"  Evaluating batch {batch_idx + 1}/{n_batches}")
        
        # Single sample (with fixed seed for reproducibility)
        pred_single = generate_single_sample(model, past, device, seed=42 + batch_idx)
        all_preds_single.append(pred_single.cpu())
        
        # Averaged samples
        n_avg = n_samples if not smoke_test else 3
        pred_avg, _ = generate_averaged_samples(model, past, device, n_samples=n_avg)
        all_preds_avg.append(pred_avg)
        
        all_targets.append(future)
    
    # Concatenate
    preds_single = torch.cat(all_preds_single, dim=0)
    preds_avg = torch.cat(all_preds_avg, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    # Compute metrics
    results = {}
    
    # Single sample metrics
    mse_single = torch.nn.functional.mse_loss(preds_single, targets).item()
    mae_single = torch.nn.functional.l1_loss(preds_single, targets).item()
    shape_single = compute_shape_metrics(preds_single, targets)
    
    results['single'] = {
        'mse': mse_single,
        'mae': mae_single,
        **shape_single,
    }
    
    # Averaged sample metrics
    mse_avg = torch.nn.functional.mse_loss(preds_avg, targets).item()
    mae_avg = torch.nn.functional.l1_loss(preds_avg, targets).item()
    shape_avg = compute_shape_metrics(preds_avg, targets)
    
    results['averaged'] = {
        'mse': mse_avg,
        'mae': mae_avg,
        **shape_avg,
    }
    
    return results


def load_model_for_eval(subset_id: str, checkpoint_dir: str = CHECKPOINT_DIR) -> Tuple[DiffusionTSF, Dict]:
    """Load a trained model for evaluation.
    
    Returns:
        Tuple of (model, metadata)
    """
    subset_dir = os.path.join(checkpoint_dir, subset_id)
    best_ckpt = os.path.join(subset_dir, 'best.pt')
    metadata_path = os.path.join(subset_dir, 'metadata.json')
    
    if not os.path.exists(best_ckpt):
        raise FileNotFoundError(f"No checkpoint found for {subset_id}")
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Create and load model
    model = create_diffusion_model(n_variates=N_VARIATES)
    checkpoint = torch.load(best_ckpt, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, metadata


# ============================================================================
# Main Evaluation
# ============================================================================

def evaluate_all_models(
    n_samples: int = 30,
    smoke_test: bool = False,
    only_subset: str = None,
    checkpoint_dir: str = CHECKPOINT_DIR,
    results_dir: str = RESULTS_DIR,
):
    """Evaluate all completed models."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load manifest — derive path from checkpoint_dir so cluster storage works
    manifest_path = os.path.join(checkpoint_dir, 'training_manifest.json')
    if not os.path.exists(manifest_path):
        logger.error(f"No training manifest found at {manifest_path}")
        return
    
    manifest = TrainingManifest.load(manifest_path)
    
    # Get completed subsets
    completed = [k for k, v in manifest.subsets.items() if v.get('status') == 'complete']
    
    if only_subset:
        if only_subset not in completed:
            logger.error(f"Subset '{only_subset}' not found or not complete")
            return
        completed = [only_subset]
    
    logger.info(f"Evaluating {len(completed)} models")
    
    # Results storage
    os.makedirs(results_dir, exist_ok=True)
    all_results = {}
    
    for subset_id in completed:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {subset_id}")
        logger.info(f"{'='*60}")
        
        try:
            # Load model
            model, metadata = load_model_for_eval(subset_id, checkpoint_dir)
            model = model.to(device)
            
            # Get dataset name and variate indices
            dataset_name = metadata['dataset_name']
            variate_indices = metadata['variate_indices']
            
            # Load test data
            _, _, test_ds, _ = load_dataset(
                dataset_name, variate_indices,
                stride=LOOKBACK_LENGTH,  # No overlap for test
            )
            
            if smoke_test:
                test_ds = Subset(test_ds, list(range(min(4, len(test_ds)))))
            
            test_loader = DataLoader(test_ds, batch_size=8 if not smoke_test else 2, shuffle=False)
            
            # Evaluate
            results = evaluate_model(
                model, test_loader, device,
                n_samples=n_samples,
                smoke_test=smoke_test,
            )
            
            logger.info(f"\nResults for {subset_id}:")
            logger.info(f"  Single sample: MSE={results['single']['mse']:.4f}, MAE={results['single']['mae']:.4f}")
            logger.info(f"  Averaged ({n_samples}): MSE={results['averaged']['mse']:.4f}, MAE={results['averaged']['mae']:.4f}")
            logger.info(f"  Shape metrics (avg): trend_acc={results['averaged']['trend_accuracy']:.3f}, "
                       f"corr={results['averaged']['correlation']:.3f}")

            # Merge into per-subset results.json (preserves itransformer_metrics if present)
            data = _load_subset_results(results_dir, subset_id)
            data.update({
                'subset_id': subset_id,
                'dataset': dataset_name,
                'variate_indices': variate_indices,
                'eval_metrics': results,
                'evaluated_at': datetime.now().isoformat(),
            })
            _save_subset_results(results_dir, subset_id, data)
            update_summary_csv(results_dir)

            all_results[subset_id] = data
            
        except Exception as e:
            logger.error(f"Error evaluating {subset_id}: {e}")
            all_results[subset_id] = {'error': str(e)}
            continue
    
    logger.info(f"\n{'='*60}")
    logger.info("EVALUATION COMPLETE")
    logger.info(f"Results saved to: {results_dir}/{{subset_id}}/results.json")
    logger.info(f"{'='*60}")


def create_summary_csv(all_results: Dict, results_dir: str):
    """Create summary CSV from results."""
    rows = []
    
    for subset_id, data in all_results.items():
        if 'error' in data:
            continue
        
        metrics = data['metrics']
        row = {
            'subset_id': subset_id,
            'dataset': data['dataset'],
            'n_variates': len(data['variate_indices']),
            # Single sample
            'single_mse': metrics['single']['mse'],
            'single_mae': metrics['single']['mae'],
            'single_trend_acc': metrics['single']['trend_accuracy'],
            'single_corr': metrics['single']['correlation'],
            # Averaged
            'avg_mse': metrics['averaged']['mse'],
            'avg_mae': metrics['averaged']['mae'],
            'avg_trend_acc': metrics['averaged']['trend_accuracy'],
            'avg_corr': metrics['averaged']['correlation'],
        }
        rows.append(row)
    
    if rows:
        df = pd.DataFrame(rows)
        
        # Sort by dataset then subset
        df = df.sort_values(['dataset', 'subset_id'])
        
        # Save
        csv_path = os.path.join(results_dir, 'summary.csv')
        df.to_csv(csv_path, index=False)
        logger.info(f"Summary CSV saved to: {csv_path}")
        
        # Print summary statistics
        logger.info("\nSummary Statistics:")
        logger.info(f"  Total models evaluated: {len(df)}")
        logger.info(f"  Avg MSE (single): {df['single_mse'].mean():.4f}")
        logger.info(f"  Avg MSE (averaged): {df['avg_mse'].mean():.4f}")
        logger.info(f"  Avg MAE (single): {df['single_mae'].mean():.4f}")
        logger.info(f"  Avg MAE (averaged): {df['avg_mae'].mean():.4f}")
        
        # Per-dataset summary
        logger.info("\nPer-Dataset Summary (averaged metrics):")
        for dataset in df['dataset'].unique():
            ds_df = df[df['dataset'] == dataset]
            logger.info(f"  {dataset}: MSE={ds_df['avg_mse'].mean():.4f}, "
                       f"MAE={ds_df['avg_mae'].mean():.4f}, "
                       f"n_subsets={len(ds_df)}")


# ============================================================================
# Standalone iTransformer Baseline Runner
# ============================================================================

def run_baseline_eval(
    checkpoint_dir: str = CHECKPOINT_DIR,
    results_dir: str = RESULTS_DIR,
    smoke_test: bool = False,
    only_subset: str = None,
):
    """Re-run iTransformer baseline eval on all completed subsets.

    Useful for populating itransformer_baseline.json on runs that predate
    the automatic baseline evaluation, or to regenerate it after changes.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    manifest_path = os.path.join(checkpoint_dir, 'training_manifest.json')
    if not os.path.exists(manifest_path):
        logger.error(f"No training manifest found at {manifest_path}")
        return

    manifest = TrainingManifest.load(manifest_path)
    completed = [k for k, v in manifest.subsets.items() if v.get('status') == 'complete']
    if only_subset:
        completed = [only_subset] if only_subset in completed else []
        if not completed:
            logger.error(f"Subset '{only_subset}' not found or not complete")
            return

    # Locate pretrained iTransformer checkpoint
    itrans_ckpt = os.path.join(checkpoint_dir, 'pretrained_itransformer.pt')
    if not os.path.exists(itrans_ckpt):
        logger.error(f"iTransformer checkpoint not found: {itrans_ckpt}")
        return

    logger.info(f"Running iTransformer baseline on {len(completed)} subsets")
    for subset_id in completed:
        info = manifest.subsets[subset_id]
        dataset_name = info.get('dataset', subset_id)
        variate_indices = info.get('variate_indices', [])
        try:
            evaluate_itransformer_baseline(
                subset_id, dataset_name, variate_indices,
                itrans_ckpt, results_dir, device, smoke_test=smoke_test,
            )
        except Exception as e:
            logger.error(f"Baseline eval failed for {subset_id}: {e}")

    logger.info(f"Baseline results saved to: {os.path.join(results_dir, 'itransformer_baseline.json')}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate 7-Variate Models')
    parser.add_argument('--subset', type=str, default=None,
                        help='Evaluate only specific subset')
    parser.add_argument('--n-samples', type=int, default=30,
                        help='Number of samples for averaging (default: 30)')
    parser.add_argument('--smoke-test', action='store_true',
                        help='Quick test with minimal data')
    parser.add_argument('--checkpoint-dir', type=str, default=CHECKPOINT_DIR,
                        help='Directory containing checkpoints')
    parser.add_argument('--results-dir', type=str, default=RESULTS_DIR,
                        help='Directory to save results')
    parser.add_argument('--baseline', action='store_true',
                        help='Run iTransformer-only baseline eval on all completed subsets '
                             '(generates itransformer_baseline.json for summarize_results.py)')
    
    args = parser.parse_args()

    if args.baseline:
        run_baseline_eval(
            checkpoint_dir=args.checkpoint_dir,
            results_dir=args.results_dir,
            smoke_test=args.smoke_test,
            only_subset=args.subset,
        )
        return
    
    evaluate_all_models(
        n_samples=args.n_samples,
        smoke_test=args.smoke_test,
        only_subset=args.subset,
        checkpoint_dir=args.checkpoint_dir,
        results_dir=args.results_dir,
    )


if __name__ == '__main__':
    main()

