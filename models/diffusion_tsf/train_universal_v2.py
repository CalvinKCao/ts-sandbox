""" 
Universal Training Script v2 - Trains on all datasets with CCM support.

Phases:
  1. PRETRAIN: Train iTransformer + Diffusion on 1M synthetic samples (7 variates)
  2. FINETUNE: Fine-tune on real datasets
     - 7-variate datasets: Direct fine-tuning
     - >7-variate datasets: Use CCM to cluster channels to 7

Usage:
    # Pre-train (run once)
    python -m models.diffusion_tsf.train_universal_v2 --mode pretrain --synthetic-samples 1000000
    
    # Fine-tune on specific dataset
    python -m models.diffusion_tsf.train_universal_v2 --mode finetune --dataset ETTh1
    
    # Fine-tune with CCM (for >7 variate datasets)
    python -m models.diffusion_tsf.train_universal_v2 --mode finetune --dataset electricity
    
    # Smoke test (minimal resources)
    python -m models.diffusion_tsf.train_universal_v2 --smoke-test
"""

import argparse
import logging
import os
import sys
import json
import time
from datetime import datetime
from dataclasses import asdict
from typing import Optional, Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Local imports
from models.diffusion_tsf.config import DiffusionTSFConfig
from models.diffusion_tsf.diffusion_model import DiffusionTSF
from models.diffusion_tsf.dataset import get_synthetic_dataloader
from models.diffusion_tsf.ccm_adapter import CCMAdapter, visualize_clusters
from models.diffusion_tsf.train_electricity import (
    ElectricityDataset,
    DATASET_REGISTRY,
    load_itransformer_from_checkpoint,
    train_epoch,
    validate,
    save_checkpoint,
    EarlyStopping,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)],
    force=True
)
logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

DATASETS_DIR = os.path.join(project_root, 'datasets')
CHECKPOINT_DIR = os.path.join(current_dir, 'checkpoints', 'universal_v2')

# Dataset info: (path, default_target, seasonal_period, n_variates)
DATASET_INFO = {
    'ETTh1': ('ETT-small/ETTh1.csv', 'OT', 24, 7),
    'ETTh2': ('ETT-small/ETTh2.csv', 'OT', 24, 7),
    'ETTm1': ('ETT-small/ETTm1.csv', 'OT', 96, 7),
    'ETTm2': ('ETT-small/ETTm2.csv', 'OT', 96, 7),
    'illness': ('illness/national_illness.csv', 'OT', 52, 7),
    'electricity': ('electricity/electricity.csv', 'OT', 96, 321),
    'weather': ('weather/weather.csv', 'OT', 144, 21),
    'exchange_rate': ('exchange_rate/exchange_rate.csv', 'OT', 5, 8),
    'traffic': ('traffic/traffic.csv', 'OT', 24, 862),
}

# Datasets that need CCM (>7 variates)
CCM_DATASETS = ['electricity', 'weather', 'exchange_rate', 'traffic']

# Fixed architecture params
LOOKBACK_LENGTH = 512
FORECAST_LENGTH = 96
IMAGE_HEIGHT = 64
N_CLUSTERS = 7  # Match pretrained model's variate count


# ============================================================================
# Data Preparation
# ============================================================================

def combine_traffic_files(datasets_dir: str) -> str:
    """Combine traffic_part1.csv and traffic_part2.csv into traffic.csv.
    
    Returns path to combined file.
    """
    traffic_dir = os.path.join(datasets_dir, 'traffic')
    combined_path = os.path.join(traffic_dir, 'traffic.csv')
    part1_path = os.path.join(traffic_dir, 'traffic_part1.csv')
    part2_path = os.path.join(traffic_dir, 'traffic_part2.csv')
    
    if os.path.exists(combined_path):
        logger.info(f"traffic.csv already exists at {combined_path}")
        return combined_path
    
    if not os.path.exists(part1_path) or not os.path.exists(part2_path):
        raise FileNotFoundError(
            f"Cannot find traffic parts. Expected:\n"
            f"  {part1_path}\n  {part2_path}"
        )
    
    logger.info("Combining traffic_part1.csv and traffic_part2.csv...")
    
    with open(part1_path, 'r') as f1, open(part2_path, 'r') as f2, open(combined_path, 'w') as out:
        # Write part1 entirely
        for line in f1:
            out.write(line)
        
        # Skip header of part2, write rest
        next(f2)  # Skip header
        for line in f2:
            out.write(line)
    
    logger.info(f"Created {combined_path}")
    return combined_path


def get_dataset_path(dataset_name: str) -> str:
    """Get full path to dataset CSV."""
    if dataset_name not in DATASET_INFO:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_INFO.keys())}")
    
    rel_path = DATASET_INFO[dataset_name][0]
    full_path = os.path.join(DATASETS_DIR, rel_path)
    
    # Special handling for traffic
    if dataset_name == 'traffic' and not os.path.exists(full_path):
        combine_traffic_files(DATASETS_DIR)
    
    return full_path


def create_dataloaders(
    dataset_name: str,
    batch_size: int,
    lookback: int = LOOKBACK_LENGTH,
    forecast: int = FORECAST_LENGTH,
    max_samples: Optional[int] = None,
    use_multivariate: bool = True,
) -> Tuple[DataLoader, DataLoader, int]:
    """Create train/val dataloaders for a dataset.
    
    Returns:
        (train_loader, val_loader, n_variates)
    """
    data_path = get_dataset_path(dataset_name)
    
    # Load full dataset
    full_dataset = ElectricityDataset(
        data_path,
        lookback=lookback,
        forecast=forecast,
        use_all_columns=use_multivariate,
        max_samples=max_samples,
        augment=False,
    )
    
    n_variates = full_dataset.num_variables
    total = len(full_dataset)
    
    # Chronological split: 70% train, 10% val, 20% test
    train_end = int(total * 0.7)
    val_end = int(total * 0.8)
    
    train_indices = list(range(train_end))
    val_indices = list(range(train_end, val_end))
    
    if max_samples:
        train_indices = train_indices[:max_samples]
        val_indices = val_indices[:max(1, max_samples // 5)]
    
    train_dataset = ElectricityDataset(
        data_path,
        lookback=lookback,
        forecast=forecast,
        use_all_columns=use_multivariate,
        augment=True,
        data_tensor=full_dataset.data,
        indices=train_indices,
    )
    
    val_dataset = ElectricityDataset(
        data_path,
        lookback=lookback,
        forecast=forecast,
        use_all_columns=use_multivariate,
        augment=False,
        data_tensor=full_dataset.data,
        indices=val_indices,
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    logger.info(f"Dataset {dataset_name}: {n_variates} variates, "
                f"{len(train_dataset)} train, {len(val_dataset)} val samples")
    
    return train_loader, val_loader, n_variates


# ============================================================================
# Model Creation
# ============================================================================

def create_diffusion_config(
    n_variates: int = 7,
    lookback: int = LOOKBACK_LENGTH,
    forecast: int = FORECAST_LENGTH,
    smoke_test: bool = False,
) -> DiffusionTSFConfig:
    """Create diffusion model config."""
    
    if smoke_test:
        return DiffusionTSFConfig(
            lookback_length=64,
            forecast_length=16,
            image_height=32,
            num_variables=n_variates,
            unet_channels=[16, 32],
            attention_levels=[1],
            num_res_blocks=1,
            num_diffusion_steps=10,
            ddim_steps=5,
            representation_mode='cdf',
            use_guidance_channel=False,
            use_hybrid_condition=False,
        )
    
    return DiffusionTSFConfig(
        lookback_length=lookback,
        forecast_length=forecast,
        image_height=IMAGE_HEIGHT,
        num_variables=n_variates,
        unet_channels=[64, 128, 256],
        attention_levels=[2],
        num_res_blocks=2,
        num_diffusion_steps=200,
        ddim_steps=20,
        representation_mode='cdf',
        use_guidance_channel=False,
        use_hybrid_condition=True,
    )


# ============================================================================
# Training Functions
# ============================================================================

def pretrain_on_synthetic(
    n_samples: int,
    n_variates: int = 7,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    checkpoint_dir: str = CHECKPOINT_DIR,
    smoke_test: bool = False,
) -> str:
    """Phase 1 & 2: Pretrain diffusion model on synthetic data.
    
    Returns path to saved checkpoint.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info("=" * 60)
    logger.info(f"PRETRAINING ON SYNTHETIC DATA ({n_samples} samples, {n_variates} variates)")
    logger.info("=" * 60)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create config
    config = create_diffusion_config(n_variates=n_variates, smoke_test=smoke_test)
    
    # Create synthetic dataloader
    synthetic_loader = get_synthetic_dataloader(
        num_samples=n_samples,
        lookback_length=config.lookback_length,
        forecast_length=config.forecast_length,
        batch_size=batch_size,
        num_variables=n_variates,
        shuffle=True,
        seed=42,
    )
    
    # Create model
    model = DiffusionTSF(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(epochs):
        epoch_start = time.time()
        
        model.train()
        total_loss = 0
        n_batches = 0
        
        for batch_idx, (past, future) in enumerate(synthetic_loader):
            past = past.to(device)
            future = future.to(device)
            
            optimizer.zero_grad()
            outputs = model(past, future)
            loss = outputs['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
            if batch_idx % 100 == 0:
                logger.info(f"  Epoch {epoch+1} [{batch_idx}/{len(synthetic_loader)}] Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / n_batches
        elapsed = time.time() - epoch_start
        
        logger.info(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
    
    # Save checkpoint
    ckpt_path = os.path.join(checkpoint_dir, 'pretrained_diffusion.pt')
    save_checkpoint(
        model, optimizer, epochs - 1, best_loss, best_loss,
        asdict(config), ckpt_path
    )
    
    logger.info(f"Pretrained checkpoint saved to {ckpt_path}")
    return ckpt_path


def finetune_on_dataset(
    dataset_name: str,
    pretrained_path: str,
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-5,
    checkpoint_dir: str = CHECKPOINT_DIR,
    smoke_test: bool = False,
    visualize: bool = True,
) -> str:
    """Fine-tune pretrained model on a real dataset.
    
    Automatically uses CCM for datasets with >7 variates.
    
    Returns path to saved checkpoint.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Determine if we need CCM
    needs_ccm = dataset_name in CCM_DATASETS
    n_pretrained_variates = N_CLUSTERS
    
    logger.info("=" * 60)
    logger.info(f"FINE-TUNING ON {dataset_name.upper()}")
    if needs_ccm:
        logger.info(f"  Using CCM (clustering to {N_CLUSTERS} channels)")
    logger.info("=" * 60)
    
    # Create dataloaders
    lookback = 64 if smoke_test else LOOKBACK_LENGTH
    forecast = 16 if smoke_test else FORECAST_LENGTH
    max_samples = 2 if smoke_test else None
    
    train_loader, val_loader, n_variates = create_dataloaders(
        dataset_name,
        batch_size=batch_size,
        lookback=lookback,
        forecast=forecast,
        max_samples=max_samples,
        use_multivariate=True,
    )
    
    # Load pretrained config
    if os.path.exists(pretrained_path):
        ckpt = torch.load(pretrained_path, map_location='cpu')
        pretrained_config_dict = ckpt.get('config', {})
        logger.info(f"Loaded pretrained config from {pretrained_path}")
    else:
        logger.warning(f"No pretrained checkpoint at {pretrained_path}, starting fresh")
        pretrained_config_dict = {}
    
    # Create config (use pretrained settings for model architecture)
    config = create_diffusion_config(
        n_variates=n_pretrained_variates if needs_ccm else n_variates,
        lookback=lookback,
        forecast=forecast,
        smoke_test=smoke_test,
    )
    
    # Create model
    model = DiffusionTSF(config).to(device)
    
    # Load pretrained weights if available
    if os.path.exists(pretrained_path):
        state_dict = ckpt.get('model_state_dict', ckpt)
        try:
            model.load_state_dict(state_dict, strict=False)
            logger.info("Loaded pretrained weights")
        except Exception as e:
            logger.warning(f"Could not load pretrained weights: {e}")
    
    # Create CCM adapter if needed
    ccm_adapter = None
    if needs_ccm:
        ccm_adapter = CCMAdapter(
            n_original_vars=n_variates,
            n_clusters=N_CLUSTERS,
            seq_len=lookback,
        ).to(device)
        logger.info(f"Created CCM adapter: {n_variates} -> {N_CLUSTERS} channels")
    
    # Optimizer
    params = list(model.parameters())
    if ccm_adapter:
        params += list(ccm_adapter.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr)
    
    logger.info(f"Total params: {sum(p.numel() for p in params):,}")
    
    # Training loop
    early_stopping = EarlyStopping(patience=5 if not smoke_test else 2)
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Train
        model.train()
        if ccm_adapter:
            ccm_adapter.train()
        
        total_loss = 0
        total_cluster_loss = 0
        n_batches = 0
        
        for batch_idx, (past, future) in enumerate(train_loader):
            past = past.to(device)
            future = future.to(device)
            
            # Apply CCM if needed
            if ccm_adapter:
                past_clustered = ccm_adapter.aggregate(past, compute_prob=True)
                future_clustered = ccm_adapter.aggregate(future, compute_prob=False)  # Reuse probs from past
            else:
                past_clustered = past
                future_clustered = future
            
            optimizer.zero_grad()
            outputs = model(past_clustered, future_clustered)
            loss = outputs['loss']
            
            # Add cluster loss if using CCM
            if ccm_adapter:
                cluster_loss = ccm_adapter.get_cluster_loss(past)
                loss = loss + cluster_loss
                total_cluster_loss += cluster_loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_train_loss = total_loss / max(n_batches, 1)
        
        # Validate
        model.eval()
        if ccm_adapter:
            ccm_adapter.eval()
        
        val_loss = 0
        n_val = 0
        with torch.no_grad():
            for past, future in val_loader:
                past = past.to(device)
                future = future.to(device)
                
                if ccm_adapter:
                    past_clustered = ccm_adapter.aggregate(past, compute_prob=True)
                    future_clustered = ccm_adapter.aggregate(future, compute_prob=False)
                else:
                    past_clustered = past
                    future_clustered = future
                
                outputs = model(past_clustered, future_clustered)
                val_loss += outputs['loss'].item()
                n_val += 1
        
        avg_val_loss = val_loss / max(n_val, 1)
        elapsed = time.time() - epoch_start
        
        cluster_str = f" | ClusterLoss: {total_cluster_loss/max(n_batches,1):.4f}" if ccm_adapter else ""
        logger.info(f"Epoch {epoch+1}/{epochs} | Train: {avg_train_loss:.4f} | "
                   f"Val: {avg_val_loss:.4f}{cluster_str} | Time: {elapsed:.1f}s")
        
        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        
        # Early stopping
        if early_stopping(avg_val_loss):
            logger.info("Early stopping triggered")
            break
    
    # Save checkpoint
    dataset_ckpt_dir = os.path.join(checkpoint_dir, dataset_name)
    os.makedirs(dataset_ckpt_dir, exist_ok=True)
    
    ckpt_path = os.path.join(dataset_ckpt_dir, 'finetuned.pt')
    
    save_dict = {
        'model_state_dict': model.state_dict(),
        'config': asdict(config),
        'dataset': dataset_name,
        'val_loss': best_val_loss,
    }
    if ccm_adapter:
        save_dict['ccm_state_dict'] = ccm_adapter.state_dict()
        save_dict['n_original_vars'] = n_variates
    
    torch.save(save_dict, ckpt_path)
    logger.info(f"Saved checkpoint to {ckpt_path}")
    
    # Visualize clusters if using CCM
    if ccm_adapter and visualize and len(val_loader) > 0:
        viz_path = os.path.join(dataset_ckpt_dir, 'cluster_visualization.png')
        try:
            sample_batch = next(iter(val_loader))
            sample_past = sample_batch[0].to(device)
            visualize_clusters(ccm_adapter, sample_past, save_path=viz_path)
            logger.info(f"Saved cluster visualization to {viz_path}")
        except Exception as e:
            logger.warning(f"Could not create cluster visualization: {e}")
    
    return ckpt_path


# ============================================================================
# Main
# ============================================================================

def run_smoke_test():
    """Run minimal smoke test on all datasets."""
    
    logger.info("=" * 60)
    logger.info("SMOKE TEST - Minimal validation run")
    logger.info("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")
    
    # Test 1: Synthetic pretraining (1 sample, 1 epoch)
    logger.info("\n[1/3] Testing synthetic pretraining...")
    ckpt_path = pretrain_on_synthetic(
        n_samples=2,
        n_variates=7,
        epochs=1,
        batch_size=1,
        smoke_test=True,
    )
    logger.info("[OK] Synthetic pretraining works\n")
    
    # Test 2: Fine-tune on 7-variate dataset
    logger.info("[2/3] Testing 7-variate fine-tuning (ETTh1)...")
    finetune_on_dataset(
        dataset_name='ETTh1',
        pretrained_path=ckpt_path,
        epochs=1,
        batch_size=1,
        smoke_test=True,
        visualize=False,
    )
    logger.info("[OK] 7-variate fine-tuning works\n")
    
    # Test 3: Fine-tune with CCM (>7 variates)
    logger.info("[3/3] Testing CCM fine-tuning (weather - 21 variates)...")
    finetune_on_dataset(
        dataset_name='weather',
        pretrained_path=ckpt_path,
        epochs=1,
        batch_size=1,
        smoke_test=True,
        visualize=True,  # Test visualization
    )
    logger.info("[OK] CCM fine-tuning works\n")
    
    logger.info("=" * 60)
    logger.info("ALL SMOKE TESTS PASSED!")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Universal Training Script v2')
    
    parser.add_argument('--mode', type=str, choices=['pretrain', 'finetune', 'all'],
                        default='finetune', help='Training mode')
    parser.add_argument('--dataset', type=str, default='ETTh1',
                        choices=list(DATASET_INFO.keys()),
                        help='Dataset for fine-tuning')
    parser.add_argument('--synthetic-samples', type=int, default=100000,
                        help='Number of synthetic samples for pretraining')
    parser.add_argument('--pretrain-epochs', type=int, default=10,
                        help='Epochs for pretraining')
    parser.add_argument('--finetune-epochs', type=int, default=20,
                        help='Epochs for fine-tuning')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--smoke-test', action='store_true',
                        help='Run minimal smoke test')
    parser.add_argument('--checkpoint-dir', type=str, default=CHECKPOINT_DIR,
                        help='Directory for checkpoints')
    parser.add_argument('--pretrained-path', type=str, default=None,
                        help='Path to pretrained checkpoint (for finetune mode)')
    
    args = parser.parse_args()
    
    # Smoke test
    if args.smoke_test:
        run_smoke_test()
        return
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Determine pretrained path
    pretrained_path = args.pretrained_path
    if pretrained_path is None:
        pretrained_path = os.path.join(args.checkpoint_dir, 'pretrained_diffusion.pt')
    
    # Pretrain
    if args.mode in ['pretrain', 'all']:
        pretrain_on_synthetic(
            n_samples=args.synthetic_samples,
            epochs=args.pretrain_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            checkpoint_dir=args.checkpoint_dir,
        )
    
    # Fine-tune
    if args.mode in ['finetune', 'all']:
        if args.mode == 'all':
            # Fine-tune on all datasets
            for dataset in DATASET_INFO.keys():
                try:
                    finetune_on_dataset(
                        dataset_name=dataset,
                        pretrained_path=pretrained_path,
                        epochs=args.finetune_epochs,
                        batch_size=args.batch_size,
                        lr=args.lr * 0.1,  # Lower LR for fine-tuning
                        checkpoint_dir=args.checkpoint_dir,
                    )
                except Exception as e:
                    logger.error(f"Failed on {dataset}: {e}")
        else:
            # Fine-tune on specific dataset
            finetune_on_dataset(
                dataset_name=args.dataset,
                pretrained_path=pretrained_path,
                epochs=args.finetune_epochs,
                batch_size=args.batch_size,
                lr=args.lr * 0.1,
                checkpoint_dir=args.checkpoint_dir,
            )


if __name__ == '__main__':
    main()

