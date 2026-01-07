"""
Test script for ShapeAugmentediTransformer on ETTh1 dataset.

This script:
1. Runs 3 quick Optuna trials to tune key hyperparameters
2. Compares vanilla iTransformer vs iTransformer+CNN
3. Plots predictions from both models
4. Saves best model checkpoints for reuse
5. Optionally visualizes smudged PDF maps to verify 2D encoding

Usage:
    python test_etth1_comparison.py                 # Run with defaults (train both)
    python test_etth1_comparison.py --n-trials 5   # More trials
    python test_etth1_comparison.py --quick        # Very quick test (fewer epochs)
    python test_etth1_comparison.py --only-vanilla # Only train/search vanilla iTransformer
    python test_etth1_comparison.py --only-augmented # Only train/search augmented model
    python test_etth1_comparison.py --eval-only    # Only evaluate existing checkpoints
    python test_etth1_comparison.py --force-retrain # Force retrain even if checkpoints exist
"""

import os
import sys
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.insert(0, script_dir)
sys.path.insert(0, project_root)

# Import our modules
from shape_augmented_itransformer import (
    ShapeAugmentedConfig,
    ShapeAugmentediTransformer,
    VanillaiTransformer,
    count_parameters,
)

# Import preprocessing from diffusion_tsf
diffusion_path = os.path.join(project_root, 'models', 'diffusion_tsf')
sys.path.insert(0, diffusion_path)
from preprocessing import TimeSeriesTo2D, VerticalGaussianBlur, Standardizer

# ============================================================================
# Configuration
# ============================================================================

DATASETS_DIR = os.path.join(project_root, 'datasets')
ETTH1_PATH = os.path.join(DATASETS_DIR, 'ETT-small', 'ETTh1.csv')
RESULTS_DIR = os.path.join(script_dir, 'results')
CHECKPOINTS_DIR = os.path.join(script_dir, 'checkpoints')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

# Checkpoint file paths
VANILLA_CHECKPOINT_PATH = os.path.join(CHECKPOINTS_DIR, 'vanilla_itransformer_best.pt')
AUGMENTED_CHECKPOINT_PATH = os.path.join(CHECKPOINTS_DIR, 'augmented_itransformer_cnn_best.pt')
VANILLA_PARAMS_PATH = os.path.join(CHECKPOINTS_DIR, 'vanilla_best_params.json')
AUGMENTED_PARAMS_PATH = os.path.join(CHECKPOINTS_DIR, 'augmented_best_params.json')

# Fixed parameters for fair comparison
LOOKBACK_LENGTH = 96   # Shorter for faster testing
FORECAST_LENGTH = 24    # Predict 24 hours
IMAGE_HEIGHT = 64       # 2D representation height
BLUR_KERNEL = 15        # Gaussian blur kernel
BLUR_SIGMA = 2.0        # Blur sigma
MAX_SCALE = 3.5         # Standardization max scale

# Data split: 70% train, 10% val, 20% test (chronological)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2

# ============================================================================
# Dataset
# ============================================================================

class ETTh1Dataset(Dataset):
    """ETTh1 dataset with 2D smudged representation generation.
    
    CRITICAL: Uses CHRONOLOGICAL splits to avoid data leakage.
    Time series data must NOT use random splits due to sliding window overlap.
    """
    
    def __init__(
        self,
        data_path: str,
        lookback: int,
        forecast: int,
        split: str = 'train',  # 'train', 'val', or 'test'
        image_height: int = 64,
        max_scale: float = 3.5,
        blur_kernel: int = 15,
        blur_sigma: float = 2.0,
        stride: int = 1,
        transform_ts: bool = True,  # Whether to apply augmentation
    ):
        """
        Args:
            data_path: Path to ETTh1.csv
            lookback: Lookback window length
            forecast: Forecast horizon length
            split: 'train', 'val', or 'test'
            image_height: Height of 2D representation
            max_scale: Max scale for standardization
            blur_kernel: Gaussian blur kernel size
            blur_sigma: Gaussian blur sigma
            stride: Stride for sliding window
            transform_ts: Whether to apply data augmentation (only for train)
        """
        logger.info(f"Loading ETTh1 dataset from {data_path}")
        df = pd.read_csv(data_path)
        
        # Get all numeric columns (7 variates: HUFL, HULL, MUFL, MULL, LUFL, LULL, OT)
        self.columns = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
        # Verify columns exist
        missing = [c for c in self.columns if c not in df.columns]
        if missing:
            # Fallback to all numeric columns
            self.columns = df.select_dtypes(include=[np.number]).columns.tolist()
            logger.warning(f"Some columns missing, using: {self.columns}")
        
        self.data = torch.tensor(df[self.columns].values, dtype=torch.float32)  # (T, N)
        self.num_variates = len(self.columns)
        total_len = len(self.data)
        
        self.lookback = lookback
        self.forecast = forecast
        self.window_size = lookback + forecast
        self.image_height = image_height
        self.transform_ts = transform_ts and (split == 'train')
        
        # ===================================================================
        # CHRONOLOGICAL SPLIT - Critical for time series to avoid leakage
        # ===================================================================
        # With sliding windows, adjacent samples share most of their data.
        # Random splits would cause train/val/test to overlap temporally.
        # We use chronological splits with GAPS to prevent any overlap.
        
        # Calculate gap size (in samples) to prevent window overlap
        gap_samples = (self.window_size + stride - 1) // stride
        
        # Calculate split boundaries (in timesteps)
        train_end_ts = int(total_len * TRAIN_RATIO)
        val_end_ts = int(total_len * (TRAIN_RATIO + VAL_RATIO))
        
        # Convert to valid sample indices
        # Sample i covers timesteps [i*stride, i*stride + window_size)
        max_start_ts = total_len - self.window_size
        
        if split == 'train':
            # Train: timesteps [0, train_end_ts)
            start_idx = 0
            end_idx = max(0, (train_end_ts - self.window_size) // stride)
        elif split == 'val':
            # Val: timesteps [train_end_ts + gap, val_end_ts)
            # Add gap to ensure no overlap with train windows
            start_ts = train_end_ts + gap_samples * stride
            start_idx = start_ts // stride
            end_idx = max(start_idx, (val_end_ts - self.window_size) // stride)
        else:  # test
            # Test: timesteps [val_end_ts + gap, end)
            start_ts = val_end_ts + gap_samples * stride
            start_idx = start_ts // stride
            end_idx = max(start_idx, max_start_ts // stride)
        
        # Ensure valid ranges (prevent negative indices)
        start_idx = max(0, start_idx)
        end_idx = max(start_idx, min(max_start_ts // stride, end_idx))
        
        self.indices = list(range(start_idx, end_idx + 1, max(1, stride)))
        
        # Validate that we have samples
        if len(self.indices) == 0:
            raise ValueError(
                f"No valid samples for {split} split! "
                f"Total timesteps: {total_len}, window_size: {self.window_size}, "
                f"start_idx: {start_idx}, end_idx: {end_idx}. "
                f"Consider using a smaller lookback/forecast or different split ratios."
            )
        
        # 2D encoding modules
        self.to_2d = TimeSeriesTo2D(
            height=image_height,
            max_scale=max_scale,
            representation_mode='pdf'  # One-hot stripe representation
        )
        self.blur = VerticalGaussianBlur(kernel_size=blur_kernel, sigma=blur_sigma)
        
        logger.info(f"{split.upper()} split: {len(self.indices)} samples")
        logger.info(f"  Index range: [{start_idx}, {end_idx}]")
        logger.info(f"  Timestep range: [{start_idx * stride}, {end_idx * stride + self.window_size}]")
    
    def __len__(self):
        return len(self.indices)
    
    def _normalize_window(self, window: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Z-score normalize a window. Returns normalized, mean, std."""
        # window: (window_size, num_variates)
        mean = window.mean(dim=0, keepdim=True)  # (1, N)
        std = window.std(dim=0, keepdim=True) + 1e-8  # (1, N)
        normalized = (window - mean) / std
        return normalized, mean.squeeze(0), std.squeeze(0)
    
    def _denormalize(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Reverse z-score normalization."""
        return x * std + mean
    
    def _create_smudged_image(self, x_normalized: torch.Tensor) -> torch.Tensor:
        """Convert normalized time series to smudged 2D representation.
        
        Args:
            x_normalized: (seq_len, num_variates) normalized time series
        
        Returns:
            (num_variates, image_height, seq_len) smudged image
        """
        # Transpose to (num_variates, seq_len) for TimeSeriesTo2D
        x_transposed = x_normalized.T  # (N, L)
        
        # Convert to 2D: (N, L) -> (1, N, H, L) -> squeeze -> (N, H, L)
        # Note: TimeSeriesTo2D expects (batch, num_vars, seq_len) or (batch, seq_len)
        with torch.no_grad():  # Ensure no gradients leak during preprocessing
            image_2d = self.to_2d(x_transposed.unsqueeze(0))  # (1, N, H, L)
            
            # Apply vertical blur: (1, N, H, L) -> (1, N, H, L)
            blurred = self.blur(image_2d)
        
        return blurred.squeeze(0)  # (N, H, L)
    
    def __getitem__(self, idx):
        """
        Returns:
            past_ts: (lookback, num_variates) - raw time series for iTransformer
            past_img: (num_variates, image_height, lookback) - smudged image for CNN
            future_ts: (forecast, num_variates) - ground truth
            norm_stats: dict with mean, std for denormalization
        
        CRITICAL: Normalization uses ONLY past data to avoid data leakage!
        The smudged image is created from past-normalized data, consistent with
        what the model sees (RevIN normalizes on past only).
        """
        start = self.indices[idx]
        window = self.data[start:start + self.window_size]  # (window_size, N)
        
        # Split into past and future FIRST (before any normalization)
        past_ts = window[:self.lookback]      # (lookback, N) - raw values
        future_ts = window[self.lookback:]    # (forecast, N) - raw values
        
        # CRITICAL: Normalize using ONLY past data to match model's RevIN behavior
        # This prevents data leakage from future values into normalization stats
        past_norm, mean, std = self._normalize_window(past_ts)  # Only past!
        
        # Create smudged image from past-only normalized data
        # This ensures the 2D representation uses the same normalization as the model
        past_img = self._create_smudged_image(past_norm)  # (N, H, lookback)
        
        return {
            'past_ts': past_ts,      # Raw past for model (model applies RevIN internally)
            'past_img': past_img,    # 2D representation from past-only normalized data
            'future_ts': future_ts,  # Raw future for loss computation
            'mean': mean,            # Normalization stats (past-only)
            'std': std,
        }


def get_dataloaders(
    batch_size: int,
    lookback: int = LOOKBACK_LENGTH,
    forecast: int = FORECAST_LENGTH,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """Create train, val, test dataloaders.
    
    Returns:
        train_loader, val_loader, test_loader, num_variates
    """
    train_ds = ETTh1Dataset(ETTH1_PATH, lookback, forecast, split='train')
    val_ds = ETTh1Dataset(ETTH1_PATH, lookback, forecast, split='val')
    test_ds = ETTh1Dataset(ETTH1_PATH, lookback, forecast, split='test')
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader, train_ds.num_variates


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    use_img: bool = True,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for batch in train_loader:
        past_ts = batch['past_ts'].to(device)
        future_ts = batch['future_ts'].to(device)
        
        optimizer.zero_grad()
        
        if use_img:
            past_img = batch['past_img'].to(device)
            pred = model(past_ts, past_img)
        else:
            pred = model(past_ts)
        
        loss = criterion(pred, future_ts)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str,
    use_img: bool = True,
) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    n_samples = 0
    
    for batch in val_loader:
        past_ts = batch['past_ts'].to(device)
        future_ts = batch['future_ts'].to(device)
        
        if use_img:
            past_img = batch['past_img'].to(device)
            pred = model(past_ts, past_img)
        else:
            pred = model(past_ts)
        
        loss = criterion(pred, future_ts)
        total_loss += loss.item()
        
        # Compute MSE and MAE
        mse = ((pred - future_ts) ** 2).mean().item()
        mae = (pred - future_ts).abs().mean().item()
        
        total_mse += mse * past_ts.shape[0]
        total_mae += mae * past_ts.shape[0]
        n_samples += past_ts.shape[0]
    
    return {
        'loss': total_loss / len(val_loader),
        'mse': total_mse / n_samples,
        'mae': total_mae / n_samples,
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: str,
    use_img: bool = True,
    max_epochs: int = 75,
    patience: int = 25,
) -> Tuple[nn.Module, Dict]:
    """Full training loop with early stopping."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=config['learning_rate'] * 0.1
    )
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_mse': [], 'val_mae': []}
    
    for epoch in range(max_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, use_img)
        val_metrics = evaluate(model, val_loader, criterion, device, use_img)
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_mse'].append(val_metrics['mse'])
        history['val_mae'].append(val_metrics['mae'])
        
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 5 == 0 or patience_counter == 0:
            logger.info(f"Epoch {epoch:3d} | Train: {train_loss:.4f} | "
                       f"Val: {val_metrics['loss']:.4f} | MSE: {val_metrics['mse']:.4f}")
        
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model, history


# ============================================================================
# Checkpoint Save/Load Functions
# ============================================================================

def save_checkpoint(
    model: nn.Module,
    config: dict,
    params: dict,
    metrics: Dict[str, float],
    path: str,
    params_path: str,
):
    """Save model checkpoint and best parameters."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'params': params,
        'metrics': metrics,
    }
    torch.save(checkpoint, path)
    logger.info(f"Saved checkpoint to {path}")
    
    # Also save params as JSON for easy viewing
    with open(params_path, 'w') as f:
        json.dump({
            'params': params,
            'metrics': metrics,
            'config': config,
        }, f, indent=2)
    logger.info(f"Saved params to {params_path}")


def load_checkpoint(
    path: str,
    model_class,
    device: str,
) -> Tuple[nn.Module, dict, dict, Dict[str, float]]:
    """Load model from checkpoint.
    
    Returns:
        model, config, params, metrics
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint['config']
    params = checkpoint['params']
    metrics = checkpoint.get('metrics', {})
    
    # Reconstruct model config
    model_config = ShapeAugmentedConfig(**config)
    
    # Create model and load weights
    model = model_class(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    logger.info(f"Loaded checkpoint from {path}")
    logger.info(f"  Metrics: {metrics}")
    
    return model, config, params, metrics


def checkpoint_exists(path: str) -> bool:
    """Check if a checkpoint file exists."""
    return os.path.exists(path)


def load_params(params_path: str) -> Optional[dict]:
    """Load saved parameters from JSON."""
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            data = json.load(f)
        return data.get('params', data)
    return None


# ============================================================================
# Optuna Hyperparameter Search
# ============================================================================

def run_optuna_search_vanilla(
    n_trials: int = 3,
    max_epochs: int = 30,
    device: str = 'cuda',
    quick: bool = False,
    train_loader: DataLoader = None,
    val_loader: DataLoader = None,
    num_variates: int = 7,
) -> dict:
    """Run Optuna hyperparameter search for vanilla iTransformer only."""
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    
    if quick:
        max_epochs = 10
    
    def objective_vanilla(trial):
        config = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True),
            'd_model': trial.suggest_categorical('d_model', [64, 128, 256]),
            'n_heads': trial.suggest_categorical('n_heads', [4, 8]),
            'e_layers': trial.suggest_int('e_layers', 1, 3),
            'd_ff': trial.suggest_categorical('d_ff', [128, 256, 512]),
            'dropout': trial.suggest_float('dropout', 0.05, 0.3),
        }
        
        model_config = ShapeAugmentedConfig(
            seq_len=LOOKBACK_LENGTH,
            pred_len=FORECAST_LENGTH,
            num_variates=num_variates,
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            e_layers=config['e_layers'],
            d_ff=config['d_ff'],
            dropout=config['dropout'],
            image_height=IMAGE_HEIGHT,
        )
        
        model = VanillaiTransformer(model_config).to(device)
        
        try:
            model, history = train_model(
                model, train_loader, val_loader, config, device,
                use_img=False, max_epochs=max_epochs, patience=8
            )
            return min(history['val_loss'])
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                return float('inf')
            raise
    
    logger.info("=" * 60)
    logger.info("Searching hyperparameters for VANILLA iTransformer...")
    logger.info("=" * 60)
    
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=1, n_warmup_steps=5)
    )
    study.optimize(objective_vanilla, n_trials=n_trials, show_progress_bar=True)
    
    logger.info(f"Best vanilla val_loss: {study.best_value:.4f}")
    logger.info(f"Best vanilla params: {study.best_params}")
    
    return study.best_params


def run_optuna_search_augmented(
    n_trials: int = 3,
    max_epochs: int = 30,
    device: str = 'cuda',
    quick: bool = False,
    train_loader: DataLoader = None,
    val_loader: DataLoader = None,
    num_variates: int = 7,
) -> dict:
    """Run Optuna hyperparameter search for augmented iTransformer+CNN only."""
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    
    if quick:
        max_epochs = 10
    
    def objective_augmented(trial):
        config = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True),
            'd_model': trial.suggest_categorical('d_model', [64, 128, 256]),
            'n_heads': trial.suggest_categorical('n_heads', [4, 8]),
            'e_layers': trial.suggest_int('e_layers', 1, 3),
            'd_ff': trial.suggest_categorical('d_ff', [128, 256, 512]),
            'dropout': trial.suggest_float('dropout', 0.05, 0.3),
            'cnn_depth': trial.suggest_int('cnn_depth', 2, 4),
            'fusion_mode': trial.suggest_categorical('fusion_mode', ['add', 'concat']),
        }
        
        cnn_channels = [32 * (2 ** i) for i in range(config['cnn_depth'])]
        cnn_kernel_sizes = [(3, 3)] * config['cnn_depth']
        
        model_config = ShapeAugmentedConfig(
            seq_len=LOOKBACK_LENGTH,
            pred_len=FORECAST_LENGTH,
            num_variates=num_variates,
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            e_layers=config['e_layers'],
            d_ff=config['d_ff'],
            dropout=config['dropout'],
            image_height=IMAGE_HEIGHT,
            cnn_channels=cnn_channels,
            cnn_kernel_sizes=cnn_kernel_sizes,
            fusion_mode=config['fusion_mode'],
        )
        
        model = ShapeAugmentediTransformer(model_config).to(device)
        
        try:
            model, history = train_model(
                model, train_loader, val_loader, config, device,
                use_img=True, max_epochs=max_epochs, patience=8
            )
            return min(history['val_loss'])
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                return float('inf')
            raise
    
    logger.info("=" * 60)
    logger.info("Searching hyperparameters for AUGMENTED iTransformer+CNN...")
    logger.info("=" * 60)
    
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=1, n_warmup_steps=5)
    )
    study.optimize(objective_augmented, n_trials=n_trials, show_progress_bar=True)
    
    logger.info(f"Best augmented val_loss: {study.best_value:.4f}")
    logger.info(f"Best augmented params: {study.best_params}")
    
    return study.best_params


def run_optuna_search(
    n_trials: int = 3,
    max_epochs: int = 30,
    device: str = 'cuda',
    quick: bool = False,
) -> Tuple[dict, dict]:
    """
    Run Optuna hyperparameter search for both vanilla and augmented models.
    
    Returns:
        best_params_vanilla, best_params_augmented
    """
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    
    # Use shorter epochs for quick mode
    if quick:
        max_epochs = 10
        logger.info("Quick mode: using 10 epochs per trial")
    
    # Get data
    train_loader, val_loader, test_loader, num_variates = get_dataloaders(
        batch_size=32, lookback=LOOKBACK_LENGTH, forecast=FORECAST_LENGTH
    )
    
    # Define objective for vanilla iTransformer
    def objective_vanilla(trial):
        config = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True),
            'd_model': trial.suggest_categorical('d_model', [64, 128, 256]),
            'n_heads': trial.suggest_categorical('n_heads', [4, 8]),
            'e_layers': trial.suggest_int('e_layers', 1, 3),
            'd_ff': trial.suggest_categorical('d_ff', [128, 256, 512]),
            'dropout': trial.suggest_float('dropout', 0.05, 0.3),
        }
        
        model_config = ShapeAugmentedConfig(
            seq_len=LOOKBACK_LENGTH,
            pred_len=FORECAST_LENGTH,
            num_variates=num_variates,
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            e_layers=config['e_layers'],
            d_ff=config['d_ff'],
            dropout=config['dropout'],
            image_height=IMAGE_HEIGHT,
        )
        
        model = VanillaiTransformer(model_config).to(device)
        
        try:
            model, history = train_model(
                model, train_loader, val_loader, config, device,
                use_img=False, max_epochs=max_epochs, patience=8
            )
            return min(history['val_loss'])
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                return float('inf')
            raise
    
    # Define objective for augmented model
    def objective_augmented(trial):
        config = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True),
            'd_model': trial.suggest_categorical('d_model', [64, 128, 256]),
            'n_heads': trial.suggest_categorical('n_heads', [4, 8]),
            'e_layers': trial.suggest_int('e_layers', 1, 3),
            'd_ff': trial.suggest_categorical('d_ff', [128, 256, 512]),
            'dropout': trial.suggest_float('dropout', 0.05, 0.3),
            'cnn_depth': trial.suggest_int('cnn_depth', 2, 4),
            'fusion_mode': trial.suggest_categorical('fusion_mode', ['add', 'concat']),
        }
        
        # Build CNN channel list based on depth
        cnn_channels = [32 * (2 ** i) for i in range(config['cnn_depth'])]
        cnn_kernel_sizes = [(3, 3)] * config['cnn_depth']
        
        model_config = ShapeAugmentedConfig(
            seq_len=LOOKBACK_LENGTH,
            pred_len=FORECAST_LENGTH,
            num_variates=num_variates,
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            e_layers=config['e_layers'],
            d_ff=config['d_ff'],
            dropout=config['dropout'],
            image_height=IMAGE_HEIGHT,
            cnn_channels=cnn_channels,
            cnn_kernel_sizes=cnn_kernel_sizes,
            fusion_mode=config['fusion_mode'],
        )
        
        model = ShapeAugmentediTransformer(model_config).to(device)
        
        try:
            model, history = train_model(
                model, train_loader, val_loader, config, device,
                use_img=True, max_epochs=max_epochs, patience=8
            )
            return min(history['val_loss'])
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                return float('inf')
            raise
    
    # Run search for vanilla model
    logger.info("=" * 60)
    logger.info("Searching hyperparameters for VANILLA iTransformer...")
    logger.info("=" * 60)
    
    study_vanilla = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=1, n_warmup_steps=5)
    )
    study_vanilla.optimize(objective_vanilla, n_trials=n_trials, show_progress_bar=True)
    
    logger.info(f"Best vanilla val_loss: {study_vanilla.best_value:.4f}")
    logger.info(f"Best vanilla params: {study_vanilla.best_params}")
    
    # Run search for augmented model
    logger.info("=" * 60)
    logger.info("Searching hyperparameters for AUGMENTED iTransformer+CNN...")
    logger.info("=" * 60)
    
    study_augmented = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=1, n_warmup_steps=5)
    )
    study_augmented.optimize(objective_augmented, n_trials=n_trials, show_progress_bar=True)
    
    logger.info(f"Best augmented val_loss: {study_augmented.best_value:.4f}")
    logger.info(f"Best augmented params: {study_augmented.best_params}")
    
    return study_vanilla.best_params, study_augmented.best_params


# ============================================================================
# Evaluation and Plotting
# ============================================================================

@torch.no_grad()
def get_predictions(
    model: nn.Module,
    test_loader: DataLoader,
    device: str,
    use_img: bool = True,
    max_samples: int = 5,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Get predictions from model for plotting."""
    model.eval()
    
    pasts = []
    preds = []
    targets = []
    
    for i, batch in enumerate(test_loader):
        if i >= max_samples:
            break
        
        past_ts = batch['past_ts'].to(device)
        future_ts = batch['future_ts']
        
        if use_img:
            past_img = batch['past_img'].to(device)
            pred = model(past_ts, past_img)
        else:
            pred = model(past_ts)
        
        pasts.append(past_ts.cpu().numpy()[0])
        preds.append(pred.cpu().numpy()[0])
        targets.append(future_ts.numpy()[0])
    
    return pasts, preds, targets


def plot_comparison(
    pasts_vanilla: List[np.ndarray],
    preds_vanilla: List[np.ndarray],
    targets_vanilla: List[np.ndarray],
    pasts_augmented: List[np.ndarray],
    preds_augmented: List[np.ndarray],
    targets_augmented: List[np.ndarray],
    save_path: str,
    num_samples: int = 3,
    variate_idx: int = -1,  # OT column by default
):
    """Plot comparison of vanilla vs augmented predictions."""
    fig, axes = plt.subplots(num_samples, 2, figsize=(16, 4 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(min(num_samples, len(pasts_vanilla))):
        # Get data for one variate (OT by default, which is the last column)
        past_v = pasts_vanilla[i][:, variate_idx]
        pred_v = preds_vanilla[i][:, variate_idx]
        target_v = targets_vanilla[i][:, variate_idx]
        
        past_a = pasts_augmented[i][:, variate_idx]
        pred_a = preds_augmented[i][:, variate_idx]
        target_a = targets_augmented[i][:, variate_idx]
        
        lookback = len(past_v)
        forecast = len(target_v)
        
        # Create time indices
        time_past = np.arange(lookback)
        time_future = np.arange(lookback, lookback + forecast)
        
        # Plot vanilla
        ax = axes[i, 0]
        ax.plot(time_past, past_v, 'b-', label='Past', linewidth=1.5)
        ax.plot(time_future, target_v, 'g-', label='Ground Truth', linewidth=2)
        ax.plot(time_future, pred_v, 'r--', label='Prediction', linewidth=2)
        ax.axvline(x=lookback, color='gray', linestyle=':', alpha=0.5)
        ax.set_title(f'Sample {i+1} - Vanilla iTransformer')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value (OT)')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Compute MSE for this sample
        mse_v = ((pred_v - target_v) ** 2).mean()
        ax.text(0.98, 0.02, f'MSE: {mse_v:.4f}', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Plot augmented
        ax = axes[i, 1]
        ax.plot(time_past, past_a, 'b-', label='Past', linewidth=1.5)
        ax.plot(time_future, target_a, 'g-', label='Ground Truth', linewidth=2)
        ax.plot(time_future, pred_a, 'r--', label='Prediction', linewidth=2)
        ax.axvline(x=lookback, color='gray', linestyle=':', alpha=0.5)
        ax.set_title(f'Sample {i+1} - Augmented iTransformer+CNN')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value (OT)')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Compute MSE for this sample
        mse_a = ((pred_a - target_a) ** 2).mean()
        ax.text(0.98, 0.02, f'MSE: {mse_a:.4f}', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved comparison plot to {save_path}")


def visualize_smudged_samples(
    loader: DataLoader,
    save_dir: str,
    max_samples: int = 2,
    max_variates: int = 3,
) -> None:
    """Save a few smudged PDF maps from the loader for quick visual checks."""
    os.makedirs(save_dir, exist_ok=True)
    saved = 0
    for batch_idx, batch in enumerate(loader):
        imgs = batch['past_img']  # (B, N, H, L)
        # Ensure CPU numpy for plotting
        imgs_np = imgs.detach().cpu().numpy()
        B, N, H, L = imgs_np.shape
        for b in range(B):
            if saved >= max_samples:
                return
            for v in range(min(max_variates, N)):
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.imshow(imgs_np[b, v], aspect='auto', origin='lower', cmap='magma')
                ax.set_title(f"Sample {saved+1} | Variate {v} | Shape: {H}x{L}")
                ax.set_xlabel("Time (lookback steps)")
                ax.set_ylabel("Value bins (PDF height)")
                ax.grid(False)
                fname = os.path.join(save_dir, f"sample{saved+1}_var{v}.png")
                plt.tight_layout()
                plt.savefig(fname, dpi=150, bbox_inches='tight')
                plt.close(fig)
            saved += 1
        if saved >= max_samples:
            return


def compute_test_metrics(
    model: nn.Module,
    test_loader: DataLoader,
    device: str,
    use_img: bool,
) -> Dict[str, float]:
    """Compute test set metrics."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            past_ts = batch['past_ts'].to(device)
            future_ts = batch['future_ts']
            
            if use_img:
                past_img = batch['past_img'].to(device)
                pred = model(past_ts, past_img)
            else:
                pred = model(past_ts)
            
            all_preds.append(pred.cpu())
            all_targets.append(future_ts)
    
    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    mse = ((preds - targets) ** 2).mean().item()
    mae = (preds - targets).abs().mean().item()
    
    return {'mse': mse, 'mae': mae, 'rmse': np.sqrt(mse)}


# ============================================================================
# Main
# ============================================================================

def train_and_save_vanilla(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    num_variates: int,
    device: str,
    n_trials: int,
    max_epochs: int,
    final_epochs: int,
    quick: bool,
    force_retrain: bool,
) -> Tuple[nn.Module, dict, Dict[str, float]]:
    """Train vanilla iTransformer and save checkpoint."""
    
    # Check if we can load existing checkpoint
    if checkpoint_exists(VANILLA_CHECKPOINT_PATH) and not force_retrain:
        logger.info("Found existing vanilla iTransformer checkpoint!")
        saved_params = load_params(VANILLA_PARAMS_PATH)
        if saved_params:
            logger.info(f"Using saved params: {saved_params}")
            model, config, params, metrics = load_checkpoint(
                VANILLA_CHECKPOINT_PATH, VanillaiTransformer, device
            )
            return model, params, metrics
    
    # Run Optuna search
    best_params = run_optuna_search_vanilla(
        n_trials=n_trials,
        max_epochs=max_epochs,
        device=device,
        quick=quick,
        train_loader=train_loader,
        val_loader=val_loader,
        num_variates=num_variates,
    )
    
    # Train final model with best params
    logger.info("=" * 60)
    logger.info("Training FINAL vanilla iTransformer with best hyperparameters...")
    logger.info("=" * 60)
    
    config_dict = {
        'seq_len': LOOKBACK_LENGTH,
        'pred_len': FORECAST_LENGTH,
        'num_variates': num_variates,
        'd_model': best_params.get('d_model', 128),
        'n_heads': best_params.get('n_heads', 4),
        'e_layers': best_params.get('e_layers', 2),
        'd_ff': best_params.get('d_ff', 256),
        'dropout': best_params.get('dropout', 0.1),
        'image_height': IMAGE_HEIGHT,
    }
    
    model_config = ShapeAugmentedConfig(**config_dict)
    model = VanillaiTransformer(model_config).to(device)
    logger.info(f"Vanilla model params: {count_parameters(model):,}")
    
    model, history = train_model(
        model, train_loader, val_loader,
        {'learning_rate': best_params.get('learning_rate', 5e-4)},
        device, use_img=False, max_epochs=final_epochs, patience=15
    )
    
    # Evaluate on test set
    test_metrics = compute_test_metrics(model, test_loader, device, use_img=False)
    
    # Save checkpoint
    save_checkpoint(
        model=model,
        config=config_dict,
        params=best_params,
        metrics=test_metrics,
        path=VANILLA_CHECKPOINT_PATH,
        params_path=VANILLA_PARAMS_PATH,
    )
    
    return model, best_params, test_metrics


def train_and_save_augmented(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    num_variates: int,
    device: str,
    n_trials: int,
    max_epochs: int,
    final_epochs: int,
    quick: bool,
    force_retrain: bool,
) -> Tuple[nn.Module, dict, Dict[str, float]]:
    """Train augmented iTransformer+CNN and save checkpoint."""
    
    # Check if we can load existing checkpoint
    if checkpoint_exists(AUGMENTED_CHECKPOINT_PATH) and not force_retrain:
        logger.info("Found existing augmented iTransformer+CNN checkpoint!")
        saved_params = load_params(AUGMENTED_PARAMS_PATH)
        if saved_params:
            logger.info(f"Using saved params: {saved_params}")
            model, config, params, metrics = load_checkpoint(
                AUGMENTED_CHECKPOINT_PATH, ShapeAugmentediTransformer, device
            )
            return model, params, metrics
    
    # Run Optuna search
    best_params = run_optuna_search_augmented(
        n_trials=n_trials,
        max_epochs=max_epochs,
        device=device,
        quick=quick,
        train_loader=train_loader,
        val_loader=val_loader,
        num_variates=num_variates,
    )
    
    # Train final model with best params
    logger.info("=" * 60)
    logger.info("Training FINAL augmented iTransformer+CNN with best hyperparameters...")
    logger.info("=" * 60)
    
    cnn_depth = best_params.get('cnn_depth', 3)
    cnn_channels = [32 * (2 ** i) for i in range(cnn_depth)]
    cnn_kernel_sizes = [(3, 3)] * cnn_depth
    
    config_dict = {
        'seq_len': LOOKBACK_LENGTH,
        'pred_len': FORECAST_LENGTH,
        'num_variates': num_variates,
        'd_model': best_params.get('d_model', 128),
        'n_heads': best_params.get('n_heads', 4),
        'e_layers': best_params.get('e_layers', 2),
        'd_ff': best_params.get('d_ff', 256),
        'dropout': best_params.get('dropout', 0.1),
        'image_height': IMAGE_HEIGHT,
        'cnn_channels': cnn_channels,
        'cnn_kernel_sizes': cnn_kernel_sizes,
        'fusion_mode': best_params.get('fusion_mode', 'add'),
    }
    
    model_config = ShapeAugmentedConfig(**config_dict)
    model = ShapeAugmentediTransformer(model_config).to(device)
    logger.info(f"Augmented model params: {count_parameters(model):,}")
    
    model, history = train_model(
        model, train_loader, val_loader,
        {'learning_rate': best_params.get('learning_rate', 5e-4)},
        device, use_img=True, max_epochs=final_epochs, patience=15
    )
    
    # Evaluate on test set
    test_metrics = compute_test_metrics(model, test_loader, device, use_img=True)
    
    # Save checkpoint
    save_checkpoint(
        model=model,
        config=config_dict,
        params=best_params,
        metrics=test_metrics,
        path=AUGMENTED_CHECKPOINT_PATH,
        params_path=AUGMENTED_PARAMS_PATH,
    )
    
    return model, best_params, test_metrics


def main():
    parser = argparse.ArgumentParser(description='Test ShapeAugmentediTransformer on ETTh1')
    parser.add_argument('--n-trials', type=int, default=3, help='Number of Optuna trials')
    parser.add_argument('--max-epochs', type=int, default=50, help='Max epochs per trial')
    parser.add_argument('--quick', action='store_true', help='Quick test mode (fewer epochs)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model selection options
    parser.add_argument('--only-vanilla', action='store_true', 
                       help='Only train/search vanilla iTransformer (skip augmented)')
    parser.add_argument('--only-augmented', action='store_true',
                       help='Only train/search augmented iTransformer+CNN (skip vanilla)')
    parser.add_argument('--eval-only', action='store_true',
                       help='Only evaluate existing checkpoints (no training)')
    parser.add_argument('--force-retrain', action='store_true',
                       help='Force retrain even if checkpoints exist')
    parser.add_argument('--list-checkpoints', action='store_true',
                       help='List available checkpoints and exit')
    parser.add_argument('--viz-smudges', action='store_true',
                       help='Visualize and save a few smudged PDF maps')
    parser.add_argument('--viz-samples', type=int, default=2,
                       help='How many samples to visualize for smudges')
    parser.add_argument('--viz-variates', type=int, default=3,
                       help='How many variates per sample to visualize')
    
    args = parser.parse_args()
    
    # Handle conflicting options
    if args.only_vanilla and args.only_augmented:
        logger.error("Cannot use --only-vanilla and --only-augmented together")
        return
    
    # List checkpoints mode
    if args.list_checkpoints:
        logger.info("Available checkpoints:")
        logger.info(f"  Vanilla: {VANILLA_CHECKPOINT_PATH}")
        logger.info(f"    Exists: {checkpoint_exists(VANILLA_CHECKPOINT_PATH)}")
        if checkpoint_exists(VANILLA_CHECKPOINT_PATH):
            params = load_params(VANILLA_PARAMS_PATH)
            if params:
                logger.info(f"    Params: {params}")
        
        logger.info(f"  Augmented: {AUGMENTED_CHECKPOINT_PATH}")
        logger.info(f"    Exists: {checkpoint_exists(AUGMENTED_CHECKPOINT_PATH)}")
        if checkpoint_exists(AUGMENTED_CHECKPOINT_PATH):
            params = load_params(AUGMENTED_PARAMS_PATH)
            if params:
                logger.info(f"    Params: {params}")
        return
    
    device = args.device
    logger.info(f"Using device: {device}")
    logger.info(f"ETTh1 data path: {ETTH1_PATH}")
    logger.info(f"Checkpoints dir: {CHECKPOINTS_DIR}")
    
    # Verify data file exists
    if not os.path.exists(ETTH1_PATH):
        logger.error(f"ETTh1.csv not found at {ETTH1_PATH}")
        logger.error("Please ensure the datasets/ETT-small/ directory contains ETTh1.csv")
        return
    
    # Get data loaders
    train_loader, val_loader, test_loader, num_variates = get_dataloaders(
        batch_size=32, lookback=LOOKBACK_LENGTH, forecast=FORECAST_LENGTH
    )
    
    # Optional: visualize smudged PDF maps from train loader
    if args.viz_smudges:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        smudge_dir = os.path.join(RESULTS_DIR, f'smudges_{timestamp}')
        visualize_smudged_samples(
            train_loader,
            smudge_dir,
            max_samples=max(1, args.viz_samples),
            max_variates=max(1, args.viz_variates),
        )

    final_epochs = 30 if args.quick else 100
    
    vanilla_model = None
    augmented_model = None
    best_vanilla = {}
    best_augmented = {}
    vanilla_test = {}
    augmented_test = {}
    
    # Determine what to train
    train_vanilla = not args.only_augmented
    train_augmented = not args.only_vanilla
    
    # Eval-only mode
    if args.eval_only:
        if train_vanilla:
            if checkpoint_exists(VANILLA_CHECKPOINT_PATH):
                vanilla_model, _, best_vanilla, vanilla_test = load_checkpoint(
                    VANILLA_CHECKPOINT_PATH, VanillaiTransformer, device
                )
                # Re-evaluate to ensure fresh metrics
                vanilla_test = compute_test_metrics(vanilla_model, test_loader, device, use_img=False)
            else:
                logger.warning("Vanilla checkpoint not found, skipping evaluation")
                train_vanilla = False
        
        if train_augmented:
            if checkpoint_exists(AUGMENTED_CHECKPOINT_PATH):
                augmented_model, _, best_augmented, augmented_test = load_checkpoint(
                    AUGMENTED_CHECKPOINT_PATH, ShapeAugmentediTransformer, device
                )
                # Re-evaluate to ensure fresh metrics
                augmented_test = compute_test_metrics(augmented_model, test_loader, device, use_img=True)
            else:
                logger.warning("Augmented checkpoint not found, skipping evaluation")
                train_augmented = False
    else:
        # Training mode
        if train_vanilla:
            vanilla_model, best_vanilla, vanilla_test = train_and_save_vanilla(
                train_loader, val_loader, test_loader, num_variates, device,
                args.n_trials, args.max_epochs, final_epochs, args.quick, args.force_retrain
            )
        
        if train_augmented:
            augmented_model, best_augmented, augmented_test = train_and_save_augmented(
                train_loader, val_loader, test_loader, num_variates, device,
                args.n_trials, args.max_epochs, final_epochs, args.quick, args.force_retrain
            )
    
    # Report results
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    
    if vanilla_test:
        logger.info(f"VANILLA iTransformer TEST metrics:")
        logger.info(f"  MSE: {vanilla_test['mse']:.4f}")
        logger.info(f"  MAE: {vanilla_test['mae']:.4f}")
        logger.info(f"  RMSE: {vanilla_test['rmse']:.4f}")
    
    if augmented_test:
        logger.info(f"AUGMENTED iTransformer+CNN TEST metrics:")
        logger.info(f"  MSE: {augmented_test['mse']:.4f}")
        logger.info(f"  MAE: {augmented_test['mae']:.4f}")
        logger.info(f"  RMSE: {augmented_test['rmse']:.4f}")
    
    # Compare if both models available
    if vanilla_test and augmented_test:
        mse_improvement = (vanilla_test['mse'] - augmented_test['mse']) / vanilla_test['mse'] * 100
        logger.info(f"\nMSE improvement (augmented vs vanilla): {mse_improvement:.2f}%")
        
        # Plot comparison
        if vanilla_model and augmented_model:
            pasts_v, preds_v, targets_v = get_predictions(vanilla_model, test_loader, device, use_img=False)
            pasts_a, preds_a, targets_a = get_predictions(augmented_model, test_loader, device, use_img=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_path = os.path.join(RESULTS_DIR, f'comparison_{timestamp}.png')
            plot_comparison(pasts_v, preds_v, targets_v, pasts_a, preds_a, targets_a, plot_path)
            
            # Save results summary
            results = {
                'timestamp': timestamp,
                'best_params_vanilla': best_vanilla,
                'best_params_augmented': best_augmented,
                'test_metrics_vanilla': vanilla_test,
                'test_metrics_augmented': augmented_test,
                'mse_improvement_pct': mse_improvement,
                'config': {
                    'lookback': LOOKBACK_LENGTH,
                    'forecast': FORECAST_LENGTH,
                    'image_height': IMAGE_HEIGHT,
                    'num_variates': num_variates,
                }
            }
            
            results_path = os.path.join(RESULTS_DIR, f'results_{timestamp}.json')
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved results to {results_path}")
    
    logger.info("=" * 60)
    logger.info("DONE!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

