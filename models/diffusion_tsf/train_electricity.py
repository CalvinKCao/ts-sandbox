"""
Training script for Diffusion TSF on Electricity dataset.

Features:
- Optuna for efficient hyperparameter search (Bayesian + pruning)
- Early stopping based on validation loss
- Checkpoint saving (resume anytime)
- Focuses on most important hyperparameters only

Usage:
    python train_electricity.py                    # Start fresh (creates new study if exists)
    python train_electricity.py --resume           # Resume latest study
    python train_electricity.py --best             # Train with best found params of latest study
"""

import os
import sys
import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# Setup path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from config import DiffusionTSFConfig
from model import DiffusionTSF
from metrics import compute_metrics, log_metrics
from dataset import apply_1d_augmentations

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(script_dir, 'training.log'))
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

# Paths
DATA_PATH = os.path.join(script_dir, '..', '..', 'datasets', 'electricity', 'electricity.csv')
BASE_CHECKPOINT_DIR = os.path.join(script_dir, 'checkpoints')
CHECKPOINT_DIR = BASE_CHECKPOINT_DIR  # Will be updated by run_optuna_search or train_with_best_params
OPTUNA_DB = os.path.join(script_dir, 'optuna_study.db')

# Fixed parameters (aligned with ViTime paper)
LOOKBACK_LENGTH = 512      # Same as ViTime paper
FORECAST_LENGTH = 96       # Common benchmark (paper uses 96, 192, 336, 720)
IMAGE_HEIGHT = 128         # ViTime paper: h=128
BLUR_KERNEL = 31           # ViTime paper: kernel=31
BLUR_SIGMA = 1.0           # Sharper labels; EMD handles non-overlap
EMD_LAMBDA = 0.2           # Weight for EMD loss term
MAX_SCALE = 3.5            # ViTime paper: MS=3.5

# ============================================================================
# Hardware-Adaptive Configuration
# ============================================================================

def get_gpu_memory_gb() -> float:
    """Get available GPU memory in GB."""
    if not torch.cuda.is_available():
        return 0.0
    try:
        # Get total memory of the first GPU
        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / (1024 ** 3)
        return total_gb
    except Exception:
        return 8.0  # Assume 8GB if detection fails

def get_hardware_config():
    """Get hardware-adaptive search space based on GPU memory."""
    gpu_mem = get_gpu_memory_gb()
    
    if gpu_mem >= 40:  # Ada 6000, A100, etc.
        logger.info(f"Detected high-end GPU ({gpu_mem:.1f}GB) - using extensive search space")
        return {
            'learning_rate': (1e-5, 1e-3),  # Much wider LR range: 1e-5 to 1e-3
            'model_size': ['tiny', 'small', 'medium', 'large'],  # Include all model sizes
            'diffusion_steps': [100, 250, 500, 1000, 2000],  # More diffusion options
            'batch_size': [16, 32, 64, 128, 256],  # Much larger batch sizes
            'noise_schedule': ['linear', 'cosine'],  # Well-established noise schedules
        }
    elif gpu_mem >= 16:  # RTX 3090, 4080, etc.
        logger.info(f"Detected mid-range GPU ({gpu_mem:.1f}GB) - using medium batch sizes")
        return {
            'learning_rate': (5e-5, 5e-4),
            'model_size': ['small', 'medium'],
            'diffusion_steps': [250, 500],
            'batch_size': [16, 32],
            'noise_schedule': ['linear', 'cosine'],
        }
    elif gpu_mem >= 8:  # RTX 3070, laptop GPUs, etc.
        logger.info(f"Detected mid-low GPU ({gpu_mem:.1f}GB) - using small batch sizes")
        return {
            'learning_rate': (5e-5, 5e-4),
            'model_size': ['tiny', 'small'],
            'diffusion_steps': [250],
            'batch_size': [8, 16],
            'noise_schedule': ['cosine'],
        }
    else:  # Low-end or integrated GPU
        logger.info(f"Detected low-end GPU ({gpu_mem:.1f}GB) - using minimal settings")
        return {
            'learning_rate': (1e-4, 3e-4),
            'model_size': ['tiny'],
            'diffusion_steps': [250],
            'batch_size': [4, 8],
            'noise_schedule': ['cosine'],
        }

# Get hardware-adaptive search space
SEARCH_SPACE = None  # Will be set at runtime
# Selected model type (set from CLI)
SELECTED_MODEL_TYPE = "unet"
# Selected representation mode (stripe/pdf vs occupancy/cdf)
SELECTED_REPR_MODE = "pdf"
# Transformer patch sizes (set from CLI, default 16x16)
TRANSFORMER_PATCH_HEIGHT = 16
TRANSFORMER_PATCH_WIDTH = 16

MODEL_SIZES = {
    'tiny': [32, 64],           # ~1M params, for quick tests only
    'small': [64, 128, 256],    # ~10M params
    'medium': [64, 128, 256, 512],  # ~40M params
    'large': [128, 256, 512],   # ~80M params (close to ViTime's 93M)
}

# Training settings
MAX_EPOCHS = 200
PATIENCE = 15                   # Early stopping patience (increased for longer training)
VAL_SPLIT = 0.1
NUM_OPTUNA_TRIALS = 20          # Total trials to run
PRUNING_WARMUP = 20             # Don't prune before this epoch (increased for longer training)

# ============================================================================
# Dataset
# ============================================================================

class ElectricityDataset(Dataset):
    """Electricity dataset for time series forecasting.
    
    Uses a single variable (OT - Oil Temperature) for univariate forecasting.
    Creates sliding windows of (past, future) pairs.
    """
    
    def __init__(
        self,
        data_path: str,
        lookback: int = 512,
        forecast: int = 96,
        column: str = 'OT',
        stride: int = 24,  # Stride for sliding window (1 day = 24 hours)
        max_samples: Optional[int] = None,
        augment: bool = True,
        data_tensor: Optional[torch.Tensor] = None,
        indices: Optional[List[int]] = None
    ):
        if data_tensor is None:
            logger.info(f"Loading electricity data from {data_path}")
            df = pd.read_csv(data_path)
            
            # Use OT column (Oil Temperature) as target
            if column not in df.columns:
                # If OT not available, use first numeric column
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                column = numeric_cols[0]
                logger.warning(f"Column 'OT' not found, using '{column}' instead")
            
            self.data = torch.tensor(df[column].values, dtype=torch.float32)
        else:
            # Reuse already loaded tensor to avoid extra IO/memory
            self.data = data_tensor
        
        self.lookback = lookback
        self.forecast = forecast
        self.total_len = lookback + forecast
        self.augment = augment
        
        # Calculate number of samples
        if indices is not None:
            self.indices = indices
        else:
            num_windows = (len(self.data) - self.total_len) // stride + 1
            self.indices = [i * stride for i in range(num_windows)]
        
        if max_samples and indices is None and len(self.indices) > max_samples:
            self.indices = self.indices[:max_samples]
        
        logger.info(f"Created dataset: {len(self.indices)} samples, "
                   f"lookback={lookback}, forecast={forecast}")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        start = self.indices[idx]
        window = self.data[start:start + self.total_len]
        
        if self.augment:
            window = apply_1d_augmentations(window)
        
        past = window[:self.lookback]
        future = window[self.lookback:]
        return past, future


def get_dataloaders(
    batch_size: int,
    val_split: float = 0.1,
    max_samples: Optional[int] = None,
    lookback: int = LOOKBACK_LENGTH,
    forecast: int = FORECAST_LENGTH
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    
    # Base dataset (no augmentation) to derive indices and reuse tensor
    base_dataset = ElectricityDataset(
        DATA_PATH,
        lookback=lookback,
        forecast=forecast,
        max_samples=max_samples,
        augment=False
    )
    
    # Split into train/val
    val_size = int(len(base_dataset) * val_split)
    train_size = len(base_dataset) - val_size
    
    train_subset, val_subset = random_split(
        base_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_dataset = ElectricityDataset(
        DATA_PATH,
        lookback=lookback,
        forecast=forecast,
        max_samples=max_samples,
        augment=True,
        data_tensor=base_dataset.data,
        indices=train_subset.indices
    )
    
    val_dataset = ElectricityDataset(
        DATA_PATH,
        lookback=lookback,
        forecast=forecast,
        max_samples=max_samples,
        augment=False,
        data_tensor=base_dataset.data,
        indices=val_subset.indices
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    logger.info(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    
    return train_loader, val_loader


# ============================================================================
# Training
# ============================================================================

class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience: int = 7, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    config: dict,
    path: str
):
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'config': config,
    }, path)
    logger.info(f"Checkpoint saved: {path}")


def load_checkpoint(path: str, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None):
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    return checkpoint


def train_epoch(
    model: DiffusionTSF,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)
    
    for batch_idx, (past, future) in enumerate(train_loader):
        past = past.to(device)
        future = future.to(device)
        
        optimizer.zero_grad()
        outputs = model(past, future)
        loss = outputs['loss']
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            logger.debug(f"Epoch {epoch} [{batch_idx}/{num_batches}] Loss: {loss.item():.4f}")
    
    return total_loss / num_batches


@torch.no_grad()
def validate(
    model: DiffusionTSF,
    val_loader: DataLoader,
    device: str,
    use_generation: bool = False
) -> Dict[str, float]:
    """Validate model on validation set."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    for past, future in val_loader:
        past = past.to(device)
        future = future.to(device)
        
        # Compute diffusion loss
        outputs = model(past, future)
        total_loss += outputs['loss'].item()
        
        # Optionally generate predictions (slower but more accurate metrics)
        if use_generation:
            gen_out = model.generate(past, use_ddim=True, num_ddim_steps=20)
            all_preds.append(gen_out['prediction'].cpu())
            all_targets.append(future.cpu())
    
    metrics = {'val_loss': total_loss / len(val_loader)}
    
    if use_generation and all_preds:
        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)
        gen_metrics = compute_metrics(preds, targets)
        metrics.update({f'val_{k}': v.item() for k, v in gen_metrics.items()})
    
    return metrics


def train(
    config: dict,
    trial=None,  # Optuna trial for pruning
    max_epochs: int = MAX_EPOCHS,
    checkpoint_path: Optional[str] = None
) -> float:
    """Full training loop.
    
    Args:
        config: Hyperparameter configuration
        trial: Optuna trial for pruning (optional)
        max_epochs: Maximum epochs to train
        checkpoint_path: Path to save checkpoints
        
    Returns:
        Best validation loss
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Training on device: {device}")
    logger.info(f"Config: {config}")
    
    # Create model
    # Attention at deeper levels for larger models
    model_size = config['model_size']
    channels = MODEL_SIZES[model_size]
    if model_size == 'large':
        attention_levels = [1, 2]  # Attention at 256 and 512 channel levels
        num_res_blocks = 3
    elif model_size == 'medium':
        attention_levels = [1, 2, 3]  # Attention at 128, 256, 512
        num_res_blocks = 2
    else:  # small
        attention_levels = [1, 2]
        num_res_blocks = 2
    
    model_config = DiffusionTSFConfig(
        lookback_length=LOOKBACK_LENGTH,
        forecast_length=FORECAST_LENGTH,
        image_height=IMAGE_HEIGHT,
        max_scale=MAX_SCALE,
        blur_kernel_size=BLUR_KERNEL,
        blur_sigma=config.get('blur_sigma', BLUR_SIGMA),
        emd_lambda=config.get('emd_lambda', EMD_LAMBDA),
        representation_mode=config.get('representation_mode', SELECTED_REPR_MODE),
        unet_channels=channels,
        num_res_blocks=num_res_blocks,
        attention_levels=attention_levels,
        num_diffusion_steps=config['diffusion_steps'],
        noise_schedule=config['noise_schedule'],
        ddim_steps=50,
        model_type=config.get('model_type', 'unet'),
        transformer_patch_height=config.get('transformer_patch_height', TRANSFORMER_PATCH_HEIGHT),
        transformer_patch_width=config.get('transformer_patch_width', TRANSFORMER_PATCH_WIDTH),
        use_coordinate_channel=config.get('use_coordinate_channel', True),
    )
    
    model = DiffusionTSF(model_config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")
    
    # Create dataloaders
    train_loader, val_loader = get_dataloaders(
        batch_size=config['batch_size'],
        val_split=VAL_SPLIT
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=0.01
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_epochs,
        eta_min=config['learning_rate'] * 0.1  # Higher minimum LR for longer training
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=PATIENCE)
    
    best_val_loss = float('inf')
    start_epoch = 0
    
    # Resume from checkpoint if exists
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = load_checkpoint(checkpoint_path, model, optimizer)
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt['val_loss']
        logger.info(f"Resuming from epoch {start_epoch}, best val_loss: {best_val_loss:.4f}")
    
    # Training loop
    for epoch in range(start_epoch, max_epochs):
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        
        # Validate
        val_metrics = validate(model, val_loader, device, use_generation=(epoch % 5 == 0))
        val_loss = val_metrics['val_loss']
        
        # Update scheduler
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch {epoch}/{max_epochs} | "
                   f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                   f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                   f"Time: {epoch_time:.1f}s")
        
        if 'val_mse' in val_metrics:
            logger.info(f"  Generation metrics: {log_metrics({k: v for k, v in val_metrics.items() if k != 'val_loss'})}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if checkpoint_path:
                save_checkpoint(
                    model, optimizer, epoch, train_loss, val_loss,
                    config, checkpoint_path.replace('.pt', '_best.pt')
                )
        
        # Save regular checkpoint
        if checkpoint_path and epoch % 5 == 0:
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss,
                config, checkpoint_path
            )
        
        # Optuna pruning
        if trial is not None:
            trial.report(val_loss, epoch)
            if epoch >= PRUNING_WARMUP and trial.should_prune():
                logger.info(f"Trial pruned at epoch {epoch}")
                raise optuna.TrialPruned()
        
        # Early stopping
        if early_stopping(val_loss):
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    return best_val_loss


# ============================================================================
# Hyperparameter Search with Optuna
# ============================================================================

def objective(trial) -> float:
    """Optuna objective function."""
    
    # Sample hyperparameters
    config = {
        'learning_rate': trial.suggest_float('learning_rate', *SEARCH_SPACE['learning_rate'], log=True),
        'model_size': trial.suggest_categorical('model_size', SEARCH_SPACE['model_size']),
        'diffusion_steps': trial.suggest_categorical('diffusion_steps', SEARCH_SPACE['diffusion_steps']),
        'batch_size': trial.suggest_categorical('batch_size', SEARCH_SPACE['batch_size']),
        'noise_schedule': trial.suggest_categorical('noise_schedule', SEARCH_SPACE['noise_schedule']),
        'model_type': SELECTED_MODEL_TYPE,
        'blur_sigma': trial.suggest_categorical('blur_sigma', SEARCH_SPACE['blur_sigma']),
        'emd_lambda': trial.suggest_categorical('emd_lambda', SEARCH_SPACE['emd_lambda']),
        'representation_mode': SELECTED_REPR_MODE,
        'transformer_patch_height': TRANSFORMER_PATCH_HEIGHT,
        'transformer_patch_width': TRANSFORMER_PATCH_WIDTH,
        'use_coordinate_channel': True,  # Enable vertical spatial awareness
    }
    
    # Checkpoint for this trial
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f'trial_{trial.number}.pt')
    
    # Try training with automatic batch size reduction on OOM
    original_batch = config['batch_size']
    current_batch = original_batch
    
    while current_batch >= 2:
        try:
            config['batch_size'] = current_batch
            best_val_loss = train(config, trial=trial, checkpoint_path=checkpoint_path)
            return best_val_loss
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                new_batch = current_batch // 2
                logger.warning(f"OOM with batch_size={current_batch}, retrying with batch_size={new_batch}")
                current_batch = new_batch
            else:
                raise
    
    logger.error(f"OOM even with batch_size=2, skipping trial")
    return float('inf')


def run_optuna_search(n_trials: int = NUM_OPTUNA_TRIALS, resume: bool = True):
    """Run Optuna hyperparameter search."""
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    
    # Create or load study
    storage = f"sqlite:///{OPTUNA_DB}"
    base_study_name = "diffusion_tsf_electricity"
    study_name = base_study_name
    
    if resume:
        # Try to find the latest study in the storage
        try:
            summaries = optuna.get_all_study_summaries(storage=storage)
            if summaries:
                # Filter for studies that start with our base name
                related_studies = [s for s in summaries if s.study_name.startswith(base_study_name)]
                if related_studies:
                    # Sort by last trial time or creation time if available, 
                    # but summaries don't have creation time. We'll use the name which often has a timestamp.
                    # Actually, if we use timestamps, sorting by name works.
                    # If some don't have timestamps (like the default one), we'll handle that.
                    study_name = max(related_studies, key=lambda s: s.study_name).study_name
                    logger.info(f"Resuming latest study: {study_name}")
        except Exception as e:
            logger.warning(f"Could not list studies: {e}. Using default study name.")
    else:
        # Check if default study exists
        try:
            optuna.load_study(study_name=study_name, storage=storage)
            # If it exists and we are NOT resuming, we should create a new one
            study_name = f"{base_study_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            logger.info(f"Existing study found and resume=False. Creating new study: {study_name}")
        except KeyError:
            # Study doesn't exist, use default name
            pass
    
    # Update checkpoint directory for this study
    global CHECKPOINT_DIR
    CHECKPOINT_DIR = os.path.join(BASE_CHECKPOINT_DIR, study_name)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    logger.info(f"Study checkpoints will be saved to: {CHECKPOINT_DIR}")
    
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        storage=storage,
        load_if_exists=resume,
        sampler=TPESampler(seed=42),  # Bayesian optimization
        pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=PRUNING_WARMUP)
    )
    
    logger.info(f"Starting Optuna search: {n_trials} trials")
    logger.info(f"Previous trials: {len(study.trials)}")
    
    remaining_trials = max(0, n_trials - len(study.trials))
    
    if remaining_trials > 0:
        study.optimize(objective, n_trials=remaining_trials, show_progress_bar=True)
    
    # Log results
    logger.info("=" * 60)
    logger.info("SEARCH COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best value: {study.best_trial.value:.4f}")
    logger.info(f"Best params: {study.best_trial.params}")
    
    # Save best params
    best_params_path = os.path.join(CHECKPOINT_DIR, 'best_params.json')
    params_to_save = dict(study.best_trial.params)
    params_to_save['representation_mode'] = SELECTED_REPR_MODE
    with open(best_params_path, 'w') as f:
        json.dump(params_to_save, f, indent=2)
    logger.info(f"Best params saved to {best_params_path}")
    
    # Copy best trial checkpoint to best_model.pt
    best_trial_ckpt = os.path.join(CHECKPOINT_DIR, f"trial_{study.best_trial.number}_best.pt")
    dest_best_model = os.path.join(CHECKPOINT_DIR, 'best_model.pt')
    if os.path.exists(best_trial_ckpt):
        import shutil
        shutil.copy(best_trial_ckpt, dest_best_model)
        logger.info(f"Best trial checkpoint copied to {dest_best_model}")
    
    return study.best_trial.params


def train_with_best_params():
    """Train with the best found parameters."""
    global CHECKPOINT_DIR
    
    # If CHECKPOINT_DIR is still the base, find the latest study directory
    if CHECKPOINT_DIR == BASE_CHECKPOINT_DIR:
        subdirs = [os.path.join(BASE_CHECKPOINT_DIR, d) for d in os.listdir(BASE_CHECKPOINT_DIR) 
                  if os.path.isdir(os.path.join(BASE_CHECKPOINT_DIR, d))]
        if subdirs:
            # Sort by modification time to find the latest
            CHECKPOINT_DIR = max(subdirs, key=os.path.getmtime)
            logger.info(f"Using latest checkpoint directory: {CHECKPOINT_DIR}")
        else:
            # Fallback to base if no subdirs exist (for backward compatibility or first run)
            pass

    best_params_path = os.path.join(CHECKPOINT_DIR, 'best_params.json')
    
    if not os.path.exists(best_params_path):
        logger.error(f"No best_params.json found in {CHECKPOINT_DIR}. Run search first.")
        return
    
    with open(best_params_path, 'r') as f:
        config = json.load(f)
    
    # Backward compatibility for older studies
    config.setdefault('blur_sigma', BLUR_SIGMA)
    config.setdefault('emd_lambda', EMD_LAMBDA)
    config.setdefault('representation_mode', SELECTED_REPR_MODE)
    
    logger.info("Training with best params:")
    logger.info(json.dumps(config, indent=2))
    
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'best_model.pt')
    
    # Train for longer with best params
    best_val_loss = train(
        config,
        max_epochs=MAX_EPOCHS,  # Use the increased MAX_EPOCHS (200)
        checkpoint_path=checkpoint_path
    )
    
    logger.info(f"Final best validation loss: {best_val_loss:.4f}")


# ============================================================================
# Main
# ============================================================================

def main():
    global SEARCH_SPACE, SELECTED_MODEL_TYPE, SELECTED_REPR_MODE, TRANSFORMER_PATCH_HEIGHT, TRANSFORMER_PATCH_WIDTH
    
    parser = argparse.ArgumentParser(description='Train Diffusion TSF on Electricity dataset')
    parser.add_argument('--resume', action='store_true', help='Resume Optuna search')
    parser.add_argument('--best', action='store_true', help='Train with best found params')
    parser.add_argument('--trials', type=int, default=NUM_OPTUNA_TRIALS, help='Number of Optuna trials')
    parser.add_argument('--quick', action='store_true', help='Quick test with fewer samples')
    parser.add_argument('--model-type', choices=['unet', 'transformer'], default='unet', help='Backbone: unet (default) or transformer (DiT-style)')
    parser.add_argument('--blur-sigma', type=float, default=BLUR_SIGMA, help='Vertical blur sigma for preprocessing (label smoothing)')
    parser.add_argument('--emd-lambda', type=float, default=EMD_LAMBDA, help='Weight for EMD loss term')
    parser.add_argument('--repr-mode', choices=['pdf', 'cdf'], default='pdf', help='Representation: pdf (stripe) or cdf (occupancy)')
    parser.add_argument('--patch-height', type=int, default=16, choices=[4, 8, 16, 32], help='Transformer patch height (value axis, default: 16)')
    parser.add_argument('--patch-width', type=int, default=16, choices=[1, 2, 4, 8, 16, 32], help='Transformer patch width (time axis, default: 16). Smaller = finer temporal detail')
    args = parser.parse_args()
    
    # Check for optuna
    try:
        import optuna
        globals()['optuna'] = optuna
    except ImportError:
        logger.error("Optuna not installed. Run: pip install optuna")
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("DIFFUSION TSF - ELECTRICITY TRAINING")
    logger.info("=" * 60)
    logger.info(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    logger.info(f"Data: {DATA_PATH}")
    logger.info(f"Base checkpoints: {BASE_CHECKPOINT_DIR}")
    
    # Initialize hardware-adaptive search space
    SEARCH_SPACE = get_hardware_config()
    SEARCH_SPACE['blur_sigma'] = [args.blur_sigma]
    SEARCH_SPACE['emd_lambda'] = [args.emd_lambda]
    SELECTED_REPR_MODE = args.repr_mode
    # Set selected model type for downstream use
    SELECTED_MODEL_TYPE = args.model_type
    # Set transformer patch sizes
    TRANSFORMER_PATCH_HEIGHT = args.patch_height
    TRANSFORMER_PATCH_WIDTH = args.patch_width
    logger.info(f"Search space: batch_sizes={SEARCH_SPACE['batch_size']}, model_sizes={SEARCH_SPACE['model_size']}")
    if args.model_type == 'transformer':
        logger.info(f"Transformer patch size: {TRANSFORMER_PATCH_HEIGHT}x{TRANSFORMER_PATCH_WIDTH} (HxW)")
    
    if args.quick:
        # Quick test mode - minimal settings for fast verification
        logger.info("Quick test mode - using minimal settings (16 samples, 2 epochs)")
        
        config = {
            'learning_rate': 1e-4,
            'model_size': 'tiny',
            'diffusion_steps': 50,
            'batch_size': 4,
            'noise_schedule': 'linear',
            'model_type': args.model_type,
            'blur_sigma': args.blur_sigma,
            'emd_lambda': args.emd_lambda,
            'transformer_patch_height': args.patch_height,
            'transformer_patch_width': args.patch_width,
            'use_coordinate_channel': True,  # Enable vertical spatial awareness
        }
        
        # Use tiny dataset for quick test
        from torch.utils.data import Subset
        base_dataset = ElectricityDataset(
            DATA_PATH, lookback=64, forecast=16, max_samples=20, augment=False
        )
        train_indices = list(range(16))
        val_indices = list(range(16, 20))
        
        train_dataset = ElectricityDataset(
            DATA_PATH,
            lookback=64,
            forecast=16,
            augment=True,
            data_tensor=base_dataset.data,
            indices=train_indices
        )
        val_dataset = ElectricityDataset(
            DATA_PATH,
            lookback=64,
            forecast=16,
            augment=False,
            data_tensor=base_dataset.data,
            indices=val_indices
        )
        
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=4)
        
        # Create tiny model
        tiny_config = DiffusionTSFConfig(
            lookback_length=64, forecast_length=16, image_height=32,
            unet_channels=[16, 32], num_res_blocks=1, attention_levels=[1],
            num_diffusion_steps=50, ddim_steps=5, model_type=args.model_type,
            blur_sigma=args.blur_sigma, emd_lambda=args.emd_lambda,
            representation_mode=args.repr_mode,
            transformer_patch_height=min(args.patch_height, 8),  # Use smaller patch for tiny test
            transformer_patch_width=min(args.patch_width, 8),
            use_coordinate_channel=config.get('use_coordinate_channel', True),
        )
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = DiffusionTSF(tiny_config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
        
        for epoch in range(2):
            model.train()
            total_loss = 0
            for past, future in train_loader:
                past, future = past.to(device), future.to(device)
                optimizer.zero_grad()
                loss = model(past, future)['loss']
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            logger.info(f"Epoch {epoch}: train_loss={total_loss/len(train_loader):.4f}")
        
        logger.info("[OK] Quick test passed!")
        
    elif args.best:
        train_with_best_params()
        
    else:
        # Run Optuna search
        run_optuna_search(n_trials=args.trials, resume=args.resume)


if __name__ == "__main__":
    main()

