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
    python train_electricity.py --use-defaults     # Train with pre-tuned default params (no Optuna)
    python train_electricity.py --list-checkpoints # List all available checkpoint files
    python train_electricity.py --resume-checkpoint PATH/TO/CHECKPOINT.pt  # Resume from specific checkpoint
    python train_electricity.py --params-file X    # Train with params from a JSON file
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
from dataset import apply_1d_augmentations, get_synthetic_dataloader
from realts import RealTS
from guidance import (
    GuidanceModel, 
    LastValueGuidance, 
    LinearRegressionGuidance, 
    iTransformerGuidance,
    create_guidance_model
)

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
DATASETS_DIR = os.path.join(script_dir, '..', '..', 'datasets')
BASE_CHECKPOINT_DIR = os.path.join(script_dir, 'checkpoints')
CHECKPOINT_DIR = BASE_CHECKPOINT_DIR  # Will be updated by run_optuna_search or train_with_best_params
OPTUNA_DB = os.path.join(script_dir, 'optuna_study.db')

# Dataset registry: name -> (relative_path, default_target_column, seasonal_period)
# seasonal_period: 96 for hourly (daily cycle), 24 for 15-min (daily cycle), 168 for weekly
DATASET_REGISTRY = {
    'electricity': ('electricity/electricity.csv', 'OT', 96),  # Hourly data
    'ETTh1': ('ETT-small/ETTh1.csv', 'OT', 24),  # Hourly data (24h cycle)
    'ETTh2': ('ETT-small/ETTh2.csv', 'OT', 24),
    'ETTm1': ('ETT-small/ETTm1.csv', 'OT', 96),  # 15-min data (96 = 24h)
    'ETTm2': ('ETT-small/ETTm2.csv', 'OT', 96),
    'exchange_rate': ('exchange_rate/exchange_rate.csv', 'OT', 5),  # Daily data (5 = weekly business days)
    'illness': ('illness/national_illness.csv', 'OT', 52),  # Weekly data (52 = yearly cycle)
    'traffic': ('traffic/traffic.csv', 'OT', 24),  # Hourly data
    'weather': ('weather/weather.csv', 'OT', 144),  # 10-min data (144 = daily cycle)
}

# Default dataset (will be updated from CLI)
SELECTED_DATASET = 'electricity'
DATA_PATH = os.path.join(DATASETS_DIR, DATASET_REGISTRY[SELECTED_DATASET][0])
TARGET_COLUMN = None  # None = use dataset default from registry

# Fixed parameters (aligned with ViTime paper)
LOOKBACK_LENGTH = 512      # Same as ViTime paper
FORECAST_LENGTH = 96       # Common benchmark (paper uses 96, 192, 336, 720)
IMAGE_HEIGHT = 128         # ViTime paper: h=128
BLUR_KERNEL = 31           # ViTime paper: kernel=31
BLUR_SIGMA = 1.0           # Sharper labels; EMD handles non-overlap
EMD_LAMBDA = 0.0           # Best found value (was 0.2)
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

def get_high_end_search_space():
    """Return the high-end GPU search space (for synthetic pre-training or forced mode)."""
    return {
        'learning_rate': (1e-6, 1e-3),  # Wide LR range: 1e-5 to 1e-3
        'model_size': ['small', 'large'],  # Include all model sizes
        'diffusion_steps': [2000, 4000],  # More diffusion options
        'batch_size': [128, 512],  # Much larger batch sizes
        'noise_schedule': ['linear'],  # Well-established noise schedules
    }


def get_finetune_search_space():
    """Return a more conservative search space for fine-tuning pre-trained models.
    
    Uses lower learning rates to avoid catastrophic forgetting when fine-tuning
    a universal pre-trained model on domain-specific data.
    """
    return {
        'learning_rate': (1e-6, 1e-4),  # More conservative LR for fine-tuning
        'model_size': ['small', 'large'],  # Same model sizes
        'diffusion_steps': [2000, 4000],  # Same diffusion options
        'batch_size': [128, 512],  # Same batch sizes
        'noise_schedule': ['linear'],  # Same noise schedule
    }


def get_hardware_config():
    """Get hardware-adaptive search space based on GPU memory."""
    # If fine-tuning from pretrained checkpoint, use conservative LR range
    if FINETUNE_MODE and PRETRAINED_CHECKPOINT_PATH:
        logger.info("Fine-tuning mode enabled - using conservative LR range (1e-6, 1e-4)")
        return get_finetune_search_space()
    
    # If forced, use high-end search space regardless of hardware
    if FORCE_HIGH_END_SEARCH:
        logger.info("Forced high-end search space enabled - using extensive search space")
        return get_high_end_search_space()
    
    gpu_mem = get_gpu_memory_gb()
    
    if gpu_mem >= 40:  # Ada 6000, A100, etc.
        logger.info(f"Detected high-end GPU ({gpu_mem:.1f}GB) - using extensive search space")
        return get_high_end_search_space()
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
# U-Net kernel size (height, width) - set from CLI
UNET_KERNEL_SIZE = (3, 3)
# Time channels for U-Net temporal awareness (set from CLI)
USE_TIME_RAMP = True  # Linear ramp channel
USE_TIME_SINE = True  # Sine wave channel
USE_VALUE_CHANNEL = False  # Value channel (normalized values broadcast)
USE_COORDINATE_CHANNEL = True  # Vertical coordinate channel
SEASONAL_PERIOD = 96

# Multivariate mode (set from CLI)
USE_ALL_COLUMNS = False  # If True, use all numeric columns for multivariate forecasting

# Dilated middle block (set from CLI)
USE_DILATED_MIDDLE = True  # If True, use dilated convolutions in U-Net bottleneck

# Monotonicity loss (CDF regularization)
USE_MONOTONICITY_LOSS = False
MONOTONICITY_WEIGHT = 10.0

# Hybrid 1D cross-attention conditioning (set from CLI)
USE_HYBRID_CONDITION = True  # If True, use 1D context encoder + cross-attention

# Visual Guide (Stage 1 predictor) settings (set from CLI)
USE_GUIDANCE_CHANNEL = False  # If True, use Stage 1 predictor as guidance
GUIDANCE_TYPE = "linear"  # "linear", "last_value", or "itransformer"
GUIDANCE_CHECKPOINT = None  # Path to pre-trained iTransformer checkpoint (if guidance_type="itransformer")

# Synthetic data pre-training settings (set from CLI)
# Two-phase training: (1) pre-train on synthetic data, (2) fine-tune on real data
SYNTHETIC_PRETRAIN_EPOCHS = 0  # Number of epochs to pre-train on synthetic data (0 = disabled)
SYNTHETIC_SIZE = 10000  # Number of synthetic samples to generate for pre-training

# Pure synthetic training mode (for hyperparameter search on synthetic data)
SYNTHETIC_ONLY_MODE = False  # If True, train ONLY on synthetic data (no real data)

# Pre-trained checkpoint for fine-tuning (set from CLI)
PRETRAINED_CHECKPOINT_PATH = None  # Path to a pre-trained model to start fine-tuning from

# Force high-end search space regardless of GPU detection
FORCE_HIGH_END_SEARCH = False

# Custom run name for Optuna studies (overrides default naming)
CUSTOM_RUN_NAME = None

# Fine-tuning mode (enables more conservative LR range when loading pretrained checkpoint)
# When True and pretrained_checkpoint is set, uses LR range (1e-6, 1e-4) instead of (1e-5, 1e-3)
FINETUNE_MODE = False

# Data split settings
# IMPORTANT: Time series data MUST use chronological splits to avoid data leakage.
# With sliding windows, adjacent samples share ~90% of their data. Random splitting
# causes train and val sets to overlap temporally, making validation metrics meaningless.
# Split: Train (first 70%), Val (next 10%), Test (last 20%)
USE_CHRONOLOGICAL_SPLIT = True  # ALWAYS True for time series - random split causes severe data leakage

DATASET_STRIDE = 24  # Stride for sliding window (configurable via CLI)

MODEL_SIZES = {
    'tiny': [32, 64],           # ~1M params, for quick tests only
    'small': [64, 128, 256],    # ~10M params
    'medium': [64, 128, 256, 512],  # ~40M params
    'large': [128, 256, 512],   # ~80M params (close to ViTime's 93M)
}

# ============================================================================
# Guidance Model Loading
# ============================================================================

def load_itransformer_from_checkpoint(
    checkpoint_path: str,
    seq_len: int = LOOKBACK_LENGTH,
    pred_len: int = FORECAST_LENGTH,
    num_variables: int = 1,
    device: str = 'cpu'
) -> iTransformerGuidance:
    """Load a pre-trained iTransformer model as guidance.
    
    Args:
        checkpoint_path: Path to iTransformer checkpoint (.pt file)
        seq_len: Input sequence length (should match training)
        pred_len: Prediction length (should match training)
        num_variables: Number of variables in the dataset
        device: Device to load model on
        
    Returns:
        iTransformerGuidance wrapper around the loaded model
    """
    import sys
    import importlib.util
    
    # Use importlib to load from absolute path to avoid conflicts with local model.py
    itrans_model_path = os.path.join(script_dir, '..', 'iTransformer', 'model', 'iTransformer.py')
    itrans_model_path = os.path.abspath(itrans_model_path)
    
    # Also need to add iTransformer to path for its internal imports (layers, etc.)
    itrans_dir = os.path.join(script_dir, '..', 'iTransformer')
    itrans_dir = os.path.abspath(itrans_dir)
    if itrans_dir not in sys.path:
        sys.path.insert(0, itrans_dir)
    
    # Load the module using spec
    spec = importlib.util.spec_from_file_location("iTransformer_module", itrans_model_path)
    itrans_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(itrans_module)
    iTransformerModel = itrans_module.Model
    
    # Load checkpoint
    logger.info(f"Loading iTransformer from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Try to extract config from checkpoint
    if 'config' in checkpoint:
        ckpt_config = checkpoint['config']
        logger.info(f"Found config in checkpoint: seq_len={ckpt_config.get('seq_len')}, pred_len={ckpt_config.get('pred_len')}")
    else:
        ckpt_config = {}
    
    # Get the state dict to infer model architecture
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Auto-detect e_layers from state_dict by counting encoder.attn_layers.X
    detected_e_layers = 0
    for key in state_dict.keys():
        if key.startswith('encoder.attn_layers.'):
            layer_idx = int(key.split('.')[2])
            detected_e_layers = max(detected_e_layers, layer_idx + 1)
    
    # Auto-detect d_model from embedding weight shape
    detected_d_model = 512  # default
    detected_d_ff = 2048  # default
    if 'enc_embedding.value_embedding.weight' in state_dict:
        detected_d_model = state_dict['enc_embedding.value_embedding.weight'].shape[0]
    # Auto-detect d_ff from conv1 weight shape
    if 'encoder.attn_layers.0.conv1.weight' in state_dict:
        detected_d_ff = state_dict['encoder.attn_layers.0.conv1.weight'].shape[0]
    
    logger.info(f"Auto-detected from state_dict: e_layers={detected_e_layers}, d_model={detected_d_model}, d_ff={detected_d_ff}")
    
    # Create a config object for iTransformer
    class iTransConfig:
        def __init__(self):
            self.seq_len = ckpt_config.get('seq_len', seq_len)
            self.pred_len = ckpt_config.get('pred_len', pred_len)
            self.output_attention = False
            self.use_norm = True
            self.d_model = ckpt_config.get('d_model', detected_d_model)
            self.embed = 'fixed'
            self.freq = 'h'
            self.dropout = 0.1
            self.factor = 1
            self.n_heads = ckpt_config.get('n_heads', 8)
            self.d_ff = ckpt_config.get('d_ff', detected_d_ff)
            self.activation = 'gelu'
            self.e_layers = ckpt_config.get('e_layers', detected_e_layers if detected_e_layers > 0 else 3)
            self.class_strategy = 'projection'
            self.enc_in = num_variables
    
    config = iTransConfig()
    
    # Create model
    model = iTransformerModel(config)
    
    # Load weights
    model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    
    logger.info(f"iTransformer loaded: seq_len={config.seq_len}, pred_len={config.pred_len}")
    
    # Wrap in guidance interface
    guidance = iTransformerGuidance(
        model=model,
        use_norm=config.use_norm,
        seq_len=config.seq_len,
        pred_len=config.pred_len
    )
    
    return guidance


def create_guidance_for_training(
    guidance_type: str,
    guidance_checkpoint: Optional[str],
    seq_len: int,
    pred_len: int,
    num_variables: int,
    device: str
) -> Optional[GuidanceModel]:
    """Create guidance model based on configuration.
    
    Args:
        guidance_type: Type of guidance ("linear", "last_value", "itransformer")
        guidance_checkpoint: Path to checkpoint (required for itransformer)
        seq_len: Lookback length
        pred_len: Forecast length
        num_variables: Number of variables
        device: Device to use
        
    Returns:
        GuidanceModel instance or None if guidance is disabled
    """
    if guidance_type == "linear":
        logger.info("Using LinearRegressionGuidance for Stage 1 predictions")
        return LinearRegressionGuidance()
    
    elif guidance_type == "last_value":
        logger.info("Using LastValueGuidance for Stage 1 predictions")
        return LastValueGuidance()
    
    elif guidance_type == "itransformer":
        if not guidance_checkpoint:
            raise ValueError(
                "guidance_type='itransformer' requires --guidance-checkpoint PATH"
            )
        return load_itransformer_from_checkpoint(
            checkpoint_path=guidance_checkpoint,
            seq_len=seq_len,
            pred_len=pred_len,
            num_variables=num_variables,
            device=device
        )
    
    else:
        raise ValueError(f"Unknown guidance_type: {guidance_type}")

# Training settings
MAX_EPOCHS = 275
PATIENCE = 25                   # Early stopping patience (increased for longer training)
VAL_SPLIT = 0.1
NUM_OPTUNA_TRIALS = 10          # Total trials to run
PRUNING_WARMUP = 20             # Don't prune before this epoch (increased for longer training)
OVERALL_SAVE_INTERVAL = 25      # Re-save the global best every N epochs to avoid loss on interrupts

# Pre-tuned default parameters (from best Optuna run)
# These are good starting points that work well across datasets
DEFAULT_PARAMS = {
    "learning_rate": 4.25e-05,
    "model_size": "small",
    "diffusion_steps": 200,
    "batch_size": 16,
    "noise_schedule": "cosine",
    "blur_sigma": 1.0,
    "emd_lambda": 0.0,
    "representation_mode": "cdf",
    "use_monotonicity_loss": False,
    "monotonicity_weight": 1.0,
}

# ============================================================================
# Dataset
# ============================================================================

class ElectricityDataset(Dataset):
    """Electricity dataset for time series forecasting.
    
    Supports both univariate and multivariate forecasting.
    - Univariate: Single column (e.g., 'OT') -> tensors of shape (seq_len,)
    - Multivariate: Multiple columns or 'all' -> tensors of shape (num_vars, seq_len)
    
    Creates sliding windows of (past, future) pairs.
    """
    
    def __init__(
        self,
        data_path: str,
        lookback: int = 512,
        forecast: int = 96,
        column: str = 'OT',
        columns: Optional[List[str]] = None,  # For multivariate: list of columns or None
        use_all_columns: bool = False,  # If True, use all numeric columns
        stride: int = 24,  # Stride for sliding window (1 day = 24 hours)
        max_samples: Optional[int] = None,
        augment: bool = True,
        data_tensor: Optional[torch.Tensor] = None,
        indices: Optional[List[int]] = None
    ):
        if data_tensor is None:
            logger.info(f"Loading electricity data from {data_path}")
            df = pd.read_csv(data_path)
            
            # Determine which columns to use
            if use_all_columns:
                # Use all numeric columns (exclude date column)
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                self.columns = numeric_cols
                logger.info(f"Using all {len(self.columns)} numeric columns: {self.columns[:5]}...")
                self.data = torch.tensor(df[self.columns].values, dtype=torch.float32)  # (time, num_vars)
                self.data = self.data.T  # -> (num_vars, time) for channel-first
            elif columns is not None:
                # Use specified columns (multivariate)
                self.columns = columns
                for col in columns:
                    if col not in df.columns:
                        raise ValueError(f"Column '{col}' not found in dataset")
                logger.info(f"Using {len(columns)} columns: {columns}")
                self.data = torch.tensor(df[columns].values, dtype=torch.float32)  # (time, num_vars)
                self.data = self.data.T  # -> (num_vars, time)
            else:
                # Univariate mode (backwards compatible)
                if column not in df.columns:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    column = numeric_cols[0]
                    logger.warning(f"Column 'OT' not found, using '{column}' instead")
                self.columns = [column]
                self.data = torch.tensor(df[column].values, dtype=torch.float32)  # (time,)
        else:
            # Reuse already loaded tensor to avoid extra IO/memory
            self.data = data_tensor
            self.columns = ['unknown']
        
        self.lookback = lookback
        self.forecast = forecast
        self.total_len = lookback + forecast
        self.augment = augment
        self.multivariate = self.data.dim() == 2  # True if (num_vars, time)
        self.num_variables = self.data.shape[0] if self.multivariate else 1
        
        # Calculate number of samples
        # data_len is the time dimension
        data_len = self.data.shape[-1] if self.multivariate else len(self.data)
        if indices is not None:
            self.indices = indices
        else:
            num_windows = (data_len - self.total_len) // stride + 1
            self.indices = [i * stride for i in range(num_windows)]
        
        if max_samples and indices is None and len(self.indices) > max_samples:
            self.indices = self.indices[:max_samples]
        
        logger.info(f"Created dataset: {len(self.indices)} samples, "
                   f"lookback={lookback}, forecast={forecast}, "
                   f"variables={self.num_variables}")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        start = self.indices[idx]
        
        if self.multivariate:
            # window shape: (num_vars, total_len)
            window = self.data[:, start:start + self.total_len]
            
            if self.augment:
                # Apply augmentation to each variable separately
                augmented = []
                for v in range(window.shape[0]):
                    augmented.append(apply_1d_augmentations(window[v]))
                window = torch.stack(augmented)
            
            past = window[:, :self.lookback]  # (num_vars, lookback)
            future = window[:, self.lookback:]  # (num_vars, forecast)
        else:
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
    forecast: int = FORECAST_LENGTH,
    use_all_columns: bool = False,
    columns: Optional[List[str]] = None,
    column: Optional[str] = None,  # Target column for univariate (None = use global TARGET_COLUMN)
    use_chronological_split: bool = True  # MUST be True for time series to avoid data leakage
) -> Tuple[DataLoader, DataLoader, int]:
    """Create train and validation dataloaders for REAL data.
    
    NOTE: For synthetic pre-training, use get_synthetic_dataloader() from dataset.py.
    
    Args:
        batch_size: Batch size for dataloaders
        val_split: Fraction of data to use for validation (only used if not chronological)
        max_samples: Maximum number of samples to use (None = all)
        lookback: Lookback window length
        forecast: Forecast horizon length
        use_all_columns: If True, use all columns (multivariate)
        columns: Specific columns to use (if not use_all_columns)
        column: Target column for univariate forecasting (overrides global TARGET_COLUMN)
        use_chronological_split: If True (DEFAULT), use chronological split (70% train, 10% val, 20% test).
                                 WARNING: Setting to False causes severe data leakage in time series!
    
    Returns:
        (train_loader, val_loader, num_variables)
    """
    # Use provided column, else fall back to global TARGET_COLUMN
    target_column = column if column is not None else TARGET_COLUMN
    # CRITICAL: Warn if someone tries to use random split
    if not use_chronological_split:
        logger.warning(
            "⚠️  RANDOM SPLIT ENABLED - THIS CAUSES SEVERE DATA LEAKAGE FOR TIME SERIES!\n"
            "   With sliding windows (stride=24, window=608), adjacent samples share 584 timesteps.\n"
            "   Random shuffle means train/val samples are interleaved temporally.\n"
            "   Your validation metrics will be MEANINGLESS. Use use_chronological_split=True."
        )
    
    # Base dataset (no augmentation) to derive indices and reuse tensor
    base_dataset = ElectricityDataset(
        DATA_PATH,
        lookback=lookback,
        forecast=forecast,
        column=target_column,
        max_samples=max_samples,
        augment=False,
        use_all_columns=use_all_columns,
        columns=columns
    )
    
    total_samples = len(base_dataset)
    
    if use_chronological_split:
        # CHRONOLOGICAL SPLIT WITH GAPS to prevent window overlap
        # 
        # Problem: With stride=24 and window=608, adjacent samples share data:
        #   Sample i covers timesteps [i*stride, i*stride + window)
        #   Sample i+1 covers timesteps [(i+1)*stride, (i+1)*stride + window)
        #   Overlap = window - stride = 608 - 24 = 584 timesteps!
        #
        # Solution: Insert a GAP of ceil(window/stride) indices between splits
        # This ensures the last window of train doesn't overlap with first window of val.
        
        window_size = lookback + forecast  # Total timesteps per sample
        stride = DATASET_STRIDE  # Configurable stride from CLI/global
        gap_indices = (window_size + stride - 1) // stride  # Ceiling division
        
        # Target proportions: ~70% train, ~10% val, ~20% test
        # But we need gaps, so effective sizes are smaller
        raw_train_end = int(total_samples * 0.7)
        raw_val_end = int(total_samples * 0.8)
        
        train_end = raw_train_end
        val_start = train_end + gap_indices  # Gap after train
        val_end = raw_val_end
        test_start = val_end + gap_indices   # Gap after val
        
        # Ensure we have valid ranges
        if val_start >= val_end:
            logger.warning(f"Gap too large for val split! Reducing gap.")
            val_start = train_end + 1
        if test_start >= total_samples:
            test_start = val_end + 1
        
        train_indices = list(range(0, train_end))
        val_indices = list(range(val_start, val_end))
        # test_indices would be range(test_start, total_samples) if needed
        
        # Calculate actual timestep ranges for logging
        train_ts_end = (train_end - 1) * stride + window_size if train_end > 0 else 0
        val_ts_start = val_start * stride
        val_ts_end = (val_end - 1) * stride + window_size if val_end > val_start else val_ts_start
        
        logger.info(f"Using CHRONOLOGICAL split with gaps (no window overlap):")
        logger.info(f"  Window size: {window_size} timesteps, stride: {stride}, gap: {gap_indices} indices")
        logger.info(f"  Train: indices 0-{train_end-1} ({len(train_indices)} samples)")
        logger.info(f"         timesteps 0-{train_ts_end}")
        logger.info(f"  [GAP]: {gap_indices} indices ({gap_indices * stride} timesteps)")
        logger.info(f"  Val:   indices {val_start}-{val_end-1} ({len(val_indices)} samples)")
        logger.info(f"         timesteps {val_ts_start}-{val_ts_end}")
        logger.info(f"  [GAP]: {gap_indices} indices")
        logger.info(f"  Test:  indices {test_start}-{total_samples-1} (held out)")
    else:
        # RANDOM SPLIT: original behavior
        val_size = int(total_samples * val_split)
        train_size = total_samples - val_size
        
        train_subset, val_subset = random_split(
            base_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        train_indices = list(train_subset.indices)
        val_indices = list(val_subset.indices)
        
        logger.info(f"Using RANDOM split (seed=42):")
        logger.info(f"  Train: {len(train_indices)} samples ({100-val_split*100:.0f}%)")
        logger.info(f"  Val:   {len(val_indices)} samples ({val_split*100:.0f}%)")
    
    train_dataset = ElectricityDataset(
        DATA_PATH,
        lookback=lookback,
        forecast=forecast,
        max_samples=max_samples,
        augment=True,
        data_tensor=base_dataset.data,
        indices=train_indices
    )
    
    val_dataset = ElectricityDataset(
        DATA_PATH,
        lookback=lookback,
        forecast=forecast,
        max_samples=max_samples,
        augment=False,
        data_tensor=base_dataset.data,
        indices=val_indices
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
    
    logger.info(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples, Variables: {base_dataset.num_variables}")
    
    return train_loader, val_loader, base_dataset.num_variables


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


def _load_val_loss_from_checkpoint(path: str) -> float:
    """Safely load val_loss from a checkpoint, return inf if unavailable."""
    try:
        ckpt = torch.load(path, map_location="cpu")
        return float(ckpt.get("val_loss", float("inf")))
    except Exception:
        return float("inf")


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
    
    # Create dataloaders FIRST to get actual num_variables from dataset
    # Use chronological split when using iTransformer guidance to match its training split
    train_loader, val_loader, num_variables = get_dataloaders(
        batch_size=config['batch_size'],
        val_split=VAL_SPLIT,
        use_all_columns=config.get('use_all_columns', False),
        use_chronological_split=USE_CHRONOLOGICAL_SPLIT
    )
    
    # Create synthetic dataloader for pre-training phase (if enabled)
    synthetic_pretrain_epochs = config.get('synthetic_pretrain_epochs', SYNTHETIC_PRETRAIN_EPOCHS)
    synthetic_loader = None
    if synthetic_pretrain_epochs > 0:
        synthetic_size = config.get('synthetic_size', SYNTHETIC_SIZE)
        synthetic_loader = get_synthetic_dataloader(
            num_samples=synthetic_size,
            lookback_length=LOOKBACK_LENGTH,
            forecast_length=FORECAST_LENGTH,
            batch_size=config['batch_size'],
            shuffle=True,
            seed=42  # Fixed seed for reproducibility
        )
        logger.info(f"Synthetic pre-training enabled: {synthetic_pretrain_epochs} epochs on {synthetic_size} samples")
    
    # Update config with actual number of variables from dataset
    config['num_variables'] = num_variables
    logger.info(f"Dataset num_variables: {num_variables}")
    
    model_config = DiffusionTSFConfig(
        lookback_length=LOOKBACK_LENGTH,
        forecast_length=FORECAST_LENGTH,
        image_height=IMAGE_HEIGHT,
        max_scale=MAX_SCALE,
        blur_kernel_size=BLUR_KERNEL,
        blur_sigma=config.get('blur_sigma', BLUR_SIGMA),
        emd_lambda=config.get('emd_lambda', EMD_LAMBDA),
        use_monotonicity_loss=config.get('use_monotonicity_loss', USE_MONOTONICITY_LOSS),
        monotonicity_weight=config.get('monotonicity_weight', MONOTONICITY_WEIGHT),
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
        unet_kernel_size=config.get('unet_kernel_size', UNET_KERNEL_SIZE),
        use_dilated_middle=config.get('use_dilated_middle', USE_DILATED_MIDDLE),
        use_time_ramp=config.get('use_time_ramp', USE_TIME_RAMP),
        use_time_sine=config.get('use_time_sine', USE_TIME_SINE),
        use_value_channel=config.get('use_value_channel', USE_VALUE_CHANNEL),
        seasonal_period=config.get('seasonal_period', SEASONAL_PERIOD),
        num_variables=num_variables,  # Use actual value from dataset
        # Hybrid 1D cross-attention conditioning
        use_hybrid_condition=config.get('use_hybrid_condition', USE_HYBRID_CONDITION),
        context_embedding_dim=config.get('context_embedding_dim', 128),
        context_encoder_layers=config.get('context_encoder_layers', 2),
        # Visual Guide (Stage 1 predictor)
        use_guidance_channel=config.get('use_guidance_channel', USE_GUIDANCE_CHANNEL),
    )
    
    # Create guidance model if enabled
    guidance_model = None
    if model_config.use_guidance_channel:
        guidance_model = create_guidance_for_training(
            guidance_type=config.get('guidance_type', GUIDANCE_TYPE),
            guidance_checkpoint=config.get('guidance_checkpoint', GUIDANCE_CHECKPOINT),
            seq_len=LOOKBACK_LENGTH,
            pred_len=FORECAST_LENGTH,
            num_variables=num_variables,  # Use actual value from dataset
            device=device
        )
    
    model = DiffusionTSF(model_config, guidance_model=guidance_model).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")
    logger.info(f"Representation mode: {model_config.representation_mode}")
    logger.info(f"Blur sigma: {model_config.blur_sigma}, EMD lambda: {model_config.emd_lambda}")
    
    # Load pre-trained weights if specified (for fine-tuning)
    pretrained_path = config.get('pretrained_checkpoint', PRETRAINED_CHECKPOINT_PATH)
    if pretrained_path and os.path.exists(pretrained_path):
        logger.info(f"Loading pre-trained weights from: {pretrained_path}")
        pretrained_ckpt = torch.load(pretrained_path, map_location=device)
        
        # Load model weights with PARTIAL loading (handles size mismatches gracefully)
        pretrained_state = pretrained_ckpt.get('model_state_dict', pretrained_ckpt)
        model_state = model.state_dict()
        
        loaded_keys = []
        skipped_keys = []
        
        for key in pretrained_state.keys():
            if key in model_state:
                # Check if shapes match
                if pretrained_state[key].shape == model_state[key].shape:
                    model_state[key] = pretrained_state[key]
                    loaded_keys.append(key)
                else:
                    # Handle partial loading for first conv (guidance channel added)
                    if 'init_conv.weight' in key:
                        # Copy the channels that exist in pretrained, leave new channel random
                        pretrained_channels = pretrained_state[key].shape[1]
                        model_state[key][:, :pretrained_channels, :, :] = pretrained_state[key]
                        loaded_keys.append(f"{key} (partial: {pretrained_channels}/{model_state[key].shape[1]} channels)")
                    else:
                        skipped_keys.append(f"{key} (shape mismatch: {pretrained_state[key].shape} vs {model_state[key].shape})")
            else:
                skipped_keys.append(f"{key} (not in model)")
        
        model.load_state_dict(model_state)
        
        logger.info(f"Loaded {len(loaded_keys)} weight tensors from pretrained checkpoint")
        if skipped_keys:
            logger.warning(f"Skipped {len(skipped_keys)} tensors due to shape/key mismatch:")
            for k in skipped_keys[:10]:  # Show first 10
                logger.warning(f"  - {k}")
            if len(skipped_keys) > 10:
                logger.warning(f"  ... and {len(skipped_keys) - 10} more")
    
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
    # Track global best across all trials/runs inside the current CHECKPOINT_DIR
    best_model_path = os.path.join(CHECKPOINT_DIR, 'best_model.pt')
    overall_best_loss = _load_val_loss_from_checkpoint(best_model_path)
    logger.info(f"Current overall best (if any): {overall_best_loss:.4f}" if overall_best_loss < float('inf') else "No overall best model yet.")
    
    # Resume from checkpoint if exists
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = load_checkpoint(checkpoint_path, model, optimizer)
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt['val_loss']
        logger.info(f"Resuming from epoch {start_epoch}, best val_loss: {best_val_loss:.4f}")
    
    # Check if we're in synthetic-only mode
    synthetic_only = config.get('synthetic_only', SYNTHETIC_ONLY_MODE)
    
    # =========================================================================
    # SYNTHETIC-ONLY MODE: Train purely on synthetic data (for HP search)
    # =========================================================================
    if synthetic_only:
        logger.info("=" * 60)
        logger.info("SYNTHETIC-ONLY MODE: Training on synthetic data, validating on real data")
        logger.info("=" * 60)
        
        # Create synthetic loader for all training
        synthetic_loader_for_training = get_synthetic_dataloader(
            num_samples=config.get('synthetic_size', SYNTHETIC_SIZE),
            lookback_length=LOOKBACK_LENGTH,
            forecast_length=FORECAST_LENGTH,
            batch_size=config['batch_size'],
            shuffle=True,
            seed=42
        )
        
        for epoch in range(start_epoch, max_epochs):
            epoch_start = time.time()
            
            # Train on SYNTHETIC data
            train_loss = train_epoch(model, synthetic_loader_for_training, optimizer, device, epoch)
            
            # Validate on REAL validation data (to track transfer quality)
            val_metrics = validate(model, val_loader, device, use_generation=(epoch % 5 == 0))
            val_loss = val_metrics['val_loss']
            
            # Update scheduler
            scheduler.step()
            
            epoch_time = time.time() - epoch_start
            logger.info(f"[SYNTH-ONLY] Epoch {epoch}/{max_epochs} | "
                       f"Synth Train: {train_loss:.4f} | Real Val: {val_loss:.4f} | "
                       f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                       f"Time: {epoch_time:.1f}s")
            
            if 'val_mse' in val_metrics:
                logger.info(f"  Generation metrics: {log_metrics({k: v for k, v in val_metrics.items() if k != 'val_loss'})}")
            
            # Save best model (based on real validation performance)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if checkpoint_path:
                    save_checkpoint(
                        model, optimizer, epoch, train_loss, val_loss,
                        config, checkpoint_path.replace('.pt', '_best.pt')
                    )
                    # Also update the global overall best
                    if val_loss < overall_best_loss:
                        overall_best_loss = val_loss
                        save_checkpoint(
                            model, optimizer, epoch, train_loss, val_loss,
                            config, best_model_path
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
    
    # =========================================================================
    # PHASE 1: Pre-train on synthetic data (if enabled, not synthetic-only)
    # =========================================================================
    if synthetic_loader is not None and start_epoch == 0:
        logger.info("=" * 60)
        logger.info("PHASE 1: PRE-TRAINING ON SYNTHETIC DATA")
        logger.info("=" * 60)
        
        for pretrain_epoch in range(synthetic_pretrain_epochs):
            epoch_start = time.time()
            
            # Train on synthetic data
            train_loss = train_epoch(model, synthetic_loader, optimizer, device, pretrain_epoch)
            
            # Validate on REAL validation data (to track transfer quality)
            val_metrics = validate(model, val_loader, device, use_generation=False)
            val_loss = val_metrics['val_loss']
            
            # Update scheduler
            scheduler.step()
            
            epoch_time = time.time() - epoch_start
            logger.info(f"[PRETRAIN] Epoch {pretrain_epoch}/{synthetic_pretrain_epochs} | "
                       f"Synth Train: {train_loss:.4f} | Real Val: {val_loss:.4f} | "
                       f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                       f"Time: {epoch_time:.1f}s")
        
        # Save pre-trained checkpoint before fine-tuning
        if checkpoint_path:
            pretrain_ckpt_path = checkpoint_path.replace('.pt', '_pretrained.pt')
            save_checkpoint(
                model, optimizer, synthetic_pretrain_epochs - 1, train_loss, val_loss,
                config, pretrain_ckpt_path
            )
            logger.info(f"Pre-trained checkpoint saved: {pretrain_ckpt_path}")
        
        logger.info("=" * 60)
        logger.info("PHASE 2: FINE-TUNING ON REAL DATA")
        logger.info("=" * 60)
        
        # Reset early stopping for fine-tuning phase
        early_stopping = EarlyStopping(patience=PATIENCE)
    
    # =========================================================================
    # PHASE 2: Fine-tune on real data (or regular training if no pre-training)
    # =========================================================================
    for epoch in range(start_epoch, max_epochs):
        epoch_start = time.time()
        
        # Train on REAL data
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        
        # Validate
        val_metrics = validate(model, val_loader, device, use_generation=(epoch % 5 == 0))
        val_loss = val_metrics['val_loss']
        
        # Update scheduler
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        phase_label = "[FINETUNE] " if synthetic_loader is not None else ""
        logger.info(f"{phase_label}Epoch {epoch}/{max_epochs} | "
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
                # Also update the global overall best if this run beats it
                if val_loss < overall_best_loss:
                    overall_best_loss = val_loss
                    save_checkpoint(
                        model, optimizer, epoch, train_loss, val_loss,
                        config, best_model_path
                    )
        
        # Save regular checkpoint
        if checkpoint_path and epoch % 5 == 0:
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss,
                config, checkpoint_path
            )

        # Periodically re-save the best checkpoint of this run as the global best (defensive)
        if checkpoint_path and epoch % OVERALL_SAVE_INTERVAL == 0 and epoch > 0:
            run_best_path = checkpoint_path.replace('.pt', '_best.pt')
            if os.path.exists(run_best_path):
                run_best_loss = _load_val_loss_from_checkpoint(run_best_path)
                if run_best_loss < overall_best_loss:
                    overall_best_loss = run_best_loss
                    import shutil
                    shutil.copy2(run_best_path, best_model_path)
                    logger.info(f"Updated best_model.pt at epoch {epoch} (val_loss={overall_best_loss:.4f})")
        
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

# Cache for pretrained checkpoint config (to avoid reloading for every trial)
_PRETRAINED_CONFIG_CACHE = {}

def _get_pretrained_config(checkpoint_path: str) -> dict:
    """Load and cache config from pretrained checkpoint."""
    if checkpoint_path not in _PRETRAINED_CONFIG_CACHE:
        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            _PRETRAINED_CONFIG_CACHE[checkpoint_path] = ckpt.get('config', {})
        else:
            _PRETRAINED_CONFIG_CACHE[checkpoint_path] = {}
    return _PRETRAINED_CONFIG_CACHE[checkpoint_path]


def objective(trial) -> float:
    """Optuna objective function."""
    
    # If fine-tuning from pretrained checkpoint, extract architecture params that MUST match
    forced_model_size = None
    forced_diffusion_steps = None
    if PRETRAINED_CHECKPOINT_PATH and os.path.exists(PRETRAINED_CHECKPOINT_PATH):
        pretrained_config = _get_pretrained_config(PRETRAINED_CHECKPOINT_PATH)
        if pretrained_config:
            # MUST use same model_size as pretrained to load weights correctly
            forced_model_size = pretrained_config.get('model_size')
            forced_diffusion_steps = pretrained_config.get('diffusion_steps')
            # Only log on first trial to avoid spam
            if trial.number == 0:
                if forced_model_size:
                    logger.info(f"Fine-tuning: forcing model_size={forced_model_size} from pretrained checkpoint")
                if forced_diffusion_steps:
                    logger.info(f"Fine-tuning: forcing diffusion_steps={forced_diffusion_steps} from pretrained checkpoint")
                if not forced_model_size:
                    logger.warning("Pretrained checkpoint has no model_size in config - architecture mismatch may occur!")
    
    # Sample hyperparameters (force architecture params if fine-tuning)
    config = {
        'learning_rate': trial.suggest_float('learning_rate', *SEARCH_SPACE['learning_rate'], log=True),
        'model_size': forced_model_size if forced_model_size else trial.suggest_categorical('model_size', SEARCH_SPACE['model_size']),
        'diffusion_steps': forced_diffusion_steps if forced_diffusion_steps else trial.suggest_categorical('diffusion_steps', SEARCH_SPACE['diffusion_steps']),
        'batch_size': trial.suggest_categorical('batch_size', SEARCH_SPACE['batch_size']),
        'noise_schedule': trial.suggest_categorical('noise_schedule', SEARCH_SPACE['noise_schedule']),
        'model_type': SELECTED_MODEL_TYPE,
        'blur_sigma': trial.suggest_categorical('blur_sigma', SEARCH_SPACE['blur_sigma']),
        'emd_lambda': trial.suggest_categorical('emd_lambda', SEARCH_SPACE['emd_lambda']),
        'representation_mode': SELECTED_REPR_MODE,
        'transformer_patch_height': TRANSFORMER_PATCH_HEIGHT,
        'transformer_patch_width': TRANSFORMER_PATCH_WIDTH,
        'use_coordinate_channel': USE_COORDINATE_CHANNEL,  # Vertical spatial awareness
        'unet_kernel_size': UNET_KERNEL_SIZE,
        'use_dilated_middle': USE_DILATED_MIDDLE,  # Dilated bottleneck
        'use_time_ramp': USE_TIME_RAMP,  # Enable linear ramp time channel
        'use_time_sine': USE_TIME_SINE,  # Enable sine wave time channel
        'use_value_channel': USE_VALUE_CHANNEL,  # Enable value channel
        'seasonal_period': SEASONAL_PERIOD,
        'use_all_columns': USE_ALL_COLUMNS,  # Multivariate mode
        'use_hybrid_condition': USE_HYBRID_CONDITION,  # Hybrid 1D conditioning
        'use_guidance_channel': USE_GUIDANCE_CHANNEL,  # Visual Guide (Stage 1)
        'guidance_type': GUIDANCE_TYPE,
        'guidance_checkpoint': GUIDANCE_CHECKPOINT,
        'dataset': SELECTED_DATASET,  # Dataset name for visualization
        'use_monotonicity_loss': USE_MONOTONICITY_LOSS,
        'monotonicity_weight': MONOTONICITY_WEIGHT,
        # Synthetic pre-training (two-phase: pretrain on synthetic, then fine-tune on real)
        'synthetic_pretrain_epochs': SYNTHETIC_PRETRAIN_EPOCHS,
        'synthetic_size': SYNTHETIC_SIZE,
        # Pure synthetic training mode (no real data training)
        'synthetic_only': SYNTHETIC_ONLY_MODE,
        # Pre-trained checkpoint for fine-tuning
        'pretrained_checkpoint': PRETRAINED_CHECKPOINT_PATH,
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


def run_optuna_search(n_trials: int = NUM_OPTUNA_TRIALS, resume: bool = True, run_name: Optional[str] = None):
    """Run Optuna hyperparameter search.
    
    Args:
        n_trials: Number of Optuna trials to run
        resume: Whether to resume from existing study
        run_name: Custom name for the study/checkpoint directory (overrides default naming)
    """
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    
    # Create or load study
    storage = f"sqlite:///{OPTUNA_DB}"
    
    # Use custom run name if provided, otherwise default naming
    if run_name or CUSTOM_RUN_NAME:
        study_name = run_name or CUSTOM_RUN_NAME
        logger.info(f"Using custom study name: {study_name}")
    else:
        base_study_name = f"diffusion_tsf_{SELECTED_DATASET}"
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
    
    # When using custom run_name, always load_if_exists to allow resuming
    # (the user explicitly wants this specific study name)
    should_load_if_exists = resume or (run_name is not None) or (CUSTOM_RUN_NAME is not None)
    
    if should_load_if_exists:
        logger.info(f"Will resume existing study '{study_name}' if it exists")
    
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        storage=storage,
        load_if_exists=should_load_if_exists,
        sampler=TPESampler(seed=42),  # Bayesian optimization
        pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=PRUNING_WARMUP)
    )
    
    logger.info(f"Starting Optuna search: {n_trials} trials")
    logger.info(f"Previous trials in study: {len(study.trials)}")
    
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


def train_with_params(params: dict, run_name: Optional[str] = None):
    """Train with specified parameters (no Optuna).
    
    Args:
        params: Dictionary of hyperparameters to use
        run_name: Optional name for the run (used for checkpoint directory)
    """
    global CHECKPOINT_DIR
    
    # Create checkpoint directory for this run
    if run_name:
        CHECKPOINT_DIR = os.path.join(BASE_CHECKPOINT_DIR, run_name)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        CHECKPOINT_DIR = os.path.join(BASE_CHECKPOINT_DIR, f"direct_train_{timestamp}")
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    logger.info(f"Checkpoints will be saved to: {CHECKPOINT_DIR}")
    
    # Build config from params, applying defaults for missing keys
    config = dict(DEFAULT_PARAMS)  # Start with defaults
    config.update(params)  # Override with provided params
    
    # CLI flags ALWAYS override defaults (user intent takes precedence)
    config['model_type'] = SELECTED_MODEL_TYPE
    config['representation_mode'] = SELECTED_REPR_MODE
    config['blur_sigma'] = SEARCH_SPACE['blur_sigma'][0]
    config['emd_lambda'] = SEARCH_SPACE['emd_lambda'][0]
    config['unet_kernel_size'] = UNET_KERNEL_SIZE
    config['use_dilated_middle'] = USE_DILATED_MIDDLE
    config['use_time_ramp'] = USE_TIME_RAMP
    config['use_time_sine'] = USE_TIME_SINE
    config['use_value_channel'] = USE_VALUE_CHANNEL
    config['use_coordinate_channel'] = USE_COORDINATE_CHANNEL
    config['seasonal_period'] = SEASONAL_PERIOD
    config['use_all_columns'] = USE_ALL_COLUMNS
    config['use_hybrid_condition'] = USE_HYBRID_CONDITION
    config['use_guidance_channel'] = USE_GUIDANCE_CHANNEL
    config['guidance_type'] = GUIDANCE_TYPE
    config['guidance_checkpoint'] = GUIDANCE_CHECKPOINT
    config['transformer_patch_height'] = TRANSFORMER_PATCH_HEIGHT
    config['transformer_patch_width'] = TRANSFORMER_PATCH_WIDTH
    config['dataset'] = SELECTED_DATASET  # Dataset name for visualization
    config['use_monotonicity_loss'] = USE_MONOTONICITY_LOSS
    config['monotonicity_weight'] = MONOTONICITY_WEIGHT
    # Synthetic pre-training (two-phase: pretrain on synthetic, then fine-tune on real)
    config['synthetic_pretrain_epochs'] = SYNTHETIC_PRETRAIN_EPOCHS
    config['synthetic_size'] = SYNTHETIC_SIZE
    # Pure synthetic training mode
    config['synthetic_only'] = SYNTHETIC_ONLY_MODE
    # Pre-trained checkpoint for fine-tuning
    config['pretrained_checkpoint'] = PRETRAINED_CHECKPOINT_PATH
    
    logger.info("Training with params:")
    logger.info(json.dumps(config, indent=2, default=str))
    
    # Save params for reproducibility
    params_path = os.path.join(CHECKPOINT_DIR, 'params.json')
    with open(params_path, 'w') as f:
        # Convert tuples to lists for JSON serialization
        serializable_config = {k: list(v) if isinstance(v, tuple) else v for k, v in config.items()}
        json.dump(serializable_config, f, indent=2)
    logger.info(f"Params saved to {params_path}")
    
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'model.pt')
    
    # Train with automatic batch size reduction on OOM
    original_batch = config['batch_size']
    current_batch = original_batch
    
    while current_batch >= 2:
        try:
            config['batch_size'] = current_batch
            best_val_loss = train(
                config,
                max_epochs=MAX_EPOCHS,
                checkpoint_path=checkpoint_path
            )
            logger.info(f"Final best validation loss: {best_val_loss:.4f}")
            return best_val_loss
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                new_batch = current_batch // 2
                logger.warning(f"OOM with batch_size={current_batch}, retrying with batch_size={new_batch}")
                current_batch = new_batch
            else:
                raise
    
    logger.error(f"OOM even with batch_size=2, training failed")
    return float('inf')


# ============================================================================
# Main
# ============================================================================

def list_available_checkpoints():
    """List all checkpoint files in the checkpoints directory."""
    if not os.path.exists(BASE_CHECKPOINT_DIR):
        logger.info(f"No checkpoints directory found at: {BASE_CHECKPOINT_DIR}")
        return

    checkpoints = []
    for root, dirs, files in os.walk(BASE_CHECKPOINT_DIR):
        for file in files:
            if file.endswith('.pt'):
                checkpoints.append(os.path.join(root, file))

    if not checkpoints:
        logger.info("No checkpoint files (.pt) found in checkpoints directory")
        return

    logger.info("Available checkpoint files:")
    for i, ckpt in enumerate(sorted(checkpoints), 1):
        # Get relative path for cleaner display
        rel_path = os.path.relpath(ckpt, BASE_CHECKPOINT_DIR)
        logger.info(f"  {i}. {rel_path}")


def main():
    global SEARCH_SPACE, SELECTED_MODEL_TYPE, SELECTED_REPR_MODE, TRANSFORMER_PATCH_HEIGHT, TRANSFORMER_PATCH_WIDTH, UNET_KERNEL_SIZE, USE_TIME_RAMP, USE_TIME_SINE, USE_VALUE_CHANNEL, SEASONAL_PERIOD, USE_ALL_COLUMNS, SELECTED_DATASET, DATA_PATH, USE_DILATED_MIDDLE, USE_HYBRID_CONDITION, USE_COORDINATE_CHANNEL, USE_GUIDANCE_CHANNEL, GUIDANCE_TYPE, GUIDANCE_CHECKPOINT, USE_CHRONOLOGICAL_SPLIT
    
    parser = argparse.ArgumentParser(description='Train Diffusion TSF on Electricity dataset')
    parser.add_argument('--resume', action='store_true', help='Resume Optuna search')
    parser.add_argument('--best', action='store_true', help='Train with best found params from latest Optuna study')
    parser.add_argument('--use-defaults', action='store_true', 
                        help='Train with pre-tuned default params (no Optuna, fast start)')
    parser.add_argument('--params-file', type=str, default=None, metavar='PATH',
                        help='Train with params from a JSON file (e.g., best_params.json from a previous run)')
    parser.add_argument('--resume-checkpoint', type=str, default=None, metavar='PATH',
                        help='Resume training from a specific checkpoint file (.pt)')
    parser.add_argument('--list-checkpoints', action='store_true',
                        help='List all available checkpoint files and exit')
    parser.add_argument('--run-name', type=str, default=None, metavar='NAME',
                        help='Name for the training run (used with --use-defaults or --params-file)')
    parser.add_argument('--trials', type=int, default=NUM_OPTUNA_TRIALS, help='Number of Optuna trials')
    parser.add_argument('--quick', action='store_true', help='Quick test with fewer samples')
    parser.add_argument('--model-type', choices=['unet', 'transformer'], default='unet', help='Backbone: unet (default) or transformer (DiT-style)')
    parser.add_argument('--blur-sigma', type=float, default=BLUR_SIGMA, help='Vertical blur sigma for preprocessing (label smoothing)')
    parser.add_argument('--emd-lambda', type=float, default=EMD_LAMBDA, help='Weight for EMD loss term')
    parser.add_argument('--repr-mode', choices=['pdf', 'cdf'], default='cdf', help='Representation: pdf (stripe) or cdf (occupancy)')
    parser.add_argument('--use-monotonicity-loss', action='store_true', default=False,
                        help='Add monotonicity regularization on predicted CDF (optional)')
    parser.add_argument('--monotonicity-weight', type=float, default=10.0,
                        help='Weight for monotonicity loss term')
    parser.add_argument('--patch-height', type=int, default=16, choices=[4, 8, 16, 32], help='Transformer patch height (value axis, default: 16)')
    parser.add_argument('--patch-width', type=int, default=16, choices=[1, 2, 4, 8, 16, 32], help='Transformer patch width (time axis, default: 16). Smaller = finer temporal detail')
    parser.add_argument('--use-coordinate-channel', action='store_true', default=True,
                        help='Add vertical coordinate channel (gradient +1 to -1) for spatial awareness (default: True)')
    parser.add_argument('--no-coordinate-channel', dest='use_coordinate_channel', action='store_false',
                        help='Disable vertical coordinate channel')
    parser.add_argument('--kernel-size', type=int, nargs=2, default=[3, 3], metavar=('H', 'W'),
                        help='U-Net conv kernel size as (height, width). Height=value axis, Width=time axis. '
                             'E.g., --kernel-size 3 5 for wider temporal receptive field. Must be odd numbers. (default: 3 3)')
    parser.add_argument('--dilated-middle', action='store_true', default=False,
                        help='Use dilated convolutions in U-Net bottleneck for expanded temporal receptive field')
    parser.add_argument('--use-time-ramp', action='store_true', default=False,
                        help='Add linear ramp time channel (-1 to +1 "progress bar")')
    parser.add_argument('--no-time-ramp', dest='use_time_ramp', action='store_false',
                        help='Disable linear ramp time channel')
    parser.add_argument('--use-time-sine', action='store_true', default=False,
                        help='Add sine wave time channel (periodic "clock")')
    parser.add_argument('--no-time-sine', dest='use_time_sine', action='store_false',
                        help='Disable sine wave time channel')
    parser.add_argument('--use-value-channel', action='store_true', default=False,
                        help='Add 2D channel with normalized values broadcast across height')
    parser.add_argument('--no-value-channel', dest='use_value_channel', action='store_false',
                        help='Disable value channel')
    parser.add_argument('--seasonal-period', type=int, default=96,
                        help='Period for sine wave time channel (e.g., 96 for hourly data with daily seasonality)')
    parser.add_argument('--use-hybrid-condition', action='store_true', default=True,
                        help='Enable hybrid 1D cross-attention conditioning (default: True)')
    parser.add_argument('--no-hybrid-condition', dest='use_hybrid_condition', action='store_false',
                        help='Disable hybrid 1D cross-attention conditioning')
    parser.add_argument('--multivariate', action='store_true', default=False,
                        help='Enable multivariate forecasting using all columns in the dataset')
    parser.add_argument('--dataset', type=str, default='electricity',
                        choices=list(DATASET_REGISTRY.keys()),
                        help=f'Dataset to use for training. Available: {", ".join(DATASET_REGISTRY.keys())}')
    parser.add_argument('--target', type=str, default=None, metavar='COLUMN',
                        help='Target column for univariate forecasting (overrides dataset default). '
                             'For ETTh1/ETTh2 use HUFL, HULL, MUFL, MULL, LUFL, LULL, or OT')
    # Visual Guide (Stage 1 predictor) arguments
    parser.add_argument('--use-guidance', action='store_true', default=False,
                        help='Enable Visual Guide: use Stage 1 predictor to guide diffusion')
    parser.add_argument('--guidance-type', type=str, default='linear',
                        choices=['linear', 'last_value', 'itransformer'],
                        help='Type of guidance model: linear (regression), last_value (naive), '
                             'or itransformer (requires --guidance-checkpoint)')
    parser.add_argument('--guidance-checkpoint', type=str, default=None, metavar='PATH',
                        help='Path to pre-trained iTransformer checkpoint for guidance '
                             '(required when --guidance-type=itransformer)')
    parser.add_argument('--stride', type=int, default=24,
                        help='Stride for sliding window (default: 24, meaning 1 day for hourly data)')
    # Synthetic data pre-training arguments (two-phase: pretrain on synthetic, then fine-tune on real)
    parser.add_argument('--synthetic-pretrain-epochs', type=int, default=0, metavar='N',
                        help='Number of epochs to PRE-TRAIN on synthetic RealTS data before fine-tuning '
                             'on real data. Set to 0 to disable (default: 0). '
                             'Recommended: 20-50 epochs for small datasets.')
    parser.add_argument('--synthetic-size', type=int, default=10000, metavar='N',
                        help='Number of synthetic samples to generate for pre-training '
                             '(default: 10000)')
    # Pure synthetic training mode (for hyperparameter search)
    parser.add_argument('--synthetic-only', action='store_true', default=False,
                        help='Train ONLY on synthetic data (no real data). Used for hyperparameter '
                             'search on synthetic data to find best params for universal pre-training.')
    # Pre-trained checkpoint for fine-tuning
    parser.add_argument('--pretrained-checkpoint', type=str, default=None, metavar='PATH',
                        help='Path to a pre-trained model checkpoint (.pt) to initialize weights from. '
                             'Used for fine-tuning a universal pre-trained model on real data.')
    # Force high-end search space
    parser.add_argument('--force-high-end-search', action='store_true', default=False,
                        help='Force using high-end GPU search space regardless of actual hardware. '
                             'Useful for consistent hyperparameter ranges across different machines.')
    parser.add_argument('--finetune-mode', action='store_true', default=False,
                        help='Enable fine-tuning mode with conservative LR range (1e-6, 1e-4). '
                             'Automatically enabled when --pretrained-checkpoint is set.')
    args = parser.parse_args()
    
    # Check for optuna
    try:
        import optuna
        globals()['optuna'] = optuna
    except ImportError:
        logger.error("Optuna not installed. Run: pip install optuna")
        sys.exit(1)
    
    # Set all config globals from args FIRST before any logging
    # Set dataset
    global TARGET_COLUMN
    SELECTED_DATASET = args.dataset
    dataset_info = DATASET_REGISTRY[SELECTED_DATASET]
    DATA_PATH = os.path.join(DATASETS_DIR, dataset_info[0])
    # Set target column: use CLI arg if provided, else dataset default from registry
    TARGET_COLUMN = args.target if args.target else dataset_info[1]
    
    # Set time channel settings
    USE_TIME_RAMP = args.use_time_ramp
    USE_TIME_SINE = args.use_time_sine
    USE_VALUE_CHANNEL = args.use_value_channel
    USE_COORDINATE_CHANNEL = args.use_coordinate_channel
    # Use dataset-specific seasonal period unless overridden by CLI
    if args.seasonal_period == 96:  # Default value means user didn't specify
        SEASONAL_PERIOD = dataset_info[2]  # Use dataset-specific default
    else:
        SEASONAL_PERIOD = args.seasonal_period
    # Set multivariate mode
    USE_ALL_COLUMNS = args.multivariate
    
    # Set hybrid conditioning mode
    USE_HYBRID_CONDITION = args.use_hybrid_condition
    
    # Set Visual Guide (Stage 1 predictor) settings
    USE_GUIDANCE_CHANNEL = args.use_guidance
    GUIDANCE_TYPE = args.guidance_type
    GUIDANCE_CHECKPOINT = args.guidance_checkpoint

    # Set dataset stride
    global DATASET_STRIDE
    DATASET_STRIDE = args.stride
    
    # Validate guidance settings
    if USE_GUIDANCE_CHANNEL and GUIDANCE_TYPE == 'itransformer' and not GUIDANCE_CHECKPOINT:
        logger.error("--guidance-type=itransformer requires --guidance-checkpoint PATH")
        sys.exit(1)
    
    # Set synthetic pre-training settings
    global SYNTHETIC_PRETRAIN_EPOCHS, SYNTHETIC_SIZE, SYNTHETIC_ONLY_MODE, PRETRAINED_CHECKPOINT_PATH, FORCE_HIGH_END_SEARCH, CUSTOM_RUN_NAME, FINETUNE_MODE
    SYNTHETIC_PRETRAIN_EPOCHS = args.synthetic_pretrain_epochs
    SYNTHETIC_SIZE = args.synthetic_size
    SYNTHETIC_ONLY_MODE = args.synthetic_only
    PRETRAINED_CHECKPOINT_PATH = args.pretrained_checkpoint
    FORCE_HIGH_END_SEARCH = args.force_high_end_search
    CUSTOM_RUN_NAME = args.run_name
    # Auto-enable finetune mode when pretrained checkpoint is provided
    FINETUNE_MODE = args.finetune_mode or (args.pretrained_checkpoint is not None)
    
    # Validate: synthetic-only requires synthetic-pretrain-epochs or will be used as pure synthetic training
    if SYNTHETIC_ONLY_MODE and SYNTHETIC_SIZE <= 0:
        logger.error("--synthetic-only requires --synthetic-size > 0")
        sys.exit(1)
    
    # Validate: pretrained-checkpoint must exist if specified
    if PRETRAINED_CHECKPOINT_PATH and not os.path.exists(PRETRAINED_CHECKPOINT_PATH):
        logger.error(f"Pre-trained checkpoint not found: {PRETRAINED_CHECKPOINT_PATH}")
        sys.exit(1)
    
    # Validate synthetic pre-training constraint: only CDF mode is supported
    if (SYNTHETIC_PRETRAIN_EPOCHS > 0 or SYNTHETIC_ONLY_MODE) and args.repr_mode == 'pdf':
        logger.error(
            "Synthetic training is currently only supported in CDF representation mode. "
            "Please switch to --repr-mode cdf."
        )
        sys.exit(1)
    
    # IMPORTANT: Time series ALWAYS needs chronological split to avoid data leakage.
    # With sliding windows (stride=24), adjacent samples share ~583 of 608 timesteps.
    # Random splitting means train sample 0 and val sample 5 share 90%+ of data!
    # USE_CHRONOLOGICAL_SPLIT is now always True by default (set at module level).
    logger.info("Using CHRONOLOGICAL split (70% train / 10% val / 20% test) - required for time series")
    
    # Initialize hardware-adaptive search space
    SEARCH_SPACE = get_hardware_config()
    SEARCH_SPACE['blur_sigma'] = [args.blur_sigma]
    SEARCH_SPACE['emd_lambda'] = [args.emd_lambda]
    SELECTED_REPR_MODE = args.repr_mode
    USE_MONOTONICITY_LOSS = args.use_monotonicity_loss
    MONOTONICITY_WEIGHT = args.monotonicity_weight
    # Set selected model type for downstream use
    SELECTED_MODEL_TYPE = args.model_type
    # Set transformer patch sizes
    TRANSFORMER_PATCH_HEIGHT = args.patch_height
    TRANSFORMER_PATCH_WIDTH = args.patch_width
    # Set U-Net kernel size
    UNET_KERNEL_SIZE = tuple(args.kernel_size)
    # Set dilated middle block
    USE_DILATED_MIDDLE = args.dilated_middle
    
    # Now log everything
    logger.info("=" * 60)
    logger.info(f"DIFFUSION TSF - {SELECTED_DATASET.upper()} TRAINING")
    logger.info("=" * 60)
    logger.info(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    logger.info(f"Dataset: {SELECTED_DATASET}")
    logger.info(f"Data: {DATA_PATH}")
    logger.info(f"Target column: {TARGET_COLUMN} (univariate)" if not USE_ALL_COLUMNS else "Multivariate (all columns)")
    logger.info(f"Seasonal period: {SEASONAL_PERIOD}")
    logger.info(f"Base checkpoints: {BASE_CHECKPOINT_DIR}")
    logger.info(f"Search space: batch_sizes={SEARCH_SPACE['batch_size']}, model_sizes={SEARCH_SPACE['model_size']}")
    if args.model_type == 'unet':
        logger.info(f"U-Net kernel size: {UNET_KERNEL_SIZE} (height x width)")
        logger.info(f"Dilated middle block: {USE_DILATED_MIDDLE}")
        logger.info(f"Time ramp: {USE_TIME_RAMP}, Time sine: {USE_TIME_SINE}, Value channel: {USE_VALUE_CHANNEL} (period={SEASONAL_PERIOD})")
    if args.model_type == 'transformer':
        logger.info(f"Transformer patch size: {TRANSFORMER_PATCH_HEIGHT}x{TRANSFORMER_PATCH_WIDTH} (HxW)")
    logger.info(f"Multivariate mode: {USE_ALL_COLUMNS}")
    logger.info(f"Hybrid 1D conditioning: {USE_HYBRID_CONDITION}")
    logger.info(f"Monotonicity loss: {USE_MONOTONICITY_LOSS} (weight={MONOTONICITY_WEIGHT})")
    if USE_GUIDANCE_CHANNEL:
        logger.info(f"Visual Guide: enabled (type={GUIDANCE_TYPE})")
        if GUIDANCE_CHECKPOINT:
            logger.info(f"  Guidance checkpoint: {GUIDANCE_CHECKPOINT}")
    if SYNTHETIC_PRETRAIN_EPOCHS > 0:
        logger.info(f"Synthetic pre-training: {SYNTHETIC_PRETRAIN_EPOCHS} epochs on {SYNTHETIC_SIZE} samples")
    if SYNTHETIC_ONLY_MODE:
        logger.info(f"SYNTHETIC-ONLY MODE: Training purely on {SYNTHETIC_SIZE} synthetic samples (no real data)")
    if PRETRAINED_CHECKPOINT_PATH:
        logger.info(f"Fine-tuning from pre-trained checkpoint: {PRETRAINED_CHECKPOINT_PATH}")
    if FINETUNE_MODE:
        logger.info("Fine-tuning mode: ENABLED (conservative LR range 1e-6 to 1e-4)")
    if FORCE_HIGH_END_SEARCH:
        logger.info("Force high-end search space: ENABLED")
    
    if args.list_checkpoints:
        # List all checkpoint files and exit
        list_available_checkpoints()
        sys.exit(0)

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
            'use_coordinate_channel': USE_COORDINATE_CHANNEL,
            'unet_kernel_size': UNET_KERNEL_SIZE,
            'use_dilated_middle': USE_DILATED_MIDDLE,  # Dilated bottleneck
            'use_time_ramp': USE_TIME_RAMP,  # Enable linear ramp time channel
            'use_time_sine': USE_TIME_SINE,  # Enable sine wave time channel
            'use_value_channel': USE_VALUE_CHANNEL,  # Enable value channel
            'seasonal_period': SEASONAL_PERIOD,
            'use_all_columns': USE_ALL_COLUMNS,  # Multivariate mode
            'use_hybrid_condition': USE_HYBRID_CONDITION,  # Hybrid 1D conditioning
            'use_guidance_channel': USE_GUIDANCE_CHANNEL,  # Visual Guide
            'guidance_type': GUIDANCE_TYPE,
            'guidance_checkpoint': GUIDANCE_CHECKPOINT,
            'dataset': SELECTED_DATASET,  # Dataset name for visualization
            'use_monotonicity_loss': USE_MONOTONICITY_LOSS,
            'monotonicity_weight': MONOTONICITY_WEIGHT,
            # Synthetic pre-training not used in quick test
            'synthetic_pretrain_epochs': 0,
            'synthetic_size': 0,
        }
        
        # Use tiny dataset for quick test
        from torch.utils.data import Subset
        base_dataset = ElectricityDataset(
            DATA_PATH, lookback=64, forecast=16, max_samples=20, augment=False,
            use_all_columns=USE_ALL_COLUMNS
        )
        train_indices = list(range(16))
        val_indices = list(range(16, 20))
        num_variables = base_dataset.num_variables
        
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
            unet_kernel_size=UNET_KERNEL_SIZE,
            use_dilated_middle=USE_DILATED_MIDDLE,
            use_time_ramp=USE_TIME_RAMP,
            use_time_sine=USE_TIME_SINE,
            use_value_channel=USE_VALUE_CHANNEL,
            seasonal_period=SEASONAL_PERIOD,
            num_variables=num_variables,
            use_hybrid_condition=USE_HYBRID_CONDITION,  # Hybrid 1D conditioning
            context_embedding_dim=32,   # Smaller for quick test
            context_encoder_layers=1,
            use_guidance_channel=USE_GUIDANCE_CHANNEL,  # Visual Guide
            use_monotonicity_loss=USE_MONOTONICITY_LOSS,
            monotonicity_weight=MONOTONICITY_WEIGHT,
        )
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create guidance model for quick test if enabled
        guidance_model = None
        if USE_GUIDANCE_CHANNEL:
            guidance_model = create_guidance_for_training(
                guidance_type=GUIDANCE_TYPE,
                guidance_checkpoint=GUIDANCE_CHECKPOINT,
                seq_len=64,  # Quick test uses shorter sequences
                pred_len=16,
                num_variables=num_variables,
                device=device
            )
        
        model = DiffusionTSF(tiny_config, guidance_model=guidance_model).to(device)
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
    
    elif args.use_defaults:
        # Train with pre-tuned default params (no Optuna)
        logger.info("Using pre-tuned default parameters (no Optuna search)")
        train_with_params(DEFAULT_PARAMS, run_name=args.run_name)
    
    elif args.params_file:
        # Train with params from a JSON file
        params_path = args.params_file
        if not os.path.exists(params_path):
            logger.error(f"Params file not found: {params_path}")
            sys.exit(1)
        
        logger.info(f"Loading params from: {params_path}")
        with open(params_path, 'r') as f:
            custom_params = json.load(f)
        
        train_with_params(custom_params, run_name=args.run_name)

    elif args.resume_checkpoint:
        # Resume training from a specific checkpoint file
        checkpoint_path = args.resume_checkpoint
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint file not found: {checkpoint_path}")
            sys.exit(1)

        logger.info(f"Resuming training from checkpoint: {checkpoint_path}")

        # Load checkpoint to get the config
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        config = checkpoint['config']

        logger.info("Resuming with checkpoint's original configuration:")
        logger.info(f"  use_coordinate_channel: {config.get('use_coordinate_channel', True)}")
        logger.info(f"  use_time_ramp: {config.get('use_time_ramp', False)}")
        logger.info(f"  use_time_sine: {config.get('use_time_sine', False)}")
        logger.info(f"  use_value_channel: {config.get('use_value_channel', False)}")
        logger.info(f"  use_hybrid_condition: {config.get('use_hybrid_condition', True)}")
        logger.info(f"  use_guidance_channel: {config.get('use_guidance_channel', False)}")
        if config.get('use_guidance_channel'):
            logger.info(f"  guidance_type: {config.get('guidance_type', 'linear')}")

        # Warn if CLI flags might conflict
        cli_flags_set = []
        if args.use_coordinate_channel is not True: cli_flags_set.append('no_coordinate_channel')
        if args.use_time_ramp: cli_flags_set.append('use_time_ramp')
        if args.use_time_sine: cli_flags_set.append('use_time_sine')
        if args.use_value_channel: cli_flags_set.append('use_value_channel')
        if not args.use_hybrid_condition: cli_flags_set.append('use_hybrid_condition (disabled)')
        if args.use_guidance: cli_flags_set.append('use_guidance')

        if cli_flags_set:
            logger.warning("WARNING: You specified CLI flags that may conflict with checkpoint config:")
            for flag in cli_flags_set:
                logger.warning(f"  --{flag.replace('_', '-')}")
            logger.warning("When resuming, the checkpoint's saved configuration takes precedence.")
            logger.warning("The CLI flags are ignored for model architecture settings.")

        # Override run name if specified
        run_name = args.run_name or f"resume_{Path(checkpoint_path).stem}"

        # Call train with the checkpoint path (it will auto-resume)
        train(
            config,
            max_epochs=MAX_EPOCHS,
            checkpoint_path=checkpoint_path
        )

    else:
        # Run Optuna search
        run_optuna_search(n_trials=args.trials, resume=args.resume, run_name=args.run_name)


if __name__ == "__main__":
    main()

