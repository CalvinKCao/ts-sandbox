"""
7-Variate Training Pipeline with Full HP Tuning.

PHASE 1: Synthetic Pretraining (with HP tuning)
  1A. iTransformer HP Tuning (20 trials, 100k samples)
      → tune lr, batch_size, dropout
  1B. Diffusion HP Tuning with iTransformer guidance (8 trials, 10k samples)
      → tune lr, batch_size
  1C. Full Pretraining (200 epochs, patience 20, 1M samples)
      → First train iTransformer, then Diffusion with guidance

PHASE 2: Fine-tuning per Dataset (simplified HP tuning)
  2A. HP Tune (8 trials, 200 epochs, patience 20)
      → tune lr, batch_size only
  2B. Full Fine-tune (200 epochs, patience 25)
  2C. Evaluate

Usage:
    # Single GPU
    python -m models.diffusion_tsf.train_7var_pipeline
    python -m models.diffusion_tsf.train_7var_pipeline --resume
    python -m models.diffusion_tsf.train_7var_pipeline --smoke-test
    
    # Multi-GPU (DDP)
    torchrun --nproc_per_node=4 -m models.diffusion_tsf.train_7var_pipeline --ddp
    torchrun --nproc_per_node=2 -m models.diffusion_tsf.train_7var_pipeline --ddp --resume
"""

import argparse
import importlib.util
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import optuna
from optuna.samplers import TPESampler
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Optional wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

# Setup path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.diffusion_tsf.config import DiffusionTSFConfig
from models.diffusion_tsf.diffusion_model import DiffusionTSF
from models.diffusion_tsf.metrics import compute_metrics
from models.diffusion_tsf.dataset import get_synthetic_dataloader
from models.diffusion_tsf.guidance import iTransformerGuidance

# ============================================================================
# DDP (Multi-GPU) Support
# ============================================================================

# Global DDP state
_ddp_enabled = False
_rank = 0
_world_size = 1
_local_rank = 0


def setup_ddp():
    """Initialize DDP. Call before any model/data creation."""
    global _ddp_enabled, _rank, _world_size, _local_rank
    
    if not dist.is_available():
        return False
    
    # Check if launched with torchrun
    if 'RANK' not in os.environ:
        return False
    
    _rank = int(os.environ['RANK'])
    _world_size = int(os.environ['WORLD_SIZE'])
    _local_rank = int(os.environ['LOCAL_RANK'])
    
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(_local_rank)
    
    _ddp_enabled = True
    return True


def cleanup_ddp():
    """Clean up DDP."""
    if _ddp_enabled:
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Returns True if this is the main process (rank 0)."""
    return _rank == 0


def get_rank() -> int:
    return _rank


def get_world_size() -> int:
    return _world_size


def get_device() -> torch.device:
    """Get device for current process."""
    if _ddp_enabled:
        return torch.device(f'cuda:{_local_rank}')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def wrap_model_ddp(model: nn.Module) -> nn.Module:
    """Wrap model with DDP if enabled."""
    if _ddp_enabled:
        model = model.to(get_device())
        return DDP(model, device_ids=[_local_rank], output_device=_local_rank)
    return model.to(get_device())


def unwrap_model(model: nn.Module) -> nn.Module:
    """Get the underlying model from DDP wrapper."""
    if isinstance(model, DDP):
        return model.module
    return model


def create_dataloader_ddp(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    drop_last: bool = False,
) -> Tuple[DataLoader, Optional[DistributedSampler]]:
    """Create DataLoader with DDP support."""
    sampler = None
    if _ddp_enabled:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        # When using sampler, don't pass shuffle to DataLoader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=drop_last,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            drop_last=drop_last,
        )
    return loader, sampler


def sync_across_processes(tensor: torch.Tensor) -> torch.Tensor:
    """Average tensor across all processes."""
    if not _ddp_enabled:
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= _world_size
    return tensor


def barrier():
    """Synchronize all processes."""
    if _ddp_enabled:
        dist.barrier()


# ============================================================================
# Parallel Optuna Workers (Multi-GPU HP Tuning)
# ============================================================================

_parallel_worker_id = None  # None = single process, 0-N = parallel worker ID
_optuna_storage = None  # Shared storage path for parallel workers


def setup_parallel_worker(worker_id: int, storage_path: str = None):
    """Configure this process as a parallel Optuna worker."""
    global _parallel_worker_id, _optuna_storage
    _parallel_worker_id = worker_id
    
    # Use env var or provided path for shared storage
    _optuna_storage = storage_path or os.environ.get('OPTUNA_STORAGE')
    if not _optuna_storage:
        # Default to SQLite in checkpoint dir
        _optuna_storage = f"sqlite:///{os.path.join(script_dir, 'checkpoints_7var', 'optuna_shared.db')}"
    
    logger = get_logger()
    logger.info(f"Parallel worker {worker_id} initialized with storage: {_optuna_storage}")


def is_parallel_mode() -> bool:
    """Check if running in parallel worker mode."""
    return _parallel_worker_id is not None


def get_worker_id() -> int:
    """Get worker ID (0 for single process or main worker)."""
    return _parallel_worker_id if _parallel_worker_id is not None else 0


def is_worker_zero() -> bool:
    """Returns True if this is worker 0 (or single process mode)."""
    return _parallel_worker_id is None or _parallel_worker_id == 0


def create_shared_study(study_name: str, direction: str = 'minimize') -> optuna.Study:
    """Create an Optuna study that can be shared across parallel workers.
    
    In parallel mode, uses shared SQLite storage so multiple workers
    can run trials concurrently. Optuna handles the coordination.
    """
    if is_parallel_mode() and _optuna_storage:
        # Shared storage for parallel workers
        return optuna.create_study(
            study_name=study_name,
            storage=_optuna_storage,
            direction=direction,
            load_if_exists=True,  # Workers join existing study
            sampler=TPESampler(),  # No fixed seed - let workers explore
        )
    else:
        # In-memory study for single process
        return optuna.create_study(
            direction=direction,
            sampler=TPESampler(seed=42),
        )


def parallel_worker_barrier():
    """Simple file-based barrier for parallel workers (not DDP)."""
    if not is_parallel_mode():
        return
    
    # Use filesystem for coordination
    barrier_dir = os.path.join(script_dir, 'checkpoints_7var', '.barriers')
    os.makedirs(barrier_dir, exist_ok=True)
    
    barrier_file = os.path.join(barrier_dir, f'worker_{_parallel_worker_id}.ready')
    
    # Signal this worker is ready
    Path(barrier_file).touch()
    
    # Wait for all workers (assume 4 workers max, adjust if needed)
    n_workers = int(os.environ.get('SLURM_GPUS_ON_NODE', 4))
    while True:
        ready = sum(1 for i in range(n_workers) 
                   if os.path.exists(os.path.join(barrier_dir, f'worker_{i}.ready')))
        if ready >= n_workers:
            break
        time.sleep(0.5)
    
    # Clean up
    if is_worker_zero():
        time.sleep(0.1)  # Let others finish reading
        for f in Path(barrier_dir).glob('worker_*.ready'):
            f.unlink()


# Logging - only main process/worker 0 logs fully
def setup_logging():
    """Setup logging - only rank 0 / worker 0 logs to file/stdout."""
    is_main = is_main_process() and is_worker_zero()
    level = logging.INFO if is_main else logging.WARNING
    handlers = []
    if is_main:
        handlers.append(logging.StreamHandler(sys.stdout))
        handlers.append(logging.FileHandler(os.path.join(script_dir, 'train_7var.log')))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers if handlers else [logging.NullHandler()],
        force=True,  # Override any existing config
    )
    return logging.getLogger(__name__)


# Deferred logger initialization (called after DDP setup)
logger = None


def get_logger():
    global logger
    if logger is None:
        logger = setup_logging()
    return logger


# ============================================================================
# Weights & Biases Integration (Comprehensive Logging)
# ============================================================================

_wandb_run = None
_wandb_enabled = False
_global_step = 0


def get_git_info() -> dict:
    """Get git commit info for reproducibility."""
    import subprocess
    info = {}
    try:
        info['git_commit'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=project_root, stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        info['git_branch'] = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=project_root, stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        info['git_dirty'] = len(subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=project_root, stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()) > 0
        # Get diff if dirty
        if info['git_dirty']:
            diff = subprocess.check_output(
                ['git', 'diff', '--stat'], cwd=project_root, stderr=subprocess.DEVNULL
            ).decode('utf-8').strip()
            info['git_diff_summary'] = diff[:1000] if len(diff) > 1000 else diff
    except Exception:
        info['git_commit'] = 'unknown'
        info['git_branch'] = 'unknown'
        info['git_dirty'] = False
    return info


def get_system_info() -> dict:
    """Get system info for reproducibility."""
    import platform
    info = {
        'python_version': platform.python_version(),
        'pytorch_version': torch.__version__,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'N/A',
        'cudnn_version': str(torch.backends.cudnn.version()) if torch.cuda.is_available() else 'N/A',
        'num_gpus': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'hostname': platform.node(),
        'platform': platform.platform(),
    }
    if torch.cuda.is_available():
        info['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        info['gpu_memory_gb'] = [
            round(torch.cuda.get_device_properties(i).total_memory / 1e9, 2) 
            for i in range(torch.cuda.device_count())
        ]
    return info


def init_wandb(
    project: str = "diffusion-tsf-7var",
    config: dict = None,
    resume: bool = False,
    tags: list = None,
) -> bool:
    """Initialize wandb with comprehensive logging (only on main process)."""
    global _wandb_run, _wandb_enabled, _global_step
    
    if not WANDB_AVAILABLE:
        logger.warning("wandb not installed. Run: pip install wandb")
        return False
    
    if not is_main_process():
        _wandb_enabled = False
        return False
    
    # Build comprehensive config
    full_config = {
        # Training constants
        'lookback_length': LOOKBACK_LENGTH,
        'forecast_length': FORECAST_LENGTH,
        'image_height': IMAGE_HEIGHT,
        'n_variates': N_VARIATES,
        'pretrain_epochs': PRETRAIN_EPOCHS,
        'pretrain_patience': PRETRAIN_PATIENCE,
        'finetune_epochs': FINETUNE_EPOCHS,
        'finetune_patience': FINETUNE_PATIENCE,
        'synthetic_samples_full': SYNTHETIC_SAMPLES_FULL,
        'synthetic_samples_hp_tune': SYNTHETIC_SAMPLES_HP_TUNE,
        'synthetic_samples_diff_tune': SYNTHETIC_SAMPLES_DIFF_TUNE,
        'n_itrans_hp_trials': N_ITRANS_HP_TRIALS,
        'n_diffusion_hp_trials': N_DIFFUSION_HP_TRIALS,
        'n_finetune_hp_trials': N_FINETUNE_HP_TRIALS,
        'itrans_batch_sizes': ITRANS_BATCH_SIZES,
        'diffusion_batch_sizes': DIFFUSION_BATCH_SIZES,
        'finetune_batch_sizes': FINETUNE_BATCH_SIZES,
        # DDP info
        'ddp_enabled': _ddp_enabled,
        'world_size': get_world_size(),
        # Directories
        'checkpoint_dir': CHECKPOINT_DIR,
        'results_dir': RESULTS_DIR,
        'datasets_dir': DATASETS_DIR,
    }
    
    # Add user config
    if config:
        full_config.update(config)
    
    # Add git info
    full_config.update(get_git_info())
    
    # Add system info
    full_config.update(get_system_info())
    
    # Handle resume
    run_id = None
    if resume:
        run_id_path = os.path.join(CHECKPOINT_DIR, 'wandb_run_id.txt')
        if os.path.exists(run_id_path):
            with open(run_id_path, 'r') as f:
                run_id = f.read().strip()
            logger.info(f"Resuming wandb run: {run_id}")
    
    # Default tags
    if tags is None:
        tags = ['7var-pipeline']
    if _ddp_enabled:
        tags.append(f'ddp-{get_world_size()}gpu')
    
    try:
        _wandb_run = wandb.init(
            project=project,
            config=full_config,
            resume="allow" if resume else None,
            id=run_id,
            reinit=True,
            tags=tags,
            save_code=True,  # Save code for reproducibility
        )
        
        # Save run ID for resume
        if _wandb_run:
            run_id_path = os.path.join(CHECKPOINT_DIR, 'wandb_run_id.txt')
            os.makedirs(os.path.dirname(run_id_path), exist_ok=True)
            with open(run_id_path, 'w') as f:
                f.write(_wandb_run.id)
            
            # Log config files as artifacts
            artifact = wandb.Artifact('config-files', type='config')
            # Add this script
            artifact.add_file(__file__)
            # Add config.py if exists
            config_path = os.path.join(script_dir, 'config.py')
            if os.path.exists(config_path):
                artifact.add_file(config_path)
            _wandb_run.log_artifact(artifact)
            
            logger.info(f"wandb initialized: {_wandb_run.url}")
        
        _wandb_enabled = True
        _global_step = 0
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize wandb: {e}")
        _wandb_enabled = False
        return False


def log_wandb(metrics: dict, step: int = None, commit: bool = True, prefix: str = None):
    """Log metrics to wandb with optional prefix."""
    global _global_step
    if not _wandb_enabled or not is_main_process() or _wandb_run is None:
        return
    
    if prefix:
        metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
    
    if step is None:
        step = _global_step
        _global_step += 1
    
    wandb.log(metrics, step=step, commit=commit)


def log_wandb_summary(metrics: dict):
    """Log summary metrics (shown in wandb dashboard)."""
    if not _wandb_enabled or not is_main_process() or _wandb_run is None:
        return
    for k, v in metrics.items():
        wandb.run.summary[k] = v


def log_wandb_hp_search(study_name: str, best_params: dict, best_value: float, n_trials: int):
    """Log hyperparameter search results."""
    if not _wandb_enabled or not is_main_process():
        return
    
    log_wandb({
        f'hp_search/{study_name}/best_value': best_value,
        f'hp_search/{study_name}/n_trials': n_trials,
        **{f'hp_search/{study_name}/best_{k}': v for k, v in best_params.items()}
    })
    
    # Also add to summary
    log_wandb_summary({
        f'{study_name}_best_value': best_value,
        **{f'{study_name}_best_{k}': v for k, v in best_params.items()}
    })


def log_wandb_model_checkpoint(path: str, name: str = None):
    """Log model checkpoint as artifact."""
    if not _wandb_enabled or not is_main_process() or _wandb_run is None:
        return
    
    if name is None:
        name = os.path.basename(os.path.dirname(path)) or 'checkpoint'
    
    try:
        artifact = wandb.Artifact(f'model-{name}', type='model')
        artifact.add_file(path)
        _wandb_run.log_artifact(artifact)
    except Exception as e:
        logger.warning(f"Failed to log checkpoint artifact: {e}")


def log_wandb_eval_results(subset_id: str, eval_results: dict, train_metrics: dict):
    """Log evaluation results for a subset."""
    if not _wandb_enabled or not is_main_process():
        return
    
    flat_metrics = {
        f'eval/{subset_id}/single_mse': eval_results['single']['mse'],
        f'eval/{subset_id}/single_mae': eval_results['single']['mae'],
        f'eval/{subset_id}/single_trend_acc': eval_results['single']['trend_accuracy'],
        f'eval/{subset_id}/avg_mse': eval_results['averaged']['mse'],
        f'eval/{subset_id}/avg_mae': eval_results['averaged']['mae'],
        f'eval/{subset_id}/avg_trend_acc': eval_results['averaged']['trend_accuracy'],
        f'eval/{subset_id}/best_val_loss': train_metrics.get('best_val_loss', 0),
        f'eval/{subset_id}/final_epoch': train_metrics.get('final_epoch', 0),
    }
    log_wandb(flat_metrics)
    
    # Table for comparison
    if hasattr(wandb, 'Table'):
        table_data = [[
            subset_id,
            eval_results['single']['mse'],
            eval_results['single']['mae'],
            eval_results['averaged']['mse'],
            eval_results['averaged']['mae'],
            train_metrics.get('best_val_loss', 0),
        ]]
        table = wandb.Table(
            columns=['subset', 'single_mse', 'single_mae', 'avg_mse', 'avg_mae', 'val_loss'],
            data=table_data
        )
        log_wandb({f'eval_table/{subset_id}': table})


def finish_wandb():
    """Finish wandb run and upload final artifacts."""
    global _wandb_run, _wandb_enabled
    if _wandb_run is not None and is_main_process():
        # Log final manifest as artifact
        if os.path.exists(MANIFEST_PATH):
            try:
                artifact = wandb.Artifact('training-manifest', type='metadata')
                artifact.add_file(MANIFEST_PATH)
                _wandb_run.log_artifact(artifact)
            except Exception:
                pass
        
        wandb.finish()
        _wandb_run = None
        _wandb_enabled = False

# ============================================================================
# Constants
# ============================================================================

DATASETS_DIR = os.path.join(project_root, 'datasets')
CHECKPOINT_DIR = os.path.join(script_dir, 'checkpoints_7var')
MANIFEST_PATH = os.path.join(CHECKPOINT_DIR, 'training_manifest.json')
RESULTS_DIR = os.path.join(script_dir, 'results_7var')

# Training settings
LOOKBACK_LENGTH = 512
FORECAST_LENGTH = 96
IMAGE_HEIGHT = 64
N_VARIATES = 7

# Phase 1: Synthetic pretraining
PRETRAIN_EPOCHS = 200
PRETRAIN_PATIENCE = 20
SYNTHETIC_SAMPLES_FULL = 1000000
SYNTHETIC_SAMPLES_HP_TUNE = 100000  # For iTransformer HP tuning
SYNTHETIC_SAMPLES_DIFF_TUNE = 10000  # For Diffusion HP tuning (smaller for speed)

# Phase 2: Fine-tuning
FINETUNE_EPOCHS = 200
FINETUNE_PATIENCE = 25
HP_TUNE_EPOCHS = 200
HP_TUNE_PATIENCE = 20

# Optuna settings
N_ITRANS_HP_TRIALS = 20
N_DIFFUSION_HP_TRIALS = 8
N_FINETUNE_HP_TRIALS = 8

# Batch size ranges for A6000 (48GB)
ITRANS_BATCH_SIZES = [64, 128, 256]
DIFFUSION_BATCH_SIZES = [32, 64, 96]  # 64x96x7 images
FINETUNE_BATCH_SIZES = [16, 32, 64]

# Dataset registry: name -> (path, date_col, seasonal_period)
DATASET_REGISTRY = {
    'ETTh1': ('ETT-small/ETTh1.csv', 'date', 24),
    'ETTh2': ('ETT-small/ETTh2.csv', 'date', 24),
    'ETTm1': ('ETT-small/ETTm1.csv', 'date', 96),
    'ETTm2': ('ETT-small/ETTm2.csv', 'date', 96),
    'illness': ('illness/national_illness.csv', 'date', 52),
    'exchange_rate': ('exchange_rate/exchange_rate.csv', 'date', 5),
    'weather': ('weather/weather.csv', 'date', 144),
    'electricity': ('electricity/electricity.csv', 'date', 96),
    'traffic': ('traffic/traffic.csv', 'date', 24),
}

# ============================================================================
# iTransformer Model Creation
# ============================================================================

def get_itransformer_class():
    """Dynamically load iTransformer model class."""
    itrans_path = os.path.join(script_dir, '..', 'iTransformer', 'model', 'iTransformer.py')
    itrans_path = os.path.abspath(itrans_path)
    
    # Add iTransformer directory to path for internal imports
    itrans_dir = os.path.join(script_dir, '..', 'iTransformer')
    itrans_dir = os.path.abspath(itrans_dir)
    if itrans_dir not in sys.path:
        sys.path.insert(0, itrans_dir)
    
    spec = importlib.util.spec_from_file_location("iTransformer_module", itrans_path)
    itrans_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(itrans_module)
    return itrans_module.Model


def create_itransformer_config(
    seq_len: int = LOOKBACK_LENGTH,
    pred_len: int = FORECAST_LENGTH,
    num_vars: int = N_VARIATES,
    d_model: int = 512,
    d_ff: int = 512,
    e_layers: int = 4,
    n_heads: int = 8,
    dropout: float = 0.1,
):
    """Create iTransformer config object."""
    class iTransConfig:
        def __init__(self):
            self.seq_len = seq_len
            self.pred_len = pred_len
            self.output_attention = False
            self.use_norm = True
            self.d_model = d_model
            self.d_ff = d_ff
            self.e_layers = e_layers
            self.n_heads = n_heads
            self.dropout = dropout
            self.activation = 'gelu'
            self.embed = 'fixed'
            self.freq = 'h'
            self.factor = 1
            self.enc_in = num_vars
            self.class_strategy = 'projection'
    return iTransConfig()


def create_itransformer(
    seq_len: int = LOOKBACK_LENGTH,
    pred_len: int = FORECAST_LENGTH,
    num_vars: int = N_VARIATES,
    dropout: float = 0.1,
) -> nn.Module:
    """Create iTransformer model."""
    iTransformerModel = get_itransformer_class()
    config = create_itransformer_config(
        seq_len=seq_len, pred_len=pred_len, num_vars=num_vars, dropout=dropout
    )
    return iTransformerModel(config)


# ============================================================================
# Diffusion Model Creation (with guidance support)
# ============================================================================

def create_diffusion_model(
    n_variates: int = N_VARIATES,
    lookback: int = LOOKBACK_LENGTH,
    horizon: int = FORECAST_LENGTH,
    use_guidance: bool = True,
) -> DiffusionTSF:
    """Create DiffusionTSF model with optional guidance channel."""
    config = DiffusionTSFConfig(
        num_variables=n_variates,
        lookback_length=lookback,
        forecast_length=horizon,
        image_height=IMAGE_HEIGHT,
        representation_mode='cdf',
        use_coordinate_channel=True,
        use_guidance_channel=use_guidance,
        use_hybrid_condition=True,
        num_diffusion_steps=1000,
        unet_channels=[64, 128, 256],
        attention_levels=[2],
        num_res_blocks=2,
    )
    return DiffusionTSF(config)


# ============================================================================
# Dataset Classes
# ============================================================================

class TimeSeriesDataset(Dataset):
    """Dataset for multivariate time series forecasting."""
    
    def __init__(
        self,
        data: np.ndarray,
        lookback: int = 512,
        horizon: int = 96,
        stride: int = 1,
    ):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.lookback = lookback
        self.horizon = horizon
        self.stride = stride
        total_len = lookback + horizon
        self.n_samples = max(0, (len(data) - total_len) // stride + 1)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        start = idx * self.stride
        past = self.data[start:start + self.lookback].T
        future = self.data[start + self.lookback:start + self.lookback + self.horizon].T
        return past, future


def load_dataset(
    dataset_name: str,
    variate_indices: List[int] = None,
    lookback: int = LOOKBACK_LENGTH,
    horizon: int = FORECAST_LENGTH,
    stride: int = 1,
) -> Tuple[Dataset, Dataset, Dataset, Dict]:
    """Load dataset and return train/val/test splits."""
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
    
    # Chronological split: 70/10/20
    n = len(data)
    train_end = int(n * 0.7)
    val_end = int(n * 0.8)
    
    train_ds = TimeSeriesDataset(data[:train_end], lookback, horizon, stride)
    val_ds = TimeSeriesDataset(data[train_end:val_end], lookback, horizon, stride=lookback)
    test_ds = TimeSeriesDataset(data[val_end:], lookback, horizon, stride=lookback)
    
    return train_ds, val_ds, test_ds, {'mean': mean, 'std': std}


# ============================================================================
# Variate Subset Management
# ============================================================================

def generate_variate_subsets(dataset_name: str, n_variates: int = 7, seed: int = 42) -> List[Dict]:
    """Generate non-overlapping 7-variate subsets for a dataset."""
    path = os.path.join(DATASETS_DIR, DATASET_REGISTRY[dataset_name][0])
    df = pd.read_csv(path, nrows=1)
    date_col = DATASET_REGISTRY[dataset_name][1]
    all_cols = [c for c in df.columns if c != date_col]
    n_total = len(all_cols)
    
    if n_total <= n_variates:
        indices = list(range(n_total))
        while len(indices) < n_variates:
            indices.append(indices[len(indices) % n_total])
        return [{'subset_id': dataset_name, 'variate_indices': indices, 'variate_names': [all_cols[i % n_total] for i in indices]}]
    
    rng = random.Random(seed)
    shuffled = list(range(n_total))
    rng.shuffle(shuffled)
    
    subsets = []
    for i in range(n_total // n_variates):
        start = i * n_variates
        indices = shuffled[start:start + n_variates]
        subsets.append({
            'subset_id': f"{dataset_name}-{i}" if n_total > n_variates else dataset_name,
            'variate_indices': indices,
            'variate_names': [all_cols[j] for j in indices],
        })
    return subsets


def generate_all_subsets(seed: int = 42) -> Dict[str, List[Dict]]:
    """Generate all variate subsets for all datasets."""
    return {name: generate_variate_subsets(name, seed=seed) for name in DATASET_REGISTRY}


# ============================================================================
# Training Manifest
# ============================================================================

@dataclass
class TrainingManifest:
    """Tracks training progress for resumability."""
    seed: int = 42
    created_at: str = ""
    
    # Phase 1 status
    itrans_hp_done: bool = False
    itrans_best_params: Dict = field(default_factory=dict)
    diffusion_hp_done: bool = False
    diffusion_best_params: Dict = field(default_factory=dict)
    pretrain_complete: bool = False
    pretrain_checkpoint: str = ""
    itrans_checkpoint: str = ""
    
    # Phase 2 status
    subsets: Dict = field(default_factory=dict)
    
    def save(self, path: str = MANIFEST_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str = MANIFEST_PATH) -> 'TrainingManifest':
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            m = cls()
            for k, v in data.items():
                if hasattr(m, k):
                    setattr(m, k, v)
            return m
        return cls(created_at=datetime.now().isoformat())
    
    def mark_complete(self, subset_id: str, checkpoint_path: str, metrics: Dict):
        self.subsets[subset_id] = {
            'status': 'complete',
            'checkpoint': checkpoint_path,
            'metrics': metrics,
            'completed_at': datetime.now().isoformat(),
        }
        self.save()


# ============================================================================
# Early Stopping & Checkpointing
# ============================================================================

class EarlyStopping:
    def __init__(self, patience: int = 25, min_delta: float = 1e-4):
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


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, config, path, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'config': config,
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


# ============================================================================
# PHASE 1A: iTransformer HP Tuning
# ============================================================================

def train_itransformer_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    """Train iTransformer for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for past, future in loader:
        # past: (B, vars, seq_len), future: (B, vars, pred_len)
        # iTransformer expects: (B, seq_len, vars)
        x_enc = past.permute(0, 2, 1).to(device)
        y_true = future.permute(0, 2, 1).to(device)
        
        optimizer.zero_grad()
        # iTransformer forward: (x_enc, x_mark_enc, x_dec, x_mark_dec)
        y_pred = model(x_enc, None, None, None)
        loss = criterion(y_pred, y_true)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


def validate_itransformer(model, loader, criterion, device):
    """Validate iTransformer."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for past, future in loader:
            x_enc = past.permute(0, 2, 1).to(device)
            y_true = future.permute(0, 2, 1).to(device)
            y_pred = model(x_enc, None, None, None)
            loss = criterion(y_pred, y_true)
            total_loss += loss.item()
            n_batches += 1
    
    return total_loss / max(n_batches, 1)


def itrans_hp_objective(trial, synthetic_loader, val_loader, device, smoke_test=False):
    """Optuna objective for iTransformer HP search."""
    # Suggest hyperparameters
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128] if smoke_test else ITRANS_BATCH_SIZES)
    dropout = trial.suggest_float('dropout', 0.0, 0.3)
    
    # Create model
    model = create_itransformer(dropout=dropout).to(device)
    
    # Rebuild loaders with new batch size
    train_loader = DataLoader(synthetic_loader.dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    epochs = 30 if not smoke_test else 1
    patience = 5 if not smoke_test else 1
    early_stop = EarlyStopping(patience=patience)
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        train_itransformer_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate_itransformer(model, val_loader, criterion, device)
        
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        if early_stop(val_loss):
            break
    
    return best_val_loss


def run_itransformer_hp_tuning(n_trials: int, smoke_test: bool = False) -> Dict:
    """Run Optuna HP search for iTransformer."""
    logger.info("=" * 60)
    logger.info("PHASE 1A: iTransformer HP Tuning")
    logger.info(f"Trials: {n_trials}")
    logger.info("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create synthetic data loaders
    n_samples = 16 if smoke_test else SYNTHETIC_SAMPLES_HP_TUNE
    synthetic_loader = get_synthetic_dataloader(
        batch_size=64,
        lookback_length=LOOKBACK_LENGTH,
        forecast_length=FORECAST_LENGTH,
        num_variables=N_VARIATES,
        num_samples=n_samples,
        num_workers=0,
    )
    
    # Split for validation
    dataset = synthetic_loader.dataset
    n_val = min(len(dataset) // 10, 1000)
    train_indices = list(range(len(dataset) - n_val))
    val_indices = list(range(len(dataset) - n_val, len(dataset)))
    
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False, num_workers=0)
    
    # Run Optuna (shared study in parallel mode)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = create_shared_study('itrans_hp_tuning')
    
    # In parallel mode, each worker runs n_trials / n_workers
    n_workers = int(os.environ.get('SLURM_GPUS_ON_NODE', 1)) if is_parallel_mode() else 1
    trials_per_worker = max(1, n_trials // n_workers)
    
    study.optimize(
        lambda trial: itrans_hp_objective(trial, train_loader, val_loader, device, smoke_test),
        n_trials=trials_per_worker,
        show_progress_bar=is_worker_zero(),  # Only worker 0 shows progress
    )
    
    # Wait for all workers to finish their trials
    if is_parallel_mode():
        parallel_worker_barrier()
    
    best_params = study.best_params
    logger.info(f"Best iTransformer params: lr={best_params['learning_rate']:.2e}, "
               f"bs={best_params['batch_size']}, dropout={best_params['dropout']:.3f}")
    logger.info(f"Best val loss: {study.best_value:.4f}")
    
    return best_params


# ============================================================================
# PHASE 1B: Diffusion HP Tuning (with iTransformer guidance)
# ============================================================================

def diffusion_hp_objective(
    trial, 
    synthetic_loader, 
    val_loader,
    itrans_guidance: iTransformerGuidance,
    device, 
    smoke_test=False
):
    """Optuna objective for Diffusion HP search."""
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64] if smoke_test else DIFFUSION_BATCH_SIZES)
    
    # Create model with guidance
    model = create_diffusion_model(use_guidance=True).to(device)
    model.set_guidance_model(itrans_guidance)
    
    # Rebuild loader with new batch size
    train_loader = DataLoader(synthetic_loader.dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    epochs = 30 if not smoke_test else 1
    patience = 10 if not smoke_test else 1
    early_stop = EarlyStopping(patience=patience)
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Train
        model.train()
        for past, future in train_loader:
            past, future = past.to(device), future.to(device)
            optimizer.zero_grad()
            loss = model.get_loss(past, future)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        # Validate
        model.eval()
        val_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for past, future in val_loader:
                past, future = past.to(device), future.to(device)
                loss = model.get_loss(past, future)
                val_loss += loss.item()
                n_batches += 1
        val_loss /= max(n_batches, 1)
        
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        if early_stop(val_loss):
            break
    
    return best_val_loss


def run_diffusion_hp_tuning(
    itrans_checkpoint: str,
    n_trials: int,
    smoke_test: bool = False
) -> Dict:
    """Run Optuna HP search for Diffusion model."""
    logger.info("=" * 60)
    logger.info("PHASE 1B: Diffusion HP Tuning (with iTransformer guidance)")
    logger.info(f"Trials: {n_trials}")
    logger.info("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load iTransformer as guidance
    itrans_model = create_itransformer().to(device)
    ckpt = torch.load(itrans_checkpoint, map_location=device, weights_only=False)
    itrans_model.load_state_dict(ckpt['model_state_dict'])
    itrans_guidance = iTransformerGuidance(
        model=itrans_model,
        use_norm=True,
        seq_len=LOOKBACK_LENGTH,
        pred_len=FORECAST_LENGTH
    )
    
    # Create small synthetic dataset for fast iteration
    n_samples = 16 if smoke_test else SYNTHETIC_SAMPLES_DIFF_TUNE
    synthetic_loader = get_synthetic_dataloader(
        batch_size=32,
        lookback_length=LOOKBACK_LENGTH,
        forecast_length=FORECAST_LENGTH,
        num_variables=N_VARIATES,
        num_samples=n_samples,
        num_workers=0,
    )
    
    dataset = synthetic_loader.dataset
    n_val = min(len(dataset) // 10, 500)
    train_subset = Subset(dataset, list(range(len(dataset) - n_val)))
    val_subset = Subset(dataset, list(range(len(dataset) - n_val, len(dataset))))
    
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=0)
    
    # Run Optuna (shared study in parallel mode)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = create_shared_study('diffusion_hp_tuning')
    
    # In parallel mode, each worker runs n_trials / n_workers
    n_workers = int(os.environ.get('SLURM_GPUS_ON_NODE', 1)) if is_parallel_mode() else 1
    trials_per_worker = max(1, n_trials // n_workers)
    
    study.optimize(
        lambda trial: diffusion_hp_objective(trial, train_loader, val_loader, itrans_guidance, device, smoke_test),
        n_trials=trials_per_worker,
        show_progress_bar=is_worker_zero(),
    )
    
    # Wait for all workers to finish
    if is_parallel_mode():
        parallel_worker_barrier()
    
    best_params = study.best_params
    logger.info(f"Best Diffusion params: lr={best_params['learning_rate']:.2e}, bs={best_params['batch_size']}")
    logger.info(f"Best val loss: {study.best_value:.4f}")
    
    return best_params


# ============================================================================
# PHASE 1C: Full Pretraining
# ============================================================================

def pretrain_itransformer(
    best_params: Dict,
    n_samples: int,
    epochs: int,
    patience: int,
    checkpoint_dir: str,
    smoke_test: bool = False,
) -> str:
    """Train iTransformer on synthetic data with tuned params (DDP-aware)."""
    logger.info("=" * 60)
    logger.info("PHASE 1C-1: Full iTransformer Pretraining")
    logger.info(f"Samples: {n_samples}, Epochs: {epochs}, Patience: {patience}")
    logger.info(f"Params: {best_params}")
    if _ddp_enabled:
        logger.info(f"DDP: {get_world_size()} GPUs")
    logger.info("=" * 60)
    
    device = get_device()
    
    lr = best_params.get('learning_rate', 1e-4)
    batch_size = best_params.get('batch_size', 64)
    dropout = best_params.get('dropout', 0.1)
    
    # Effective batch size scales with world size
    effective_batch_size = batch_size // get_world_size() if _ddp_enabled else batch_size
    effective_batch_size = max(1, effective_batch_size)
    
    # Create data
    synthetic_loader = get_synthetic_dataloader(
        batch_size=effective_batch_size,
        lookback_length=LOOKBACK_LENGTH,
        forecast_length=FORECAST_LENGTH,
        num_variables=N_VARIATES,
        num_samples=n_samples,
        num_workers=0 if smoke_test else 4,
    )
    
    # Split for validation
    dataset = synthetic_loader.dataset
    n_val = min(len(dataset) // 10, 5000)
    train_subset = Subset(dataset, list(range(len(dataset) - n_val)))
    val_subset = Subset(dataset, list(range(len(dataset) - n_val, len(dataset))))
    
    # Use DDP-aware data loaders
    train_loader, train_sampler = create_dataloader_ddp(
        train_subset, effective_batch_size, shuffle=True,
        num_workers=0 if smoke_test else 4
    )
    val_loader, _ = create_dataloader_ddp(
        val_subset, effective_batch_size, shuffle=False, num_workers=0
    )
    
    # Create and wrap model with DDP
    model = create_itransformer(dropout=dropout)
    model = wrap_model_ddp(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    criterion = nn.MSELoss()
    
    early_stop = EarlyStopping(patience=patience)
    best_val_loss = float('inf')
    ckpt_path = os.path.join(checkpoint_dir, 'pretrained_itransformer.pt')
    
    for epoch in range(epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)  # Crucial for DDP shuffling
        
        t0 = time.time()
        train_loss = train_itransformer_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate_itransformer(model, val_loader, criterion, device)
        
        # Average loss across GPUs for consistent logging
        if _ddp_enabled:
            train_loss_t = torch.tensor([train_loss], device=device)
            val_loss_t = torch.tensor([val_loss], device=device)
            train_loss = sync_across_processes(train_loss_t).item()
            val_loss = sync_across_processes(val_loss_t).item()
        
        scheduler.step()
        
        logger.info(f"[iTransformer] Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | "
                   f"Val: {val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e} | Time: {time.time()-t0:.1f}s")
        
        # Wandb logging
        log_wandb({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': scheduler.get_last_lr()[0],
            'epoch': epoch + 1,
            'epoch_time_s': time.time() - t0,
        }, prefix='itrans_pretrain')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Only main process saves checkpoint
            if is_main_process():
                save_checkpoint(unwrap_model(model), optimizer, epoch, train_loss, val_loss, best_params, ckpt_path)
                logger.info(f"  -> New best! Saved to {ckpt_path}")
            barrier()  # Sync before continuing
        
        if early_stop(val_loss):
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    barrier()  # Ensure all processes finish
    logger.info(f"iTransformer pretraining complete. Best val loss: {best_val_loss:.4f}")
    log_wandb_summary({'itrans_pretrain_best_val_loss': best_val_loss})
    return ckpt_path


def pretrain_diffusion(
    best_params: Dict,
    itrans_checkpoint: str,
    n_samples: int,
    epochs: int,
    patience: int,
    checkpoint_dir: str,
    smoke_test: bool = False,
) -> str:
    """Train Diffusion model on synthetic data with iTransformer guidance (DDP-aware)."""
    logger.info("=" * 60)
    logger.info("PHASE 1C-2: Full Diffusion Pretraining (with iTransformer guidance)")
    logger.info(f"Samples: {n_samples}, Epochs: {epochs}, Patience: {patience}")
    logger.info(f"Params: {best_params}")
    if _ddp_enabled:
        logger.info(f"DDP: {get_world_size()} GPUs")
    logger.info("=" * 60)
    
    device = get_device()
    
    lr = best_params.get('learning_rate', 1e-4)
    batch_size = best_params.get('batch_size', 64)
    
    # Effective batch size scales with world size
    effective_batch_size = batch_size // get_world_size() if _ddp_enabled else batch_size
    effective_batch_size = max(1, effective_batch_size)
    
    # Load iTransformer as guidance (not wrapped in DDP - used in eval mode only)
    itrans_model = create_itransformer().to(device)
    ckpt = torch.load(itrans_checkpoint, map_location=device, weights_only=False)
    itrans_model.load_state_dict(ckpt['model_state_dict'])
    itrans_guidance = iTransformerGuidance(
        model=itrans_model,
        use_norm=True,
        seq_len=LOOKBACK_LENGTH,
        pred_len=FORECAST_LENGTH
    )
    
    # Create data
    synthetic_loader = get_synthetic_dataloader(
        batch_size=effective_batch_size,
        lookback_length=LOOKBACK_LENGTH,
        forecast_length=FORECAST_LENGTH,
        num_variables=N_VARIATES,
        num_samples=n_samples,
        num_workers=0 if smoke_test else 4,
    )
    
    dataset = synthetic_loader.dataset
    n_val = min(len(dataset) // 10, 5000)
    train_subset = Subset(dataset, list(range(len(dataset) - n_val)))
    val_subset = Subset(dataset, list(range(len(dataset) - n_val, len(dataset))))
    
    # Use DDP-aware data loaders
    train_loader, train_sampler = create_dataloader_ddp(
        train_subset, effective_batch_size, shuffle=True,
        num_workers=0 if smoke_test else 4
    )
    val_loader, _ = create_dataloader_ddp(
        val_subset, effective_batch_size, shuffle=False, num_workers=0
    )
    
    # Create model with guidance and wrap with DDP
    model = create_diffusion_model(use_guidance=True)
    model.set_guidance_model(itrans_guidance)
    model = wrap_model_ddp(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    
    early_stop = EarlyStopping(patience=patience)
    best_val_loss = float('inf')
    ckpt_path = os.path.join(checkpoint_dir, 'pretrained_diffusion.pt')
    
    for epoch in range(epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        t0 = time.time()
        
        # Train
        model.train()
        total_loss = 0.0
        n_batches = 0
        for past, future in train_loader:
            past, future = past.to(device), future.to(device)
            optimizer.zero_grad()
            # Handle DDP wrapper
            base_model = unwrap_model(model)
            loss = base_model.get_loss(past, future)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        train_loss = total_loss / max(n_batches, 1)
        
        # Validate
        model.eval()
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for past, future in val_loader:
                past, future = past.to(device), future.to(device)
                base_model = unwrap_model(model)
                loss = base_model.get_loss(past, future)
                total_loss += loss.item()
                n_batches += 1
        val_loss = total_loss / max(n_batches, 1)
        
        # Average loss across GPUs
        if _ddp_enabled:
            train_loss_t = torch.tensor([train_loss], device=device)
            val_loss_t = torch.tensor([val_loss], device=device)
            train_loss = sync_across_processes(train_loss_t).item()
            val_loss = sync_across_processes(val_loss_t).item()
        
        scheduler.step()
        
        logger.info(f"[Diffusion] Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | "
                   f"Val: {val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e} | Time: {time.time()-t0:.1f}s")
        
        # Wandb logging
        log_wandb({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': scheduler.get_last_lr()[0],
            'epoch': epoch + 1,
            'epoch_time_s': time.time() - t0,
        }, prefix='diffusion_pretrain')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if is_main_process():
                save_checkpoint(unwrap_model(model), optimizer, epoch, train_loss, val_loss, 
                              {'diffusion_params': best_params, 'itrans_checkpoint': itrans_checkpoint}, ckpt_path)
                logger.info(f"  -> New best! Saved to {ckpt_path}")
            barrier()
        
        if early_stop(val_loss):
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    barrier()
    logger.info(f"Diffusion pretraining complete. Best val loss: {best_val_loss:.4f}")
    log_wandb_summary({'diffusion_pretrain_best_val_loss': best_val_loss})
    log_wandb_model_checkpoint(ckpt_path, 'pretrained_diffusion')
    return ckpt_path


# ============================================================================
# PHASE 2: Fine-tuning HP Search & Training
# ============================================================================

def finetune_hp_objective(
    trial,
    dataset_name: str,
    variate_indices: List[int],
    pretrained_path: str,
    itrans_checkpoint: str,
    device: torch.device,
    smoke_test: bool = False,
) -> float:
    """Optuna objective for fine-tuning HP search (lr and batch_size only)."""
    lr = trial.suggest_float('learning_rate', 1e-6, 1e-4, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32] if smoke_test else FINETUNE_BATCH_SIZES)
    
    # Load data
    train_ds, val_ds, _, _ = load_dataset(
        dataset_name, variate_indices,
        stride=24 if not smoke_test else LOOKBACK_LENGTH,
    )
    
    if smoke_test:
        train_ds = Subset(train_ds, list(range(min(2, len(train_ds)))))
        val_ds = Subset(val_ds, list(range(min(2, len(val_ds)))))
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Load iTransformer guidance
    itrans_model = create_itransformer().to(device)
    ckpt = torch.load(itrans_checkpoint, map_location=device, weights_only=False)
    itrans_model.load_state_dict(ckpt['model_state_dict'])
    itrans_guidance = iTransformerGuidance(itrans_model, use_norm=True, seq_len=LOOKBACK_LENGTH, pred_len=FORECAST_LENGTH)
    
    # Load pretrained diffusion
    model = create_diffusion_model(use_guidance=True).to(device)
    model.set_guidance_model(itrans_guidance)
    ckpt = torch.load(pretrained_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    epochs = HP_TUNE_EPOCHS if not smoke_test else 1
    patience = HP_TUNE_PATIENCE if not smoke_test else 1
    early_stop = EarlyStopping(patience=patience)
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        for past, future in train_loader:
            past, future = past.to(device), future.to(device)
            optimizer.zero_grad()
            loss = model.get_loss(past, future)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        model.eval()
        val_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for past, future in val_loader:
                past, future = past.to(device), future.to(device)
                loss = model.get_loss(past, future)
                val_loss += loss.item()
                n_batches += 1
        val_loss /= max(n_batches, 1)
        
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        if early_stop(val_loss):
            break
    
    return best_val_loss


def finetune_on_dataset(
    subset_info: Dict,
    pretrained_path: str,
    itrans_checkpoint: str,
    tuned_params: Dict,
    epochs: int = FINETUNE_EPOCHS,
    patience: int = FINETUNE_PATIENCE,
    checkpoint_dir: str = CHECKPOINT_DIR,
    smoke_test: bool = False,
) -> Tuple[str, Dict]:
    """Fine-tune on a dataset with tuned params (DDP-aware)."""
    subset_id = subset_info['subset_id']
    variate_indices = subset_info['variate_indices']
    
    if '-' in subset_id and subset_id.split('-')[-1].isdigit():
        dataset_name = '-'.join(subset_id.split('-')[:-1])
    else:
        dataset_name = subset_id
    
    lr = tuned_params.get('learning_rate', 1e-5)
    batch_size = tuned_params.get('batch_size', 32)
    
    # Effective batch size for DDP
    effective_batch_size = batch_size // get_world_size() if _ddp_enabled else batch_size
    effective_batch_size = max(1, effective_batch_size)
    
    logger.info("=" * 60)
    logger.info(f"FINE-TUNING: {subset_id}")
    logger.info(f"Params: lr={lr:.2e}, batch_size={batch_size}" + 
                (f" (effective={effective_batch_size} per GPU)" if _ddp_enabled else ""))
    logger.info("=" * 60)
    
    device = get_device()
    
    # Load data
    train_ds, val_ds, _, norm_stats = load_dataset(
        dataset_name, variate_indices,
        stride=24 if not smoke_test else LOOKBACK_LENGTH,
    )
    
    if smoke_test:
        train_ds = Subset(train_ds, list(range(min(2, len(train_ds)))))
        val_ds = Subset(val_ds, list(range(min(2, len(val_ds)))))
    
    # DDP-aware data loaders
    train_loader, train_sampler = create_dataloader_ddp(
        train_ds, effective_batch_size, shuffle=True, num_workers=0
    )
    val_loader, _ = create_dataloader_ddp(
        val_ds, effective_batch_size, shuffle=False, num_workers=0
    )
    
    # Load iTransformer guidance (not wrapped - eval mode only)
    itrans_model = create_itransformer().to(device)
    ckpt = torch.load(itrans_checkpoint, map_location=device, weights_only=False)
    itrans_model.load_state_dict(ckpt['model_state_dict'])
    itrans_guidance = iTransformerGuidance(itrans_model, use_norm=True, seq_len=LOOKBACK_LENGTH, pred_len=FORECAST_LENGTH)
    
    # Load pretrained diffusion and wrap with DDP
    model = create_diffusion_model(use_guidance=True)
    model.set_guidance_model(itrans_guidance)
    ckpt = torch.load(pretrained_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model = wrap_model_ddp(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    
    early_stop = EarlyStopping(patience=patience)
    best_val_loss = float('inf')
    
    subset_dir = os.path.join(checkpoint_dir, subset_id)
    if is_main_process():
        os.makedirs(subset_dir, exist_ok=True)
    barrier()
    best_ckpt_path = os.path.join(subset_dir, 'best.pt')
    
    # Save metadata (main process only)
    if is_main_process():
        with open(os.path.join(subset_dir, 'metadata.json'), 'w') as f:
            json.dump({
                'subset_id': subset_id,
                'dataset_name': dataset_name,
                'variate_indices': variate_indices,
                'variate_names': subset_info.get('variate_names', []),
                'norm_mean': norm_stats['mean'].tolist(),
                'norm_std': norm_stats['std'].tolist(),
                'tuned_params': tuned_params,
            }, f, indent=2)
    
    final_epoch = 0
    for epoch in range(epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        final_epoch = epoch
        t0 = time.time()
        
        model.train()
        total_loss = 0.0
        n_batches = 0
        for past, future in train_loader:
            past, future = past.to(device), future.to(device)
            optimizer.zero_grad()
            base_model = unwrap_model(model)
            loss = base_model.get_loss(past, future)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        train_loss = total_loss / max(n_batches, 1)
        
        model.eval()
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for past, future in val_loader:
                past, future = past.to(device), future.to(device)
                base_model = unwrap_model(model)
                loss = base_model.get_loss(past, future)
                total_loss += loss.item()
                n_batches += 1
        val_loss = total_loss / max(n_batches, 1)
        
        # Average loss across GPUs
        if _ddp_enabled:
            train_loss_t = torch.tensor([train_loss], device=device)
            val_loss_t = torch.tensor([val_loss], device=device)
            train_loss = sync_across_processes(train_loss_t).item()
            val_loss = sync_across_processes(val_loss_t).item()
        
        scheduler.step()
        
        logger.info(f"[{subset_id}] Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | "
                   f"Val: {val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e} | Time: {time.time()-t0:.1f}s")
        
        # Wandb logging
        log_wandb({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': scheduler.get_last_lr()[0],
            'epoch': epoch + 1,
            'epoch_time_s': time.time() - t0,
        }, prefix=f'finetune/{subset_id}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if is_main_process():
                save_checkpoint(unwrap_model(model), optimizer, epoch, train_loss, val_loss, 
                              {'tuned_params': tuned_params}, best_ckpt_path)
                logger.info(f"  -> New best!")
            barrier()
        
        if early_stop(val_loss):
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    barrier()
    return best_ckpt_path, {'best_val_loss': best_val_loss, 'final_epoch': final_epoch + 1}


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_model(
    model: DiffusionTSF,
    test_loader: DataLoader,
    device: torch.device,
    n_samples: int = 30,
    smoke_test: bool = False,
) -> Dict:
    """Evaluate model on test set."""
    model.eval()
    
    all_preds_single = []
    all_preds_avg = []
    all_targets = []
    
    n_batches = min(1, len(test_loader)) if smoke_test else len(test_loader)
    
    with torch.no_grad():
        for batch_idx, (past, future) in enumerate(test_loader):
            if batch_idx >= n_batches:
                break
            
            past = past.to(device)
            
            # Single sample
            torch.manual_seed(42 + batch_idx)
            result = model.generate(past)
            all_preds_single.append(result['prediction'].cpu())
            
            # Averaged
            samples = []
            n_avg = n_samples if not smoke_test else 2
            for _ in range(n_avg):
                result = model.generate(past)
                samples.append(result['prediction'].cpu())
            all_preds_avg.append(torch.stack(samples).mean(dim=0))
            
            all_targets.append(future)
    
    preds_single = torch.cat(all_preds_single, dim=0)
    preds_avg = torch.cat(all_preds_avg, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    # Compute metrics
    def compute_metrics(pred, target):
        mse = torch.nn.functional.mse_loss(pred, target).item()
        mae = torch.nn.functional.l1_loss(pred, target).item()
        
        # Trend accuracy
        pred_diff = pred[:, :, 1:] - pred[:, :, :-1]
        target_diff = target[:, :, 1:] - target[:, :, :-1]
        trend_acc = ((pred_diff > 0) == (target_diff > 0)).float().mean().item()
        
        return {'mse': mse, 'mae': mae, 'trend_accuracy': trend_acc}
    
    return {
        'single': compute_metrics(preds_single, targets),
        'averaged': compute_metrics(preds_avg, targets),
    }


def save_eval_results(subset_id, dataset_name, variate_indices, train_metrics, eval_results, results_dir):
    """Save evaluation results."""
    os.makedirs(results_dir, exist_ok=True)
    
    result = {
        'subset_id': subset_id,
        'dataset': dataset_name,
        'variate_indices': variate_indices,
        'train_metrics': train_metrics,
        'eval_metrics': eval_results,
        'evaluated_at': datetime.now().isoformat(),
    }
    
    with open(os.path.join(results_dir, f'{subset_id}_results.json'), 'w') as f:
        json.dump(result, f, indent=2)
    
    # Update summary CSV
    update_summary_csv(results_dir)


def update_summary_csv(results_dir):
    """Update summary CSV."""
    rows = []
    for fname in os.listdir(results_dir):
        if fname.endswith('_results.json'):
            try:
                with open(os.path.join(results_dir, fname), 'r') as f:
                    data = json.load(f)
                if 'eval_metrics' not in data:
                    continue
                metrics = data['eval_metrics']
                rows.append({
                    'subset_id': data['subset_id'],
                    'dataset': data['dataset'],
                    'best_val_loss': data.get('train_metrics', {}).get('best_val_loss'),
                    'single_mse': metrics['single']['mse'],
                    'single_mae': metrics['single']['mae'],
                    'avg_mse': metrics['averaged']['mse'],
                    'avg_mae': metrics['averaged']['mae'],
                })
            except:
                continue
    
    if rows:
        df = pd.DataFrame(rows).sort_values(['dataset', 'subset_id'])
        df.to_csv(os.path.join(results_dir, 'summary.csv'), index=False)


# ============================================================================
# Traffic Recombination
# ============================================================================

def recombine_traffic_data():
    """Recombine traffic_part1.csv and traffic_part2.csv."""
    traffic_dir = os.path.join(DATASETS_DIR, 'traffic')
    combined_path = os.path.join(traffic_dir, 'traffic.csv')
    
    if os.path.exists(combined_path):
        logger.info("traffic.csv already exists")
        return
    
    part1 = os.path.join(traffic_dir, 'traffic_part1.csv')
    part2 = os.path.join(traffic_dir, 'traffic_part2.csv')
    
    if not os.path.exists(part1) or not os.path.exists(part2):
        logger.warning("Traffic part files not found")
        return
    
    logger.info("Recombining traffic data...")
    df = pd.concat([pd.read_csv(part1), pd.read_csv(part2)], ignore_index=True)
    df.to_csv(combined_path, index=False)
    logger.info(f"Created traffic.csv with {len(df)} rows")


# ============================================================================
# Main Pipeline
# ============================================================================

def run_pipeline(
    resume: bool = False, 
    smoke_test: bool = False, 
    seed: int = 42,
    use_wandb: bool = False,
    wandb_project: str = "diffusion-tsf-7var",
):
    """Run the full training pipeline."""
    # Seed with rank offset for DDP to ensure different data sampling
    effective_seed = seed + get_rank()
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    
    # Only main process handles data prep
    if is_main_process():
        recombine_traffic_data()
    barrier()  # Wait for data prep
    
    # Load or create manifest (only main process writes)
    if resume and os.path.exists(MANIFEST_PATH):
        manifest = TrainingManifest.load()
        logger.info(f"Resuming from manifest (created: {manifest.created_at})")
    else:
        manifest = TrainingManifest(seed=seed, created_at=datetime.now().isoformat())
    
    if is_main_process():
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    barrier()
    
    device = get_device()
    gpu_info = f"GPU {get_rank()}/{get_world_size()}" if _ddp_enabled else "single GPU"
    logger.info(f"Using device: {device} ({gpu_info})")
    
    # Initialize wandb
    if use_wandb:
        tags = ['smoke-test'] if smoke_test else []
        init_wandb(
            project=wandb_project,
            config={'seed': seed, 'smoke_test': smoke_test, 'resume': resume},
            resume=resume,
            tags=tags,
        )
    
    # Smoke test config
    if smoke_test:
        n_itrans_trials = 1
        n_diff_trials = 1
        n_finetune_trials = 1
        pretrain_samples = 16  # Ultra minimal
        pretrain_epochs = 1
        pretrain_patience = 1
        finetune_epochs = 1
        finetune_patience = 1
    else:
        n_itrans_trials = N_ITRANS_HP_TRIALS
        n_diff_trials = N_DIFFUSION_HP_TRIALS
        n_finetune_trials = N_FINETUNE_HP_TRIALS
        pretrain_samples = SYNTHETIC_SAMPLES_FULL
        pretrain_epochs = PRETRAIN_EPOCHS
        pretrain_patience = PRETRAIN_PATIENCE
        finetune_epochs = FINETUNE_EPOCHS
        finetune_patience = FINETUNE_PATIENCE
    
    # =========== PHASE 1A: iTransformer HP Tuning ===========
    # In parallel mode, all workers run HP trials concurrently
    # In DDP mode, only main process runs HP tuning
    if not manifest.itrans_hp_done:
        should_run_hp = is_parallel_mode() or is_main_process()
        if should_run_hp:
            manifest.itrans_best_params = run_itransformer_hp_tuning(n_itrans_trials, smoke_test)
            
            if is_worker_zero() or (not is_parallel_mode() and is_main_process()):
                manifest.itrans_hp_done = True
                manifest.save()
                # Log HP results to wandb
                log_wandb_hp_search('itransformer', manifest.itrans_best_params, 
                                   manifest.itrans_best_params.get('best_val_loss', 0), n_itrans_trials)
        
        barrier()  # DDP barrier
        if is_parallel_mode():
            parallel_worker_barrier()
            manifest = TrainingManifest.load()  # Reload with results
        elif not is_main_process():
            manifest = TrainingManifest.load()
    else:
        logger.info(f"Using cached iTransformer params: {manifest.itrans_best_params}")
    
    # =========== PHASE 1C-1: Full iTransformer Pretraining ===========
    # In parallel mode, only worker 0 trains (others wait for checkpoint to exist)
    itrans_ckpt = os.path.join(CHECKPOINT_DIR, 'pretrained_itransformer.pt')
    if not manifest.itrans_checkpoint or not os.path.exists(itrans_ckpt):
        if is_parallel_mode() and not is_worker_zero():
            logger.info(f"Worker {get_worker_id()}: Waiting for iTransformer training (worker 0)...")
            while not os.path.exists(itrans_ckpt):
                time.sleep(5)
            logger.info(f"Worker {get_worker_id()}: iTransformer checkpoint found, continuing")
        else:
            itrans_ckpt = pretrain_itransformer(
                manifest.itrans_best_params,
                n_samples=pretrain_samples,
                epochs=pretrain_epochs,
                patience=pretrain_patience,
                checkpoint_dir=CHECKPOINT_DIR,
                smoke_test=smoke_test,
            )
            manifest.itrans_checkpoint = itrans_ckpt
            if is_main_process() or is_worker_zero():
                manifest.save()
    else:
        logger.info(f"Using existing iTransformer checkpoint: {itrans_ckpt}")
    
    # =========== PHASE 1B: Diffusion HP Tuning ===========
    # In parallel mode, all workers run HP trials concurrently
    if not manifest.diffusion_hp_done:
        should_run_hp = is_parallel_mode() or is_main_process()
        if should_run_hp:
            manifest.diffusion_best_params = run_diffusion_hp_tuning(itrans_ckpt, n_diff_trials, smoke_test)
            
            if is_worker_zero() or (not is_parallel_mode() and is_main_process()):
                manifest.diffusion_hp_done = True
                manifest.save()
                # Log HP results to wandb
                log_wandb_hp_search('diffusion', manifest.diffusion_best_params,
                                   manifest.diffusion_best_params.get('best_val_loss', 0), n_diff_trials)
        
        barrier()  # DDP barrier
        if is_parallel_mode():
            parallel_worker_barrier()
            manifest = TrainingManifest.load()
        elif not is_main_process():
            manifest = TrainingManifest.load()
    else:
        logger.info(f"Using cached Diffusion params: {manifest.diffusion_best_params}")
    
    # =========== PHASE 1C-2: Full Diffusion Pretraining ===========
    # In parallel mode, only worker 0 does full pretraining (other workers exit after HP tuning)
    if is_parallel_mode() and not is_worker_zero():
        logger.info(f"Worker {get_worker_id()}: HP tuning complete, exiting (worker 0 will do full training)")
        return
    
    diff_ckpt = os.path.join(CHECKPOINT_DIR, 'pretrained_diffusion.pt')
    if not manifest.pretrain_complete or not os.path.exists(diff_ckpt):
        diff_ckpt = pretrain_diffusion(
            manifest.diffusion_best_params,
            itrans_ckpt,
            n_samples=pretrain_samples,
            epochs=pretrain_epochs,
            patience=pretrain_patience,
            checkpoint_dir=CHECKPOINT_DIR,
            smoke_test=smoke_test,
        )
        manifest.pretrain_checkpoint = diff_ckpt
        manifest.pretrain_complete = True
        if is_main_process():
            manifest.save()
    else:
        logger.info(f"Using existing Diffusion checkpoint: {diff_ckpt}")
    
    # =========== PHASE 2: Fine-tuning per Dataset ===========
    all_subsets = generate_all_subsets(seed=seed)
    subset_list = []
    for dataset_name, subsets in all_subsets.items():
        for subset in subsets:
            subset_list.append(subset)
            if subset['subset_id'] not in manifest.subsets:
                manifest.subsets[subset['subset_id']] = {
                    'status': 'pending',
                    'dataset': dataset_name,
                    'variate_indices': subset['variate_indices'],
                }
    if is_main_process():
        manifest.save()
    barrier()
    
    if smoke_test:
        subset_list = subset_list[:1]  # Just 1 dataset for ultra-fast smoke test
    
    for subset_info in subset_list:
        subset_id = subset_info['subset_id']
        variate_indices = subset_info['variate_indices']
        
        if '-' in subset_id and subset_id.split('-')[-1].isdigit():
            dataset_name = '-'.join(subset_id.split('-')[:-1])
        else:
            dataset_name = subset_id
        
        if manifest.subsets.get(subset_id, {}).get('status') == 'complete':
            logger.info(f"Skipping {subset_id} (already complete)")
            continue
        
        manifest.subsets[subset_id]['status'] = 'in_progress'
        if is_main_process():
            manifest.save()
        
        try:
            # HP Tuning for this dataset
            tuned_params = manifest.subsets[subset_id].get('tuned_params')
            if not tuned_params:
                logger.info(f"Running HP search for {subset_id}...")
                optuna.logging.set_verbosity(optuna.logging.WARNING)
                
                if is_parallel_mode():
                    # Parallel workers: all workers run trials concurrently
                    study = create_shared_study(f'finetune_{subset_id}')
                    n_workers = int(os.environ.get('SLURM_GPUS_ON_NODE', 1))
                    trials_per_worker = max(1, n_finetune_trials // n_workers)
                    
                    study.optimize(
                        lambda trial: finetune_hp_objective(
                            trial, dataset_name, variate_indices, diff_ckpt, itrans_ckpt, device, smoke_test
                        ),
                        n_trials=trials_per_worker,
                        show_progress_bar=is_worker_zero(),
                    )
                    parallel_worker_barrier()  # All workers sync
                    tuned_params = study.best_params
                    
                    # Only worker 0 saves to manifest
                    if is_worker_zero():
                        manifest.subsets[subset_id]['tuned_params'] = tuned_params
                        manifest.save()
                        logger.info(f"Best params for {subset_id}: {tuned_params}")
                    parallel_worker_barrier()  # Sync again before continuing
                    
                elif is_main_process():
                    # DDP mode: only rank 0 does HP search
                    study = create_shared_study(f'finetune_{subset_id}')
                    study.optimize(
                        lambda trial: finetune_hp_objective(
                            trial, dataset_name, variate_indices, diff_ckpt, itrans_ckpt, device, smoke_test
                        ),
                        n_trials=n_finetune_trials,
                        show_progress_bar=True,
                    )
                    tuned_params = study.best_params
                    manifest.subsets[subset_id]['tuned_params'] = tuned_params
                    manifest.save()
                    logger.info(f"Best params for {subset_id}: {tuned_params}")
                    
                barrier()  # DDP barrier
                if not is_main_process() and not is_parallel_mode():
                    manifest = TrainingManifest.load()
                    tuned_params = manifest.subsets[subset_id].get('tuned_params')
            
            # Full fine-tuning (all GPUs)
            ckpt_path, train_metrics = finetune_on_dataset(
                subset_info, diff_ckpt, itrans_ckpt, tuned_params,
                epochs=finetune_epochs, patience=finetune_patience,
                checkpoint_dir=CHECKPOINT_DIR, smoke_test=smoke_test,
            )
            
            # Evaluation (main process only - simpler and eval is fast)
            if is_main_process():
                logger.info(f"Evaluating {subset_id}...")
                itrans_model = create_itransformer().to(device)
                ckpt = torch.load(itrans_ckpt, map_location=device, weights_only=False)
                itrans_model.load_state_dict(ckpt['model_state_dict'])
                itrans_guidance = iTransformerGuidance(itrans_model, use_norm=True, seq_len=LOOKBACK_LENGTH, pred_len=FORECAST_LENGTH)
                
                model = create_diffusion_model(use_guidance=True).to(device)
                model.set_guidance_model(itrans_guidance)
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                model.load_state_dict(ckpt['model_state_dict'])
                
                _, _, test_ds, _ = load_dataset(dataset_name, variate_indices, stride=LOOKBACK_LENGTH)
                if smoke_test:
                    test_ds = Subset(test_ds, list(range(min(2, len(test_ds)))))
                test_loader = DataLoader(test_ds, batch_size=8 if not smoke_test else 2, shuffle=False)
                
                eval_results = evaluate_model(model, test_loader, device, n_samples=30, smoke_test=smoke_test)
                
                logger.info(f"[{subset_id}] Single: MSE={eval_results['single']['mse']:.4f}, MAE={eval_results['single']['mae']:.4f}")
                logger.info(f"[{subset_id}] Avg: MSE={eval_results['averaged']['mse']:.4f}, MAE={eval_results['averaged']['mae']:.4f}")
                
                save_eval_results(subset_id, dataset_name, variate_indices, 
                                {**train_metrics, 'tuned_params': tuned_params}, eval_results, RESULTS_DIR)
                
                # Log to wandb
                log_wandb_eval_results(subset_id, eval_results, train_metrics)
                log_wandb_model_checkpoint(ckpt_path, subset_id)
                
                manifest.mark_complete(subset_id, ckpt_path, {**train_metrics, 'eval': eval_results})
            barrier()  # Sync after eval
            
        except KeyboardInterrupt:
            logger.info(f"\nInterrupted during {subset_id}. Progress saved.")
            return
        except Exception as e:
            logger.error(f"Error with {subset_id}: {e}")
            import traceback
            traceback.print_exc()
            manifest.subsets[subset_id]['status'] = 'error'
            manifest.subsets[subset_id]['error'] = str(e)
            if is_main_process():
                manifest.save()
    
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    complete = len([s for s in manifest.subsets.values() if s.get('status') == 'complete'])
    logger.info(f"Trained {complete} models")
    logger.info("=" * 60)


# ============================================================================
# CLI
# ============================================================================

def main():
    global logger
    
    parser = argparse.ArgumentParser(description='7-Variate Training Pipeline')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--smoke-test', action='store_true', help='Quick validation run')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--status', action='store_true', help='Show status')
    parser.add_argument('--ddp', action='store_true', help='Enable multi-GPU DDP training')
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--wandb-project', type=str, default='diffusion-tsf-7var', help='Wandb project name')
    parser.add_argument('--checkpoint-dir', type=str, default=None, help='Override checkpoint directory (for cluster storage)')
    parser.add_argument('--results-dir', type=str, default=None, help='Override results directory')
    parser.add_argument('--parallel-worker', type=int, default=None, help='Parallel worker ID for multi-GPU Optuna (0-N)')
    
    args = parser.parse_args()
    
    # Override directories if specified (useful for cluster storage)
    global CHECKPOINT_DIR, RESULTS_DIR, MANIFEST_PATH
    if args.checkpoint_dir:
        CHECKPOINT_DIR = args.checkpoint_dir
        MANIFEST_PATH = os.path.join(CHECKPOINT_DIR, 'training_manifest.json')
    if args.results_dir:
        RESULTS_DIR = args.results_dir
    
    # Setup DDP if requested
    if args.ddp:
        if not setup_ddp():
            print("ERROR: --ddp flag set but DDP init failed. Use: torchrun --nproc_per_node=N -m ...")
            sys.exit(1)
    
    # Setup parallel worker mode (multi-GPU Optuna without DDP)
    if args.parallel_worker is not None:
        setup_parallel_worker(args.parallel_worker)
    
    # Initialize logger (respects DDP rank)
    logger = setup_logging()
    
    if args.status:
        if is_main_process():
            if os.path.exists(MANIFEST_PATH):
                m = TrainingManifest.load()
                print(f"Created: {m.created_at}")
                print(f"iTransformer HP done: {m.itrans_hp_done}")
                print(f"Diffusion HP done: {m.diffusion_hp_done}")
                print(f"Pretrain complete: {m.pretrain_complete}")
                complete = len([s for s in m.subsets.values() if s.get('status') == 'complete'])
                pending = len([s for s in m.subsets.values() if s.get('status') == 'pending'])
                print(f"Subsets: {complete} complete, {pending} pending")
                if _ddp_enabled:
                    print(f"DDP: {_world_size} GPUs")
            else:
                print("No manifest found")
        return
    
    try:
        run_pipeline(
            resume=args.resume, 
            smoke_test=args.smoke_test, 
            seed=args.seed,
            use_wandb=args.wandb,
            wandb_project=args.wandb_project,
        )
    finally:
        finish_wandb()
        cleanup_ddp()


if __name__ == '__main__':
    main()
