"""
Multivariate diffusion TSF training pipeline (any `--n-variates`; default 7 for ETT-style runs).

PHASE 1: Synthetic Pretraining (with HP tuning)
  1A. iTransformer HP Tuning (Optuna; modest synthetic pool by default)
  1B. Diffusion HP Tuning with iTransformer guidance
  1C. Full Pretraining — default 10-epoch-style budgets; synthetic pool auto-sized
      from PRETRAIN_EPOCHS unless --synthetic-samples is set (disk cache reused when compatible)

PHASE 2: Fine-tuning per Dataset (simplified HP tuning)
  2A. HP Tune (Optuna)
  2B. Full Fine-tune (default 10 epochs / patience 5; override via CLI)
  2C. Evaluate

Usage:
    # Single GPU
    python -m models.diffusion_tsf.train_multivariate_pipeline
    python -m models.diffusion_tsf.train_multivariate_pipeline --resume
    python -m models.diffusion_tsf.train_multivariate_pipeline --smoke-test
    
    # Multi-GPU (DDP)
    torchrun --nproc_per_node=4 -m models.diffusion_tsf.train_multivariate_pipeline --ddp
    torchrun --nproc_per_node=2 -m models.diffusion_tsf.train_multivariate_pipeline --ddp --resume
"""

import argparse
import gc
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
from models.diffusion_tsf.storage_paths import resolve_checkpoint_dir, resolve_results_dir

DATASETS_DIR = os.path.join(project_root, "datasets")
CHECKPOINT_DIR = resolve_checkpoint_dir(script_dir)
RESULTS_DIR = resolve_results_dir(script_dir)
MANIFEST_PATH = os.path.join(CHECKPOINT_DIR, "training_manifest.json")
SYNTH_CACHE_DIR: Optional[str] = None

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


def require_tuned_param(params: Dict, key: str, stage_name: str):
    """Fail fast when a required tuned hyperparameter is missing."""
    if params is None:
        raise RuntimeError(f"{stage_name} requires tuned params, got None.")
    if key not in params:
        raise RuntimeError(
            f"{stage_name} requires tuned param '{key}', but tuning output is missing it."
        )
    return params[key]


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
        _optuna_storage = f"sqlite:///{os.path.join(CHECKPOINT_DIR, 'optuna_shared.db')}"
    
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
    can run trials concurrently. Worker 0 creates, others wait and join.
    """
    if is_parallel_mode() and _optuna_storage:
        # Worker 0 creates the study first
        if is_worker_zero():
            study = optuna.create_study(
                study_name=study_name,
                storage=_optuna_storage,
                direction=direction,
                load_if_exists=True,
                sampler=TPESampler(),
            )
            # Signal that study is ready
            ready_file = os.path.join(os.path.dirname(_optuna_storage.replace('sqlite:///', '')), 
                                      f'.{study_name}_ready')
            Path(ready_file).touch()
            return study
        else:
            # Other workers wait for study to be created
            ready_file = os.path.join(os.path.dirname(_optuna_storage.replace('sqlite:///', '')), 
                                      f'.{study_name}_ready')
            logger.info(f"Worker {get_worker_id()}: Waiting for study '{study_name}' to be created...")
            for _ in range(120):  # Wait up to 2 minutes
                if os.path.exists(ready_file):
                    break
                time.sleep(1)
            
            # Now join the existing study
            time.sleep(get_worker_id() * 0.5)  # Stagger connections
            return optuna.load_study(
                study_name=study_name,
                storage=_optuna_storage,
                sampler=TPESampler(),
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
    barrier_dir = os.path.join(CHECKPOINT_DIR, '.barriers')
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
        handlers.append(logging.FileHandler(os.path.join(script_dir, 'train_multivariate.log')))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers if handlers else [logging.NullHandler()],
        force=True,  # Override any existing config
    )
    return logging.getLogger(__name__)


# Deferred logger initialization (called after DDP setup).
# Falls back to module-level logger when imported by other scripts.
logger = logging.getLogger(__name__)


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
    name: str = None,
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
        'hp_tune_epochs': HP_TUNE_EPOCHS,
        'hp_tune_patience': HP_TUNE_PATIENCE,
        'pretrain_virtual_samples': resolve_pretrain_virtual_dataset_size(False),
        'pretrain_synthetic_override': PRETRAIN_SYNTHETIC_SAMPLES_OVERRIDE,
        'synthetic_samples_full_cap': SYNTHETIC_SAMPLES_CAP,
        'synthetic_samples_hp_tune': SYNTHETIC_SAMPLES_HP_TUNE,
        'synthetic_samples_diff_tune': SYNTHETIC_SAMPLES_DIFF_TUNE,
        'n_itrans_hp_trials': N_ITRANS_HP_TRIALS,
        'n_diffusion_hp_trials': N_DIFFUSION_HP_TRIALS,
        'n_finetune_hp_trials': N_FINETUNE_HP_TRIALS,
        'itrans_batch_sizes': ITRANS_BATCH_SIZES,
        'diffusion_batch_sizes': DIFFUSION_BATCH_SIZES,
        'finetune_batch_sizes': FINETUNE_BATCH_SIZES,
        'diffusion_probe_target_effective_batch': DIFFUSION_PROBE_TARGET_EFFECTIVE_BATCH,
        'diffusion_probe_max_batch_cap': DIFFUSION_PROBE_MAX_BATCH_CAP,
        'diffusion_probe_max_candidate_default_v': diffusion_probe_max_candidate(N_VARIATES, False),
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
        tags = ['multivariate-pipeline']
    if _ddp_enabled:
        tags.append(f'ddp-{get_world_size()}gpu')

    run_name = (name or os.environ.get("WANDB_NAME") or "").strip() or None

    try:
        init_kw = dict(
            project=project,
            config=full_config,
            resume="allow" if resume else None,
            id=run_id,
            reinit=True,
            tags=tags,
            save_code=True,
        )
        if run_name is not None:
            init_kw["name"] = run_name
        _wandb_run = wandb.init(**init_kw)
        
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
# Constants — single source of truth lives in `pipeline_config.py`.
# Edit that file to change defaults; nothing here overrides those values.
# ============================================================================

from models.diffusion_tsf.pipeline_config import (
    LOOKBACK_LENGTH,
    FORECAST_LENGTH,
    ITRANSFORMER_SEQ_LEN,
    IMAGE_HEIGHT,
    LOOKBACK_OVERLAP,
    PAST_LOSS_WEIGHT,
    N_VARIATES_DEFAULT,
    PRETRAIN_EPOCHS,
    PRETRAIN_PATIENCE,
    PRETRAIN_DIFFUSION_EPOCHS,
    PRETRAIN_DIFFUSION_MAX_EPOCHS,
    DIFFUSION_HP_MAX_EPOCHS,
    DIFFUSION_HP_PATIENCE,
    SYNTHETIC_SAMPLES_HP_TUNE,
    SYNTHETIC_SAMPLES_DIFF_TUNE,
    SYNTHETIC_SAMPLES_MIN,
    SYNTHETIC_SAMPLES_CAP,
    resolve_synthetic_params,
    PRETRAIN_SYNTHETIC_SAMPLES_OVERRIDE,
    HP_TUNE_EPOCHS,
    HP_TUNE_PATIENCE,
    N_ITRANS_HP_TRIALS,
    N_DIFFUSION_HP_TRIALS,
    N_FINETUNE_HP_TRIALS,
    ITRANS_HP_PRETRAIN_MAX_EPOCHS,
    ITRANS_HP_PRETRAIN_PATIENCE,
    ITRANS_HP_FINETUNE_MAX_EPOCHS,
    ITRANS_HP_FINETUNE_PATIENCE,
    ITRANS_REAL_COLD_START,
    ITRANS_BATCH_SIZES,
    DIFFUSION_BATCH_SIZES,
    DIFFUSION_PROBE_TARGET_EFFECTIVE_BATCH,
    DIFFUSION_PROBE_MAX_BATCH_CAP,
    DIFFUSION_PROBE_MIN_BATCH,
    FINETUNE_BATCH_SIZES,
    FINETUNE_HP_LR_MIN,
    FINETUNE_HP_LR_MAX,
    diffusion_probe_max_candidate,
    USE_AMP,
    USE_GRADIENT_CHECKPOINTING,
    NUM_DIFFUSION_STEPS,
    NOISE_SCHEDULE,
    USE_TIME_RAMP,
    USE_TIME_SINE,
    UNET_CHANNELS,
    ATTENTION_LEVELS,
    DISABLE_CROSS_ATTENTION,
    MODEL_TYPE,
    DIT_PATCH_SIZE,
    DIT_EMBED_DIM,
    DIT_DEPTH,
    DIT_NUM_HEADS,
    DIT_MLP_RATIO,
    DIT_DROPOUT,
    GUIDANCE_PENALTY_WEIGHT,
    EVAL_NUM_SAMPLES,
    resolve_pretrain_virtual_dataset_size,
    synthetic_epoch_capacity_itrans_hp,
    synthetic_epoch_capacity_diff_hp,
    synthetic_epoch_capacity_pretrain_itrans,
    synthetic_epoch_capacity_pretrain_diffusion,
)

# Per-run dispatch knob — set from --n-variates at CLI parse time.
N_VARIATES = N_VARIATES_DEFAULT

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


def set_realts_training_epoch(loader_or_subset_or_dataset, epoch: int) -> None:
    """If ``loader_or_subset_or_dataset`` wraps a RealTS, set its strided synthetic epoch."""
    ds = loader_or_subset_or_dataset
    if isinstance(ds, DataLoader):
        ds = ds.dataset
    if isinstance(ds, Subset):
        ds = ds.dataset
    if hasattr(ds, 'set_synthetic_epoch'):
        ds.set_synthetic_epoch(epoch)


def get_synth_cache_dir(checkpoint_dir: Optional[str] = None, smoke_test: bool = False) -> Optional[str]:
    """Resolve synthetic cache dir; prefer shared cache when configured."""
    if smoke_test:
        return None
    if SYNTH_CACHE_DIR:
        os.makedirs(SYNTH_CACHE_DIR, exist_ok=True)
        return SYNTH_CACHE_DIR
    path = os.path.join(project_root, 'synth_data')
    os.makedirs(path, exist_ok=True)
    return path


# ============================================================================
# Dimensionality Helpers
# ============================================================================

def get_dataset_n_cols(dataset_name: str) -> int:
    """Return the number of numeric columns in a dataset (excluding date)."""
    path = os.path.join(DATASETS_DIR, DATASET_REGISTRY[dataset_name][0])
    df = pd.read_csv(path, nrows=1)
    date_col = DATASET_REGISTRY[dataset_name][1]
    return sum(1 for c in df.columns if c != date_col)


def get_dim_for_dataset(dataset_name: str) -> int:
    """Return native dataset dimensionality (always full variates)."""
    return get_dataset_n_cols(dataset_name)


def get_all_pretrain_dims() -> Dict[int, List[str]]:
    """Return {dim: [dataset_names]} grouping for pretraining.

    Each unique dim needs its own pretrained iTransformer + Diffusion.
    """
    groups: Dict[int, List[str]] = {}
    for name in DATASET_REGISTRY:
        dim = get_dim_for_dataset(name)
        groups.setdefault(dim, []).append(name)
    return groups


def pretrain_dir_for_dim(dim: int, base_dir: str = None) -> str:
    """Checkpoint subdirectory for a specific pretrain dimensionality."""
    base = base_dir or CHECKPOINT_DIR
    return os.path.join(base, f'pretrained_dim{dim}')


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
    seq_len: int = None,
    pred_len: int = FORECAST_LENGTH,
    num_vars: int = None,
    d_model: int = 512,
    d_ff: int = 512,
    e_layers: int = 4,
    n_heads: int = 8,
    dropout: float = 0.1,
):
    """Create iTransformer config object."""
    if seq_len is None:
        seq_len = ITRANSFORMER_SEQ_LEN
    if num_vars is None:
        num_vars = N_VARIATES
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
    seq_len: int = None,
    pred_len: int = FORECAST_LENGTH,
    num_vars: int = None,
    dropout: float = 0.1,
) -> nn.Module:
    """Create iTransformer model."""
    if seq_len is None:
        seq_len = ITRANSFORMER_SEQ_LEN
    if num_vars is None:
        num_vars = N_VARIATES
    iTransformerModel = get_itransformer_class()
    config = create_itransformer_config(
        seq_len=seq_len, pred_len=pred_len, num_vars=num_vars, dropout=dropout
    )
    return iTransformerModel(config)


def load_itransformer_from_checkpoint(
    path: str,
    num_vars: int,
    device: torch.device,
    dropout: float = 0.1,
) -> nn.Module:
    """Load iTransformer weights, inferring ``seq_len`` from the checkpoint itself.

    The inverted-embedding layer stores a Linear of shape ``[d_model, seq_len]`` at
    key ``enc_embedding.value_embedding.weight``. Reading that shape lets us build
    a model that matches whatever seq_len the checkpoint was saved with — no
    hardcoded fallback list.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt['model_state_dict']
    weight_key = 'enc_embedding.value_embedding.weight'
    if weight_key not in state:
        raise RuntimeError(
            f"iTransformer checkpoint {path} is missing key {weight_key!r}; "
            f"cannot infer seq_len."
        )
    ckpt_seq_len = int(state[weight_key].shape[1])

    model = create_itransformer(
        seq_len=ckpt_seq_len, num_vars=num_vars, dropout=dropout,
    ).to(device)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        raise RuntimeError(
            f"Cannot load iTransformer checkpoint {path} "
            f"(inferred seq_len={ckpt_seq_len}): {e}"
        ) from e
    model.eval()
    if ckpt_seq_len != ITRANSFORMER_SEQ_LEN:
        logger.warning(
            f"iTransformer checkpoint {path} has seq_len={ckpt_seq_len}, "
            f"differs from current ITRANSFORMER_SEQ_LEN={ITRANSFORMER_SEQ_LEN}. "
            f"Loaded with checkpoint's seq_len; ensure callers slice past inputs accordingly."
        )
    return model


def load_diffusion_state_keep_attached_guidance(model: nn.Module, ckpt_state: Dict) -> None:
    """Load a diffusion checkpoint while preserving the guidance submodule that the
    caller already attached via ``set_guidance_model``.

    The diffusion checkpoint's ``model_state_dict`` includes ``guidance_model.*``
    keys (PyTorch saves all submodules). Reloading those would overwrite the
    freshly-attached guidance — and breaks loudly when the saved guidance has a
    different ``seq_len`` than the attached one (e.g. synthetic-pretrain vs
    real-finetuned iTransformer). We always want to keep the attached guidance
    and only restore the diffusion backbone weights.
    """
    filtered = {k: v for k, v in ckpt_state.items() if not k.startswith('guidance_model.')}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    leaked = [k for k in (missing or []) if not k.startswith('guidance_model.')]
    if leaked:
        raise RuntimeError(f"Diffusion ckpt missing non-guidance keys: {leaked[:5]}...")
    real_unexpected = [k for k in (unexpected or []) if not k.startswith('guidance_model.')]
    if real_unexpected:
        raise RuntimeError(f"Diffusion ckpt has unexpected keys: {real_unexpected[:5]}...")


# ============================================================================
# Diffusion Model Creation (with guidance support)
# ============================================================================

def create_diffusion_model(
    n_variates: int = None,
    lookback: int = LOOKBACK_LENGTH,
    horizon: int = FORECAST_LENGTH,
    lookback_overlap: int = LOOKBACK_OVERLAP,
    past_loss_weight: float = PAST_LOSS_WEIGHT,
    guidance_penalty_weight: Optional[float] = None,
) -> DiffusionTSF:
    """Create DiffusionTSF model with iTransformer guidance channel enabled."""
    if n_variates is None:
        n_variates = N_VARIATES
    if guidance_penalty_weight is None:
        guidance_penalty_weight = GUIDANCE_PENALTY_WEIGHT

    logger.info(f"Creating diffusion model: guidance_penalty_weight={guidance_penalty_weight}")

    config = DiffusionTSFConfig(
        num_variables=n_variates,
        lookback_length=lookback,
        forecast_length=horizon + lookback_overlap,
        lookback_overlap=lookback_overlap,
        past_loss_weight=past_loss_weight,
        image_height=IMAGE_HEIGHT,
        use_coordinate_channel=True,
        use_time_ramp=USE_TIME_RAMP,
        use_time_sine=USE_TIME_SINE,
        use_guidance_channel=True,
        guidance_penalty_weight=guidance_penalty_weight,
        num_diffusion_steps=NUM_DIFFUSION_STEPS,
        noise_schedule=NOISE_SCHEDULE,
        model_type=MODEL_TYPE,
        unet_channels=UNET_CHANNELS,
        attention_levels=ATTENTION_LEVELS,
        disable_cross_attention=DISABLE_CROSS_ATTENTION,
        num_res_blocks=2,
        dit_patch_size=DIT_PATCH_SIZE,
        dit_embed_dim=DIT_EMBED_DIM,
        dit_depth=DIT_DEPTH,
        dit_num_heads=DIT_NUM_HEADS,
        dit_mlp_ratio=DIT_MLP_RATIO,
        dit_dropout=DIT_DROPOUT,
        use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
        use_amp=USE_AMP,
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
        lookback_overlap: int = 0,
    ):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.lookback = lookback
        self.horizon = horizon
        self.stride = stride
        self.lookback_overlap = lookback_overlap
        total_len = lookback + horizon
        self.n_samples = max(0, (len(data) - total_len) // stride + 1)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        start = idx * self.stride
        past = self.data[start:start + self.lookback].T
        # Target includes last K observed steps + H forecast steps
        target_start = start + self.lookback - self.lookback_overlap
        target_end = start + self.lookback + self.horizon
        future = self.data[target_start:target_end].T
        return past, future


def load_dataset(
    dataset_name: str,
    variate_indices: List[int] = None,
    lookback: int = LOOKBACK_LENGTH,
    horizon: int = FORECAST_LENGTH,
    stride: int = 1,
    lookback_overlap: int = LOOKBACK_OVERLAP,
) -> Tuple[Dataset, Dataset, Dataset, Dict]:
    """Load dataset and return train/val/test splits."""
    path = os.path.join(DATASETS_DIR, DATASET_REGISTRY[dataset_name][0])
    date_col = DATASET_REGISTRY[dataset_name][1]
    
    df = pd.read_csv(path)
    data_cols = [c for c in df.columns if c != date_col]
    data = df[data_cols].values.astype(np.float32)
    
    if variate_indices is not None:
        data = data[:, variate_indices]
    
    # Chronological split: 70/10/20 (boundaries first; z-score uses train slice only)
    n = len(data)
    total_window = lookback + horizon
    if n < total_window:
        raise ValueError(
            f"Dataset '{dataset_name}' has {n} rows but needs at least "
            f"{total_window} (lookback={lookback} + horizon={horizon}). "
            f"Skipping this dataset."
        )
    
    train_end = int(n * 0.7)
    val_end = int(n * 0.8)
    
    train_slice = data[:train_end]
    mean = train_slice.mean(axis=0, keepdims=True)
    std = train_slice.std(axis=0, keepdims=True) + 1e-8
    data = (data - mean) / std
    
    train_ds = TimeSeriesDataset(data[:train_end], lookback, horizon, stride, lookback_overlap=lookback_overlap)
    val_ds = TimeSeriesDataset(data[train_end:val_end], lookback, horizon, stride=lookback, lookback_overlap=lookback_overlap)
    test_ds = TimeSeriesDataset(data[val_end:], lookback, horizon, stride=lookback, lookback_overlap=lookback_overlap)
    
    return train_ds, val_ds, test_ds, {'mean': mean, 'std': std}


# ============================================================================
# Variate Subset Management
# ============================================================================

def generate_dataset_job(dataset_name: str, n_variates: int = None, seed: int = 42) -> Dict:
    """Return one full-dataset training job (no variate partitioning)."""
    path = os.path.join(DATASETS_DIR, DATASET_REGISTRY[dataset_name][0])
    df = pd.read_csv(path, nrows=1)
    date_col = DATASET_REGISTRY[dataset_name][1]
    all_cols = [c for c in df.columns if c != date_col]
    indices = list(range(len(all_cols)))
    return {'dataset_id': dataset_name, 'variate_indices': indices, 'variate_names': all_cols}


def generate_all_dataset_jobs(seed: int = 42) -> Dict[str, Dict]:
    """Return one full-dataset job per dataset, filtered to those whose
    variate count matches N_VARIATES exactly.

    This avoids needing separate pretrained models for different dataset sizes.
    Datasets with a different variate count are skipped silently.
    """
    result = {}
    for name in DATASET_REGISTRY:
        try:
            n_cols = get_dataset_n_cols(name)
        except Exception:
            continue
        if n_cols != N_VARIATES:
            logger.debug(f"Skipping {name}: {n_cols} variates (need {N_VARIATES})")
            continue
        result[name] = generate_dataset_job(name, seed=seed)
    if not result:
        logger.warning(
            f"No datasets found with exactly {N_VARIATES} variates. "
            "Check --n-variates matches your target datasets."
        )
    return result


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


def amp_context():
    """Return the appropriate autocast context for mixed precision."""
    if USE_AMP and torch.cuda.is_available():
        return torch.amp.autocast('cuda', dtype=torch.bfloat16)
    from contextlib import nullcontext
    return nullcontext()


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
        x_enc = past.permute(0, 2, 1).to(device)
        seq_sl = getattr(model, 'seq_len', x_enc.shape[1])
        if x_enc.shape[1] > seq_sl:
            x_enc = x_enc[:, -seq_sl:, :]
        y_true = future.permute(0, 2, 1).to(device)
        # iTransformer predicts H steps; strip the K overlap from target
        if LOOKBACK_OVERLAP > 0:
            y_true = y_true[:, LOOKBACK_OVERLAP:, :]
        
        optimizer.zero_grad()
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
            seq_sl = getattr(model, 'seq_len', x_enc.shape[1])
            if x_enc.shape[1] > seq_sl:
                x_enc = x_enc[:, -seq_sl:, :]
            y_true = future.permute(0, 2, 1).to(device)
            if LOOKBACK_OVERLAP > 0:
                y_true = y_true[:, LOOKBACK_OVERLAP:, :]
            y_pred = model(x_enc, None, None, None)
            loss = criterion(y_pred, y_true)
            total_loss += loss.item()
            n_batches += 1
    
    return total_loss / max(n_batches, 1)


def auto_select_max_even_batch_size(
    phase_name: str,
    max_candidate: int,
    try_step_fn,
    min_candidate: int = 2,
) -> int:
    """Pick the largest even batch size that passes ``try_step_fn`` without OOM."""
    max_candidate = max(min_candidate, max_candidate)
    if max_candidate % 2 != 0 and max_candidate > 1:
        max_candidate -= 1
    min_candidate = max(min_candidate, 1)

    lo = min_candidate
    hi = max_candidate
    best = min_candidate

    while lo <= hi:
        mid = (lo + hi) // 2
        if mid % 2 != 0 and mid > 1:
            mid -= 1
        if mid < min_candidate:
            mid = min_candidate

        try:
            ok = bool(try_step_fn(mid))
        except torch.OutOfMemoryError:
            ok = False
        except RuntimeError as exc:
            ok = 'out of memory' not in str(exc).lower()
            if not ok:
                pass
            else:
                raise

        if ok:
            best = mid
            # If mid is 1, stepping by 2 would jump to 3, but lo is updated to mid + 1 if mid == 1, or mid + 2 if even
            lo = mid + 1 if mid == 1 else mid + 2
        else:
            hi = mid - 1 if mid == 1 else mid - 2

    # Apply safety margin: a single probe step underestimates sustained
    # training memory (optimizer state, gradient accumulation, etc.).
    safe = max(min_candidate, int(best * 0.8))
    if safe % 2 != 0 and safe > 1:
        safe = max(min_candidate, safe - 1)
    logger.info(f"[AutoBS] {phase_name}: selected batch_size={safe} (probe_max={best}, tested_max={max_candidate})")
    return safe


def select_itrans_batch_size(
    phase_name: str,
    dataset,
    device: torch.device,
    dropout: float,
    max_candidate: int,
) -> int:
    """Probe iTransformer memory with one train step and pick largest safe even batch."""
    sample_past, sample_future = dataset[0]

    def _try(bs: int) -> bool:
        model = None
        optimizer = None
        x_enc = None
        y_true = None
        y_pred = None
        loss = None
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model = create_itransformer(dropout=dropout).to(device)
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            optimizer.zero_grad(set_to_none=True)

            past = sample_past.unsqueeze(0).repeat(bs, 1, 1).to(device)
            future = sample_future.unsqueeze(0).repeat(bs, 1, 1).to(device)
            x_enc = past.permute(0, 2, 1)
            seq_sl = getattr(model, 'seq_len', x_enc.shape[1])
            if x_enc.shape[1] > seq_sl:
                x_enc = x_enc[:, -seq_sl:, :]
            y_true = future.permute(0, 2, 1)
            if LOOKBACK_OVERLAP > 0:
                y_true = y_true[:, LOOKBACK_OVERLAP:, :]
            y_pred = model(x_enc, None, None, None)
            loss = nn.functional.mse_loss(y_pred, y_true)
            loss.backward()
            optimizer.step()
            return True
        finally:
            del loss, y_pred, y_true, x_enc, optimizer, model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    return auto_select_max_even_batch_size(phase_name, max_candidate, _try, min_candidate=2)


def select_diffusion_batch_size(
    phase_name: str,
    dataset,
    device: torch.device,
    itrans_guidance: iTransformerGuidance,
    max_candidate: int,
) -> int:
    """Probe diffusion memory with one train step and pick largest safe even batch."""
    sample_past, sample_future = dataset[0]

    def _try(bs: int) -> bool:
        model = None
        optimizer = None
        past = None
        future = None
        loss = None
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model = create_diffusion_model().to(device)
            model.set_guidance_model(itrans_guidance)
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            optimizer.zero_grad(set_to_none=True)

            past = sample_past.unsqueeze(0).repeat(bs, 1, 1).to(device)
            future = sample_future.unsqueeze(0).repeat(bs, 1, 1).to(device)
            with amp_context():
                loss = model.get_loss(past, future)
            loss.backward()
            optimizer.step()
            
            return True
        finally:
            del loss, future, past, optimizer, model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    return auto_select_max_even_batch_size(phase_name, max_candidate, _try, min_candidate=DIFFUSION_PROBE_MIN_BATCH)


def get_itrans_batch_size_candidates(smoke_test: bool) -> List[int]:
    """Return a safe iTransformer HP batch-size search space for current N_VARIATES."""
    if smoke_test:
        return [8, 16]
    if N_VARIATES >= 512:
        return [8, 16, 32]
    if N_VARIATES >= 256:
        return [16, 32, 64]
    return ITRANS_BATCH_SIZES


def itrans_hp_objective(
    trial,
    synthetic_loader,
    val_loader,
    device,
    smoke_test=False,
    fixed_batch_size: Optional[int] = None,
    best_state: Optional[dict] = None,
    pretrained_ckpt: Optional[str] = None,
    max_epochs: int = ITRANS_HP_PRETRAIN_MAX_EPOCHS,
    early_stop_patience: int = ITRANS_HP_PRETRAIN_PATIENCE,
):
    """Optuna objective for iTransformer HP search.

    pretrained_ckpt: if provided, warm-starts each trial from those weights
        (used for finetune HP search on real data).
    best_state: shared mutable dict; updated with best cross-trial model state
        whenever a new minimum val loss is achieved.
    """
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    if fixed_batch_size is None:
        batch_size = trial.suggest_categorical('batch_size', get_itrans_batch_size_candidates(smoke_test))
    else:
        batch_size = fixed_batch_size
    dropout = trial.suggest_float('dropout', 0.0, 0.3)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = create_itransformer(dropout=dropout).to(device)
    if pretrained_ckpt is not None and os.path.exists(pretrained_ckpt):
        ckpt = torch.load(pretrained_ckpt, map_location=device, weights_only=False)
        try:
            model.load_state_dict(ckpt['model_state_dict'], strict=True)
        except RuntimeError as e:
            logger.warning(
                "iTransformer warm-start failed (%s); training this trial from scratch.",
                e,
            )

    train_loader = DataLoader(synthetic_loader.dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_bs = min(batch_size, 32)
    val_loader_local = DataLoader(val_loader.dataset, batch_size=val_bs, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    epochs = max_epochs if not smoke_test else 1
    patience = early_stop_patience if not smoke_test else 1
    early_stop = EarlyStopping(patience=patience)
    best_val_loss = float('inf')

    try:
        for epoch in range(epochs):
            set_realts_training_epoch(synthetic_loader, epoch)
            train_itransformer_epoch(model, train_loader, optimizer, criterion, device)
            val_loss = validate_itransformer(model, val_loader_local, criterion, device)

            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if best_state is not None and val_loss < best_state.get('val_loss', float('inf')):
                    best_state['model_state'] = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    best_state['val_loss'] = val_loss

            if early_stop(val_loss):
                break
    except torch.OutOfMemoryError:
        logger.warning(f"[iTransformer HP] OOM at batch_size={batch_size}; pruning trial {trial.number}.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise optuna.TrialPruned()

    return best_val_loss


def run_itransformer_hp_tuning(
    n_trials: int,
    smoke_test: bool = False,
    checkpoint_dir: Optional[str] = None,
) -> Tuple[Dict, Optional[str]]:
    """Run Optuna HP search for iTransformer.

    Returns (best_params, path_to_best_model_or_None). The best model state
    across all trials is saved to itrans_hp_best.pt in checkpoint_dir so the
    caller can use it directly without a separate full-pretrain step.
    """
    logger.info("=" * 60)
    logger.info("PHASE 1A: iTransformer HP Tuning")
    logger.info(f"Trials: {n_trials}")
    logger.info(
        f"iTransformer seq_len={ITRANSFORMER_SEQ_LEN} (diffusion lookback={LOOKBACK_LENGTH})"
    )
    logger.info("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    requested_n = SYNTHETIC_SAMPLES_HP_TUNE
    requested_cap = synthetic_epoch_capacity_itrans_hp()
    n_samples, epoch_cap = resolve_synthetic_params(requested_n, requested_cap, smoke_test)

    n_val = 0 if smoke_test else min(n_samples // 10, 1000)
    synth_cache = get_synth_cache_dir()
    synthetic_loader = get_synthetic_dataloader(
        batch_size=64,
        lookback_length=LOOKBACK_LENGTH,
        forecast_length=FORECAST_LENGTH,
        num_variables=N_VARIATES,
        num_samples=n_samples,
        num_workers=0,
        lookback_overlap=LOOKBACK_OVERLAP,
        cache_dir=synth_cache,
        skip_cross_var_aug=(N_VARIATES > 32),
        val_tail_n=n_val,
        synthetic_epoch_capacity=epoch_cap,
    )

    dataset = synthetic_loader.dataset
    n_val = min(len(dataset) // 10, 1000)
    train_subset = Subset(dataset, list(range(len(dataset) - n_val)))
    val_subset   = Subset(dataset, list(range(len(dataset) - n_val, len(dataset))))

    train_bs = select_itrans_batch_size(
        phase_name='iTransformer HP tune',
        dataset=train_subset,
        device=device,
        dropout=0.1,
        max_candidate=32 if smoke_test else 256,
    )
    train_loader = DataLoader(train_subset, batch_size=train_bs, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_subset,   batch_size=min(train_bs, 32), shuffle=False, num_workers=0)

    _best_state: dict = {'model_state': None, 'val_loss': float('inf')}

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2),
    )

    logger.info(f"Starting iTransformer HP search: {n_trials} trials")

    def log_trial(study, trial):
        logger.info(f"[iTransformer HP] Trial {trial.number}/{n_trials}: "
                   f"loss={trial.value:.4f}, lr={trial.params['learning_rate']:.2e}, "
                   f"bs={train_bs}, dropout={trial.params['dropout']:.3f}")

    study.optimize(
        lambda trial: itrans_hp_objective(
            trial, train_loader, val_loader, device, smoke_test,
            fixed_batch_size=train_bs, best_state=_best_state,
            max_epochs=ITRANS_HP_PRETRAIN_MAX_EPOCHS,
            early_stop_patience=ITRANS_HP_PRETRAIN_PATIENCE,
        ),
        n_trials=n_trials,
        show_progress_bar=True,
        callbacks=[log_trial],
    )

    best_params = study.best_params
    best_params['batch_size'] = train_bs
    logger.info(f"Best iTransformer params: lr={best_params['learning_rate']:.2e}, "
               f"bs={best_params['batch_size']}, dropout={best_params['dropout']:.3f}")
    logger.info(f"Best val loss: {study.best_value:.4f}")

    ckpt_path = None
    if checkpoint_dir is not None and _best_state.get('model_state') is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(checkpoint_dir, 'itrans_hp_best.pt')
        torch.save({'model_state_dict': _best_state['model_state'], 'best_params': best_params}, ckpt_path)
        logger.info(f"  Saved best iTrans HP model → {ckpt_path} (val_loss={_best_state['val_loss']:.4f})")

    return best_params, ckpt_path


# ============================================================================
# PHASE 1B: Diffusion HP Tuning (with iTransformer guidance)
# ============================================================================

def diffusion_hp_objective(
    trial,
    synthetic_loader,
    val_loader,
    itrans_guidance: iTransformerGuidance,
    device,
    smoke_test=False,
    fixed_batch_size: Optional[int] = None,
    best_state: Optional[dict] = None,
):
    """Optuna objective for Diffusion HP search.

    best_state is a shared mutable dict; when provided, updates
    best_state['model_state'] and best_state['val_loss'] whenever this
    trial achieves a new cross-trial best (used to skip a separate pretrain).
    """
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    if fixed_batch_size is None:
        batch_size = trial.suggest_categorical('batch_size', [2, 4] if smoke_test else DIFFUSION_BATCH_SIZES)
    else:
        batch_size = fixed_batch_size

    model = create_diffusion_model().to(device)
    model.set_guidance_model(itrans_guidance)

    train_loader = DataLoader(synthetic_loader.dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    epochs = DIFFUSION_HP_MAX_EPOCHS if not smoke_test else 1
    patience = DIFFUSION_HP_PATIENCE if not smoke_test else 1
    early_stop = EarlyStopping(patience=patience)
    best_val_loss = float('inf')

    for epoch in range(epochs):
        set_realts_training_epoch(synthetic_loader, epoch)
        model.train()
        for past, future in train_loader:
            past, future = past.to(device), future.to(device)
            optimizer.zero_grad()
            with amp_context():
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
                with amp_context():
                    loss = model.get_loss(past, future)
                val_loss += loss.item()
                n_batches += 1
        val_loss /= max(n_batches, 1)

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if best_state is not None and val_loss < best_state.get('val_loss', float('inf')):
                best_state['model_state'] = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_state['val_loss'] = val_loss

        if early_stop(val_loss):
            break

    return best_val_loss


def run_diffusion_hp_tuning(
    itrans_checkpoint: str,
    n_trials: int,
    smoke_test: bool = False,
    checkpoint_dir: Optional[str] = None,
) -> Tuple[Dict, Optional[str]]:
    """Run Optuna HP search for Diffusion model."""
    logger.info("=" * 60)
    logger.info("PHASE 1B: Diffusion HP Tuning (with iTransformer guidance)")
    logger.info(f"Trials: {n_trials}")
    logger.info("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load iTransformer as guidance (supports legacy 512-wide and current shorter seq checkpoints)
    itrans_model = load_itransformer_from_checkpoint(itrans_checkpoint, N_VARIATES, device)
    itrans_guidance = iTransformerGuidance(itrans_model)
    
    # Create small synthetic dataset for fast iteration
    requested_n = SYNTHETIC_SAMPLES_DIFF_TUNE
    requested_cap = synthetic_epoch_capacity_diff_hp()
    n_samples, epoch_cap = resolve_synthetic_params(requested_n, requested_cap, smoke_test)

    n_val = 0 if smoke_test else min(n_samples // 10, 500)
    synth_cache = get_synth_cache_dir()
    synthetic_loader = get_synthetic_dataloader(
        batch_size=32,
        lookback_length=LOOKBACK_LENGTH,
        forecast_length=FORECAST_LENGTH,
        num_variables=N_VARIATES,
        num_samples=n_samples,
        num_workers=0,
        lookback_overlap=LOOKBACK_OVERLAP,
        cache_dir=synth_cache,
        skip_cross_var_aug=(N_VARIATES > 32),
        val_tail_n=n_val,
        synthetic_epoch_capacity=epoch_cap,
    )
    
    dataset = synthetic_loader.dataset
    n_val = min(len(dataset) // 10, 500)
    train_subset = Subset(dataset, list(range(len(dataset) - n_val)))
    val_subset = Subset(dataset, list(range(len(dataset) - n_val, len(dataset))))
    
    train_bs = select_diffusion_batch_size(
        phase_name='Diffusion HP tune',
        dataset=train_subset,
        device=device,
        itrans_guidance=itrans_guidance,
        max_candidate=diffusion_probe_max_candidate(N_VARIATES, smoke_test),
    )
    train_loader = DataLoader(train_subset, batch_size=train_bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=min(train_bs, 16), shuffle=False, num_workers=0)
    
    # Run Optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
    
    logger.info(f"Starting Diffusion HP search: {n_trials} trials")
    
    def log_trial(study, trial):
        bs = trial.params.get('batch_size', train_bs)
        logger.info(f"[Diffusion HP] Trial {trial.number}/{n_trials}: "
                   f"loss={trial.value:.4f}, lr={trial.params['learning_rate']:.2e}, "
                   f"bs={bs}")
    
    _best_state: dict = {'model_state': None, 'val_loss': float('inf')}

    study.optimize(
        lambda trial: diffusion_hp_objective(
            trial, train_loader, val_loader, itrans_guidance, device, smoke_test,
            fixed_batch_size=train_bs, best_state=_best_state,
        ),
        n_trials=n_trials,
        show_progress_bar=True,
        callbacks=[log_trial],
    )

    best_params = study.best_params
    best_params['batch_size'] = train_bs
    logger.info(f"Best Diffusion params: lr={best_params['learning_rate']:.2e}, bs={best_params['batch_size']}")
    logger.info(f"Best val loss: {study.best_value:.4f}")

    ckpt_path = None
    if checkpoint_dir is not None and _best_state.get('model_state') is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(checkpoint_dir, 'diff_hp_best.pt')
        torch.save({'model_state_dict': _best_state['model_state'], 'best_params': best_params}, ckpt_path)
        logger.info(f"  Saved best diffusion HP model → {ckpt_path} (val_loss={_best_state['val_loss']:.4f})")

    return best_params, ckpt_path


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
    
    lr = require_tuned_param(best_params, 'learning_rate', 'iTransformer pretraining')
    tuned_batch_size = require_tuned_param(best_params, 'batch_size', 'iTransformer pretraining')
    dropout = require_tuned_param(best_params, 'dropout', 'iTransformer pretraining')
    batch_size = tuned_batch_size
    
    # Create data
    synth_cache = get_synth_cache_dir(checkpoint_dir=checkpoint_dir, smoke_test=smoke_test)
    n_val = 0 if smoke_test else min(n_samples // 10, 5000)
    epoch_cap = 1 if smoke_test else synthetic_epoch_capacity_pretrain_itrans()
    synthetic_loader = get_synthetic_dataloader(
        batch_size=min(32, max(2, tuned_batch_size)),
        lookback_length=LOOKBACK_LENGTH,
        forecast_length=FORECAST_LENGTH,
        num_variables=N_VARIATES,
        num_samples=n_samples,
        num_workers=0 if smoke_test else 4,
        lookback_overlap=LOOKBACK_OVERLAP,
        cache_dir=synth_cache,
        skip_cross_var_aug=(N_VARIATES > 32),
        val_tail_n=n_val,
        synthetic_epoch_capacity=epoch_cap,
    )
    
    # Split for validation (indices must match ``val_tail_n`` above)
    dataset = synthetic_loader.dataset
    train_subset = Subset(dataset, list(range(len(dataset) - n_val)))
    val_subset = Subset(dataset, list(range(len(dataset) - n_val, len(dataset))))
    if not _ddp_enabled:
        batch_size = select_itrans_batch_size(
            phase_name='iTransformer pretrain',
            dataset=train_subset,
            device=device,
            dropout=dropout,
            max_candidate=max(2, tuned_batch_size),
        )
    effective_batch_size = batch_size // get_world_size() if _ddp_enabled else batch_size
    effective_batch_size = max(1, effective_batch_size)
    
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
        
        set_realts_training_epoch(train_loader, epoch)
        
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
    
    lr = require_tuned_param(best_params, 'learning_rate', 'Diffusion pretraining')
    tuned_batch_size = require_tuned_param(best_params, 'batch_size', 'Diffusion pretraining')
    batch_size = tuned_batch_size
    
    # Load iTransformer as guidance (not wrapped in DDP - used in eval mode only)
    itrans_model = load_itransformer_from_checkpoint(itrans_checkpoint, N_VARIATES, device)
    itrans_guidance = iTransformerGuidance(itrans_model)
    
    # Create data
    synth_cache = get_synth_cache_dir(checkpoint_dir=checkpoint_dir, smoke_test=smoke_test)
    n_val = 0 if smoke_test else min(n_samples // 10, 5000)
    epoch_cap = 1 if smoke_test else synthetic_epoch_capacity_pretrain_diffusion()
    synthetic_loader = get_synthetic_dataloader(
        batch_size=min(16, max(2, tuned_batch_size)),
        lookback_length=LOOKBACK_LENGTH,
        forecast_length=FORECAST_LENGTH,
        num_variables=N_VARIATES,
        num_samples=n_samples,
        num_workers=0 if smoke_test else 4,
        lookback_overlap=LOOKBACK_OVERLAP,
        cache_dir=synth_cache,
        skip_cross_var_aug=(N_VARIATES > 32),
        val_tail_n=n_val,
        synthetic_epoch_capacity=epoch_cap,
    )
    
    dataset = synthetic_loader.dataset
    train_subset = Subset(dataset, list(range(len(dataset) - n_val)))
    val_subset = Subset(dataset, list(range(len(dataset) - n_val, len(dataset))))
    if not _ddp_enabled:
        batch_size = select_diffusion_batch_size(
            phase_name='Diffusion pretrain',
            dataset=train_subset,
            device=device,
            itrans_guidance=itrans_guidance,
            max_candidate=max(
                tuned_batch_size,
                diffusion_probe_max_candidate(N_VARIATES, smoke_test),
            ),
        )
    effective_batch_size = batch_size // get_world_size() if _ddp_enabled else batch_size
    effective_batch_size = max(1, effective_batch_size)
    
    # Use DDP-aware data loaders
    train_loader, train_sampler = create_dataloader_ddp(
        train_subset, effective_batch_size, shuffle=True,
        num_workers=0 if smoke_test else 4
    )
    val_loader, _ = create_dataloader_ddp(
        val_subset, effective_batch_size, shuffle=False, num_workers=0
    )
    
    # Create model with guidance and wrap with DDP
    model = create_diffusion_model()
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
        
        set_realts_training_epoch(train_loader, epoch)
        
        t0 = time.time()
        
        # Train
        model.train()
        total_loss = 0.0
        n_batches = 0
        for past, future in train_loader:
            past, future = past.to(device), future.to(device)
            optimizer.zero_grad()
            base_model = unwrap_model(model)
            with amp_context():
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
                with amp_context():
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
    fixed_batch_size: Optional[int] = None,
    trial_ckpt_dir: Optional[str] = None,
) -> float:
    """Optuna objective for fine-tuning HP search (lr only; batch_size auto-probed or fixed).

    If ``trial_ckpt_dir`` is provided, this trial's best-epoch model state is saved
    to ``{trial_ckpt_dir}/_diff_ft_trial_{trial.number}_best.pt``. The caller picks
    the best study trial and promotes its file to the final ``best.pt`` — no
    separate "Phase 2C" retrain is performed.
    """
    lr = trial.suggest_float(
        'learning_rate', FINETUNE_HP_LR_MIN, FINETUNE_HP_LR_MAX, log=True,
    )
    if fixed_batch_size is not None:
        batch_size = fixed_batch_size
    else:
        batch_size = trial.suggest_categorical('batch_size', [2, 4] if smoke_test else FINETUNE_BATCH_SIZES)
    
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
    n_iv = len(variate_indices)
    itrans_model = load_itransformer_from_checkpoint(itrans_checkpoint, n_iv, device)
    itrans_guidance = iTransformerGuidance(itrans_model)
    
    # Load pretrained diffusion (skip guidance keys — keep the attached one)
    model = create_diffusion_model().to(device)
    model.set_guidance_model(itrans_guidance)
    ckpt = torch.load(pretrained_path, map_location=device, weights_only=False)
    load_diffusion_state_keep_attached_guidance(model, ckpt['model_state_dict'])

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    epochs = HP_TUNE_EPOCHS if not smoke_test else 1
    patience = HP_TUNE_PATIENCE if not smoke_test else 1
    early_stop = EarlyStopping(patience=patience)
    best_val_loss = float('inf')

    trial_ckpt_path: Optional[str] = None
    if trial_ckpt_dir is not None:
        os.makedirs(trial_ckpt_dir, exist_ok=True)
        trial_ckpt_path = os.path.join(trial_ckpt_dir, f'_diff_ft_trial_{trial.number}_best.pt')

    for epoch in range(epochs):
        model.train()
        for past, future in train_loader:
            past, future = past.to(device), future.to(device)
            optimizer.zero_grad()
            with amp_context():
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
                with amp_context():
                    loss = model.get_loss(past, future)
                val_loss += loss.item()
                n_batches += 1
        val_loss /= max(n_batches, 1)
        
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if trial_ckpt_path is not None and is_main_process():
                save_checkpoint(
                    unwrap_model(model), optimizer, epoch, float('nan'), val_loss,
                    {
                        'tuned_params': {'learning_rate': lr, 'batch_size': batch_size},
                        'trial_number': trial.number,
                    },
                    trial_ckpt_path,
                )

        if early_stop(val_loss):
            break

    return best_val_loss


def _promote_best_trial_to_final(
    study,
    subset_dir: str,
    subset_info: Dict,
    dataset_name: str,
    norm_stats: Dict,
    fixed_batch_size: int,
) -> Tuple[str, Dict]:
    """Copy the best Optuna trial's checkpoint to ``{subset_dir}/best.pt``, write
    metadata.json, clean up other trials' temp files, and return the final
    checkpoint path + a small train-metrics dict.

    Replaces the old "Phase 2C" full retrain — we just keep the best trial.
    """
    if study.best_trial is None:
        raise RuntimeError("Optuna study has no successful trials")

    subset_id = subset_info['subset_id']
    variate_indices = subset_info['variate_indices']
    best_num = study.best_trial.number
    best_val_loss = float(study.best_value)
    tuned_params = dict(study.best_params)
    tuned_params['batch_size'] = fixed_batch_size

    src = os.path.join(subset_dir, f'_diff_ft_trial_{best_num}_best.pt')
    if not os.path.exists(src):
        raise RuntimeError(f"Best trial checkpoint missing: {src}")

    dst = os.path.join(subset_dir, 'best.pt')
    if is_main_process():
        os.makedirs(subset_dir, exist_ok=True)
        import shutil
        shutil.copy2(src, dst)
        with open(os.path.join(subset_dir, 'metadata.json'), 'w') as f:
            json.dump({
                'subset_id': subset_id,
                'dataset_name': dataset_name,
                'variate_indices': variate_indices,
                'variate_names': subset_info.get('variate_names', []),
                'norm_mean': norm_stats['mean'].tolist(),
                'norm_std': norm_stats['std'].tolist(),
                'tuned_params': tuned_params,
                'best_trial': best_num,
                'best_val_loss': best_val_loss,
            }, f, indent=2)
        for fn in os.listdir(subset_dir):
            if fn.startswith('_diff_ft_trial_') and fn.endswith('_best.pt'):
                try:
                    os.remove(os.path.join(subset_dir, fn))
                except OSError:
                    pass
    barrier()

    return dst, {'best_val_loss': best_val_loss, 'best_trial': best_num}


def finetune_on_dataset(*args, **kwargs):
    """Removed. The Phase 2C full-finetune step has been eliminated; the best
    Phase 2B Optuna trial's checkpoint is now reused as the final fine-tuned
    model. See ``_promote_best_trial_to_final``."""
    raise RuntimeError(
        "finetune_on_dataset() was removed — Phase 2B's best trial checkpoint is "
        "the final model. Use _promote_best_trial_to_final() after study.optimize()."
    )


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
    
    gen_kwargs = {'num_ddim_steps': 5} if smoke_test else {}

    K = getattr(model.config, 'lookback_overlap', 0)

    with torch.no_grad():
        for batch_idx, (past, future) in enumerate(test_loader):
            if batch_idx >= n_batches:
                break
            
            past = past.to(device)
            
            # Single sample
            torch.manual_seed(42 + batch_idx)
            result = model.generate(past, **gen_kwargs)
            all_preds_single.append(result['prediction'].cpu())
            
            # Averaged (skip in smoke test — 1 sample is enough to verify the path)
            if smoke_test:
                all_preds_avg.append(result['prediction'].cpu())
            else:
                samples = []
                for _ in range(n_samples):
                    result = model.generate(past, **gen_kwargs)
                    samples.append(result['prediction'].cpu())
                all_preds_avg.append(torch.stack(samples).mean(dim=0))
            
            # Trim overlap from target so it matches the H-step forecast
            if K > 0:
                future = future[..., K:]
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


def _subset_results_path(results_dir: str, subset_id: str) -> str:
    """Return path to the canonical results.json for a subset."""
    return os.path.join(results_dir, subset_id, 'results.json')


def _load_subset_results(results_dir: str, subset_id: str) -> dict:
    path = _subset_results_path(results_dir, subset_id)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_subset_results(results_dir: str, subset_id: str, data: dict):
    subset_dir = os.path.join(results_dir, subset_id)
    os.makedirs(subset_dir, exist_ok=True)
    with open(os.path.join(subset_dir, 'results.json'), 'w') as f:
        json.dump(data, f, indent=2)


def save_eval_results(subset_id, dataset_name, variate_indices, train_metrics, eval_results, results_dir):
    """Save diffusion evaluation results to per-subset subdirectory."""
    data = _load_subset_results(results_dir, subset_id)
    data.update({
        'subset_id': subset_id,
        'dataset': dataset_name,
        'variate_indices': variate_indices,
        'train_metrics': train_metrics,
        'eval_metrics': eval_results,
        'evaluated_at': datetime.now().isoformat(),
    })
    _save_subset_results(results_dir, subset_id, data)
    update_summary_csv(results_dir)


def update_summary_csv(results_dir):
    """Rebuild summary CSV by walking per-subset subdirectories."""
    rows = []
    results_path = Path(results_dir)
    for subset_dir in sorted(results_path.iterdir()):
        if not subset_dir.is_dir():
            continue
        rfile = subset_dir / 'results.json'
        if not rfile.exists():
            continue
        try:
            with open(rfile) as f:
                data = json.load(f)
            if 'eval_metrics' not in data:
                continue
            m = data['eval_metrics']
            itrans = data.get('itransformer_metrics', {})
            row = {
                'subset_id': data['subset_id'],
                'dataset': data['dataset'],
                'best_val_loss': data.get('train_metrics', {}).get('best_val_loss'),
                'single_mse': m['single']['mse'],
                'single_mae': m['single']['mae'],
                'avg_mse': m['averaged']['mse'],
                'avg_mae': m['averaged']['mae'],
                'avg_trend_acc': m['averaged'].get('trend_accuracy'),
                'itrans_mse': itrans.get('mse'),
                'itrans_mae': itrans.get('mae'),
                'itrans_trend_acc': itrans.get('trend_accuracy'),
            }
            rows.append(row)
        except Exception:
            continue

    if rows:
        df = pd.DataFrame(rows).sort_values(['dataset', 'subset_id'])
        df.to_csv(os.path.join(results_dir, 'summary.csv'), index=False)


# ============================================================================
# iTransformer Baseline Evaluation
# ============================================================================

def evaluate_itransformer_baseline(
    subset_id: str,
    dataset_name: str,
    variate_indices: List[int],
    itrans_checkpoint: str,
    results_dir: str,
    device: torch.device,
    smoke_test: bool = False,
) -> Dict:
    """Run iTransformer-only forecast on test set and save to itransformer_baseline.json.

    Reuses the same test split as diffusion eval so the numbers are directly
    comparable. Results are merged into a single baseline file so summarize_results.py
    can produce the comparison table automatically.
    """
    _, _, test_ds, _ = load_dataset(dataset_name, variate_indices, stride=LOOKBACK_LENGTH)
    if smoke_test:
        test_ds = Subset(test_ds, list(range(min(2, len(test_ds)))))
    test_loader = DataLoader(test_ds, batch_size=8 if not smoke_test else 2, shuffle=False)

    n_iv = len(variate_indices)
    itrans_model = load_itransformer_from_checkpoint(itrans_checkpoint, n_iv, device)

    all_preds, all_targets = [], []
    with torch.no_grad():
        for past, future in test_loader:
            past = past.to(device)
            B, C, L = past.shape
            x_enc = past.permute(0, 2, 1)
            seq_sl = getattr(itrans_model, 'seq_len', L)
            if x_enc.shape[1] > seq_sl:
                x_enc = x_enc[:, -seq_sl:, :]
            x_dec = torch.zeros(B, FORECAST_LENGTH, C, device=device, dtype=past.dtype)
            output = itrans_model(x_enc, None, x_dec, None)
            if isinstance(output, tuple):
                output = output[0]
            all_preds.append(output.permute(0, 2, 1).cpu())
            # Strip overlap from target to match H-step prediction
            if LOOKBACK_OVERLAP > 0:
                future = future[..., LOOKBACK_OVERLAP:]
            all_targets.append(future)

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    mse = torch.nn.functional.mse_loss(preds, targets).item()
    mae = torch.nn.functional.l1_loss(preds, targets).item()
    pred_diff = preds[:, :, 1:] - preds[:, :, :-1]
    tgt_diff = targets[:, :, 1:] - targets[:, :, :-1]
    trend_acc = ((pred_diff > 0) == (tgt_diff > 0)).float().mean().item()

    metrics = {'mse': mse, 'mae': mae, 'trend_accuracy': trend_acc}
    logger.info(f"[{subset_id}] iTransformer baseline: MSE={mse:.4f}, MAE={mae:.4f}, trend={trend_acc:.3f}")

    # Merge into the per-subset results.json (same file as diffusion eval)
    data = _load_subset_results(results_dir, subset_id)
    data.setdefault('subset_id', subset_id)
    data.setdefault('dataset', dataset_name)
    data.setdefault('variate_indices', variate_indices)
    data['itransformer_metrics'] = metrics
    data['itransformer_evaluated_at'] = datetime.now().isoformat()
    _save_subset_results(results_dir, subset_id, data)
    update_summary_csv(results_dir)

    return metrics


# ============================================================================
# Full-Dimensionality iTransformer Baseline (for high-variate comparison)
# ============================================================================

def train_full_dim_itransformer_baseline(
    dataset_name: str,
    epochs: int = 50,
    patience: int = 15,
    batch_size: int = 32,
    lr: float = 1e-4,
    smoke_test: bool = False,
) -> str:
    """Train an iTransformer on ALL columns of a dataset.

    Used as the comparison baseline for high-variate datasets:
    avg(subset diffusion models) vs single full-dim iTransformer.

    Returns path to the saved checkpoint.
    """
    n_cols = get_dataset_n_cols(dataset_name)
    logger.info("=" * 60)
    logger.info(f"FULL-DIM ITRANSFORMER BASELINE: {dataset_name} ({n_cols} vars)")
    logger.info(f"Epochs: {epochs}, LR: {lr}, Batch: {batch_size}")
    logger.info("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds, val_ds, test_ds, norm_stats = load_dataset(
        dataset_name, variate_indices=None,
        stride=24 if not smoke_test else LOOKBACK_LENGTH,
    )
    if smoke_test:
        train_ds = Subset(train_ds, list(range(min(4, len(train_ds)))))
        val_ds = Subset(val_ds, list(range(min(2, len(val_ds)))))
        test_ds = Subset(test_ds, list(range(min(2, len(test_ds)))))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = create_itransformer(num_vars=n_cols).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    criterion = nn.MSELoss()

    baseline_dir = os.path.join(CHECKPOINT_DIR, f'{dataset_name}-baseline')
    os.makedirs(baseline_dir, exist_ok=True)
    ckpt_path = os.path.join(baseline_dir, 'itransformer_full.pt')

    if smoke_test:
        epochs = 1
        patience_val = 1
    else:
        patience_val = patience

    early_stop = EarlyStopping(patience=patience_val)
    best_val_loss = float('inf')

    for epoch in range(epochs):
        t0 = time.time()
        train_loss = train_itransformer_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate_itransformer(model, val_loader, criterion, device)
        scheduler.step()

        logger.info(f"[{dataset_name}-baseline] Epoch {epoch+1}/{epochs} | "
                     f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | {time.time()-t0:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss,
                          {'dataset': dataset_name, 'n_cols': n_cols}, ckpt_path)

        if early_stop(val_loss):
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    # Evaluate on test set
    model_eval = create_itransformer(num_vars=n_cols).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_eval.load_state_dict(ckpt['model_state_dict'])
    model_eval.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for past, future in test_loader:
            past = past.to(device)
            x_enc = past.permute(0, 2, 1)
            sl = getattr(model_eval, 'seq_len', x_enc.shape[1])
            if x_enc.shape[1] > sl:
                x_enc = x_enc[:, -sl:, :]
            output = model_eval(x_enc, None, None, None)
            if isinstance(output, tuple):
                output = output[0]
            all_preds.append(output.permute(0, 2, 1).cpu())
            if LOOKBACK_OVERLAP > 0:
                future = future[..., LOOKBACK_OVERLAP:]
            all_targets.append(future)

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    mse = torch.nn.functional.mse_loss(preds, targets).item()
    mae = torch.nn.functional.l1_loss(preds, targets).item()

    metrics = {'mse': mse, 'mae': mae, 'n_cols': n_cols}
    logger.info(f"[{dataset_name}-baseline] Test MSE={mse:.4f}, MAE={mae:.4f}")

    # Save results
    results_path = os.path.join(RESULTS_DIR, f'{dataset_name}-baseline', 'results.json')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump({
            'dataset': dataset_name,
            'type': 'full_dim_itransformer_baseline',
            'n_cols': n_cols,
            'metrics': metrics,
            'checkpoint': ckpt_path,
            'evaluated_at': datetime.now().isoformat(),
        }, f, indent=2)

    return ckpt_path


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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    recombine_traffic_data()
    
    # Load or create manifest
    if resume and os.path.exists(MANIFEST_PATH):
        manifest = TrainingManifest.load()
        logger.info(f"Resuming from manifest (created: {manifest.created_at})")
    else:
        manifest = TrainingManifest(seed=seed, created_at=datetime.now().isoformat())
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    device = get_device()
    logger.info(f"Using device: {device}")
    
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
        pretrain_samples = 4  # Ultra minimal
        itrans_pretrain_epochs = 1
        pretrain_patience = 1
    else:
        n_itrans_trials = N_ITRANS_HP_TRIALS
        n_diff_trials = N_DIFFUSION_HP_TRIALS
        n_finetune_trials = N_FINETUNE_HP_TRIALS
        pretrain_samples = resolve_pretrain_virtual_dataset_size(False)
        itrans_pretrain_epochs = PRETRAIN_EPOCHS
        pretrain_patience = PRETRAIN_PATIENCE
    
    # =========== PHASE 1A: iTransformer HP Tuning (best model saved directly) ===========
    itrans_tune_ckpt = os.path.join(CHECKPOINT_DIR, 'itrans_hp_best.pt')
    if not manifest.itrans_hp_done:
        manifest.itrans_best_params, _ = run_itransformer_hp_tuning(
            n_itrans_trials, smoke_test, checkpoint_dir=CHECKPOINT_DIR,
        )
        manifest.itrans_hp_done = True
        manifest.save()
        log_wandb_hp_search('itransformer', manifest.itrans_best_params,
                           manifest.itrans_best_params.get('best_val_loss', 0), n_itrans_trials)
    else:
        logger.info(f"Using cached iTransformer params: {manifest.itrans_best_params}")

    # Phase 1B eliminated: use best HP model directly as itransformer checkpoint
    itrans_ckpt = os.path.join(CHECKPOINT_DIR, 'pretrained_itransformer.pt')
    if not manifest.itrans_checkpoint or not os.path.exists(itrans_ckpt):
        if os.path.exists(itrans_tune_ckpt):
            import shutil
            shutil.copy2(itrans_tune_ckpt, itrans_ckpt)
            logger.info(f"Using best HP tuning model as iTransformer checkpoint: {itrans_ckpt}")
        else:
            itrans_ckpt = pretrain_itransformer(
                manifest.itrans_best_params,
                n_samples=pretrain_samples,
                epochs=itrans_pretrain_epochs,
                patience=pretrain_patience,
                checkpoint_dir=CHECKPOINT_DIR,
                smoke_test=smoke_test,
            )
        manifest.itrans_checkpoint = itrans_ckpt
        manifest.save()
    else:
        logger.info(f"Using existing iTransformer checkpoint: {itrans_ckpt}")
    
    # =========== PHASE 1C: Diffusion HP tuning (best ckpt saved under CHECKPOINT_DIR) ===========
    diff_tune_ckpt = os.path.join(CHECKPOINT_DIR, 'diff_hp_best.pt')
    if not manifest.diffusion_hp_done:
        manifest.diffusion_best_params, _ = run_diffusion_hp_tuning(
            itrans_ckpt, n_diff_trials, smoke_test, checkpoint_dir=CHECKPOINT_DIR,
        )
        manifest.diffusion_hp_done = True
        manifest.save()
        log_wandb_hp_search('diffusion', manifest.diffusion_best_params,
                           manifest.diffusion_best_params.get('best_val_loss', 0), n_diff_trials)
    else:
        logger.info(f"Using cached Diffusion params: {manifest.diffusion_best_params}")

    # =========== Diffusion checkpoint: best from HP only (no separate full synthetic pretrain) ===========
    diff_ckpt = os.path.join(CHECKPOINT_DIR, 'pretrained_diffusion.pt')
    if not manifest.pretrain_complete or not os.path.exists(diff_ckpt):
        if os.path.exists(diff_tune_ckpt):
            import shutil
            shutil.copy2(diff_tune_ckpt, diff_ckpt)
            logger.info(f"Using best diffusion HP model as pretrained checkpoint: {diff_ckpt}")
        else:
            fallback_epochs = 1 if smoke_test else PRETRAIN_DIFFUSION_MAX_EPOCHS
            diff_ckpt = pretrain_diffusion(
                manifest.diffusion_best_params,
                itrans_ckpt,
                n_samples=pretrain_samples,
                epochs=fallback_epochs,
                patience=pretrain_patience,
                checkpoint_dir=CHECKPOINT_DIR,
                smoke_test=smoke_test,
            )
        manifest.pretrain_checkpoint = diff_ckpt
        manifest.pretrain_complete = True
        manifest.save()
    else:
        logger.info(f"Using existing Diffusion checkpoint: {diff_ckpt}")
    
    # =========== PHASE 2: Fine-tuning per Dataset (full variates only) ===========
    all_jobs = generate_all_dataset_jobs(seed=seed)
    job_list = list(all_jobs.values())
    if smoke_test:
        job_list = job_list[:1]  # Just 1 dataset for ultra-fast smoke test

    for job in job_list:
        dataset_name = job['dataset_id']
        variate_indices = job['variate_indices']

        try:
            # Probe max safe batch size once before HP trials
            n_iv = len(variate_indices)
            _p_itrans = load_itransformer_from_checkpoint(itrans_ckpt, n_iv, device)
            _p_guidance = iTransformerGuidance(_p_itrans)
            _p_ds, _, _, _ = load_dataset(dataset_name, variate_indices, stride=LOOKBACK_LENGTH)
            ft_diff_bs = select_diffusion_batch_size(
                phase_name=f'Diff FT HP ({dataset_name})',
                dataset=_p_ds,
                device=device,
                itrans_guidance=_p_guidance,
                max_candidate=diffusion_probe_max_candidate(len(variate_indices), smoke_test),
            )
            del _p_itrans, _p_guidance, _p_ds
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            subset_dir = os.path.join(CHECKPOINT_DIR, dataset_name)
            os.makedirs(subset_dir, exist_ok=True)
            logger.info(
                f"Phase 2B — diffusion finetune HP ({dataset_name}): "
                f"{n_finetune_trials} Optuna trials (N_FINETUNE_HP_TRIALS), bs={ft_diff_bs}, "
                f"epochs<={HP_TUNE_EPOCHS} patience={HP_TUNE_PATIENCE}; best trial → final ckpt..."
            )
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            def log_finetune_trial(study, trial):
                logger.info(f"[{dataset_name} HP] Trial {trial.number}/{n_finetune_trials}: "
                            f"loss={trial.value:.4f}, lr={trial.params['learning_rate']:.2e}, "
                            f"bs={ft_diff_bs}")

            study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
            study.optimize(
                lambda trial: finetune_hp_objective(
                    trial, dataset_name, variate_indices, diff_ckpt, itrans_ckpt, device, smoke_test,
                    fixed_batch_size=ft_diff_bs, trial_ckpt_dir=subset_dir,
                ),
                n_trials=n_finetune_trials,
                show_progress_bar=True,
                callbacks=[log_finetune_trial],
            )
            tuned_params = study.best_params
            tuned_params['batch_size'] = ft_diff_bs
            logger.info(f"Best params for {dataset_name}: {tuned_params}")

            # Reuse the best trial's checkpoint as the final fine-tuned model
            # (no separate Phase 2C retrain).
            _, _, _, norm_stats = load_dataset(dataset_name, variate_indices, stride=LOOKBACK_LENGTH)
            ckpt_path, train_metrics = _promote_best_trial_to_final(
                study, subset_dir,
                {'subset_id': dataset_name, 'variate_indices': variate_indices},
                dataset_name, norm_stats, ft_diff_bs,
            )
            
            # Evaluation
            if True:
                logger.info(f"Evaluating {dataset_name}...")
                n_iv = len(variate_indices)
                itrans_model = load_itransformer_from_checkpoint(itrans_ckpt, n_iv, device)
                itrans_guidance = iTransformerGuidance(itrans_model)
                
                model = create_diffusion_model().to(device)
                model.set_guidance_model(itrans_guidance)
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                load_diffusion_state_keep_attached_guidance(model, ckpt['model_state_dict'])

                _, _, test_ds, _ = load_dataset(dataset_name, variate_indices, stride=LOOKBACK_LENGTH)
                if smoke_test:
                    test_ds = Subset(test_ds, list(range(min(2, len(test_ds)))))
                test_loader = DataLoader(test_ds, batch_size=8 if not smoke_test else 2, shuffle=False)
                
                eval_results = evaluate_model(model, test_loader, device, n_samples=30, smoke_test=smoke_test)
                
                logger.info(f"[{dataset_name}] Single: MSE={eval_results['single']['mse']:.4f}, MAE={eval_results['single']['mae']:.4f}")
                logger.info(f"[{dataset_name}] Avg: MSE={eval_results['averaged']['mse']:.4f}, MAE={eval_results['averaged']['mae']:.4f}")
                
                save_eval_results(dataset_name, dataset_name, variate_indices,
                                {**train_metrics, 'tuned_params': tuned_params}, eval_results, RESULTS_DIR)
                
                # iTransformer-only baseline (for comparison table in summarize_results.py)
                try:
                    evaluate_itransformer_baseline(
                        dataset_name, dataset_name, variate_indices,
                        itrans_ckpt, RESULTS_DIR, device, smoke_test=smoke_test,
                    )
                except Exception as be:
                    logger.warning(f"iTransformer baseline eval failed for {dataset_name}: {be}")
                
                # Log to wandb
                log_wandb_eval_results(dataset_name, eval_results, train_metrics)
                log_wandb_model_checkpoint(ckpt_path, dataset_name)
            
        except KeyboardInterrupt:
            logger.info(f"\nInterrupted during {dataset_name}.")
            return
        except Exception as e:
            logger.error(f"Error with {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Trained {len(job_list)} models")
    logger.info("=" * 60)


# ============================================================================
# Mode-Specific Entry Points (Slurm / CLI orchestration calls these modes)
# ============================================================================

def find_existing_itrans_checkpoint(n_variates: int) -> Optional[str]:
    """Scan known cluster locations for a previously-trained V=n_variates iTransformer.

    Returns the first usable path found, or None if nothing exists.
    Checks (in order):
      1. The canonical local pretrain dir for this dim
      2. The storage roots used by past slurm jobs (SCRATCH / PROJECT variants)
      3. Any checkpoints/ subtree under the project root
    """
    # 1. Local canonical path
    local = os.path.join(pretrain_dir_for_dim(n_variates), 'itransformer.pt')
    if os.path.exists(local):
        return local

    # 2. Cluster storage roots referenced in slurm scripts
    scratch = os.environ.get('SCRATCH', '')
    project = os.environ.get('PROJECT', '')
    user = os.environ.get('USER', os.environ.get('LOGNAME', ''))

    candidate_roots = []
    if project and user:
        candidate_roots += [
            os.path.join(project, user, 'diffusion-tsf-fullvar', 'checkpoints'),
            os.path.join(project, user, 'diffusion-tsf', 'checkpoints'),
        ]
    if scratch:
        candidate_roots += [
            os.path.join(scratch, 'ts-sandbox', 'checkpoints'),
        ]
    # also check siblings of the current checkpoint dir
    candidate_roots.append(os.path.dirname(CHECKPOINT_DIR))

    dim_subdirs = [f'pretrained_dim{n_variates}', f'pretrain_dim{n_variates}']
    filenames   = ['itransformer.pt', 'pretrained_itransformer.pt']

    for root in candidate_roots:
        for subdir in dim_subdirs:
            for fname in filenames:
                p = os.path.join(root, subdir, fname)
                if os.path.exists(p):
                    if is_itrans_checkpoint_compatible(p, n_variates):
                        return p

    # 3. Broad project-tree search (limited depth to avoid being slow)
    project_root_local = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for fname in filenames:
        for dirpath, dirnames, files in os.walk(project_root_local):
            # skip venv and hidden dirs
            dirnames[:] = [d for d in dirnames if d not in ('.git', '.venv', 'venv', '__pycache__')]
            if fname in files:
                candidate = os.path.join(dirpath, fname)
                # lightweight sanity: the file must be a valid torch checkpoint
                try:
                    meta = torch.load(candidate, map_location='cpu', weights_only=False)
                    if 'model_state_dict' in meta and is_itrans_checkpoint_compatible(candidate, n_variates):
                        return candidate
                except Exception:
                    pass

    return None


def is_itrans_checkpoint_compatible(path: str, n_variates: int) -> bool:
    """Return True if checkpoint can be loaded into current iTransformer config."""
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        state = ckpt.get('model_state_dict')
        if state is None:
            return False
        last_err: Optional[Exception] = None
        for seq_len in (ITRANSFORMER_SEQ_LEN, LOOKBACK_LENGTH):
            model = create_itransformer(num_vars=n_variates, seq_len=seq_len).cpu()
            try:
                model.load_state_dict(state, strict=True)
                return True
            except RuntimeError as e:
                last_err = e
                continue
        logger.warning(f"Skipping incompatible iTransformer checkpoint {path}: {last_err}")
        return False
    except Exception as e:
        logger.warning(f"Skipping incompatible iTransformer checkpoint {path}: {e}")
        return False


def run_pretrain_mode(n_variates: int, smoke_test: bool = False, seed: int = 42):
    """Pretrain iTransformer + Diffusion for a specific dimensionality.

    Called once per unique dim by the shell script.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    recombine_traffic_data()

    dim_dir = pretrain_dir_for_dim(n_variates)
    os.makedirs(dim_dir, exist_ok=True)

    itrans_ckpt = os.path.join(dim_dir, 'itransformer.pt')
    diff_ckpt   = os.path.join(dim_dir, 'diffusion.pt')
    smoke_flag  = os.path.join(dim_dir, '.smoke_test')  # marks partial smoke-test checkpoints

    # If a previous smoke test left checkpoints here, wipe them so a real run
    # doesn't skip pretraining on the basis of a 1-epoch model.
    if not smoke_test and os.path.exists(smoke_flag):
        logger.info(f"  Removing smoke-test checkpoints from {dim_dir} — re-running for real")
        for f in [itrans_ckpt, diff_ckpt,
                  os.path.join(dim_dir, 'itrans_hp.json'),
                  os.path.join(dim_dir, 'diff_hp.json'),
                  os.path.join(dim_dir, 'itrans_hp_best.pt'),
                  os.path.join(dim_dir, 'diff_hp_best.pt'),
                  smoke_flag]:
            if os.path.exists(f):
                os.remove(f)

    n_itrans_trials = 1 if smoke_test else N_ITRANS_HP_TRIALS
    n_diff_trials = 1 if smoke_test else N_DIFFUSION_HP_TRIALS
    pretrain_samples = resolve_pretrain_virtual_dataset_size(smoke_test)
    itrans_pretrain_epochs = 1 if smoke_test else PRETRAIN_EPOCHS
    diff_pretrain_epochs = 1 if smoke_test else min(PRETRAIN_DIFFUSION_EPOCHS, PRETRAIN_DIFFUSION_MAX_EPOCHS)
    pretrain_patience = 1 if smoke_test else PRETRAIN_PATIENCE

    itrans_hp_path = os.path.join(dim_dir, 'itrans_hp.json')
    diff_hp_path   = os.path.join(dim_dir, 'diff_hp.json')

    logger.info(f"Pretraining dim={n_variates}")

    # Guard against stale checkpoints copied from a mismatched architecture
    if os.path.exists(itrans_ckpt) and not is_itrans_checkpoint_compatible(itrans_ckpt, n_variates):
        logger.info(f"  Removing incompatible iTransformer ckpt: {itrans_ckpt}")
        os.remove(itrans_ckpt)

    # Try to reuse an existing V=n_variates iTransformer from previous runs
    # (searches slurm storage roots and the project tree — see find_existing_itrans_checkpoint)
    if not os.path.exists(itrans_ckpt) and not smoke_test:
        found = find_existing_itrans_checkpoint(n_variates)
        if found:
            import shutil
            logger.info(f"  Found existing iTransformer checkpoint: {found}")
            logger.info(f"  Copying to {itrans_ckpt} — skipping iTransformer pretrain")
            os.makedirs(os.path.dirname(itrans_ckpt), exist_ok=True)
            shutil.copy2(found, itrans_ckpt)

    # Phase 1A: iTransformer HP tuning — cached to disk so reruns skip it
    itrans_tune_ckpt = os.path.join(dim_dir, 'itrans_hp_best.pt')
    if os.path.exists(itrans_hp_path):
        with open(itrans_hp_path) as f:
            best_itrans_params = json.load(f)
        logger.info(f"  iTransformer HP loaded from cache: {itrans_hp_path}")
    else:
        best_itrans_params, _ = run_itransformer_hp_tuning(
            n_itrans_trials, smoke_test, checkpoint_dir=dim_dir,
        )
        with open(itrans_hp_path, 'w') as f:
            json.dump(best_itrans_params, f, indent=2)

    # Phase 1B (eliminated): use best HP-tuning model directly as itransformer.pt
    if not os.path.exists(itrans_ckpt):
        if os.path.exists(itrans_tune_ckpt):
            import shutil
            shutil.copy2(itrans_tune_ckpt, itrans_ckpt)
            logger.info(f"  Using best HP tuning model as iTransformer checkpoint: {itrans_ckpt}")
        else:
            # fallback: HP cache from an older run without itrans_hp_best.pt
            fallback_epochs = 1 if smoke_test else PRETRAIN_EPOCHS
            pretrain_itransformer(
                best_itrans_params,
                n_samples=pretrain_samples,
                epochs=fallback_epochs,
                patience=pretrain_patience,
                checkpoint_dir=dim_dir,
                smoke_test=smoke_test,
            )
            saved = os.path.join(dim_dir, 'pretrained_itransformer.pt')
            if saved != itrans_ckpt and os.path.exists(saved):
                os.rename(saved, itrans_ckpt)
    else:
        logger.info(f"  iTransformer ckpt exists: {itrans_ckpt}")

    if not is_itrans_checkpoint_compatible(itrans_ckpt, n_variates):
        raise RuntimeError(
            f"iTransformer checkpoint remains incompatible after pretrain/reuse: {itrans_ckpt}"
        )

    # Phase 1B: Diffusion HP tuning — cached to disk so reruns skip it
    diff_tune_ckpt = os.path.join(dim_dir, 'diff_hp_best.pt')
    if os.path.exists(diff_hp_path):
        with open(diff_hp_path) as f:
            best_diff_params = json.load(f)
        logger.info(f"  Diffusion HP loaded from cache: {diff_hp_path}")
    else:
        best_diff_params, _ = run_diffusion_hp_tuning(
            itrans_ckpt, n_diff_trials, smoke_test, checkpoint_dir=dim_dir,
        )
        with open(diff_hp_path, 'w') as f:
            json.dump(best_diff_params, f, indent=2)

    # Phase 1C-2: use the best model saved during HP tuning directly — no separate full pretrain
    if not os.path.exists(diff_ckpt):
        if os.path.exists(diff_tune_ckpt):
            import shutil
            shutil.copy2(diff_tune_ckpt, diff_ckpt)
            logger.info(f"  Using best HP tuning model as diffusion checkpoint: {diff_ckpt}")
        else:
            # fallback: HP cache from an older run that didn't save diff_hp_best.pt —
            # the already-trained HP model is gone, so run a full training pass
            fallback_epochs = 1 if smoke_test else PRETRAIN_DIFFUSION_MAX_EPOCHS
            pretrain_diffusion(
                best_diff_params, itrans_ckpt,
                n_samples=pretrain_samples,
                epochs=fallback_epochs,
                patience=pretrain_patience,
                checkpoint_dir=dim_dir,
                smoke_test=smoke_test,
            )
            saved = os.path.join(dim_dir, 'pretrained_diffusion.pt')
            if saved != diff_ckpt and os.path.exists(saved):
                os.rename(saved, diff_ckpt)
    else:
        logger.info(f"  Diffusion ckpt exists: {diff_ckpt}")

    if smoke_test:
        # Mark so a subsequent real run knows to discard these
        open(smoke_flag, 'w').close()

    logger.info(f"Pretrain dim={n_variates} complete")


def run_finetune_mode(
    dataset_name: str,
    n_variates: int,
    variate_indices: Optional[List[int]] = None,
    subset_id: Optional[str] = None,
    smoke_test: bool = False,
    seed: int = 42,
):
    """Fine-tune + evaluate one full-dataset model."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    recombine_traffic_data()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dim_dir = pretrain_dir_for_dim(n_variates)
    itrans_ckpt = os.path.join(dim_dir, 'itransformer.pt')
    diff_ckpt   = os.path.join(dim_dir, 'diffusion.pt')
    smoke_flag  = os.path.join(dim_dir, '.smoke_test')

    if not os.path.exists(diff_ckpt):
        logger.error(f"Pretrained checkpoint not found: {diff_ckpt}")
        logger.error(f"Run --mode pretrain --n-variates {n_variates} first")
        sys.exit(1)

    if not smoke_test and os.path.exists(smoke_flag):
        logger.error(
            f"Pretrain checkpoints in {dim_dir} are from a smoke test. "
            f"Run --mode pretrain --n-variates {n_variates} first to replace them."
        )
        sys.exit(1)

    n_finetune_trials = 1 if smoke_test else N_FINETUNE_HP_TRIALS

    if variate_indices is None:
        variate_indices = generate_dataset_job(dataset_name)['variate_indices']
    if not subset_id:
        subset_id = dataset_name

    _finetune_and_eval_one_subset(
        {'subset_id': subset_id, 'variate_indices': variate_indices},
        dataset_name, diff_ckpt, itrans_ckpt,
        n_finetune_trials, device, smoke_test,
    )


def run_itransformer_finetune_hp_tuning(
    dataset_name: str,
    variate_indices: List[int],
    pretrained_ckpt: str,
    n_trials: int,
    device: torch.device,
    smoke_test: bool = False,
    checkpoint_dir: Optional[str] = None,
    subset_id: Optional[str] = None,
) -> Tuple[Dict, Optional[str]]:
    """HP tune iTransformer on real data.

    If ``ITRANS_REAL_COLD_START`` is True, ignore the synthetic warm-start so each
    trial trains from scratch (synthetic pretrain on this corpus tends to converge
    near a unit-variance mean predictor, which makes warm-started fine-tunes barely
    move). Returns (best_params, path_to_best_model_or_None).
    """
    label = subset_id or dataset_name
    warm = (None if ITRANS_REAL_COLD_START else pretrained_ckpt)
    logger.info("=" * 60)
    logger.info(f"iTrans Finetune HP Tuning: {label} ({n_trials} trials)")
    logger.info(
        f"Up to {ITRANS_HP_FINETUNE_MAX_EPOCHS} epochs per trial, patience={ITRANS_HP_FINETUNE_PATIENCE}, "
        f"warm_start={'no (cold start)' if warm is None else os.path.basename(warm)}"
    )
    logger.info("=" * 60)

    train_ds, val_ds, _, _ = load_dataset(
        dataset_name, variate_indices,
        stride=24 if not smoke_test else LOOKBACK_LENGTH,
    )
    if smoke_test:
        train_ds = Subset(train_ds, list(range(min(2, len(train_ds)))))
        val_ds = Subset(val_ds, list(range(min(2, len(val_ds)))))

    train_bs = select_itrans_batch_size(
        phase_name=f'iTransformer FT HP ({label})',
        dataset=train_ds,
        device=device,
        dropout=0.1,
        max_candidate=32 if smoke_test else 256,
    )
    train_loader = DataLoader(train_ds, batch_size=train_bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=min(train_bs, 32), shuffle=False, num_workers=0)

    _best_state: dict = {'model_state': None, 'val_loss': float('inf')}

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2),
    )

    def log_trial(study, trial):
        logger.info(f"[iTrans FT HP {label}] Trial {trial.number}/{n_trials}: "
                   f"loss={trial.value:.4f}, lr={trial.params['learning_rate']:.2e}, "
                   f"dropout={trial.params['dropout']:.3f}")

    study.optimize(
        lambda trial: itrans_hp_objective(
            trial, train_loader, val_loader, device, smoke_test,
            fixed_batch_size=train_bs, best_state=_best_state,
            pretrained_ckpt=warm,
            max_epochs=ITRANS_HP_FINETUNE_MAX_EPOCHS,
            early_stop_patience=ITRANS_HP_FINETUNE_PATIENCE,
        ),
        n_trials=n_trials,
        show_progress_bar=False,
        callbacks=[log_trial],
    )

    best_params = study.best_params
    best_params['batch_size'] = train_bs
    logger.info(f"Best iTrans FT params for {label}: lr={best_params['learning_rate']:.2e}, "
               f"dropout={best_params['dropout']:.3f} → val_loss={_best_state.get('val_loss', float('inf')):.4f}")

    ckpt_path = None
    if checkpoint_dir is not None and _best_state.get('model_state') is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(checkpoint_dir, f'{label}_itrans_ft_hp_best.pt')
        torch.save({'model_state_dict': _best_state['model_state'], 'best_params': best_params}, ckpt_path)
        logger.info(f"  Saved best iTrans FT HP model → {ckpt_path} (val_loss={_best_state['val_loss']:.4f})")

    return best_params, ckpt_path



def _finetune_and_eval_one_subset(
    subset_info, dataset_name, diff_ckpt, itrans_ckpt,
    n_finetune_trials, device, smoke_test,
):
    """Internal: HP tune, fine-tune, and evaluate a single subset.

    Three phases:
      A. HP tune iTransformer on real data (warm-start from pretrained) -> best weights promoted.
      B. HP tune Diffusion on real data (finetuned iTrans as guidance).
      C. Full Diffusion finetune on real data (finetuned iTrans as guidance).
    """
    subset_id = subset_info['subset_id']
    variate_indices = subset_info['variate_indices']

    # Preflight: check dataset has enough rows before wasting a trial slot
    try:
        load_dataset(dataset_name, variate_indices, stride=LOOKBACK_LENGTH)
    except ValueError as ve:
        logger.warning(f"Skipping {subset_id}: {ve}")
        return

    try:
        n_itrans_ft_trials = 1 if smoke_test else N_ITRANS_HP_TRIALS

        # Phase A: iTransformer HP tune on real data (warm-start from pretrained)
        itrans_hp_cache = os.path.join(CHECKPOINT_DIR, f'{subset_id}_itrans_ft_hp.json')
        itrans_tune_ckpt = os.path.join(CHECKPOINT_DIR, f'{subset_id}_itrans_ft_hp_best.pt')
        ft_itrans_ckpt = os.path.join(CHECKPOINT_DIR, f'{subset_id}_itransformer_finetuned.pt')

        if os.path.exists(ft_itrans_ckpt):
            logger.info(f"  Using cached finetuned iTransformer: {ft_itrans_ckpt}")
        else:
            if os.path.exists(itrans_hp_cache):
                with open(itrans_hp_cache) as f:
                    itrans_ft_params = json.load(f)
                logger.info(f"  iTrans FT HP loaded from cache: {itrans_hp_cache}")
            else:
                itrans_ft_params, _ = run_itransformer_finetune_hp_tuning(
                    dataset_name, variate_indices, itrans_ckpt,
                    n_trials=n_itrans_ft_trials, device=device,
                    smoke_test=smoke_test, checkpoint_dir=CHECKPOINT_DIR,
                    subset_id=subset_id,
                )
                with open(itrans_hp_cache, 'w') as f:
                    json.dump(itrans_ft_params, f, indent=2)
            
            if os.path.exists(itrans_tune_ckpt):
                import shutil
                shutil.copy2(itrans_tune_ckpt, ft_itrans_ckpt)
                logger.info(f"  Using best HP tuning model as finetuned iTransformer: {ft_itrans_ckpt}")
            else:
                # fallback for missing tune ckpt
                raise RuntimeError(f"Expected to find {itrans_tune_ckpt} but it was missing.")

        # Phase C: Diffusion HP search using finetuned iTransformer as guidance
        # Probe max safe batch size once before HP trials start
        _ft_itrans_model = load_itransformer_from_checkpoint(ft_itrans_ckpt, len(variate_indices), device)
        _ft_itrans_guidance = iTransformerGuidance(_ft_itrans_model)
        _probe_ds, _, _, _ = load_dataset(
            dataset_name, variate_indices, stride=LOOKBACK_LENGTH,
        )
        ft_diff_bs = select_diffusion_batch_size(
            phase_name=f'Diff FT HP ({subset_id})',
            dataset=_probe_ds,
            device=device,
            itrans_guidance=_ft_itrans_guidance,
            max_candidate=diffusion_probe_max_candidate(len(variate_indices), smoke_test),
        )
        del _ft_itrans_model, _ft_itrans_guidance, _probe_ds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        subset_dir = os.path.join(CHECKPOINT_DIR, subset_id)
        os.makedirs(subset_dir, exist_ok=True)
        logger.info(
            f"Phase 2B — diffusion finetune HP ({subset_id}): "
            f"{n_finetune_trials} Optuna trials (N_FINETUNE_HP_TRIALS), bs={ft_diff_bs}, "
            f"epochs<={HP_TUNE_EPOCHS} patience={HP_TUNE_PATIENCE}; best trial → final ckpt "
            f"(pretrain diffusion HP uses N_DIFFUSION_HP_TRIALS={N_DIFFUSION_HP_TRIALS})..."
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=2),
        )
        study.optimize(
            lambda trial: finetune_hp_objective(
                trial, dataset_name, variate_indices, diff_ckpt, ft_itrans_ckpt, device, smoke_test,
                fixed_batch_size=ft_diff_bs, trial_ckpt_dir=subset_dir,
            ),
            n_trials=n_finetune_trials,
            show_progress_bar=False,
            catch=(ValueError,),
        )
        if study.best_trial is None:
            logger.warning(f"All diffusion HP trials failed for {subset_id} — skipping")
            return
        tuned_params = study.best_params
        tuned_params['batch_size'] = ft_diff_bs
        logger.info(f"Best diffusion params for {subset_id}: {tuned_params}")

        # Reuse the best Phase 2B trial's checkpoint as the final fine-tuned model.
        _, _, _, norm_stats = load_dataset(dataset_name, variate_indices, stride=LOOKBACK_LENGTH)
        ckpt_path, train_metrics = _promote_best_trial_to_final(
            study, subset_dir, subset_info, dataset_name, norm_stats, ft_diff_bs,
        )

        # Evaluate diffusion model
        logger.info(f"Evaluating {subset_id}...")
        itrans_model = load_itransformer_from_checkpoint(ft_itrans_ckpt, len(variate_indices), device)
        itrans_guidance = iTransformerGuidance(itrans_model)

        model = create_diffusion_model().to(device)
        model.set_guidance_model(itrans_guidance)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        load_diffusion_state_keep_attached_guidance(model, ckpt['model_state_dict'])

        _, _, test_ds, _ = load_dataset(dataset_name, variate_indices, stride=LOOKBACK_LENGTH)
        if smoke_test:
            test_ds = Subset(test_ds, list(range(min(2, len(test_ds)))))
        test_loader = DataLoader(test_ds, batch_size=8 if not smoke_test else 2, shuffle=False)

        eval_results = evaluate_model(model, test_loader, device, n_samples=30, smoke_test=smoke_test)
        logger.info(f"[{subset_id}] Avg: MSE={eval_results['averaged']['mse']:.4f}, "
                     f"MAE={eval_results['averaged']['mae']:.4f}")

        save_eval_results(
            subset_id, dataset_name, variate_indices,
            {**train_metrics, 'tuned_params': tuned_params}, eval_results, RESULTS_DIR,
        )

        # Finetuned iTransformer as baseline for comparison
        try:
            evaluate_itransformer_baseline(
                subset_id, dataset_name, variate_indices,
                ft_itrans_ckpt, RESULTS_DIR, device, smoke_test=smoke_test,
            )
        except Exception as be:
            logger.warning(f"iTransformer baseline eval failed for {subset_id}: {be}")

    except KeyboardInterrupt:
        logger.info(f"\nInterrupted during {subset_id}.")
        raise
    except Exception as e:
        logger.error(f"Error with {subset_id}: {e}")
        import traceback
        traceback.print_exc()
        raise

def run_baseline_mode(dataset_name: str, smoke_test: bool = False):
    """Train full-dimensionality iTransformer baseline for a high-variate dataset."""
    recombine_traffic_data()
    train_full_dim_itransformer_baseline(dataset_name, smoke_test=smoke_test)


# ============================================================================
# CLI
# ============================================================================

def main():
    global logger, N_VARIATES, CHECKPOINT_DIR, RESULTS_DIR, MANIFEST_PATH, SYNTH_CACHE_DIR, GUIDANCE_PENALTY_WEIGHT
    global IMAGE_HEIGHT, UNET_CHANNELS, ATTENTION_LEVELS, DISABLE_CROSS_ATTENTION, LOOKBACK_LENGTH, FORECAST_LENGTH
    global MODEL_TYPE

    parser = argparse.ArgumentParser(description='Diffusion TSF Training Pipeline')
    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'pretrain', 'finetune', 'baseline', 'evaluate', 'status'],
                        help='Pipeline mode (default: full = run everything)')
    parser.add_argument('--n-variates', type=int, default=None,
                        help='Override variate count (default: auto per dataset)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Single dataset to process')
    parser.add_argument('--variate-indices', type=str, default=None,
                        help='Comma-separated variate indices for subset finetune')
    parser.add_argument('--subset-id', type=str, default=None,
                        help='Optional subset id label used in saved results')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--smoke-test', action='store_true', help='Quick validation run')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--status', action='store_true', help='Show status (legacy flag)')
    parser.add_argument('--ddp', action='store_true', help='Enable multi-GPU DDP training')
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--wandb-project', type=str, default='diffusion-tsf', help='Wandb project')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Override checkpoint directory')
    parser.add_argument('--results-dir', type=str, default=None,
                        help='Override results directory')
    parser.add_argument('--synth-cache-dir', type=str, default=None,
                        help='Shared synthetic pool cache directory for reuse across runs')
    parser.add_argument('--parallel-worker', type=int, default=None,
                        help='Parallel worker ID for multi-GPU Optuna (0-N)')
    parser.add_argument('--fresh', action='store_true',
                        help='Wipe manifest and checkpoints, start from scratch')
    parser.add_argument('--guidance-penalty-weight', type=float, default=GUIDANCE_PENALTY_WEIGHT,
                        help='Weight for guidance penalty loss (default from pipeline_config)')
    parser.add_argument('--image-height', type=int, default=IMAGE_HEIGHT,
                        help='Override image height')
    parser.add_argument('--unet-channels', type=str, default=None,
                        help='Comma-separated UNet channels (e.g. 64,128,256)')
    parser.add_argument('--attention-levels', type=str, default=None,
                        help='Comma-separated attention levels (e.g. 1,2)')
    parser.add_argument('--disable-cross-attention', action='store_true',
                        help='Disable cross-variate attention (fully univariate baseline)')
    parser.add_argument('--model-type', type=str, default=None, choices=['unet', 'dit'],
                        help="Diffusion backbone: 'unet' (default) or 'dit'")
    parser.add_argument('--lookback-length', type=int, default=LOOKBACK_LENGTH,
                        help='Override lookback length')
    parser.add_argument('--forecast-length', type=int, default=FORECAST_LENGTH,
                        help='Override forecast length')

    args = parser.parse_args()

    # Legacy flag compat
    if args.status:
        args.mode = 'status'

    if args.checkpoint_dir:
        CHECKPOINT_DIR = args.checkpoint_dir
        MANIFEST_PATH = os.path.join(CHECKPOINT_DIR, 'training_manifest.json')
    if args.results_dir:
        RESULTS_DIR = args.results_dir
    if args.synth_cache_dir:
        SYNTH_CACHE_DIR = args.synth_cache_dir

    if args.n_variates is not None:
        N_VARIATES = args.n_variates
    
    # Global overrides from CLI
    GUIDANCE_PENALTY_WEIGHT = args.guidance_penalty_weight
    IMAGE_HEIGHT = args.image_height
    if args.unet_channels:
        UNET_CHANNELS = [int(x.strip()) for x in args.unet_channels.split(',') if x.strip()]
    if args.attention_levels is not None:
        ATTENTION_LEVELS = [int(x.strip()) for x in args.attention_levels.split(',') if x.strip()]
    if args.disable_cross_attention:
        DISABLE_CROSS_ATTENTION = True
    if args.model_type is not None:
        MODEL_TYPE = args.model_type
    LOOKBACK_LENGTH = args.lookback_length
    FORECAST_LENGTH = args.forecast_length
    
    # DDP setup
    if args.ddp:
        if not setup_ddp():
            print("ERROR: --ddp flag set but DDP init failed.")
            sys.exit(1)
    
    if args.parallel_worker is not None:
        setup_parallel_worker(args.parallel_worker)
    
    logger = setup_logging()
    
    # ---- Mode dispatch ----
    
    if args.mode == 'status':
        if is_main_process():
            if os.path.exists(MANIFEST_PATH):
                m = TrainingManifest.load()
                print(f"Created: {m.created_at}")
                print(f"iTransformer HP done: {m.itrans_hp_done}")
                print(f"Diffusion HP done: {m.diffusion_hp_done}")
                print(f"Pretrain complete: {m.pretrain_complete}")
            else:
                print("No manifest found")
        return

    if args.mode == 'pretrain':
        nv = args.n_variates
        if nv is None:
            print("ERROR: --n-variates required for pretrain mode")
            sys.exit(1)
        N_VARIATES = nv
        try:
            if args.wandb:
                tags = ['smoke-test'] if args.smoke_test else []
                tags.append('mode-pretrain')
                init_wandb(
                    project=args.wandb_project,
                    config={
                        'seed': args.seed,
                        'smoke_test': args.smoke_test,
                        'resume': args.resume,
                        'mode': 'pretrain',
                        'n_variates': nv,
                    },
                    resume=args.resume,
                    tags=tags,
                )
            run_pretrain_mode(nv, smoke_test=args.smoke_test, seed=args.seed)
        finally:
            finish_wandb()
            cleanup_ddp()
        return

    if args.mode == 'finetune':
        if not args.dataset:
            print("ERROR: --dataset required for finetune mode")
            sys.exit(1)
        variate_indices = None
        if args.variate_indices:
            variate_indices = [int(x.strip()) for x in args.variate_indices.split(',') if x.strip()]
        nv = args.n_variates or (len(variate_indices) if variate_indices else get_dim_for_dataset(args.dataset))
        N_VARIATES = nv
        try:
            if args.wandb:
                tags = ['smoke-test'] if args.smoke_test else []
                tags.append('mode-finetune')
                init_wandb(
                    project=args.wandb_project,
                    config={
                        'seed': args.seed,
                        'smoke_test': args.smoke_test,
                        'resume': args.resume,
                        'mode': 'finetune',
                        'dataset': args.dataset,
                        'n_variates': nv,
                    },
                    resume=args.resume,
                    tags=tags,
                )
            run_finetune_mode(
                args.dataset,
                nv,
                variate_indices=variate_indices,
                subset_id=args.subset_id,
                smoke_test=args.smoke_test,
                seed=args.seed,
            )
        finally:
            finish_wandb()
            cleanup_ddp()
        return

    if args.mode == 'baseline':
        if not args.dataset:
            print("ERROR: --dataset required for baseline mode")
            sys.exit(1)
        run_baseline_mode(args.dataset, smoke_test=args.smoke_test)
        return

    if args.mode == 'evaluate':
        # Just rebuild summary from existing results
        update_summary_csv(RESULTS_DIR)
        logger.info(f"Summary updated: {os.path.join(RESULTS_DIR, 'summary.csv')}")
        return

    # ---- mode == 'full': legacy run-everything path ----
    if args.fresh:
        if os.path.exists(MANIFEST_PATH):
            os.remove(MANIFEST_PATH)
            logger.info(f"Removed old manifest: {MANIFEST_PATH}")
        for ckpt_file in ['pretrained_itransformer.pt', 'pretrained_diffusion.pt']:
            p = os.path.join(CHECKPOINT_DIR, ckpt_file)
            if os.path.exists(p):
                os.remove(p)
                logger.info(f"Removed old checkpoint: {p}")
        args.resume = False
    
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
