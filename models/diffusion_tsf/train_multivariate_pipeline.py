
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
import errno
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
from models.diffusion_tsf.dalia_data import (
    DALIA_CHANNEL_NAMES,
    DALIA_DEFAULT_FORECAST,
    DALIA_DEFAULT_LOOKBACK,
    DALIA_N_VARS,
    dalia_window_count,
    dalia_window_lengths,
    ensure_dalia_csv,
    load_dalia_dataset,
)
from models.diffusion_tsf.pipeline.data_subset import resolve_data_subset

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


def fixed_deterministic_anchor_hp() -> Tuple[float, float]:
    """Fixed anchor hyperparameters (CLI / pipeline_config; not Optuna-tuned)."""
    return DETERMINISTIC_ANCHOR_LAMBDA, DETERMINISTIC_ANCHOR_ALPHA


def anchor_kwargs_from_params(params: Optional[Dict] = None) -> Dict:
    """Kwargs for create_diffusion_model from CLI anchor settings."""
    del params  # kept for call-site compatibility; anchor HP is not tuned
    if not DETERMINISTIC_ANCHOR_LOSS:
        return {}
    anchor_lambda, anchor_alpha = fixed_deterministic_anchor_hp()
    return {
        'use_deterministic_anchor_loss': True,
        'deterministic_anchor_lambda': anchor_lambda,
        'deterministic_anchor_alpha': anchor_alpha,
    }


def diffusion_arch_config_dict() -> Dict[str, Any]:
    """Architecture/runtime flags needed to reconstruct diffusion checkpoints."""
    return {
        'image_height': IMAGE_HEIGHT,
        'coarse_image_height': COARSE_IMAGE_HEIGHT,
        'fine_image_height': FINE_IMAGE_HEIGHT,
        'finer_image_height': FINER_IMAGE_HEIGHT,
        'max_scale': MAX_SCALE,
        'window_norm_std_floor': WINDOW_NORM_STD_FLOOR,
        'use_dual_scale': USE_DUAL_SCALE,
        'use_triple_scale': USE_TRIPLE_SCALE,
        'diffusion_stage': DIFFUSION_STAGE,
        'dual_scale_fine_weight': DUAL_SCALE_FINE_WEIGHT,
        'dual_scale_independent_timesteps': DUAL_SCALE_INDEPENDENT_TIMESTEPS,
        'use_guidance_channel': USE_GUIDANCE_CHANNEL,
        'cfg_dropout': CFG_DROPOUT,
        'cfg_scale': CFG_SCALE,
        'use_cfg_inference': USE_CFG_INFERENCE,
        'disable_cross_attention': DISABLE_CROSS_ATTENTION,
        'cross_variate_context_bias': CROSS_VARIATE_CONTEXT_BIAS,
        'model_type': MODEL_TYPE,
        'diffusion_type': DIFFUSION_TYPE,
        'dit_patch_size': DIT_PATCH_SIZE,
        'dit_embed_dim': DIT_EMBED_DIM,
        'dit_depth': DIT_DEPTH,
        'dit_num_heads': DIT_NUM_HEADS,
        'dit_mlp_ratio': DIT_MLP_RATIO,
        'dit_dropout': DIT_DROPOUT,
        'use_window_normalization': USE_WINDOW_NORMALIZATION,
        'zero_guidance_forecast': ZERO_GUIDANCE_FORECAST,
        'window_stride': WINDOW_STRIDE,
    }


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


def _wandb_api_key_usable() -> bool:
    """True if WANDB_API_KEY is set and matches wandb's allowed charset."""
    import re
    key = os.environ.get("WANDB_API_KEY", "").strip()
    if not key:
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9_]+", key))


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

    if not _wandb_api_key_usable():
        if os.environ.get("WANDB_API_KEY", "").strip():
            logger.warning(
                "WANDB_API_KEY is set but invalid (wandb allows only A-Z, 0-9, _). "
                "Skipping wandb for this run."
            )
        else:
            logger.warning("WANDB_API_KEY not set; skipping wandb.")
        os.environ.pop("WANDB_API_KEY", None)
        _wandb_enabled = False
        return False
    
    # Build comprehensive config
    full_config = {
        # Training constants
        'lookback_length': LOOKBACK_LENGTH,
        'forecast_length': FORECAST_LENGTH,
        'image_height': IMAGE_HEIGHT,
        'max_scale': MAX_SCALE,
        'window_norm_std_floor': WINDOW_NORM_STD_FLOOR,
        'use_dual_scale': USE_DUAL_SCALE,
        'dual_scale_fine_weight': DUAL_SCALE_FINE_WEIGHT,
        'dual_scale_independent_timesteps': DUAL_SCALE_INDEPENDENT_TIMESTEPS,
        'cfg_dropout': CFG_DROPOUT,
        'cfg_scale': CFG_SCALE,
        'use_cfg_inference': USE_CFG_INFERENCE,
        'cross_variate_context_bias': CROSS_VARIATE_CONTEXT_BIAS,
        'n_variates': N_VARIATES,
        'diffusion_type': DIFFUSION_TYPE,
        'deterministic_anchor_loss': DETERMINISTIC_ANCHOR_LOSS,
        'deterministic_anchor_lambda': DETERMINISTIC_ANCHOR_LAMBDA,
        'deterministic_anchor_alpha': DETERMINISTIC_ANCHOR_ALPHA,
        'anchor_hp_lambda_min': ANCHOR_HP_LAMBDA_MIN,
        'anchor_hp_lambda_max': ANCHOR_HP_LAMBDA_MAX,
        'anchor_hp_alpha_min': ANCHOR_HP_ALPHA_MIN,
        'anchor_hp_alpha_max': ANCHOR_HP_ALPHA_MAX,
        'eval_sampler': EVAL_SAMPLER,
        'pretrain_epochs': PRETRAIN_EPOCHS,
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
        'itrans_paper_batch_size': ITRANS_PAPER_BATCH_SIZE,
        'itrans_paper_lr_grid': ITRANS_PAPER_LR_GRID,
        'itrans_paper_dropout': ITRANS_PAPER_DROPOUT,
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
        f'eval/{subset_id}/best_val_loss': train_metrics.get('best_val_loss', 0),
        f'eval/{subset_id}/final_epoch': train_metrics.get('final_epoch', 0),
    }
    if 'mse' in eval_results.get('averaged', {}):
        flat_metrics.update({
            f'eval/{subset_id}/avg_mse': eval_results['averaged']['mse'],
            f'eval/{subset_id}/avg_mae': eval_results['averaged']['mae'],
        })
    log_wandb(flat_metrics)

    # Table for comparison
    if hasattr(wandb, 'Table'):
        avg = eval_results.get('averaged', {})
        table_data = [[
            subset_id,
            eval_results['single']['mse'],
            eval_results['single']['mae'],
            avg.get('mse'),
            avg.get('mae'),
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
    COARSE_IMAGE_HEIGHT,
    FINE_IMAGE_HEIGHT,
    FINER_IMAGE_HEIGHT,
    MAX_SCALE,
    WINDOW_NORM_STD_FLOOR,
    LOOKBACK_OVERLAP,
    PAST_LOSS_WEIGHT,
    N_VARIATES_DEFAULT,
    PRETRAIN_EPOCHS,
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
    ITRANS_HP_FINETUNE_MAX_EPOCHS,
    ITRANS_REAL_COLD_START,
    ITRANS_PAPER_BATCH_SIZE,
    ITRANS_PAPER_LR_GRID,
    ITRANS_PAPER_DROPOUT,
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
    UNET_MAX_CHUNK_SIZE,
    DISABLE_CROSS_ATTENTION,
    USE_DUAL_SCALE,
    USE_TRIPLE_SCALE,
    DIFFUSION_STAGE,
    DUAL_SCALE_FINE_WEIGHT,
    DUAL_SCALE_INDEPENDENT_TIMESTEPS,
    USE_GUIDANCE_CHANNEL,
    CFG_DROPOUT,
    CFG_SCALE,
    USE_CFG_INFERENCE,
    MODEL_TYPE,
    DIFFUSION_TYPE,
    DIT_PATCH_SIZE,
    DIT_EMBED_DIM,
    DIT_DEPTH,
    DIT_NUM_HEADS,
    DIT_MLP_RATIO,
    DIT_DROPOUT,
    CROSS_VARIATE_CONTEXT_BIAS,
    GUIDANCE_PENALTY_WEIGHT,
    DETERMINISTIC_ANCHOR_LOSS,
    DETERMINISTIC_ANCHOR_LAMBDA,
    DETERMINISTIC_ANCHOR_ALPHA,
    USE_WINDOW_NORMALIZATION,
    ZERO_GUIDANCE_FORECAST,
    WINDOW_STRIDE,
    ANCHOR_HP_LAMBDA_MIN,
    ANCHOR_HP_LAMBDA_MAX,
    ANCHOR_HP_ALPHA_MIN,
    ANCHOR_HP_ALPHA_MAX,
    EVAL_NUM_SAMPLES,
    EVAL_SAMPLER,
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
    # PeMS benchmarks ship as NPZ (iTransformer Dataset_PEMS); see scripts/fetch_pems_solar.sh
    'PeMS': ('PeMS/PEMS04.npz', None, 24),
    'solar_Alabama': ('solar_Alabama/solar_Alabama.csv', 'Unnamed: 0', 96),
    'dalia': ('dalia/dalia.csv', 'window_id', 96),
}

# First existing path wins (under DATASETS_DIR).
PEMS_DATA_CANDIDATES = (
    'PeMS/PEMS04.npz',
    'PeMS/PEMS08.npz',
    'PeMS/PEMS03.npz',
    'PeMS/PEMS07.npz',
    'PeMS/PeMS.csv',
)

SOLAR_DATA_CANDIDATES = (
    'solar_Alabama/solar_Alabama.csv',
    'solar_Alabama/solar_AL.csv',
    'Solar/solar_AL.csv',
)


def _datasets_root() -> str:
    return os.path.abspath(os.path.expanduser(DATASETS_DIR))


def _path_is_file(path: str, retries: int = 3, delay_s: float = 0.5) -> bool:
    """os.path.isfile with brief retries for NFS stale-file-handle (errno 116)."""
    for attempt in range(retries):
        try:
            return os.path.isfile(path)
        except OSError as exc:
            if exc.errno == errno.ESTALE and attempt + 1 < retries:
                time.sleep(delay_s)
                continue
            raise
    return False


def _resolve_registry_path(dataset_name: str) -> Tuple[str, Optional[str]]:
    """Return (absolute path, date_col or None for NPZ/headerless)."""
    if dataset_name == 'PeMS':
        for rel in PEMS_DATA_CANDIDATES:
            path = os.path.join(_datasets_root(), rel)
            if _path_is_file(path):
                date_col = None if rel.endswith('.npz') else DATASET_REGISTRY['PeMS'][1]
                return path, date_col
        raise FileNotFoundError(
            f"No PeMS file under {_datasets_root()}/PeMS/. "
            f"Run setup/fetch_pems_solar.sh from the repo root (login node)."
        )
    if dataset_name == 'solar_Alabama':
        for rel in SOLAR_DATA_CANDIDATES:
            path = os.path.join(_datasets_root(), rel)
            if _path_is_file(path):
                return path, DATASET_REGISTRY['solar_Alabama'][1]
        raise FileNotFoundError(
            f"No solar file under {_datasets_root()}/. "
            f"Run setup/fetch_pems_solar.sh from the repo root (login node)."
        )
    if dataset_name == 'dalia':
        path = ensure_dalia_csv(_datasets_root())
        return path, DATASET_REGISTRY['dalia'][1]
    rel, date_col, _ = DATASET_REGISTRY[dataset_name]
    path = os.path.join(_datasets_root(), rel)
    if not _path_is_file(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    return path, date_col


def _load_pems_npz(path: str) -> np.ndarray:
    raw = np.load(path, allow_pickle=True)
    data = raw['data']
    if data.ndim == 3:
        data = data[:, :, 0]
    return np.asarray(data, dtype=np.float32)


def _load_solar_lines(path: str) -> np.ndarray:
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append([float(x) for x in line.split(',')])
    if not rows:
        raise ValueError(f"Empty solar dataset: {path}")
    return np.stack(rows, axis=0).astype(np.float32)


def _load_dataset_array(path: str, date_col: Optional[str]) -> np.ndarray:
    if path.endswith('.npz'):
        return _load_pems_npz(path)
    try:
        df_head = pd.read_csv(path, nrows=1)
    except Exception:
        df_head = None
    if df_head is not None and date_col and date_col in df_head.columns:
        df = pd.read_csv(path)
        cols = [c for c in df.columns if c != date_col]
        return df[cols].values.astype(np.float32)
    if df_head is not None and len(df_head.columns) > 1:
        df = pd.read_csv(path)
        cols = list(df.columns)
        if date_col and date_col in cols:
            cols = [c for c in cols if c != date_col]
        return df[cols].values.astype(np.float32)
    return _load_solar_lines(path)


def _dataset_variate_names(path: str, date_col: Optional[str], n_cols: int) -> List[str]:
    if path.endswith('.npz'):
        return [f"var_{i}" for i in range(n_cols)]
    try:
        df = pd.read_csv(path, nrows=1)
        if date_col and date_col in df.columns:
            return [c for c in df.columns if c != date_col]
        return list(df.columns)
    except Exception:
        return [f"var_{i}" for i in range(n_cols)]


def dataset_window_lengths(dataset_name: str) -> Tuple[int, int]:
    """Per-dataset (lookback, forecast) for finetune/eval; pretrain stays on pipeline defaults."""
    if dataset_name == 'dalia':
        return dalia_window_lengths()
    return LOOKBACK_LENGTH, FORECAST_LENGTH


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
    if dataset_name == 'dalia':
        ensure_dalia_csv(DATASETS_DIR)
        return DALIA_N_VARS
    path, date_col = _resolve_registry_path(dataset_name)
    if path.endswith('.npz'):
        data = _load_pems_npz(path)
        return int(data.shape[1])
    try:
        df = pd.read_csv(path, nrows=1)
        if date_col and date_col in df.columns:
            return sum(1 for c in df.columns if c != date_col)
        return len(df.columns)
    except Exception:
        return int(_load_solar_lines(path).shape[1])


_DATASET_SHAPE_CACHE: Dict[Tuple[str, str], Tuple[int, int]] = {}


def get_dataset_shape(dataset_name: str) -> Tuple[int, int]:
    """Return raw row/variate counts without materializing the full numeric array."""
    key = (DATASETS_DIR, dataset_name)
    if key in _DATASET_SHAPE_CACHE:
        return _DATASET_SHAPE_CACHE[key]
    if dataset_name == 'dalia':
        ensure_dalia_csv(DATASETS_DIR)
        n_win = dalia_window_count(DATASETS_DIR)
        shape = (n_win, DALIA_N_VARS)
    else:
        path, date_col = _resolve_registry_path(dataset_name)
        data = _load_dataset_array(path, date_col)
        shape = (int(data.shape[0]), int(data.shape[1]))
    _DATASET_SHAPE_CACHE[key] = shape
    return shape


def resolve_pipeline_data_subset(state) -> Dict[str, Any]:
    """Resolve state.data_subset and write concrete variates/strides to state."""
    base_indices = state.variate_indices
    if base_indices is None:
        base_indices = generate_dataset_job(state.dataset)["variate_indices"]
    raw_rows, raw_variates = get_dataset_shape(state.dataset)
    policy = dict(state.data_subset or {})
    target_dataset = policy.get("target_dataset")
    target_rows = target_variates = None
    if target_dataset:
        try:
            target_rows, target_variates = get_dataset_shape(str(target_dataset))
        except Exception as exc:
            raise ValueError(f"Could not resolve data_subset target_dataset={target_dataset!r}: {exc}") from exc
    resolved = resolve_data_subset(
        dataset_name=state.dataset,
        raw_rows=raw_rows,
        raw_variates=raw_variates,
        base_variate_indices=list(base_indices),
        default_subset_id=state.subset_id,
        default_window_stride=state.window_stride,
        seed=state.seed,
        policy=policy,
        target_rows=target_rows,
        target_variates=target_variates,
    )
    state.variate_indices = list(resolved["variate_indices"])
    state.n_variates = int(resolved["n_variates"])
    state.subset_id = str(resolved["subset_id"])
    state.data_subset_resolved = resolved
    return resolved


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

def _purge_repo_utils_for_itransformer() -> None:
    """Repo ``utils/`` (eval scripts) shadows iTransformer's ``utils.masking`` imports."""
    repo_utils = Path(__file__).resolve().parents[2] / "utils"
    drop: List[str] = []
    for name, mod in list(sys.modules.items()):
        if name != "utils" and not name.startswith("utils."):
            continue
        mod_file = getattr(mod, "__file__", None)
        if mod_file is None:
            if name == "utils":
                drop.append(name)
            continue
        try:
            mod_path = Path(mod_file).resolve()
        except OSError:
            continue
        if "iTransformer" in mod_path.parts:
            continue
        if mod_path == repo_utils or repo_utils in mod_path.parents:
            drop.append(name)
    for name in drop:
        sys.modules.pop(name, None)


def get_itransformer_class():
    """Dynamically load iTransformer model class."""
    itrans_path = os.path.join(script_dir, '..', 'iTransformer', 'model', 'iTransformer.py')
    itrans_path = os.path.abspath(itrans_path)
    
    # Add iTransformer directory to path for internal imports
    itrans_dir = os.path.join(script_dir, '..', 'iTransformer')
    itrans_dir = os.path.abspath(itrans_dir)
    _purge_repo_utils_for_itransformer()
    sys.path = [p for p in sys.path if os.path.abspath(p) != itrans_dir]
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
    proj_key = 'projector.weight'
    if proj_key in state:
        ckpt_pred_len = int(state[proj_key].shape[0])
    else:
        ckpt_pred_len = FORECAST_LENGTH

    model = create_itransformer(
        seq_len=ckpt_seq_len,
        pred_len=ckpt_pred_len,
        num_vars=num_vars,
        dropout=dropout,
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
    model_state = model.state_dict()
    filtered = {}
    for k, v in ckpt_state.items():
        if k.startswith('guidance_model.'):
            continue
        if k in model_state and model_state[k].shape != v.shape:
            logger.warning(f"Skipping {k} due to shape mismatch: ckpt {v.shape} vs model {model_state[k].shape}")
            continue
        filtered[k] = v
        
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    leaked = [k for k in (missing or []) if not k.startswith('guidance_model.')]
    if leaked:
        logger.warning(f"Diffusion ckpt missing non-guidance keys: {leaked[:5]}...")
    real_unexpected = [k for k in (unexpected or []) if not k.startswith('guidance_model.')]
    if real_unexpected:
        logger.warning(f"Diffusion ckpt has unexpected keys: {real_unexpected[:5]}...")


# ============================================================================
# Diffusion Model Creation (with guidance support)
# ============================================================================

def create_diffusion_model(
    n_variates: int = None,
    lookback: Optional[int] = None,
    horizon: Optional[int] = None,
    lookback_overlap: int = LOOKBACK_OVERLAP,
    past_loss_weight: float = PAST_LOSS_WEIGHT,
    guidance_penalty_weight: Optional[float] = None,
    diffusion_type: str = None,
    model_type: str = None,
    use_deterministic_anchor_loss: Optional[bool] = None,
    deterministic_anchor_lambda: Optional[float] = None,
    deterministic_anchor_alpha: Optional[float] = None,
    cross_variate_context_bias: Optional[float] = None,
    diffusion_stage: Optional[str] = None,
    use_guidance_channel: Optional[bool] = None,
    max_scale: Optional[float] = None,
    window_norm_std_floor: Optional[float] = None,
    binary_noise_schedule: Optional[str] = None,
    prediction_target: Optional[str] = None,
    loss_weighting: Optional[str] = None,
    min_snr_gamma: Optional[float] = None,
    cfg_dropout: Optional[float] = None,
    cfg_scale: Optional[float] = None,
    use_cfg_inference: Optional[bool] = None,
    use_window_normalization: Optional[bool] = None,
    zero_guidance_forecast: Optional[bool] = None,
    image_height: Optional[int] = None,
    coarse_image_height: Optional[int] = None,
    fine_image_height: Optional[int] = None,
    finer_image_height: Optional[int] = None,
    guidance_model=None,
) -> DiffusionTSF:
    """Create DiffusionTSF model with iTransformer guidance channel enabled."""
    if lookback is None:
        lookback = LOOKBACK_LENGTH
    if horizon is None:
        horizon = FORECAST_LENGTH
    if n_variates is None:
        n_variates = N_VARIATES
    if guidance_penalty_weight is None:
        guidance_penalty_weight = GUIDANCE_PENALTY_WEIGHT
    if diffusion_type is None:
        diffusion_type = DIFFUSION_TYPE
    if model_type is None:
        model_type = MODEL_TYPE
    if use_deterministic_anchor_loss is None:
        use_deterministic_anchor_loss = DETERMINISTIC_ANCHOR_LOSS
    if deterministic_anchor_lambda is None:
        deterministic_anchor_lambda = DETERMINISTIC_ANCHOR_LAMBDA
    if deterministic_anchor_alpha is None:
        deterministic_anchor_alpha = DETERMINISTIC_ANCHOR_ALPHA
    if cross_variate_context_bias is None:
        cross_variate_context_bias = CROSS_VARIATE_CONTEXT_BIAS
    if diffusion_stage is None:
        diffusion_stage = DIFFUSION_STAGE
    if use_guidance_channel is None:
        use_guidance_channel = USE_GUIDANCE_CHANNEL
    if max_scale is None:
        max_scale = MAX_SCALE
    if window_norm_std_floor is None:
        window_norm_std_floor = WINDOW_NORM_STD_FLOOR
    if binary_noise_schedule is None:
        binary_noise_schedule = "sqrt_linear"
    if prediction_target is None:
        prediction_target = "x0"
    if loss_weighting is None:
        loss_weighting = "none"
    if min_snr_gamma is None:
        min_snr_gamma = 5.0
    if cfg_dropout is None:
        cfg_dropout = CFG_DROPOUT
    if cfg_scale is None:
        cfg_scale = CFG_SCALE
    if use_cfg_inference is None:
        use_cfg_inference = USE_CFG_INFERENCE
    if use_window_normalization is None:
        use_window_normalization = USE_WINDOW_NORMALIZATION
    if zero_guidance_forecast is None:
        zero_guidance_forecast = ZERO_GUIDANCE_FORECAST
    if image_height is None:
        image_height = IMAGE_HEIGHT
    if coarse_image_height is None:
        coarse_image_height = COARSE_IMAGE_HEIGHT
    if fine_image_height is None:
        fine_image_height = FINE_IMAGE_HEIGHT
    if finer_image_height is None:
        finer_image_height = FINER_IMAGE_HEIGHT
    logger.info(
        f"Creating diffusion model: guidance_penalty_weight={guidance_penalty_weight}, "
        f"diffusion_type={diffusion_type}, "
        f"deterministic_anchor_loss={use_deterministic_anchor_loss}, "
        f"anchor_lambda={deterministic_anchor_lambda}, anchor_alpha={deterministic_anchor_alpha}"
    )

    config = DiffusionTSFConfig(
        num_variables=n_variates,
        lookback_length=lookback,
        forecast_length=horizon + lookback_overlap,
        lookback_overlap=lookback_overlap,
        past_loss_weight=past_loss_weight,
        image_height=image_height,
        coarse_image_height=coarse_image_height,
        fine_image_height=fine_image_height,
        finer_image_height=finer_image_height,
        max_scale=max_scale,
        binary_noise_schedule=binary_noise_schedule,
        prediction_target=prediction_target,
        loss_weighting=loss_weighting,
        min_snr_gamma=min_snr_gamma,
        use_coordinate_channel=True,
        use_guidance_channel=use_guidance_channel,
        guidance_penalty_weight=guidance_penalty_weight,
        model_type=model_type,
        disable_cross_attention=DISABLE_CROSS_ATTENTION,
        diffusion_stage=diffusion_stage,
        use_dual_scale=USE_DUAL_SCALE,
        use_triple_scale=USE_TRIPLE_SCALE,
        dual_scale_fine_weight=DUAL_SCALE_FINE_WEIGHT,
        dual_scale_independent_timesteps=DUAL_SCALE_INDEPENDENT_TIMESTEPS,
        dit_patch_size=DIT_PATCH_SIZE,
        dit_embed_dim=DIT_EMBED_DIM,
        dit_depth=DIT_DEPTH,
        dit_num_heads=DIT_NUM_HEADS,
        dit_mlp_ratio=DIT_MLP_RATIO,
        dit_dropout=DIT_DROPOUT,
        use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
        unet_max_chunk_size=UNET_MAX_CHUNK_SIZE,
        use_amp=USE_AMP,
        diffusion_type=diffusion_type,
        use_deterministic_anchor_loss=use_deterministic_anchor_loss,
        deterministic_anchor_lambda=deterministic_anchor_lambda,
        deterministic_anchor_alpha=deterministic_anchor_alpha,
        cross_variate_context_bias=cross_variate_context_bias,
        cfg_dropout=cfg_dropout,
        cfg_scale=cfg_scale,
        use_cfg_inference=use_cfg_inference,
        use_window_normalization=use_window_normalization,
        window_norm_std_floor=window_norm_std_floor,
        zero_guidance_forecast=zero_guidance_forecast,
    )
    return DiffusionTSF(config, guidance_model=guidance_model)


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


def _paper_split_borders(dataset_name: str, n: int, seq_len: int) -> Tuple[List[int], List[int]]:
    """Return (border1s, border2s) following the iTransformer / TimesNet protocol.

    Each split's window-construction array runs from ``border1s[i]`` to
    ``border2s[i]``. ``border1s[i] = boundary - seq_len`` for val/test so the
    first val/test lookback reaches back into the previous split (no "dead
    zone" of unevaluated steps right after the train boundary).

    ETTh{1,2} and ETTm{1,2} use fixed month-based boundaries. Every other
    dataset uses the 70/10/20 length-based convention with the same overlap
    trick — this mirrors ``Dataset_Custom`` in the upstream iTransformer repo.
    """
    if dataset_name in ('ETTh1', 'ETTh2'):
        b2 = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
    elif dataset_name in ('ETTm1', 'ETTm2'):
        b2 = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
    elif dataset_name == 'PeMS':
        n_train = int(n * 0.6)
        n_val = int(n * 0.2)
        n_test = n - n_train - n_val
        b2 = [n_train, n_train + n_val, n_train + n_val + n_test]
    else:
        n_train = int(n * 0.7)
        n_test = int(n * 0.2)
        n_val = n - n_train - n_test
        b2 = [n_train, n_train + n_val, n_train + n_val + n_test]
    b1 = [0, b2[0] - seq_len, b2[1] - seq_len]
    return b1, b2


def load_dataset(
    dataset_name: str,
    variate_indices: List[int] = None,
    lookback: Optional[int] = None,
    horizon: Optional[int] = None,
    stride: int = 1,
    test_stride: Optional[int] = None,
    lookback_overlap: int = LOOKBACK_OVERLAP,
) -> Tuple[Dataset, Dataset, Dataset, Dict]:
    """Load dataset and return train/val/test splits matching iTransformer paper.

    Splits follow the upstream iTransformer / TimesNet data loaders: fixed
    month-based boundaries for ETT* and 70/10/20 length-based otherwise, with
    the standard overlap trick (val/test windows can reach back into the
    previous split by ``lookback`` steps). Train/val use ``stride``; test uses
    ``test_stride`` (default: same as ``stride``). Finetune jobs pass
    ``test_stride=1`` so eval keeps the dense paper protocol while train/val
  can use a larger stride to cut redundant overlap.
    """
    if test_stride is None:
        test_stride = stride
    if lookback is None:
        lookback = DALIA_DEFAULT_LOOKBACK if dataset_name == 'dalia' else LOOKBACK_LENGTH
    if horizon is None:
        horizon = DALIA_DEFAULT_FORECAST if dataset_name == 'dalia' else FORECAST_LENGTH
    if dataset_name == 'dalia':
        return load_dalia_dataset(
            variate_indices=variate_indices,
            lookback=lookback,
            horizon=horizon,
            stride=stride,
            test_stride=test_stride,
            lookback_overlap=lookback_overlap,
            datasets_dir=DATASETS_DIR,
        )
    path, date_col = _resolve_registry_path(dataset_name)
    data = _load_dataset_array(path, date_col)

    if variate_indices is not None:
        data = data[:, variate_indices]

    n = len(data)
    total_window = lookback + horizon
    if n < total_window:
        raise ValueError(
            f"Dataset '{dataset_name}' has {n} rows but needs at least "
            f"{total_window} (lookback={lookback} + horizon={horizon}). "
            f"Skipping this dataset."
        )

    border1s, border2s = _paper_split_borders(dataset_name, n, lookback)
    train_end = border2s[0]

    train_slice = data[:train_end]
    mean = train_slice.mean(axis=0, keepdims=True)
    std = train_slice.std(axis=0, keepdims=True) + 1e-8
    data = (data - mean) / std

    train_ds = TimeSeriesDataset(
        data[border1s[0]:border2s[0]], lookback, horizon, stride,
        lookback_overlap=lookback_overlap,
    )
    val_ds = TimeSeriesDataset(
        data[border1s[1]:border2s[1]], lookback, horizon, stride,
        lookback_overlap=lookback_overlap,
    )
    test_ds = TimeSeriesDataset(
        data[border1s[2]:border2s[2]], lookback, horizon, test_stride,
        lookback_overlap=lookback_overlap,
    )

    return train_ds, val_ds, test_ds, {'mean': mean, 'std': std}


# ============================================================================
# Variate Subset Management
# ============================================================================

def generate_dataset_job(dataset_name: str, n_variates: int = None, seed: int = 42) -> Dict:
    """Return one full-dataset training job (no variate partitioning)."""
    if dataset_name == 'dalia':
        indices = list(range(DALIA_N_VARS))
        return {
            'dataset_id': dataset_name,
            'variate_indices': indices,
            'variate_names': DALIA_CHANNEL_NAMES,
        }
    path, date_col = _resolve_registry_path(dataset_name)
    n_cols = get_dataset_n_cols(dataset_name)
    all_cols = _dataset_variate_names(path, date_col, n_cols)
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
    seq_len: Optional[int] = None,
    pred_len: Optional[int] = None,
) -> int:
    """Probe iTransformer memory with one train step and pick largest safe even batch."""
    sample_past, sample_future = dataset[0]
    if seq_len is None:
        seq_len = int(sample_past.shape[-1])
    if pred_len is None:
        pred_len = int(sample_future.shape[-1]) - LOOKBACK_OVERLAP

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
            model = create_itransformer(
                seq_len=seq_len, pred_len=pred_len, dropout=dropout,
            ).to(device)
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
    smoke_test: bool = False,
) -> int:
    """Probe diffusion memory with one train step and pick largest safe even batch."""
    if smoke_test:
        hi = max(DIFFUSION_PROBE_MIN_BATCH, int(max_candidate))
        if hi % 2 != 0:
            hi -= 1
        return min(4, hi)

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
            model = create_diffusion_model(
                guidance_model=itrans_guidance,
                **anchor_kwargs_from_params(),
            ).to(device)
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
    seq_len: Optional[int] = None,
    pred_len: Optional[int] = None,
):
    """Optuna objective for iTransformer HP search.

    Paper-faithful setup: only learning rate is searched, over the categorical
    grid {1e-3, 5e-4, 1e-4}. Batch size is fixed at 32, dropout at 0.1, optimizer
    is plain Adam, no gradient clipping, no early stopping, exactly 10 epochs.

    pretrained_ckpt: if provided, warm-starts each trial from those weights
        (used for finetune HP search on real data).
    best_state: shared mutable dict; updated with best cross-trial model state
        whenever a new minimum val loss is achieved.
    """
    lr = trial.suggest_categorical('learning_rate', ITRANS_PAPER_LR_GRID)
    batch_size = fixed_batch_size if fixed_batch_size is not None else ITRANS_PAPER_BATCH_SIZE
    dropout = ITRANS_PAPER_DROPOUT

    if seq_len is None:
        seq_len = ITRANSFORMER_SEQ_LEN
    if pred_len is None:
        pred_len = FORECAST_LENGTH

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = create_itransformer(seq_len=seq_len, pred_len=pred_len, dropout=dropout).to(device)
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

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    epochs = max_epochs if not smoke_test else 1
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
    synth_cache = get_synth_cache_dir(smoke_test=smoke_test)
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
    if smoke_test:
        n_val = max(1, min(len(dataset) // 4, len(dataset) - 1))
    else:
        n_val = min(len(dataset) // 10, 1000)
    train_subset = Subset(dataset, list(range(len(dataset) - n_val)))
    val_subset   = Subset(dataset, list(range(len(dataset) - n_val, len(dataset))))

    train_bs = ITRANS_PAPER_BATCH_SIZE
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
                   f"bs={train_bs}, dropout={ITRANS_PAPER_DROPOUT}")

    study.optimize(
        lambda trial: itrans_hp_objective(
            trial, train_loader, val_loader, device, smoke_test,
            fixed_batch_size=train_bs, best_state=_best_state,
            max_epochs=ITRANS_HP_PRETRAIN_MAX_EPOCHS,
        ),
        n_trials=n_trials,
        show_progress_bar=not smoke_test,
        callbacks=[log_trial],
    )

    best_params = study.best_params
    best_params['batch_size'] = train_bs
    best_params['dropout'] = ITRANS_PAPER_DROPOUT
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
    disable_anchor_loss: bool = False,
):
    """Optuna objective for Diffusion HP search.

    best_state is a shared mutable dict; when provided, updates
    best_state['model_state'] and best_state['val_loss'] whenever this
    trial achieves a new cross-trial best (used to skip a separate pretrain).

    disable_anchor_loss: skip the anchor forward pass during HP search to
        halve per-step cost. The anchor regularizer doesn't help rank LR
        candidates on synthetic data.
    """
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    if fixed_batch_size is None:
        batch_size = trial.suggest_categorical('batch_size', [2, 4] if smoke_test else DIFFUSION_BATCH_SIZES)
    else:
        batch_size = fixed_batch_size

    anchor_kw = anchor_kwargs_from_params()
    if disable_anchor_loss:
        anchor_kw = {'use_deterministic_anchor_loss': False}
    model = create_diffusion_model(
        guidance_model=itrans_guidance,
        **anchor_kw,
    ).to(device)

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
    synth_cache = get_synth_cache_dir(smoke_test=smoke_test)
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
    if smoke_test:
        n_val = max(1, min(len(dataset) // 4, len(dataset) - 1))
    else:
        n_val = min(len(dataset) // 10, 500)
    train_subset = Subset(dataset, list(range(len(dataset) - n_val)))
    val_subset = Subset(dataset, list(range(len(dataset) - n_val, len(dataset))))
    
    train_bs = select_diffusion_batch_size(
        phase_name='Diffusion HP tune',
        dataset=train_subset,
        device=device,
        itrans_guidance=itrans_guidance,
        max_candidate=diffusion_probe_max_candidate(N_VARIATES, smoke_test),
        smoke_test=smoke_test,
    )
    train_loader = DataLoader(train_subset, batch_size=train_bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=min(train_bs, 16), shuffle=False, num_workers=0)
    
    # Run Optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
    
    # Anchor loss doubles per-step compute (extra DiT fwd+bwd). Disable it
    # during Phase 1B since the regularizer doesn't help rank LR candidates.
    skip_anchor = DETERMINISTIC_ANCHOR_LOSS
    if skip_anchor:
        logger.info("Phase 1B: anchor loss disabled for HP search (2× speedup)")

    logger.info(f"Starting Diffusion HP search: {n_trials} trials")
    
    def log_trial(study, trial):
        bs = trial.params.get('batch_size', train_bs)
        msg = (
            f"[Diffusion HP] Trial {trial.number}/{n_trials}: "
            f"loss={trial.value:.4f}, lr={trial.params['learning_rate']:.2e}, bs={bs}"
        )
        if DETERMINISTIC_ANCHOR_LOSS and not skip_anchor:
            lam, alpha = fixed_deterministic_anchor_hp()
            msg += f", anchor_lambda={lam:.4f}, anchor_alpha={alpha:.4f} (fixed)"
        logger.info(msg)
    
    _best_state: dict = {'model_state': None, 'val_loss': float('inf')}

    study.optimize(
        lambda trial: diffusion_hp_objective(
            trial, train_loader, val_loader, itrans_guidance, device, smoke_test,
            fixed_batch_size=train_bs, best_state=_best_state,
            disable_anchor_loss=skip_anchor,
        ),
        n_trials=n_trials,
        show_progress_bar=not smoke_test,
        callbacks=[log_trial],
    )

    best_params = study.best_params
    best_params['batch_size'] = train_bs
    msg = (
        f"Best Diffusion params: lr={best_params['learning_rate']:.2e}, "
        f"bs={best_params['batch_size']}"
    )
    if DETERMINISTIC_ANCHOR_LOSS:
        lam, alpha = fixed_deterministic_anchor_hp()
        msg += f", anchor_lambda={lam:.4f}, anchor_alpha={alpha:.4f} (fixed)"
    logger.info(msg)
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
    checkpoint_dir: str,
    smoke_test: bool = False,
) -> str:
    """Train iTransformer on synthetic data with tuned params (DDP-aware).

    Paper-faithful: Adam (no AdamW), no LR scheduler, no early stopping,
    no gradient clipping. Fixed epoch count.
    """
    logger.info("=" * 60)
    logger.info("PHASE 1C-1: Full iTransformer Pretraining")
    logger.info(f"Samples: {n_samples}, Epochs: {epochs}")
    logger.info(f"Params: {best_params}")
    if _ddp_enabled:
        logger.info(f"DDP: {get_world_size()} GPUs")
    logger.info("=" * 60)
    
    device = get_device()
    
    lr = require_tuned_param(best_params, 'learning_rate', 'iTransformer pretraining')
    batch_size = ITRANS_PAPER_BATCH_SIZE
    dropout = ITRANS_PAPER_DROPOUT
    
    # Create data
    synth_cache = get_synth_cache_dir(checkpoint_dir=checkpoint_dir, smoke_test=smoke_test)
    n_val = 0 if smoke_test else min(n_samples // 10, 5000)
    epoch_cap = 1 if smoke_test else synthetic_epoch_capacity_pretrain_itrans()
    synthetic_loader = get_synthetic_dataloader(
        batch_size=min(32, batch_size),
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
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    ckpt_path = os.path.join(checkpoint_dir, 'pretrained_itransformer.pt')

    for epoch in range(epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)  # Crucial for DDP shuffling

        set_realts_training_epoch(train_loader, epoch)

        t0 = time.time()
        train_loss = train_itransformer_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate_itransformer(model, val_loader, criterion, device)

        if _ddp_enabled:
            train_loss_t = torch.tensor([train_loss], device=device)
            val_loss_t = torch.tensor([val_loss], device=device)
            train_loss = sync_across_processes(train_loss_t).item()
            val_loss = sync_across_processes(val_loss_t).item()

        logger.info(f"[iTransformer] Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | "
                   f"Val: {val_loss:.4f} | LR: {lr:.2e} | Time: {time.time()-t0:.1f}s")

        log_wandb({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': lr,
            'epoch': epoch + 1,
            'epoch_time_s': time.time() - t0,
        }, prefix='itrans_pretrain')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if is_main_process():
                save_checkpoint(unwrap_model(model), optimizer, epoch, train_loss, val_loss, best_params, ckpt_path)
                logger.info(f"  -> New best! Saved to {ckpt_path}")
            barrier()

    barrier()
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
            smoke_test=smoke_test,
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
    model = create_diffusion_model(
        guidance_model=itrans_guidance,
        **anchor_kwargs_from_params(best_params),
    )
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
    train_stride: Optional[int] = None,
    test_stride: Optional[int] = None,
    train_ds: Any = None,
    val_ds: Any = None,
) -> float:
    """Optuna objective for fine-tuning HP search (lr only; batch_size auto-probed or fixed).

    If ``trial_ckpt_dir`` is provided, this trial's best-epoch model state is saved
    to ``{trial_ckpt_dir}/_diff_ft_trial_{trial.number}_best.pt``. The caller picks
    the best study trial and promotes its file to the final ``best.pt`` — no
    separate "Phase 2C" retrain is performed.

    When ``train_ds`` / ``val_ds`` are passed (recommended), the caller loads real data
    once before Optuna — avoids re-reading CSVs every trial and NFS stale-handle flakes.
    """
    lr = trial.suggest_float(
        'learning_rate', FINETUNE_HP_LR_MIN, FINETUNE_HP_LR_MAX, log=True,
    )
    if fixed_batch_size is not None:
        batch_size = fixed_batch_size
    else:
        batch_size = trial.suggest_categorical('batch_size', [2, 4] if smoke_test else FINETUNE_BATCH_SIZES)

    anchor_lambda, anchor_alpha = fixed_deterministic_anchor_hp()
    
    if train_ds is None or val_ds is None:
        train_ds, val_ds, _, _ = load_dataset(
            dataset_name, variate_indices,
            stride=train_stride or WINDOW_STRIDE,
            test_stride=1 if test_stride is None else test_stride,
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
    ds_lb, ds_hz = dataset_window_lengths(dataset_name)
    model = create_diffusion_model(
        n_variates=n_iv,
        lookback=ds_lb,
        horizon=ds_hz,
        guidance_model=itrans_guidance,
        **anchor_kwargs_from_params(),
    ).to(device)
    ckpt = torch.load(pretrained_path, map_location=device, weights_only=False)
    load_diffusion_state_keep_attached_guidance(model, ckpt['model_state_dict'])

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    epochs = HP_TUNE_EPOCHS if not smoke_test else 1
    patience = HP_TUNE_PATIENCE if not smoke_test else 1
    early_stop = EarlyStopping(patience=patience)
    best_val_loss = float('inf')
    trial_tuned_params = {'learning_rate': lr, 'batch_size': batch_size}
    if DETERMINISTIC_ANCHOR_LOSS:
        trial_tuned_params['deterministic_anchor_lambda'] = anchor_lambda
        trial_tuned_params['deterministic_anchor_alpha'] = anchor_alpha

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
                ckpt_config = diffusion_arch_config_dict()
                ckpt_config.update({
                    'tuned_params': trial_tuned_params,
                    'trial_number': trial.number,
                })
                save_checkpoint(
                    unwrap_model(model), optimizer, epoch, float('nan'), val_loss,
                    ckpt_config,
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
    pretrained_path: str,
    itrans_checkpoint: str,
    device: torch.device,
    smoke_test: bool = False,
) -> Tuple[str, Dict]:
    """Copy the best Phase 2B trial checkpoint to best.pt (no extra retrain)."""
    del pretrained_path, itrans_checkpoint, device, smoke_test  # kept for call-site compatibility
    if study.best_trial is None:
        raise RuntimeError("Optuna study has no successful trials")

    subset_id = subset_info['subset_id']
    variate_indices = subset_info['variate_indices']
    best_num = study.best_trial.number
    tuned_params = dict(study.best_params)
    tuned_params['batch_size'] = fixed_batch_size

    src = os.path.join(subset_dir, f'_diff_ft_trial_{best_num}_best.pt')
    if not os.path.exists(src):
        raise RuntimeError(f"Best trial checkpoint missing: {src}")

    dst = os.path.join(subset_dir, 'best.pt')
    logger.info(
        f"Promoting trial {best_num} to {dst} for {subset_id} "
        f"(lr={float(tuned_params['learning_rate']):.2e}, batch_size={int(tuned_params['batch_size'])})"
    )

    import shutil
    if is_main_process():
        shutil.copy2(src, dst)
        trial_ckpt = torch.load(src, map_location='cpu', weights_only=False)
        ckpt_config = trial_ckpt.get('config', {})
        if isinstance(ckpt_config, dict) and 'tuned_params' in ckpt_config:
            tuned_params = dict(ckpt_config['tuned_params'])
            tuned_params['batch_size'] = fixed_batch_size
        best_val_loss = float(trial_ckpt.get('val_loss', study.best_value))
        best_epoch = int(trial_ckpt.get('epoch', 0)) + 1
        if DETERMINISTIC_ANCHOR_LOSS:
            lam, alpha = fixed_deterministic_anchor_hp()
            tuned_params['deterministic_anchor_lambda'] = lam
            tuned_params['deterministic_anchor_alpha'] = alpha
        os.makedirs(subset_dir, exist_ok=True)
        with open(os.path.join(subset_dir, 'metadata.json'), 'w') as f:
            json.dump({
                'subset_id': subset_id,
                'dataset_name': dataset_name,
                'variate_indices': variate_indices,
                'data_subset': subset_info.get('data_subset', {}),
                'variate_names': subset_info.get('variate_names', []),
                'norm_mean': norm_stats['mean'].tolist(),
                'norm_std': norm_stats['std'].tolist(),
                'tuned_params': tuned_params,
                'best_trial': best_num,
                'hp_best_val_loss': float(study.best_value),
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch,
                'promoted_from_trial_ckpt': True,
                'diffusion_type': DIFFUSION_TYPE,
                'image_height': IMAGE_HEIGHT,
                'max_scale': MAX_SCALE,
                'window_norm_std_floor': WINDOW_NORM_STD_FLOOR,
                'use_dual_scale': USE_DUAL_SCALE,
                'diffusion_stage': DIFFUSION_STAGE,
                'dual_scale_fine_weight': DUAL_SCALE_FINE_WEIGHT,
                'dual_scale_independent_timesteps': DUAL_SCALE_INDEPENDENT_TIMESTEPS,
                'use_guidance_channel': USE_GUIDANCE_CHANNEL,
                'cfg_dropout': CFG_DROPOUT,
                'cfg_scale': CFG_SCALE,
                'use_cfg_inference': USE_CFG_INFERENCE,
                'disable_cross_attention': DISABLE_CROSS_ATTENTION,
                'cross_variate_context_bias': CROSS_VARIATE_CONTEXT_BIAS,
                'use_window_normalization': USE_WINDOW_NORMALIZATION,
                'zero_guidance_forecast': ZERO_GUIDANCE_FORECAST,
                'window_stride': WINDOW_STRIDE,
                'lookback_length': LOOKBACK_LENGTH,
                'forecast_length': FORECAST_LENGTH,
                'dit_patch_size': list(DIT_PATCH_SIZE),
            }, f, indent=2)
        for fn in os.listdir(subset_dir):
            if fn.startswith('_diff_ft_trial_') and fn.endswith('_best.pt'):
                try:
                    os.remove(os.path.join(subset_dir, fn))
                except OSError:
                    pass
    else:
        best_val_loss = float(study.best_value)
        best_epoch = 0
    barrier()

    return dst, {
        'best_val_loss': best_val_loss,
        'best_trial': best_num,
        'hp_best_val_loss': float(study.best_value),
        'best_epoch': best_epoch,
    }


def finetune_on_dataset(*args, **kwargs):
    """Removed. Phase 2B HP search plus ``_promote_best_trial_to_final`` (copy best trial ckpt)."""
    raise RuntimeError(
        "finetune_on_dataset() was removed — promote the best Phase 2B trial via "
        "_promote_best_trial_to_final() after study.optimize()."
    )


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_model(
    model: DiffusionTSF,
    test_loader: DataLoader,
    device: torch.device,
    n_samples: int = 3,
    probabilistic_n_samples: Optional[int] = None,
    probabilistic_sampler: Optional[str] = None,
    probabilistic_num_inference_steps: Optional[int] = None,
    smoke_test: bool = False,
) -> Dict:
    """Evaluate model on test set.

    Uses a single deterministic anchor decode for deterministic metrics when
    EVAL_SAMPLER is anchor / deterministic_anchor. Probabilistic CRPS/top-k and
    sample texture always come from a non-anchor stochastic sampler.
    Logs periodic progress (batch index, throughput, ETA) for Slurm logs.
    Note: ``test_loader`` should already be subsetted by the caller if a
    half-test sweep is desired (see eval call site).

    Returns:
        ``single``: MSE/MAE/trend + texture on the first draw (anchor pred when
            ``eval_sampler`` is anchor).
        ``averaged``: texture_* keys are the mean of per-draw texture metrics
            (not texture of the mean). Mean-forecast MSE/MAE is intentionally
            omitted for now because the current MMPD matrix reports
            deterministic-output MSE/MAE in its full profile.
    """
    from tqdm import tqdm
    model.eval()

    all_preds_single = []
    all_preds_avg = []
    all_samples = []
    all_prob_samples = []
    all_targets = []

    n_batches = min(1, len(test_loader)) if smoke_test else len(test_loader)
    batch_size = getattr(test_loader, 'batch_size', None) or 1
    ds = getattr(test_loader, 'dataset', None)
    n_windows = len(ds) if ds is not None else None

    def _gen_kwargs_for_sampler(sampler_name: str, *, default_steps: int) -> Dict[str, Any]:
        if sampler_name in ("anchor", "deterministic_anchor"):
            return {'sampler': 'anchor'}
        if sampler_name == "ddpm":
            return {'sampler': 'ddpm', 'use_ddim': False}
        return {'sampler': sampler_name, 'num_inference_steps': default_steps}

    eval_sampler = EVAL_SAMPLER
    det_steps = 1 if eval_sampler in ("anchor", "deterministic_anchor") else (5 if smoke_test else 20)
    det_gen_kwargs = _gen_kwargs_for_sampler(eval_sampler, default_steps=det_steps)
    anchor_sampler = det_gen_kwargs.get('sampler') == 'anchor'
    prob_sampler = probabilistic_sampler or ("dpmpp" if anchor_sampler else eval_sampler)
    prob_steps = probabilistic_num_inference_steps or (5 if smoke_test else 20)
    prob_gen_kwargs = _gen_kwargs_for_sampler(prob_sampler, default_steps=prob_steps)
    if prob_gen_kwargs.get('sampler') == 'anchor':
        raise ValueError("probabilistic_sampler must not be anchor/deterministic_anchor")

    K = getattr(model.config, 'lookback_overlap', 0)
    if probabilistic_n_samples is None:
        probabilistic_n_samples = n_samples
    effective_avg_samples = 1 if (smoke_test or anchor_sampler) else n_samples
    effective_prob_samples = 1 if (smoke_test or anchor_sampler) else probabilistic_n_samples
    if anchor_sampler and not smoke_test:
        effective_prob_samples = probabilistic_n_samples
    effective_n_samples = max(effective_avg_samples, effective_prob_samples)
    det_steps_for_log = det_gen_kwargs.get('num_inference_steps', 1 if anchor_sampler else 20)
    prob_steps_for_log = prob_gen_kwargs.get(
        'num_inference_steps',
        1 if prob_gen_kwargs.get('sampler') == 'anchor' else 20,
    )
    nfe_total = n_batches * (det_steps_for_log + effective_prob_samples * prob_steps_for_log)

    logger.info(
        "eval: start | windows=%s batches=%d batch_size=%d avg_samples=%d prob_samples=%d "
        "sampler=%s steps=%d lookback_overlap=%d device=%s "
        "(~%d U-Net forward passes across eval)",
        n_windows if n_windows is not None else '?',
        n_batches,
        batch_size,
        effective_avg_samples,
        effective_prob_samples,
        f"{det_gen_kwargs.get('sampler')}+prob:{prob_gen_kwargs.get('sampler')}",
        prob_steps_for_log,
        K,
        device,
        nfe_total,
    )

    use_tqdm = sys.stdout.isatty()
    pbar = tqdm(
        enumerate(test_loader),
        total=n_batches,
        desc='eval',
        mininterval=2.0,
        disable=not use_tqdm,
        file=sys.stdout,
    )
    log_every = max(1, min(50, n_batches // 40 or 1))
    t0 = time.perf_counter()

    with torch.no_grad():
        for batch_idx, (past, future) in pbar:
            if batch_idx >= n_batches:
                break

            past = past.to(device)
            t_batch = time.perf_counter()

            torch.manual_seed(42 + batch_idx)
            result = model.generate(past, **det_gen_kwargs)
            all_preds_single.append(result.get('prediction_global_norm', result['prediction']).cpu())

            if smoke_test:
                pred_cpu = result.get('prediction_global_norm', result['prediction']).cpu()
                all_preds_avg.append(pred_cpu)
                all_samples.append(pred_cpu.unsqueeze(0))
                all_prob_samples.append(pred_cpu.unsqueeze(0))
            else:
                samples = []
                for s_idx in range(effective_n_samples):
                    torch.manual_seed(1000 + s_idx * 17 + batch_idx)
                    result = model.generate(past, **prob_gen_kwargs)
                    samples.append(result.get('prediction_global_norm', result['prediction']).cpu())
                stacked_samples = torch.stack(samples)
                all_preds_avg.append(stacked_samples[:effective_avg_samples].mean(dim=0))
                all_samples.append(stacked_samples[:effective_avg_samples])
                all_prob_samples.append(stacked_samples[:effective_prob_samples])

            if K > 0:
                future = future[..., K:]
            all_targets.append(future)

            done = batch_idx + 1
            elapsed = time.perf_counter() - t0
            batch_wall = time.perf_counter() - t_batch
            rate = done / elapsed if elapsed > 1e-6 else 0.0
            eta = (n_batches - done) / rate if rate > 1e-9 else float('nan')
            mem_mb = ''
            if torch.cuda.is_available() and device.type == 'cuda':
                try:
                    mem_mb = f" cuda_alloc_MiB={torch.cuda.memory_allocated() / (1024 ** 2):.0f}"
                except Exception:
                    mem_mb = ''

            if (
                batch_idx < 3
                or batch_idx % log_every == 0
                or done == n_batches
            ):
                logger.info(
                    "eval: batch %d/%d (%.1f%%) | last_batch_wall=%.2fs | "
                    "avg_rate=%.3f batch/s | elapsed=%.1fs | eta=%.1fs%s",
                    done,
                    n_batches,
                    100.0 * done / n_batches,
                    batch_wall,
                    rate,
                    elapsed,
                    eta,
                    mem_mb,
                )

    logger.info(
        "eval: done | total_wall=%.1fs | batches=%d",
        time.perf_counter() - t0,
        n_batches,
    )

    preds_single = torch.cat(all_preds_single, dim=0)
    preds_avg = torch.cat(all_preds_avg, dim=0)
    samples_tensor = torch.cat(all_samples, dim=1) if len(all_samples) > 0 else preds_avg.unsqueeze(0)
    prob_samples_tensor = (
        torch.cat(all_prob_samples, dim=1)
        if len(all_prob_samples) > 0
        else samples_tensor
    )
    targets = torch.cat(all_targets, dim=0)
    
    # Compute metrics in the dataset's global z-scored space. Diffusion generate()
    # returns prediction_global_norm after undoing only the per-window model norm.
    def compute_metrics(pred, target):
        mse = torch.nn.functional.mse_loss(pred, target).item()
        mae = torch.nn.functional.l1_loss(pred, target).item()
        
        return {'mse': mse, 'mae': mae}
    
    from models.diffusion_tsf.metrics import (
        aggregate_texture_per_sample,
        probabilistic_forecast_metrics,
        texture_metrics,
    )

    n_series = int(targets.shape[0] * targets.shape[1])
    n_draws = int(samples_tensor.shape[0]) if samples_tensor.ndim > 3 else 1
    n_prob_draws = int(prob_samples_tensor.shape[0]) if prob_samples_tensor.ndim > 3 else 1
    logger.info(
        "eval: computing MSE/MAE/trend + texture on %d windows, %d variate-series, "
        "%d avg draw(s), %d paper probabilistic draw(s) "
        "(CPU texture/CRPS; no batch logs — can take a long time on ETTm-scale data)",
        int(targets.shape[0]),
        n_series,
        n_draws,
        n_prob_draws,
    )
    t_metrics = time.perf_counter()

    t_np = targets.numpy()
    single_metrics = compute_metrics(preds_single, targets)
    # First draw (anchor when eval_sampler=anchor, else one stochastic sample).
    single_metrics.update(texture_metrics(t_np, preds_single.numpy()))

    avg_metrics = {
        "n_samples": float(n_draws),
        "point_metrics_disabled": True,
        "point_metrics_note": (
            "Mean-sample MSE/MAE disabled; MMPD full-profile mse/mae are "
            "deterministic-output metrics, not sample-mean metrics."
        ),
    }
    prob_samples_np = prob_samples_tensor.numpy()
    samples_bvsl = np.moveaxis(prob_samples_np, 0, 2)
    prob_metrics = {}
    prob_metrics.update(
        probabilistic_forecast_metrics(
            t_np,
            samples_bvsl,
            gmm_components=10,
            topk_max=3,
            seed=42,
        )
    )
    prob_texture = aggregate_texture_per_sample(t_np, prob_samples_np, max_draws=3)
    for key, val in prob_texture.items():
        prob_metrics[f"prob_{key}"] = val

    logger.info(
        "eval: metrics done | wall=%.1fs (includes texture)",
        time.perf_counter() - t_metrics,
    )

    return {
        'single': single_metrics,
        'averaged': avg_metrics,
        'deterministic_anchor': single_metrics,
        'probabilistic_averaged': avg_metrics,
        'probabilistic_avg30': avg_metrics,
        'probabilistic': prob_metrics,
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


def save_eval_results(
    subset_id,
    dataset_name,
    variate_indices,
    train_metrics,
    eval_results,
    results_dir,
    data_subset: Optional[Dict] = None,
):
    """Save diffusion evaluation results to per-subset subdirectory."""
    data = _load_subset_results(results_dir, subset_id)
    data.update({
        'subset_id': subset_id,
        'dataset': dataset_name,
        'variate_indices': variate_indices,
        'data_subset': data_subset or {},
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
                'avg_mse': m.get('averaged', {}).get('mse'),
                'avg_mae': m.get('averaged', {}).get('mae'),
                'itrans_mse': itrans.get('mse'),
                'itrans_mae': itrans.get('mae'),
            }
            for src, pfx in ((m['single'], 'single'), (m.get('averaged', {}), 'avg')):
                for key, val in src.items():
                    if key.startswith('texture_'):
                        row[f'{pfx}_{key}'] = val
            rows.append(row)
        except Exception:
            continue

    if rows:
        df = pd.DataFrame(rows).sort_values(['dataset', 'subset_id'])
        df.to_csv(os.path.join(results_dir, 'summary.csv'), index=False)


# ============================================================================
# iTransformer Baseline Evaluation
# ============================================================================

def train_subset_itransformer_full_baseline(
    dataset_name: str,
    variate_indices: List[int],
    subset_id: str,
    device: torch.device,
    smoke_test: bool = False,
    epochs: int = None,
    patience: int = None,
    train_stride: Optional[int] = None,
    test_stride: Optional[int] = None,
    data_subset: Optional[Dict] = None,
) -> str:
    """Train iTransformer from scratch on the full train split (no diffusion, no warm-start).

    This is the fair ``iTrans-only'' comparison: same variates and train/val windows as the
    diffusion finetune job, but a separate model not used as guidance.
    """
    ds_lb, ds_hz = dataset_window_lengths(dataset_name)
    ckpt_path = os.path.join(CHECKPOINT_DIR, f'{subset_id}_itrans_full_dataset.pt')
    if os.path.exists(ckpt_path) and not smoke_test:
        logger.info(f"  Using cached full-dataset iTransformer baseline: {ckpt_path}")
        return ckpt_path

    global LOOKBACK_LENGTH, FORECAST_LENGTH, ITRANSFORMER_SEQ_LEN
    saved_lens = (LOOKBACK_LENGTH, FORECAST_LENGTH, ITRANSFORMER_SEQ_LEN)
    LOOKBACK_LENGTH, FORECAST_LENGTH, ITRANSFORMER_SEQ_LEN = ds_lb, ds_hz, ds_lb
    try:
        train_ds, val_ds, _, _ = load_dataset(
            dataset_name, variate_indices,
            stride=train_stride or WINDOW_STRIDE,
            test_stride=1 if test_stride is None else test_stride,
        )
        if smoke_test:
            train_ds = Subset(train_ds, list(range(min(4, len(train_ds)))))
            val_ds = Subset(val_ds, list(range(min(2, len(val_ds)))))

        train_loader = DataLoader(
            train_ds, batch_size=ITRANS_PAPER_BATCH_SIZE, shuffle=True, num_workers=0,
        )
        val_loader = DataLoader(
            val_ds, batch_size=min(ITRANS_PAPER_BATCH_SIZE, 32), shuffle=False, num_workers=0,
        )

        n_iv = len(variate_indices)
        model = create_itransformer(
            seq_len=ds_lb, pred_len=ds_hz, num_vars=n_iv, dropout=ITRANS_PAPER_DROPOUT,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()

        max_epochs = 1 if smoke_test else (epochs if epochs is not None else ITRANS_HP_FINETUNE_MAX_EPOCHS)
        patience_val = 1 if smoke_test else (patience if patience is not None else 5)
        early_stop = EarlyStopping(patience=patience_val)
        best_val = float('inf')

        logger.info(
            f"[{subset_id}] Training full-dataset iTransformer baseline "
            f"({max_epochs} epochs, lookback={ds_lb}, forecast={ds_hz}, n={n_iv})..."
        )
        for epoch in range(max_epochs):
            train_loss = train_itransformer_epoch(model, train_loader, optimizer, criterion, device)
            val_loss = validate_itransformer(model, val_loader, criterion, device)
            logger.info(
                f"[{subset_id}] iTrans full-baseline epoch {epoch + 1}/{max_epochs} "
                f"train={train_loss:.4f} val={val_loss:.4f}"
            )
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    model, optimizer, epoch, train_loss, val_loss,
                    {
                        'subset_id': subset_id,
                        'dataset_name': dataset_name,
                        'variate_indices': variate_indices,
                        'data_subset': data_subset or {},
                        'lookback_length': ds_lb,
                        'forecast_length': ds_hz,
                        'type': 'itrans_full_dataset_baseline',
                    },
                    ckpt_path,
                )
            if early_stop(val_loss):
                break
        logger.info(f"  Full-dataset iTransformer baseline saved → {ckpt_path} (val={best_val:.4f})")
        return ckpt_path
    finally:
        LOOKBACK_LENGTH, FORECAST_LENGTH, ITRANSFORMER_SEQ_LEN = saved_lens


def evaluate_itransformer_baseline(
    subset_id: str,
    dataset_name: str,
    variate_indices: List[int],
    itrans_checkpoint: str,
    results_dir: str,
    device: torch.device,
    smoke_test: bool = False,
    test_indices: Optional[List[int]] = None,
    test_stride: Optional[int] = None,
    data_subset: Optional[Dict] = None,
) -> Dict:
    """Run iTransformer-only forecast on the test split (same windows as diffusion eval).

    Results are merged into the per-subset ``results.json`` for summary tables.
    """
    ds_lb, ds_hz = dataset_window_lengths(dataset_name)
    global LOOKBACK_LENGTH, FORECAST_LENGTH, ITRANSFORMER_SEQ_LEN
    saved_lens = (LOOKBACK_LENGTH, FORECAST_LENGTH, ITRANSFORMER_SEQ_LEN)
    LOOKBACK_LENGTH, FORECAST_LENGTH, ITRANSFORMER_SEQ_LEN = ds_lb, ds_hz, ds_lb
    try:
        _, _, test_ds, _ = load_dataset(
            dataset_name, variate_indices,
            stride=1,
            test_stride=1 if test_stride is None else test_stride,
        )
        if test_indices is not None:
            test_ds = Subset(test_ds, list(test_indices))
        elif smoke_test:
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
                x_dec = torch.zeros(B, ds_hz, C, device=device, dtype=past.dtype)
                output = itrans_model(x_enc, None, x_dec, None)
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
        pred_diff = preds[:, :, 1:] - preds[:, :, :-1]
        tgt_diff = targets[:, :, 1:] - targets[:, :, :-1]
        trend_acc = ((pred_diff > 0) == (tgt_diff > 0)).float().mean().item()

        metrics = {'mse': mse, 'mae': mae, 'trend_accuracy': trend_acc}
        logger.info(
            f"[{subset_id}] iTransformer full-dataset baseline: "
            f"MSE={mse:.4f}, MAE={mae:.4f}, trend={trend_acc:.3f}"
        )

        data = _load_subset_results(results_dir, subset_id)
        data.setdefault('subset_id', subset_id)
        data.setdefault('dataset', dataset_name)
        data.setdefault('variate_indices', variate_indices)
        data.setdefault('data_subset', data_subset or {})
        data['itransformer_metrics'] = metrics
        data['itransformer_baseline_ckpt'] = itrans_checkpoint
        data['itransformer_evaluated_at'] = datetime.now().isoformat()
        _save_subset_results(results_dir, subset_id, data)
        update_summary_csv(results_dir)

        return metrics
    finally:
        LOOKBACK_LENGTH, FORECAST_LENGTH, ITRANSFORMER_SEQ_LEN = saved_lens


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

    ds_lb, ds_hz = dataset_window_lengths(dataset_name)
    train_ds, val_ds, test_ds, norm_stats = load_dataset(
        dataset_name, variate_indices=None,
        stride=1,
        lookback=ds_lb,
        horizon=ds_hz,
    )
    if smoke_test:
        train_ds = Subset(train_ds, list(range(min(4, len(train_ds)))))
        val_ds = Subset(val_ds, list(range(min(2, len(val_ds)))))
        test_ds = Subset(test_ds, list(range(min(2, len(test_ds)))))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = create_itransformer(seq_len=ds_lb, pred_len=ds_hz, num_vars=n_cols).to(device)
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
    model_eval = create_itransformer(seq_len=ds_lb, pred_len=ds_hz, num_vars=n_cols).to(device)
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
    datasets: Optional[List[str]] = None,
    pretrained_diff_ckpt: Optional[str] = None,
    variate_indices: Optional[List[int]] = None,
    subset_id: Optional[str] = None,
):
    """Run the per-dataset fine-tune pipeline (Phase 2A + 2B + eval).

    If ``pretrained_diff_ckpt`` is provided and the file exists, Phase 1 (synthetic
    pretrain) is skipped entirely and that checkpoint is used to warm-start the
    diffusion finetune.  iTransformer Phase 2A always trains cold-start on real data
    (ITRANS_REAL_COLD_START=True), so no synthetic iTrans checkpoint is needed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    recombine_traffic_data()
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = get_device()
    logger.info(f"Using device: {device}")

    if use_wandb:
        tags = ['smoke-test'] if smoke_test else []
        init_wandb(
            project=wandb_project,
            config={'seed': seed, 'smoke_test': smoke_test, 'resume': resume},
            resume=resume,
            tags=tags,
        )

    n_finetune_trials = 1 if smoke_test else N_FINETUNE_HP_TRIALS

    # =========== Diffusion pretrain checkpoint ===========
    # Use the provided 1B checkpoint if it exists, otherwise fall back to running
    # Phase 1 so the script still works on a fresh directory.
    diff_ckpt_candidates = [
        pretrained_diff_ckpt,
        os.path.join(CHECKPOINT_DIR, 'diff_hp_best.pt'),
        os.path.join(CHECKPOINT_DIR, 'pretrained_diffusion.pt'),
    ]
    diff_ckpt = next((p for p in diff_ckpt_candidates if p and os.path.exists(p)), None)

    if diff_ckpt is None:
        logger.info("No pretrained diffusion checkpoint found — running Phase 1 (synthetic pretrain).")
        manifest_path = os.path.join(CHECKPOINT_DIR, 'training_manifest.json')
        if resume and os.path.exists(manifest_path):
            manifest = TrainingManifest.load()
        else:
            manifest = TrainingManifest(seed=seed, created_at=datetime.now().isoformat())

        n_itrans_trials = 1 if smoke_test else N_ITRANS_HP_TRIALS
        n_diff_trials = 1 if smoke_test else N_DIFFUSION_HP_TRIALS
        pretrain_samples = resolve_pretrain_virtual_dataset_size(smoke_test)

        itrans_tune_ckpt = os.path.join(CHECKPOINT_DIR, 'itrans_hp_best.pt')
        if not manifest.itrans_hp_done:
            manifest.itrans_best_params, _ = run_itransformer_hp_tuning(
                n_itrans_trials, smoke_test, checkpoint_dir=CHECKPOINT_DIR,
            )
            manifest.itrans_hp_done = True
            manifest.save()
        else:
            logger.info(f"Cached iTransformer HP: {manifest.itrans_best_params}")

        itrans_ckpt_p1 = os.path.join(CHECKPOINT_DIR, 'pretrained_itransformer.pt')
        if not manifest.itrans_checkpoint or not os.path.exists(itrans_ckpt_p1):
            if os.path.exists(itrans_tune_ckpt):
                import shutil
                shutil.copy2(itrans_tune_ckpt, itrans_ckpt_p1)
            else:
                pretrain_itransformer(
                    manifest.itrans_best_params, n_samples=pretrain_samples,
                    epochs=1 if smoke_test else PRETRAIN_EPOCHS,
                    checkpoint_dir=CHECKPOINT_DIR, smoke_test=smoke_test,
                )
                saved = os.path.join(CHECKPOINT_DIR, 'pretrained_itransformer.pt')
                if saved != itrans_ckpt_p1 and os.path.exists(saved):
                    import shutil; shutil.copy2(saved, itrans_ckpt_p1)
            manifest.itrans_checkpoint = itrans_ckpt_p1
            manifest.save()

        diff_tune_ckpt = os.path.join(CHECKPOINT_DIR, 'diff_hp_best.pt')
        if not manifest.diffusion_hp_done:
            manifest.diffusion_best_params, _ = run_diffusion_hp_tuning(
                itrans_ckpt_p1, n_diff_trials, smoke_test, checkpoint_dir=CHECKPOINT_DIR,
            )
            manifest.diffusion_hp_done = True
            manifest.save()
        else:
            logger.info(f"Cached Diffusion HP: {manifest.diffusion_best_params}")

        diff_ckpt = os.path.join(CHECKPOINT_DIR, 'pretrained_diffusion.pt')
        if not manifest.pretrain_complete or not os.path.exists(diff_ckpt):
            if os.path.exists(diff_tune_ckpt):
                import shutil
                shutil.copy2(diff_tune_ckpt, diff_ckpt)
            else:
                pretrain_diffusion(
                    manifest.diffusion_best_params, itrans_ckpt_p1,
                    n_samples=pretrain_samples,
                    epochs=1 if smoke_test else PRETRAIN_DIFFUSION_MAX_EPOCHS,
                    patience=1 if smoke_test else DIFFUSION_HP_PATIENCE,
                    checkpoint_dir=CHECKPOINT_DIR, smoke_test=smoke_test,
                )
                saved = os.path.join(CHECKPOINT_DIR, 'pretrained_diffusion.pt')
                if saved != diff_ckpt and os.path.exists(saved):
                    import shutil; shutil.copy2(saved, diff_ckpt)
            manifest.pretrain_complete = True
            manifest.save()
        else:
            logger.info(f"Using existing diffusion checkpoint: {diff_ckpt}")
    else:
        logger.info(f"Skipping Phase 1 — using pretrained diffusion checkpoint: {diff_ckpt}")

    # =========== PHASE 2: per-dataset iTrans finetune (2A) + diffusion finetune (2B) + eval ===========
    if datasets and variate_indices is not None:
        if len(datasets) != 1:
            raise ValueError("--variate-indices with --mode full requires exactly one --dataset")
        if len(variate_indices) != N_VARIATES:
            raise ValueError(
                f"--n-variates ({N_VARIATES}) must match --variate-indices length "
                f"({len(variate_indices)}) in --mode full"
            )
        dataset_name = datasets[0]
        all_jobs = {
            dataset_name: {
                'dataset_id': dataset_name,
                'variate_indices': variate_indices,
                'subset_id': subset_id or dataset_name,
            }
        }
    else:
        all_jobs = generate_all_dataset_jobs(seed=seed)
    if datasets and variate_indices is None:
        want = set(datasets)
        all_jobs = {k: v for k, v in all_jobs.items() if k in want}
        missing = want - set(all_jobs)
        if missing:
            logger.warning(
                f"--dataset filter {sorted(missing)} not found in {N_VARIATES}-variate job list "
                f"(available: {sorted(all_jobs)})"
            )
        if not all_jobs:
            raise ValueError(
                f"No finetune jobs for datasets={sorted(want)} with n_variates={N_VARIATES}"
            )
    job_list = list(all_jobs.values())
    if smoke_test:
        job_list = job_list[:1]

    for job in job_list:
        dataset_name = job['dataset_id']
        variate_indices = job['variate_indices']
        job_subset_id = job.get('subset_id') or dataset_name
        subset_info = {'subset_id': job_subset_id, 'variate_indices': variate_indices}

        prior_results = _load_subset_results(RESULTS_DIR, job_subset_id)
        ft_itrans_ckpt = os.path.join(CHECKPOINT_DIR, f'{job_subset_id}_itransformer_finetuned.pt')
        if resume and prior_results.get('eval_metrics') and os.path.exists(ft_itrans_ckpt):
            logger.info(
                f"[Resume] Skipping {job_subset_id}: eval_metrics already present in "
                f"{_subset_results_path(RESULTS_DIR, job_subset_id)} and Phase 2A checkpoint exists"
            )
            continue

        try:
            # _finetune_and_eval_one_subset handles:
            #   Phase 2A — iTrans HP tune + promote best (cold start on real data)
            #   Phase 2B — Diffusion HP tune using finetuned iTrans + promote best
            #   Eval + iTransformer baseline eval + results save
            _finetune_and_eval_one_subset(
                subset_info, dataset_name, diff_ckpt,
                itrans_ckpt="",  # cold start (ITRANS_REAL_COLD_START=True) — path unused
                n_finetune_trials=n_finetune_trials,
                device=device,
                smoke_test=smoke_test,
            )
            if use_wandb:
                _r = _load_subset_results(RESULTS_DIR, dataset_name)
                if _r.get('eval_metrics'):
                    log_wandb_eval_results(dataset_name, _r['eval_metrics'], _r.get('train_metrics', {}))
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
    pretrain_patience = 1 if smoke_test else DIFFUSION_HP_PATIENCE  # diffusion fallback only

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
    train_stride: Optional[int] = None,
    test_stride: Optional[int] = None,
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
        f"{ITRANS_HP_FINETUNE_MAX_EPOCHS} epochs per trial (no early stopping), "
        f"warm_start={'no (cold start)' if warm is None else os.path.basename(warm)}"
    )
    logger.info("=" * 60)

    train_ds, val_ds, _, _ = load_dataset(
        dataset_name, variate_indices,
        stride=train_stride or WINDOW_STRIDE,
        test_stride=1 if test_stride is None else test_stride,
    )
    if smoke_test:
        train_ds = Subset(train_ds, list(range(min(2, len(train_ds)))))
        val_ds = Subset(val_ds, list(range(min(2, len(val_ds)))))

    train_bs = ITRANS_PAPER_BATCH_SIZE
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
                   f"dropout={ITRANS_PAPER_DROPOUT}")

    ds_lb, ds_hz = dataset_window_lengths(dataset_name)
    study.optimize(
        lambda trial: itrans_hp_objective(
            trial, train_loader, val_loader, device, smoke_test,
            fixed_batch_size=train_bs, best_state=_best_state,
            pretrained_ckpt=warm,
            max_epochs=ITRANS_HP_FINETUNE_MAX_EPOCHS,
            seq_len=ds_lb,
            pred_len=ds_hz,
        ),
        n_trials=n_trials,
        show_progress_bar=False,
        callbacks=[log_trial],
    )

    best_params = study.best_params
    best_params['batch_size'] = train_bs
    best_params['dropout'] = ITRANS_PAPER_DROPOUT
    best_params['lookback_length'] = ds_lb
    best_params['forecast_length'] = ds_hz
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

    global LOOKBACK_LENGTH, FORECAST_LENGTH, ITRANSFORMER_SEQ_LEN
    ds_lb, ds_hz = dataset_window_lengths(dataset_name)
    saved_lens = (LOOKBACK_LENGTH, FORECAST_LENGTH, ITRANSFORMER_SEQ_LEN)
    LOOKBACK_LENGTH, FORECAST_LENGTH, ITRANSFORMER_SEQ_LEN = ds_lb, ds_hz, ds_lb
    if dataset_name == 'dalia':
        logger.info(
            f"DALIA finetune windows: lookback={ds_lb}, forecast={ds_hz} "
            f"(train/val stride={WINDOW_STRIDE})"
        )

    # Preflight: check dataset has enough rows before wasting a trial slot
    try:
        load_dataset(
            dataset_name, variate_indices,
            stride=WINDOW_STRIDE, test_stride=1,
        )
    except ValueError as ve:
        logger.warning(f"Skipping {subset_id}: {ve}")
        LOOKBACK_LENGTH, FORECAST_LENGTH, ITRANSFORMER_SEQ_LEN = saved_lens
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

        subset_dir = os.path.join(CHECKPOINT_DIR, subset_id)
        os.makedirs(subset_dir, exist_ok=True)

        # Eval-resume: skip diffusion HP if best.pt + metadata.json exist but
        # results.json has no eval_metrics yet.
        existing_best = os.path.join(subset_dir, 'best.pt')
        existing_meta = os.path.join(subset_dir, 'metadata.json')
        prior_results = _load_subset_results(RESULTS_DIR, subset_id)
        can_resume_eval = (
            os.path.exists(existing_best)
            and os.path.exists(existing_meta)
            and 'eval_metrics' not in prior_results
        )
        if can_resume_eval:
            with open(existing_meta) as f:
                md = json.load(f)
            tuned_params = md.get('tuned_params', {})
            ft_diff_bs = int(tuned_params.get('batch_size', 8))
            ckpt_path = existing_best
            train_metrics = {
                'best_val_loss': md.get('best_val_loss', float('nan')),
                'best_trial': md.get('best_trial', -1),
            }
            logger.info(
                f"[Resume] Found existing fine-tuned checkpoint for {subset_id} "
                f"at {ckpt_path}; skipping diffusion HP and going straight to eval."
            )
        else:
            # Phase C: Diffusion HP search using finetuned iTransformer as guidance
            _ft_itrans_model = load_itransformer_from_checkpoint(ft_itrans_ckpt, len(variate_indices), device)
            _ft_itrans_guidance = iTransformerGuidance(_ft_itrans_model)
            _probe_ds, _, _, _ = load_dataset(
                dataset_name, variate_indices,
                stride=WINDOW_STRIDE, test_stride=1,
            )
            ft_diff_bs = select_diffusion_batch_size(
                phase_name=f'Diff FT HP ({subset_id})',
                dataset=_probe_ds,
                device=device,
                itrans_guidance=_ft_itrans_guidance,
                max_candidate=diffusion_probe_max_candidate(len(variate_indices), smoke_test),
                smoke_test=smoke_test,
            )
            del _ft_itrans_model, _ft_itrans_guidance, _probe_ds
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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

            _, _, _, norm_stats = load_dataset(
                dataset_name, variate_indices,
                stride=WINDOW_STRIDE, test_stride=1,
            )
            ckpt_path, train_metrics = _promote_best_trial_to_final(
                study, subset_dir, subset_info, dataset_name, norm_stats, ft_diff_bs,
                diff_ckpt, ft_itrans_ckpt, device, smoke_test,
            )

        # Evaluate diffusion model
        logger.info(f"Evaluating {subset_id}...")
        itrans_model = load_itransformer_from_checkpoint(ft_itrans_ckpt, len(variate_indices), device)
        itrans_guidance = iTransformerGuidance(itrans_model)

        model = create_diffusion_model(
            guidance_model=itrans_guidance,
            **anchor_kwargs_from_params(tuned_params),
        ).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        load_diffusion_state_keep_attached_guidance(model, ckpt['model_state_dict'])

        _, _, test_ds, _ = load_dataset(
            dataset_name, variate_indices, stride=1, test_stride=1,
        )
        if smoke_test:
            test_ds = Subset(test_ds, list(range(min(2, len(test_ds)))))
        else:
            n_full = len(test_ds)
            n_eval = max(1, n_full // 2)
            rng = np.random.default_rng(42)
            eval_idx = sorted(rng.choice(n_full, size=n_eval, replace=False).tolist())
            test_ds = Subset(test_ds, eval_idx)
            logger.info(f"[{subset_id}] eval subset: {n_eval}/{n_full} windows (seeded random half)")
        test_loader = DataLoader(test_ds, batch_size=8 if not smoke_test else 2, shuffle=False)

        eval_results = evaluate_model(model, test_loader, device, n_samples=3, smoke_test=smoke_test)
        avg_block = eval_results.get('averaged', {})
        if 'mse' in avg_block:
            logger.info(f"[{subset_id}] Avg: MSE={avg_block['mse']:.4f}, "
                        f"MAE={avg_block['mae']:.4f}")
        else:
            logger.info(f"[{subset_id}] Avg point MSE/MAE disabled")

        save_eval_results(
            subset_id, dataset_name, variate_indices,
            {**train_metrics, 'tuned_params': tuned_params}, eval_results, RESULTS_DIR,
        )

        # iTransformer-only baseline (trained on full train split, not the guidance ckpt)
        try:
            full_itrans_ckpt = os.path.join(
                CHECKPOINT_DIR, f'{subset_id}_itrans_full_dataset.pt',
            )
            if not os.path.exists(full_itrans_ckpt):
                full_itrans_ckpt = train_subset_itransformer_full_baseline(
                    dataset_name, variate_indices, subset_id, device, smoke_test=smoke_test,
                )
            eval_test_indices = None
            if not smoke_test and isinstance(test_ds, Subset):
                eval_test_indices = list(test_ds.indices)
            evaluate_itransformer_baseline(
                subset_id, dataset_name, variate_indices,
                full_itrans_ckpt, RESULTS_DIR, device, smoke_test=smoke_test,
                test_indices=eval_test_indices,
            )
        except Exception as be:
            logger.warning(f"iTransformer full-dataset baseline failed for {subset_id}: {be}")

    except KeyboardInterrupt:
        logger.info(f"\nInterrupted during {subset_id}.")
        raise
    except Exception as e:
        logger.error(f"Error with {subset_id}: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        LOOKBACK_LENGTH, FORECAST_LENGTH, ITRANSFORMER_SEQ_LEN = saved_lens


def run_baseline_mode(dataset_name: str, smoke_test: bool = False):
    """Train full-dimensionality iTransformer baseline for a high-variate dataset."""
    recombine_traffic_data()
    train_full_dim_itransformer_baseline(dataset_name, smoke_test=smoke_test)


# ============================================================================
# CLI
# ============================================================================

def main():
    global logger, N_VARIATES, CHECKPOINT_DIR, RESULTS_DIR, MANIFEST_PATH, SYNTH_CACHE_DIR, GUIDANCE_PENALTY_WEIGHT
    global IMAGE_HEIGHT, MAX_SCALE, WINDOW_NORM_STD_FLOOR, DISABLE_CROSS_ATTENTION, CROSS_VARIATE_CONTEXT_BIAS
    global USE_DUAL_SCALE, USE_TRIPLE_SCALE, DIFFUSION_STAGE, DUAL_SCALE_FINE_WEIGHT, DUAL_SCALE_INDEPENDENT_TIMESTEPS
    global USE_GUIDANCE_CHANNEL
    global CFG_DROPOUT, CFG_SCALE, USE_CFG_INFERENCE
    global LOOKBACK_LENGTH, FORECAST_LENGTH, ITRANSFORMER_SEQ_LEN
    global MODEL_TYPE, DIFFUSION_TYPE, DETERMINISTIC_ANCHOR_LOSS, DETERMINISTIC_ANCHOR_LAMBDA
    global DETERMINISTIC_ANCHOR_ALPHA, EVAL_SAMPLER
    global USE_WINDOW_NORMALIZATION, ZERO_GUIDANCE_FORECAST, WINDOW_STRIDE

    parser = argparse.ArgumentParser(description='Diffusion TSF Training Pipeline')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML experiment config (new modular pipeline)')
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
    parser.add_argument('--datasets-dir', type=str, default=None,
                        help='Directory with benchmark CSV/NPZ trees (ETT-small, etc.)')
    parser.add_argument('--synth-cache-dir', type=str, default=None,
                        help='Shared synthetic pool cache directory for reuse across runs')
    parser.add_argument('--pretrained-diff-ckpt', type=str, default=None,
                        help='Path to a pretrained (Phase 1B) diffusion checkpoint. '
                             'When provided, Phase 1 synthetic pretrain is skipped entirely.')
    parser.add_argument('--fresh', action='store_true',
                        help='Wipe manifest and checkpoints, start from scratch')
    parser.add_argument('--guidance-penalty-weight', type=float, default=GUIDANCE_PENALTY_WEIGHT,
                        help='Weight for guidance penalty loss (default from pipeline_config)')
    parser.add_argument('--deterministic-anchor-loss', action='store_true',
                        help='Add deterministic anchor loss at alpha_bar closest to --deterministic-anchor-alpha')
    parser.add_argument('--deterministic-anchor-lambda', type=float, default=DETERMINISTIC_ANCHOR_LAMBDA,
                        help='Weight on standard diffusion MSE when anchor loss is enabled')
    parser.add_argument('--deterministic-anchor-alpha', type=float, default=None,
                        help='Target alpha_bar for Gaussian anchor; binary clean-bit anchor uses alpha=0')
    parser.add_argument('--eval-sampler', type=str, default=EVAL_SAMPLER,
                        choices=['dpmpp', 'ddim', 'ddpm', 'anchor', 'deterministic_anchor'],
                        help='Sampler used by diffusion eval')
    parser.add_argument('--image-height', type=int, default=IMAGE_HEIGHT,
                        help='Override image height')
    parser.add_argument('--max-scale', type=float, default=MAX_SCALE,
                        help='Representation range in per-window normalized units')
    parser.add_argument('--window-norm-std-floor', type=float, default=WINDOW_NORM_STD_FLOOR,
                        help='Minimum per-window std for diffusion normalization')
    parser.add_argument('--dual-scale', action='store_true',
                        help='Use paired 16-bin coarse/residual binary CDF maps')
    parser.add_argument('--triple-scale', action='store_true',
                        help='Use staged coarse/fine/finer residual binary CDF maps')
    parser.add_argument('--diffusion-stage', type=str, default=DIFFUSION_STAGE,
                        choices=['joint', 'coarse', 'fine', 'finer'],
                        help='Joint dual-scale, staged coarse, staged fine, or staged finer diffusion model')
    parser.add_argument('--disable-guidance-channel', action='store_true',
                        help='Disable iTransformer forecast ghost channel while keeping encoder tokens if cross-attention is on')
    parser.add_argument('--dual-scale-fine-weight', type=float, default=DUAL_SCALE_FINE_WEIGHT,
                        help='Weight on fine residual BCE in dual-scale binary diffusion')
    parser.add_argument('--dual-scale-independent-timesteps', action='store_true',
                        default=DUAL_SCALE_INDEPENDENT_TIMESTEPS,
                        help='Draw independent diffusion timesteps for coarse and fine scales')
    parser.add_argument('--cfg-dropout', type=float, default=CFG_DROPOUT,
                        help='Classifier-free conditioning dropout during diffusion training')
    parser.add_argument('--cfg-scale', type=float, default=CFG_SCALE,
                        help='Classifier-free guidance scale used when inference CFG is enabled')
    parser.add_argument('--use-cfg-inference', action='store_true', default=USE_CFG_INFERENCE,
                        help='Blend conditional and null predictions during diffusion inference')
    parser.add_argument('--disable-cross-attention', action='store_true',
                        help='Disable cross-variate attention (fully univariate baseline)')
    parser.add_argument('--cross-variate-context-bias', type=float, default=CROSS_VARIATE_CONTEXT_BIAS,
                        help='Additive bottleneck cross-attention bias for the target variate token')
    parser.add_argument('--disable-window-normalization', action='store_true',
                        help='Use only dataset-level normalization from load_dataset; skip per-window z-score in DiffusionTSF')
    parser.add_argument('--zero-guidance-forecast', action='store_true',
                        help='Zero the iTransformer forecast ghost image while keeping encoder tokens for cross attention')
    parser.add_argument('--lookback-length', type=int, default=LOOKBACK_LENGTH,
                        help='Override lookback length')
    parser.add_argument('--forecast-length', type=int, default=FORECAST_LENGTH,
                        help='Override forecast length')
    parser.add_argument('--window-stride', type=int, default=WINDOW_STRIDE,
                        help='Train/val sliding-window stride (test stays stride=1)')

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
    DETERMINISTIC_ANCHOR_LOSS = args.deterministic_anchor_loss
    DETERMINISTIC_ANCHOR_LAMBDA = args.deterministic_anchor_lambda
    EVAL_SAMPLER = "anchor" if args.eval_sampler == "deterministic_anchor" else args.eval_sampler
    USE_DUAL_SCALE = args.dual_scale
    USE_TRIPLE_SCALE = args.triple_scale
    DIFFUSION_STAGE = args.diffusion_stage
    DUAL_SCALE_FINE_WEIGHT = args.dual_scale_fine_weight
    DUAL_SCALE_INDEPENDENT_TIMESTEPS = args.dual_scale_independent_timesteps
    USE_GUIDANCE_CHANNEL = not args.disable_guidance_channel
    CFG_DROPOUT = args.cfg_dropout
    CFG_SCALE = args.cfg_scale
    USE_CFG_INFERENCE = args.use_cfg_inference
    CROSS_VARIATE_CONTEXT_BIAS = args.cross_variate_context_bias
    staged_model = DIFFUSION_STAGE in {"coarse", "fine", "finer"}
    IMAGE_HEIGHT = 16 if (USE_DUAL_SCALE or staged_model) and args.image_height == parser.get_default('image_height') else args.image_height
    MAX_SCALE = args.max_scale
    WINDOW_NORM_STD_FLOOR = args.window_norm_std_floor
    if args.disable_cross_attention:
        DISABLE_CROSS_ATTENTION = True
    if args.disable_window_normalization:
        USE_WINDOW_NORMALIZATION = False
    if args.zero_guidance_forecast:
        ZERO_GUIDANCE_FORECAST = True
    
    # We enforce Binary DiT only since user asked to remove others
    MODEL_TYPE = "dit"
    DIFFUSION_TYPE = "binary"

    if args.deterministic_anchor_alpha is None:
        if DETERMINISTIC_ANCHOR_LOSS:
            DETERMINISTIC_ANCHOR_ALPHA = 0.0
    else:
        DETERMINISTIC_ANCHOR_ALPHA = args.deterministic_anchor_alpha
        
    if DETERMINISTIC_ANCHOR_LOSS and DETERMINISTIC_ANCHOR_ALPHA != 0.0:
        parser.error(
            "Binary anchor is a max-noise Bernoulli clean-bit anchor; use "
            "--deterministic-anchor-alpha 0.0 or omit the flag."
        )
    _cfg_lookback = LOOKBACK_LENGTH
    LOOKBACK_LENGTH = args.lookback_length
    FORECAST_LENGTH = args.forecast_length
    if args.lookback_length != _cfg_lookback:
        ITRANSFORMER_SEQ_LEN = args.lookback_length
    if args.window_stride < 1:
        parser.error('--window-stride must be >= 1')
    WINDOW_STRIDE = args.window_stride
    
    # DDP setup
    if args.ddp:
        if not setup_ddp():
            print("ERROR: --ddp flag set but DDP init failed.")
            sys.exit(1)
    
    logger = setup_logging()
    
    # ---- New Pipeline Path ----
    if args.config:
        from models.diffusion_tsf.pipeline import load_experiment_config, PipelineState, Pipeline
        from models.diffusion_tsf.pipeline.phases import PHASE_REGISTRY
        
        cli_overrides = {}
        if args.dataset: cli_overrides["dataset"] = args.dataset
        
        nv = args.n_variates
        variate_indices = None
        if args.variate_indices:
            variate_indices = [int(x.strip()) for x in args.variate_indices.split(',') if x.strip()]
            cli_overrides["variate_indices"] = variate_indices
            if not nv: nv = len(variate_indices)
            
        if not nv and args.dataset:
            try:
                nv = get_dim_for_dataset(args.dataset)
            except Exception:
                pass
                
        if nv: cli_overrides["n_variates"] = nv
        
        if args.seed != 42: cli_overrides["seed"] = args.seed
        if args.smoke_test: cli_overrides["smoke_test"] = True
        if args.checkpoint_dir: cli_overrides["checkpoint_dir"] = args.checkpoint_dir
        if args.results_dir: cli_overrides["results_dir"] = args.results_dir
        if args.datasets_dir: cli_overrides["datasets_dir"] = os.path.abspath(args.datasets_dir)
        if args.synth_cache_dir: cli_overrides["synth_cache_dir"] = args.synth_cache_dir
        if args.wandb: cli_overrides["wandb_enabled"] = True
        if args.wandb_project: cli_overrides["wandb_project"] = args.wandb_project
        if args.fresh: cli_overrides["fresh"] = True
        if args.resume: cli_overrides["resume"] = True
        if args.subset_id: cli_overrides["subset_id"] = args.subset_id
        if args.cfg_dropout != parser.get_default('cfg_dropout'):
            cli_overrides["cfg_dropout"] = args.cfg_dropout
        if args.cfg_scale != parser.get_default('cfg_scale'):
            cli_overrides["cfg_scale"] = args.cfg_scale
        if args.use_cfg_inference != parser.get_default('use_cfg_inference'):
            cli_overrides["use_cfg_inference"] = args.use_cfg_inference
        if args.cross_variate_context_bias != parser.get_default('cross_variate_context_bias'):
            cli_overrides["cross_variate_context_bias"] = args.cross_variate_context_bias
        if args.max_scale != parser.get_default('max_scale'):
            cli_overrides["max_scale"] = args.max_scale
        if args.window_norm_std_floor != parser.get_default('window_norm_std_floor'):
            cli_overrides["window_norm_std_floor"] = args.window_norm_std_floor

        cfg = load_experiment_config(args.config, cli_overrides)
        state = PipelineState.from_config(cfg)
        subset_meta = resolve_pipeline_data_subset(state)
        if subset_meta.get("enabled"):
            logger.info(
                "Data subset resolved: %s -> %s vars, train_stride=%s, test_stride=%s, "
                "raw=%.2f MiB, reduced≈%.2f MiB",
                state.subset_id,
                subset_meta.get("n_variates"),
                subset_meta.get("train_stride"),
                subset_meta.get("test_stride"),
                float(subset_meta.get("raw_size_mb") or 0.0),
                float(subset_meta.get("reduced_size_mb") or 0.0),
            )
        
        phases = []
        for p in cfg.get("phases", []):
            p_class = PHASE_REGISTRY.get(p["phase"])
            if not p_class:
                logger.error(f"Unknown phase: {p['phase']}")
                sys.exit(1)
            phases.append(p_class(**p))
            
        try:
            Pipeline(phases, state).run()
        finally:
            if args.wandb:
                from models.diffusion_tsf.pipeline import wandb_utils
                wandb_utils.finish_phase_run()
            cleanup_ddp()
        return

    # ---- Legacy Mode dispatch ----
    
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
    variate_indices = None
    if args.variate_indices:
        variate_indices = [int(x.strip()) for x in args.variate_indices.split(',') if x.strip()]
    if args.fresh:
        if os.path.exists(MANIFEST_PATH):
            os.remove(MANIFEST_PATH)
            logger.info(f"Removed old manifest: {MANIFEST_PATH}")
        for ckpt_file in [
            'itrans_hp_best.pt',
            'diff_hp_best.pt',
            'pretrained_itransformer.pt',
            'pretrained_diffusion.pt',
        ]:
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
            datasets=[args.dataset] if args.dataset else None,
            pretrained_diff_ckpt=args.pretrained_diff_ckpt,
            variate_indices=variate_indices,
            subset_id=args.subset_id,
        )
    finally:
        finish_wandb()
        cleanup_ddp()


if __name__ == '__main__':
    main()
