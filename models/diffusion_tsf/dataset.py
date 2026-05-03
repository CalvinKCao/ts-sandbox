"""
Dataset helpers.

got some 1D augs and RealTS stuff for pre-training.
"""

import torch
from torch.utils.data import DataLoader
import logging
from typing import Optional
# load RealTS for synthetic data
try:
    from .realts import RealTS
except ImportError:
    from realts import RealTS

logger = logging.getLogger(__name__)

# ============================================================================
# RealTS Synthetic Data for Pre-training
# ============================================================================

def get_synthetic_dataloader(
    num_samples: int = 10000,
    lookback_length: int = 512,
    forecast_length: int = 96,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    seed: Optional[int] = None,
    num_variables: int = 1,
    pool_size: Optional[int] = None,
    cache_dir: Optional[str] = None,
    lookback_overlap: int = 0,
    skip_cross_var_aug: bool = False,
) -> DataLoader:
    """Create a DataLoader with ONLY synthetic RealTS data for pre-training.
    
    This is used for the pre-training phase where the model learns general
    time series structure from diverse synthetic patterns before fine-tuning
    on real data.
    
    Args:
        num_samples: Number of synthetic samples to generate per epoch (virtual size)
        lookback_length: Past context window length
        forecast_length: Forecast horizon length
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        seed: Random seed for reproducibility (None for random)
        num_variables: Number of variables (default: 1)
        pool_size: Total number of samples in the cached pool (randomly sampled)
        cache_dir: Directory to cache the pool (enables large disk-based pools)
        lookback_overlap: Number of past steps to include in the target (K)
        skip_cross_var_aug: Skip O(V²) cross-variate augmentation for high-V
        
    Returns:
        DataLoader with synthetic-only data
    """
    synthetic_dataset = RealTS(
        num_samples=num_samples,
        lookback_length=lookback_length,
        forecast_length=forecast_length,
        seed=seed,
        num_variables=num_variables,
        pool_size=pool_size,
        cache_dir=cache_dir,
        lookback_overlap=lookback_overlap,
        skip_cross_var_aug=skip_cross_var_aug,
    )
    
    logger.info(
        f"Created synthetic-only dataloader: {num_samples} samples/epoch "
        f"(Pool: {pool_size or num_samples}), "
        f"lookback={lookback_length}, forecast={forecast_length}, "
        f"variables={num_variables}"
    )
    
    return DataLoader(
        synthetic_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )



