"""
Dataset helpers.

got some 1D augs and RealTS stuff for pre-training.
"""

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import logging
from typing import Optional
# load RealTS for synthetic data
try:
    from .realts import RealTS
except ImportError:
    from realts import RealTS

logger = logging.getLogger(__name__)


def apply_1d_augmentations(
    seq: torch.Tensor,
    scale_prob: float = 0.5,
    scale_min: float = 0.8,
    scale_max: float = 1.2,
    warp_prob: float = 0.3,
    warp_min: float = 0.9,
    warp_max: float = 1.1,
    stretch_prob: float = 0.3,
    stretch_min: float = 0.7,
    stretch_max: float = 1.3,
) -> torch.Tensor:
    """
    Apply some lightweight 1D augs (scaling, warp, stretch).
    
    stretch > 1.0 repeats stuff, < 1.0 averages.
    always resizes back to original len.
    """
    seq = seq.clone()
    
    # random scaling
    if torch.rand(1).item() < scale_prob:
        scale = torch.empty(1).uniform_(scale_min, scale_max).item()
        seq = seq * scale
    
    # time-warp
    if torch.rand(1).item() < warp_prob:
        factor = torch.empty(1).uniform_(warp_min, warp_max).item()
        orig_len = seq.shape[-1]
        new_len = max(1, int(round(orig_len * factor)))
        
        warped = F.interpolate(
            seq.unsqueeze(0).unsqueeze(0),
            size=new_len,
            mode='linear',
            align_corners=False
        ).squeeze(0).squeeze(0)
        
        if new_len > orig_len:
            # crop it
            start = torch.randint(0, new_len - orig_len + 1, (1,)).item()
            seq = warped[start:start + orig_len]
        elif new_len < orig_len:
            # pad it
            pad_total = orig_len - new_len
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            left_pad = warped[0].repeat(pad_left) if pad_left > 0 else torch.tensor([], device=warped.device)
            right_pad = warped[-1].repeat(pad_right) if pad_right > 0 else torch.tensor([], device=warped.device)
            seq = torch.cat([left_pad, warped, right_pad], dim=0)
        else:
            seq = warped
    
    # time-stretch
    if torch.rand(1).item() < stretch_prob:
        factor = torch.empty(1).uniform_(stretch_min, stretch_max).item()
        orig_len = seq.shape[-1]
        target_len = max(2, int(round(orig_len * factor)))

        # if factor > 1 use nearest, else linear
        mode = 'nearest' if factor >= 1.0 else 'linear'
        stretched = F.interpolate(
            seq.unsqueeze(0).unsqueeze(0),
            size=target_len,
            mode=mode,
            align_corners=False if mode == 'linear' else None
        ).squeeze(0).squeeze(0)

        # Resize back to original length (linear for smoothness)
        seq = F.interpolate(
            stretched.unsqueeze(0).unsqueeze(0),
            size=orig_len,
            mode='linear',
            align_corners=False
        ).squeeze(0).squeeze(0)

    return seq




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



