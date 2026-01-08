"""
Dataset utilities for Diffusion TSF.

Provides:
- Synthetic data generation for testing
- Simple loading from common TSF datasets
- RealTS synthetic data mixing for improved generalizability
"""

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn.functional as F
import numpy as np
import logging
from typing import Tuple, Optional, List, Union
from dataclasses import dataclass

# Import RealTS for synthetic data generation (ViTime-inspired)
# Use try/except to handle both relative imports (when used as package)
# and absolute imports (when used as script)
try:
    from .realts import RealTS, create_realts_dataset
except ImportError:
    from realts import RealTS, create_realts_dataset

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Dataset-level configuration toggles."""
    representation_mode: str = "pdf"  # "pdf" (stripe) or "cdf" (occupancy)

    def __post_init__(self):
        if self.representation_mode not in ["pdf", "cdf"]:
            raise ValueError("representation_mode must be 'pdf' or 'cdf'")


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
    """Apply lightweight 1D augmentations (scaling, warp, and stretch) to a sequence.

    stretch > 1.0 repeats/holds values (nearest), stretch < 1.0 averages/merges (linear),
    and the result is always resized back to the original length to preserve shape.
    """
    seq = seq.clone()
    
    # Random scaling
    if torch.rand(1).item() < scale_prob:
        scale = torch.empty(1).uniform_(scale_min, scale_max).item()
        seq = seq * scale
    
    # Time-warp via interpolation
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
            # Random crop back to original length
            start = torch.randint(0, new_len - orig_len + 1, (1,)).item()
            seq = warped[start:start + orig_len]
        elif new_len < orig_len:
            # Pad (replicate edges) to original length; manual to support 1D
            pad_total = orig_len - new_len
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            left_pad = warped[0].repeat(pad_left) if pad_left > 0 else torch.tensor([], device=warped.device)
            right_pad = warped[-1].repeat(pad_right) if pad_right > 0 else torch.tensor([], device=warped.device)
            seq = torch.cat([left_pad, warped, right_pad], dim=0)
        else:
            seq = warped
    
    # Time-stretch (hold or average contiguous values), then resize back
    if torch.rand(1).item() < stretch_prob:
        factor = torch.empty(1).uniform_(stretch_min, stretch_max).item()
        orig_len = seq.shape[-1]
        target_len = max(2, int(round(orig_len * factor)))

        # If factor > 1 -> nearest neighbor repeat; if < 1 -> linear (averaging)
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


class SyntheticTimeSeriesDataset(Dataset):
    """Synthetic time series dataset for testing.
    
    Generates time series with various patterns:
    - Sinusoidal waves with multiple frequencies
    - Trends (linear, exponential)
    - Noise
    - W/V shapes (high-frequency patterns)
    """
    
    def __init__(
        self,
        num_samples: int = 100,
        lookback_length: int = 512,
        forecast_length: int = 96,
        seed: int = 42,
        augment: bool = True
    ):
        """
        Args:
            num_samples: Number of samples to generate
            lookback_length: Length of past context
            forecast_length: Length of forecast horizon
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.num_samples = num_samples
        self.lookback_length = lookback_length
        self.forecast_length = forecast_length
        self.total_length = lookback_length + forecast_length
        self.augment = augment
        
        # Set seed for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Pre-generate all data
        self.data = self._generate_data()
        
        logger.info(f"SyntheticTimeSeriesDataset created: {num_samples} samples, "
                   f"lookback={lookback_length}, forecast={forecast_length}")
    
    def _generate_data(self) -> torch.Tensor:
        """Generate synthetic time series data."""
        data = []
        
        for i in range(self.num_samples):
            # Time axis
            t = np.linspace(0, 10 * np.pi, self.total_length)
            
            # Base pattern: combination of sinusoids
            freq1 = np.random.uniform(0.5, 2.0)
            freq2 = np.random.uniform(2.0, 5.0)
            phase1 = np.random.uniform(0, 2 * np.pi)
            phase2 = np.random.uniform(0, 2 * np.pi)
            
            signal = (
                np.sin(freq1 * t + phase1) + 
                0.5 * np.sin(freq2 * t + phase2)
            )
            
            # Add trend
            trend_type = np.random.choice(['none', 'linear', 'quadratic'])
            if trend_type == 'linear':
                trend = np.random.uniform(-0.1, 0.1) * t
                signal += trend
            elif trend_type == 'quadratic':
                trend = np.random.uniform(-0.01, 0.01) * t ** 2
                signal += trend
            
            # Add high-frequency "texture" (W/V shapes)
            if np.random.random() > 0.3:
                hf_freq = np.random.uniform(10, 20)
                hf_amp = np.random.uniform(0.1, 0.3)
                signal += hf_amp * np.sin(hf_freq * t)
            
            # Add some sharp spikes/dips (to test shape preservation)
            num_spikes = np.random.randint(0, 5)
            for _ in range(num_spikes):
                spike_pos = np.random.randint(0, self.total_length)
                spike_width = np.random.randint(3, 10)
                spike_height = np.random.uniform(-1, 1)
                
                start = max(0, spike_pos - spike_width // 2)
                end = min(self.total_length, spike_pos + spike_width // 2)
                
                # Create triangular spike (W or V shape)
                mid = (start + end) // 2
                for j in range(start, mid):
                    signal[j] += spike_height * (j - start) / (mid - start)
                for j in range(mid, end):
                    signal[j] += spike_height * (end - j) / (end - mid)
            
            # Add noise
            noise_level = np.random.uniform(0.05, 0.2)
            signal += np.random.normal(0, noise_level, self.total_length)
            
            data.append(signal)
        
        return torch.tensor(np.array(data), dtype=torch.float32)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (past, future) tuple where:
            - past: shape (lookback_length,)
            - future: shape (forecast_length,)
        """
        full_seq = self.data[idx]
        
        if self.augment:
            full_seq = apply_1d_augmentations(full_seq)
        
        past = full_seq[:self.lookback_length]
        future = full_seq[self.lookback_length:]
        return past, future


def create_toy_batch(
    batch_size: int = 4,
    lookback_length: int = 512,
    forecast_length: int = 96,
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create a single toy batch for quick testing.
    
    Args:
        batch_size: Number of samples in batch
        lookback_length: Length of past context
        forecast_length: Length of forecast horizon
        device: Device to place tensors on
        
    Returns:
        (past, future) tuple
    """
    dataset = SyntheticTimeSeriesDataset(
        num_samples=batch_size,
        lookback_length=lookback_length,
        forecast_length=forecast_length
    )
    
    # Stack all samples into a batch
    past_list = []
    future_list = []
    for i in range(batch_size):
        p, f = dataset[i]
        past_list.append(p)
        future_list.append(f)
    
    past = torch.stack(past_list).to(device)
    future = torch.stack(future_list).to(device)
    
    logger.info(f"Created toy batch: past={past.shape}, future={future.shape}")
    
    return past, future


def get_dataloader(
    num_samples: int = 100,
    lookback_length: int = 512,
    forecast_length: int = 96,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    augment: bool = True
) -> DataLoader:
    """Create a DataLoader for synthetic data.
    
    Args:
        num_samples: Total number of samples
        lookback_length: Length of past context
        forecast_length: Length of forecast horizon
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        
    Returns:
        DataLoader instance
    """
    dataset = SyntheticTimeSeriesDataset(
        num_samples=num_samples,
        lookback_length=lookback_length,
        forecast_length=forecast_length,
        augment=augment
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )


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
    seed: Optional[int] = None
) -> DataLoader:
    """Create a DataLoader with ONLY synthetic RealTS data for pre-training.
    
    This is used for the pre-training phase where the model learns general
    time series structure from diverse synthetic patterns before fine-tuning
    on real data.
    
    Args:
        num_samples: Number of synthetic samples to generate
        lookback_length: Past context window length
        forecast_length: Forecast horizon length
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        seed: Random seed for reproducibility (None for random)
        
    Returns:
        DataLoader with synthetic-only data
    """
    synthetic_dataset = RealTS(
        num_samples=num_samples,
        lookback_length=lookback_length,
        forecast_length=forecast_length,
        seed=seed
    )
    
    logger.info(
        f"Created synthetic-only dataloader: {num_samples} samples, "
        f"lookback={lookback_length}, forecast={forecast_length}"
    )
    
    return DataLoader(
        synthetic_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


def create_mixed_dataset(
    real_dataset: Dataset,
    synthetic_size: int = 10000,
    lookback_length: int = 512,
    forecast_length: int = 96,
    seed: Optional[int] = None
) -> ConcatDataset:
    """Create a combined dataset of real and synthetic data.
    
    NOTE: For pre-train + fine-tune workflow, use get_synthetic_dataloader()
    for pre-training and real dataloader for fine-tuning instead.
    
    This function is kept for backward compatibility but the two-phase
    approach (pre-train on synthetic, then fine-tune on real) is preferred.
    
    Args:
        real_dataset: The real data dataset (e.g., ElectricityDataset)
        synthetic_size: Number of synthetic samples to generate
        lookback_length: Past context window length
        forecast_length: Forecast horizon length
        seed: Random seed for synthetic data (None for random)
        
    Returns:
        ConcatDataset combining real and synthetic data
    """
    # Create synthetic dataset
    synthetic_dataset = RealTS(
        num_samples=synthetic_size,
        lookback_length=lookback_length,
        forecast_length=forecast_length,
        seed=seed
    )
    
    # Combine datasets
    combined = ConcatDataset([real_dataset, synthetic_dataset])
    
    logger.info(
        f"Created mixed dataset: {len(real_dataset)} real + "
        f"{len(synthetic_dataset)} synthetic = {len(combined)} total samples"
    )
    
    return combined


def get_mixed_dataloader(
    real_dataset: Dataset,
    synthetic_size: int = 10000,
    lookback_length: int = 512,
    forecast_length: int = 96,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    seed: Optional[int] = None
) -> DataLoader:
    """Create a DataLoader for mixed real + synthetic data.
    
    NOTE: For pre-train + fine-tune workflow, use get_synthetic_dataloader()
    for pre-training and real dataloader for fine-tuning instead.
    
    Args:
        real_dataset: The real data dataset
        synthetic_size: Number of synthetic samples
        lookback_length: Past context window length
        forecast_length: Forecast horizon length
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        seed: Random seed for synthetic data
        
    Returns:
        DataLoader with mixed real and synthetic data
    """
    combined = create_mixed_dataset(
        real_dataset=real_dataset,
        synthetic_size=synthetic_size,
        lookback_length=lookback_length,
        forecast_length=forecast_length,
        seed=seed
    )
    
    return DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

