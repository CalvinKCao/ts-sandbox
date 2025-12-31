"""
Dataset utilities for Diffusion TSF.

Provides:
- Synthetic data generation for testing
- Simple loading from common TSF datasets
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


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
        seed: int = 42
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
    num_workers: int = 0
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
        forecast_length=forecast_length
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

