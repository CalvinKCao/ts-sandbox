"""
RealTS: Synthetic Time Series Generation inspired by ViTime paper.

This module implements various time series generation functions that produce
diverse synthetic patterns for training diffusion models. These help with
generalizability and structural learning, especially for small datasets.

Generator Functions:
- RWB: Random Walk Behavior
- PWB: Periodic Wave Behavior  
- LGB: Logistic Growth Behavior
- TWDB: Trend + Wave Data Behavior
- IFFTB: Inverse FFT Behavior (synthetic spectrum)
- seasonal_periodicity: Complex seasonal patterns

Reference: ViTime Paper - "Foundation Model for Time Series Forecasting 
           Powered by Vision Intelligence" (Yang et al., 2025)
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple
import logging

try:
    from .augmentation import generate_multivariate_synthetic_data
except ImportError:
    from augmentation import generate_multivariate_synthetic_data

logger = logging.getLogger(__name__)


# ============================================================================
# Generator Functions
# ============================================================================

def RWB(length: int) -> np.ndarray:
    """Random Walk Behavior.
    
    Generates a random walk process where each value is the previous value
    plus a random step drawn from a normal distribution.
    
    Formula: x_t = x_{t-1} + ε_t, where ε_t ~ N(0, σ²)
    
    Args:
        length: Number of time steps to generate
        
    Returns:
        1D numpy array of shape (length,)
    """
    # Sample noise scale uniformly from [0.1, 1.0] for volatility variation
    sigma = np.random.uniform(0.1, 1.0)
    
    # Generate random steps
    steps = np.random.normal(0, sigma, length)
    
    # Random starting point
    start = np.random.uniform(-1, 1)
    
    # Cumulative sum to create random walk
    walk = np.cumsum(steps) + start
    
    return walk


def PWB(length: int) -> np.ndarray:
    """Periodic Wave Behavior.
    
    Generates time series by superimposing multiple periodic waves (sin/cos)
    with varying amplitudes, frequencies, and phases.
    
    Formula: x_t = Σ A_k * sin(2π * f_k * t + φ_k)
    
    Args:
        length: Number of time steps to generate
        
    Returns:
        1D numpy array of shape (length,)
    """
    # Number of components: randomly choose 1-5
    num_components = np.random.randint(1, 6)
    
    # Time axis normalized to [0, 1]
    t = np.linspace(0, 1, length)
    
    signal = np.zeros(length)
    
    for _ in range(num_components):
        # Amplitude: uniform [0.5, 2.0]
        amplitude = np.random.uniform(0.5, 2.0)
        
        # Frequency: log-uniform distribution for 1 to ~10 cycles per window
        # Sample log(f) uniformly, then exponentiate
        log_freq = np.random.uniform(np.log(1), np.log(10))
        frequency = np.exp(log_freq)
        
        # Phase: uniform [0, 2π]
        phase = np.random.uniform(0, 2 * np.pi)
        
        # Add this component
        signal += amplitude * np.sin(2 * np.pi * frequency * t + phase)
    
    return signal


def LGB(length: int) -> np.ndarray:
    """Logistic Growth Behavior.
    
    Simulates S-curve trends using the logistic function.
    
    Formula: x_t = K / (1 + exp(-r * (t - t0)))
    
    Args:
        length: Number of time steps to generate
        
    Returns:
        1D numpy array of shape (length,)
    """
    # Time axis
    t = np.arange(length, dtype=np.float64)
    
    # Carrying capacity K: log-uniform between 1 and 10
    log_K = np.random.uniform(np.log(1), np.log(10))
    K = np.exp(log_K)
    
    # Growth rate r: log-uniform between 0.01 and 0.1
    log_r = np.random.uniform(np.log(0.01), np.log(0.1))
    r = np.exp(log_r)
    
    # Midpoint t0: uniform across the sequence range
    t0 = np.random.uniform(0, length)
    
    # Logistic function
    signal = K / (1 + np.exp(-r * (t - t0)))
    
    # Add small Gaussian noise
    noise = np.random.normal(0, 0.05 * K, length)
    signal += noise
    
    return signal


def TWDB(length: int) -> np.ndarray:
    """Trend + Wave Data Behavior.
    
    Combines a linear trend with periodic waves from PWB.
    
    Formula: x_t = slope * t + intercept + PWB(t)
    
    Args:
        length: Number of time steps to generate
        
    Returns:
        1D numpy array of shape (length,)
    """
    # Time axis normalized
    t = np.linspace(0, 1, length)
    
    # Linear trend parameters
    # Slope: uniform [-2, 2]
    slope = np.random.uniform(-2, 2)
    
    # Intercept: uniform [-1, 1]
    intercept = np.random.uniform(-1, 1)
    
    # Linear component
    linear = slope * t + intercept
    
    # Add periodic waves
    waves = PWB(length)
    
    # Scale waves to be subordinate to trend
    wave_scale = np.random.uniform(0.3, 0.7)
    
    signal = linear + wave_scale * waves
    
    return signal


def IFFTB(length: int) -> np.ndarray:
    """Inverse FFT Behavior (Synthetic Spectrum Generator).
    
    Creates complex periodicities by generating a synthetic frequency spectrum
    with sparse peaks and noise floor, then applying inverse FFT.
    
    This simulates real-world data that has sparse dominant frequencies
    plus background noise.
    
    Args:
        length: Number of time steps to generate
        
    Returns:
        1D numpy array of shape (length,)
    """
    # Create frequency domain array (complex numbers)
    freq_domain = np.zeros(length, dtype=complex)
    
    # Number of sparse peaks: 2-5 dominant frequencies
    num_peaks = np.random.randint(2, 6)
    
    # Select random frequency indices (excluding DC and Nyquist)
    max_freq_idx = length // 2
    peak_indices = np.random.choice(
        range(1, max_freq_idx), 
        size=min(num_peaks, max_freq_idx - 1), 
        replace=False
    )
    
    for idx in peak_indices:
        # Magnitude: randomly between 1 and 5
        magnitude = np.random.uniform(1, 5)
        
        # Phase: random [0, 2π]
        phase = np.random.uniform(0, 2 * np.pi)
        
        # Set positive frequency
        freq_domain[idx] = magnitude * np.exp(1j * phase)
        
        # Set corresponding negative frequency for real signal (conjugate symmetry)
        if idx < length - idx:
            freq_domain[length - idx] = magnitude * np.exp(-1j * phase)
    
    # Add noise floor to all frequencies
    noise_floor = np.random.normal(0, 0.1, length) + 1j * np.random.normal(0, 0.1, length)
    freq_domain += noise_floor
    
    # Apply inverse FFT
    signal = np.fft.ifft(freq_domain)
    
    # Take real part and normalize
    signal = np.real(signal)
    
    # Normalize to reasonable range
    if np.std(signal) > 1e-8:
        signal = signal / np.std(signal)
    
    return signal


def seasonal_periodicity(length: int) -> np.ndarray:
    """Seasonal Periodicity Pattern.
    
    Generates complex seasonal patterns with multiple harmonics,
    similar to real-world seasonal data (daily, weekly, yearly cycles).
    
    Args:
        length: Number of time steps to generate
        
    Returns:
        1D numpy array of shape (length,)
    """
    t = np.linspace(0, 1, length)
    
    # Base seasonal period (normalized)
    # Choose a period that creates 2-8 complete cycles
    num_cycles = np.random.uniform(2, 8)
    base_period = 1 / num_cycles
    
    signal = np.zeros(length)
    
    # Add fundamental frequency
    amplitude1 = np.random.uniform(1.0, 2.0)
    phase1 = np.random.uniform(0, 2 * np.pi)
    signal += amplitude1 * np.sin(2 * np.pi * t / base_period + phase1)
    
    # Add harmonics (2nd and 3rd)
    for harmonic in [2, 3]:
        if np.random.random() > 0.3:  # 70% chance to add each harmonic
            amp = amplitude1 / (harmonic * np.random.uniform(1.5, 3))
            phase = np.random.uniform(0, 2 * np.pi)
            signal += amp * np.sin(2 * np.pi * harmonic * t / base_period + phase)
    
    # Add slow trend modulation
    if np.random.random() > 0.5:
        trend_amp = np.random.uniform(0.1, 0.5)
        signal *= (1 + trend_amp * t)
    
    # Add small noise
    signal += np.random.normal(0, 0.1, length)
    
    return signal


# ============================================================================
# RealTS Dataset Class
# ============================================================================

class RealTS(Dataset):
    """Synthetic Time Series Dataset for training data augmentation.
    
    Generates diverse synthetic time series using multiple generator functions.
    Returns raw sequences that match the format expected by the diffusion
    training pipeline (past, future) tuples.
    
    Supports both univariate and multivariate generation.
    
    Mixing probabilities (from ViTime paper):
    - IFFTB: 60% (complex periodicities)
    - PWB: 16% (periodic waves)
    - RWB: 8% (random walks)
    - LGB: 8% (logistic growth)
    - TWDB: 8% (trend + waves)
    
    Args:
        num_samples: Number of synthetic samples to generate
        lookback_length: Length of past context window
        forecast_length: Length of forecast horizon
        seed: Random seed for reproducibility (None for random)
        augment: Whether to apply additional augmentations (reserved for future)
        num_variables: Number of variables to generate (default: 1)
    """
    
    # Generator functions and their probabilities
    GENERATORS = [
        (IFFTB, 0.30),           # Complex periodicities (Type 1)
        (seasonal_periodicity, 0.30),  # Seasonal patterns (Type 2)
        (PWB, 0.16),             # Periodic waves
        (RWB, 0.08),             # Random walks
        (LGB, 0.08),             # Logistic growth
        (TWDB, 0.08),            # Trend + waves
    ]
    
    def __init__(
        self,
        num_samples: int = 10000,
        lookback_length: int = 512,
        forecast_length: int = 96,
        seed: Optional[int] = None,
        augment: bool = False,
        num_variables: int = 1,
        pregenerate: bool = True,
        pool_size: Optional[int] = None,
        cache_dir: Optional[str] = None
    ):
        self.num_samples = num_samples  # Virtual epoch size
        self.lookback_length = lookback_length
        self.forecast_length = forecast_length
        self.total_length = lookback_length + forecast_length
        self.augment = augment
        self.num_variables = num_variables
        self.pregenerate = pregenerate
        self.pool_size = pool_size or num_samples
        if self.pool_size < num_samples: self.pool_size = num_samples
        
        self.data_cache = None
        self.use_disk_cache = False
        
        # Extract generators and probabilities
        self.generators = [g for g, _ in self.GENERATORS]
        self.probabilities = [p for _, p in self.GENERATORS]
        
        # Set seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        logger.info(
            f"RealTS initialized: {num_samples} samples/epoch, "
            f"lookback={lookback_length}, forecast={forecast_length}, "
            f"variables={num_variables}"
        )
        
        # Disk Caching Logic (Large Pool)
        if cache_dir:
            import os
            os.makedirs(cache_dir, exist_ok=True)
            self.use_disk_cache = True
            
            # Helper to generate data
            def gen_data(size):
                if self.num_variables > 1:
                    return generate_multivariate_synthetic_data(
                        num_samples=size,
                        num_vars=self.num_variables,
                        length=self.total_length,
                        seed=seed
                    )
                else:
                    # Generate univariate in bulk
                    data = np.zeros((size, self.total_length), dtype=np.float32)
                    for i in range(size):
                        gen = np.random.choice(self.generators, p=self.probabilities)
                        seq = gen(self.total_length)
                        if np.random.random() < 0.5: seq = seq[::-1].copy()
                        if np.random.random() < 0.5: seq = -seq
                        data[i] = self._normalize_sequence(seq)
                    return data

            cache_filename = f"synth_pool_v{self.num_variables}_L{self.total_length}_N{self.pool_size}.npy"
            if seed is not None:
                cache_filename = f"synth_pool_v{self.num_variables}_L{self.total_length}_N{self.pool_size}_seed{seed}.npy"
            
            cache_path = os.path.join(cache_dir, cache_filename)
            
            if os.path.exists(cache_path):
                logger.info(f"Loading synthetic pool from {cache_path}")
                # Use mmap_mode='r' to avoid loading everything into RAM if it's huge
                # But for <10GB, loading into RAM is usually faster for random access training
                # Let's try mmap first, if user has RAM they can load it.
                self.data_cache = np.load(cache_path, mmap_mode='r')
            else:
                logger.info(f"Generating synthetic pool of {self.pool_size} samples to {cache_path}...")
                data = gen_data(self.pool_size)
                np.save(cache_path, data)
                self.data_cache = np.load(cache_path, mmap_mode='r')
                logger.info("Pool generation complete.")
                
        elif self.pregenerate and self.num_variables > 1:
            # Memory Caching Logic (Small Pool / Legacy)
            logger.info(f"Pre-generating {self.num_samples} multivariate samples (RAM)...")
            self.data_cache = generate_multivariate_synthetic_data(
                num_samples=self.num_samples,
                num_vars=self.num_variables,
                length=self.total_length,
                seed=seed
            )
            self.pool_size = self.num_samples # Pool size is fixed to what we generated
            logger.info("Pre-generation complete.")
    
    def __len__(self) -> int:
        return self.num_samples
    
    def _normalize_sequence(self, seq: np.ndarray) -> np.ndarray:
        """Apply ViTime-style normalization."""
        # Compute mean and std
        mean = np.mean(seq)
        std = np.std(seq) + 1e-7
        
        # Z-score normalization
        normalized = (seq - mean) / std
        
        return normalized
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a synthetic (past, future) pair.
        
        Args:
            idx: Sample index (ignored if using random sampling from pool)
            
        Returns:
            Tuple of (past, future):
            - past: shape (lookback_length,) or (num_vars, lookback_length)
            - future: shape (forecast_length,) or (num_vars, forecast_length)
        """
        
        # Case 1: Using Cached Pool (Disk or RAM)
        if self.data_cache is not None:
            # Pick a random index from the pool to simulate infinite data cycling
            # We ignore 'idx' which cycles 0..num_samples
            real_idx = np.random.randint(0, self.pool_size)
            seq = self.data_cache[real_idx]
            
            # Note: For cached univariate, we already did augmentations (flip/negate) at generation time.
            # But we could do more here if needed. For now, assume pool is sufficient.
            
            if self.num_variables > 1:
                past = seq[:, :self.lookback_length]
                future = seq[:, self.lookback_length:]
            else:
                past = seq[:self.lookback_length]
                future = seq[self.lookback_length:]
                
            # If using mmap, need to copy to array to make it writable/torch-compatible
            past = np.array(past)
            future = np.array(future)
            
            return torch.tensor(past, dtype=torch.float32), torch.tensor(future, dtype=torch.float32)

        # Case 2: On-the-fly Generation (Legacy / Univariate RAM)
        if self.num_variables > 1:
            # Multivariate generation using augmentation module
            # Generate 1 sample with num_variables
            seq_batch = generate_multivariate_synthetic_data(
                num_samples=1,
                num_vars=self.num_variables,
                length=self.total_length
            )
            seq = seq_batch[0]  # (num_vars, total_length)
            
            # Split into past and future
            past = seq[:, :self.lookback_length]
            future = seq[:, self.lookback_length:]
            
        else:
            # Univariate generation (original logic)
            generator = np.random.choice(self.generators, p=self.probabilities)
            seq = generator(self.total_length)
            
            # Randomly flip the sequence (data augmentation)
            if np.random.random() < 0.5:
                seq = seq[::-1].copy()
            
            # Randomly negate (data augmentation)
            if np.random.random() < 0.5:
                seq = -seq
            
            # Normalize
            seq = self._normalize_sequence(seq)
            
            # Split into past and future
            past = seq[:self.lookback_length]
            future = seq[self.lookback_length:]
        
        # Convert to tensors
        past_tensor = torch.tensor(past, dtype=torch.float32)
        future_tensor = torch.tensor(future, dtype=torch.float32)
        
        return past_tensor, future_tensor



