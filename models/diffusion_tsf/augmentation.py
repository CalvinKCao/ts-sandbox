
import numpy as np
import torch
import logging
from typing import List, Optional, Callable, Dict, Any, Union, Tuple

logger = logging.getLogger(__name__)

# ============================================================================
# Core Augmentation Logic
# ============================================================================

def informative_covariate_augmentation(
    y: np.ndarray,
    corpus: List[np.ndarray],
    synth_generator: Callable[[int, Dict], np.ndarray],
    hyperparams: Dict[str, Any],
    rng: np.random.Generator
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Algorithm 1: Informative Covariate Augmentation.
    
    Augments a target series y by adding impacts from sampled covariates.
    
    Args:
        y: Target series of shape (T,)
        corpus: List of candidate covariate series
        synth_generator: Function to generate synthetic covariates
        hyperparams: Hyperparameters dictionary
        rng: Random number generator
        
    Returns:
        y_aug: Augmented target series (T,)
        covariates: List of covariates used (k, T)
    """
    T = len(y)
    p = hyperparams.get('p', 0.25)
    kmax = hyperparams.get('kmax', 10)
    
    # 1. Sample k (number of covariates)
    # geometric distribution in numpy: rng.geometric(p)
    kappa = rng.geometric(p=p)
    k = min(kappa, kmax)
    
    covariates = []
    impact_functions = []
    
    for i in range(k):
        # 2. Sample covariate either from corpus or synthetic generator
        # 50% chance to use corpus if available
        use_corpus = (len(corpus) > 0) and (rng.random() < 0.5)
        
        if use_corpus:
            # Sample from corpus and clip/pad to length T
            idx = rng.integers(0, len(corpus))
            x_raw = corpus[idx]
            if len(x_raw) >= T:
                start = rng.integers(0, len(x_raw) - T + 1)
                x_i = x_raw[start:start+T]
            else:
                # Pad if too short (shouldn't happen with proper corpus prep, but safety first)
                pad_len = T - len(x_raw)
                x_i = np.pad(x_raw, (0, pad_len), mode='edge')
        else:
            x_i = synth_generator(T, hyperparams, rng)
            
        covariates.append(x_i)
        
        # 3. Sample impact function f_i
        f_i = sample_impact_function(hyperparams, rng)
        impact_functions.append(f_i)
        
    # 4. Apply impacts
    y_aug = y.copy()
    for f_i, x_i in zip(impact_functions, covariates):
        # f_i is a callable: f(x, y) -> impact_array
        impact = f_i(x_i, y)
        y_aug += impact
        
    return y_aug, covariates


def sample_impact_function(
    hyperparams: Dict[str, Any],
    rng: np.random.Generator
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Algorithm 2: Sample Impact Function.
    
    Returns a callable f(x, y) -> impact_array.
    """
    pFO = hyperparams.get('pFO', 0.2)
    pPW = hyperparams.get('pPW', 0.15)
    plagcount = hyperparams.get('plagcount', 0.85)
    plagpos = hyperparams.get('plagpos', 0.15)
    l_max = hyperparams.get('l', 500)
    s_epsilon = hyperparams.get('s_epsilon', 0.02)
    
    # Initialize coefficients
    a = np.zeros(l_max + 1)
    
    if rng.random() > pFO:
        # Sample number of active lags
        clag = rng.geometric(p=plagcount)
        
        # Sample lag positions
        L = [rng.geometric(p=plagpos) for _ in range(clag)]
        
        # Sample coefficients
        A = [rng.normal(0, 1) for _ in range(clag)]
        
        for lag, coeff in zip(L, A):
            if lag <= l_max:
                a[lag] = coeff
                
        # Piecewise parameters
        if rng.random() > pPW:
            a0 = rng.normal(0, 1)
            z_type = rng.choice(['y', 'x'])
            relation = rng.choice(['>', '<'])
            q = rng.random()
            is_piecewise = True
        else:
            a0 = 0.0
            z_type = 'y'
            relation = '>'
            q = 0.0
            is_piecewise = False
            
    else:
        # No impact
        def zero_impact(x, y):
            return np.zeros_like(x)
        return zero_impact
    
    def f_callable(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        T = len(x)
        
        # Domain selection S(x,y)
        if is_piecewise:
            target_series = y if z_type == 'y' else x
            threshold = np.quantile(target_series, q)
            
            if relation == '>':
                S_mask = target_series > threshold
            else:
                S_mask = target_series < threshold
        else:
            S_mask = np.ones(T, dtype=bool)
            
        # Compute convolution <a, x_{t-l:t}>
        # We can use np.convolve. 
        # a is indices 0..l_max. a[k] is coeff for lag k (x_{t-k}).
        # To use np.convolve, we reverse 'a' because convolve(x, a) corresponds to sum x[t-k]*a[k].
        # However, np.convolve handles the flip. 
        # Standard discrete convolution: (x * h)[n] = sum x[k] h[n-k].
        # We want sum a[k] * x[t-k]. This is exactly convolution if a is the filter.
        
        # a has shape (l_max+1,). a[0] corresponds to lag 0 (current time).
        # We want output length T.
        # mode='full' gives length T + len(a) - 1. We take the first T points.
        conv_result = np.convolve(x, a, mode='full')[:T]
        
        # Compute impact
        impact = np.zeros(T)
        
        # Scale of impact for noise calculation (simple heuristic: std of conv result)
        impact_std = np.std(conv_result) if np.std(conv_result) > 1e-9 else 1.0
        epsilon = rng.normal(0, impact_std * s_epsilon, size=T)
        
        # Apply formula only on S
        impact[S_mask] = a0 + conv_result[S_mask] + epsilon[S_mask]
        
        return impact
        
    return f_callable


# ============================================================================
# Synthetic Covariate Generation
# ============================================================================

def generate_synthetic_covariate(
    T: int, 
    hyperparams: Dict[str, Any],
    rng: np.random.Generator
) -> np.ndarray:
    """
    Algorithm 3: Synthetic Covariates Generation.
    """
    cmax_e = hyperparams.get('cmax_e', 20)
    cmax_cp = hyperparams.get('cmax_cp', 8)
    sigma_cp = hyperparams.get('sigma_cp', 2.0)
    
    # 1. Sample event count
    ce = rng.integers(1, cmax_e + 1)
    
    # 2. Sample event positions
    Pe = rng.integers(0, T, size=ce)
    
    # 3. Sample type
    event_type = rng.choice(['step', 'gauss'])
    
    # 4. Build xe (events)
    xe = np.zeros(T)
    for pos in Pe:
        amplitude = rng.normal(0, 1) # sample_amplitude
        
        if event_type == 'gauss':
            sigma = rng.uniform(1.0, 50.0) # sample_width
            
            # Gaussian kernel
            # G(t) = exp(-0.5 * ((t-pos)/sigma)^2)
            t_indices = np.arange(T)
            kernel = np.exp(-0.5 * ((t_indices - pos) / sigma)**2)
            xe += amplitude * kernel
            
        else: # step
            # Alternating step
            # The paper says "step alternates at each event position".
            # We implement a cumulative step function.
            # Here we just add a step starting at pos.
            xe[pos:] += amplitude
            # To "alternate", we could flip the sign of amplitude for subsequent events,
            # but random amplitude sign is sufficient.
            
    # 5. Sample change-point count
    ccp = rng.integers(0, cmax_cp + 1)
    
    # 6. Sample change-points
    if ccp > 0:
        pi = np.sort(rng.integers(0, T, size=ccp))
        # 7. Sample amplitudes
        # points: 0, pi_1, ..., pi_ccp, T-1
        # values: a_0, ..., a_{ccp+1}
        points = np.concatenate(([0], pi, [T-1]))
        values = rng.normal(0, sigma_cp, size=len(points))
        
        # Interpolate
        xtrend = np.interp(np.arange(T), points, values)
    else:
        # Just a linear trend from start to end
        values = rng.normal(0, sigma_cp, size=2)
        xtrend = np.linspace(values[0], values[1], T)
        
    return xe + xtrend

# ============================================================================
# API
# ============================================================================

def generate_multivariate_synthetic_data(
    num_samples: int,
    num_vars: int,
    length: int,
    hyperparams: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate a batch of synthetic multivariate time series.
    
    Uses the informative covariate augmentation to couple independently generated series.
    Now utilizes organic generators (RWB, IFFTB, etc.) for base series.
    
    Args:
        num_samples: Number of samples (batch size)
        num_vars: Number of variables (channels)
        length: Sequence length (T+h)
        
    Returns:
        Data tensor of shape (num_samples, num_vars, length)
    """
    # Import here to avoid circular dependency with realts.py
    try:
        from .realts import IFFTB, seasonal_periodicity, PWB, RWB, LGB, TWDB
    except ImportError:
        # Fallback for different execution contexts
        from models.diffusion_tsf.realts import IFFTB, seasonal_periodicity, PWB, RWB, LGB, TWDB

    if seed is not None:
        rng = np.random.default_rng(seed)
        np.random.seed(seed)
    else:
        rng = np.random.default_rng()
        
    if hyperparams is None:
        hyperparams = {
            'p': 0.25, 'pFO': 0.2, 'pPW': 0.15, 'kmax': 5,
            'plagcount': 0.85, 'plagpos': 0.15, 'l': min(100, length // 2),
            's_epsilon': 0.02, 'cmax_e': 10, 'cmax_cp': 5, 'sigma_cp': 2.0
        }
    
    # List of available organic generators
    ORGANIC_GENERATORS = [IFFTB, seasonal_periodicity, PWB, RWB, LGB, TWDB]
    
    data = np.zeros((num_samples, num_vars, length))
    
    for s in range(num_samples):
        # 1. Generate base independent series using organic generators
        series_list = []
        for v in range(num_vars):
            # Pick a random generator for this variable's base behavior
            gen_idx = rng.integers(0, len(ORGANIC_GENERATORS))
            gen_func = ORGANIC_GENERATORS[gen_idx]
            base = gen_func(length)
            
            # Normalize base
            base = (base - np.mean(base)) / (np.std(base) + 1e-6)
            series_list.append(base)
            
        # 2. Apply augmentation to couple them
        augmented_series = []
        for v in range(num_vars):
            y = series_list[v]
            # Potential covariates are all other series
            others = [series_list[j] for j in range(num_vars) if j != v]
            
            # Define a wrapper for the synth generator to be used in augmentation
            # so it also picks from organic behaviors
            def organic_synth_wrapper(T, hp, r):
                idx = rng.integers(0, len(ORGANIC_GENERATORS))
                g = ORGANIC_GENERATORS[idx]
                return g(T)
            
            y_aug, _ = informative_covariate_augmentation(
                y, others, organic_synth_wrapper, hyperparams, rng
            )
            
            # Normalize result
            y_aug = (y_aug - np.mean(y_aug)) / (np.std(y_aug) + 1e-6)
            augmented_series.append(y_aug)
            
        data[s] = np.stack(augmented_series)
        
    return data
