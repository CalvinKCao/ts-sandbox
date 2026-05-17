import os
import sys
import numpy as np
import pandas as pd
import scipy.signal as signal
from pathlib import Path

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.insert(0, project_root)

from models.diffusion_tsf.train_multivariate_pipeline import DATASET_REGISTRY, DATASETS_DIR

def test_butterworth_residual():
    path, date_col, _ = DATASET_REGISTRY["ETTh1"]
    full_path = os.path.join(DATASETS_DIR, path)
    
    df = pd.read_csv(full_path)
    # Get just the numeric data
    data = df.drop(columns=[date_col]).values
    
    # create a mock future window (batch_size, num_vars, pred_len)
    pred_len = 96
    batch_size = 32
    num_vars = data.shape[1]
    
    future = np.zeros((batch_size, num_vars, pred_len))
    for b in range(batch_size):
        start = np.random.randint(0, len(data) - pred_len)
        future[b] = data[start:start+pred_len].T

    cutoff_freq = 0.1 # Example cutoff
    b, a = signal.butter(4, cutoff_freq, btype='low')
    
    # Apply filtfilt over the sequence dimension (axis=-1)
    trend = signal.filtfilt(b, a, future, axis=-1)
    residual = future - trend
    
    print(f"Original target mean: {future.mean():.4f}, std: {future.std():.4f}")
    print(f"Trend mean: {trend.mean():.4f}, std: {trend.std():.4f}")
    print(f"Residual mean: {residual.mean():.4f}, std: {residual.std():.4f}")
    
    # Check if residual is approximately mean-zero
    is_mean_zero = np.abs(residual.mean()) < 0.05
    print(f"Is residual approximately mean-zero? {is_mean_zero}")

if __name__ == "__main__":
    test_butterworth_residual()
