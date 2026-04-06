import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants (matching project structure)
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "fft_visualizations")

DATASET_REGISTRY = {
    'ETTh1': ('ETT-small/ETTh1.csv', 'date'),
    'ETTh2': ('ETT-small/ETTh2.csv', 'date'),
    'ETTm1': ('ETT-small/ETTm1.csv', 'date'),
    'ETTm2': ('ETT-small/ETTm2.csv', 'date'),
    'illness': ('illness/national_illness.csv', 'date'),
    'exchange_rate': ('exchange_rate/exchange_rate.csv', 'date'),
    'weather': ('weather/weather.csv', 'date'),
    'electricity': ('electricity/electricity.csv', 'date'),
    'traffic': ('traffic/traffic.csv', 'date'),
}

def apply_fft_lowpass(signal, n_remove):
    """
    Apply FFT, remove n_remove highest frequency components, and IFFT back.
    n_remove is the total number of components to zero out from the high-frequency end.
    """
    L = len(signal)
    if n_remove >= L:
        return np.zeros_like(signal)
    
    # Compute FFT
    coeffs = fft(signal)
    
    # Zero out high frequencies. 
    # For real signals, high frequencies are in the middle of the FFT array.
    # Indices: 0 (DC), 1...L/2 (positive), L/2+1...L-1 (negative)
    # We zero out n_remove elements centered around L/2.
    
    mid = (L + 1) // 2
    half_remove = n_remove // 2
    
    # Create mask
    mask = np.ones(L, dtype=bool)
    if n_remove > 0:
        start = max(1, mid - half_remove)
        end = min(L, mid + (n_remove - half_remove))
        mask[start:end] = False
    
    coeffs_filtered = coeffs.copy()
    coeffs_filtered[~mask] = 0
    
    # Inverse FFT
    reconstructed = ifft(coeffs_filtered)
    return reconstructed.real

def visualize_dataset(name, path_info, n_samples=512):
    rel_path, date_col = path_info
    full_path = os.path.join(DATASETS_DIR, rel_path)
    
    if not os.path.exists(full_path):
        if name == 'traffic':
            logger.info("traffic.csv not found, attempting to recombine...")
            from models.diffusion_tsf.train_multivariate_pipeline import recombine_traffic_data
            recombine_traffic_data()
            if not os.path.exists(full_path):
                logger.error(f"Failed to find or create {full_path}")
                return
        else:
            logger.error(f"Dataset path {full_path} does not exist")
            return

    logger.info(f"Processing dataset: {name}")
    df = pd.read_csv(full_path)
    data_cols = [c for c in df.columns if c != date_col]
    
    # Pick the first variable and a segment of data
    # We use a power-of-2 length for cleaner FFT behavior
    segment_len = n_samples
    if len(df) < segment_len:
        segment_len = len(df)
    
    # Just take the first column for visualization
    target_col = data_cols[0]
    signal = df[target_col].values[:segment_len]
    
    # Normalize for better visualization
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
    
    # Define cutoff levels (number of frequencies to remove)
    # We'll use percentages of the total length
    cutoffs = [0, int(segment_len * 0.5), int(segment_len * 0.8), int(segment_len * 0.95), int(segment_len * 0.99)]
    
    plt.figure(figsize=(15, 12))
    
    # Original signal
    plt.subplot(len(cutoffs), 1, 1)
    plt.plot(signal, label='Original', color='black', alpha=0.7)
    plt.title(f"Dataset: {name} | Variable: {target_col}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    for i, n_remove in enumerate(cutoffs[1:], 2):
        reconstructed = apply_fft_lowpass(signal, n_remove)
        plt.subplot(len(cutoffs), 1, i)
        plt.plot(signal, color='gray', alpha=0.3, label='Original')
        plt.plot(reconstructed, label=f'Removed {n_remove} High Freqs ({n_remove/segment_len:.1%})', color='blue')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    out_path = os.path.join(RESULTS_DIR, f"{name}_fft_cutoff.png")
    plt.savefig(out_path)
    plt.close()
    logger.info(f"Saved visualization to {out_path}")

def main():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        logger.info(f"Created results directory: {RESULTS_DIR}")
    
    for name, path_info in DATASET_REGISTRY.items():
        try:
            visualize_dataset(name, path_info)
        except Exception as e:
            logger.error(f"Error processing {name}: {e}")

if __name__ == "__main__":
    main()
