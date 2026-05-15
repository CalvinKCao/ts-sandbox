import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

def remove_high_freqs(signal, num_to_remove):
    """
    Remove the num_to_remove highest frequency components from the signal.
    Assumes signal is 1D.
    """
    n = len(signal)
    fft_coeffs = np.fft.rfft(signal)
    
    # In rfft, we have (n//2 + 1) coefficients.
    # The highest frequencies are at the end of the array.
    if num_to_remove > 0:
        # Avoid removing DC component (index 0)
        start_idx = max(1, len(fft_coeffs) - num_to_remove)
        fft_coeffs[start_idx:] = 0
        
    return np.fft.irfft(fft_coeffs, n=n)

def analyze_datasets(base_dir='datasets', output_dir='temp/freq_decomp_plots'):
    os.makedirs(output_dir, exist_ok=True)
    
    csv_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    # Filtering levels
    removal_levels = [0, 50, 100, 200, 400, 800, 1600, 3200, 6400]
    window_size = 16384
    all_mses = {}
    
    for csv_path in csv_files:
        print(f"Processing {csv_path}...")
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Failed to read {csv_path}: {e}")
            continue
            
        # Identify numerical columns (excluding 'date' or 'time')
        cols = [c for c in df.columns if c.lower() not in ['date', 'time']]
        if not cols:
            continue
            
        # Sample 3 variates
        sampled_cols = random.sample(cols, min(3, len(cols)))
        
        # Take a random starting point for the window
        if len(df) < window_size:
            start_idx = 0
            actual_window = len(df)
        else:
            start_idx = random.randint(0, len(df) - window_size)
            actual_window = window_size
            
        dataset_name = os.path.basename(csv_path).replace('.csv', '')
        
        for col in sampled_cols:
            signal = df[col].iloc[start_idx:start_idx + actual_window].values
            
            # Normalize signal for fair MSE comparison across datasets
            signal_norm = (signal - signal.mean()) / (signal.std() + 1e-8)
            
            plt.figure(figsize=(15, 18))
            plt.subplot(len(removal_levels), 1, 1)
            plt.plot(signal, label='Original', color='black', alpha=0.7)
            plt.title(f"Dataset: {dataset_name} | Column: {col} | Start: {start_idx}")
            plt.legend()
            
            for i, k in enumerate(removal_levels[1:], 1):
                if k >= len(np.fft.rfft(signal)):
                    continue
                filtered = remove_high_freqs(signal, k)
                
                # Calculate MSE on normalized signal
                filtered_norm = (filtered - filtered.mean()) / (filtered.std() + 1e-8)
                mse = np.mean((signal_norm - filtered_norm)**2)
                
                if k not in all_mses:
                    all_mses[k] = []
                all_mses[k].append(mse)
                
                plt.subplot(len(removal_levels), 1, i + 1)
                plt.plot(signal, color='gray', alpha=0.3, label='Original')
                plt.plot(filtered, label=f'Removed Top {k} Freqs (MSE: {mse:.4f})', color='tab:blue')
                plt.legend(loc='upper right')
            
            plt.tight_layout()
            out_name = f"{dataset_name}_{col.replace('/', '_')}.png"
            plt.savefig(os.path.join(output_dir, out_name))
            plt.close()

    print("\n" + "="*50)
    print(f"{'Removed Freqs':<15} | {'Avg MSE (Normalized)':<20}")
    print("-" * 50)
    for k in sorted(all_mses.keys()):
        avg_mse = np.mean(all_mses[k])
        print(f"{k:<15} | {avg_mse:<20.6f}")
    print("="*50)

if __name__ == "__main__":
    analyze_datasets()
