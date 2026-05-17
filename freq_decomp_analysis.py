import os
import argparse
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

def analyze_datasets(mode='absolute', base_dir='datasets', output_dir='temp/freq_decomp_plots'):
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    csv_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    window_size = 131072
    plot_limit = 1000 # Only plot first 1000 steps for readability
    all_mses = {}
    
    if mode == 'absolute':
        # Filtering levels (increased to support higher frequency removals)
        levels = [0, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200]
    elif mode == 'percent':
        # Percentage of frequencies to keep (lowest frequencies)
        levels = [100.0, 64.0, 32.0, 16.0, 8.0, 4.0, 2.0, 1.0, 0.5]
    else:
        # Granular percentage mode: 28% to 12% with 4% intervals
        levels = [100.0, 28.0, 24.0, 20.0, 16.0, 12.0]
    
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
            num_coeffs = len(np.fft.rfft(signal))
            
            plt.figure(figsize=(15, 20))
            plt.subplot(len(levels), 1, 1)
            plt.plot(signal[:plot_limit], label=f'Original (first {plot_limit} steps)', color='black', alpha=0.7)
            plt.title(f"Dataset: {dataset_name} | Column: {col} | Start: {start_idx} (Showing first {plot_limit} steps)")
            plt.legend()
            
            for i, val in enumerate(levels[1:], 1):
                if mode == 'absolute':
                    k = int(val)
                    label = f'Removed Top {k} Freqs'
                else:
                    # Calculate how many to remove to keep `val` percentage of frequencies
                    keep_count = max(1, int(num_coeffs * (val / 100.0)))
                    k = num_coeffs - keep_count
                    label = f'Kept Lowest {val}% Freqs'
                
                if k >= num_coeffs:
                    continue
                    
                filtered = remove_high_freqs(signal, k)
                
                # Calculate MSE on normalized signal
                filtered_norm = (filtered - filtered.mean()) / (filtered.std() + 1e-8)
                mse = np.mean((signal_norm - filtered_norm)**2)
                
                if val not in all_mses:
                    all_mses[val] = []
                all_mses[val].append(mse)
                
                plt.subplot(len(levels), 1, i + 1)
                plt.plot(signal[:plot_limit], color='gray', alpha=0.3, label='Original')
                plt.plot(filtered[:plot_limit], label=f'{label} (MSE: {mse:.4f})', color='tab:blue')
                plt.legend(loc='upper right')
            
            plt.tight_layout()
            out_name = f"{dataset_name}_{col.replace('/', '_')}.png"
            plt.savefig(os.path.join(output_dir, out_name))
            plt.close()

    print("\n" + "="*50)
    if mode == 'absolute':
        print(f"{'Removed Freqs':<15} | {'Avg MSE (Normalized)':<20}")
        sort_reverse = False
    else:
        print(f"{'Kept % Freqs':<15} | {'Avg MSE (Normalized)':<20}")
        sort_reverse = True # For percentages, decreasing is more intuitive

    print("-" * 50)
    for k in sorted(all_mses.keys(), reverse=sort_reverse):
        avg_mse = np.mean(all_mses[k])
        print(f"{k:<15} | {avg_mse:<20.6f}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Frequency Decomposition Analysis")
    parser.add_argument("--mode", type=str, choices=['absolute', 'percent', 'granular'], default='absolute', help="Filtering mode (absolute, percent, or granular)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for plots")
    args = parser.parse_args()
    
    out_dir = args.output_dir
    if out_dir is None:
        if args.mode == 'percent':
            out_dir = 'temp/freq_decomp_plots_percent'
        elif args.mode == 'granular':
            out_dir = 'temp/freq_decomp_plots_granular'
        else:
            out_dir = 'temp/freq_decomp_plots'
        
    analyze_datasets(mode=args.mode, output_dir=out_dir)
