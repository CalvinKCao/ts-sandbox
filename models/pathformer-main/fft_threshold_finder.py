"""
FFT Threshold Finder - Visualize Frequency Decomposition

This script helps you find the optimal frequency threshold for FFT-DILATE loss
by visualizing how different thresholds separate high and low frequency components.

Usage:
    python fft_threshold_finder.py --dataset datasets/ETT-small/ETTh1.csv --seq_len 96 --num_samples 3
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path


def load_data(file_path, seq_len=96, num_samples=3):
    """Load time series data and extract samples"""
    print(f"Loading data from: {file_path}")
    
    # Determine file type and load accordingly
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.txt'):
        df = pd.read_csv(file_path, sep='\t')
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    
    # Remove date/time columns if present
    date_cols = ['date', 'Date', 'time', 'Time', 'timestamp', 'Timestamp']
    for col in date_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # Convert to numpy array
    data = df.values.astype(np.float32)
    print(f"Data shape: {data.shape} (timesteps, features)")
    print(f"Features: {df.columns.tolist()}")
    
    # Extract random samples
    num_timesteps = data.shape[0]
    max_start_idx = num_timesteps - seq_len
    
    if max_start_idx <= 0:
        raise ValueError(f"Sequence length {seq_len} is too long for data with {num_timesteps} timesteps")
    
    # Get random start indices
    np.random.seed(42)
    start_indices = np.random.choice(max_start_idx, size=min(num_samples, max_start_idx), replace=False)
    
    samples = []
    for idx in start_indices:
        sample = data[idx:idx+seq_len, :]
        samples.append(sample)
    
    return np.array(samples), df.columns.tolist()


def fft_decomposition(signal, freq_threshold_percentile):
    """
    Decompose signal into high and low frequency components using FFT
    
    Args:
        signal: Input signal (seq_len, n_features)
        freq_threshold_percentile: Percentile threshold (0-100)
        
    Returns:
        high_freq: High frequency component
        low_freq: Low frequency component
        freq_mask: Boolean mask of high frequency components
        magnitude_spectrum: Magnitude spectrum for visualization
    """
    seq_len, n_features = signal.shape
    
    # Perform FFT along time dimension
    fft_result = np.fft.rfft(signal, axis=0)
    
    # Compute magnitude spectrum
    magnitude = np.abs(fft_result)
    
    # Average magnitude across features for threshold calculation
    avg_magnitude = magnitude.mean(axis=1)
    
    # Determine threshold
    threshold_value = np.percentile(avg_magnitude, freq_threshold_percentile)
    
    # Create mask for high frequencies
    freq_mask = avg_magnitude > threshold_value
    
    # Create filtered signals
    high_freq_fft = fft_result.copy()
    low_freq_fft = fft_result.copy()
    
    # Zero out frequencies based on mask
    high_freq_fft[~freq_mask, :] = 0  # Keep only high frequencies
    low_freq_fft[freq_mask, :] = 0    # Keep only low frequencies
    
    # Inverse FFT to get time-domain signals
    high_freq = np.fft.irfft(high_freq_fft, n=seq_len, axis=0)
    low_freq = np.fft.irfft(low_freq_fft, n=seq_len, axis=0)
    
    return high_freq, low_freq, freq_mask, magnitude


def plot_frequency_spectrum(magnitude, freq_mask, threshold, ax, feature_name):
    """Plot frequency spectrum with threshold"""
    n_freqs = len(magnitude)
    freqs = np.arange(n_freqs)
    avg_magnitude = magnitude.mean(axis=1) if magnitude.ndim > 1 else magnitude
    
    # Plot spectrum
    ax.plot(freqs, avg_magnitude, 'b-', linewidth=1, label='Magnitude')
    
    # Highlight high frequencies
    high_freq_indices = np.where(freq_mask)[0]
    if len(high_freq_indices) > 0:
        ax.scatter(high_freq_indices, avg_magnitude[high_freq_indices], 
                  c='red', s=30, alpha=0.6, label='High freq', zorder=5)
    
    ax.axhline(y=np.percentile(avg_magnitude, threshold), 
              color='r', linestyle='--', linewidth=1.5, 
              label=f'{threshold}% threshold')
    
    ax.set_xlabel('Frequency Bin', fontsize=9)
    ax.set_ylabel('Magnitude', fontsize=9)
    ax.set_title(f'Frequency Spectrum - {feature_name}', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)


def plot_decomposition(sample, feature_idx, feature_name, thresholds, output_dir):
    """
    Create comprehensive visualization for one feature across multiple thresholds
    
    Args:
        sample: Time series sample (seq_len, n_features)
        feature_idx: Index of feature to plot
        feature_name: Name of the feature
        thresholds: List of threshold percentiles to test
        output_dir: Directory to save plots
    """
    seq_len = sample.shape[0]
    n_thresholds = len(thresholds)
    
    # Create figure
    fig = plt.figure(figsize=(20, 4 * n_thresholds))
    
    # Original signal (top row)
    original = sample[:, feature_idx]
    
    for i, threshold in enumerate(thresholds):
        # Decompose signal
        high_freq, low_freq, freq_mask, magnitude = fft_decomposition(sample, threshold)
        
        high_freq_signal = high_freq[:, feature_idx]
        low_freq_signal = low_freq[:, feature_idx]
        
        # Calculate statistics
        high_freq_ratio = freq_mask.sum() / len(freq_mask)
        high_energy = np.sum(high_freq_signal ** 2)
        low_energy = np.sum(low_freq_signal ** 2)
        total_energy = high_energy + low_energy
        
        # Row for this threshold (4 subplots)
        row_start = i * 4
        
        # 1. Original signal
        ax1 = plt.subplot(n_thresholds, 4, row_start + 1)
        ax1.plot(original, 'b-', linewidth=1.5, label='Original')
        ax1.set_ylabel('Value', fontsize=9)
        ax1.set_title(f'Original Signal (Threshold={threshold}%)', fontsize=10, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8)
        ax1.tick_params(labelsize=8)
        if i == n_thresholds - 1:
            ax1.set_xlabel('Time', fontsize=9)
        
        # 2. Low frequency component
        ax2 = plt.subplot(n_thresholds, 4, row_start + 2)
        ax2.plot(low_freq_signal, 'g-', linewidth=1.5, label='Low Freq')
        ax2.set_ylabel('Value', fontsize=9)
        energy_pct = (low_energy / total_energy * 100) if total_energy > 0 else 0
        ax2.set_title(f'Low Freq ({100-threshold}% of spectrum, {energy_pct:.1f}% energy)', 
                     fontsize=10, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8)
        ax2.tick_params(labelsize=8)
        if i == n_thresholds - 1:
            ax2.set_xlabel('Time', fontsize=9)
        
        # 3. High frequency component
        ax3 = plt.subplot(n_thresholds, 4, row_start + 3)
        ax3.plot(high_freq_signal, 'r-', linewidth=1.5, label='High Freq')
        ax3.set_ylabel('Value', fontsize=9)
        energy_pct = (high_energy / total_energy * 100) if total_energy > 0 else 0
        ax3.set_title(f'High Freq ({threshold}% of spectrum, {energy_pct:.1f}% energy)', 
                     fontsize=10, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=8)
        ax3.tick_params(labelsize=8)
        if i == n_thresholds - 1:
            ax3.set_xlabel('Time', fontsize=9)
        
        # 4. Frequency spectrum
        ax4 = plt.subplot(n_thresholds, 4, row_start + 4)
        plot_frequency_spectrum(magnitude[:, feature_idx], freq_mask, threshold, ax4, feature_name)
        if i == n_thresholds - 1:
            ax4.set_xlabel('Frequency Bin', fontsize=9)
    
    plt.tight_layout()
    
    # Save figure
    safe_name = feature_name.replace('/', '_').replace('\\', '_')
    output_path = output_dir / f'fft_decomposition_{safe_name}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_comparison_grid(samples, feature_names, threshold, output_dir):
    """
    Create comparison grid showing all features for one threshold
    
    Args:
        samples: Array of samples (n_samples, seq_len, n_features)
        feature_names: List of feature names
        threshold: Threshold percentile to use
        output_dir: Directory to save plots
    """
    n_samples, seq_len, n_features = samples.shape
    
    # Create figure with grid
    fig = plt.figure(figsize=(6 * n_features, 4 * n_samples))
    
    for sample_idx in range(n_samples):
        sample = samples[sample_idx]
        
        for feat_idx, feat_name in enumerate(feature_names):
            # Decompose
            high_freq, low_freq, freq_mask, magnitude = fft_decomposition(sample, threshold)
            
            original = sample[:, feat_idx]
            high_freq_signal = high_freq[:, feat_idx]
            low_freq_signal = low_freq[:, feat_idx]
            
            # Calculate energy
            high_energy = np.sum(high_freq_signal ** 2)
            low_energy = np.sum(low_freq_signal ** 2)
            total_energy = high_energy + low_energy
            high_energy_pct = (high_energy / total_energy * 100) if total_energy > 0 else 0
            
            # Plot in grid
            col = feat_idx + 1
            row = sample_idx + 1
            subplot_idx = (sample_idx * n_features) + feat_idx + 1
            
            ax = plt.subplot(n_samples, n_features, subplot_idx)
            
            # Plot all three components
            time = np.arange(seq_len)
            ax.plot(time, original, 'b-', linewidth=1, alpha=0.7, label='Original')
            ax.plot(time, low_freq_signal, 'g--', linewidth=1.5, label='Low Freq')
            ax.plot(time, high_freq_signal, 'r:', linewidth=1.5, label='High Freq')
            
            ax.set_title(f'Sample {sample_idx+1}: {feat_name}\nHigh Freq Energy: {high_energy_pct:.1f}%',
                        fontsize=9, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, loc='best')
            ax.tick_params(labelsize=7)
            
            if sample_idx == n_samples - 1:
                ax.set_xlabel('Time', fontsize=8)
            if feat_idx == 0:
                ax.set_ylabel('Value', fontsize=8)
    
    plt.suptitle(f'Frequency Decomposition Comparison (Threshold={threshold}%)',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save
    output_path = output_dir / f'comparison_grid_threshold_{threshold}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def create_summary_report(samples, feature_names, thresholds, output_dir):
    """Create summary report with statistics"""
    n_samples, seq_len, n_features = samples.shape
    
    # Create summary figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Energy distribution across thresholds
    ax1 = axes[0, 0]
    for feat_idx, feat_name in enumerate(feature_names):
        high_energies = []
        for threshold in thresholds:
            energies = []
            for sample in samples:
                high_freq, low_freq, _, _ = fft_decomposition(sample, threshold)
                high_energy = np.sum(high_freq[:, feat_idx] ** 2)
                total_energy = high_energy + np.sum(low_freq[:, feat_idx] ** 2)
                energies.append((high_energy / total_energy * 100) if total_energy > 0 else 0)
            high_energies.append(np.mean(energies))
        
        ax1.plot(thresholds, high_energies, 'o-', linewidth=2, label=feat_name, markersize=8)
    
    ax1.set_xlabel('Frequency Threshold (%)', fontsize=11)
    ax1.set_ylabel('High Freq Energy (%)', fontsize=11)
    ax1.set_title('High Frequency Energy vs Threshold', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=10)
    
    # 2. Frequency component ratio
    ax2 = axes[0, 1]
    freq_ratios = []
    for threshold in thresholds:
        ratios = []
        for sample in samples:
            _, _, freq_mask, _ = fft_decomposition(sample, threshold)
            ratios.append(freq_mask.sum() / len(freq_mask) * 100)
        freq_ratios.append(np.mean(ratios))
    
    ax2.plot(thresholds, freq_ratios, 'o-', linewidth=3, color='purple', markersize=10)
    ax2.fill_between(thresholds, 0, freq_ratios, alpha=0.3, color='purple')
    ax2.set_xlabel('Frequency Threshold (%)', fontsize=11)
    ax2.set_ylabel('High Freq Components (%)', fontsize=11)
    ax2.set_title('Frequency Components Classified as High', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=10)
    
    # 3. Variance explained by components
    ax3 = axes[1, 0]
    for feat_idx, feat_name in enumerate(feature_names):
        variance_ratios = []
        for threshold in thresholds:
            ratios = []
            for sample in samples:
                high_freq, low_freq, _, _ = fft_decomposition(sample, threshold)
                high_var = np.var(high_freq[:, feat_idx])
                total_var = np.var(sample[:, feat_idx])
                ratios.append((high_var / total_var * 100) if total_var > 0 else 0)
            variance_ratios.append(np.mean(ratios))
        
        ax3.plot(thresholds, variance_ratios, 's-', linewidth=2, label=feat_name, markersize=8)
    
    ax3.set_xlabel('Frequency Threshold (%)', fontsize=11)
    ax3.set_ylabel('High Freq Variance (%)', fontsize=11)
    ax3.set_title('Variance Explained by High Frequencies', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9, loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(labelsize=10)
    
    # 4. Recommendations text
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate recommendations
    recommended_threshold = None
    avg_energies = []
    for threshold in thresholds:
        energies = []
        for sample in samples:
            for feat_idx in range(n_features):
                high_freq, low_freq, _, _ = fft_decomposition(sample, threshold)
                high_energy = np.sum(high_freq[:, feat_idx] ** 2)
                total_energy = high_energy + np.sum(low_freq[:, feat_idx] ** 2)
                energies.append((high_energy / total_energy * 100) if total_energy > 0 else 0)
        avg_energies.append(np.mean(energies))
    
    # Find threshold where high freq contains 10-30% of energy (good balance)
    target_energy = 20  # Target 20% energy in high freq
    closest_idx = np.argmin(np.abs(np.array(avg_energies) - target_energy))
    recommended_threshold = thresholds[closest_idx]
    
    recommendations = f"""
THRESHOLD RECOMMENDATIONS

Dataset: {n_samples} samples, {seq_len} timesteps, {n_features} features

Recommended Threshold: {recommended_threshold}%
  • High freq energy: {avg_energies[closest_idx]:.1f}%
  • Balances shape preservation with trend fitting

Alternative Thresholds:

"""
    
    for i, threshold in enumerate(thresholds):
        recommendations += f"  {threshold}%: {avg_energies[i]:.1f}% high-freq energy"
        if threshold == recommended_threshold:
            recommendations += " ← RECOMMENDED"
        elif avg_energies[i] < 10:
            recommendations += " (may be too selective)"
        elif avg_energies[i] > 40:
            recommendations += " (may apply DILATE too broadly)"
        recommendations += "\n"
    
    recommendations += f"""
Guidelines:
  • 10-30% energy in high freq is ideal
  • Lower threshold = more aggressive DILATE
  • Higher threshold = more selective DILATE
  
Command to use:
  python run.py --loss_type fft_dilate \\
                --freq_threshold {recommended_threshold}
"""
    
    ax4.text(0.05, 0.95, recommendations, 
            transform=ax4.transAxes,
            fontsize=11,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / 'summary_report.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()
    
    return recommended_threshold, avg_energies[closest_idx]


def main():
    parser = argparse.ArgumentParser(description='FFT Threshold Finder - Find optimal frequency threshold')
    
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to dataset CSV file')
    parser.add_argument('--seq_len', type=int, default=96,
                       help='Sequence length to analyze (default: 96)')
    parser.add_argument('--num_samples', type=int, default=3,
                       help='Number of random samples to visualize (default: 3)')
    parser.add_argument('--thresholds', nargs='+', type=float, 
                       default=[50, 60, 70, 80, 85, 90, 95],
                       help='List of thresholds to test (default: 50 60 70 80 85 90 95)')
    parser.add_argument('--output_dir', type=str, default='visualizations/fft_threshold_analysis',
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("FFT THRESHOLD FINDER")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Thresholds to test: {args.thresholds}")
    print(f"Output directory: {output_dir}")
    print("="*70)
    print()
    
    # Load data
    samples, feature_names = load_data(args.dataset, args.seq_len, args.num_samples)
    n_samples, seq_len, n_features = samples.shape
    
    print(f"\nLoaded {n_samples} samples of shape ({seq_len}, {n_features})")
    print()
    
    # Create detailed visualizations for each feature
    print("Creating detailed decomposition visualizations...")
    for feat_idx, feat_name in enumerate(feature_names):
        print(f"  Processing feature {feat_idx+1}/{n_features}: {feat_name}")
        plot_decomposition(samples[0], feat_idx, feat_name, args.thresholds, output_dir)
    print()
    
    # Create comparison grids for each threshold
    print("Creating comparison grids...")
    for threshold in [70, 80, 90]:  # Show a few key thresholds
        print(f"  Threshold: {threshold}%")
        plot_comparison_grid(samples, feature_names, threshold, output_dir)
    print()
    
    # Create summary report
    print("Creating summary report...")
    recommended_threshold, rec_energy = create_summary_report(
        samples, feature_names, args.thresholds, output_dir
    )
    print()
    
    print("="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nRecommended Threshold: {recommended_threshold}%")
    print(f"  (Results in ~{rec_energy:.1f}% of energy in high frequencies)")
    print(f"\nVisualization files saved to: {output_dir}")
    print("\nKey files:")
    print(f"  • summary_report.png - Overall analysis and recommendations")
    for feat_name in feature_names[:3]:
        safe_name = feat_name.replace('/', '_').replace('\\', '_')
        print(f"  • fft_decomposition_{safe_name}.png - Detailed analysis")
    print(f"  • comparison_grid_threshold_*.png - Side-by-side comparisons")
    print()
    print("Next steps:")
    print(f"  1. Review visualizations in {output_dir}")
    print(f"  2. Use recommended threshold in training:")
    print(f"     python run.py --loss_type fft_dilate --freq_threshold {recommended_threshold}")
    print("="*70)


if __name__ == '__main__':
    main()
