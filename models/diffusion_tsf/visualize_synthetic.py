#!/usr/bin/env python3
"""
Visualize synthetic time series samples from RealTS generators.

This script generates and visualizes samples from each generator function
to help understand what the synthetic training data looks like.

Usage:
    python visualize_synthetic.py
    python visualize_synthetic.py --num-samples 5
    python visualize_synthetic.py --output-dir ./my_plots
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add parent to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from realts import RWB, PWB, LGB, TWDB, IFFTB, seasonal_periodicity, RealTS

# Styling
plt.style.use('seaborn-v0_8-darkgrid')
COLORS = {
    'RWB': '#E74C3C',      # Red
    'PWB': '#3498DB',      # Blue
    'LGB': '#27AE60',      # Green
    'TWDB': '#9B59B6',     # Purple
    'IFFTB': '#F39C12',    # Orange
    'seasonal': '#1ABC9C', # Teal
}


def plot_generator_samples(output_dir: str, num_samples: int = 3, length: int = 608):
    """Generate and plot samples from each generator function.
    
    Args:
        output_dir: Directory to save plots
        num_samples: Number of samples per generator
        length: Length of each time series (512 + 96 = 608)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    generators = [
        ("RWB", "Random Walk Behavior", RWB, 
         "Cumulative sum of random steps\nSimulates stock prices, sensor drift"),
        ("PWB", "Periodic Wave Behavior", PWB,
         "Superposition of 1-5 sine waves\nSimulates seasonal patterns"),
        ("LGB", "Logistic Growth Behavior", LGB,
         "S-curve (logistic function)\nSimulates adoption curves, saturation"),
        ("TWDB", "Trend + Wave Data Behavior", TWDB,
         "Linear trend + periodic waves\nSimulates trending seasonal data"),
        ("IFFTB", "Inverse FFT Behavior", IFFTB,
         "Synthetic spectrum with sparse peaks\nSimulates complex periodicities"),
        ("seasonal", "Seasonal Periodicity", seasonal_periodicity,
         "Multi-harmonic seasonal patterns\nSimulates daily/weekly/yearly cycles"),
    ]
    
    # Figure 1: All generators overview (one sample each)
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('Synthetic Time Series Generators - Overview', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    for ax, (key, name, gen, desc) in zip(axes, generators):
        # Generate sample
        np.random.seed(42)  # For reproducibility in overview
        data = gen(length)
        
        # Normalize for display
        data = (data - np.mean(data)) / (np.std(data) + 1e-8)
        
        # Split into past and future
        past = data[:512]
        future = data[512:]
        
        # Plot
        ax.plot(range(512), past, color=COLORS[key], linewidth=0.8, label='Past (512)')
        ax.plot(range(512, length), future, color=COLORS[key], linewidth=1.5, 
                linestyle='--', alpha=0.8, label='Future (96)')
        ax.axvline(x=512, color='gray', linestyle=':', alpha=0.7, label='Forecast start')
        
        ax.set_title(f'{name}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Time step')
        ax.set_ylabel('Normalized value')
        ax.text(0.02, 0.98, desc, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.legend(loc='lower right', fontsize=7)
        ax.set_xlim(0, length)
    
    plt.tight_layout()
    overview_path = os.path.join(output_dir, 'generators_overview.png')
    plt.savefig(overview_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {overview_path}")
    plt.close()
    
    # Figure 2: Multiple samples per generator
    for key, name, gen, desc in generators:
        fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3 * num_samples))
        fig.suptitle(f'{name} - {num_samples} Random Samples', fontsize=14, fontweight='bold')
        
        if num_samples == 1:
            axes = [axes]
        
        for i, ax in enumerate(axes):
            # Generate with different random seeds
            np.random.seed(i * 100 + 42)
            data = gen(length)
            
            # Normalize
            data = (data - np.mean(data)) / (np.std(data) + 1e-8)
            
            past = data[:512]
            future = data[512:]
            
            ax.plot(range(512), past, color=COLORS[key], linewidth=0.8)
            ax.plot(range(512, length), future, color=COLORS[key], linewidth=1.5, 
                    linestyle='--', alpha=0.8)
            ax.axvline(x=512, color='gray', linestyle=':', alpha=0.7)
            
            ax.set_ylabel(f'Sample {i+1}')
            ax.set_xlim(0, length)
            
            if i == num_samples - 1:
                ax.set_xlabel('Time step')
        
        plt.tight_layout()
        sample_path = os.path.join(output_dir, f'generator_{key}_samples.png')
        plt.savefig(sample_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {sample_path}")
        plt.close()


def plot_realts_dataset_samples(output_dir: str, num_samples: int = 6):
    """Generate and plot samples from the RealTS dataset class.
    
    This shows what the actual training data looks like, including
    the mixing probabilities from the ViTime paper.
    
    Args:
        output_dir: Directory to save plots
        num_samples: Number of samples to show
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset
    dataset = RealTS(
        num_samples=num_samples * 2,  # Generate extra for variety
        lookback_length=512,
        forecast_length=96,
        seed=42
    )
    
    # Plot samples
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('RealTS Dataset Samples (Mixed Generators)\n'
                 'IFFTB: 60%, PWB: 16%, RWB: 8%, LGB: 8%, TWDB: 8%', 
                 fontsize=14, fontweight='bold')
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i >= num_samples:
            ax.axis('off')
            continue
            
        past, future = dataset[i]
        past = past.numpy()
        future = future.numpy()
        
        # Plot
        ax.plot(range(512), past, color='#2C3E50', linewidth=0.8, label='Past')
        ax.plot(range(512, 608), future, color='#E74C3C', linewidth=1.2, 
                linestyle='--', label='Future')
        ax.axvline(x=512, color='gray', linestyle=':', alpha=0.7)
        
        ax.set_title(f'Sample {i+1}', fontsize=11)
        ax.set_xlabel('Time step')
        ax.set_ylabel('Normalized value')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlim(0, 608)
    
    plt.tight_layout()
    dataset_path = os.path.join(output_dir, 'realts_dataset_samples.png')
    plt.savefig(dataset_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {dataset_path}")
    plt.close()


def plot_comparison_with_real_data(output_dir: str):
    """Plot synthetic samples in a grid format similar to the visualization output.
    
    This creates a detailed view showing how synthetic data would appear
    during training.
    
    Args:
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    dataset = RealTS(num_samples=10, lookback_length=512, forecast_length=96, seed=123)
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.25)
    
    fig.suptitle('Synthetic Pre-training Data: RealTS Samples\n'
                 'These diverse patterns help the model learn general time series structure',
                 fontsize=14, fontweight='bold')
    
    sample_idx = 0
    for row in range(4):
        for col in range(3):
            if sample_idx >= 10:
                break
                
            ax = fig.add_subplot(gs[row, col])
            
            past, future = dataset[sample_idx]
            past = past.numpy()
            future = future.numpy()
            
            # Combine for full sequence
            full = np.concatenate([past, future])
            
            # Plot with gradient coloring
            ax.plot(range(512), past, color='#3498DB', linewidth=0.7, alpha=0.9)
            ax.plot(range(512, 608), future, color='#E74C3C', linewidth=1.0)
            ax.axvline(x=512, color='#95A5A6', linestyle='--', linewidth=1, alpha=0.8)
            
            # Add shading for future region
            ax.axvspan(512, 608, alpha=0.1, color='red')
            
            ax.set_title(f'Sample {sample_idx + 1}', fontsize=10)
            ax.tick_params(axis='both', which='major', labelsize=8)
            
            if col == 0:
                ax.set_ylabel('Value', fontsize=9)
            if row == 3:
                ax.set_xlabel('Time Step', fontsize=9)
            
            sample_idx += 1
    
    # Add legend in the empty spot
    ax_legend = fig.add_subplot(gs[3, 2])
    ax_legend.axis('off')
    ax_legend.text(0.5, 0.7, 'Legend:', fontsize=11, fontweight='bold',
                   ha='center', transform=ax_legend.transAxes)
    ax_legend.text(0.5, 0.5, '--- Past (512 steps)', fontsize=10, color='#3498DB',
                   ha='center', transform=ax_legend.transAxes)
    ax_legend.text(0.5, 0.3, '--- Future (96 steps)', fontsize=10, color='#E74C3C',
                   ha='center', transform=ax_legend.transAxes)
    ax_legend.text(0.5, 0.1, ' |  Forecast boundary', fontsize=10, color='#95A5A6',
                   ha='center', transform=ax_legend.transAxes)
    
    grid_path = os.path.join(output_dir, 'realts_training_grid.png')
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {grid_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize synthetic time series samples from RealTS generators'
    )
    parser.add_argument('--output-dir', type=str, default='synthetic_visualizations',
                        help='Directory to save plots (default: synthetic_visualizations)')
    parser.add_argument('--num-samples', type=int, default=3,
                        help='Number of samples per generator (default: 3)')
    args = parser.parse_args()
    
    output_dir = os.path.join(script_dir, args.output_dir)
    
    print("=" * 60)
    print("  SYNTHETIC TIME SERIES VISUALIZATION")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print(f"Samples per generator: {args.num_samples}")
    print()
    
    # Generate all plots
    print("Generating generator overview and samples...")
    plot_generator_samples(output_dir, num_samples=args.num_samples)
    
    print("\nGenerating RealTS dataset samples...")
    plot_realts_dataset_samples(output_dir, num_samples=6)
    
    print("\nGenerating training grid visualization...")
    plot_comparison_with_real_data(output_dir)
    
    print("\n" + "=" * 60)
    print("  VISUALIZATION COMPLETE")
    print("=" * 60)
    print(f"\nAll plots saved to: {output_dir}")
    print("\nGenerated files:")
    for f in sorted(os.listdir(output_dir)):
        if f.endswith('.png'):
            print(f"  - {f}")


if __name__ == "__main__":
    main()

