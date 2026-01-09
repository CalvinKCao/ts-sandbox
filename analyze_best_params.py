#!/usr/bin/env python3
"""
Script to analyze and plot distributions of best hyperparameters
from all best_params.json files found recursively in the project.
"""

import json
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_best_params():
    """Load all best_params.json files recursively"""
    params_list = []

    # Find all best_params.json files
    pattern = "**/best_params.json"
    param_files = glob.glob(pattern, recursive=True)

    print(f"Found {len(param_files)} best_params.json files")

    for file_path in param_files:
        try:
            with open(file_path, 'r') as f:
                params = json.load(f)
                params['source_file'] = file_path
                params['dataset'] = extract_dataset_name(file_path)
                params_list.append(params)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return pd.DataFrame(params_list)

def extract_dataset_name(file_path):
    """Extract dataset name from file path"""
    path_parts = Path(file_path).parts

    # Look for dataset names in path
    for part in path_parts:
        if any(dataset in part.lower() for dataset in ['etth1', 'etth2', 'ettm1', 'ettm2', 'electricity', 'exchange', 'illness']):
            return part.split('_')[1] if '_' in part else part

    return 'unknown'

def plot_continuous_distributions(df):
    """Plot distributions for continuous parameters"""
    continuous_params = ['learning_rate', 'blur_sigma', 'emd_lambda']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Distributions of Continuous Hyperparameters', fontsize=16)

    for i, param in enumerate(continuous_params):
        if param in df.columns:
            ax = axes[i]
            sns.histplot(data=df, x=param, ax=ax, kde=True)
            ax.set_title(f'{param.replace("_", " ").title()}')
            ax.set_xlabel(param.replace("_", " ").title())
            ax.set_ylabel('Count')

            # Add mean line
            mean_val = df[param].mean()
            ax.axvline(mean_val, color='red', linestyle='--', alpha=0.7,
                      label=f'Mean: {mean_val:.6f}')
            ax.legend()

    plt.tight_layout()
    plt.savefig('continuous_params_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_categorical_distributions(df):
    """Plot distributions for categorical parameters"""
    categorical_params = ['model_size', 'diffusion_steps', 'batch_size', 'noise_schedule', 'representation_mode']

    # Calculate grid dimensions
    n_params = len(categorical_params)
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
    fig.suptitle('Distributions of Categorical Hyperparameters', fontsize=16)

    axes = axes.flatten()

    for i, param in enumerate(categorical_params):
        if param in df.columns:
            ax = axes[i]
            value_counts = df[param].value_counts()

            # Create bar plot
            bars = ax.bar(range(len(value_counts)), value_counts.values)
            ax.set_xticks(range(len(value_counts)))
            ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
            ax.set_title(f'{param.replace("_", " ").title()}')
            ax.set_ylabel('Count')

            # Add value labels on bars
            for bar, count in zip(bars, value_counts.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{count}', ha='center', va='bottom')

    # Hide empty subplots
    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig('categorical_params_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_dataset_comparison(df):
    """Plot parameter distributions by dataset"""
    if 'dataset' not in df.columns or df['dataset'].nunique() <= 1:
        return

    # Plot key parameters by dataset
    key_params = ['learning_rate', 'model_size', 'batch_size', 'diffusion_steps']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Hyperparameter Distributions by Dataset', fontsize=16)

    axes = axes.flatten()

    for i, param in enumerate(key_params):
        ax = axes[i]

        if df[param].dtype in ['int64', 'float64']:
            # Box plot for continuous
            sns.boxplot(data=df, x='dataset', y=param, ax=ax)
            ax.set_title(f'{param.replace("_", " ").title()} by Dataset')
            ax.set_xlabel('Dataset')
            ax.set_ylabel(param.replace("_", " ").title())
            ax.tick_params(axis='x', rotation=45)
        else:
            # Count plot for categorical
            sns.countplot(data=df, x=param, hue='dataset', ax=ax)
            ax.set_title(f'{param.replace("_", " ").title()} by Dataset')
            ax.set_xlabel(param.replace("_", " ").title())
            ax.set_ylabel('Count')
            ax.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig('params_by_dataset.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_summary_stats(df):
    """Print summary statistics"""
    print("\n" + "="*50)
    print("HYPERPARAMETER ANALYSIS SUMMARY")
    print("="*50)

    print(f"\nTotal configurations analyzed: {len(df)}")

    print("\nCONTINUOUS PARAMETERS:")
    continuous_params = ['learning_rate', 'blur_sigma', 'emd_lambda']
    for param in continuous_params:
        if param in df.columns:
            values = df[param]
            print(f"{param}: Mean={values.mean():.6f}, Std={values.std():.6f}, Min={values.min():.6f}, Max={values.max():.6f}")

    print("\nCATEGORICAL PARAMETERS:")
    categorical_params = ['model_size', 'diffusion_steps', 'batch_size', 'noise_schedule', 'representation_mode', 'dataset']
    for param in categorical_params:
        if param in df.columns:
            value_counts = df[param].value_counts()
            print(f"\n{param}:")
            for val, count in value_counts.items():
                print(f"  {val}: {count} ({count/len(df)*100:.1f}%)")

def main():
    """Main analysis function"""
    print("Loading best parameters from all JSON files...")

    # Load all parameters
    df = load_best_params()

    if df.empty:
        print("No best_params.json files found!")
        return

    print(f"Loaded {len(df)} parameter configurations")

    # Print summary stats
    print_summary_stats(df)

    # Create plots
    print("\nCreating plots...")

    try:
        plot_continuous_distributions(df)
        print("✓ Created continuous parameter distributions plot")
    except Exception as e:
        print(f"✗ Error creating continuous plots: {e}")

    try:
        plot_categorical_distributions(df)
        print("✓ Created categorical parameter distributions plot")
    except Exception as e:
        print(f"✗ Error creating categorical plots: {e}")

    try:
        plot_dataset_comparison(df)
        print("✓ Created dataset comparison plot")
    except Exception as e:
        print(f"✗ Error creating dataset comparison plot: {e}")

    print("\nPlots saved as PNG files in current directory")
    print("- continuous_params_distributions.png")
    print("- categorical_params_distributions.png")
    print("- params_by_dataset.png")

if __name__ == "__main__":
    main()
