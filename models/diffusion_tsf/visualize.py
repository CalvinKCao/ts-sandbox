"""
Visualization script for Diffusion TSF.

Loads a trained model and generates plots for evenly sampled windows across the dataset.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
from typing import Tuple, Optional

# Setup path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from config import DiffusionTSFConfig
from model import DiffusionTSF
from train_electricity import ElectricityDataset, MODEL_SIZES

def visualize_samples(
    model_path: str,
    data_path: str,
    num_samples: int = 5,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    # 1. Load checkpoint
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    config_dict = checkpoint['config']
    
    # 2. Reconstruct model
    model_config = DiffusionTSFConfig(
        lookback_length=512,
        forecast_length=96,
        image_height=128,
        unet_channels=MODEL_SIZES[config_dict['model_size']],
        num_res_blocks=2 if config_dict['model_size'] != 'large' else 3,
        attention_levels=[1, 2],
        num_diffusion_steps=config_dict['diffusion_steps'],
        noise_schedule=config_dict['noise_schedule']
    )
    
    model = DiffusionTSF(model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 3. Load Dataset (Last windows)
    dataset = ElectricityDataset(
        data_path,
        lookback=512,
        forecast=96,
        stride=96 # Use non-overlapping windows for test
    )
    
    # Evenly sample across the dataset for diverse visualizations
    total_samples = len(dataset)
    if total_samples <= num_samples:
        indices = list(range(total_samples))
    else:
        # Use linspace to get evenly spaced indices across the full dataset
        indices = np.linspace(0, total_samples - 1, num_samples, dtype=int).tolist()
    
    print(f"Generating {num_samples} visualizations...")
    
    os.makedirs('visualizations', exist_ok=True)
    
    for i, idx in enumerate(indices):
        past, future = dataset[idx]
        past_tensor = past.unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Generate prediction
            out = model.generate(past_tensor, use_ddim=True, num_ddim_steps=50, verbose=True)
            pred = out['prediction'].cpu().squeeze(0).numpy()
            
            # Extract 2D maps
            past_2d = out['past_2d'].cpu().squeeze(0).squeeze(0).numpy()
            future_2d = out['future_2d'].cpu().squeeze(0).squeeze(0).numpy()
        
        # Plotting
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [1, 1]})
        
        # 1. Plot 1D Time Series
        # Combine past and future for context
        time_past = np.arange(len(past))
        time_future = np.arange(len(past), len(past) + len(future))
        
        ax1.plot(time_past, past.numpy(), label='Past (Context)', color='gray', alpha=0.6)
        ax1.plot(time_future, future.numpy(), label='True Future', color='blue', linewidth=2)
        ax1.plot(time_future, pred, label='Diffusion Forecast', color='red', linestyle='--', linewidth=2)
        
        ax1.set_title(f"Diffusion TSF Forecast - Sample {i+1} (Dataset Index {idx})", fontsize=14)
        ax1.set_xlabel("Time Steps")
        ax1.set_ylabel("Value (Normalized)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Plot 2D Probability Map
        # Concatenate past and future 2D maps for full context
        full_2d = np.concatenate([past_2d, future_2d], axis=1)
        
        im = ax2.imshow(full_2d, aspect='auto', origin='lower', cmap='magma', interpolation='nearest')
        ax2.set_title("Diffusion Probability Map (2D Stripe Representation)", fontsize=14)
        ax2.set_xlabel("Time Steps")
        ax2.set_ylabel("Normalized Value Bins")
        
        # Add vertical line at the forecast start
        ax2.axvline(x=len(past), color='white', linestyle='-', linewidth=2, alpha=0.8, label='Forecast Start')
        
        # Add colorbar
        plt.colorbar(im, ax=ax2, label='Density')
        
        plt.tight_layout()
        
        save_path = f"visualizations/sample_{i+1}_full.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  Saved {save_path}")

def find_best_model(base_dir: str) -> Optional[str]:
    """Find the best model checkpoint in the directory structure."""
    # 1. Look for best_model.pt in subdirectories (study folders)
    study_dirs = [d for d in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(d)]
    # Sort by modification time, newest first
    study_dirs.sort(key=os.path.getmtime, reverse=True)
    
    # Also check the base directory itself for backward compatibility
    search_dirs = study_dirs + [base_dir]
    
    best_overall_model = None
    min_val_loss = float('inf')
    
    for d in search_dirs:
        # Priority 1: best_model.pt in this directory
        best_model_path = os.path.join(d, 'best_model.pt')
        if os.path.exists(best_model_path):
            print(f"Found best_model.pt in {d}")
            return best_model_path
            
        # Priority 2: trial_*_best.pt in this directory
        trial_checkpoints = glob.glob(os.path.join(d, "trial_*_best.pt"))
        for cp in trial_checkpoints:
            try:
                ckpt = torch.load(cp, map_location='cpu')
                if 'val_loss' in ckpt:
                    loss = ckpt['val_loss']
                    if loss < min_val_loss:
                        min_val_loss = loss
                        best_overall_model = cp
            except Exception:
                continue
                
    if best_overall_model:
        print(f"Found best trial checkpoint: {best_overall_model} (val_loss: {min_val_loss:.4f})")
    return best_overall_model

if __name__ == "__main__":
    # Setup paths
    BASE_CHECKPOINT_DIR = os.path.join(script_dir, "checkpoints")
    DATA = os.path.join(script_dir, "../../datasets/electricity/electricity.csv")
    
    best_model = find_best_model(BASE_CHECKPOINT_DIR)
    
    if best_model and os.path.exists(best_model):
        visualize_samples(best_model, DATA)
    else:
        print(f"Error: No suitable model checkpoint found in {BASE_CHECKPOINT_DIR}.")
        print("Run training first with: python train_electricity.py")

