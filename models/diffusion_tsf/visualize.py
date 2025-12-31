"""
Visualization script for Diffusion TSF.

Loads a trained model and generates plots for the last windows of the dataset.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple

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
    
    # Take the last num_samples
    total_samples = len(dataset)
    indices = range(total_samples - num_samples, total_samples)
    
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

if __name__ == "__main__":
    # Update paths as needed
    BEST_MODEL = "checkpoints/best_model.pt"
    DATA = "../../datasets/electricity/electricity.csv"
    
    if os.path.exists(BEST_MODEL):
        visualize_samples(BEST_MODEL, DATA)
    else:
        print(f"Error: Model not found at {BEST_MODEL}. Run training first.")

