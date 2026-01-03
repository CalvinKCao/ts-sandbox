#!/usr/bin/env python3
"""Visualize multiple trial checkpoints for comparison."""
import os
import sys
import glob

# Add script dir to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from visualize import visualize_samples

CKPT_DIR = os.path.join(script_dir, "checkpoints/diffusion_tsf_electricity_20260103_223916")
DATA_PATH = os.path.join(script_dir, "../../datasets/electricity/electricity.csv")

# Find all *_best.pt checkpoints
best_checkpoints = sorted(glob.glob(os.path.join(CKPT_DIR, "trial_*_best.pt")))

print(f"Found {len(best_checkpoints)} best checkpoints")

for ckpt_path in best_checkpoints:
    trial_name = os.path.basename(ckpt_path).replace(".pt", "")
    output_dir = os.path.join(script_dir, f"visualizations/{trial_name}")
    
    print(f"\n{'='*60}")
    print(f"Visualizing {trial_name}")
    print(f"{'='*60}")
    
    visualize_samples(
        model_path=ckpt_path,
        data_path=DATA_PATH,
        num_samples=2,
        output_dir=output_dir,
        decoder_method="mean"
    )

print("\n" + "="*60)
print("Done! Check visualizations/trial_*_best/sample_*.png")
print("="*60)

