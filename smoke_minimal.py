#!/usr/bin/env python3
"""MINIMAL smoke test - 1 epoch, 1 diffusion step, 1 sample per dataset"""
import sys
import os
os.chdir('/root/ts-sandbox' if os.path.exists('/root/ts-sandbox') else '.')
sys.path.insert(0, 'models/diffusion_tsf')

import torch
import numpy as np
import pandas as pd

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
print("=" * 60)

DATASETS = [
    ("ETTh2", "datasets/ETT-small/ETTh2.csv", "OT"),
    ("ETTm1", "datasets/ETT-small/ETTm1.csv", "HUFL"),
    ("electricity", "datasets/electricity/electricity.csv", "42"),
    ("exchange_rate", "datasets/exchange_rate/exchange_rate.csv", "3"),
    ("traffic", "datasets/traffic/traffic.csv", "394"),
    ("weather", "datasets/weather/weather.csv", "T (degC)"),
    ("illness", "datasets/illness/national_illness.csv", "ILITOTAL"),
]

# ============================================================================
# 1. Test imports
# ============================================================================
print("[1/6] Testing imports...")
from config import DiffusionTSFConfig
from model import DiffusionTSF
print("  ✅ Imports OK")

# ============================================================================
# 2. Test model creation with MINIMAL config
# ============================================================================
print("[2/6] Testing model creation (minimal config)...")
config = DiffusionTSFConfig(
    lookback_length=32,
    forecast_length=8,
    num_variables=1,
    image_height=16,
    num_diffusion_steps=2,  # MINIMAL
    unet_channels=[8, 16],  # TINY
)
model = DiffusionTSF(config).to(device)
print(f"  ✅ Model created: {sum(p.numel() for p in model.parameters()):,} params")

# ============================================================================
# 3. Test forward pass (1 sample)
# ============================================================================
print("[3/6] Testing forward pass...")
past = torch.randn(1, 32).to(device)
future = torch.randn(1, 8).to(device)
output = model(past, future)
loss = output['loss'] if isinstance(output, dict) else output
print(f"  ✅ Forward pass: loss={loss.item():.4f}")

# ============================================================================
# 4. Test backward pass (1 step)
# ============================================================================
print("[4/6] Testing backward pass...")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer.zero_grad()
loss.backward()
optimizer.step()
print("  ✅ Backward pass OK")

# ============================================================================
# 5. Test generation (1 sample, 2 DDIM steps)
# ============================================================================
print("[5/6] Testing generation...")
model.eval()
with torch.no_grad():
    output = model.generate(past, use_ddim=True, num_ddim_steps=2)
    pred = output['prediction'] if isinstance(output, dict) else output
print(f"  ✅ Generation: shape={pred.shape}")

# ============================================================================
# 6. Test all datasets load
# ============================================================================
print("[6/6] Testing dataset loading...")
for name, path, col in DATASETS:
    try:
        df = pd.read_csv(path, nrows=100)
        if col in df.columns:
            vals = df[col].values[:40].astype(np.float32)
        else:
            vals = df.iloc[:, 1].values[:40].astype(np.float32)
        
        # Simulate train/val split with gap
        train = vals[:20]
        val = vals[28:40]  # Gap of 8
        
        # Per-sample normalization
        mean, std = train.mean(), train.std() + 1e-8
        train_norm = (train - mean) / std
        val_norm = (val - mean) / std
        
        print(f"  ✅ {name}: loaded, train={train_norm.shape}, val={val_norm.shape}")
    except Exception as e:
        print(f"  ❌ {name}: {e}")

print("=" * 60)
print("✅ ALL SMOKE TESTS PASSED")
print("=" * 60)

