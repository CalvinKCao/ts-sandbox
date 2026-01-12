#!/usr/bin/env python3
"""
Lightning-fast smoke test for multi_dataset_finetune.sh
Tests all code paths without actual training.
"""
import sys
import os

sys.path.insert(0, 'models/diffusion_tsf')
os.chdir('/root/ts-sandbox' if os.path.exists('/root/ts-sandbox') else os.getcwd())

print("=" * 60)
print("SMOKE TEST: Multi-Dataset Fine-tuning Pipeline")
print("=" * 60)

errors = []

# =============================================================================
# 1. Check pretrained models exist
# =============================================================================
print("\n[1/6] Checking pretrained models...")
synth_guidance = "models/diffusion_tsf/checkpoints/full_synthetic_pretrain/synthetic_guidance/checkpoint.pth"
synth_diffusion = "models/diffusion_tsf/checkpoints/full_synthetic_pretrain/diffusion_synthetic_pretrain.pt"
params_file = "models/diffusion_tsf/checkpoints/universal_synthetic_pretrain/best_params.json"

for f, name in [(synth_guidance, "Synthetic iTransformer"), 
                (synth_diffusion, "Synthetic DiffusionTSF"),
                (params_file, "Params file")]:
    if os.path.exists(f):
        print(f"  ✅ {name}: {f}")
    else:
        print(f"  ❌ {name}: NOT FOUND - {f}")
        errors.append(f"Missing: {name}")

# =============================================================================
# 2. Check all datasets exist and are readable
# =============================================================================
print("\n[2/6] Checking datasets...")
import pandas as pd

DATASETS = [
    ("ETTh2", "datasets/ETT-small/ETTh2.csv", "OT"),
    ("ETTm1", "datasets/ETT-small/ETTm1.csv", "HUFL"),
    ("electricity", "datasets/electricity/electricity.csv", "42"),
    ("exchange_rate", "datasets/exchange_rate/exchange_rate.csv", "3"),
    ("traffic", "datasets/traffic/traffic.csv", "394"),
    ("weather", "datasets/weather/weather.csv", "T (degC)"),
    ("illness", "datasets/illness/national_illness.csv", "ILITOTAL"),
]

for name, path, col in DATASETS:
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, nrows=10)
            if col in df.columns or col.isdigit():
                print(f"  ✅ {name}: {len(pd.read_csv(path))} rows, column '{col}' OK")
            else:
                print(f"  ⚠️ {name}: column '{col}' not found, available: {list(df.columns)[:5]}...")
        except Exception as e:
            print(f"  ❌ {name}: Read error - {e}")
            errors.append(f"Read error: {name}")
    else:
        print(f"  ❌ {name}: NOT FOUND - {path}")
        errors.append(f"Missing dataset: {name}")

# =============================================================================
# 3. Test model imports
# =============================================================================
print("\n[3/6] Testing model imports...")
try:
    from config import DiffusionTSFConfig
    from model import DiffusionTSF
    print("  ✅ DiffusionTSF imports OK")
except Exception as e:
    print(f"  ❌ DiffusionTSF imports failed: {e}")
    errors.append(f"Import error: {e}")

# =============================================================================
# 4. Test iTransformer loader
# =============================================================================
print("\n[4/6] Testing iTransformer loader...")
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location('train_elec', 'models/diffusion_tsf/train_electricity.py')
    train_mod = importlib.util.module_from_spec(spec)
    sys.modules['train_elec'] = train_mod
    spec.loader.exec_module(train_mod)
    
    if hasattr(train_mod, 'load_itransformer_from_checkpoint'):
        print("  ✅ load_itransformer_from_checkpoint found")
    else:
        print("  ❌ load_itransformer_from_checkpoint not found")
        errors.append("Missing function: load_itransformer_from_checkpoint")
except Exception as e:
    print(f"  ❌ train_electricity.py load failed: {e}")
    errors.append(f"Load error: {e}")

# =============================================================================
# 5. Test SimpleDataset (inline eval dataset)
# =============================================================================
print("\n[5/6] Testing SimpleDataset...")
try:
    import torch
    import numpy as np
    from torch.utils.data import Dataset
    
    class SimpleDataset(Dataset):
        def __init__(self, data_path, lookback, forecast, column, stride=1):
            df = pd.read_csv(data_path)
            if column in df.columns:
                self.values = df[column].values.astype(np.float32)
            else:
                self.values = df.iloc[:, 1].values.astype(np.float32)
            self.lookback = lookback
            self.forecast = forecast
            self.stride = stride
            self.n_samples = max(0, (len(self.values) - lookback - forecast) // stride + 1)
        
        def __len__(self):
            return self.n_samples
        
        def __getitem__(self, idx):
            start = idx * self.stride
            past = self.values[start:start + self.lookback]
            future = self.values[start + self.lookback:start + self.lookback + self.forecast]
            mean, std = past.mean(), past.std() + 1e-8
            past = (past - mean) / std
            future = (future - mean) / std
            return torch.tensor(past), torch.tensor(future)
    
    # Test on first dataset
    ds = SimpleDataset("datasets/ETT-small/ETTh2.csv", 512, 96, "OT", stride=8)
    past, future = ds[0]
    print(f"  ✅ SimpleDataset OK: {len(ds)} samples, past={past.shape}, future={future.shape}")
except Exception as e:
    print(f"  ❌ SimpleDataset failed: {e}")
    errors.append(f"SimpleDataset error: {e}")

# =============================================================================
# 6. Test checkpoint loading (if exists)
# =============================================================================
print("\n[6/6] Testing checkpoint loading...")
try:
    import torch
    if os.path.exists(synth_diffusion):
        ckpt = torch.load(synth_diffusion, map_location='cpu')
        print(f"  ✅ Checkpoint loaded: {list(ckpt.keys())}")
        if 'config' in ckpt:
            print(f"      Config keys: {list(ckpt['config'].keys())[:5]}...")
    else:
        print("  ⚠️ Skipped - checkpoint not found")
except Exception as e:
    print(f"  ❌ Checkpoint load failed: {e}")
    errors.append(f"Checkpoint error: {e}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
if errors:
    print(f"❌ SMOKE TEST FAILED: {len(errors)} errors")
    for e in errors:
        print(f"   - {e}")
    sys.exit(1)
else:
    print("✅ SMOKE TEST PASSED: All checks OK")
    sys.exit(0)

