import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Add paths
script_dir = "/home/cao/ts-sandbox/models/diffusion_tsf"
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from config import DiffusionTSFConfig
from model import DiffusionTSF
from train_electricity import ElectricityDataset, DATASET_REGISTRY, MODEL_SIZES
from metrics import compute_metrics, log_metrics
from visualize import create_guidance_for_visualization

datasets = {
    "ETTh2": "MUFL",
    "ETTm1": "HUFL",
    "illness": "AGE 0-4",
    "exchange_rate": "OT",
    "traffic": "394",
    "weather": "raining (s)"
}

checkpoint_base = "/home/cao/ts-sandbox/models/diffusion_tsf/checkpoints"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def sanitize_name(name):
    # This must match exactly how the shell script sanitized the names
    import re
    # Replace anything not alphanumeric, hyphen or underscore with underscore
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    # Replace multiple underscores with one
    sanitized = re.sub(r'__+', '_', sanitized)
    # Strip leading/trailing underscores
    sanitized = sanitized.strip('_')
    return sanitized

def evaluate_model(ds_name, col):
    sanitized_col = sanitize_name(col)
    ds_var = f"{ds_name}_{sanitized_col}"
    model_path = os.path.join(checkpoint_base, ds_var, "best_model.pt")
    
    if not os.path.exists(model_path):
        return None
        
    print(f"\n>>> Evaluating {ds_var} on full test set...")
    
    # 1. Load checkpoint and reconstruct model
    checkpoint = torch.load(model_path, map_location=device)
    config_dict = checkpoint['config']
    state_dict = checkpoint['model_state_dict']
    
    # Remap legacy unet.* keys to noise_predictor.*
    if any(k.startswith("unet.") for k in state_dict.keys()):
        state_dict = {"noise_predictor." + k[len("unet."):] if k.startswith("unet.") else k: v for k, v in state_dict.items()}
    
    model_size = config_dict.get('model_size', 'small')
    if model_size == 'large':
        attention_levels, num_res_blocks = [1, 2], 3
    elif model_size == 'medium':
        attention_levels, num_res_blocks = [1, 2, 3], 2
    else:
        attention_levels, num_res_blocks = [1, 2], 2
        
    num_variables = config_dict.get('num_variables', 1)
    
    model_config = DiffusionTSFConfig(
        lookback_length=512, forecast_length=96, image_height=128,
        max_scale=config_dict.get('max_scale', 3.5),
        blur_kernel_size=config_dict.get('blur_kernel_size', 31),
        blur_sigma=config_dict.get('blur_sigma', 1.0),
        emd_lambda=config_dict.get('emd_lambda', 0.0),
        representation_mode=config_dict.get('representation_mode', 'pdf'),
        unet_channels=MODEL_SIZES.get(model_size, [64, 128, 256]),
        num_res_blocks=num_res_blocks, attention_levels=attention_levels,
        num_diffusion_steps=config_dict['diffusion_steps'],
        noise_schedule=config_dict['noise_schedule'],
        use_coordinate_channel=config_dict.get('use_coordinate_channel', True),
        unet_kernel_size=config_dict.get('unet_kernel_size', (3, 9)),
        use_time_ramp=config_dict.get('use_time_ramp', True),
        use_time_sine=config_dict.get('use_time_sine', False),
        use_value_channel=config_dict.get('use_value_channel', True),
        seasonal_period=config_dict.get('seasonal_period', 24),
        num_variables=num_variables,
        use_hybrid_condition=config_dict.get('use_hybrid_condition', False),
        use_guidance_channel=config_dict.get('use_guidance_channel', False)
    )
    
    model = DiffusionTSF(model_config).to(device)
    
    # Load guidance if enabled
    guidance_path = os.path.join(checkpoint_base, ds_var, "guidance", "checkpoint.pth")
    if model_config.use_guidance_channel and os.path.exists(guidance_path):
        guidance_model = create_guidance_for_visualization(
            guidance_type=config_dict.get('guidance_type', 'itransformer'),
            guidance_checkpoint=guidance_path,
            seq_len=512, pred_len=96, num_variables=num_variables, device=device
        )
        model.set_guidance_model(guidance_model)
        
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # 2. Load test set
    data_rel_path = DATASET_REGISTRY[ds_name][0]
    data_path = os.path.join("/home/cao/ts-sandbox/datasets", data_rel_path)
    
    base_dataset = ElectricityDataset(data_path, lookback=512, forecast=96, augment=False, use_all_columns=False)
    total_samples = len(base_dataset)
    test_start = int(total_samples * 0.8) + (512 + 96 + 23) // 24 # Rough estimate of gap
    test_indices = list(range(test_start, total_samples))
    
    test_dataset = ElectricityDataset(
        data_path, lookback=512, forecast=96, augment=False, 
        use_all_columns=False, data_tensor=base_dataset.data, indices=test_indices
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 3. Predict and compute metrics
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for past, future in test_loader:
            past = past.to(device)
            out = model.generate(past, use_ddim=True, num_ddim_steps=50)
            all_preds.append(out['prediction'].cpu())
            all_targets.append(future)
            
    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    # Ensure shapes are (batch, forecast_len)
    if preds.ndim == 3: preds = preds.squeeze(1)
    if targets.ndim == 3: targets = targets.squeeze(1)
    
    metrics = compute_metrics(preds, targets)
    return metrics

results = {}
for ds_name, col in datasets.items():
    metrics = evaluate_model(ds_name, col)
    if metrics:
        results[ds_name] = metrics
        print(f"Results for {ds_name}: {log_metrics(metrics)}")

print("\n" + "="*60)
print("FINAL TEST SET EVALUATION SUMMARY")
print("="*60)
print(f"{'Dataset':<15} | {'MSE':<8} | {'MAE':<8} | {'GradMAE':<8} | {'GradCorr':<8}")
print("-" * 60)
for ds, m in results.items():
    print(f"{ds:<15} | {m['mse']:.4f} | {m['mae']:.4f} | {m['gradient_mae']:.4f} | {m['gradient_correlation']:.4f}")
print("="*60)


