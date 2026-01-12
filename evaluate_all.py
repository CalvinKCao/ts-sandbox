import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Add paths
# Use relative path to find model files regardless of where the script is run
current_dir = os.path.dirname(os.path.abspath(__file__))
script_dir = os.path.join(current_dir, "models", "diffusion_tsf")
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from config import DiffusionTSFConfig
from model import DiffusionTSF
from train_electricity import ElectricityDataset, DATASET_REGISTRY, MODEL_SIZES
from metrics import compute_metrics, log_metrics
from visualize import create_guidance_for_visualization

# Parse arguments
parser = argparse.ArgumentParser(description="Evaluate DiffusionTSF models")
parser.add_argument('--dry-run', action='store_true', help="Quick test with 1 dataset, 3 samples, no real checkpoints needed")
parser.add_argument('--stride', type=int, default=8, help="Sample every N-th test point (default: 8 for speed, use 1 for full eval)")
parser.add_argument('--batch-size', type=int, default=32, help="Batch size for inference (default: 32)")
parser.add_argument('--ddim-steps', type=int, default=50, help="DDIM sampling steps (default: 50, lower=faster)")
args = parser.parse_args()

DRY_RUN = args.dry_run
EVAL_STRIDE = args.stride
BATCH_SIZE = args.batch_size
DDIM_STEPS = args.ddim_steps

datasets = {
    "ETTh2": "OT",
    "ETTm1": "HUFL",
    "electricity": "42",
    "exchange_rate": "3",
    "traffic": "394",
    "weather": "T (degC)",
    "illness": "ILITOTAL",
}

# In dry run, only test ETTh2
if DRY_RUN:
    print("🧪 DRY RUN MODE: Testing pipeline with minimal data")
    datasets = {"ETTh2": "MUFL"}

checkpoint_base = os.path.join(script_dir, "checkpoints", "multi_dataset_finetune")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
print(f"Settings: stride={EVAL_STRIDE}, batch_size={BATCH_SIZE}, ddim_steps={DDIM_STEPS}")

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
    
    use_dummy_model = False
    if not os.path.exists(model_path):
        if DRY_RUN:
            print(f"🧪 DRY RUN: Creating dummy model for {ds_var} (no checkpoint)")
            use_dummy_model = True
        else:
            print(f"Skipping {ds_var}, model not found at {model_path}")
            return None
        
    print(f"\n>>> Evaluating {ds_var} on {'dry run' if DRY_RUN else 'full'} test set...")
    
    # 1. Load checkpoint and reconstruct model
    if use_dummy_model:
        # Minimal config for dry run
        config_dict = {
            'model_size': 'small',
            'diffusion_steps': 100,  # Very few steps for speed
            'noise_schedule': 'linear',
            'num_variables': 1,
            'use_guidance_channel': False,
        }
        state_dict = None
    else:
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
    
    # Use tiny architecture for dry run
    if DRY_RUN and use_dummy_model:
        unet_channels = [32, 64]  # Tiny model
        diffusion_steps = 100
    else:
        unet_channels = MODEL_SIZES.get(model_size, [64, 128, 256])
        diffusion_steps = config_dict.get('diffusion_steps', 2000)
    
    model_config = DiffusionTSFConfig(
        lookback_length=512, forecast_length=96, image_height=128,
        max_scale=config_dict.get('max_scale', 3.5),
        blur_kernel_size=config_dict.get('blur_kernel_size', 31),
        blur_sigma=config_dict.get('blur_sigma', 1.0),
        emd_lambda=config_dict.get('emd_lambda', 0.0),
        representation_mode=config_dict.get('representation_mode', 'pdf'),
        unet_channels=unet_channels,
        num_res_blocks=num_res_blocks, attention_levels=attention_levels,
        num_diffusion_steps=diffusion_steps,
        noise_schedule=config_dict.get('noise_schedule', 'linear'),
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
    standalone_guidance_model = None
    if model_config.use_guidance_channel and os.path.exists(guidance_path):
        guidance_model = create_guidance_for_visualization(
            guidance_type=config_dict.get('guidance_type', 'itransformer'),
            guidance_checkpoint=guidance_path,
            seq_len=512, pred_len=96, num_variables=num_variables, device=device
        )
        model.set_guidance_model(guidance_model)
        # Also keep a standalone copy for baseline evaluation
        standalone_guidance_model = guidance_model
    
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # 2. Load test set
    data_rel_path = DATASET_REGISTRY[ds_name][0]
    data_path = os.path.join(current_dir, "datasets", data_rel_path)
    
    # Use stride=1 for evaluation to maximize sample count
    eval_stride = 1
    base_dataset = ElectricityDataset(data_path, lookback=512, forecast=96, stride=eval_stride, augment=False, use_all_columns=False, column=col)
    total_samples = len(base_dataset)
    
    if total_samples == 0:
        print(f"❌ Error: Dataset {ds_name} has 0 possible windows with lookback=512, forecast=96")
        return None

    # Use same gap logic as training: ceil(window_size / stride)
    window_size = 512 + 96
    gap_indices = (window_size + eval_stride - 1) // eval_stride
    
    # Split: 70% Train, 10% Val, 20% Test
    train_end = int(total_samples * 0.7)
    val_end = int(total_samples * 0.8)
    
    test_start = val_end + gap_indices
    
    # Fallback for small datasets: if gap is too large, reduce it to 1 step 
    # to at least get SOME test data (with warning)
    if test_start >= total_samples - 1:
        print(f"⚠️  Warning: Dataset {ds_name} too small for full gap. Reducing gap to 1.")
        test_start = min(val_end + 1, total_samples - 1)
        
    # Apply stride to subsample test set (use every N-th sample)
    all_test_indices = list(range(test_start, total_samples))
    
    if not all_test_indices:
        print(f"❌ Error: No test samples available for {ds_name} after splitting.")
        return None
    
    # Subsample with stride
    test_indices = all_test_indices[::EVAL_STRIDE]
    
    # In dry run, limit to 3 samples
    if DRY_RUN and len(test_indices) > 3:
        test_indices = test_indices[:3]
        print(f"   🧪 DRY RUN: Limiting to {len(test_indices)} samples")
        
    print(f"   Test set: {len(test_indices)} samples (from {len(all_test_indices)} total, stride={EVAL_STRIDE})")
    
    test_dataset = ElectricityDataset(
        data_path, lookback=512, forecast=96, stride=eval_stride, augment=False, 
        use_all_columns=False, data_tensor=base_dataset.data, indices=test_indices
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Predict and compute metrics
    all_preds = []
    all_targets = []
    
    # Fewer DDIM steps for dry run
    ddim_steps = 10 if DRY_RUN else DDIM_STEPS
    total_batches = len(test_loader)
    
    start_time = time.time()
    with torch.no_grad():
        for batch_idx, (past, future) in enumerate(test_loader):
            if batch_idx % 5 == 0 or DRY_RUN:
                elapsed = time.time() - start_time
                eta = (elapsed / (batch_idx + 1)) * (total_batches - batch_idx - 1) if batch_idx > 0 else 0
                print(f"   Batch {batch_idx + 1}/{total_batches} (ETA: {eta:.0f}s)")
            past = past.to(device)
            out = model.generate(past, use_ddim=True, num_ddim_steps=ddim_steps)
            all_preds.append(out['prediction'].cpu())
            all_targets.append(future)
            
    if not all_preds:
        print(f"❌ Error: No predictions generated for {ds_name}")
        return None

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    # Ensure shapes are (batch, forecast_len)
    if preds.ndim == 3: preds = preds.squeeze(1)
    if targets.ndim == 3: targets = targets.squeeze(1)

    diffusion_metrics = compute_metrics(preds, targets)

    # Also evaluate standalone iTransformer baseline if available
    itransformer_metrics = None
    if standalone_guidance_model is not None:
        print(f"   Evaluating iTransformer baseline...")
        guidance_preds = []

        with torch.no_grad():
            for past, future in test_loader:
                past = past.to(device)
                # Get guidance prediction directly (pass forecast_length=96)
                guidance_pred = standalone_guidance_model(past, forecast_length=96)
                guidance_preds.append(guidance_pred.cpu())

        guidance_preds = torch.cat(guidance_preds, dim=0)
        if guidance_preds.ndim == 3: guidance_preds = guidance_preds.squeeze(1)

        itransformer_metrics = compute_metrics(guidance_preds, targets)

    return diffusion_metrics, itransformer_metrics

results = {}
itransformer_results = {}

for ds_name, col in datasets.items():
    eval_result = evaluate_model(ds_name, col)
    if eval_result:
        diffusion_metrics, itransformer_metrics = eval_result
        results[ds_name] = diffusion_metrics

        print(f"Diffusion ({ds_name}): {log_metrics(diffusion_metrics)}")
        if itransformer_metrics:
            itransformer_results[ds_name] = itransformer_metrics
            print(f"iTransformer ({ds_name}): {log_metrics(itransformer_metrics)}")

            # Calculate improvement
            mse_improvement = ((itransformer_metrics['mse'] - diffusion_metrics['mse']) / itransformer_metrics['mse']) * 100
            mae_improvement = ((itransformer_metrics['mae'] - diffusion_metrics['mae']) / itransformer_metrics['mae']) * 100
            grad_mae_improvement = ((itransformer_metrics['gradient_mae'] - diffusion_metrics['gradient_mae']) / itransformer_metrics['gradient_mae']) * 100

            print(f"Improvement: MSE {mse_improvement:+.1f}% | MAE {mae_improvement:+.1f}% | GradMAE {grad_mae_improvement:+.1f}%")

print("\n" + "="*80)
print("FINAL TEST SET EVALUATION SUMMARY")
print("="*80)
print(f"{'Dataset':<15} | {'Model':<12} | {'MSE':<8} | {'MAE':<8} | {'GradMAE':<8} | {'GradCorr':<8}")
print("-" * 80)
for ds, diffusion_metrics in results.items():
    print(f"{ds:<15} | {'Diffusion':<12} | {diffusion_metrics['mse']:.4f} | {diffusion_metrics['mae']:.4f} | {diffusion_metrics['gradient_mae']:.4f} | {diffusion_metrics['gradient_correlation']:.4f}")

if itransformer_results:
    print("-" * 80)
    for ds, itransformer_metrics in itransformer_results.items():
        print(f"{ds:<15} | {'iTransformer':<12} | {itransformer_metrics['mse']:.4f} | {itransformer_metrics['mae']:.4f} | {itransformer_metrics['gradient_mae']:.4f} | {itransformer_metrics['gradient_correlation']:.4f}")

print("="*80)


