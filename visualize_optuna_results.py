#!/usr/bin/env python3
"""
Visualize results from train_multi_dataset_optuna.sh Optuna tuning runs.

This script SPECIFICALLY targets checkpoints from the multi-dataset Optuna script:
- Looks for directories matching: diffusion_tsf_{ETTh2|ETTm1|illness|exchange_rate|traffic|weather}_*
- Handles missing/incomplete runs gracefully (some training may not have finished)
- Creates summary report and comparison plots

Usage:
    python visualize_optuna_results.py
    python visualize_optuna_results.py --checkpoint-dir models/diffusion_tsf/checkpoints
    python visualize_optuna_results.py --num-samples 3
    python visualize_optuna_results.py --dataset ETTh2  # Visualize only ETTh2
"""

import os
import sys
import glob
import json
import re
import argparse
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DIFFUSION_TSF_DIR = os.path.join(SCRIPT_DIR, 'models', 'diffusion_tsf')
DATASETS_DIR = os.path.join(SCRIPT_DIR, 'datasets')

# Add diffusion_tsf to path for imports
if DIFFUSION_TSF_DIR not in sys.path:
    sys.path.insert(0, DIFFUSION_TSF_DIR)

from config import DiffusionTSFConfig
from model import DiffusionTSF
from train_electricity import ElectricityDataset, MODEL_SIZES, DATASET_REGISTRY
from guidance import iTransformerGuidance, LinearRegressionGuidance
import importlib.util

# Datasets from train_multi_dataset_optuna.sh
OPTUNA_DATASETS = ['ETTh2', 'ETTm1', 'illness', 'exchange_rate', 'traffic', 'weather']

# Regex pattern to match Optuna study directories from train_multi_dataset_optuna.sh
# Pattern: diffusion_tsf_{dataset}_{timestamp} or diffusion_tsf_{dataset}
OPTUNA_DIR_PATTERN = re.compile(
    r'^diffusion_tsf_(ETTh2|ETTm1|illness|exchange_rate|traffic|weather)(?:_\d{8}_\d{6})?$'
)


def load_itransformer_guidance(
    checkpoint_path: str,
    seq_len: int = 512,
    pred_len: int = 96,
    num_variables: int = 1,
    device: str = 'cpu'
) -> iTransformerGuidance:
    """Load a pre-trained iTransformer model as guidance.
    
    Args:
        checkpoint_path: Path to iTransformer checkpoint (.pth file)
        seq_len: Input sequence length
        pred_len: Prediction length
        num_variables: Number of variables in the dataset
        device: Device to load model on
        
    Returns:
        iTransformerGuidance wrapper around the loaded model
    """
    # Use importlib to load from absolute path to avoid conflicts
    itrans_model_path = os.path.join(DIFFUSION_TSF_DIR, '..', 'iTransformer', 'model', 'iTransformer.py')
    itrans_model_path = os.path.abspath(itrans_model_path)
    
    # Add iTransformer to path for its internal imports
    itrans_dir = os.path.join(DIFFUSION_TSF_DIR, '..', 'iTransformer')
    itrans_dir = os.path.abspath(itrans_dir)
    if itrans_dir not in sys.path:
        sys.path.insert(0, itrans_dir)
    
    # Load the module using spec
    spec = importlib.util.spec_from_file_location("iTransformer_module", itrans_model_path)
    itrans_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(itrans_module)
    iTransformerModel = itrans_module.Model
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Try to extract config from checkpoint
    if 'config' in checkpoint:
        ckpt_config = checkpoint['config']
    else:
        ckpt_config = {}
    
    # Get the state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Auto-detect e_layers from state_dict
    detected_e_layers = 0
    for key in state_dict.keys():
        if key.startswith('encoder.attn_layers.'):
            layer_idx = int(key.split('.')[2])
            detected_e_layers = max(detected_e_layers, layer_idx + 1)
    
    # Auto-detect d_model from embedding weight shape
    detected_d_model = 512
    if 'enc_embedding.value_embedding.weight' in state_dict:
        detected_d_model = state_dict['enc_embedding.value_embedding.weight'].shape[0]
    
    # Create config object for iTransformer
    class iTransConfig:
        def __init__(self):
            self.seq_len = ckpt_config.get('seq_len', seq_len)
            self.pred_len = ckpt_config.get('pred_len', pred_len)
            self.output_attention = False
            self.use_norm = True
            self.d_model = ckpt_config.get('d_model', detected_d_model)
            self.embed = 'fixed'
            self.freq = 'h'
            self.dropout = 0.1
            self.factor = 1
            self.n_heads = ckpt_config.get('n_heads', 8)
            self.d_ff = ckpt_config.get('d_ff', 2048)
            self.activation = 'gelu'
            self.e_layers = ckpt_config.get('e_layers', detected_e_layers if detected_e_layers > 0 else 4)
            self.class_strategy = 'projection'
            self.enc_in = num_variables
    
    config = iTransConfig()
    
    # Create and load model
    model = iTransformerModel(config)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    # Wrap in guidance interface
    guidance = iTransformerGuidance(
        model=model,
        use_norm=config.use_norm,
        seq_len=config.seq_len,
        pred_len=config.pred_len
    )
    
    return guidance


def find_itransformer_checkpoint(dataset: str, target_column: str) -> Optional[str]:
    """Find the iTransformer checkpoint for a dataset/target combination.
    
    The train_multi_dataset_optuna.sh saves iTransformer checkpoints at:
        checkpoints/itransformer_optuna_{dataset}_{target}/.../{model_id}/checkpoint.pth
    
    Args:
        dataset: Dataset name (e.g., 'ETTh2')
        target_column: Target column name
        
    Returns:
        Path to checkpoint.pth or None if not found
    """
    base_dir = os.path.join(SCRIPT_DIR, 'checkpoints')
    
    if not os.path.exists(base_dir):
        return None
    
    # Pattern: itransformer_optuna_{dataset}_{target}
    # The target column might have special characters, so we search flexibly
    patterns = [
        f'itransformer_optuna_{dataset}_{target_column}',
        f'itransformer_optuna_{dataset}_*',
        f'itransformer*{dataset}*',
    ]
    
    for pattern in patterns:
        matches = glob.glob(os.path.join(base_dir, pattern))
        for match in matches:
            if not os.path.isdir(match):
                continue
            # Look for checkpoint.pth in subdirectories
            ckpt_files = glob.glob(os.path.join(match, '**', 'checkpoint.pth'), recursive=True)
            if ckpt_files:
                # Return the most recent one
                ckpt_files.sort(key=os.path.getmtime, reverse=True)
                return ckpt_files[0]
    
    return None


def find_optuna_checkpoints(base_dir: str, dataset_filter: Optional[str] = None) -> Dict[str, List[dict]]:
    """Find checkpoints from train_multi_dataset_optuna.sh runs ONLY.
    
    This specifically looks for directories matching the pattern:
        diffusion_tsf_{ETTh2|ETTm1|illness|exchange_rate|traffic|weather}[_timestamp]
    
    Args:
        base_dir: Base checkpoint directory (models/diffusion_tsf/checkpoints)
        dataset_filter: If set, only find checkpoints for this dataset
        
    Returns:
        Dict mapping dataset name to list of checkpoint info dicts
    """
    checkpoints = {}
    
    if not os.path.exists(base_dir):
        print(f"⚠️  Checkpoint directory not found: {base_dir}")
        return checkpoints
    
    print(f"🔍 Scanning for Optuna study directories in: {base_dir}")
    
    # Helper to register a checkpoint entry
    def _add_checkpoint(dataset_name: str, ckpt_path: str, item_dir: str, item_name: str):
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            val_loss = ckpt.get('val_loss', None)
            epoch = ckpt.get('epoch', None)
            config = ckpt.get('config', {})
        except Exception as e:
            print(f"      ⚠️  Failed to load {ckpt_path}: {e}")
            return
        
        target_column = config.get('target_column', config.get('target', 'unknown'))
        info = {
            'path': ckpt_path,
            'dir': item_dir,
            'name': item_name,
            'val_loss': val_loss,
            'epoch': epoch,
            'config': config,
            'dataset': dataset_name,
            'target_column': target_column,
        }
        checkpoints.setdefault(dataset_name, []).append(info)
        print(f"      ✓ {os.path.basename(ckpt_path)}, val_loss={val_loss}, target={target_column}")
    
    # Detect direct checkpoint files when base_dir itself is the dataset directory
    direct_candidates = []
    for fname in ["best_model.pt", "model_best.pt"]:
        fpath = os.path.join(base_dir, fname)
        if os.path.exists(fpath):
            direct_candidates.append(fpath)
    direct_candidates.extend(sorted(glob.glob(os.path.join(base_dir, "trial_*_best.pt"))))
    
    if direct_candidates:
        base_name = os.path.basename(os.path.normpath(base_dir))
        # Try to extract dataset from base name (e.g., diffusion_tsf_ETTh2 -> ETTh2)
        m = re.match(r"diffusion_tsf_([^/]+)", base_name)
        dataset_name = m.group(1) if m else base_name
        if dataset_filter and dataset_name.lower() != dataset_filter.lower():
            pass  # ignore if filtered out
        else:
            print(f"   Found direct checkpoints in base_dir -> dataset={dataset_name}")
            best_loss = float('inf')
            best_path = None
            for cand in direct_candidates:
                try:
                    ckpt = torch.load(cand, map_location='cpu')
                    loss = ckpt.get('val_loss', float('inf'))
                except Exception:
                    continue
                if loss < best_loss:
                    best_loss = loss
                    best_path = cand
            if best_path is None:
                best_path = direct_candidates[0]
            _add_checkpoint(dataset_name, best_path, base_dir, base_name)
    
    # Look for study directories matching our pattern
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if not os.path.isdir(item_path):
            continue
        
        # Check if this matches the Optuna pattern from train_multi_dataset_optuna.sh
        match = OPTUNA_DIR_PATTERN.match(item)
        if not match:
            # Skip non-matching directories (e.g., electricity runs, other experiments)
            continue
        
        # Extract dataset name from the match
        dataset_name = match.group(1)
        
        # Apply filter if specified
        if dataset_filter and dataset_name.lower() != dataset_filter.lower():
            continue
        
        print(f"   Found: {item} -> dataset={dataset_name}")
        
        # Find best checkpoint in this directory (priority order)
        # 1. best_model.pt (overall best from Optuna)
        # 2. model_best.pt (alternative naming)
        # 3. trial_*_best.pt (individual trial bests)
        checkpoint_path = None
        
        best_model_path = os.path.join(item_path, 'best_model.pt')
        model_best_path = os.path.join(item_path, 'model_best.pt')
        
        if os.path.exists(best_model_path):
            checkpoint_path = best_model_path
        elif os.path.exists(model_best_path):
            checkpoint_path = model_best_path
        else:
            # Try to find trial_*_best.pt and pick the one with lowest val_loss
            trial_checkpoints = glob.glob(os.path.join(item_path, 'trial_*_best.pt'))
            if trial_checkpoints:
                best_loss = float('inf')
                for tc in trial_checkpoints:
                    try:
                        ckpt = torch.load(tc, map_location='cpu')
                        loss = ckpt.get('val_loss', float('inf'))
                        if loss < best_loss:
                            best_loss = loss
                            checkpoint_path = tc
                    except Exception:
                        pass
        
        if checkpoint_path is None:
            print(f"      ⚠️  No checkpoint files found in {item}")
            continue
        
        _add_checkpoint(dataset_name, checkpoint_path, item_path, item)
    
    # Sort each dataset's checkpoints by val_loss (best first)
    for ds in checkpoints:
        checkpoints[ds].sort(key=lambda x: x['val_loss'] if x['val_loss'] is not None else float('inf'))
    
    return checkpoints


def find_guidance_checkpoint(dataset: str, target_column: Optional[str] = None) -> Optional[str]:
    """Find iTransformer guidance checkpoint for a dataset.
    
    Args:
        dataset: Dataset name
        target_column: Optional target column name
        
    Returns:
        Path to guidance checkpoint or None
    """
    guidance_base = os.path.join(SCRIPT_DIR, 'checkpoints')
    
    if not os.path.exists(guidance_base):
        return None
    
    # Look for matching guidance directories
    patterns = [
        f'itransformer_optuna_{dataset}*',
        f'itransformer_guidance*{dataset}*',
        f'itransformer_{dataset}*',
    ]
    
    for pattern in patterns:
        matches = glob.glob(os.path.join(guidance_base, pattern))
        for match in matches:
            ckpt_path = os.path.join(match, 'checkpoint.pth')
            if os.path.exists(ckpt_path):
                return ckpt_path
            # Also check subdirectories
            sub_ckpts = glob.glob(os.path.join(match, '*', 'checkpoint.pth'))
            if sub_ckpts:
                return sub_ckpts[0]
    
    return None


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: str = 'cpu',
    dataset: str = None,
    target_column: str = None
) -> Tuple[DiffusionTSF, dict, bool]:
    """Load a trained model from checkpoint, optionally with guidance.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        dataset: Dataset name (for finding guidance checkpoint)
        target_column: Target column name (for finding guidance checkpoint)
        
    Returns:
        Tuple of (model, config_dict, guidance_loaded)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config_dict = checkpoint['config']
    state_dict = checkpoint['model_state_dict']
    
    # Remap legacy "unet.*" keys to "noise_predictor.*"
    if any(k.startswith("unet.") for k in state_dict.keys()):
        remapped = {}
        for k, v in state_dict.items():
            if k.startswith("unet."):
                remapped["noise_predictor." + k[len("unet."):]] = v
            else:
                remapped[k] = v
        state_dict = remapped
    
    # Get model configuration
    model_type = config_dict.get('model_type', 'unet')
    model_size = config_dict.get('model_size', 'small')
    num_variables = config_dict.get('num_variables', 1)
    
    # Determine attention and res blocks
    if model_size == 'large':
        attention_levels = [1, 2]
        num_res_blocks = 3
    elif model_size == 'medium':
        attention_levels = [1, 2, 3]
        num_res_blocks = 2
    else:
        attention_levels = [1, 2]
        num_res_blocks = 2
    
    # Auto-detect settings from state_dict
    init_key = 'noise_predictor.init_conv.weight'
    if init_key in state_dict:
        init_weight_shape = state_dict[init_key].shape
        unet_kernel_size = (init_weight_shape[2], init_weight_shape[3])
    else:
        unet_kernel_size = config_dict.get('unet_kernel_size', (3, 3))
    
    # Detect conditioning mode
    cond_encoder_key = 'noise_predictor.cond_encoder.local_encoder.0.weight'
    has_cond_encoder = cond_encoder_key in state_dict
    conditioning_mode = config_dict.get('conditioning_mode', 
                                         'vector_embedding' if has_cond_encoder else 'visual_concat')
    
    # Detect hybrid conditioning
    context_encoder_key = 'context_encoder.time_embed.weight'
    has_context_encoder = context_encoder_key in state_dict
    use_hybrid_condition = config_dict.get('use_hybrid_condition', has_context_encoder)
    
    # Check if model was trained with guidance
    use_guidance_channel = config_dict.get('use_guidance_channel', False)
    guidance_type = config_dict.get('guidance_type', None)
    
    # Build config
    model_config = DiffusionTSFConfig(
        lookback_length=512,
        forecast_length=96,
        image_height=128,
        max_scale=config_dict.get('max_scale', 3.5),
        blur_kernel_size=config_dict.get('blur_kernel_size', 31),
        blur_sigma=config_dict.get('blur_sigma', 1.0),
        emd_lambda=config_dict.get('emd_lambda', 0.0),
        representation_mode=config_dict.get('representation_mode', 'cdf'),
        unet_channels=MODEL_SIZES.get(model_size, [64, 128, 256]),
        num_res_blocks=num_res_blocks,
        attention_levels=attention_levels,
        num_diffusion_steps=config_dict.get('diffusion_steps', 500),
        noise_schedule=config_dict.get('noise_schedule', 'cosine'),
        model_type=model_type,
        use_coordinate_channel=config_dict.get('use_coordinate_channel', True),
        unet_kernel_size=unet_kernel_size,
        use_time_ramp=config_dict.get('use_time_ramp', False),
        use_time_sine=config_dict.get('use_time_sine', False),
        use_value_channel=config_dict.get('use_value_channel', False),
        seasonal_period=config_dict.get('seasonal_period', 96),
        num_variables=num_variables,
        conditioning_mode=conditioning_mode,
        use_hybrid_condition=use_hybrid_condition,
        use_guidance_channel=use_guidance_channel,
    )
    
    # Filter out guidance_model.* keys (we'll load guidance separately)
    guidance_keys = [k for k in state_dict.keys() if k.startswith('guidance_model.')]
    for k in guidance_keys:
        del state_dict[k]
    
    model = DiffusionTSF(model_config).to(device)
    
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(f"  ⚠️  Strict loading failed, trying non-strict: {e}")
        model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    
    # Try to load guidance model if the diffusion model was trained with guidance
    guidance_loaded = False
    if use_guidance_channel and guidance_type == 'itransformer' and dataset and target_column:
        itrans_ckpt = find_itransformer_checkpoint(dataset, target_column)
        
        if itrans_ckpt:
            try:
                print(f"  📦 Loading iTransformer guidance from: {itrans_ckpt}")
                guidance_model = load_itransformer_guidance(
                    checkpoint_path=itrans_ckpt,
                    seq_len=512,
                    pred_len=96,
                    num_variables=num_variables,
                    device=device
                )
                model.set_guidance_model(guidance_model)
                guidance_loaded = True
                print(f"  ✅ Guidance model loaded!")
            except Exception as e:
                print(f"  ⚠️  Failed to load guidance model: {e}")
        else:
            print(f"  ⚠️  No iTransformer checkpoint found for {dataset}/{target_column}")
    
    return model, config_dict, guidance_loaded


def visualize_checkpoint(
    checkpoint_info: dict,
    output_base_dir: str,
    num_samples: int = 5,
    device: str = 'cpu',
    decoder_method: str = 'mean'
) -> dict:
    """Visualize a single checkpoint from train_multi_dataset_optuna.sh.
    
    Args:
        checkpoint_info: Checkpoint info dict from find_optuna_checkpoints
        output_base_dir: Base directory for output
        num_samples: Number of samples to visualize
        device: Device to use
        
    Returns:
        Results dict with metrics
    """
    dataset_name = checkpoint_info['dataset']
    checkpoint_path = checkpoint_info['path']
    config_dict = checkpoint_info['config']
    target_column = checkpoint_info.get('target_column', 'unknown')
    
    print(f"\n{'='*60}")
    print(f"Visualizing: {dataset_name} (target: {target_column})")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*60}")
    
    # Create output directory
    output_dir = os.path.join(output_base_dir, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'dataset': dataset_name,
        'target_column': target_column,
        'checkpoint': checkpoint_path,
        'val_loss': checkpoint_info['val_loss'],
        'success': False,
        'error': None,
        'metrics': {}
    }
    
    try:
        # Load model (with guidance if available)
        model, config, guidance_loaded = load_model_from_checkpoint(
            checkpoint_path, device, 
            dataset=dataset_name, 
            target_column=target_column
        )
        
        # Get dataset info
        if dataset_name in DATASET_REGISTRY:
            dataset_info = DATASET_REGISTRY[dataset_name]
            data_path = os.path.join(DATASETS_DIR, dataset_info[0])
        else:
            print(f"  ⚠️  Dataset {dataset_name} not in registry, skipping")
            results['error'] = f"Dataset {dataset_name} not in registry"
            return results
        
        if not os.path.exists(data_path):
            print(f"  ⚠️  Data file not found: {data_path}")
            results['error'] = f"Data file not found: {data_path}"
            return results
        
        print(f"  Data path: {data_path}")
        print(f"  Config: {config.get('representation_mode', 'cdf')} mode, "
              f"model_size={config.get('model_size', 'small')}, "
              f"decoder={decoder_method}")
        print(f"  Guidance: {'✅ Loaded' if guidance_loaded else '❌ Not available'}")
        
        # Load dataset
        num_variables = config.get('num_variables', 1)
        use_all_columns = config.get('use_all_columns', False) or num_variables > 1
        
        base_dataset = ElectricityDataset(
            data_path,
            lookback=512,
            forecast=96,
            augment=False,
            use_all_columns=use_all_columns
        )
        
        # Use chronological test set
        total_samples = len(base_dataset)
        window_size = 512 + 96
        stride = 24
        gap_indices = (window_size + stride - 1) // stride
        
        raw_train_end = int(total_samples * 0.7)
        raw_val_end = int(total_samples * 0.8)
        test_start = raw_val_end + gap_indices
        
        if test_start >= total_samples:
            test_start = raw_val_end + 1
        
        eval_indices = list(range(test_start, total_samples))
        
        if len(eval_indices) == 0:
            print(f"  ⚠️  No test samples available for {dataset_name}")
            results['error'] = "No test samples available"
            return results
        
        eval_dataset = ElectricityDataset(
            data_path,
            lookback=512,
            forecast=96,
            augment=False,
            use_all_columns=use_all_columns,
            data_tensor=base_dataset.data,
            indices=eval_indices
        )
        
        print(f"  Test set: {len(eval_indices)} samples (indices {test_start}-{total_samples-1})")
        
        # Select samples evenly spaced
        n_eval = len(eval_dataset)
        if n_eval <= num_samples:
            indices = list(range(n_eval))
        else:
            indices = np.linspace(0, n_eval - 1, num_samples, dtype=int).tolist()
        
        # Track metrics
        diffusion_maes = []
        diffusion_rmses = []
        guidance_maes = []
        guidance_rmses = []
        
        for i, idx in enumerate(indices):
            past, future = eval_dataset[idx]
            past_tensor = past.unsqueeze(0).to(device)
            
            with torch.no_grad():
                out = model.generate(
                    past_tensor,
                    use_ddim=True,
                    num_ddim_steps=50,
                    verbose=False,
                    decoder_method=decoder_method
                )
                pred = out['prediction'].cpu().squeeze(0).numpy()
                
                # Extract guidance prediction if available
                guidance_pred = None
                if 'guidance_1d' in out:
                    guidance_pred = out['guidance_1d'].cpu().squeeze(0).numpy()
            
            future_np = future.numpy()
            
            # Calculate diffusion metrics
            mae = np.mean(np.abs(pred - future_np))
            rmse = np.sqrt(np.mean((pred - future_np) ** 2))
            diffusion_maes.append(mae)
            diffusion_rmses.append(rmse)
            
            # Calculate guidance metrics if available
            guidance_mae = None
            guidance_rmse = None
            if guidance_pred is not None:
                guidance_mae = np.mean(np.abs(guidance_pred - future_np))
                guidance_rmse = np.sqrt(np.mean((guidance_pred - future_np) ** 2))
                guidance_maes.append(guidance_mae)
                guidance_rmses.append(guidance_rmse)
            
            # Handle multivariate - plot first variable
            if pred.ndim == 2:
                pred_1d = pred[0]
                future_1d = future_np[0]
                past_1d = past.numpy()[0]
                guidance_1d = guidance_pred[0] if guidance_pred is not None and guidance_pred.ndim == 2 else guidance_pred
            else:
                pred_1d = pred
                future_1d = future_np
                past_1d = past.numpy()
                guidance_1d = guidance_pred
            
            # Plot with guidance if available
            has_guidance = guidance_1d is not None
            
            if has_guidance:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            else:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            
            # 1D time series
            time_past = np.arange(len(past_1d))
            time_future = np.arange(len(past_1d), len(past_1d) + len(future_1d))
            
            ax1.plot(time_past[-96:], past_1d[-96:], label='Past (last 96)', color='gray', alpha=0.6)
            ax1.plot(time_future, future_1d, label='True Future', color='blue', linewidth=2)
            ax1.plot(time_future, pred_1d, label=f'Diffusion (MAE={mae:.3f})', color='red', 
                    linestyle='--', linewidth=2)
            
            # Add guidance prediction if available
            if has_guidance:
                ax1.plot(time_future, guidance_1d, label=f'iTransformer (MAE={guidance_mae:.3f})', 
                        color='green', linestyle=':', linewidth=2, alpha=0.8)
            
            title = f'{dataset_name} - Sample {i+1} | Target: {target_column}'
            if has_guidance:
                improvement = (guidance_mae - mae) / guidance_mae * 100 if guidance_mae > 0 else 0
                title += f' | Diff vs iTrans: {improvement:+.1f}%'
            ax1.set_title(title)
            ax1.set_xlabel('Time Steps')
            ax1.set_ylabel('Value')
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # 2D representation
            past_2d_raw = out['past_2d'].cpu().squeeze(0).numpy()
            future_2d_raw = out['future_2d'].cpu().squeeze(0).numpy()
            
            if past_2d_raw.ndim == 3:
                past_2d_raw = past_2d_raw[0]
                future_2d_raw = future_2d_raw[0]
            
            past_2d = np.clip((past_2d_raw + 1.0) / 2.0, 0.0, 1.0)
            future_2d = np.clip((future_2d_raw + 1.0) / 2.0, 0.0, 1.0)
            full_2d = np.concatenate([past_2d, future_2d], axis=1)
            
            im = ax2.imshow(full_2d, aspect='auto', origin='lower', cmap='magma',
                           interpolation='nearest', vmin=0.0, vmax=1.0)
            ax2.axvline(x=past_2d.shape[1], color='white', linestyle='-', linewidth=2, alpha=0.8)
            mode_label = "PDF" if config.get('representation_mode', 'cdf') == 'pdf' else "CDF"
            ax2.set_title(f'2D Representation ({mode_label} mode)')
            ax2.set_xlabel('Time Steps')
            ax2.set_ylabel('Value Bins')
            plt.colorbar(im, ax=ax2, label='Density')
            
            plt.tight_layout()
            save_path = os.path.join(output_dir, f'sample_{i+1}.png')
            plt.savefig(save_path, dpi=150)
            plt.close()
            
            # Log sample metrics
            if has_guidance:
                print(f"    Sample {i+1}: Diff MAE={mae:.4f} | iTrans MAE={guidance_mae:.4f} -> {save_path}")
            else:
                print(f"    Sample {i+1}: MAE={mae:.4f}, RMSE={rmse:.4f} -> {save_path}")
        
        # Store metrics
        results['metrics'] = {
            'diffusion_mean_mae': float(np.mean(diffusion_maes)),
            'diffusion_std_mae': float(np.std(diffusion_maes)),
            'diffusion_mean_rmse': float(np.mean(diffusion_rmses)),
            'diffusion_std_rmse': float(np.std(diffusion_rmses)),
            'num_samples': len(indices),
            'guidance_loaded': guidance_loaded,
        }
        
        # Add guidance metrics if available
        if guidance_maes:
            results['metrics']['guidance_mean_mae'] = float(np.mean(guidance_maes))
            results['metrics']['guidance_std_mae'] = float(np.std(guidance_maes))
            results['metrics']['guidance_mean_rmse'] = float(np.mean(guidance_rmses))
            results['metrics']['guidance_std_rmse'] = float(np.std(guidance_rmses))
            
            # Calculate improvement
            improvement = (np.mean(guidance_maes) - np.mean(diffusion_maes)) / np.mean(guidance_maes) * 100
            results['metrics']['improvement_pct'] = float(improvement)
        
        results['success'] = True
        
        # Print summary
        print(f"\n  Summary:")
        print(f"    Diffusion: MAE={np.mean(diffusion_maes):.4f}±{np.std(diffusion_maes):.4f}, "
              f"RMSE={np.mean(diffusion_rmses):.4f}±{np.std(diffusion_rmses):.4f}")
        
        if guidance_maes:
            print(f"    iTransformer: MAE={np.mean(guidance_maes):.4f}±{np.std(guidance_maes):.4f}, "
                  f"RMSE={np.mean(guidance_rmses):.4f}±{np.std(guidance_rmses):.4f}")
            improvement = (np.mean(guidance_maes) - np.mean(diffusion_maes)) / np.mean(guidance_maes) * 100
            status = "✅ Diffusion better" if improvement > 0 else "❌ iTransformer better"
            print(f"    Improvement: {improvement:+.1f}% {status}")
        
    except Exception as e:
        print(f"  ❌ Error visualizing {dataset_name}: {e}")
        traceback.print_exc()
        results['error'] = str(e)
    
    return results


def create_summary_report(
    all_results: List[dict],
    output_dir: str
):
    """Create a summary report of all visualizations.
    
    Args:
        all_results: List of result dicts from visualize_checkpoint
        output_dir: Output directory
    """
    report_path = os.path.join(output_dir, 'summary_report.txt')
    json_path = os.path.join(output_dir, 'summary_results.json')
    
    # Save JSON
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create text report
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TRAIN_MULTI_DATASET_OPTUNA.SH VISUALIZATION SUMMARY\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Datasets: {', '.join(OPTUNA_DATASETS)}\n")
        f.write("=" * 80 + "\n\n")
        
        # Count successes
        successes = [r for r in all_results if r['success']]
        failures = [r for r in all_results if not r['success']]
        
        f.write(f"Total checkpoints found: {len(all_results)}\n")
        f.write(f"Successful visualizations: {len(successes)}\n")
        f.write(f"Failed visualizations: {len(failures)}\n\n")
        
        if successes:
            f.write("-" * 100 + "\n")
            f.write("RESULTS BY DATASET (from train_multi_dataset_optuna.sh)\n")
            f.write("-" * 100 + "\n\n")
            
            # Sort by MAE
            successes.sort(key=lambda x: x['metrics'].get('diffusion_mean_mae', float('inf')))
            
            # Check if any have guidance
            has_any_guidance = any(r['metrics'].get('guidance_loaded', False) for r in successes)
            
            if has_any_guidance:
                f.write(f"{'Dataset':<13} {'Target':<10} {'Diff MAE':<14} {'iTrans MAE':<14} {'Improve':<10} {'Val Loss':<10}\n")
                f.write("-" * 100 + "\n")
                
                for r in successes:
                    m = r['metrics']
                    diff_mae = f"{m['diffusion_mean_mae']:.4f}±{m['diffusion_std_mae']:.4f}"
                    
                    if m.get('guidance_loaded', False) and 'guidance_mean_mae' in m:
                        guid_mae = f"{m['guidance_mean_mae']:.4f}±{m['guidance_std_mae']:.4f}"
                        improve = f"{m.get('improvement_pct', 0):+.1f}%"
                    else:
                        guid_mae = "N/A"
                        improve = "N/A"
                    
                    val_loss = f"{r['val_loss']:.4f}" if r['val_loss'] else "N/A"
                    target = r.get('target_column', 'unknown')[:9]
                    f.write(f"{r['dataset']:<13} {target:<10} {diff_mae:<14} {guid_mae:<14} {improve:<10} {val_loss:<10}\n")
            else:
                f.write(f"{'Dataset':<15} {'Target':<12} {'MAE':<16} {'RMSE':<16} {'Val Loss':<12}\n")
                f.write("-" * 80 + "\n")
                
                for r in successes:
                    m = r['metrics']
                    mae = f"{m['diffusion_mean_mae']:.4f}±{m['diffusion_std_mae']:.4f}"
                    rmse = f"{m['diffusion_mean_rmse']:.4f}±{m['diffusion_std_rmse']:.4f}"
                    val_loss = f"{r['val_loss']:.4f}" if r['val_loss'] else "N/A"
                    target = r.get('target_column', 'unknown')[:10]
                    f.write(f"{r['dataset']:<15} {target:<12} {mae:<16} {rmse:<16} {val_loss:<12}\n")
            
            f.write("\n")
        
        if failures:
            f.write("-" * 70 + "\n")
            f.write("FAILED VISUALIZATIONS\n")
            f.write("-" * 70 + "\n\n")
            
            for r in failures:
                f.write(f"Dataset: {r['dataset']}\n")
                f.write(f"  Checkpoint: {r['checkpoint']}\n")
                f.write(f"  Error: {r['error']}\n\n")
        
        # Missing datasets
        found_datasets = set(r['dataset'] for r in all_results)
        missing = set(OPTUNA_DATASETS) - found_datasets
        
        if missing:
            f.write("-" * 70 + "\n")
            f.write("MISSING DATASETS (no checkpoints found)\n")
            f.write("-" * 70 + "\n\n")
            for ds in missing:
                f.write(f"  - {ds}\n")
    
    print(f"\n📊 Summary saved to: {report_path}")
    print(f"📊 JSON results saved to: {json_path}")
    
    # Also create a comparison plot
    if len(successes) >= 2:
        create_comparison_plot(successes, output_dir)


def create_comparison_plot(results: List[dict], output_dir: str):
    """Create a bar chart comparing Diffusion vs iTransformer MAE across datasets.
    
    Args:
        results: List of successful result dicts
        output_dir: Output directory
    """
    datasets = [r['dataset'] for r in results]
    
    # Extract diffusion metrics
    diff_maes = [r['metrics']['diffusion_mean_mae'] for r in results]
    diff_mae_stds = [r['metrics']['diffusion_std_mae'] for r in results]
    
    # Extract guidance metrics (if available)
    has_guidance = [r['metrics'].get('guidance_loaded', False) and 'guidance_mean_mae' in r['metrics'] 
                   for r in results]
    any_guidance = any(has_guidance)
    
    if any_guidance:
        guid_maes = [r['metrics'].get('guidance_mean_mae', 0) for r in results]
        guid_mae_stds = [r['metrics'].get('guidance_std_mae', 0) for r in results]
    
    x = np.arange(len(datasets))
    
    if any_guidance:
        # Side-by-side bar chart comparing Diffusion vs iTransformer
        fig, ax = plt.subplots(figsize=(14, 7))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, diff_maes, width, yerr=diff_mae_stds, 
                      label='Diffusion', color='#e74c3c', capsize=4, alpha=0.85)
        bars2 = ax.bar(x + width/2, guid_maes, width, yerr=guid_mae_stds, 
                      label='iTransformer', color='#2ecc71', capsize=4, alpha=0.85)
        
        ax.set_ylabel('MAE (lower is better)', fontsize=12)
        ax.set_title('Diffusion vs iTransformer Forecast Error by Dataset', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=45, ha='right', fontsize=11)
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val, has_g in zip(bars1, diff_maes, has_guidance):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        
        for bar, val, has_g in zip(bars2, guid_maes, has_guidance):
            if has_g:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Add improvement annotations
        for i, (d_mae, g_mae, has_g) in enumerate(zip(diff_maes, guid_maes, has_guidance)):
            if has_g and g_mae > 0:
                improvement = (g_mae - d_mae) / g_mae * 100
                color = '#27ae60' if improvement > 0 else '#c0392b'
                symbol = '↑' if improvement > 0 else '↓'
                ax.annotate(f'{symbol}{abs(improvement):.1f}%', 
                           xy=(x[i], max(d_mae, g_mae) + 0.02),
                           ha='center', fontsize=9, fontweight='bold', color=color)
        
    else:
        # Single bar chart for diffusion only
        fig, ax = plt.subplots(figsize=(12, 6))
        width = 0.6
        
        bars = ax.bar(x, diff_maes, width, yerr=diff_mae_stds, capsize=5, 
                     color='#e74c3c', alpha=0.85)
        ax.set_ylabel('MAE (lower is better)', fontsize=12)
        ax.set_title('Diffusion Forecast Error by Dataset', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=45, ha='right', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, diff_maes):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'dataset_comparison.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    print(f"📊 Comparison plot saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize multi-dataset Optuna results')
    parser.add_argument('--checkpoint-dir', type=str, 
                       default=os.path.join(DIFFUSION_TSF_DIR, 'checkpoints'),
                       help='Base checkpoint directory')
    parser.add_argument('--output-dir', type=str, default='optuna_visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--num-samples', type=int, default=5,
                       help='Number of samples to visualize per dataset')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Visualize only this dataset (e.g., ETTh2)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--decoder-method', type=str, default='mean',
                        choices=['mean', 'pdf_expectation', 'median', 'mode', 'beam'],
                        help='Decoder to map CDF back to 1D (CDF mode only)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("VISUALIZE train_multi_dataset_optuna.sh RESULTS")
    print("=" * 70)
    print(f"Target datasets: {', '.join(OPTUNA_DATASETS)}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Samples per dataset: {args.num_samples}")
    print(f"Device: {args.device}")
    if args.dataset:
        print(f"Dataset filter: {args.dataset}")
    print(f"Decoder method: {args.decoder_method}")
    print("=" * 70 + "\n")
    
    # Find checkpoints from train_multi_dataset_optuna.sh ONLY
    checkpoints = find_optuna_checkpoints(args.checkpoint_dir, args.dataset)
    
    if not checkpoints:
        print("\n❌ No Optuna checkpoints found from train_multi_dataset_optuna.sh!")
        print(f"   Looked in: {args.checkpoint_dir}")
        print(f"   Expected pattern: diffusion_tsf_{{ETTh2|ETTm1|illness|exchange_rate|traffic|weather}}[_timestamp]")
        print("\n   Make sure to sync checkpoints from remote first:")
        print("   ./sync_checkpoints.sh <remote_ip> --trials")
        return
    
    # Report what was found
    print(f"\n📁 Found Optuna checkpoints for {len(checkpoints)}/{len(OPTUNA_DATASETS)} datasets:")
    for ds, ckpts in checkpoints.items():
        best = ckpts[0]
        loss_str = f"{best['val_loss']:.4f}" if best['val_loss'] else 'N/A'
        target = best.get('target_column', 'unknown')
        print(f"   {ds}: target='{target}', val_loss={loss_str}")
    
    # Check for missing datasets
    found = set(checkpoints.keys())
    expected = set(OPTUNA_DATASETS)
    missing = expected - found
    if missing:
        print(f"\n⚠️  Missing datasets: {', '.join(sorted(missing))}")
        print("   (Training may not have completed before GPU credits ran out)")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Visualize each dataset
    all_results = []
    
    for dataset_name, ckpt_list in checkpoints.items():
        # Use the best checkpoint for each dataset
        best_ckpt = ckpt_list[0]
        
        result = visualize_checkpoint(
            checkpoint_info=best_ckpt,
            output_base_dir=args.output_dir,
            num_samples=args.num_samples,
            device=args.device,
            decoder_method=args.decoder_method
        )
        all_results.append(result)
    
    # Create summary report
    print("\n" + "=" * 70)
    print("CREATING SUMMARY REPORT")
    print("=" * 70)
    create_summary_report(all_results, args.output_dir)
    
    # Final summary
    successes = sum(1 for r in all_results if r['success'])
    print(f"\n✅ Visualization complete!")
    print(f"   Successful: {successes}/{len(all_results)}")
    print(f"   Output: {args.output_dir}/")


if __name__ == '__main__':
    main()

