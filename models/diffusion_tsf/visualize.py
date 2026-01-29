"""
Visualization script for Diffusion TSF.

Loads a trained model and generates plots for evenly sampled windows across the dataset.
Supports models trained with iTransformer guidance and plots guidance predictions.
"""

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import glob
import importlib.util
from typing import Tuple, Optional
from torch.utils.data import random_split

# Setup path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from dataset import apply_1d_augmentations
from train_electricity import ElectricityDataset, DATASET_REGISTRY, MODEL_SIZES
from diffusion_model import DiffusionTSF
from config import DiffusionTSFConfig
from guidance import iTransformerGuidance, LinearRegressionGuidance, LastValueGuidance

# Datasets directory (relative to script)
DATASETS_DIR = os.path.join(script_dir, '..', '..', 'datasets')


def load_itransformer_guidance(
    checkpoint_path: str,
    seq_len: int = 512,
    pred_len: int = 96,
    num_variables: int = 1,
    device: str = 'cpu'
) -> iTransformerGuidance:
    """Load a pre-trained iTransformer model as guidance for visualization.
    
    Args:
        checkpoint_path: Path to iTransformer checkpoint (.pt file)
        seq_len: Input sequence length
        pred_len: Prediction length
        num_variables: Number of variables in the dataset
        device: Device to load model on
        
    Returns:
        iTransformerGuidance wrapper around the loaded model
    """
    # Use importlib to load from absolute path to avoid conflicts with local model.py
    itrans_model_path = os.path.join(script_dir, '..', 'iTransformer', 'model', 'iTransformer.py')
    itrans_model_path = os.path.abspath(itrans_model_path)
    
    # Also need to add iTransformer to path for its internal imports (layers, etc.)
    itrans_dir = os.path.join(script_dir, '..', 'iTransformer')
    itrans_dir = os.path.abspath(itrans_dir)
    if itrans_dir not in sys.path:
        sys.path.insert(0, itrans_dir)
    
    # Load the module using spec
    spec = importlib.util.spec_from_file_location("iTransformer_module", itrans_model_path)
    itrans_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(itrans_module)
    iTransformerModel = itrans_module.Model
    
    # Load checkpoint
    print(f"Loading iTransformer from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Try to extract config from checkpoint
    if 'config' in checkpoint:
        ckpt_config = checkpoint['config']
        print(f"Found config in checkpoint: seq_len={ckpt_config.get('seq_len')}, pred_len={ckpt_config.get('pred_len')}")
    else:
        ckpt_config = {}
    
    # Get the state dict to infer model architecture
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Auto-detect e_layers from state_dict by counting encoder.attn_layers.X
    detected_e_layers = 0
    for key in state_dict.keys():
        if key.startswith('encoder.attn_layers.'):
            layer_idx = int(key.split('.')[2])
            detected_e_layers = max(detected_e_layers, layer_idx + 1)
    
    # Auto-detect d_model from embedding weight shape
    detected_d_model = 512  # default
    detected_d_ff = 2048  # default
    if 'enc_embedding.value_embedding.weight' in state_dict:
        detected_d_model = state_dict['enc_embedding.value_embedding.weight'].shape[0]
    
    # Auto-detect d_ff from conv1 weight shape
    if 'encoder.attn_layers.0.conv1.weight' in state_dict:
        detected_d_ff = state_dict['encoder.attn_layers.0.conv1.weight'].shape[0]
    
    # Auto-detect n_heads from attention projection shape
    detected_n_heads = 8  # default
    if 'encoder.attn_layers.0.attention.query_projection.weight' in state_dict:
        # query_projection.weight shape is [d_model, d_model]
        # n_heads = d_model / d_keys, where d_keys is typically d_model / n_heads
        # We can infer from the projector output size
        pass  # Keep default, hard to infer n_heads from weights
    
    print(f"Auto-detected from state_dict: e_layers={detected_e_layers}, d_model={detected_d_model}, d_ff={detected_d_ff}")
    
    # Create a config object for iTransformer
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
            self.d_ff = ckpt_config.get('d_ff', detected_d_ff)
            self.activation = 'gelu'
            self.e_layers = ckpt_config.get('e_layers', detected_e_layers if detected_e_layers > 0 else 3)
            self.class_strategy = 'projection'
            self.enc_in = num_variables
    
    config = iTransConfig()
    
    # Create model
    model = iTransformerModel(config)
    
    # Load weights
    model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    
    print(f"iTransformer loaded: seq_len={config.seq_len}, pred_len={config.pred_len}")
    
    # Wrap in guidance interface
    guidance = iTransformerGuidance(
        model=model,
        use_norm=config.use_norm,
        seq_len=config.seq_len,
        pred_len=config.pred_len
    )
    
    return guidance


def create_guidance_for_visualization(
    guidance_type: str,
    guidance_checkpoint: Optional[str],
    seq_len: int,
    pred_len: int,
    num_variables: int,
    device: str
):
    """Create guidance model based on configuration."""
    if guidance_type == "linear":
        print("Using LinearRegressionGuidance for Stage 1 predictions")
        return LinearRegressionGuidance()
    
    elif guidance_type == "last_value":
        print("Using LastValueGuidance for Stage 1 predictions")
        return LastValueGuidance()
    
    elif guidance_type == "itransformer":
        if not guidance_checkpoint:
            raise ValueError("guidance_checkpoint is required for itransformer guidance")
        return load_itransformer_guidance(
            checkpoint_path=guidance_checkpoint,
            seq_len=seq_len,
            pred_len=pred_len,
            num_variables=num_variables,
            device=device
        )
    
    else:
        raise ValueError(f"Unknown guidance type: {guidance_type}")

def visualize_samples(
    model_path: str,
    data_path: str,
    num_samples: int = 5,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    decoder_method: str = "mean",
    beam_width: int = 5,
    jump_penalty_scale: float = 1.0,
    search_radius: int = 10,
    output_dir: str = "visualizations",
    guidance_checkpoint: Optional[str] = None
):
    # 1. Load checkpoint
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    config_dict = checkpoint['config']
    
    # Resolve dataset - prioritize explicitly provided data_path (for pretrain-only mode)
    # Only fall back to checkpoint config if data_path is not provided or is generic
    num_variables = config_dict.get('num_variables', 1)
    dataset_name = config_dict.get('dataset', None)

    # Check if the provided data_path is different from the checkpoint's dataset
    # This happens in pretrain-only mode where we use one model on multiple datasets
    checkpoint_dataset_path = None
    if dataset_name and dataset_name in DATASET_REGISTRY:
        checkpoint_dataset_path = os.path.join(DATASETS_DIR, DATASET_REGISTRY[dataset_name][0])

    # If data_path is explicitly provided and different from checkpoint path, use it
    # This allows pretrain-only visualizations to work with different datasets
    if data_path and checkpoint_dataset_path and os.path.abspath(data_path) != os.path.abspath(checkpoint_dataset_path):
        resolved_data_path = data_path
        print(f"📊 Using provided data_path (differs from checkpoint): {dataset_name} → {os.path.basename(data_path)}")
        print(f"   Data path: {resolved_data_path}")
        # Keep dataset_name as is for logging, but use the provided path
    elif dataset_name and dataset_name in DATASET_REGISTRY:
        # Checkpoint has dataset name saved - use it directly
        dataset_info = DATASET_REGISTRY[dataset_name]
        resolved_data_path = os.path.join(DATASETS_DIR, dataset_info[0])
        print(f"📊 Using dataset from checkpoint: {dataset_name}")
        print(f"   Data path: {resolved_data_path}")
    else:
        # OLD checkpoint without 'dataset' key - try to infer from num_variables
        # Map num_variables to likely datasets
        dataset_by_vars = {
            321: 'electricity',  # electricity has 321 clients
            7: 'ETTh1',  # ETT datasets have 7 columns (HUFL, HULL, MUFL, MULL, LUFL, LULL, OT)
            8: 'exchange_rate',  # exchange_rate has 8 countries
            21: 'weather',  # weather has 21 features
            861: 'traffic',  # traffic has 861 sensors
        }

        inferred_dataset = dataset_by_vars.get(num_variables, None)

        if inferred_dataset and inferred_dataset in DATASET_REGISTRY:
            dataset_name = inferred_dataset
            dataset_info = DATASET_REGISTRY[dataset_name]
            resolved_data_path = os.path.join(DATASETS_DIR, dataset_info[0])
            print(f"📊 Inferred dataset from num_variables={num_variables}: {dataset_name}")
            print(f"   Data path: {resolved_data_path}")
            print(f"   (Old checkpoint without 'dataset' key - using inference)")
        else:
            # Fallback to provided data_path
            resolved_data_path = data_path
            dataset_name = 'unknown'
            print(f"⚠️  Could not determine dataset (num_variables={num_variables})")
            print(f"   Using provided data path: {data_path}")
    
    # Debug: Print what's actually in the checkpoint config
    print(f"\n=== Checkpoint Config ===")
    print(f"  dataset: {dataset_name}")
    print(f"  representation_mode: {config_dict.get('representation_mode', 'NOT FOUND (defaulting to pdf)')}")
    print(f"  blur_sigma: {config_dict.get('blur_sigma', 'NOT FOUND')}")
    print(f"  emd_lambda: {config_dict.get('emd_lambda', 'NOT FOUND')}")
    print(f"  model_size: {config_dict.get('model_size', 'NOT FOUND')}")
    print(f"=========================\n")
    
    # 2. Reconstruct model
    # Determine model type (defaults to unet if not present)
    model_type = config_dict.get('model_type', 'unet')
    
    # Reconstruct attention_levels and num_res_blocks based on model_size
    # (must match training logic exactly!)
    model_size = config_dict.get('model_size', 'small')
    if model_size == 'large':
        attention_levels = [1, 2]
        num_res_blocks = 3
    elif model_size == 'medium':
        attention_levels = [1, 2, 3]  # Different from small/large!
        num_res_blocks = 2
    else:  # small or tiny
        attention_levels = [1, 2]
        num_res_blocks = 2
    
    # Build config (pull representation_mode if present)
    # Detect usage of coordinate channel and kernel size from weight shapes
    state_dict = checkpoint['model_state_dict']

    # Remap legacy "unet.*" keys to "noise_predictor.*" first
    if any(k.startswith("unet.") for k in state_dict.keys()):
        remapped = {}
        for k, v in state_dict.items():
            if k.startswith("unet."):
                remapped["noise_predictor." + k[len("unet."):]] = v
            else:
                remapped[k] = v
        state_dict = remapped
    
    # Get basic settings from config
    num_variables = config_dict.get('num_variables', 1)
    seasonal_period = config_dict.get('seasonal_period', 96)
    
    # Infer num_variables and image_height from weights if missing/default
    if 'noise_predictor.final_conv.weight' in state_dict:
        inferred_vars = state_dict['noise_predictor.final_conv.weight'].shape[0]
        if num_variables == 1 and inferred_vars != 1:
            print(f"Auto-detected num_variables={inferred_vars} from final_conv weights (config had {num_variables})")
            num_variables = inferred_vars
            
    if 'to_2d.bin_centers' in state_dict:
        inferred_height = state_dict['to_2d.bin_centers'].shape[0]
        config_height = config_dict.get('image_height', 128)
        if inferred_height != config_height:
            print(f"Auto-detected image_height={inferred_height} from bin_centers (config/default had {config_height})")
            config_dict['image_height'] = inferred_height
    
    # Auto-detect conditioning_mode from state_dict (not config default!)
    # If cond_encoder keys exist -> vector_embedding mode
    # If no cond_encoder keys -> visual_concat mode
    cond_encoder_key = 'noise_predictor.cond_encoder.local_encoder.0.weight'
    has_cond_encoder = cond_encoder_key in state_dict
    
    if 'conditioning_mode' in config_dict:
        conditioning_mode = config_dict['conditioning_mode']
    else:
        # Auto-detect from state_dict
        conditioning_mode = 'vector_embedding' if has_cond_encoder else 'visual_concat'
    print(f"Conditioning mode: {conditioning_mode} (has_cond_encoder={has_cond_encoder})")
    
    # Detect hybrid conditioning (1D cross-attention)
    # Check for any context_encoder key, as legacy models might have different internal structure
    has_context_encoder = any(k.startswith('context_encoder.') for k in state_dict)
    use_hybrid_condition = config_dict.get('use_hybrid_condition', has_context_encoder)
    
    # Auto-detect coordinate channel and kernel size from weight shapes
    use_coord_channel = config_dict.get('use_coordinate_channel', None)
    unet_kernel_size = config_dict.get('unet_kernel_size', None)
    use_time_ramp = config_dict.get('use_time_ramp', None)
    use_time_sine = config_dict.get('use_time_sine', None)
    use_value_channel = config_dict.get('use_value_channel', None)
    
    if model_type == 'transformer':
        # For transformer: patch_embed.weight shape is [embed_dim, in_channels * pH * pW]
        if use_coord_channel is None:
            patch_height = config_dict.get('transformer_patch_height', 16)
            patch_width = config_dict.get('transformer_patch_width', 16)
            expected_2ch = 2 * patch_height * patch_width  # 512 for 16x16
            
            patch_key = 'noise_predictor.patch_embed.weight'
            if patch_key in state_dict:
                actual_dim = state_dict[patch_key].shape[1]
                use_coord_channel = (actual_dim == expected_2ch)
                print(f"Auto-detected use_coordinate_channel={use_coord_channel} from patch_embed shape {state_dict[patch_key].shape}")
            else:
                use_coord_channel = False
        # Transformer doesn't use unet_kernel_size
        if unet_kernel_size is None:
            unet_kernel_size = (3, 3)  # Default, not used for transformer
        # Default time channels for transformer
        if use_time_ramp is None: use_time_ramp = False
        if use_time_sine is None: use_time_sine = False
        if use_value_channel is None: use_value_channel = False
    else:
        # For U-Net: check init_conv weight shape
        # init_conv expects [out_ch, total_in_channels, kH, kW]
        init_key = 'noise_predictor.init_conv.weight'
        if init_key in state_dict:
            init_weight_shape = state_dict[init_key].shape
            total_in_channels = init_weight_shape[1]
            
            # Detect kernel size from weight shape: [out_ch, in_ch, kH, kW]
            if unet_kernel_size is None:
                detected_kh = init_weight_shape[2]
                detected_kw = init_weight_shape[3]
                unet_kernel_size = (detected_kh, detected_kw)
                print(f"Auto-detected unet_kernel_size={unet_kernel_size} from init_conv shape {init_weight_shape}")
            
            # Compute expected channels based on conditioning mode to infer aux channels
            # For visual_concat: total = backbone_in + visual_cond = (num_vars + aux) + num_vars = 2*num_vars + aux
            # For vector_embedding: total = backbone_in + 64 = (num_vars + aux) + 64
            if conditioning_mode == 'visual_concat':
                # Base assumption: input = noisy(N) + past(N) + aux(?)
                # If guidance enabled: input = noisy(N) + past(N) + guidance(N) + aux(?)
                residual = total_in_channels - 2 * num_variables
                
                # Check if guidance fits in the residual
                # Heuristic: if residual is large enough to contain guidance + at least one aux (or 0)
                # And if use_guidance_channel is not explicitly set to False in config
                if residual >= num_variables and config_dict.get('use_guidance_channel') is not False:
                    # Check if the remaining channels make sense as aux
                    potential_aux = residual - num_variables
                    if potential_aux <= 4: # Max 4 aux channels
                        print(f"Auto-detected use_guidance_channel=True from init_conv (residual {residual} split into {num_variables} guidance + {potential_aux} aux)")
                        use_guidance_channel = True
                        num_aux_channels = potential_aux
                    else:
                        # Maybe not guidance, just lots of aux? (Unlikely)
                        num_aux_channels = residual
                else:
                    num_aux_channels = residual
            else:
                # Vector embedding: input = noisy(N) + embedding(64) + aux
                # Guidance logic similar?
                num_aux_channels = total_in_channels - num_variables - 64
                if use_guidance_channel: # If config says so
                     num_aux_channels -= num_variables
            
            print(f"Detected {num_aux_channels} auxiliary channels from init_conv (total_in={total_in_channels})")
            
            # Infer which aux channels are enabled based on count
            # Order in model: [data, coord, time_ramp, time_sine, value]
            # Most common configurations:
            # 0 aux: none
            # 1 aux: coord only
            # 2 aux: coord + time_ramp
            # 3 aux: coord + time_ramp + value OR coord + time_ramp + time_sine
            # 4 aux: coord + time_ramp + time_sine + value
            if use_coord_channel is None:
                use_coord_channel = num_aux_channels >= 1
            if use_time_ramp is None:
                use_time_ramp = num_aux_channels >= 2
            if use_time_sine is None:
                use_time_sine = num_aux_channels >= 4  # Only if all 4 aux channels
            if use_value_channel is None:
                use_value_channel = num_aux_channels >= 3
            
            print(f"Auto-detected aux channels: coord={use_coord_channel}, time_ramp={use_time_ramp}, "
                  f"time_sine={use_time_sine}, value={use_value_channel}")
        else:
            if use_coord_channel is None: use_coord_channel = False
            if unet_kernel_size is None: unet_kernel_size = (3, 3)
            if use_time_ramp is None: use_time_ramp = False
            if use_time_sine is None: use_time_sine = False
            if use_value_channel is None: use_value_channel = False
    
    # Fallback defaults
    if use_coord_channel is None: use_coord_channel = False
    if use_time_ramp is None: use_time_ramp = False
    if use_time_sine is None: use_time_sine = False
    if use_value_channel is None: use_value_channel = False
    
    model_config = DiffusionTSFConfig(
        lookback_length=config_dict.get('lookback_length', 512),
        forecast_length=config_dict.get('forecast_length', 96),
        image_height=config_dict.get('image_height', 128),
        max_scale=config_dict.get('max_scale', 3.5),
        blur_kernel_size=config_dict.get('blur_kernel_size', 31),
        blur_sigma=config_dict.get('blur_sigma', 1.0),
        emd_lambda=config_dict.get('emd_lambda', 0.2),
        representation_mode=config_dict.get('representation_mode', 'cdf'),
        unet_channels=config_dict.get('unet_channels', MODEL_SIZES.get(model_size, [64, 128, 256])),
        num_res_blocks=num_res_blocks,
        attention_levels=attention_levels,
        num_diffusion_steps=config_dict.get('num_diffusion_steps', config_dict.get('diffusion_steps', 100)),
        noise_schedule=config_dict.get('noise_schedule', 'linear'),
        model_type=model_type,
        use_coordinate_channel=use_coord_channel,
        unet_kernel_size=unet_kernel_size,
        use_time_ramp=use_time_ramp,
        use_time_sine=use_time_sine,
        use_value_channel=use_value_channel,
        seasonal_period=seasonal_period,
        num_variables=num_variables,
        conditioning_mode=conditioning_mode,
        use_hybrid_condition=use_hybrid_condition,
        context_embedding_dim=config_dict.get('context_embedding_dim', 128),
        context_encoder_layers=config_dict.get('context_encoder_layers', 2),
        use_guidance_channel=use_guidance_channel,
    )
    # If using transformer, optionally override transformer params from checkpoint
    if model_type == 'transformer':
        for k in [
            'transformer_embed_dim',
            'transformer_depth',
            'transformer_num_heads',
            'transformer_patch_height',
            'transformer_patch_width',
            'transformer_dropout',
        ]:
            if k in config_dict:
                setattr(model_config, k, config_dict[k])
        # Handle legacy checkpoints with single patch_size
        if 'transformer_patch_size' in config_dict and 'transformer_patch_height' not in config_dict:
            legacy_size = config_dict['transformer_patch_size']
            model_config.transformer_patch_height = legacy_size
            model_config.transformer_patch_width = legacy_size
    
    # Detect guidance channel settings from config
    use_guidance_channel = config_dict.get('use_guidance_channel', False)
    guidance_type = config_dict.get('guidance_type', None)
    saved_guidance_checkpoint = config_dict.get('guidance_checkpoint', None)
    
    # Use provided guidance_checkpoint or fall back to saved one from training
    effective_guidance_checkpoint = guidance_checkpoint or saved_guidance_checkpoint
    
    # Remap remote paths to local paths (e.g., /root/ts-sandbox/... -> local project)
    if effective_guidance_checkpoint and not os.path.exists(effective_guidance_checkpoint):
        # Try to remap common remote path patterns
        remote_prefixes = [
            '/root/ts-sandbox/',
            '~/ts-sandbox/',
            '/home/cao/ts-sandbox/',
        ]
        project_root = os.path.dirname(os.path.dirname(script_dir))  # Go up from diffusion_tsf to ts-sandbox
        
        for prefix in remote_prefixes:
            if effective_guidance_checkpoint.startswith(prefix):
                relative_path = effective_guidance_checkpoint[len(prefix):]
                local_path = os.path.join(project_root, relative_path)
                if os.path.exists(local_path):
                    print(f"Remapped guidance checkpoint path:")
                    print(f"  Remote: {effective_guidance_checkpoint}")
                    print(f"  Local:  {local_path}")
                    effective_guidance_checkpoint = local_path
                    break
    
    if use_guidance_channel:
        model_config.use_guidance_channel = True
        print(f"\n=== Guidance Configuration ===")
        print(f"  use_guidance_channel: {use_guidance_channel}")
        print(f"  guidance_type: {guidance_type}")
        print(f"  guidance_checkpoint: {effective_guidance_checkpoint}")
        print(f"==============================\n")
    
    # state_dict already remapped above (legacy unet.* -> noise_predictor.*)
    model = DiffusionTSF(model_config).to(device)
    
    # Load guidance model if checkpoint was trained with guidance
    guidance_model = None
    guidance_loaded = False
    if use_guidance_channel and guidance_type:
        if guidance_type == 'itransformer' and not effective_guidance_checkpoint:
            print("WARNING: Model was trained with iTransformer guidance but no checkpoint provided!")
            print("         Use --guidance-checkpoint to specify the iTransformer checkpoint path.")
            print("         Guidance predictions will NOT be shown in visualizations.")
        else:
            try:
                guidance_model = create_guidance_for_visualization(
                    guidance_type=guidance_type,
                    guidance_checkpoint=effective_guidance_checkpoint,
                    seq_len=512,
                    pred_len=96,
                    num_variables=num_variables,
                    device=device
                )
                model.set_guidance_model(guidance_model)
                guidance_loaded = True
                print(f"✅ Guidance model loaded successfully!")
            except Exception as e:
                print(f"WARNING: Failed to load guidance model: {e}")
                print("         Guidance predictions will NOT be shown in visualizations.")
    
    # Filter out guidance_model.* keys if we couldn't load the guidance model
    # (these keys are saved with the checkpoint when training with guidance)
    if not guidance_loaded:
        guidance_keys = [k for k in state_dict.keys() if k.startswith('guidance_model.')]
        if guidance_keys:
            print(f"Filtering out {len(guidance_keys)} guidance_model.* keys from state_dict (guidance not loaded)")
            for k in guidance_keys:
                del state_dict[k]
    
    # Use strict=True to catch architecture mismatches!
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(f"WARNING: State dict loading failed with strict=True: {e}")
        print("Attempting to load with strict=False (some weights may be random!)")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    
    model.eval()
    
    # 3. Load evaluation subset using CHRONOLOGICAL splits
    # This ensures consistency with iTransformer's training and avoids data leakage
    # Split: Train (first 70%), Val (next 10%), Test (last 20%)
    
    # Detect if model was trained multivariate (num_variables > 1)
    use_all_columns = config_dict.get('use_all_columns', False) or num_variables > 1
    if use_all_columns:
        print(f"\n📊 Multivariate mode: loading all {num_variables} columns from dataset")
    
    base_dataset = ElectricityDataset(
        resolved_data_path,
        lookback=model_config.lookback_length,
        forecast=model_config.forecast_length,
        augment=False,
        use_all_columns=use_all_columns
    )
    
    # Verify dataset num_variables matches model
    dataset_num_vars = base_dataset.num_variables
    if dataset_num_vars != num_variables:
        print(f"⚠️  WARNING: Dataset has {dataset_num_vars} variables but model expects {num_variables}")
        print(f"   This may cause dimension mismatches!")
    
    total_samples = len(base_dataset)
    
    # Use same gap-based split as training to match exactly
    window_size = model_config.lookback_length + model_config.forecast_length  # lookback + forecast
    stride = 24
    gap_indices = (window_size + stride - 1) // stride  # ~26 indices
    
    raw_train_end = int(total_samples * 0.7)
    raw_val_end = int(total_samples * 0.8)
    
    train_end = raw_train_end
    val_start = train_end + gap_indices
    val_end = raw_val_end
    test_start = val_end + gap_indices
    
    # Ensure valid ranges
    if val_start >= val_end:
        val_start = train_end + 1
    if test_start >= total_samples:
        test_start = val_end + 1
    
    if use_guidance_channel and guidance_type == 'itransformer':
        # When using iTransformer, evaluate on TEST set
        # to ensure we're evaluating on data neither model has seen
        print("\n⚠️  Using CHRONOLOGICAL TEST set for fair iTransformer evaluation")
        print("   (Both iTransformer and diffusion were trained on train set)")
        
        eval_indices = list(range(test_start, total_samples))
        eval_set_name = "chronological test set"
    else:
        # When not using iTransformer, use validation set
        print("\n📊 Using CHRONOLOGICAL validation set")
        
        eval_indices = list(range(val_start, val_end))
        eval_set_name = "chronological validation set"
    
    print(f"   Dataset: {total_samples} total samples")
    print(f"   Window: {window_size} timesteps, stride: {stride}, gap: {gap_indices} indices")
    print(f"   Train:   indices 0-{train_end-1} ({train_end} samples)")
    print(f"   [GAP]:   {gap_indices} indices (no overlap zone)")
    print(f"   Val:     indices {val_start}-{val_end-1} ({val_end - val_start} samples)")
    print(f"   [GAP]:   {gap_indices} indices")
    print(f"   Test:    indices {test_start}-{total_samples-1} ({total_samples - test_start} samples)")
    print(f"   Using:   {len(eval_indices)} samples from {eval_set_name}\n")
    
    val_dataset = ElectricityDataset(
        resolved_data_path,
        lookback=model_config.lookback_length,
        forecast=model_config.forecast_length,
        augment=False,
        use_all_columns=use_all_columns,
        data_tensor=base_dataset.data,
        indices=eval_indices
    )
    
    # Evenly sample across the validation dataset for diverse visualizations
    total_samples = len(val_dataset)
    if total_samples <= num_samples:
        indices = list(range(total_samples))
    else:
        # Use linspace to get evenly spaced indices across the validation dataset
        indices = np.linspace(0, total_samples - 1, num_samples, dtype=int).tolist()

    print(f"Generating {num_samples} visualizations from {eval_set_name}...")
    print(f"  Model config: {model_size} model, {model_config.representation_mode} mode")
    print(f"  Attention levels: {attention_levels}, Res blocks: {num_res_blocks}")
    print(f"  Sampling indices (within eval_dataset): {indices}")
    print(f"  Corresponding original dataset indices: {[eval_indices[i] for i in indices]}")

    os.makedirs(output_dir, exist_ok=True)
    
    # Track metrics for summary
    diffusion_maes = []
    guidance_maes = []
    diffusion_rmses = []
    guidance_rmses = []
    
    for i, idx in enumerate(indices):
        past, future = val_dataset[idx]
        past_tensor = past.unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Generate prediction
            out = model.generate(
                past_tensor,
                use_ddim=True,
                num_ddim_steps=50,
                verbose=True,
                decoder_method=decoder_method,
                beam_width=beam_width,
                jump_penalty_scale=jump_penalty_scale,
                search_radius=search_radius
            )
            pred = out['prediction'].cpu().squeeze(0).numpy()
            
            # Extract guidance prediction if available
            guidance_pred = None
            guidance_2d = None
            if 'guidance_1d' in out:
                guidance_pred = out['guidance_1d'].cpu().squeeze(0).numpy()
            if 'guidance_2d' in out:
                guidance_2d_raw = out['guidance_2d'].cpu().squeeze(0).numpy()
                # For multivariate, take first variable for visualization
                if guidance_2d_raw.ndim == 3:
                    guidance_2d_raw = guidance_2d_raw[0]  # (height, width)
                guidance_2d = (guidance_2d_raw + 1.0) / 2.0
                guidance_2d = np.clip(guidance_2d, 0.0, 1.0)
            
            # Extract 2D maps (these are in diffusion-scaled [-1, 1] range)
            past_2d_raw = out['past_2d'].cpu().squeeze(0).numpy()
            future_2d_raw = out['future_2d'].cpu().squeeze(0).numpy()
            
            # For multivariate, take first variable for 2D visualization
            if past_2d_raw.ndim == 3:
                past_2d_raw = past_2d_raw[0]  # (height, width)
            if future_2d_raw.ndim == 3:
                future_2d_raw = future_2d_raw[0]  # (height, width)
            
            # Convert from diffusion space [-1, 1] to probability space [0, 1]
            past_2d = (past_2d_raw + 1.0) / 2.0
            future_2d = (future_2d_raw + 1.0) / 2.0
            
            # Clip to valid range (diffusion can sometimes exceed bounds slightly)
            past_2d = np.clip(past_2d, 0.0, 1.0)
            future_2d = np.clip(future_2d, 0.0, 1.0)
        
        # For multivariate, extract first variable for 1D plotting
        past_1d = past.numpy()
        future_1d = future.numpy()
        pred_1d = pred
        guidance_pred_1d = guidance_pred
        
        if past_1d.ndim == 2:
            # Multivariate: (num_vars, time) -> take first variable
            past_1d = past_1d[0]
            future_1d = future_1d[0]
            pred_1d = pred[0] if pred.ndim == 2 else pred
            if guidance_pred is not None and guidance_pred.ndim == 2:
                guidance_pred_1d = guidance_pred[0]
        
        # Determine figure layout based on whether we have guidance
        has_guidance = guidance_pred is not None
        
        if has_guidance and guidance_2d is not None:
            # 3 rows: 1D time series, 2D probability map, 2D guidance map
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 16), 
                                                 gridspec_kw={'height_ratios': [1, 1, 1]})
        else:
            # 2 rows: 1D time series, 2D probability map
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), 
                                            gridspec_kw={'height_ratios': [1, 1]})
        
        # 1. Plot 1D Time Series (first variable for multivariate)
        time_past = np.arange(len(past_1d))
        time_future = np.arange(len(past_1d), len(past_1d) + len(future_1d))
        
        var_label = " (Variable 0)" if num_variables > 1 else ""
        ax1.plot(time_past, past_1d, label=f'Past (Context){var_label}', color='gray', alpha=0.6)
        ax1.plot(time_future, future_1d, label='True Future', color='blue', linewidth=2)
        ax1.plot(time_future, pred_1d, label='Diffusion Forecast', color='red', linestyle='--', linewidth=2)
        
        # Add guidance prediction if available
        if has_guidance and guidance_pred_1d is not None:
            ax1.plot(time_future, guidance_pred_1d, label='iTransformer Guidance', 
                    color='green', linestyle=':', linewidth=2, alpha=0.8)
        
        title_suffix = " (with iTransformer Guidance)" if has_guidance else ""
        multivar_suffix = f" [showing var 0 of {num_variables}]" if num_variables > 1 else ""
        ax1.set_title(f"Diffusion TSF Forecast - Sample {i+1} (Dataset Index {idx}){title_suffix}{multivar_suffix}", fontsize=14)
        ax1.set_xlabel("Time Steps")
        ax1.set_ylabel("Value (Normalized)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Plot 2D Probability Map (Diffusion Output) - first variable for multivariate
        # Concatenate past and future 2D maps for full context (along width/time axis)
        full_2d = np.concatenate([past_2d, future_2d], axis=1)  # axis=1 is width for 2D arrays
        
        # Use vmin/vmax to ensure consistent color scaling
        im = ax2.imshow(full_2d, aspect='auto', origin='lower', cmap='magma', 
                       interpolation='nearest', vmin=0.0, vmax=1.0)
        mode_label = "PDF/Stripe" if model_config.representation_mode == "pdf" else "CDF/Occupancy"
        ax2.set_title(f"2D Representation - Diffusion Output ({mode_label} mode)", fontsize=14)
        ax2.set_xlabel("Time Steps")
        ax2.set_ylabel("Normalized Value Bins")
        
        # Add vertical line at the forecast start (use past_2d width, not past tensor length)
        ax2.axvline(x=past_2d.shape[1], color='white', linestyle='-', linewidth=2, alpha=0.8, label='Forecast Start')
        
        # Add colorbar
        plt.colorbar(im, ax=ax2, label='Density')
        
        # 3. Plot 2D Guidance Map (if available)
        if has_guidance and guidance_2d is not None:
            # Create a full 2D view by padding the guidance with zeros for the past
            guidance_full_2d = np.zeros_like(full_2d)
            # Note: past_2d.shape[1] is the width of past context
            guidance_cols = guidance_2d.shape[1]
            guidance_full_2d[:, -guidance_cols:] = guidance_2d
            
            im3 = ax3.imshow(guidance_full_2d, aspect='auto', origin='lower', cmap='magma', 
                            interpolation='nearest', vmin=0.0, vmax=1.0)
            ax3.set_title(f"2D Representation - iTransformer Guidance ({mode_label} mode)", fontsize=14)
            ax3.set_xlabel("Time Steps")
            ax3.set_ylabel("Normalized Value Bins")
            
            # Add vertical line at the forecast start
            ax3.axvline(x=past_2d.shape[1], color='white', linestyle='-', linewidth=2, alpha=0.8, label='Forecast Start')
            
            # Add colorbar
            plt.colorbar(im3, ax=ax3, label='Density')
        
        plt.tight_layout()
        
        save_path = f"{output_dir}/sample_{i+1}_full.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        # ===============================================================
        # MULTIVARIATE VISUALIZATION: Grid of all variables
        # ===============================================================
        if num_variables > 1:
            # Determine grid layout: aim for roughly square grid
            n_vars = num_variables
            n_cols = min(4, n_vars)  # Max 4 columns for readability
            n_rows = (n_vars + n_cols - 1) // n_cols  # Ceiling division
            
            # Extract raw data arrays for all variables
            past_np = past.numpy()  # (num_vars, lookback)
            future_np_full = future.numpy()  # (num_vars, forecast)
            pred_np = pred  # (num_vars, forecast)
            
            # ---- 1D Time Series Grid ----
            fig_1d, axes_1d = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            axes_1d = np.atleast_2d(axes_1d).flatten()  # Ensure 1D array of axes
            
            for var_idx in range(n_vars):
                ax = axes_1d[var_idx]
                past_v = past_np[var_idx]
                future_v = future_np_full[var_idx]
                pred_v = pred_np[var_idx] if pred_np.ndim == 2 else pred_np
                
                time_past_v = np.arange(len(past_v))
                time_future_v = np.arange(len(past_v), len(past_v) + len(future_v))
                
                ax.plot(time_past_v[-96:], past_v[-96:], label='Past (last 96)', color='gray', alpha=0.6)
                ax.plot(time_future_v, future_v, label='True', color='blue', linewidth=1.5)
                ax.plot(time_future_v, pred_v, label='Pred', color='red', linestyle='--', linewidth=1.5)
                
                if has_guidance and guidance_pred is not None and guidance_pred.ndim == 2:
                    guidance_v = guidance_pred[var_idx]
                    ax.plot(time_future_v, guidance_v, label='Guide', color='green', linestyle=':', linewidth=1.5)
                
                ax.set_title(f'Variable {var_idx}', fontsize=10)
                ax.grid(True, alpha=0.3)
                if var_idx == 0:
                    ax.legend(fontsize=8, loc='upper right')
            
            # Hide unused subplots
            for var_idx in range(n_vars, len(axes_1d)):
                axes_1d[var_idx].set_visible(False)
            
            fig_1d.suptitle(f'Multivariate Forecast - Sample {i+1} ({n_vars} variables)', fontsize=14)
            plt.tight_layout()
            
            multivar_1d_path = f"{output_dir}/sample_{i+1}_multivar_1d.png"
            plt.savefig(multivar_1d_path, dpi=150)
            plt.close()
            
            # ---- 2D Representation Grid ----
            # Get full 2D representations for all variables
            past_2d_full = out['past_2d'].cpu().squeeze(0).numpy()  # (num_vars, height, width)
            future_2d_full = out['future_2d'].cpu().squeeze(0).numpy()  # (num_vars, height, width)
            
            fig_2d, axes_2d = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))
            axes_2d = np.atleast_2d(axes_2d).flatten()
            
            for var_idx in range(n_vars):
                ax = axes_2d[var_idx]
                
                # Get 2D representation for this variable
                if past_2d_full.ndim == 3:
                    p2d = past_2d_full[var_idx]
                    f2d = future_2d_full[var_idx]
                else:
                    p2d = past_2d_full
                    f2d = future_2d_full
                
                # Convert from diffusion space [-1, 1] to [0, 1]
                p2d = np.clip((p2d + 1.0) / 2.0, 0.0, 1.0)
                f2d = np.clip((f2d + 1.0) / 2.0, 0.0, 1.0)
                
                # Concatenate past and future
                full_2d_v = np.concatenate([p2d, f2d], axis=1)
                
                im = ax.imshow(full_2d_v, aspect='auto', origin='lower', cmap='magma', 
                              interpolation='nearest', vmin=0.0, vmax=1.0)
                ax.axvline(x=p2d.shape[1], color='white', linestyle='-', linewidth=1, alpha=0.8)
                ax.set_title(f'Variable {var_idx}', fontsize=10)
                ax.set_xlabel('Time')
                ax.set_ylabel('Value')
            
            # Hide unused subplots
            for var_idx in range(n_vars, len(axes_2d)):
                axes_2d[var_idx].set_visible(False)
            
            mode_label = "PDF" if model_config.representation_mode == "pdf" else "CDF"
            fig_2d.suptitle(f'2D Representations ({mode_label}) - Sample {i+1} ({n_vars} variables)', fontsize=14)
            plt.tight_layout()
            
            multivar_2d_path = f"{output_dir}/sample_{i+1}_multivar_2d.png"
            plt.savefig(multivar_2d_path, dpi=150)
            plt.close()
            
            print(f"  Saved multivariate visualizations: {multivar_1d_path}, {multivar_2d_path}")
        
        # Calculate and track metrics
        future_np = future.numpy()
        diffusion_mae = np.mean(np.abs(pred - future_np))
        diffusion_rmse = np.sqrt(np.mean((pred - future_np) ** 2))
        diffusion_maes.append(diffusion_mae)
        diffusion_rmses.append(diffusion_rmse)
        
        if has_guidance:
            guidance_mae = np.mean(np.abs(guidance_pred - future_np))
            guidance_rmse = np.sqrt(np.mean((guidance_pred - future_np) ** 2))
            guidance_maes.append(guidance_mae)
            guidance_rmses.append(guidance_rmse)
            print(f"  Sample {i+1}: Diffusion MAE={diffusion_mae:.4f}, RMSE={diffusion_rmse:.4f} | "
                  f"Guidance MAE={guidance_mae:.4f}, RMSE={guidance_rmse:.4f}")
        else:
            print(f"  Sample {i+1}: Diffusion MAE={diffusion_mae:.4f}, RMSE={diffusion_rmse:.4f}")
        
        print(f"  Saved {save_path}")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY - Forecast Error Metrics (lower is better)")
    print("=" * 60)
    print(f"Diffusion Model:")
    print(f"  Mean MAE:  {np.mean(diffusion_maes):.4f} ± {np.std(diffusion_maes):.4f}")
    print(f"  Mean RMSE: {np.mean(diffusion_rmses):.4f} ± {np.std(diffusion_rmses):.4f}")
    
    if guidance_maes:
        print(f"\niTransformer Guidance:")
        print(f"  Mean MAE:  {np.mean(guidance_maes):.4f} ± {np.std(guidance_maes):.4f}")
        print(f"  Mean RMSE: {np.mean(guidance_rmses):.4f} ± {np.std(guidance_rmses):.4f}")
        
        # Calculate improvement
        mae_improvement = (np.mean(guidance_maes) - np.mean(diffusion_maes)) / np.mean(guidance_maes) * 100
        rmse_improvement = (np.mean(guidance_rmses) - np.mean(diffusion_rmses)) / np.mean(guidance_rmses) * 100
        
        print(f"\nDiffusion vs Guidance Improvement:")
        print(f"  MAE:  {mae_improvement:+.1f}% {'(better)' if mae_improvement > 0 else '(worse)'}")
        print(f"  RMSE: {rmse_improvement:+.1f}% {'(better)' if rmse_improvement > 0 else '(worse)'}")
    
    print("=" * 60)


def find_best_model(base_dir: str) -> Optional[str]:
    """Find the best model checkpoint in the directory structure."""
    # 1. Look for best_model.pt and model_best.pt in subdirectories (study folders)
    study_dirs = [d for d in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(d)]
    # Sort by modification time, newest first
    study_dirs.sort(key=os.path.getmtime, reverse=True)

    # Also check the base directory itself for backward compatibility
    search_dirs = study_dirs + [base_dir]

    best_overall_model = None
    min_val_loss = float('inf')

    for d in search_dirs:
        # Priority 1: best_model.pt or model_best.pt in this directory
        best_model_path = os.path.join(d, 'best_model.pt')
        model_best_path = os.path.join(d, 'model_best.pt')

        # Check both naming conventions
        candidate_paths = []
        if os.path.exists(best_model_path):
            candidate_paths.append(best_model_path)
        if os.path.exists(model_best_path):
            candidate_paths.append(model_best_path)

        # Find the newest among the candidates
        if candidate_paths:
            # Sort by modification time (newest first)
            candidate_paths.sort(key=os.path.getmtime, reverse=True)
            chosen_path = candidate_paths[0]
            print(f"Found best model checkpoint in {d}: {os.path.basename(chosen_path)}")
            return chosen_path

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
    import argparse
    parser = argparse.ArgumentParser(description="Visualize Diffusion TSF samples")
    parser.add_argument("--model-path", type=str, default=None, help="Path to checkpoint (.pt). If not set, auto-discover best_model.pt")
    parser.add_argument("--data", type=str, default=os.path.join(script_dir, "../../datasets/electricity/electricity.csv"), help="Path to dataset CSV")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to visualize")
    parser.add_argument("--output-dir", type=str, default="visualizations", help="Directory to save visualizations")
    parser.add_argument("--decoder-method", type=str, choices=["mean", "median", "mode", "beam"], default="mean", help="Decoding method for CDF occupancy maps")
    # Beam search parameters
    parser.add_argument("--beam-width", type=int, default=5, help="Beam width for beam search decoder")
    parser.add_argument("--jump-penalty", type=float, default=1.0, help="Jump penalty scale for beam search decoder")
    parser.add_argument("--search-radius", type=int, default=10, help="Search radius (pixels) for beam search decoder")
    # Guidance parameters
    parser.add_argument("--guidance-checkpoint", type=str, default=None, 
                       help="Path to iTransformer checkpoint for guidance. If model was trained with guidance "
                            "and this is not set, will try to use the checkpoint path from training config.")
    args = parser.parse_args()
    
    # Setup paths
    BASE_CHECKPOINT_DIR = os.path.join(script_dir, "checkpoints")
    
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = find_best_model(BASE_CHECKPOINT_DIR)
    
    if model_path and os.path.exists(model_path):
        visualize_samples(
            model_path,
            args.data,
            num_samples=args.num_samples,
            output_dir=args.output_dir,
            decoder_method=args.decoder_method,
            beam_width=args.beam_width,
            jump_penalty_scale=args.jump_penalty,
            search_radius=args.search_radius,
            guidance_checkpoint=args.guidance_checkpoint
        )
    else:
        print(f"Error: No suitable model checkpoint found (looked for {args.model_path or 'best_model.pt'}).")
        print("Run training first with: python train_electricity.py")

