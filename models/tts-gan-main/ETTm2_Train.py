#!/usr/bin/env python
"""
ETTm2 GAN Training Script
Main entry point for training TTS-GAN on ETTm2 dataset
"""

import os
import sys

# Configuration for ETTm2 training
# Note: These parameters must match the arguments defined in cfg.py
config = {
    # Training hyperparameters
    'batch_size': 64,           # Batch size (mapped to -bs)
    'gen_batch_size': 64,       # Generator batch size (mapped to -gen_bs)
    'dis_batch_size': 64,       # Discriminator batch size (mapped to -dis_bs)
    'max_epoch': 200,           # Maximum epochs
    'max_iter': None,           # Maximum iterations (None for epoch-based)
    'n_critic': 1,              # Discriminator updates per generator update
    'accumulated_times': 1,     # Gradient accumulation steps
    'g_accumulated_times': 1,   # Generator gradient accumulation
    
    # Optimizer settings
    'optimizer': 'adamw',       # Optimizer type: 'adam' or 'adamw'
    'g_lr': 0.0001,            # Generator learning rate
    'd_lr': 0.0003,            # Discriminator learning rate
    'beta1': 0.0,              # Adam beta1
    'beta2': 0.9,              # Adam beta2
    'wd': 1e-3,                # Weight decay
    'lr_decay': True,          # Whether to use learning rate decay
    
    # Loss settings
    'loss': 'lsgan',           # Loss type: 'hinge', 'standard', 'lsgan', 'wgangp'
    'phi': 1,                  # Phi parameter for WGANGP
    
    # Model hyperparameters (custom, handled in train_ettm2.py)
    'latent_dim': 100,          # Latent dimension for generator
    'dropout': 0.5,             # Dropout rate
    'patch_size': 12,           # Patch size for transformer
    
    # Data settings
    'data_path': '../../datasets/ETT-small/ETTm2.csv',
    'num_workers': 4,          # Number of data loading workers
    
    # Logging and checkpointing
    'exp_name': 'ettm2_gan',   # Experiment name
    'val_freq': 20,            # Validation frequency
    'print_freq': 50,          # Print frequency
    'show': False,             # Show samples during training
    'early_stop': False,       # Enable early stopping
    'early_stop_patience': 20, # Number of epochs without improvement before stopping
    
    # Initialization
    'init_type': 'xavier_uniform',  # Weight initialization
    'seed': 12345,             # Random seed (note: cfg.py uses 'seed' but might use random_seed)
    
    # Distributed training (set to defaults for single GPU)
    'gpu': 0,                  # GPU id (None for CPU)
    'rank': 0,                 # Process rank
    'world-size': 1,           # Number of processes (note: uses hyphen in cfg.py)
    'dist-url': 'tcp://localhost:4321',  # Note: uses hyphen in cfg.py
    'dist-backend': 'nccl',    # Note: uses hyphen in cfg.py
    'multiprocessing-distributed': False,  # Note: uses hyphen in cfg.py
    
    # Other
    'load_path': None,         # Path to checkpoint for resuming
    'grow_steps': [0, 0],      # Growth steps for progressive training
}

def run_training():
    """Run the training with configured parameters"""
    
    # Build command
    cmd_parts = ['python', 'train_ettm2.py']
    
    # Add all configuration parameters
    for key, value in config.items():
        if value is None:
            continue
        
        # Format parameter name (key already has correct format with _ or -)
        param_name = '--' + key
        
        # Handle different value types
        if isinstance(value, bool):
            if value:
                cmd_parts.append(param_name)
        elif isinstance(value, list):
            cmd_parts.append(param_name)
            cmd_parts.extend([str(v) for v in value])
        else:
            cmd_parts.append(param_name)
            cmd_parts.append(str(value))
    
    # Join and execute
    cmd = ' '.join(cmd_parts)
    print("=" * 80)
    print("Training TTS-GAN on ETTm2 Dataset")
    print("=" * 80)
    print(f"\nCommand: {cmd}\n")
    print("=" * 80)
    
    os.system(cmd)

if __name__ == '__main__':
    run_training()
