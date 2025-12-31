"""
Train Pathformer on a Subset of ETTm2 with Quick Hyperparameter Tuning

This script trains the Pathformer model on a specified percentage of the ETTm2 training data
and performs quick hyperparameter tuning on an even smaller subset.

Usage:
    python train_subset_with_tuning.py --train_subset 0.25 --tune_subset 0.10
"""

import torch
import numpy as np
import random
import argparse
import time
import os
import sys
import json
from itertools import product

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp.exp_main import Exp_Main
from torch.utils.data import DataLoader, Subset
from data_provider.data_factory import data_provider

fix_seed = 1024
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


class SubsetWrapper:
    """Wrapper to preserve dataset methods when using torch Subset"""
    def __init__(self, dataset, indices):
        self.dataset = dataset if not isinstance(dataset, Subset) else dataset.dataset
        self.indices = indices
        
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]
    
    def __len__(self):
        return len(self.indices)
    
    def inverse_transform(self, data):
        """Preserve inverse_transform method from original dataset"""
        if hasattr(self.dataset, 'inverse_transform'):
            return self.dataset.inverse_transform(data)
        return data


class SubsetExp_Main(Exp_Main):
    """Extended Exp_Main that supports training on data subsets"""
    
    def __init__(self, args):
        super().__init__(args)
        self.train_subset_ratio = getattr(args, 'train_subset', 1.0)
        self.tune_subset_ratio = getattr(args, 'tune_subset', 0.1)
        
    def _get_data(self, flag):
        """Override to support subset selection"""
        data_set, data_loader = data_provider(self.args, flag)
        
        # Apply subsetting for training data if needed
        if flag == 'train' and self.train_subset_ratio < 1.0:
            original_size = len(data_set)
            subset_size = int(original_size * self.train_subset_ratio)
            
            # Create random subset indices
            indices = torch.randperm(original_size)[:subset_size].tolist()
            
            # Use custom wrapper to preserve inverse_transform
            data_set = SubsetWrapper(data_set, indices)
            
            # Recreate dataloader with subset
            data_loader = DataLoader(
                data_set,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.num_workers,
                drop_last=True
            )
            
            print(f"Using {subset_size}/{original_size} training samples ({self.train_subset_ratio*100:.1f}%)")
        
        return data_set, data_loader


def quick_hyperparameter_search(base_args, tune_subset_ratio=0.1):
    """
    Perform quick hyperparameter search on a small subset
    
    Tunes the most important hyperparameters:
    - learning_rate: Controls optimization speed
    - d_model: Model embedding dimension
    - batch_size: Training batch size
    - drop: Dropout rate
    """
    
    print("\n" + "="*80)
    print("STARTING QUICK HYPERPARAMETER TUNING")
    print("="*80)
    
    # Define hyperparameter search space (focused on most important params)
    param_grid = {
        'learning_rate': [0.002, 0.005, 0.01, 0.03],
        'd_model': [32],
        'batch_size': [64],
        'drop': [0.05, 0.1, 0.2, 0.4],
    }
    
    # Create tuning args with very short training
    tune_args = argparse.Namespace(**vars(base_args))
    tune_args.train_subset = tune_subset_ratio
    tune_args.train_epochs = 2  # Very short for quick tuning
    tune_args.patience = 10  # Don't stop early during tuning
    
    best_config = None
    best_val_loss = float('inf')
    results = []
    
    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(product(*values))
    
    total_configs = len(combinations)
    print(f"\nTesting {total_configs} hyperparameter configurations on {tune_subset_ratio*100:.1f}% of training data")
    print(f"Each configuration will train for {tune_args.train_epochs} epochs\n")
    
    for idx, combo in enumerate(combinations, 1):
        # Set hyperparameters
        config = dict(zip(keys, combo))
        for key, value in config.items():
            setattr(tune_args, key, value)
        
        # Update dependent parameters
        tune_args.d_ff = tune_args.d_model * 4  # Standard ratio
        
        print(f"\n[{idx}/{total_configs}] Testing config: {config}")
        
        # Create experiment instance
        exp = SubsetExp_Main(tune_args)
        
        # Create setting name for checkpoints
        setting = 'tune_{}_{}_lr{}_dm{}_bs{}_drop{}'.format(
            tune_args.data,
            tune_args.model,
            tune_args.learning_rate,
            tune_args.d_model,
            tune_args.batch_size,
            tune_args.drop
        )
        
        try:
            # Train briefly
            exp.train(setting)
            
            # Get validation data and evaluate
            vali_data, vali_loader = exp._get_data(flag='val')
            criterion = exp._select_criterion()
            val_loss = exp.vali(vali_data, vali_loader, criterion)
            
            print(f"Validation Loss: {val_loss:.6f}")
            
            # Track results
            result = {
                'config': config,
                'val_loss': val_loss,
                'setting': setting
            }
            results.append(result)
            
            # Update best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_config = config
                print(f"★ New best configuration! Val Loss: {val_loss:.6f}")
                
        except Exception as e:
            print(f"ERROR with config {config}: {str(e)}")
            continue
    
    # Print summary
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING COMPLETE")
    print("="*80)
    
    if best_config:
        print(f"\nBest Configuration (Val Loss: {best_val_loss:.6f}):")
        for key, value in best_config.items():
            print(f"  {key}: {value}")
        
        # Save results to file
        results_file = os.path.join(base_args.checkpoints, 'tuning_results.json')
        with open(results_file, 'w') as f:
            json.dump({
                'best_config': best_config,
                'best_val_loss': best_val_loss,
                'all_results': results
            }, f, indent=2)
        print(f"\nResults saved to: {results_file}")
    else:
        print("\nWARNING: No successful configurations found!")
        best_config = {}
    
    return best_config


def train_with_best_config(base_args, best_config, train_subset_ratio):
    """
    Train the model on the specified subset with the best hyperparameters
    """
    print("\n" + "="*80)
    print("STARTING FINAL TRAINING WITH BEST CONFIG")
    print("="*80)
    
    # Apply best hyperparameters
    for key, value in best_config.items():
        setattr(base_args, key, value)
    
    # Update dependent parameters
    base_args.d_ff = base_args.d_model * 4
    base_args.train_subset = train_subset_ratio
    
    print(f"\nTraining on {train_subset_ratio*100:.1f}% of training data")
    print(f"Configuration: {best_config}\n")
    
    # Create experiment
    exp = SubsetExp_Main(base_args)
    
    # Create setting name
    setting = 'final_ETTm2_PathFormer_subset{}_lr{}_dm{}_bs{}_drop{}'.format(
        int(train_subset_ratio*100),
        base_args.learning_rate,
        base_args.d_model,
        base_args.batch_size,
        base_args.drop
    )
    
    # Train
    exp.train(setting)
    
    # Test
    print("\nEvaluating on test set...")
    exp.test(setting)
    
    print(f"\nFinal training complete! Model saved in: {os.path.join(base_args.checkpoints, setting)}")


def main():
    parser = argparse.ArgumentParser(description='Train Pathformer on ETTm2 Subset with Hyperparameter Tuning')
    
    # Subset control
    parser.add_argument('--train_subset', type=float, default=0.25,
                        help='Percentage of training data to use for final training (0.0-1.0)')
    parser.add_argument('--tune_subset', type=float, default=0.10,
                        help='Percentage of training data to use for hyperparameter tuning (0.0-1.0)')
    parser.add_argument('--skip_tuning', action='store_true',
                        help='Skip hyperparameter tuning and use default/specified params')
    
    # Data config
    parser.add_argument('--data', type=str, default='ETTm2', help='dataset type')
    parser.add_argument('--root_path', type=str, default='../../datasets/ETT-small/',
                        help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTm2.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='t', help='freq for time features encoding')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints_subset/',
                        help='location of model checkpoints')
    
    # Forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    
    # Model hyperparameters (defaults, will be tuned)
    parser.add_argument('--model', type=str, default='PathFormer', help='model name')
    parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
    parser.add_argument('--d_ff', type=int, default=64, help='dimension of fcn')
    parser.add_argument('--num_nodes', type=int, default=7, help='number of features in ETTm2')
    parser.add_argument('--layer_nums', type=int, default=3, help='num layers')
    parser.add_argument('--k', type=int, default=2, help='choose the Top K patch size at every layer')
    parser.add_argument('--num_experts_list', type=list, default=[4, 4, 4])
    parser.add_argument('--patch_size_list', nargs='+', type=int,
                        default=[16,12,8,32,12,8,6,4,8,6,4,2])
    parser.add_argument('--revin', type=int, default=1, help='whether to apply RevIN')
    parser.add_argument('--drop', type=float, default=0.1, help='dropout ratio')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')
    parser.add_argument('--residual_connection', type=int, default=0)
    parser.add_argument('--metric', type=str, default='mae')
    parser.add_argument('--batch_norm', type=int, default=0)
    parser.add_argument('--individual', action='store_true', default=False)
    
    # Training
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs for final training')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--lradj', type=str, default='TST', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', default=False, help='use automatic mixed precision')
    parser.add_argument('--pct_start', type=float, default=0.4, help='pct_start')
    
    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False, help='use multiple gpus')
    parser.add_argument('--devices', type=str, default='0', help='device ids of multiple gpus')
    
    args = parser.parse_args()
    
    # Setup GPU
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    
    # Setup patch size list
    args.patch_size_list = np.array(args.patch_size_list).reshape(args.layer_nums, -1).tolist()
    
    # Validate subset ratios
    if not (0.0 < args.train_subset <= 1.0):
        raise ValueError("train_subset must be between 0.0 and 1.0")
    if not (0.0 < args.tune_subset <= 1.0):
        raise ValueError("tune_subset must be between 0.0 and 1.0")
    
    print("\n" + "="*80)
    print("PATHFORMER SUBSET TRAINING WITH HYPERPARAMETER TUNING")
    print("="*80)
    print(f"\nDataset: {args.data}")
    print(f"Data path: {os.path.join(args.root_path, args.data_path)}")
    print(f"Training subset: {args.train_subset*100:.1f}%")
    print(f"Tuning subset: {args.tune_subset*100:.1f}%")
    print(f"Sequence length: {args.seq_len}, Prediction length: {args.pred_len}")
    print(f"Device: {'GPU' if args.use_gpu else 'CPU'}")
    
    # Hyperparameter tuning phase
    if not args.skip_tuning:
        best_config = quick_hyperparameter_search(args, args.tune_subset)
    else:
        print("\nSkipping hyperparameter tuning, using specified configuration")
        best_config = {
            'learning_rate': args.learning_rate,
            'd_model': args.d_model,
            'batch_size': args.batch_size,
            'drop': args.drop,
        }
    
    # Final training phase with best config
    if best_config:
        train_with_best_config(args, best_config, args.train_subset)
    else:
        print("\nERROR: No valid configuration found, cannot proceed with final training")
        return 1
    
    print("\n" + "="*80)
    print("ALL DONE!")
    print("="*80)
    return 0


if __name__ == '__main__':
    exit(main())
