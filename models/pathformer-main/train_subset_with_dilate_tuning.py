"""
Train Pathformer on ETTm2 Subset with DILATE Loss Hyperparameter Tuning

Tunes DILATE-specific hyperparameters:
- alpha: Weight between shape loss (soft-DTW) and temporal loss (0-1)
- gamma: Smoothing parameter for soft-DTW
- learning_rate: Optimizer learning rate  
- drop: Dropout rate

Usage:
    python train_subset_with_dilate_tuning.py --train_subset 0.25 --tune_subset 0.10
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

from train_etth2 import Exp_Main_With_Dilate
from torch.utils.data import DataLoader, Subset

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


class SubsetExp_Dilate(Exp_Main_With_Dilate):
    """Extended Exp_Main_With_Dilate that supports training on data subsets"""
    
    def __init__(self, args):
        self.train_subset_ratio = getattr(args, 'train_subset', 1.0)
        self.tune_subset_ratio = getattr(args, 'tune_subset', 0.1)
        super().__init__(args)
        
    def _get_data(self, flag):
        """Override to support subset selection"""
        from data_provider.data_factory import data_provider
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


def quick_dilate_hyperparameter_search(base_args, tune_subset_ratio=0.1):
    """
    Perform quick hyperparameter search for DILATE loss
    
    Tunes DILATE-specific hyperparameters:
    - alpha: Shape vs temporal loss weight (0-1)
           alpha=0.5: Equal weight to shape and temporal
           alpha=0.7: More weight on shape (soft-DTW)
           alpha=0.3: More weight on temporal alignment
    - gamma: Soft-DTW smoothing parameter (smaller = stricter matching)
    - learning_rate: Optimization speed
    - drop: Dropout regularization
    """
    
    print("\n" + "="*80)
    print("STARTING DILATE HYPERPARAMETER TUNING")
    print("="*80)
    
    # Define hyperparameter search space focused on DILATE params
    param_grid = {
        'dilate_alpha': [0.6, 0.9],      # Shape vs temporal weight
        'dilate_gamma': [0.01, 0.1],               # Soft-DTW smoothing (removed 0.001, too strict)
        'learning_rate': [0.001, 0.002],   # Lower LRs for stability
        'drop': [0.15],
    }
    
    # Fixed parameters
    base_args.d_model = 32
    base_args.d_ff = 128
    base_args.batch_size = 64
    base_args.loss_type = 'dilate'  # Use DILATE loss
    
    # Create tuning args with very short training
    tune_args = argparse.Namespace(**vars(base_args))
    tune_args.train_subset = tune_subset_ratio
    tune_args.train_epochs = 3  # Slightly longer for DILATE to stabilize
    tune_args.patience = 10  # Don't stop early during tuning
    
    # Load existing results if tuning was interrupted
    os.makedirs(base_args.checkpoints, exist_ok=True)
    results_file = os.path.join(base_args.checkpoints, 'dilate_tuning_results.json')
    completed_configs = set()
    results = []
    best_config = None
    best_val_loss = float('inf')
    
    if os.path.exists(results_file):
        print(f"\n✓ Found existing tuning results: {results_file}")
        with open(results_file, 'r') as f:
            saved_data = json.load(f)
            results = saved_data.get('all_results', [])
            best_config = saved_data.get('best_config')
            best_val_loss = saved_data.get('best_val_loss', float('inf'))
            
            # Track which configs were already tested
            for r in results:
                config_tuple = tuple(sorted(r['config'].items()))
                completed_configs.add(config_tuple)
        
        print(f"✓ Loaded {len(results)} previous results (best val loss: {best_val_loss:.6f})")
        print("  Resuming from where we left off...\n")
    
    # Generate all combinations and RANDOMIZE order
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(product(*values))
    
    # RANDOMIZE: Shuffle to explore diverse configs quickly
    random.shuffle(combinations)
    
    total_configs = len(combinations)
    remaining_configs = [c for c in combinations if tuple(sorted(dict(zip(keys, c)).items())) not in completed_configs]
    
    print(f"\nTesting {len(remaining_configs)}/{total_configs} DILATE configurations (randomized order)")
    print(f"  on {tune_subset_ratio*100:.1f}% of training data")
    print(f"  Each configuration will train for {tune_args.train_epochs} epochs")
    print(f"\nDILATE Loss Components:")
    print(f"  - Shape Loss (soft-DTW): Measures shape similarity")
    print(f"  - Temporal Loss: Penalizes time misalignment")
    print(f"  - Total Loss = alpha * Shape + (1-alpha) * Temporal\n")
    
    if len(remaining_configs) == 0:
        print("All configurations already tested!")
        if best_config:
            return best_config
        else:
            return {}
    
    for idx, combo in enumerate(remaining_configs, 1):
        # Set hyperparameters
        config = dict(zip(keys, combo))
        for key, value in config.items():
            setattr(tune_args, key, value)
        
        # Update dependent parameters
        tune_args.d_ff = tune_args.d_model * 4
        
        print(f"\n[{idx}/{len(remaining_configs)}] Testing config (overall progress: {len(results)+1}/{total_configs}):")
        print(f"  alpha={config['dilate_alpha']:.1f} (shape weight), gamma={config['dilate_gamma']:.3f} (smoothing)")
        print(f"  lr={config['learning_rate']:.4f}, drop={config['drop']:.2f}")
        
        # Create experiment instance
        exp = SubsetExp_Dilate(tune_args)
        
        # Create setting name for checkpoints
        setting = 'dilate_tune_a{}_g{}_lr{}_drop{}'.format(
            config['dilate_alpha'],
            config['dilate_gamma'],
            config['learning_rate'],
            config['drop']
        )
        
        try:
            # Train briefly
            exp.train(setting)
            
            # Get validation data and evaluate
            vali_data, vali_loader = exp._get_data(flag='val')
            criterion = exp._select_criterion()
            val_loss = exp.vali(vali_data, vali_loader, criterion)
            
            print(f"  Validation Loss: {val_loss:.6f}")
            
            # Track results
            result = {
                'config': config,
                'val_loss': float(val_loss),
                'setting': setting
            }
            results.append(result)
            
            # Update best
            if val_loss < best_val_loss:
                best_val_loss = float(val_loss)
                best_config = config
                print(f"  ★ New best configuration! Val Loss: {val_loss:.6f}")
            
            # SAVE AFTER EVERY CONFIG (resume-safe)
            with open(results_file, 'w') as f:
                json.dump({
                    'best_config': best_config,
                    'best_val_loss': best_val_loss,
                    'all_results': results,
                    'completed': len(results),
                    'total': total_configs
                }, f, indent=2)
            print(f"  ✓ Progress saved to {results_file}")
                
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Still save progress even on error
            result = {
                'config': config,
                'val_loss': None,
                'setting': setting,
                'error': str(e)
            }
            results.append(result)
            with open(results_file, 'w') as f:
                json.dump({
                    'best_config': best_config,
                    'best_val_loss': best_val_loss,
                    'all_results': results,
                    'completed': len(results),
                    'total': total_configs
                }, f, indent=2)
            continue
    
    # Print summary
    print("\n" + "="*80)
    print("DILATE HYPERPARAMETER TUNING COMPLETE")
    print("="*80)
    print(f"Tested {len(results)}/{total_configs} configurations")
    
    # Filter out failed configs
    successful_results = [r for r in results if r.get('val_loss') is not None]
    print(f"Successful: {len(successful_results)}, Failed: {len(results) - len(successful_results)}")
    
    if best_config:
        print(f"\nBest Configuration (Val Loss: {best_val_loss:.6f}):")
        print(f"  alpha (shape weight): {best_config['dilate_alpha']}")
        print(f"  gamma (smoothing): {best_config['dilate_gamma']}")
        print(f"  learning_rate: {best_config['learning_rate']}")
        print(f"  drop: {best_config['drop']}")
        
        # Interpret alpha
        if best_config['dilate_alpha'] > 0.6:
            print(f"\n  → Shape similarity (soft-DTW) is more important")
        elif best_config['dilate_alpha'] < 0.4:
            print(f"\n  → Temporal alignment is more important")
        else:
            print(f"\n  → Balanced between shape and temporal")
        
        print(f"\nAll results saved to: {results_file}")
    else:
        print("\nWARNING: No successful configurations found!")
        best_config = {}
    
    return best_config


def train_with_best_dilate_config(base_args, best_config, train_subset_ratio):
    """
    Train the model on the specified subset with the best DILATE hyperparameters
    """
    print("\n" + "="*80)
    print("STARTING FINAL TRAINING WITH BEST DILATE CONFIG")
    print("="*80)
    
    # Apply best hyperparameters
    for key, value in best_config.items():
        setattr(base_args, key, value)
    
    # Update dependent parameters
    base_args.d_ff = base_args.d_model * 4
    base_args.train_subset = train_subset_ratio
    base_args.loss_type = 'dilate'
    base_args.test_flop = False  # Add missing attribute for test() method
    
    print(f"\nTraining on {train_subset_ratio*100:.1f}% of training data")
    print(f"DILATE Configuration:")
    print(f"  alpha: {best_config['dilate_alpha']} (shape vs temporal weight)")
    print(f"  gamma: {best_config['dilate_gamma']} (soft-DTW smoothing)")
    print(f"  learning_rate: {best_config['learning_rate']}")
    print(f"  drop: {best_config['drop']}\n")
    
    # Create experiment
    exp = SubsetExp_Dilate(base_args)
    
    # Create setting name
    setting = 'final_dilate_subset{}_a{}_g{}_lr{}_drop{}'.format(
        int(train_subset_ratio*100),
        best_config['dilate_alpha'],
        best_config['dilate_gamma'],
        best_config['learning_rate'],
        best_config['drop']
    )
    
    # Train
    exp.train(setting)
    
    # Test
    print("\nEvaluating on test set...")
    exp.test(setting)
    
    print(f"\nFinal training complete! Model saved in: {os.path.join(base_args.checkpoints, setting)}")


def main():
    parser = argparse.ArgumentParser(description='Train Pathformer with DILATE Loss - Hyperparameter Tuning')
    
    # Subset control
    parser.add_argument('--train_subset', type=float, default=0.25,
                        help='Percentage of training data to use for final training (0.0-1.0)')
    parser.add_argument('--tune_subset', type=float, default=0.10,
                        help='Percentage of training data to use for hyperparameter tuning (0.0-1.0)')
    parser.add_argument('--skip_tuning', action='store_true',
                        help='Skip hyperparameter tuning and use default/specified params')
    
    # Data config (ETTh2 for compatibility with train_etth2.py)
    parser.add_argument('--data', type=str, default='ETTh2', help='dataset type')
    parser.add_argument('--root_path', type=str, default='../../datasets/ETT-small/',
                        help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh2.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints_dilate_subset/',
                        help='location of model checkpoints')
    
    # Forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    
    # DILATE loss parameters (will be tuned)
    parser.add_argument('--loss_type', type=str, default='dilate', help='loss type: dilate, mse, or mae')
    parser.add_argument('--dilate_alpha', type=float, default=0.5, 
                        help='DILATE alpha: weight for shape loss vs temporal (0-1)')
    parser.add_argument('--dilate_gamma', type=float, default=0.01,
                        help='DILATE gamma: smoothing parameter for soft-DTW')
    
    # Model hyperparameters (defaults, some will be tuned)
    parser.add_argument('--model', type=str, default='PathFormer', help='model name')
    parser.add_argument('--d_model', type=int, default=32, help='dimension of model')
    parser.add_argument('--d_ff', type=int, default=128, help='dimension of fcn')
    parser.add_argument('--num_nodes', type=int, default=7, help='number of features')
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
    parser.add_argument('--train_epochs', type=int, default=15, help='train epochs for final training')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
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
    print("PATHFORMER WITH DILATE LOSS - SUBSET TRAINING & TUNING")
    print("="*80)
    print(f"\nDataset: {args.data}")
    print(f"Data path: {os.path.join(args.root_path, args.data_path)}")
    print(f"Training subset: {args.train_subset*100:.1f}%")
    print(f"Tuning subset: {args.tune_subset*100:.1f}%")
    print(f"Loss type: DILATE (shape + temporal)")
    print(f"Device: {'GPU' if args.use_gpu else 'CPU'}")
    
    # Hyperparameter tuning phase
    if not args.skip_tuning:
        best_config = quick_dilate_hyperparameter_search(args, args.tune_subset)
    else:
        print("\nSkipping hyperparameter tuning, using specified configuration")
        best_config = {
            'dilate_alpha': args.dilate_alpha,
            'dilate_gamma': args.dilate_gamma,
            'learning_rate': args.learning_rate,
            'drop': args.drop,
        }
    
    # Final training phase with best config
    if best_config:
        train_with_best_dilate_config(args, best_config, args.train_subset)
    else:
        print("\nERROR: No valid configuration found, cannot proceed with final training")
        return 1
    
    print("\n" + "="*80)
    print("ALL DONE!")
    print("="*80)
    return 0


if __name__ == '__main__':
    exit(main())
