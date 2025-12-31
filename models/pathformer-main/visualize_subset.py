"""
Visualize Pathformer Model Trained on Subset
Quick script to visualize predictions from subset-trained models

Usage:
    python visualize_subset.py --setting tune_ETTm2_PathFormer_lr0.002_dm32_bs64_drop0.2
"""

import torch
import numpy as np
import argparse
import os
import sys
import json
from datetime import datetime

# Add paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from exp.exp_main import Exp_Main
from shared_utils.visualization import TimeSeriesVisualizer

def load_predictions(exp, setting, flag='test'):
    """Load model and generate predictions"""
    print(f"\nLoading model from checkpoint: {setting}")
    
    # Load the trained model
    checkpoint_path = os.path.join(exp.args.checkpoints, setting, 'checkpoint.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    exp.model.load_state_dict(torch.load(checkpoint_path))
    print(f"✓ Model loaded from: {checkpoint_path}")
    
    # Get test data
    print(f"Loading {flag} data...")
    test_data, test_loader = exp._get_data(flag=flag)
    
    preds = []
    trues = []
    inputs = []
    
    print("Generating predictions...")
    exp.model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float().to(exp.device)
            batch_x_mark = batch_x_mark.float().to(exp.device)
            batch_y_mark = batch_y_mark.float().to(exp.device)

            if exp.args.model == 'PathFormer':
                outputs, balance_loss = exp.model(batch_x)
            else:
                outputs = exp.model(batch_x)
            
            f_dim = -1 if exp.args.features == 'MS' else 0
            outputs = outputs[:, -exp.args.pred_len:, f_dim:]
            batch_y = batch_y[:, -exp.args.pred_len:, f_dim:]
            
            outputs = outputs.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy()
            batch_x = batch_x.detach().cpu().numpy()

            preds.append(outputs)
            trues.append(batch_y)
            inputs.append(batch_x)

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    inputs = np.concatenate(inputs, axis=0)
    
    print(f"✓ Predictions shape: {preds.shape}")
    print(f"✓ Ground truth shape: {trues.shape}")
    print(f"✓ Inputs shape: {inputs.shape}")
    
    return preds, trues, inputs


def compute_metrics(preds, trues):
    """Compute evaluation metrics"""
    mae = np.mean(np.abs(preds - trues))
    mse = np.mean((preds - trues) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((preds - trues) / (trues + 1e-8))) * 100
    
    # Per-feature metrics
    mae_per_feature = np.mean(np.abs(preds - trues), axis=(0, 1))
    mse_per_feature = np.mean((preds - trues) ** 2, axis=(0, 1))
    
    metrics = {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'mape': float(mape),
        'mae_per_feature': mae_per_feature.tolist(),
        'mse_per_feature': mse_per_feature.tolist()
    }
    
    return metrics


def visualize_predictions(preds, trues, inputs, args, setting):
    """Create comprehensive visualizations"""
    
    # Create visualizations directory
    viz_dir = os.path.join('visualizations', setting)
    os.makedirs(viz_dir, exist_ok=True)
    
    print(f"\nCreating visualizations in: {viz_dir}")
    
    # Feature names for ETTm2
    feature_names = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    
    # Initialize visualizer
    visualizer = TimeSeriesVisualizer(
        save_dir=viz_dir,
        feature_names=feature_names
    )
    
    # 1. Sample predictions with input context (first 5 samples)
    print("Creating sample prediction plots...")
    visualizer.plot_time_series_comparison(
        real_data=trues,
        generated_data=preds,
        input_data=inputs,
        num_samples=5,
        save_name='sample_predictions',
        title_prefix='Test Predictions'
    )
    
    # 2. Random sample comparison
    print("Creating random sample comparison...")
    visualizer.plot_random_samples(
        real_data=trues,
        generated_data=preds,
        num_samples=5,
        random_seed=42,
        save_name='random_samples'
    )
    
    # 3. Feature distributions
    print("Creating feature distribution plots...")
    visualizer.plot_feature_distributions(
        real_data=trues,
        generated_data=preds,
        save_name='feature_distributions'
    )
    
    # 4. Statistical summary
    print("Creating statistical summary...")
    metrics = compute_metrics(preds, trues)
    visualizer.plot_statistical_summary(
        real_data=trues,
        generated_data=preds,
        save_name='statistical_summary'
    )
    
    print(f"\n✓ All visualizations saved to: {viz_dir}")
    
    return viz_dir


def main():
    parser = argparse.ArgumentParser(description='Visualize Subset-Trained Pathformer Model')
    
    # Required: checkpoint setting name
    parser.add_argument('--setting', type=str, required=True,
                        help='Checkpoint directory name (e.g., tune_ETTm2_PathFormer_lr0.002_dm32_bs64_drop0.2)')
    
    # Data config
    parser.add_argument('--data', type=str, default='ETTm2')
    parser.add_argument('--root_path', type=str, default='../../datasets/ETT-small/')
    parser.add_argument('--data_path', type=str, default='ETTm2.csv')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--freq', type=str, default='t')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints_subset/')
    
    # Model config (extract from setting name if needed)
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--model', type=str, default='PathFormer')
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--d_ff', type=int, default=128)
    parser.add_argument('--num_nodes', type=int, default=7)
    parser.add_argument('--layer_nums', type=int, default=3)
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--num_experts_list', type=list, default=[4, 4, 4])
    parser.add_argument('--patch_size_list', nargs='+', type=int,
                        default=[16, 12, 8, 32, 12, 8, 6, 4, 8, 6, 4, 2])
    parser.add_argument('--revin', type=int, default=1)
    parser.add_argument('--drop', type=float, default=0.2)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--residual_connection', type=int, default=0)
    parser.add_argument('--batch_norm', type=int, default=0)
    parser.add_argument('--individual', action='store_true', default=False)
    
    # Other
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_multi_gpu', action='store_true', default=False)
    parser.add_argument('--devices', type=str, default='0')
    
    # Auto-extract hyperparameters from setting name
    parser.add_argument('--auto_extract', action='store_true', default=True,
                        help='Automatically extract hyperparameters from setting name')
    
    args = parser.parse_args()
    
    # Extract hyperparameters from setting name if it follows the pattern
    if args.auto_extract and 'lr' in args.setting and 'dm' in args.setting:
        print(f"\nExtracting hyperparameters from setting: {args.setting}")
        parts = args.setting.split('_')
        for part in parts:
            if part.startswith('lr'):
                args.learning_rate = float(part[2:])
                print(f"  learning_rate: {args.learning_rate}")
            elif part.startswith('dm'):
                args.d_model = int(part[2:])
                args.d_ff = args.d_model * 4
                print(f"  d_model: {args.d_model}")
                print(f"  d_ff: {args.d_ff}")
            elif part.startswith('bs'):
                args.batch_size = int(part[2:])
                print(f"  batch_size: {args.batch_size}")
            elif part.startswith('drop'):
                args.drop = float(part[4:])
                print(f"  drop: {args.drop}")
    
    # Setup GPU
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    
    # Setup patch size list
    args.patch_size_list = np.array(args.patch_size_list).reshape(args.layer_nums, -1).tolist()
    
    print("\n" + "="*80)
    print("PATHFORMER VISUALIZATION")
    print("="*80)
    print(f"Setting: {args.setting}")
    print(f"Checkpoint path: {os.path.join(args.checkpoints, args.setting)}")
    print(f"Device: {'GPU' if args.use_gpu else 'CPU'}")
    
    # Create experiment instance
    exp = Exp_Main(args)
    
    # Generate predictions
    preds, trues, inputs = load_predictions(exp, args.setting, flag='test')
    
    # Compute and display metrics
    print("\n" + "="*80)
    print("METRICS")
    print("="*80)
    metrics = compute_metrics(preds, trues)
    print(f"MAE:  {metrics['mae']:.6f}")
    print(f"MSE:  {metrics['mse']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"MAPE: {metrics['mape']:.2f}%")
    
    print("\nPer-feature MAE:")
    feature_names = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    for name, mae in zip(feature_names, metrics['mae_per_feature']):
        print(f"  {name}: {mae:.6f}")
    
    # Save metrics
    metrics_file = os.path.join(args.checkpoints, args.setting, 'test_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Metrics saved to: {metrics_file}")
    
    # Create visualizations
    viz_dir = visualize_predictions(preds, trues, inputs, args, args.setting)
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE!")
    print("="*80)
    print(f"Check the '{viz_dir}' folder for all visualizations")
    
    return 0


if __name__ == '__main__':
    exit(main())
