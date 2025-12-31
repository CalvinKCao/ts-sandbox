"""
Visualize Pathformer Predictions on ETTh2 Dataset
"""

import torch
import numpy as np
import argparse
import os
import sys
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from train_etth2 import Exp_Main_With_Dilate
from shared_utils.visualization import TimeSeriesVisualizer


def load_predictions(exp, setting, flag='test'):
    """Load model and generate predictions"""
    print(f"Loading model from checkpoint: {setting}")
    
    # Load the trained model
    checkpoint_path = os.path.join(exp.args.checkpoints, setting, 'checkpoint.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    exp.model.load_state_dict(torch.load(checkpoint_path))
    
    # Get test data
    test_data, test_loader = exp._get_data(flag=flag)
    
    preds = []
    trues = []
    inputs = []
    
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
    
    print(f"Predictions shape: {preds.shape}")
    print(f"Ground truth shape: {trues.shape}")
    print(f"Inputs shape: {inputs.shape}")
    
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
    
    # ETTh2 feature names
    feature_names = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    
    # Create visualization directory
    vis_dir = os.path.join('./visualizations', setting)
    os.makedirs(vis_dir, exist_ok=True)
    
    print(f"\nCreating visualizations in {vis_dir}...")
    
    # Initialize visualizer
    visualizer = TimeSeriesVisualizer(
        save_dir=vis_dir,
        feature_names=feature_names
    )
    
    # Generate comprehensive report (without input for general plots)
    visualizer.generate_full_report(
        real_data=trues,
        generated_data=preds,
        num_samples=5,
        random_seed=42,
        report_name='etth2_predictions'
    )
    
    # Create additional plots WITH input/lookback window
    print("\nCreating plots with lookback window...")
    visualizer.plot_time_series_comparison(
        real_data=trues,
        generated_data=preds,
        input_data=inputs,
        num_samples=5,
        random_seed=42,
        save_name='etth2_with_lookback',
        title_prefix='With Lookback: '
    )
    
    # Create additional specific visualizations with lookback
    print("\nCreating additional visualizations...")
    
    # First 3 samples with lookback
    visualizer.plot_time_series_comparison(
        real_data=trues,
        generated_data=preds,
        input_data=inputs,
        specific_indices=[0, 1, 2],
        num_samples=3,
        save_name='first_3_samples_with_lookback',
        title_prefix='First 3 Samples: '
    )
    
    # Last 3 samples with lookback
    n_samples = trues.shape[0]
    visualizer.plot_time_series_comparison(
        real_data=trues,
        generated_data=preds,
        input_data=inputs,
        specific_indices=[n_samples-3, n_samples-2, n_samples-1],
        num_samples=3,
        save_name='last_3_samples_with_lookback',
        title_prefix='Last 3 Samples: '
    )
    
    print(f"\nAll visualizations saved to {vis_dir}")


def main():
    parser = argparse.ArgumentParser(description='Visualize Pathformer Predictions on ETTh2')

    # Basic config
    parser.add_argument('--model', type=str, default='PathFormer')
    parser.add_argument('--model_id', type=str, default='ETTh2')
    parser.add_argument('--setting', type=str, default=None,
                        help='Specific checkpoint setting to load (if None, will construct from other args)')

    # Data loader
    parser.add_argument('--data', type=str, default='ETTh2')
    parser.add_argument('--root_path', type=str, default='../../datasets/ETT-small/')
    parser.add_argument('--data_path', type=str, default='ETTh2.csv')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

    # Forecasting task
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--individual', action='store_true', default=False)

    # Model parameters
    parser.add_argument('--d_model', type=int, default=4)
    parser.add_argument('--d_ff', type=int, default=64)
    parser.add_argument('--num_nodes', type=int, default=7)
    parser.add_argument('--layer_nums', type=int, default=3)
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--num_experts_list', type=list, default=[4, 4, 4])
    parser.add_argument('--patch_size_list', nargs='+', type=int,
                        default=[16, 12, 8, 32, 12, 8, 6, 4, 8, 6, 4, 2])
    parser.add_argument('--do_predict', action='store_true', default=False)
    parser.add_argument('--revin', type=int, default=1)
    parser.add_argument('--drop', type=float, default=0.1)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--residual_connection', type=int, default=0)
    parser.add_argument('--metric', type=str, default='mae')
    parser.add_argument('--batch_norm', type=int, default=0)

    # Loss function (for setting name)
    parser.add_argument('--loss_type', type=str, default='mae')
    parser.add_argument('--dilate_alpha', type=float, default=0.5)
    parser.add_argument('--dilate_gamma', type=float, default=0.01)
    parser.add_argument('--freq_threshold', type=float, default=80.0)
    
    # Optimization (needed for model compatibility)
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--train_epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--lradj', type=str, default='TST')
    parser.add_argument('--use_amp', action='store_true', default=False)
    parser.add_argument('--pct_start', type=float, default=0.4)

    # Other
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_multi_gpu', action='store_true', default=False)
    parser.add_argument('--devices', type=str, default='0')

    # Visualization options
    parser.add_argument('--num_samples', type=int, default=5, 
                        help='Number of random samples to visualize')
    parser.add_argument('--save_metrics', action='store_true', 
                        help='Save metrics to JSON file')

    args = parser.parse_args()
    
    # Set device
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    args.patch_size_list = np.array(args.patch_size_list).reshape(args.layer_nums, -1).tolist()

    # Construct setting name if not provided
    if args.setting is None:
        loss_suffix = f"_{args.loss_type}"
        if args.loss_type == 'dilate':
            loss_suffix += f"_a{args.dilate_alpha}_g{args.dilate_gamma}"
        
        setting = '{}_{}_ft{}_sl{}_pl{}_0{}'.format(
            args.model_id,
            args.model,
            args.features,
            args.seq_len,
            args.pred_len,
            loss_suffix
        )
    else:
        setting = args.setting

    print('='*50)
    print(f'Visualizing predictions for: {setting}')
    print('='*50)
    print(args)
    print('='*50)

    # Create experiment instance
    exp = Exp_Main_With_Dilate(args)

    # Load predictions
    preds, trues, inputs = load_predictions(exp, setting, flag='test')

    # Compute metrics
    print("\nComputing evaluation metrics...")
    metrics = compute_metrics(preds, trues)
    
    print("\n" + "="*50)
    print("EVALUATION METRICS")
    print("="*50)
    print(f"MAE:  {metrics['mae']:.4f}")
    print(f"MSE:  {metrics['mse']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAPE: {metrics['mape']:.2f}%")
    print("\nPer-feature MAE:")
    feature_names = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    for i, (name, mae) in enumerate(zip(feature_names, metrics['mae_per_feature'])):
        print(f"  {name}: {mae:.4f}")
    print("="*50)

    # Save metrics to file if requested
    if args.save_metrics:
        metrics_file = os.path.join('./visualizations', setting, 'metrics.json')
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to {metrics_file}")

    # Create visualizations
    visualize_predictions(preds, trues, inputs, args, setting)

    print("\n" + "="*50)
    print("VISUALIZATION COMPLETE!")
    print("="*50)


if __name__ == '__main__':
    main()
