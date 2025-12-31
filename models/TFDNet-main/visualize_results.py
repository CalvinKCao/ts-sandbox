import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def visualize(args):
    folder_path = args.folder_path
    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} does not exist.")
        return

    pred_path = os.path.join(folder_path, 'pred.npy')
    true_path = os.path.join(folder_path, 'true.npy')
    x_path = os.path.join(folder_path, 'x.npy')
    
    pred_seasonal_path = os.path.join(folder_path, 'pred_seasonal.npy')
    true_seasonal_path = os.path.join(folder_path, 'true_seasonal.npy')
    true_trend_path = os.path.join(folder_path, 'true_trend.npy')

    if not os.path.exists(pred_path) or not os.path.exists(true_path):
        print(f"Error: pred.npy or true.npy not found in {folder_path}")
        return

    preds = np.load(pred_path)
    trues = np.load(true_path)
    
    has_x = os.path.exists(x_path)
    if has_x:
        inputx = np.load(x_path)
        print(f"Loaded inputx shape: {inputx.shape}")
    
    has_components = os.path.exists(pred_seasonal_path) and os.path.exists(true_seasonal_path) and os.path.exists(true_trend_path)
    if has_components:
        preds_seasonal = np.load(pred_seasonal_path)
        trues_seasonal = np.load(true_seasonal_path)
        trues_trend = np.load(true_trend_path)
        
        # Calculate predicted trend
        preds_trend = preds - preds_seasonal
        
        print(f"Loaded components.")

    print(f"Loaded preds shape: {preds.shape}")
    print(f"Loaded trues shape: {trues.shape}")

    vis_folder = os.path.join(folder_path, 'visualizations')
    if not os.path.exists(vis_folder):
        os.makedirs(vis_folder)

    num_samples = min(args.num_samples, preds.shape[0])
    num_channels = min(args.num_channels, preds.shape[2])

    for i in range(num_samples):
        for j in range(num_channels):
            # 1. Full Series Plot (Lookback + Prediction)
            plt.figure(figsize=(12, 6))
            
            if has_x:
                lookback = inputx[i, :, j]
                seq_len = len(lookback)
                pred_len = preds.shape[1]
                
                # Plot lookback
                plt.plot(range(seq_len), lookback, label='Lookback', color='gray', alpha=0.7)
                
                # Plot Ground Truth (Lookback + Future)
                # Note: trues is only the future part
                plt.plot(range(seq_len, seq_len + pred_len), trues[i, :, j], label='GroundTruth', color='blue', linewidth=2)
                
                # Plot Prediction
                plt.plot(range(seq_len, seq_len + pred_len), preds[i, :, j], label='Prediction', color='red', linestyle='--', linewidth=2)
                
                # Add vertical line at prediction start
                plt.axvline(x=seq_len, color='k', linestyle=':', alpha=0.5)
                
            else:
                plt.plot(trues[i, :, j], label='GroundTruth', linewidth=2)
                plt.plot(preds[i, :, j], label='Prediction', linewidth=2)
            
            plt.title(f'Sample {i}, Channel {j} - Full Series')
            plt.legend()
            plt.grid(True)
            save_path = os.path.join(vis_folder, f'sample_{i}_channel_{j}_full.png')
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            
            # 2. Components Plot (if available)
            if has_components:
                fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
                
                # Seasonal Component
                axes[0].plot(trues_seasonal[i, :, j], label='GroundTruth Seasonal', color='blue')
                axes[0].plot(preds_seasonal[i, :, j], label='Predicted Seasonal', color='red', linestyle='--')
                axes[0].set_title(f'Sample {i}, Channel {j} - Seasonal Component')
                axes[0].legend()
                axes[0].grid(True)
                
                # Trend Component
                axes[1].plot(trues_trend[i, :, j], label='GroundTruth Trend', color='blue')
                axes[1].plot(preds_trend[i, :, j], label='Predicted Trend', color='red', linestyle='--')
                axes[1].set_title(f'Sample {i}, Channel {j} - Trend Component')
                axes[1].legend()
                axes[1].grid(True)
                
                plt.tight_layout()
                save_path = os.path.join(vis_folder, f'sample_{i}_channel_{j}_components.png')
                plt.savefig(save_path, bbox_inches='tight')
                plt.close()
            
    print(f"Saved visualizations to {vis_folder}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize TFDNet Results')
    parser.add_argument('--folder_path', type=str, required=True, help='Path to the results folder containing pred.npy and true.npy')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to visualize')
    parser.add_argument('--num_channels', type=int, default=1, help='Number of channels to visualize (starting from 0)')
    
    args = parser.parse_args()
    visualize(args)
