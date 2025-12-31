import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
import seaborn as sns

# Add parent directory to path to find modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

# Add DILATE loss path
dilate_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'losses', 'DILATE-master')
sys.path.insert(0, dilate_path)

from dilate_loss_wrapper import DilateLoss
try:
    from loss import soft_dtw
    from loss import path_soft_dtw
except ImportError:
    import importlib.util
    spec = importlib.util.spec_from_file_location("soft_dtw", os.path.join(dilate_path, 'loss', 'soft_dtw.py'))
    soft_dtw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(soft_dtw)
    
    spec = importlib.util.spec_from_file_location("path_soft_dtw", os.path.join(dilate_path, 'loss', 'path_soft_dtw.py'))
    path_soft_dtw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(path_soft_dtw)

def load_sample_data(seq_len=96):
    """Load a sample from ETTh1 dataset"""
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'datasets', 'ETT-small', 'ETTh1.csv')
    
    if os.path.exists(csv_path):
        print(f"Loading real data from {csv_path}")
        df = pd.read_csv(csv_path)
        start_idx = 1000
        data = df['OT'].values[start_idx:start_idx+seq_len]
        data = (data - np.mean(data)) / np.std(data)
        return torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(2)
    else:
        print("Dataset not found, generating synthetic data")
        t = np.linspace(0, 4*np.pi, seq_len)
        data = np.sin(t) + 0.5 * np.sin(5*t)
        return torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(2)

def create_shifted_prediction(gt, shift):
    """Create a shifted version of ground truth"""
    pred_np = gt.squeeze().numpy()
    pred_np = np.roll(pred_np, shift)
    # Add small noise
    pred_np += np.random.normal(0, 0.1, size=pred_np.shape)
    return torch.tensor(pred_np, dtype=torch.float32).unsqueeze(0).unsqueeze(2)

def visualize_gamma_effect(gt, pred, gammas=[0.001, 0.01, 0.1, 1.0]):
    """Visualize how gamma affects the alignment matrix"""
    plt.figure(figsize=(20, 5))
    plt.suptitle('Effect of Gamma on Soft-DTW Alignment Matrix (Smoothness)', fontsize=16)
    
    # Compute pairwise distances once
    D = soft_dtw.pairwise_distances(
        gt[0, :, :].view(-1, 1),
        pred[0, :, :].view(-1, 1)
    )
    D_batch = D.unsqueeze(0)
    path_dtw_func = path_soft_dtw.PathDTWBatch.apply
    
    for i, gamma in enumerate(gammas):
        plt.subplot(1, len(gammas), i+1)
        
        # Compute alignment
        alignment_matrix = path_dtw_func(D_batch, gamma).squeeze(0).detach().numpy()
        
        sns.heatmap(alignment_matrix, cmap='Greys', cbar=False, xticklabels=False, yticklabels=False)
        plt.title(f'Gamma = {gamma}')
        if i == 0:
            plt.ylabel('Ground Truth Time')
            plt.xlabel('Prediction Time')
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('dilate_gamma_effect.png')
    print("Saved dilate_gamma_effect.png")

def visualize_alpha_effect(gt, alphas=[0.1, 0.5, 0.9], shifts=range(-20, 21, 2)):
    """Visualize how alpha affects the loss sensitivity to time shifts"""
    plt.figure(figsize=(12, 8))
    
    device = 'cpu'
    gamma = 0.01 # Fixed gamma
    
    results = {alpha: [] for alpha in alphas}
    
    for shift in shifts:
        pred = create_shifted_prediction(gt, shift)
        
        for alpha in alphas:
            loss_fn = DilateLoss(alpha=alpha, gamma=gamma, device=device)
            loss, loss_shape, loss_temporal = loss_fn(pred, gt)
            results[alpha].append(loss.item())
            
    # Plotting
    for alpha in alphas:
        # Normalize loss for better comparison of shape
        losses = np.array(results[alpha])
        # losses = (losses - np.min(losses)) / (np.max(losses) - np.min(losses))
        plt.plot(shifts, losses, marker='o', label=f'Alpha = {alpha} (Shape Weight)')
        
    plt.title(f'Effect of Alpha on Loss Sensitivity to Time Shift (Gamma={gamma})')
    plt.xlabel('Time Shift (Steps)')
    plt.ylabel('Total DILATE Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    
    plt.text(0, plt.ylim()[0], "Perfect Alignment", ha='center', va='bottom')
    
    plt.savefig('dilate_alpha_effect.png')
    print("Saved dilate_alpha_effect.png")

def main():
    print("Generating DILATE parameter visualizations...")
    
    # 1. Load Data
    gt = load_sample_data(96)
    
    # 2. Visualize Gamma (using a fixed shift of 10 steps)
    print("Visualizing Gamma effect...")
    pred_shifted = create_shifted_prediction(gt, shift=10)
    visualize_gamma_effect(gt, pred_shifted, gammas=[0.001, 0.01, 0.1, 1.0])
    
    # 3. Visualize Alpha (Loss vs Shift)
    print("Visualizing Alpha effect...")
    visualize_alpha_effect(gt, alphas=[0.1, 0.2, 0.5, 0.8, 0.9])
    
    print("Done!")

if __name__ == "__main__":
    main()
