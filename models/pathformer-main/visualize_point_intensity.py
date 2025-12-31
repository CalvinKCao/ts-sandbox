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

def get_sample(seq_len=96):
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'datasets', 'ETT-small', 'ETTh1.csv')
    if not os.path.exists(csv_path):
        t = np.linspace(0, 4*np.pi, seq_len)
        data = np.sin(t) + 0.5 * np.sin(5*t)
        return torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(2)

    df = pd.read_csv(csv_path)
    data = df['OT'].values
    data = (data - np.mean(data)) / np.std(data)
    
    # Pick a sample with a clear feature
    start_idx = 1000
    sample = data[start_idx:start_idx+seq_len]
    return torch.tensor(sample, dtype=torch.float32).unsqueeze(0).unsqueeze(2)

def visualize_point_intensity(gt_tensor, pred_tensor, target_idx=48, gamma=0.01):
    # Compute alignment
    D = soft_dtw.pairwise_distances(
        gt_tensor[0, :, :].view(-1, 1),
        pred_tensor[0, :, :].view(-1, 1)
    )
    D_batch = D.unsqueeze(0)
    path_dtw_func = path_soft_dtw.PathDTWBatch.apply
    alignment_matrix = path_dtw_func(D_batch, gamma).squeeze(0).detach().numpy()
    
    # Extract the specific row for the target_idx in Ground Truth
    # This row contains the weights for how GT[target_idx] maps to every point in Pred
    weights = alignment_matrix[target_idx, :]
    
    gt = gt_tensor.squeeze().numpy()
    pred = pred_tensor.squeeze().numpy()
    seq_len = len(gt)
    
    plt.figure(figsize=(12, 10))
    plt.suptitle(f'Soft-DTW Intensity Mapping for GT Point {target_idx} (Gamma={gamma})', fontsize=16)
    
    # Plot 1: The Signals
    plt.subplot(3, 1, 1)
    plt.plot(gt, label='Ground Truth', color='black', linewidth=2)
    plt.plot(pred, label='Prediction', color='red', linestyle='--')
    # Highlight the target point
    plt.plot(target_idx, gt[target_idx], 'bo', markersize=10, label=f'Target Point (GT[{target_idx}])')
    plt.axvline(x=target_idx, color='blue', linestyle=':', alpha=0.5)
    plt.legend()
    plt.title('1. The Signals')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: The Full Alignment Matrix
    plt.subplot(3, 1, 2)
    sns.heatmap(alignment_matrix, cmap='Greys', xticklabels=10, yticklabels=10)
    # Highlight the row we are looking at
    plt.axhline(y=target_idx, color='blue', linestyle='-', linewidth=2, alpha=0.7)
    plt.title('2. Full Alignment Matrix (Blue line = Row being analyzed)')
    plt.ylabel('Ground Truth Index')
    plt.xlabel('Prediction Index')
    
    # Plot 3: The Intensity Distribution for the Target Point
    plt.subplot(3, 1, 3)
    plt.plot(weights, color='blue', linewidth=2)
    plt.fill_between(range(seq_len), weights, alpha=0.3, color='blue')
    plt.title(f'3. Alignment Intensity for GT Point {target_idx}')
    plt.xlabel('Prediction Index')
    plt.ylabel('Weight / Probability')
    plt.grid(True, alpha=0.3)
    
    # Find max weight index
    max_idx = np.argmax(weights)
    plt.plot(max_idx, weights[max_idx], 'ro')
    plt.text(max_idx, weights[max_idx], f' Max: Pred[{max_idx}]', verticalalignment='bottom')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('point_intensity_mapping.png')
    print("Saved point_intensity_mapping.png")
    
    # Print exact values around the peak
    print(f"\nExact intensities for GT point {target_idx} mapping to Prediction points:")
    print("-" * 60)
    print(f"{'Pred Index':<15} {'Intensity (Weight)':<20} {'Value Difference':<20}")
    print("-" * 60)
    
    # Show a window around the max
    window = 5
    start = max(0, max_idx - window)
    end = min(seq_len, max_idx + window + 1)
    
    for i in range(start, end):
        marker = "<-- MAX" if i == max_idx else ""
        diff = abs(gt[target_idx] - pred[i])
        print(f"{i:<15} {weights[i]:.6f}             {diff:.4f} {marker}")

def main():
    gt = get_sample()
    
    # Create a shifted prediction
    shift = 5
    pred_np = np.roll(gt.squeeze().numpy(), shift)
    pred_np += np.random.normal(0, 0.1, size=pred_np.shape)
    pred = torch.tensor(pred_np, dtype=torch.float32).unsqueeze(0).unsqueeze(2)
    
    # Visualize middle point
    visualize_point_intensity(gt, pred, target_idx=48, gamma=0.01)

if __name__ == "__main__":
    main()
