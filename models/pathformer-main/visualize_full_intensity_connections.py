import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
from matplotlib.collections import LineCollection

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
    
    # Pick a sample with clear features
    start_idx = 1000
    sample = data[start_idx:start_idx+seq_len]
    return torch.tensor(sample, dtype=torch.float32).unsqueeze(0).unsqueeze(2)

def visualize_full_intensity(gt_tensor, pred_tensor, gamma=0.01, shift_val=0):
    # Compute alignment
    D = soft_dtw.pairwise_distances(
        gt_tensor[0, :, :].view(-1, 1),
        pred_tensor[0, :, :].view(-1, 1)
    )
    D_batch = D.unsqueeze(0)
    path_dtw_func = path_soft_dtw.PathDTWBatch.apply
    alignment_matrix = path_dtw_func(D_batch, gamma).squeeze(0).detach().numpy()
    
    gt = gt_tensor.squeeze().numpy()
    pred = pred_tensor.squeeze().numpy()
    seq_len = len(gt)
    
    plt.figure(figsize=(15, 8))
    
    # Vertical offset for visualization
    offset = 4.0
    
    # Plot signals
    plt.plot(gt + offset, label='Ground Truth', color='black', linewidth=2, zorder=10)
    plt.plot(pred, label=f'Prediction (Shifted {shift_val})', color='#d62728', linewidth=2, zorder=10)
    
    # Prepare lines
    lines = []
    colors = []
    
    # Threshold to avoid drawing invisible lines
    # Soft-DTW probabilities sum to 1 along the path, but can be very small off-path
    threshold = 0.001 
    
    # Iterate over the alignment matrix
    # alignment_matrix[i, j] is weight between GT[i] and Pred[j]
    rows, cols = alignment_matrix.shape
    
    for i in range(rows): # GT index
        for j in range(cols): # Pred index
            weight = alignment_matrix[i, j]
            
            if weight > threshold:
                # Point 1: (Time i, Value GT[i] + offset)
                p1 = (i, gt[i] + offset)
                # Point 2: (Time j, Value Pred[j])
                p2 = (j, pred[j])
                
                lines.append([p1, p2])
                
                # Color: Dark Blue with Alpha = weight
                # We can scale alpha to make it more visible if needed, but raw weight is most accurate
                # Cap alpha at 1.0
                alpha = min(1.0, weight)
                colors.append((0.0, 0.0, 0.5, alpha))

    # Create LineCollection
    lc = LineCollection(lines, colors=colors, linewidths=1, zorder=1)
    plt.gca().add_collection(lc)
    
    plt.title(f'Full Soft-DTW Intensity Mapping (Gamma={gamma})\nDarker lines = Higher Probability/Weight of Alignment', fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.2)
    plt.xlabel('Time Step')
    plt.yticks([]) # Hide y ticks
    
    # Add a colorbar-like legend explanation
    plt.text(0, offset + np.max(gt) + 0.5, "Ground Truth", fontsize=12, fontweight='bold')
    plt.text(0, np.max(pred) + 0.5, "Prediction", fontsize=12, fontweight='bold', color='#d62728')
    
    plt.tight_layout()
    output_file = 'full_intensity_connections.png'
    plt.savefig(output_file, dpi=300) # High DPI for fine lines
    print(f"Saved {output_file}")

def main():
    gt = get_sample()
    
    # Create a shifted prediction
    shift = 10
    pred_np = np.roll(gt.squeeze().numpy(), shift)
    # Add some noise
    pred_np += np.random.normal(0, 0.1, size=pred_np.shape)
    pred = torch.tensor(pred_np, dtype=torch.float32).unsqueeze(0).unsqueeze(2)
    
    print(f"Visualizing full intensity connections with shift {shift}...")
    visualize_full_intensity(gt, pred, gamma=0.001, shift_val=shift)

if __name__ == "__main__":
    main()
