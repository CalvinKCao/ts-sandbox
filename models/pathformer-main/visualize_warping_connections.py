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

def get_samples(seq_len=96):
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'datasets', 'ETT-small', 'ETTh1.csv')
    if not os.path.exists(csv_path):
        print("Dataset not found, using synthetic data")
        t = np.linspace(0, 4*np.pi, seq_len)
        data = np.sin(t) + 0.5 * np.sin(5*t)
        return [torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(2)]

    df = pd.read_csv(csv_path)
    data = df['OT'].values
    data = (data - np.mean(data)) / np.std(data)
    
    samples = []
    # Pick a few interesting spots
    indices = [1000, 2500, 4000] 
    for idx in indices:
        if idx + seq_len < len(data):
            sample = data[idx:idx+seq_len]
            samples.append(torch.tensor(sample, dtype=torch.float32).unsqueeze(0).unsqueeze(2))
    return samples

def visualize_warping(gt_tensor, pred_tensor, gamma=0.01, sample_idx=0, shift_val=0):
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
    
    plt.figure(figsize=(12, 6))
    
    # Vertical offset for visualization
    offset = 4.0
    
    plt.plot(gt + offset, label='Ground Truth', color='black', linewidth=2)
    plt.plot(pred, label=f'Prediction (Shifted {shift_val})', color='#d62728', linewidth=2) # Red
    
    # Draw connections
    lines = []
    colors = []
    
    # Filter for strong connections
    threshold = 0.1
    
    # To avoid too many lines, we can subsample or just take the max for each i
    for i in range(0, seq_len, 2): # Every 2nd point
        j_probs = alignment_matrix[i, :]
        j_max = np.argmax(j_probs)
        prob = j_probs[j_max]
        
        if prob > threshold:
            # Connect (i, gt[i]) to (j_max, pred[j_max])
            # Note: x-axis is time index
            p1 = (i, gt[i] + offset)
            p2 = (j_max, pred[j_max])
            lines.append([p1, p2])
            # Alpha based on probability
            colors.append((0.2, 0.2, 0.8, min(prob, 0.8))) # Blueish lines
            
    lc = LineCollection(lines, colors=colors, linewidths=1)
    plt.gca().add_collection(lc)
    
    plt.title(f'DTW Warping Connections (Gamma={gamma})\nBlue lines connect matched points across time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlabel('Time Step')
    plt.yticks([]) # Hide y ticks as they are arbitrary due to offset
    
    plt.tight_layout()
    plt.savefig(f'warping_lines_sample_{sample_idx}.png')
    print(f"Saved warping_lines_sample_{sample_idx}.png")

def main():
    samples = get_samples()
    
    # Create shifted predictions
    shifts = [5, 10, -5] # Different shifts for different samples
    
    for i, gt in enumerate(samples):
        if i >= len(shifts): break
        shift = shifts[i]
        pred_np = np.roll(gt.squeeze().numpy(), shift)
        # Add some noise
        pred_np += np.random.normal(0, 0.1, size=pred_np.shape)
        pred = torch.tensor(pred_np, dtype=torch.float32).unsqueeze(0).unsqueeze(2)
        
        print(f"Visualizing sample {i+1} with shift {shift}...")
        visualize_warping(gt, pred, gamma=0.01, sample_idx=i+1, shift_val=shift)

if __name__ == "__main__":
    main()
