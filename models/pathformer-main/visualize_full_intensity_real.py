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

# Import Exp_Main
try:
    from train_etth2 import Exp_Main_With_Dilate as Exp_Main
except ImportError:
    from exp.exp_main import Exp_Main
class Args:
    def __init__(self):
        self.is_training = 0
        self.model_id = 'ETTh1_96_96_fft_dilate_strict'
        self.model = 'PathFormer'
        self.data = 'ETTh1'
        self.root_path = 'C:\\Users\\kevin\\dev\\ts-sandbox\\datasets\\ETT-small'
        self.data_path = 'ETTh1.csv'
        self.features = 'M'
        self.target = 'OT'
        self.freq = 'h'
        self.checkpoints = 'C:\\Users\\kevin\\dev\\ts-sandbox\\checkpoints'
        self.seq_len = 96
        self.pred_len = 96
        self.individual = False
        self.d_model = 4
        self.d_ff = 64
        self.num_nodes = 7
        self.layer_nums = 3
        self.k = 3
        self.num_experts_list = [4, 4, 4]
        self.patch_size_list = [16, 12, 8, 32, 12, 8, 6, 4, 8, 6, 4, 2]
        self.do_predict = False
        self.revin = 1
        self.drop = 0.1
        self.embed = 'timeF'
        self.residual_connection = 1
        self.metric = 'mae'
        self.batch_norm = 0
        self.loss_type = 'fft_dilate'
        self.dilate_alpha = 0.3
        self.dilate_gamma = 0.001
        self.freq_threshold = 80.0
        self.num_workers = 0
        self.itr = 1
        self.train_epochs = 20
        self.batch_size = 32
        self.patience = 5
        self.learning_rate = 0.001
        self.lradj = 'TST'
        self.use_amp = False
        self.pct_start = 0.4
        self.use_gpu = True
        self.gpu = 0
        self.use_multi_gpu = False
        self.devices = '0'
        self.test_flop = False
        self.device_ids = [0]

def visualize_full_intensity(gt_tensor, pred_tensor, gamma=0.01, sample_idx=0):
    # Compute alignment
    D = soft_dtw.pairwise_distances(
        gt_tensor[0, :, :].view(-1, 1),
        pred_tensor[0, :, :].view(-1, 1)
    )
    D_batch = D.unsqueeze(0)
    path_dtw_func = path_soft_dtw.PathDTWBatch.apply
    alignment_matrix = path_dtw_func(D_batch, gamma).squeeze(0).detach().cpu().numpy()
    
    gt = gt_tensor.squeeze().cpu().numpy()
    pred = pred_tensor.squeeze().cpu().numpy()
    seq_len = len(gt)
    
    plt.figure(figsize=(15, 8))
    
    # Vertical offset for visualization
    offset = np.max(gt) - np.min(pred) + 2.0
    
    # Plot signals
    plt.plot(gt + offset, label='Ground Truth', color='black', linewidth=2, zorder=10)
    plt.plot(pred, label='Prediction', color='#d62728', linewidth=2, zorder=10)
    
    # Prepare lines
    lines = []
    colors = []
    
    # Threshold to avoid drawing invisible lines
    threshold = 0.001 
    
    rows, cols = alignment_matrix.shape
    
    for i in range(rows): # GT index
        for j in range(cols): # Pred index
            weight = alignment_matrix[i, j]
            
            if weight > threshold:
                p1 = (i, gt[i] + offset)
                p2 = (j, pred[j])
                lines.append([p1, p2])
                alpha = min(1.0, weight)
                colors.append((0.0, 0.0, 0.5, alpha))

    lc = LineCollection(lines, colors=colors, linewidths=1, zorder=1)
    plt.gca().add_collection(lc)
    
    plt.title(f'Full Soft-DTW Intensity Mapping (Gamma={gamma})\nSample {sample_idx}', fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.2)
    plt.xlabel('Time Step')
    plt.yticks([])
    
    plt.text(0, offset + np.max(gt) + 0.5, "Ground Truth", fontsize=12, fontweight='bold')
    plt.text(0, np.max(pred) + 0.5, "Prediction", fontsize=12, fontweight='bold', color='#d62728')
    
    plt.tight_layout()
    output_file = f'full_intensity_connections_sample_{sample_idx}.png'
    plt.savefig(output_file, dpi=300)
    print(f"Saved {output_file}")

def main():
    args = Args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.patch_size_list = np.array(args.patch_size_list).reshape(args.layer_nums, -1).tolist()
    
    exp = Exp_Main(args)
    
    setting = 'ETTh1_96_96_fft_dilate_strict_PathFormer_ftETTh1_slM_pl96_96'
    checkpoint_path = os.path.join(args.checkpoints, setting, 'checkpoint.pth')
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    print(f"Loading model from {checkpoint_path}")
    exp.model.load_state_dict(torch.load(checkpoint_path))
    exp.model.eval()
    
    print("Loading test data...")
    dataset, loader = exp._get_data(flag='test')
    
    # Get a batch
    try:
        batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(loader))
    except StopIteration:
        print("Data loader is empty.")
        return

    batch_x = batch_x.float().to(exp.device)
    batch_y = batch_y.float().to(exp.device)
    
    print("Running prediction...")
    with torch.no_grad():
        if args.model == 'PathFormer':
            outputs, _ = exp.model(batch_x)
        else:
            outputs = exp.model(batch_x)
                
    f_dim = -1 if args.features == 'MS' else 0
    outputs = outputs[:, -args.pred_len:, f_dim:]
    batch_y = batch_y[:, -args.pred_len:, f_dim:]
    
    # Visualize a few samples
    indices = [0, 10, 20]
    for idx in indices:
        pred = outputs[idx:idx+1] # (1, pred_len, n_features)
        gt = batch_y[idx:idx+1]   # (1, pred_len, n_features)
        
        # Visualize the last feature (OT)
        feat_idx = -1
        gt_feat = gt[:, :, feat_idx].view(1, -1, 1)
        pred_feat = pred[:, :, feat_idx].view(1, -1, 1)
        
        visualize_full_intensity(gt_feat, pred_feat, gamma=0.01, sample_idx=idx)

if __name__ == "__main__":
    main()
