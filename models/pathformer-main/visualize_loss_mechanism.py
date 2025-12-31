import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
import seaborn as sns
import argparse

# Add parent directory to path to find modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

# Add DILATE loss path
dilate_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'losses', 'DILATE-master')
sys.path.insert(0, dilate_path)

from dilate_loss_wrapper import FrequencySelectiveDilateLoss
try:
    from loss import soft_dtw
    from loss import path_soft_dtw
except ImportError:
    # Manual import if package structure is different
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
        self.checkpoints = './checkpoints/'
        self.seq_len = 96
        self.pred_len = 96
        self.individual = False
        self.d_model = 4 # From train_etth1_fft_dilate.bat
        self.d_ff = 64 # From train_etth1_fft_dilate.bat
        self.num_nodes = 7 # From train_etth1_fft_dilate.bat
        self.layer_nums = 3
        self.k = 3 # From train_etth1_fft_dilate.bat
        self.num_experts_list = [4, 4, 4]
        self.patch_size_list = [16, 12, 8, 32, 12, 8, 6, 4, 8, 6, 4, 2]
        self.do_predict = False
        self.revin = 1
        self.drop = 0.1
        self.embed = 'timeF'
        self.residual_connection = 1 # From train_etth1_fft_dilate.bat
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

def visualize_checkpoint_dtw():
    args = Args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    
    # Reshape patch_size_list as done in run.py
    args.patch_size_list = np.array(args.patch_size_list).reshape(args.layer_nums, -1).tolist()
    
    # Initialize Experiment
    exp = Exp_Main(args)
    
    # Load Model
    setting = 'ETTh1_96_96_fft_dilate_strict_PathFormer_ftETTh1_slM_pl96_96'
    checkpoint_path = os.path.join(args.checkpoints, setting, 'checkpoint.pth')
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        # Try to find any checkpoint in the folder
        if os.path.exists(args.checkpoints):
            subdirs = [d for d in os.listdir(args.checkpoints) if os.path.isdir(os.path.join(args.checkpoints, d))]
            if subdirs:
                print(f"Found checkpoints: {subdirs}")
                setting = subdirs[0]
                checkpoint_path = os.path.join(args.checkpoints, setting, 'checkpoint.pth')
                print(f"Trying: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print("Could not find any checkpoint.")
        return

    print(f"Loading model from {checkpoint_path}")
    exp.model.load_state_dict(torch.load(checkpoint_path))
    exp.model.eval()
    
    # Get Data
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
    batch_x_mark = batch_x_mark.float().to(exp.device)
    batch_y_mark = batch_y_mark.float().to(exp.device)
    
    # Predict
    print("Running prediction...")
    with torch.no_grad():
        if args.use_amp:
            with torch.cuda.amp.autocast():
                if args.model == 'PathFormer':
                    outputs, _ = exp.model(batch_x)
                else:
                    outputs = exp.model(batch_x)
        else:
            if args.model == 'PathFormer':
                outputs, _ = exp.model(batch_x)
            else:
                outputs = exp.model(batch_x)
                
    # Extract features
    f_dim = -1 if args.features == 'MS' else 0
    outputs = outputs[:, -args.pred_len:, f_dim:]
    batch_y = batch_y[:, -args.pred_len:, f_dim:]
    
    # Select a sample (e.g. index 0)
    idx = 0
    pred = outputs[idx:idx+1] # (1, pred_len, n_features)
    gt = batch_y[idx:idx+1]   # (1, pred_len, n_features)
    
    # Initialize Loss for Frequency Extraction
    loss_fn = FrequencySelectiveDilateLoss(
        freq_threshold=args.freq_threshold,
        device=exp.device
    )
    
    # Extract Frequencies (using target mask)
    gt_high, gt_low, mask = loss_fn.extract_frequency_components(gt)
    pred_high, pred_low, _ = loss_fn.extract_frequency_components(pred, mask=mask)
    
    # Compute DTW Alignment for High Freq (Feature 0 for visualization)
    # We visualize the first feature (usually OT if it's the last one, but here we have 7 features)
    # Let's visualize the target feature (last one)
    feat_idx = -1 
    
    gt_high_feat = gt_high[:, :, feat_idx].view(-1, 1)
    pred_high_feat = pred_high[:, :, feat_idx].view(-1, 1)
    
    gamma = 0.01
    D = soft_dtw.pairwise_distances(
        gt_high_feat,
        pred_high_feat
    )
    
    path_dtw_func = path_soft_dtw.PathDTWBatch.apply
    D_batch = D.unsqueeze(0)
    alignment_matrix = path_dtw_func(D_batch, gamma).squeeze(0).detach().cpu().numpy()
    
    # Visualization
    plt.figure(figsize=(10, 8))
    
    # Use Greys colormap, reversed so higher values (1.0) are black/dark
    # alignment_matrix values are probabilities [0, 1]
    # cmap='Greys' means 0 is white, 1 is black.
    sns.heatmap(alignment_matrix, cmap='Greys', xticklabels=10, yticklabels=10)
    plt.title(f'Soft-DTW Alignment Matrix (High Freq)\nDarker = Higher Weight')
    plt.xlabel('Prediction Time Step')
    plt.ylabel('Ground Truth Time Step')
    
    # Add diagonal line
    plt.plot([0, args.pred_len], [0, args.pred_len], color='red', linestyle='--', alpha=0.5)
    
    output_path = 'dtw_checkpoint_visualization.png'
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    visualize_checkpoint_dtw()
