"""
Detailed visualization for experimental models (A, B, A+B).
Shows intermediate steps: noise, trend, conditioning channels (2D maps).
"""

import argparse
import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.diffusion_tsf.train_multivariate_pipeline import (
    create_diffusion_model, load_dataset,
    load_itransformer_from_checkpoint,
    ITRANSFORMER_SEQ_LEN
)
from models.diffusion_tsf.guidance import iTransformerGuidance
from models.diffusion_tsf.experimental_diffusion_model import apply_zero_phase_lowpass


def denorm(x, mean, std):
    """Denormalize (C, T) or (1, C, T) tensor."""
    if x.dim() == 3:
        m = mean.unsqueeze(-1) # (C, 1)
        s = std.unsqueeze(-1)
        return x * s + m
    else:
        m = mean.squeeze().unsqueeze(-1)
        s = std.squeeze().unsqueeze(-1)
        return x * s + m


def plot_detailed_sample(dataset_name, run_dir, device, n_samples=2):
    # 1. Parse config from run_dir
    run_folder = os.path.basename(run_dir)
    # 05-18-3650640-exp_A_ETTh1
    remainder = run_folder.split('-', 3)[-1]
    if remainder.startswith('exp_A_B_'):
        scenario = "A+B"
    elif remainder.startswith('exp_A_'):
        scenario = "A"
    elif remainder.startswith('exp_B_'):
        scenario = "B"
    else:
        scenario = "baseline"

    print(f"\nVisualizing {run_folder} (Scenario: {scenario})")

    # 2. Load Data
    # GetTARGET_DIM
    target_dim = 7
    if "weather" in dataset_name: target_dim = 21
    if "exchange_rate" in dataset_name: target_dim = 8
    
    # We need to set the global EXPERIMENT for create_diffusion_model
    import models.diffusion_tsf.train_multivariate_pipeline as tmp
    tmp.EXPERIMENT = scenario
    tmp.N_VARIATES = target_dim

    _, _, test_ds, norm_stats = load_dataset(dataset_name, lookback=96, horizon=96)
    mean = torch.tensor(norm_stats['mean'], dtype=torch.float32)
    std = torch.tensor(norm_stats['std'], dtype=torch.float32)

    # 3. Load Models
    ckpt_dir = os.path.join(run_dir, 'ckpts')
    subset_id = f"exp_{scenario}"
    
    itrans_path = os.path.join(ckpt_dir, f"{subset_id}_itransformer_finetuned.pt")
    if not os.path.exists(itrans_path):
        # try synthetic
        itrans_path = os.path.join(ckpt_dir, "pretrained_dim7", "itransformer.pt")
        
    itrans_model = load_itransformer_from_checkpoint(itrans_path, target_dim, device)
    itrans_guidance = iTransformerGuidance(itrans_model)
    
    diff_model = create_diffusion_model(n_variates=target_dim).to(device)
    diff_model.set_guidance_model(itrans_guidance)
    
    # Find the best.pt
    best_pt = None
    for d in os.listdir(ckpt_dir):
        dp = os.path.join(ckpt_dir, d)
        if os.path.isdir(dp) and os.path.exists(os.path.join(dp, 'best.pt')):
            best_pt = os.path.join(dp, 'best.pt')
            break
    
    if best_pt:
        print(f"Loading diffusion weights from {best_pt}")
        ckpt = torch.load(best_pt, map_location=device, weights_only=False)
        diff_model.load_state_dict(ckpt['model_state_dict'])
    else:
        print("WARNING: No best.pt found for diffusion, using random weights!")

    diff_model.eval()

    # 4. Generate and Plot
    os.makedirs('results_experimental/viz_detailed', exist_ok=True)
    
    for s_idx in range(n_samples):
        # Pick samples spaced out
        idx = (len(test_ds) // (n_samples + 1)) * (s_idx + 1)
        past, future = test_ds[idx]
        past_t = past.unsqueeze(0).to(device)
        
        with torch.no_grad():
            res = diff_model.generate(past_t, num_ddim_steps=20)
            
            # iTransformer standalone for baseline
            B, C, L = past_t.shape
            x_enc = past_t.permute(0, 2, 1)
            x_dec = torch.zeros(B, 96, C, device=device)
            itrans_out = itrans_model(x_enc, None, x_dec, None)
            if isinstance(itrans_out, tuple): itrans_out = itrans_out[0]
            itrans_pred = itrans_out.permute(0, 2, 1).cpu()[0]

        # Extract 2D maps for the first variate
        v_idx = 0 
        
        # past_2d: (B, V, H, W)
        p2d = res['past_2d'][0, v_idx].cpu().numpy()
        f2d = res['future_2d'][0, v_idx].cpu().numpy()
        g2d = res['guidance_2d'][0, v_idx].cpu().numpy() if 'guidance_2d' in res else None
        pn2d = res.get('past_noise_2d')
        if pn2d is not None: pn2d = pn2d[0, v_idx].cpu().numpy()

        # 1D components
        pred = res['prediction'].cpu()[0]
        gt = future[:, -96:] # last 96 steps
        
        # Plotting
        n_rows = 4 if pn2d is not None else 3
        fig = plt.figure(figsize=(15, 4 * n_rows))
        gs = fig.add_gridspec(n_rows, 3)
        
        # Row 0: 1D Forecast Comparison
        ax1 = fig.add_subplot(gs[0, :])
        
        t_past = np.arange(-96, 0)
        t_fut = np.arange(0, 96)
        
        # past[v_idx] is (96,), mean[0, v_idx] is scalar
        m_v = mean[0, v_idx]
        s_v = std[0, v_idx]
        
        past_dn = (past[v_idx] * s_v + m_v).numpy()
        gt_dn = (gt[v_idx] * s_v + m_v).numpy()
        pred_dn = (pred[v_idx] * s_v + m_v).numpy()
        itrans_dn = (itrans_pred[v_idx] * s_v + m_v).numpy()
        
        ax1.plot(t_past, past_dn, color='gray', label='Past', alpha=0.5)
        ax1.plot(t_fut, gt_dn, color='blue', label='Ground Truth', linewidth=2)
        ax1.plot(t_fut, itrans_dn, color='orange', linestyle='--', label='iTransformer Baseline')
        ax1.plot(t_fut, pred_dn, color='red', label='Diffusion Output')
        
        if 'trend' in res:
            trend_v = res['trend'][0, v_idx].cpu()
            trend_dn = (trend_v * s_v + m_v).numpy()
            ax1.plot(t_fut, trend_dn, color='green', linestyle=':', label='Extracted Trend')
            
        ax1.axvline(0, color='black', linestyle='--')
        ax1.legend()
        ax1.set_title(f"Sample {idx} - {dataset_name} ({scenario}) - Variate {v_idx}")
        
        # Row 1: 2D Representations (Standard)
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.imshow(p2d, aspect='auto', origin='lower', cmap='magma')
        ax2.set_title("Past Conditioning (2D CDF)")
        
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.imshow(f2d, aspect='auto', origin='lower', cmap='magma')
        ax3.set_title("Generated Future (2D CDF)")
        
        ax4 = fig.add_subplot(gs[1, 2])
        if g2d is not None:
            ax4.imshow(g2d, aspect='auto', origin='lower', cmap='magma')
            ax4.set_title("Guidance Ghost Image (2D CDF)")
        else:
            ax4.axis('off')
            
        # Row 2: Residual Logic (if applicable)
        if pn2d is not None:
            ax5 = fig.add_subplot(gs[2, 0])
            ax5.imshow(pn2d, aspect='auto', origin='lower', cmap='magma')
            ax5.set_title("Lookback Noise Conditioning (Exp A)")
            
            # Show the residual 1D
            ax6 = fig.add_subplot(gs[2, 1:])
            # residual = final - trend
            res_1d = (pred[v_idx] - res['trend'][0, v_idx].cpu()).numpy()
            # gt_residual
            gt_trend = apply_zero_phase_lowpass(future.unsqueeze(0), 0.12)[0, v_idx, -96:]
            gt_res_1d = (future[v_idx, -96:] - gt_trend).numpy()
            
            ax6.plot(t_fut, gt_res_1d, color='blue', alpha=0.4, label='GT Residual')
            ax6.plot(t_fut, res_1d, color='red', label='Generated Residual')
            ax6.set_title("Residual Comparison (Standardized Scale)")
            ax6.legend()

        plt.tight_layout()
        out_name = f"detailed_{scenario}_{dataset_name}_s{s_idx}.png"
        out_path = os.path.join('results_experimental/viz_detailed', out_name)
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved visualization to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=str, required=True, help="Path to run folder (e.g. results/runs/...)")
    parser.add_argument('--dataset', type=str, default="ETTh1")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    plot_detailed_sample(args.dataset, args.run_dir, device)


if __name__ == "__main__":
    main()
