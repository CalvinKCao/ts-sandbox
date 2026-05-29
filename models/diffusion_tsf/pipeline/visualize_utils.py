import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Tuple

def denorm(x, mean, std):
    """Denormalize (C, T) tensor using (1, C) stats."""
    m = mean.squeeze().unsqueeze(-1)   # (C, 1)
    s = std.squeeze().unsqueeze(-1)
    return x * s + m

def generate_pipeline_visualizations(
    model: torch.nn.Module,
    itrans_model: torch.nn.Module,
    dataset,
    stats: Tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
    output_dir: str,
    subset_id: str,
    n_samples: int = 1,
    forecast_length: int = 96,
    lookback_length: int = 96,
) -> List[str]:
    """
    Generates 2D denoising steps and final 1D comparison plots for a few samples.
    Saves as compressed JPEGs with alphanumeric sorting.
    Returns list of saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    mean, std = stats
    mean = torch.tensor(mean, dtype=torch.float32)
    std = torch.tensor(std, dtype=torch.float32)
    
    saved_paths = []
    
    # Pick a few samples
    indices = np.linspace(0, len(dataset) - 1, min(n_samples, len(dataset)), dtype=int).tolist()
    
    file_idx = 1
    
    for row, idx in enumerate(indices):
        past, future = dataset[idx]
        past_t = past.unsqueeze(0).to(device)  # (1, C, L)
        
        with torch.no_grad():
            # iTransformer forward
            B, C, L = past_t.shape
            x_enc = past_t.permute(0, 2, 1)
            seq_sl = getattr(itrans_model, 'seq_len', L)
            if x_enc.shape[1] > seq_sl:
                x_enc = x_enc[:, -seq_sl:, :]
            x_dec = torch.zeros(B, forecast_length, C, device=device)
            itrans_out = itrans_model(x_enc, None, x_dec, None)
            if isinstance(itrans_out, tuple):
                itrans_out = itrans_out[0]
            itrans_pred = itrans_out.permute(0, 2, 1).cpu()[0]  # (C, F)
            
            # Diffusion with intermediates
            # We want to force DDIM or similar to get intermediates, since anchor is just 1 step.
            # But the user specifically requested "various denoising timesteps".
            # We will use DDIM 20 steps just for the visualization of the FIRST sample's denoising path,
            # even if the pipeline uses anchor sampler elsewhere.
            torch.manual_seed(42 + idx)
            result = model.generate(
                past_t,
                sampler="ddim",
                num_inference_steps=20,
                yield_intermediates=True
            )
            
            diff_pred = result.get('prediction_global_norm', result['prediction']).cpu()[0]
            intermediates = result.get('intermediates', [])
            
        # 1. Plot Intermediate 2D Heatmaps (only for first sample to save space)
        if row == 0 and intermediates:
            for i, (t_step, img_tensor) in enumerate(intermediates):
                # img_tensor shape: (B, V, H, W)
                img = img_tensor[0, 0].cpu().numpy()  # Plot first variate
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.imshow(img, aspect='auto', cmap='viridis')
                ax.set_title(f"Denoising Step {t_step}")
                ax.axis('off')
                
                path = os.path.join(output_dir, f"{file_idx:03d}_2D_denoising_sample{row}_var0_step{t_step:04d}.jpg")
                fig.savefig(path, dpi=100, format='jpg', bbox_inches='tight')
                plt.close(fig)
                saved_paths.append(path)
                file_idx += 1
                
        # 2. Plot Final 1D Comparison
        past_dn = denorm(past, mean, std)
        future_sliced = future[:, -forecast_length:]
        future_dn = denorm(future_sliced, mean, std)
        itrans_dn = denorm(itrans_pred, mean, std)
        diff_pred_sliced = diff_pred[:, -forecast_length:] if diff_pred.shape[-1] > forecast_length else diff_pred
        diff_dn = denorm(diff_pred_sliced, mean, std)

        n_vars_plot = min(3, C) # Plot up to 3 variates
        fig, axes = plt.subplots(
            1, n_vars_plot, 
            figsize=(5 * n_vars_plot, 3.5),
            squeeze=False,
            constrained_layout=True
        )
        
        t_past = np.arange(-lookback_length, 0)
        t_future = np.arange(0, forecast_length)
        
        for col in range(n_vars_plot):
            ax = axes[0, col]
            gt = future_dn[col].numpy()
            it = itrans_dn[col].numpy()
            df = diff_dn[col].numpy()
            
            ax.plot(t_past, past_dn[col, -lookback_length:].numpy(), color='#9E9E9E', alpha=0.5, linewidth=1.0)
            ax.plot(t_future, gt, color='#2196F3', linewidth=1.6, label='Ground Truth' if col==0 else '')
            ax.plot(t_future, it, color='#FF9800', linewidth=1.2, linestyle='--', alpha=0.85, label='iTransformer' if col==0 else '')
            ax.plot(t_future, df, color='#E91E63', linewidth=1.2, label='Diffusion (DDIM)' if col==0 else '')
            ax.axvline(x=0, color='black', linestyle=':', alpha=0.25)
            
            it_mae = np.mean(np.abs(it - gt))
            df_mae = np.mean(np.abs(df - gt))
            ax.set_title(f"Var {col}\niTrans MAE: {it_mae:.3f} | Diff MAE: {df_mae:.3f}", fontsize=10)
            
        fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3)
        
        path = os.path.join(output_dir, f"{file_idx:03d}_1D_comparison_sample{row}.jpg")
        fig.savefig(path, dpi=120, format='jpg', bbox_inches='tight')
        plt.close(fig)
        saved_paths.append(path)
        file_idx += 1
        
    return saved_paths
