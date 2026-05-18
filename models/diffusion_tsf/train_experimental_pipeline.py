import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm

# Setup path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from models.diffusion_tsf.train_multivariate_pipeline import load_dataset, create_itransformer
from models.diffusion_tsf.guidance import iTransformerGuidance
from models.diffusion_tsf.experimental_diffusion_model import ExperimentalDiffusionTSF, ExperimentalDiffusionTSFConfig

def train_itransformer(model, train_loader, val_loader, epochs=5, lr=1e-3, device='cuda'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = []
        pbar = tqdm(train_loader, desc=f"iTransformer Epoch {epoch+1}/{epochs}")
        for batch_x, batch_y in pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            # Permute for iTransformer: (B, Seq, V)
            batch_x = batch_x.permute(0, 2, 1)
            batch_y = batch_y.permute(0, 2, 1)
            
            optimizer.zero_grad()
            # iTransformer inputs: x_enc, x_mark_enc, x_dec, x_mark_dec
            # batch_y includes overlap, we only want the actual future (last 96 steps)
            future_y = batch_y[:, -96:, :]
            x_dec = torch.zeros_like(future_y)
            out = model(batch_x, None, x_dec, None)
            if isinstance(out, tuple): out = out[0]
            
            loss = criterion(out, future_y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        model.eval()
        val_loss = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                batch_x = batch_x.permute(0, 2, 1)
                batch_y = batch_y.permute(0, 2, 1)
                future_y = batch_y[:, -96:, :]
                x_dec = torch.zeros_like(future_y)
                out = model(batch_x, None, x_dec, None)
                if isinstance(out, tuple): out = out[0]
                val_loss.append(criterion(out, future_y).item())
                
        avg_val_loss = np.mean(val_loss)
        print(f"iTransformer Epoch {epoch+1}/{epochs} - Train MSE: {np.mean(train_loss):.4f}, Val MSE: {avg_val_loss:.4f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            
    model.load_state_dict(best_state)
    return model

def evaluate_diffusion(model, val_loader, device):
    model.eval()
    val_mse = []
    with torch.no_grad():
        for past, future in val_loader:
            past, future = past.to(device), future.to(device)
            out = model.generate(past, num_ddim_steps=20)
            forecast = out['forecast']
            # future includes lookback_overlap, take only the true horizon
            true_future = future[:, :, -forecast.shape[-1]:] if future.shape[-1] > forecast.shape[-1] else future
            # MSE computation
            mse = F.mse_loss(forecast, true_future).item()
            val_mse.append(mse)
    return np.mean(val_mse)

def evaluate_detailed(model, loader, device, return_detailed=False):
    model.eval()
    mses = []
    maes = []
    residual_mses = []
    trend_mses = []
    with torch.no_grad():
        for past, future in loader:
            past, future = past.to(device), future.to(device)
            out = model.generate(past, num_ddim_steps=20)
            forecast = out['forecast']
            true_future = future[:, :, -forecast.shape[-1]:] if future.shape[-1] > forecast.shape[-1] else future
            mse = F.mse_loss(forecast, true_future).item()
            mae = F.l1_loss(forecast, true_future).item()
            mses.append(mse)
            maes.append(mae)
            
            if return_detailed and 'residual' in out:
                # Compute ground truth trend and residual
                from models.diffusion_tsf.experimental_diffusion_model import apply_zero_phase_lowpass
                true_trend = apply_zero_phase_lowpass(true_future, model.config.residual_cutoff_freq)
                true_residual = true_future - true_trend
                
                res_mse = F.mse_loss(out['residual'], true_residual).item()
                trend_mse = F.mse_loss(out['trend'], true_trend).item()
                residual_mses.append(res_mse)
                trend_mses.append(trend_mse)
                
    if return_detailed and len(residual_mses) > 0:
        return np.mean(mses), np.mean(maes), np.mean(residual_mses), np.mean(trend_mses)
    return np.mean(mses), np.mean(maes)

def train_diffusion(model, train_loader, val_loader, epochs=10, lr=1e-4, device='cuda'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_mse = float('inf')
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = []
        pbar = tqdm(train_loader, desc=f"Diffusion Epoch {epoch+1}/{epochs}")
        for past, future in pbar:
            past, future = past.to(device), future.to(device)
            optimizer.zero_grad()
            out = model(past, future)
            loss = out['loss']
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        val_mse = evaluate_diffusion(model, val_loader, device)
        print(f"Diffusion Epoch {epoch+1}/{epochs} - Train Loss: {np.mean(train_loss):.4f}, Val MSE: {val_mse:.4f}")
        
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            
    model.load_state_dict(best_state)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["ETTh1", "ETTh2", "ETTm1", "ETTm2", "exchange_rate", "weather"])
    parser.add_argument("--experiments", nargs="+", default=["baseline", "A", "B", "A+B"])
    parser.add_argument("--epochs-itrans", type=int, default=5)
    parser.add_argument("--epochs-diff", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out-dir", type=str, default="results_experimental")
    parser.add_argument("--smoke-test", action="store_true", help="Run a fast 1-epoch 1-batch smoke test")
    parser.add_argument("--job-id", type=str, default="", help="Optional job ID for parallel runs")
    args = parser.parse_args()
    
    if args.smoke_test:
        args.epochs_itrans = 1
        args.epochs_diff = 1
        args.batch_size = 2
        args.datasets = args.datasets[:1] # Just use the first dataset for smoke test
        
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    results = []
    
    for dataset_name in args.datasets:
        print(f"\\n{'='*50}\\nDataset: {dataset_name}\\n{'='*50}")
        # 1. Load Data
        from models.diffusion_tsf.train_multivariate_pipeline import get_dataset_n_cols
        n_vars = get_dataset_n_cols(dataset_name)
        
        # Dynamically adjust batch size to avoid OOM for high-variate datasets
        # Target effective batch size (B * V) of ~256 (e.g. 32 * 8 = 256)
        target_bv = 256
        effective_batch_size = max(1, min(args.batch_size, target_bv // n_vars))
        print(f"Using effective batch size {effective_batch_size} (n_vars={n_vars}) to prevent OOM.")
        
        train_ds, val_ds, test_ds, stats = load_dataset(dataset_name, lookback=512, horizon=96)
        
        if args.smoke_test:
            # Just use 2 samples
            from torch.utils.data import Subset
            train_ds = Subset(train_ds, range(min(2, len(train_ds))))
            val_ds = Subset(val_ds, range(min(2, len(val_ds))))
            test_ds = Subset(test_ds, range(min(2, len(test_ds))))
            
        train_loader = DataLoader(train_ds, batch_size=effective_batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=effective_batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=effective_batch_size, shuffle=False)
        
        # 2. Train iTransformer
        print("\\n--- Training iTransformer ---")
        itrans_model = create_itransformer(seq_len=512, pred_len=96, num_vars=n_vars)
        itrans_model = train_itransformer(itrans_model, train_loader, val_loader, epochs=args.epochs_itrans, device=device)
        
        # Wrap for diffusion
        guidance = iTransformerGuidance(itrans_model)
        
        # 3. Run Experiments
        for exp in args.experiments:
            print(f"\\n--- Experiment: {exp} ---")
            use_residual = (exp in ["A", "A+B"])
            independent_norm = (exp in ["B", "A+B"])
            
            config = ExperimentalDiffusionTSFConfig(
                num_variables=n_vars,
                lookback_length=512,
                forecast_length=96 + 8, # 8 overlap
                lookback_overlap=8,
                image_height=32,
                use_residual_diffusion=use_residual,
                independent_norm=independent_norm,
                num_diffusion_steps=1000,
                model_type="unet", # fallback to fast unet
                use_gradient_checkpointing=(n_vars >= 10)
            )
            
            diff_model = ExperimentalDiffusionTSF(config, guidance_model=guidance)
            diff_model = train_diffusion(diff_model, train_loader, val_loader, epochs=args.epochs_diff, device=device)
            
            # Evaluate detailed
            if use_residual:
                mse, mae, res_mse, trend_mse = evaluate_detailed(diff_model, test_loader, device, return_detailed=True)
                print(f"Test MSE: {mse:.4f}, MAE: {mae:.4f}, Res_MSE: {res_mse:.4f}, Trend_MSE: {trend_mse:.4f}")
                results.append({
                    "dataset": dataset_name,
                    "experiment": exp,
                    "mse": mse,
                    "mae": mae,
                    "res_mse": res_mse,
                    "trend_mse": trend_mse
                })
            else:
                mse, mae = evaluate_detailed(diff_model, test_loader, device, return_detailed=False)
                print(f"Test MSE: {mse:.4f}, MAE: {mae:.4f}")
                results.append({
                    "dataset": dataset_name,
                    "experiment": exp,
                    "mse": mse,
                    "mae": mae,
                    "res_mse": None,
                    "trend_mse": None
                })
                
            # Save results progressively to a file specific to this dataset/run
            job_suffix = f"_{args.job_id}" if args.job_id else ""
            out_file = os.path.join(args.out_dir, f"results_{dataset_name}{job_suffix}.csv")
            df = pd.DataFrame(results)
            df.to_csv(out_file, index=False)
            
    print("\\nAll experiments completed.")
    print(df)

if __name__ == "__main__":
    main()
