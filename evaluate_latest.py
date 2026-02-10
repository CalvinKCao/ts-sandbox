import os
import sys
import glob
import torch
import numpy as np
import argparse
import logging
import shutil
from datetime import datetime
from torch.utils.data import DataLoader

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Imports from existing modules
from models.diffusion_tsf.train_electricity import ElectricityDataset, validate, load_itransformer_from_checkpoint
from models.diffusion_tsf.diffusion_model import DiffusionTSF
from models.diffusion_tsf.config import DiffusionTSFConfig
from models.diffusion_tsf.visualize import visualize_samples

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def find_latest_checkpoint(base_dir, dataset):
    """Find the most recent checkpoint directory for the given dataset."""
    # Pattern matches both 'universal_ETTh2' and 'universal_ETTh2_2026...'
    search_pattern = os.path.join(base_dir, f"universal_{dataset}*")
    dirs = glob.glob(search_pattern)
    dirs = [d for d in dirs if os.path.isdir(d)]
    
    if not dirs:
        return None
    
    def get_sort_key(dirname):
        """Sort by timestamp in name if present, else by mtime."""
        basename = os.path.basename(dirname)
        # Expected format: universal_{dataset}_{TIMESTAMP}
        # TIMESTAMP format: %Y%m%d_%H%M%S (15 chars)
        parts = basename.split('_')
        if len(parts) >= 3:
            # Try to parse the last two parts as date_time
            # e.g. universal_ETTh2_20260131_120000 -> parts[-2]=20260131, parts[-1]=120000
            try:
                timestamp_str = f"{parts[-2]}_{parts[-1]}"
                dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                return dt.timestamp()
            except ValueError:
                pass
        
        # Fallback to file system modification time
        return os.path.getmtime(dirname)

    # Sort by key (newest first)
    dirs.sort(key=get_sort_key, reverse=True)
    return dirs[0]

def evaluate_model(model, val_loader, device, output_dir, guidance_model=None):
    """Run full evaluation on the validation set."""
    model.eval()
    all_preds = []
    all_targets = []
    all_guidance_preds = []
    
    logger.info("Starting full evaluation on test set...")
    
    # Get forecast length from model config
    forecast_len = model.config.forecast_length
    
    with torch.no_grad():
        for i, (past, future) in enumerate(val_loader):
            past = past.to(device)
            future = future.to(device)
            
            # Generate predictions
            # Using DDIM with 50 steps for speed/quality balance
            out = model.generate(
                past, 
                use_ddim=True, 
                num_ddim_steps=50, 
                verbose=False
            )
            
            pred = out['prediction']
            all_preds.append(pred.cpu())
            all_targets.append(future.cpu())
            
            # Generate guidance predictions if available
            if guidance_model is not None:
                g_pred = guidance_model.get_forecast(past, forecast_len)
                all_guidance_preds.append(g_pred.cpu())
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1} batches...")

    preds = torch.cat(all_preds, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()
    
    # Calculate metrics
    mae = np.mean(np.abs(preds - targets))
    mse = np.mean((preds - targets) ** 2)
    rmse = np.sqrt(mse)
    
    stats_path = os.path.join(output_dir, "stats.txt")
    with open(stats_path, "w") as f:
        f.write(f"Evaluation Results\n")
        f.write(f"==================\n")
        f.write(f"Diffusion Model:\n")
        f.write(f"MAE:  {mae:.6f}\n")
        f.write(f"MSE:  {mse:.6f}\n")
        f.write(f"RMSE: {rmse:.6f}\n")
        
        if guidance_model is not None and all_guidance_preds:
            guidance_preds = torch.cat(all_guidance_preds, dim=0).numpy()
            g_mae = np.mean(np.abs(guidance_preds - targets))
            g_mse = np.mean((guidance_preds - targets) ** 2)
            g_rmse = np.sqrt(g_mse)
            
            # Calculate improvement (positive = diffusion is better)
            imp_mae = (g_mae - mae) / g_mae * 100
            imp_mse = (g_mse - mse) / g_mse * 100
            imp_rmse = (g_rmse - rmse) / g_rmse * 100
            
            f.write(f"\nGuidance (iTransformer):\n")
            f.write(f"MAE:  {g_mae:.6f}\n")
            f.write(f"MSE:  {g_mse:.6f}\n")
            f.write(f"RMSE: {g_rmse:.6f}\n")
            
            f.write(f"\nImprovement (Diffusion vs Guidance):\n")
            f.write(f"MAE:  {imp_mae:+.2f}%\n")
            f.write(f"MSE:  {imp_mse:+.2f}%\n")
            f.write(f"RMSE: {imp_rmse:+.2f}%\n")
            
            logger.info(f"Guidance - MAE: {g_mae:.4f}, RMSE: {g_rmse:.4f}")
            logger.info(f"Improvement - MAE: {imp_mae:+.2f}%, RMSE: {imp_rmse:+.2f}%")
        
    logger.info(f"Stats saved to {stats_path}")
    logger.info(f"Diffusion - MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")
    
    return mae, mse, rmse

def main():
    parser = argparse.ArgumentParser(description='Evaluate latest Diffusion TSF run')
    parser.add_argument('--dataset', type=str, default='ETTh2', help='Dataset name')
    parser.add_argument('--sync-dir', type=str, default='synced_results', help='Directory to sync results to')
    args = parser.parse_args()
    
    # Paths
    checkpoints_root = os.path.join(project_root, 'models', 'diffusion_tsf', 'checkpoints')
    
    # Find latest run
    latest_run_dir = find_latest_checkpoint(checkpoints_root, args.dataset)
    if not latest_run_dir:
        logger.error(f"No checkpoint found for dataset {args.dataset} in {checkpoints_root}")
        sys.exit(1)
        
    run_name = os.path.basename(latest_run_dir)
    logger.info(f"Found latest run: {run_name}")
    
    # Check for finetuned model
    model_path = os.path.join(latest_run_dir, 'diffusion_finetuned.pt')
    if not os.path.exists(model_path):
        logger.warning(f"diffusion_finetuned.pt not found. Checking for diffusion_synthetic.pt...")
        model_path = os.path.join(latest_run_dir, 'diffusion_synthetic.pt')
        if not os.path.exists(model_path):
            logger.error("No valid model checkpoint found!")
            sys.exit(1)
            
    logger.info(f"Evaluating model: {model_path}")
    
    # Load Data (Replicating train_universal.py split logic)
    # Determine Dataset Paths
    # Assuming DATASET_REGISTRY is available via imports or we construct path manually
    # For now, hardcoding common pattern or importing registry if needed.
    # train_universal.py uses 'datasets' folder in project root.
    
    # Quick lookup for common datasets
    dataset_map = {
        'ETTh2': 'ETT-small/ETTh2.csv',
        'ETTh1': 'ETT-small/ETTh1.csv',
        'electricity': 'electricity/electricity.csv'
    }
    
    rel_path = dataset_map.get(args.dataset, f"{args.dataset}/{args.dataset}.csv") 
    data_path = os.path.join(project_root, 'datasets', rel_path)
    
    if not os.path.exists(data_path):
         # Try to find it recursively if map failed
         found = glob.glob(os.path.join(project_root, 'datasets', '**', f'{args.dataset}.csv'), recursive=True)
         if found:
             data_path = found[0]
         else:
             logger.error(f"Could not locate data for {args.dataset}")
             sys.exit(1)
             
    logger.info(f"Loading data from {data_path}")
    
    # Setup Model Config
    # We need to reconstruct the config used for training.
    # Ideally, we should save the config in the checkpoint, but if not, we assume defaults/universal config.
    # train_universal.py hardcodes many config values.
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load checkpoint to check if config is inside
    ckpt = torch.load(model_path, map_location=device)
    
    # Try to extract config from checkpoint
    saved_config = ckpt.get('config', {})
    
    # Dataset Parameters
    lookback = saved_config.get('lookback_length', 512)
    forecast = saved_config.get('forecast_length', 96)
    logger.info(f"Using config from checkpoint: Lookback={lookback}, Forecast={forecast}")

    # Load full dataset
    full_dataset = ElectricityDataset(
        data_path, 
        lookback=lookback, 
        forecast=forecast, 
        use_all_columns=True
    )
    
    # Split indices (80/20 split as in train_universal.py)
    total_len = len(full_dataset)
    train_len = int(0.8 * total_len)
    val_indices = list(range(train_len, total_len))
    
    val_set = ElectricityDataset(
        data_path, 
        lookback=lookback, 
        forecast=forecast, 
        use_all_columns=True, 
        data_tensor=full_dataset.data, 
        indices=val_indices
    )
    
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False)
    
    num_vars = full_dataset.num_variables
    
    # Reconstruct Diffusion Config
    if saved_config:
        # Convert dict back to Config object, handling any missing fields with defaults
        # We start with a default config and update it
        diff_config = DiffusionTSFConfig(
            lookback_length=lookback,
            forecast_length=forecast,
            num_variables=num_vars
        )
        # Update attributes
        for k, v in saved_config.items():
            if hasattr(diff_config, k):
                setattr(diff_config, k, v)
    else:
        # Fallback to defaults
        logger.warning("No config found in checkpoint! Using defaults (512/96).")
        diff_config = DiffusionTSFConfig(
            lookback_length=512,
            forecast_length=96,
            model_type='unet',
            unet_channels=[64, 128, 256],
            attention_levels=[2],
            num_variables=num_vars,
            use_time_sine=True,
            use_guidance_channel=True,
            unified_time_axis=False, 
            num_diffusion_steps=200, 
            noise_schedule='linear'
        )
    
    # Guidance Setup
    # train_universal uses phase_3_itransformer_real to get a ckpt.
    # We need to load that guidance model.
    # It saves it to: os.path.join(args.checkpoints, setting, 'checkpoint.pth')
    # inside phase_3_itransformer_real.
    # The 'setting' name is f"itrans_finetuned_{args.dataset}".
    # args.checkpoints is './checkpoints/' in train_universal.py.
    
    # Wait, train_universal.py defines checkpoints dir as './checkpoints/' for iTransformer args.
    # This resolves to /home/cao/ts-sandbox/checkpoints/ (relative to execution dir).
    itrans_ckpt_path = os.path.join(project_root, 'checkpoints', f'itrans_finetuned_{args.dataset}', 'checkpoint.pth')
    
    guidance_model = None
    if os.path.exists(itrans_ckpt_path):
        logger.info(f"Loading guidance from {itrans_ckpt_path}")
        guidance_model = load_itransformer_from_checkpoint(
            itrans_ckpt_path,
            seq_len=lookback,
            pred_len=forecast,
            num_variables=num_vars,
            device=device
        )
    else:
        logger.warning("Guidance checkpoint not found! Evaluation will proceed without guidance (performance may drop).")

    # Initialize Model
    model = DiffusionTSF(diff_config, guidance_model=guidance_model).to(device)
    
    # Load weights
    # The checkpoint from train_universal.py (save_checkpoint) saves dict with 'model_state_dict'
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt) # Fallback if direct state dict
        
    logger.info("Model loaded successfully.")
    
    # Create output directory for this run
    output_dir = os.path.join(latest_run_dir, 'eval_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Run Evaluation
    evaluate_model(model, val_loader, device, output_dir, guidance_model=guidance_model)
    
    # Generate Visualizations
    logger.info("Generating visualizations...")
    visualize_samples(
        model_path=model_path,
        data_path=data_path,
        num_samples=5,
        device=device,
        output_dir=output_dir,
        # We pass guidance checkpoint if we have it, visualize_samples loads it internally
        guidance_checkpoint=itrans_ckpt_path if os.path.exists(itrans_ckpt_path) else None
    )
    
    # Sync results
    sync_dest = os.path.join(project_root, args.sync_dir, run_name)
    os.makedirs(sync_dest, exist_ok=True)
    
    # Copy all files from output_dir to sync_dest
    for f in os.listdir(output_dir):
        src = os.path.join(output_dir, f)
        dst = os.path.join(sync_dest, f)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            
    logger.info("="*60)
    logger.info("Evaluation Complete.")
    logger.info(f"Results synced to: {sync_dest}")
    logger.info(f"Original results in: {output_dir}")
    logger.info("="*60)

if __name__ == '__main__':
    main()
