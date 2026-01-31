
import argparse
import logging
import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from datetime import datetime
from dataclasses import asdict

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# iTransformer imports
itrans_dir = os.path.join(project_root, 'models', 'iTransformer')
if itrans_dir not in sys.path:
    sys.path.append(itrans_dir)

try:
    from models.iTransformer.experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
    from models.iTransformer.utils.tools import dotdict
except ImportError:
    # Fallback
    try:
        from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
        from utils.tools import dotdict
    except ImportError:
        # Last resort if iTransformer is not in path correctly
        sys.path.append(os.path.join(project_root, 'models', 'iTransformer'))
        from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
        from utils.tools import dotdict

# Diffusion imports
from models.diffusion_tsf.dataset import get_synthetic_dataloader
from models.diffusion_tsf.diffusion_model import DiffusionTSF
from models.diffusion_tsf.config import DiffusionTSFConfig
from models.diffusion_tsf.train_electricity import (
    train_epoch, validate, save_checkpoint, EarlyStopping, 
    ElectricityDataset, DATASET_REGISTRY, load_itransformer_from_checkpoint,
    TrainingConfig
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)],
    force=True
)
logger = logging.getLogger(__name__)

# ============================================================================
# Adapters
# ============================================================================

class Exp_Synthetic(Exp_Long_Term_Forecast):
    """Adapter to train iTransformer on in-memory synthetic dataloader."""
    def __init__(self, args, synthetic_loader):
        self.synthetic_loader = synthetic_loader
        # super().__init__ calls Exp_Basic.__init__ which sets self.args and self.device
        super().__init__(args)
            
    def _get_data(self, flag):
        return None, self.synthetic_loader

class SyntheticAdapter:
    def __init__(self, loader, args, device):
        self.loader = loader
        self.args = args
        self.device = device
        
    def __iter__(self):
        for past, future in self.loader:
            B, N, L = past.shape
            F_len = future.shape[-1]
            
            # Synthetic data from RealTS is (B, N, L).
            # iTransformer expects (B, L, N).
            batch_x = past.transpose(1, 2).to(self.device)
            future_x = future.transpose(1, 2).to(self.device)
            
            label_len = self.args.label_len
            if L >= label_len:
                label_part = batch_x[:, -label_len:, :]
            else:
                label_part = torch.cat([torch.zeros(B, label_len-L, N, device=self.device), batch_x], dim=1)
                
            batch_y = torch.cat([label_part, future_x], dim=1)
            
            batch_x_mark = torch.zeros(B, L, 4, device=self.device)
            batch_y_mark = torch.zeros(B, label_len+F_len, 4, device=self.device)
            
            yield batch_x, batch_y, batch_x_mark, batch_y_mark
            
    def __len__(self):
        return len(self.loader)

# ============================================================================
# Phases
# ============================================================================

def phase_1_itransformer_synthetic(args_itrans, synthetic_loader, checkpoint_dir):
    logger.info("="*60)
    logger.info("PHASE 1: iTransformer Pre-training (Synthetic)")
    logger.info("="*60)
    
    device = torch.device(f'cuda:{args_itrans.gpu}' if args_itrans.use_gpu else 'cpu')
    adapter = SyntheticAdapter(synthetic_loader, args_itrans, device)
    
    exp = Exp_Synthetic(args_itrans, adapter)
    setting = f"itrans_synthetic_{args_itrans.model_id}"
    exp.train(setting)
    
    ckpt_path = os.path.join('./checkpoints', setting, 'checkpoint.pth')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(args_itrans.checkpoints, setting, 'checkpoint.pth')
        
    logger.info(f"Phase 1 Complete. Checkpoint: {ckpt_path}")
    return ckpt_path

def phase_2_diffusion_synthetic(config, synthetic_loader, guidance_ckpt, checkpoint_dir, lr=1e-4, epochs=10):
    logger.info("="*60)
    logger.info("PHASE 2: Diffusion Pre-training (Synthetic)")
    logger.info("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    guidance = load_itransformer_from_checkpoint(
        guidance_ckpt, 
        seq_len=config.lookback_length,
        pred_len=config.forecast_length,
        num_variables=config.num_variables,
        device=device
    )
    
    model = DiffusionTSF(config, guidance_model=guidance).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    save_path = os.path.join(checkpoint_dir, 'diffusion_synthetic.pt')
    
    model.train()
    for epoch in range(epochs):
        loss = train_epoch(model, synthetic_loader, optimizer, device, epoch)
        logger.info(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f}")
        
    save_checkpoint(model, optimizer, epochs, loss, loss, asdict(config), save_path)
    logger.info(f"Phase 2 Complete. Checkpoint: {save_path}")
    return save_path

def phase_3_itransformer_real(args_itrans, real_train_loader, real_val_loader, pretrained_ckpt):
    logger.info("="*60)
    logger.info("PHASE 3: iTransformer Fine-tuning (Real)")
    logger.info("="*60)
    
    exp = Exp_Long_Term_Forecast(args_itrans)
    
    if os.path.exists(pretrained_ckpt):
        logger.info(f"Loading weights from {pretrained_ckpt}")
        try:
            exp.model.load_state_dict(torch.load(pretrained_ckpt))
        except Exception as e:
            logger.warning(f"Failed to load weights: {e}. Starting fresh.")
    
    setting = f"itrans_finetuned_{args_itrans.model_id}"
    exp.train(setting)
    
    ckpt_path = os.path.join(args_itrans.checkpoints, setting, 'checkpoint.pth')
    logger.info(f"Phase 3 Complete. Checkpoint: {ckpt_path}")
    return ckpt_path

def phase_4_diffusion_real(config, real_train_loader, real_val_loader, diff_pretrained_ckpt, itrans_finetuned_ckpt, checkpoint_dir, epochs=20, lr=1e-4):
    logger.info("="*60)
    logger.info("PHASE 4: Diffusion Fine-tuning (Real)")
    logger.info("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    guidance = load_itransformer_from_checkpoint(
        itrans_finetuned_ckpt,
        seq_len=config.lookback_length,
        pred_len=config.forecast_length,
        num_variables=config.num_variables,
        device=device
    )
    
    model = DiffusionTSF(config, guidance_model=guidance).to(device)
    
    if os.path.exists(diff_pretrained_ckpt):
        logger.info(f"Loading diffusion weights from {diff_pretrained_ckpt}")
        ckpt = torch.load(diff_pretrained_ckpt, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr * 0.1) # Lower LR for finetune
    
    save_path = os.path.join(checkpoint_dir, 'diffusion_finetuned.pt')
    
    best_loss = float('inf')
    early_stopping = EarlyStopping(patience=5)
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, real_train_loader, optimizer, device, epoch)
        val_metrics = validate(model, real_val_loader, device)
        val_loss = val_metrics['val_loss']
        
        logger.info(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss, asdict(config), save_path)
            
        if early_stopping(val_loss):
            break
            
    logger.info(f"Phase 4 Complete. Best Loss: {best_loss:.4f}")
    return save_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ETTh2')
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--repr-mode', type=str, default='cdf', choices=['pdf', 'cdf'], help='Data representation mode')
    parser.add_argument('--unified-time-axis', action='store_true', default=False, help='Enable Unified L+F time axis (Slower)')
    parser.add_argument('--synthetic-pool-size', type=int, default=200000, help='Size of synthetic data pool')
    parser.add_argument('--synthetic-cache-dir', type=str, default='./data_cache', help='Cache directory for synthetic data')
    args = parser.parse_args()
    
    # Load Best Params
    params_path = os.path.join(project_root, 'legacy', 'params', 'best_params.json')
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            best_params = json.load(f)
    else:
        best_params = {}
    
    lr = best_params.get('learning_rate', 1e-4)
    
    # Determine Dataset Paths
    if args.dataset in DATASET_REGISTRY:
        rel_path, _, _ = DATASET_REGISTRY[args.dataset]
        data_path = os.path.join(project_root, 'datasets', rel_path)
        root_path = os.path.dirname(data_path)
        data_file = os.path.basename(data_path)
    else:
        logger.error(f"Dataset {args.dataset} not found in registry.")
        sys.exit(1)
        
    # Create Real Data Loaders Manually
    logger.info(f"Loading real dataset from {data_path}")
    real_dataset = ElectricityDataset(
        data_path, 
        lookback=512, 
        forecast=96, 
        use_all_columns=True, 
        max_samples=20 if args.smoke_test else None
    )
    num_vars = real_dataset.num_variables
    
    # Config for Diffusion
    diff_config = DiffusionTSFConfig(
        lookback_length=512 if not args.smoke_test else 64,
        forecast_length=96 if not args.smoke_test else 16,
        model_type='unet',
        unet_channels=[64, 128, 256] if not args.smoke_test else [32, 64],
        attention_levels=[2] if not args.smoke_test else [1], # Only attention at bottleneck
        num_variables=num_vars, # Updated later
        representation_mode=args.repr_mode,
        use_time_sine=True,
        use_guidance_channel=True,
        unified_time_axis=args.unified_time_axis,
        num_diffusion_steps=best_params.get('diffusion_steps', 200) if not args.smoke_test else 10,
        noise_schedule=best_params.get('noise_schedule', 'linear')
    )
    
    # Split
    total_len = len(real_dataset)
    train_len = int(0.8 * total_len)
    
    indices = list(range(len(real_dataset)))
    train_indices = indices[:train_len]
    val_indices = indices[train_len:]
    
    # Create smaller datasets for smoke test
    lookback = 512 if not args.smoke_test else 64
    forecast = 96 if not args.smoke_test else 16
    
    train_set = ElectricityDataset(data_path, lookback=lookback, forecast=forecast, use_all_columns=True, data_tensor=real_dataset.data, indices=train_indices)
    val_set = ElectricityDataset(data_path, lookback=lookback, forecast=forecast, use_all_columns=True, data_tensor=real_dataset.data, indices=val_indices)
    
    real_train_loader = DataLoader(train_set, batch_size=4 if not args.smoke_test else 2, shuffle=True)
    real_val_loader = DataLoader(val_set, batch_size=4 if not args.smoke_test else 2)
    
    # Params for iTransformer
    itrans_args = dotdict()
    itrans_args.model_id = args.dataset
    itrans_args.model = 'iTransformer'
    itrans_args.data = 'fair'
    itrans_args.target = 'OT' # Explicitly set target
    itrans_args.features = 'M'
    itrans_args.freq = 'h' # Explicitly set frequency
    itrans_args.seq_len = 512 if not args.smoke_test else 64
    itrans_args.pred_len = 96 if not args.smoke_test else 16
    itrans_args.label_len = 48 if not args.smoke_test else 16
    itrans_args.e_layers = 2 if not args.smoke_test else 1
    itrans_args.d_layers = 1
    itrans_args.factor = 1
    itrans_args.enc_in = num_vars
    itrans_args.dec_in = num_vars
    itrans_args.c_out = num_vars
    itrans_args.d_model = 256 if not args.smoke_test else 64
    itrans_args.n_heads = 8 if not args.smoke_test else 4
    itrans_args.d_ff = 1024 if not args.smoke_test else 128
    itrans_args.dropout = 0.1
    itrans_args.attn = 'prob'
    itrans_args.embed = 'timeF'
    itrans_args.activation = 'gelu'
    itrans_args.output_attention = False
    itrans_args.do_predict = False
    itrans_args.mix = True
    itrans_args.cols = None
    itrans_args.num_workers = 0
    itrans_args.itr = 1
    itrans_args.train_epochs = 10 if not args.smoke_test else 1
    itrans_args.batch_size = 8 if not args.smoke_test else 2
    itrans_args.patience = 3
    itrans_args.learning_rate = 0.0001
    itrans_args.des = 'test'
    itrans_args.loss = 'MSE'
    itrans_args.lradj = 'type1'
    itrans_args.use_amp = False
    itrans_args.use_gpu = torch.cuda.is_available() if not args.smoke_test else False
    itrans_args.gpu = args.gpu
    itrans_args.use_multi_gpu = False
    itrans_args.devices = '0'
    itrans_args.checkpoints = './checkpoints/'
    itrans_args.root_path = root_path
    itrans_args.data_path = data_file
    
    # Synthetic Data
    syn_samples = 10000 if not args.smoke_test else 4
    synthetic_loader = get_synthetic_dataloader(
        num_samples=syn_samples,
        lookback_length=512 if not args.smoke_test else 64,
        forecast_length=96 if not args.smoke_test else 16,
        batch_size=4 if not args.smoke_test else 2, # Reduced for OOM
        num_variables=num_vars,
        pool_size=args.synthetic_pool_size if not args.smoke_test else 10,
        cache_dir=args.synthetic_cache_dir
    )
    
    # Create timestamped checkpoint directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_dir = os.path.join(project_root, 'models', 'diffusion_tsf', 'checkpoints', f'universal_{args.dataset}_{timestamp}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")
    
    # EXECUTE PHASES
    ckpt_1 = phase_1_itransformer_synthetic(itrans_args, synthetic_loader, checkpoint_dir)
    
    epochs_phase2 = 10 if not args.smoke_test else 1
    epochs_phase4 = 20 if not args.smoke_test else 1
    
    ckpt_2 = phase_2_diffusion_synthetic(diff_config, synthetic_loader, ckpt_1, checkpoint_dir, lr=lr, epochs=epochs_phase2)
    
    ckpt_3 = phase_3_itransformer_real(itrans_args, real_train_loader, real_val_loader, ckpt_1)
    
    phase_4_diffusion_real(diff_config, real_train_loader, real_val_loader, ckpt_2, ckpt_3, checkpoint_dir, epochs=epochs_phase4, lr=lr)

if __name__ == '__main__':
    main()