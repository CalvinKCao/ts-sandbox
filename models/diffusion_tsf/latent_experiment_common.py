"""
Shared helpers for train_latent_experiment (no optuna / train_multivariate_pipeline import).
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_DATASETS_DIR = _PROJECT_ROOT / "datasets"

LOOKBACK_LENGTH = 1024
FORECAST_LENGTH = 192
LOOKBACK_OVERLAP = 8
PAST_LOSS_WEIGHT = 0.3
PRETRAIN_EPOCHS = 200
PRETRAIN_PATIENCE = 20
FINETUNE_EPOCHS = 200
FINETUNE_PATIENCE = 25
SYNTHETIC_SAMPLES_FULL = 100_000

# (csv path rel to datasets/, date column, seasonal period for time features, iTransformer embed freq)
DATASET_REGISTRY = {
    "ETTh1": ("ETT-small/ETTh1.csv", "date", 24, "h"),
    "ETTh2": ("ETT-small/ETTh2.csv", "date", 24, "h"),
    "ETTm1": ("ETT-small/ETTm1.csv", "date", 96, "t"),
    "ETTm2": ("ETT-small/ETTm2.csv", "date", 96, "t"),
    "exchange_rate": ("exchange_rate/exchange_rate.csv", "date", 5, "h"),
}


def dataset_registry_row(name: str) -> Tuple[str, str, int, str]:
    row = DATASET_REGISTRY[name]
    if len(row) == 3:
        return row[0], row[1], row[2], "h"
    return row[0], row[1], row[2], row[3]


def random_exchange_variate_indices(seed: int, k: int = 7) -> List[int]:
    """exchange_rate.csv has 8 numeric series (cols 0..6 and OT); pick k=7 without replacement."""
    rng = np.random.RandomState(seed)
    all_i = np.arange(8)
    rng.shuffle(all_i)
    return sorted(all_i[:k].tolist())

logger = logging.getLogger(__name__)


class EarlyStopping:
    def __init__(self, patience: int = 25, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def amp_context():
    return nullcontext()


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,
        lookback: int = 512,
        horizon: int = 96,
        stride: int = 1,
        lookback_overlap: int = 0,
    ):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.lookback = lookback
        self.horizon = horizon
        self.stride = stride
        self.lookback_overlap = lookback_overlap
        total_len = lookback + horizon
        self.n_samples = max(0, (len(data) - total_len) // stride + 1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        past = self.data[start : start + self.lookback].T
        target_start = start + self.lookback - self.lookback_overlap
        target_end = start + self.lookback + self.horizon
        future = self.data[target_start:target_end].T
        return past, future


def load_dataset(
    dataset_name: str,
    variate_indices: Optional[List[int]] = None,
    lookback: int = LOOKBACK_LENGTH,
    horizon: int = FORECAST_LENGTH,
    stride: int = 1,
    lookback_overlap: int = LOOKBACK_OVERLAP,
) -> Tuple[Dataset, Dataset, Dataset, Dict]:
    rel, date_col, _, _ = dataset_registry_row(dataset_name)
    path = _DATASETS_DIR / rel
    df = pd.read_csv(path)
    data_cols = [c for c in df.columns if c != date_col]
    data = df[data_cols].values.astype(np.float32)
    if variate_indices is not None:
        data = data[:, variate_indices]
    # Global z-score per column on the *entire* series before train/val/test split.
    # LatentDiffusionTSF then applies a second, per-window norm (past mean/std) inside
    # _normalize_sequence. iTransformer finetuning/eval uses only this global-normalized
    # tensor from the loader — same space as guidance get_forecast(past).
    # Note: stats include future test rows (mild transductive leakage), not a train/test mismatch.
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True) + 1e-8
    data = (data - mean) / std
    n = len(data)
    total_window = lookback + horizon
    if n < total_window:
        raise ValueError(f"Dataset too short: {n} < {total_window}")
    train_end = int(n * 0.7)
    val_end = int(n * 0.8)
    train_ds = TimeSeriesDataset(data[:train_end], lookback, horizon, stride, lookback_overlap=lookback_overlap)
    val_ds = TimeSeriesDataset(
        data[train_end:val_end], lookback, horizon, stride=lookback, lookback_overlap=lookback_overlap
    )
    test_ds = TimeSeriesDataset(data[val_end:], lookback, horizon, stride, lookback_overlap=lookback_overlap)
    return train_ds, val_ds, test_ds, {"mean": mean, "std": std}


def get_itransformer_class():
    itrans_path = (_SCRIPT_DIR.parent / "iTransformer" / "model" / "iTransformer.py").resolve()
    itrans_dir = str((_SCRIPT_DIR.parent / "iTransformer").resolve())
    if itrans_dir not in sys.path:
        sys.path.insert(0, itrans_dir)
    spec = importlib.util.spec_from_file_location("iTransformer_module", str(itrans_path))
    itrans_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(itrans_module)
    return itrans_module.Model


def create_itransformer_config(
    seq_len: int = LOOKBACK_LENGTH,
    pred_len: int = FORECAST_LENGTH,
    num_vars: int = 1,
    d_model: int = 512,
    d_ff: int = 512,
    e_layers: int = 4,
    n_heads: int = 8,
    dropout: float = 0.1,
    freq: str = "h",
):
    class iTransConfig:
        def __init__(self):
            self.seq_len = seq_len
            self.pred_len = pred_len
            self.output_attention = False
            self.use_norm = True
            self.d_model = d_model
            self.d_ff = d_ff
            self.e_layers = e_layers
            self.n_heads = n_heads
            self.dropout = dropout
            self.activation = "gelu"
            self.embed = "fixed"
            self.freq = freq
            self.factor = 1
            self.enc_in = num_vars
            self.class_strategy = "projection"

    return iTransConfig()


def create_itransformer(
    seq_len: int = LOOKBACK_LENGTH,
    pred_len: int = FORECAST_LENGTH,
    num_vars: int = 1,
    dropout: float = 0.1,
    freq: str = "h",
) -> nn.Module:
    Model = get_itransformer_class()
    config = create_itransformer_config(
        seq_len=seq_len, pred_len=pred_len, num_vars=num_vars, dropout=dropout, freq=freq
    )
    return Model(config)


def _ensure_univariate_bvc(past: torch.Tensor, future: torch.Tensor) -> tuple:
    """Cached RealTS univariate batches are (B, L); iTransformer expects (B, V, L)."""
    if past.dim() == 2:
        past = past.unsqueeze(1)
    if future.dim() == 2:
        future = future.unsqueeze(1)
    return past, future


def train_itransformer_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for past, future in loader:
        past, future = _ensure_univariate_bvc(past, future)
        x_enc = past.permute(0, 2, 1).to(device)
        y_true = future.permute(0, 2, 1).to(device)
        if LOOKBACK_OVERLAP > 0:
            y_true = y_true[:, LOOKBACK_OVERLAP:, :]
        optimizer.zero_grad()
        y_pred = model(x_enc, None, None, None)
        loss = criterion(y_pred, y_true)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def validate_itransformer(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for past, future in loader:
            past, future = _ensure_univariate_bvc(past, future)
            x_enc = past.permute(0, 2, 1).to(device)
            y_true = future.permute(0, 2, 1).to(device)
            if LOOKBACK_OVERLAP > 0:
                y_true = y_true[:, LOOKBACK_OVERLAP:, :]
            y_pred = model(x_enc, None, None, None)
            loss = criterion(y_pred, y_true)
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


def save_itransformer_checkpoint(model, optimizer, epoch, train_loss, val_loss, params: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "config": params,
        },
        path,
    )


def pretrain_itransformer_dim1(
    checkpoint_dir: Path,
    n_samples: int,
    epochs: int,
    patience: int,
    cache_dir: Optional[str],
    smoke_test: bool,
) -> Path:
    from torch.utils.data import DataLoader, Subset

    from models.diffusion_tsf.dataset import get_synthetic_dataloader

    device = get_device()
    lr, batch_size, dropout = 1e-4, 64 if not smoke_test else 8, 0.1
    loader = get_synthetic_dataloader(
        batch_size=batch_size,
        lookback_length=LOOKBACK_LENGTH,
        forecast_length=FORECAST_LENGTH,
        num_variables=1,
        num_samples=n_samples,
        num_workers=0,
        lookback_overlap=LOOKBACK_OVERLAP,
        cache_dir=cache_dir if not smoke_test else None,
        skip_cross_var_aug=True,
    )
    dataset = loader.dataset
    n_val = min(len(dataset) // 10, 5000)
    train_subset = Subset(dataset, list(range(len(dataset) - n_val)))
    val_subset = Subset(dataset, list(range(len(dataset) - n_val, len(dataset))))
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = create_itransformer(num_vars=1, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    criterion = nn.MSELoss()
    early_stop = EarlyStopping(patience=patience)
    best_val = float("inf")
    ckpt_path = checkpoint_dir / "pretrained_itransformer.pt"
    params = {"learning_rate": lr, "batch_size": batch_size, "dropout": dropout}

    for epoch in range(epochs):
        train_loss = train_itransformer_epoch(model, train_loader, optimizer, criterion, device, scheduler)
        val_loss = validate_itransformer(model, val_loader, criterion, device)
        logger.info(
            "[iTransformer dim1] epoch %s/%s train=%.4f val=%.4f",
            epoch + 1,
            epochs,
            train_loss,
            val_loss,
        )
        if val_loss < best_val:
            best_val = val_loss
            save_itransformer_checkpoint(model, optimizer, epoch, train_loss, val_loss, params, ckpt_path)
        if early_stop(val_loss):
            break

    return ckpt_path


def create_pixel_diffusion_baseline(image_height: int) -> "DiffusionTSF":
    from models.diffusion_tsf.config import DiffusionTSFConfig
    from models.diffusion_tsf.diffusion_model import DiffusionTSF

    config = DiffusionTSFConfig(
        num_variables=1,
        lookback_length=LOOKBACK_LENGTH,
        forecast_length=FORECAST_LENGTH + LOOKBACK_OVERLAP,
        lookback_overlap=LOOKBACK_OVERLAP,
        past_loss_weight=PAST_LOSS_WEIGHT,
        image_height=image_height,
        use_coordinate_channel=True,
        use_guidance_channel=True,
        num_diffusion_steps=1000,
        model_type="unet",
        unet_channels=[64, 128, 256],
        attention_levels=[2],
        num_res_blocks=2,
        use_hybrid_condition=True,
        unified_time_axis=True,
    )
    return DiffusionTSF(config)
