"""
CI latent diffusion on ETTh2 — full 4-stage pipeline with proper iTransformer finetuning.

Pipeline:
  Stage 0: VAE pretrain (univariate, reuse if exists)
  Stage 1: Pretrain iTransformer on synthetic 7-var data
  Stage 2: Pretrain latent diffusion on synthetic 1-var, guided by pretrained iTransformer
  Stage 3: Finetune iTransformer on ETTh2
  Stage 4: Finetune latent diffusion on ETTh2, guided by *finetuned* iTransformer
           + eval vs finetuned iTransformer baseline

Important: the pretrained iTransformer is ONLY used during stage 2 (synthetic diffusion
pretraining). After stage 3, all downstream work uses the ETTh2-finetuned iTransformer.

Run from repo root:
  python -m models.diffusion_tsf.train_ci_latent_etth2 --smoke-test
  python -m models.diffusion_tsf.train_ci_latent_etth2 --stage all
  python -m models.diffusion_tsf.train_ci_latent_etth2 --stage 3   # just finetune itrans
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

try:
    import wandb
    WANDB_OK = True
except ImportError:
    wandb = None
    WANDB_OK = False

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.diffusion_tsf.config import LatentDiffusionConfig
from models.diffusion_tsf.dataset import get_synthetic_dataloader
from models.diffusion_tsf.guidance import BaseGuidance, LinearRegressionGuidance
from models.diffusion_tsf.latent_diffusion_model import LatentDiffusionTSF
from models.diffusion_tsf.latent_experiment_common import (
    FINETUNE_EPOCHS,
    FINETUNE_PATIENCE,
    FORECAST_LENGTH,
    LOOKBACK_LENGTH,
    LOOKBACK_OVERLAP,
    PAST_LOSS_WEIGHT,
    PRETRAIN_EPOCHS,
    PRETRAIN_PATIENCE,
    SYNTHETIC_SAMPLES_FULL,
    EarlyStopping,
    amp_context,
    create_itransformer,
    get_device,
    load_dataset,
    save_itransformer_checkpoint,
    train_itransformer_epoch,
    validate_itransformer,
)
from models.diffusion_tsf.metrics import compute_metrics
from models.diffusion_tsf.preprocessing import TimeSeriesTo2D, VerticalGaussianBlur
from models.diffusion_tsf.vae import TimeSeriesVAE, estimate_vae_scale_factor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
logger = logging.getLogger(__name__)

CKPT_LATENT = _SCRIPT_DIR / "checkpoints_latent"
CKPT_DIR = _SCRIPT_DIR / "checkpoints_ci_etth2"
RESULTS_DIR = _SCRIPT_DIR / "results"

N_VARIATES = 7
DATASET = "ETTh2"
WANDB_PROJECT = "diffusion-tsf"

_wb_enabled = False


def _wb_init(cfg, args):
    """Start a wandb run. Resumes from saved run_id if available."""
    global _wb_enabled
    if not WANDB_OK or args.smoke_test:
        return
    run_id_file = CKPT_DIR / "wandb_run_id.txt"
    run_id, resume = None, None
    if run_id_file.is_file():
        run_id = run_id_file.read_text().strip()
        resume = "allow"
        logger.info("Resuming wandb run %s", run_id)

    arch_text = ""
    arch_path = _PROJECT_ROOT / "arch.md"
    if arch_path.is_file():
        arch_text = arch_path.read_text()[:3000]

    run_name = f"{DATASET}-ci-latent-4stage-H{args.image_height}"
    try:
        wandb.init(
            project=WANDB_PROJECT,
            name=run_name,
            id=run_id,
            resume=resume,
            config={
                "dataset": DATASET,
                "pipeline": "ci-latent-4stage",
                "num_variates": N_VARIATES,
                "image_height": cfg.image_height,
                "latent_channels": cfg.latent_channels,
                "lookback_length": LOOKBACK_LENGTH,
                "forecast_length": FORECAST_LENGTH,
                "lookback_overlap": LOOKBACK_OVERLAP,
                "past_loss_weight": PAST_LOSS_WEIGHT,
                "unet_channels": cfg.unet_channels,
                "attention_levels": cfg.attention_levels,
                "num_res_blocks": cfg.num_res_blocks,
                "diffusion_steps": cfg.num_diffusion_steps,
                "representation_mode": cfg.representation_mode,
                "vae_lr": cfg.vae_lr,
                "diffusion_lr": cfg.learning_rate,
                "pretrain_epochs": PRETRAIN_EPOCHS,
                "finetune_epochs": FINETUNE_EPOCHS,
                "cfg_scale": cfg.cfg_scale,
                "architecture": arch_text,
            },
            tags=[DATASET, "ci-latent", "4stage"],
        )
        # persist run_id for resume
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        run_id_file.write_text(wandb.run.id)
        _wb_enabled = True
        logger.info("wandb run: %s", wandb.run.url)
    except Exception as e:
        logger.warning("wandb init failed: %s", e)
        _wb_enabled = False


def _wb_log(metrics: dict, step: int | None = None):
    if _wb_enabled:
        wandb.log(metrics, step=step)


def _wb_summary(metrics: dict):
    if _wb_enabled:
        for k, v in metrics.items():
            wandb.run.summary[k] = v


def _wb_finish():
    global _wb_enabled
    if _wb_enabled:
        wandb.finish()
        _wb_enabled = False


# ---------------------------------------------------------------------------
# CI guidance wrapper (same as train_ci_latent_etth1)
# ---------------------------------------------------------------------------

class CIiTransformerGuidance(BaseGuidance):
    """Wraps a multivariate iTransformer for channel-independent inference.

    Past arrives as (B*V, L) from the per-variate diffusion model. This wrapper
    unflattens to (B, V, L), runs the full multivariate model, then flattens
    per-variate forecasts back to (B*V, H).
    """

    def __init__(self, model: nn.Module, num_variates: int,
                 seq_len: int | None = None, pred_len: int | None = None):
        super().__init__()
        self.model = model
        self.num_variates = num_variates
        self.seq_len = seq_len
        self.pred_len = pred_len
        for p in self.model.parameters():
            p.requires_grad = False
        self.training = False
        self.model.eval()

    def train(self, mode: bool = True):
        self.training = False
        self.model.eval()
        return self

    def eval(self):
        self.training = False
        self.model.eval()
        return self

    @torch.no_grad()
    def get_forecast(self, past: torch.Tensor, forecast_length: int) -> torch.Tensor:
        if self.pred_len is not None and forecast_length != self.pred_len:
            raise ValueError(
                f"iTransformer pred_len={self.pred_len}, got forecast_length={forecast_length}"
            )
        BV, L = past.shape
        V = self.num_variates
        B = BV // V
        x_enc = past.reshape(B, V, L).permute(0, 2, 1)
        y_pred = self.model(x_enc, None, None, None)
        if isinstance(y_pred, tuple):
            y_pred = y_pred[0]
        return y_pred.permute(0, 2, 1).reshape(BV, -1)[:, :forecast_length]


# ---------------------------------------------------------------------------
# pixel encoder for VAE training
# ---------------------------------------------------------------------------

class PixelEncoder(nn.Module):
    def __init__(self, cfg: LatentDiffusionConfig):
        super().__init__()
        self.to_2d = TimeSeriesTo2D(
            height=cfg.image_height, max_scale=cfg.max_scale,
            representation_mode=cfg.representation_mode,
        )
        self.blur = VerticalGaussianBlur(kernel_size=cfg.blur_kernel_size, sigma=cfg.blur_sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        im = self.to_2d(x)
        b = self.blur(im)
        if self.to_2d.representation_mode == "pdf":
            return b * 30.0 * 2.0 - 1.0
        return b.clamp(0.0, 1.0) * 2.0 - 1.0


def _unified_2d(pixel_enc: PixelEncoder, past: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
    mean = past.mean(dim=-1, keepdim=True)
    std = past.std(dim=-1, keepdim=True) + 1e-8
    pn = (past - mean) / std
    fn = (future - mean) / std
    if pn.dim() == 2:
        pn, fn = pn.unsqueeze(1), fn.unsqueeze(1)
    return torch.cat([pixel_enc(pn), pixel_enc(fn)], dim=-1)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def build_config(image_height: int) -> LatentDiffusionConfig:
    horizon_pixels = FORECAST_LENGTH + LOOKBACK_OVERLAP
    return LatentDiffusionConfig(
        num_variables=1,
        lookback_length=LOOKBACK_LENGTH,
        forecast_length=horizon_pixels,
        lookback_overlap=LOOKBACK_OVERLAP,
        past_loss_weight=PAST_LOSS_WEIGHT,
        image_height=image_height,
        representation_mode="cdf",
        unified_time_axis=True,
        use_coordinate_channel=True,
        use_time_ramp=False,
        use_time_sine=False,
        use_value_channel=False,
        use_guidance_channel=True,
        use_hybrid_condition=True,
        seasonal_period=24,
        unet_channels=[64, 128, 256],
        attention_levels=[2],
        num_res_blocks=2,
        num_diffusion_steps=1000,
        model_type="unet",
        cfg_scale=1.0,
        emd_lambda=0.0,
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _scalar(x):
    return float(x.detach().cpu().item()) if torch.is_tensor(x) else float(x)


def _train_ld_epoch(model, loader, optimizer, device):
    model.train()
    total, n = 0.0, 0
    for past, future in loader:
        past, future = past.to(device), future.to(device)
        optimizer.zero_grad()
        with amp_context():
            loss = model.get_loss(past, future)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), 1.0
        )
        optimizer.step()
        total += loss.item(); n += 1
    return total / max(n, 1)


@torch.no_grad()
def _eval_ld(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    for past, future in loader:
        past, future = past.to(device), future.to(device)
        with amp_context():
            total += model.get_loss(past, future).item()
        n += 1
    return total / max(n, 1)


def _ci_train_epoch(model, loader, optimizer, device, V):
    model.train()
    total, n = 0.0, 0
    for past, future in loader:
        B = past.shape[0]
        past_flat = past.reshape(B * V, -1).to(device)
        future_flat = future.reshape(B * V, -1).to(device)
        optimizer.zero_grad()
        with amp_context():
            loss = model.get_loss(past_flat, future_flat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), 1.0
        )
        optimizer.step()
        total += loss.item(); n += 1
    return total / max(n, 1)


@torch.no_grad()
def _ci_eval_loss(model, loader, device, V):
    model.eval()
    total, n = 0.0, 0
    for past, future in loader:
        B = past.shape[0]
        past_flat = past.reshape(B * V, -1).to(device)
        future_flat = future.reshape(B * V, -1).to(device)
        with amp_context():
            total += model.get_loss(past_flat, future_flat).item()
        n += 1
    return total / max(n, 1)


def _build_ci_guidance(itrans_ckpt: Path, device: torch.device) -> CIiTransformerGuidance:
    model = create_itransformer(num_vars=N_VARIATES, pred_len=FORECAST_LENGTH).to(device)
    ckpt = torch.load(itrans_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return CIiTransformerGuidance(
        model, num_variates=N_VARIATES,
        seq_len=LOOKBACK_LENGTH, pred_len=FORECAST_LENGTH,
    )


# ---------------------------------------------------------------------------
# stage 0: VAE (reuse from 1-var experiment)
# ---------------------------------------------------------------------------

def stage0_vae(
    cfg: LatentDiffusionConfig,
    device: torch.device,
    n_samples: int,
    cache_dir: Optional[str],
    smoke_test: bool,
) -> TimeSeriesVAE:
    vae_path = CKPT_LATENT / f"vae_dim1_H{cfg.image_height}.pt"
    CKPT_LATENT.mkdir(parents=True, exist_ok=True)

    if vae_path.is_file():
        logger.info("Loading existing VAE from %s", vae_path)
        ckpt = torch.load(vae_path, map_location=device, weights_only=False)
        vae = TimeSeriesVAE(latent_channels=cfg.latent_channels).to(device)
        vae.load_state_dict(ckpt["vae_state_dict"])
        if "scale_factor" in ckpt:
            vae.scale_factor.copy_(ckpt["scale_factor"].to(device).view_as(vae.scale_factor))
        return vae

    logger.info("No VAE at %s — training from scratch", vae_path)
    epochs = 2 if smoke_test else cfg.vae_epochs
    patience = 1 if smoke_test else 15

    loader = get_synthetic_dataloader(
        num_samples=n_samples, lookback_length=LOOKBACK_LENGTH,
        forecast_length=FORECAST_LENGTH, batch_size=16, num_workers=0,
        num_variables=1, pool_size=n_samples, cache_dir=cache_dir,
        lookback_overlap=LOOKBACK_OVERLAP, skip_cross_var_aug=True,
    )
    ds = loader.dataset
    n_val = min(len(ds) // 10, 2000)
    train_dl = DataLoader(Subset(ds, list(range(len(ds) - n_val))), batch_size=16, shuffle=True)
    val_dl = DataLoader(Subset(ds, list(range(len(ds) - n_val, len(ds)))), batch_size=16)

    pixel_enc = PixelEncoder(cfg).to(device).eval()
    vae = TimeSeriesVAE(latent_channels=cfg.latent_channels).to(device)
    opt = torch.optim.AdamW(vae.parameters(), lr=cfg.vae_lr)
    early = EarlyStopping(patience=patience)
    best_val = float("inf")

    for ep in range(epochs):
        vae.train()
        tr, nb = 0.0, 0
        for past, future in train_dl:
            past, future = past.to(device), future.to(device)
            with torch.no_grad():
                img = _unified_2d(pixel_enc, past, future)
            opt.zero_grad()
            out = vae(img, kl_weight=cfg.kl_weight)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            opt.step()
            tr += out["loss"].item(); nb += 1
        vae.eval()
        va, nv = 0.0, 0
        with torch.no_grad():
            for past, future in val_dl:
                past, future = past.to(device), future.to(device)
                out = vae(_unified_2d(pixel_enc, past, future), kl_weight=cfg.kl_weight)
                va += out["loss"].item(); nv += 1
        tr /= max(nb, 1); va /= max(nv, 1)
        logger.info("VAE ep %d  train=%.4f  val=%.4f", ep + 1, tr, va)
        if va < best_val:
            best_val = va
            imgs = []
            with torch.no_grad():
                for p, f in val_dl:
                    imgs.append(_unified_2d(pixel_enc, p.to(device), f.to(device)))
                    if len(imgs) >= 4:
                        break
            estimate_vae_scale_factor(vae, torch.cat(imgs, 0))
            torch.save({
                "vae_state_dict": vae.state_dict(),
                "scale_factor": vae.scale_factor.detach().cpu(),
                "latent_channels": cfg.latent_channels,
                "image_height": cfg.image_height,
            }, vae_path)
        if early(va):
            break

    ckpt = torch.load(vae_path, map_location=device, weights_only=False)
    vae.load_state_dict(ckpt["vae_state_dict"])
    vae.scale_factor.copy_(ckpt["scale_factor"].to(device).view_as(vae.scale_factor))
    return vae


# ---------------------------------------------------------------------------
# stage 1: pretrain iTransformer on synthetic 7-var
# ---------------------------------------------------------------------------

def stage1_pretrain_itransformer(
    device: torch.device,
    n_samples: int,
    cache_dir: Optional[str],
    smoke_test: bool,
) -> Path:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / "itransformer_pretrained_synth.pt"
    if ckpt_path.is_file():
        logger.info("Pretrained iTransformer exists: %s", ckpt_path)
        return ckpt_path

    logger.info("=== Stage 1: pretrain iTransformer on synthetic 7-var ===")
    loader = get_synthetic_dataloader(
        num_samples=n_samples, lookback_length=LOOKBACK_LENGTH,
        forecast_length=FORECAST_LENGTH, batch_size=64 if not smoke_test else 8,
        num_workers=0, num_variables=N_VARIATES, pool_size=n_samples,
        cache_dir=cache_dir, lookback_overlap=LOOKBACK_OVERLAP,
        skip_cross_var_aug=True,
    )
    ds = loader.dataset
    n_val = min(len(ds) // 10, 5000)
    train_dl = DataLoader(Subset(ds, list(range(len(ds) - n_val))),
                          batch_size=64 if not smoke_test else 8, shuffle=True)
    val_dl = DataLoader(Subset(ds, list(range(len(ds) - n_val, len(ds)))),
                        batch_size=64 if not smoke_test else 8)

    model = create_itransformer(num_vars=N_VARIATES, pred_len=FORECAST_LENGTH).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=PRETRAIN_EPOCHS if not smoke_test else 3, eta_min=1e-6)
    criterion = nn.MSELoss()
    epochs = 3 if smoke_test else PRETRAIN_EPOCHS
    patience = 1 if smoke_test else PRETRAIN_PATIENCE
    early = EarlyStopping(patience=patience)
    best_val = float("inf")

    for ep in range(epochs):
        tr = train_itransformer_epoch(model, train_dl, opt, criterion, device, sched)
        va = validate_itransformer(model, val_dl, criterion, device)
        logger.info("[iTrans pretrain synth] ep %d  train=%.4f  val=%.4f", ep + 1, tr, va)
        _wb_log({"s1_itrans_pretrain/train_loss": tr, "s1_itrans_pretrain/val_loss": va})
        if va < best_val:
            best_val = va
            save_itransformer_checkpoint(
                model, opt, ep, tr, va,
                {"num_variates": N_VARIATES, "lr": 1e-4, "stage": "pretrain_synth"},
                ckpt_path,
            )
        if early(va):
            break

    _wb_summary({"s1_itrans_pretrain/best_val": best_val})
    logger.info("Pretrained iTransformer → %s", ckpt_path)
    return ckpt_path


# ---------------------------------------------------------------------------
# stage 2: pretrain latent diffusion on synthetic, guided by pretrained iTrans
# ---------------------------------------------------------------------------

def stage2_pretrain_diffusion(
    cfg: LatentDiffusionConfig,
    vae: TimeSeriesVAE,
    itrans_pretrained: Path,
    device: torch.device,
    n_samples: int,
    cache_dir: Optional[str],
    smoke_test: bool,
) -> Path:
    out_path = CKPT_DIR / f"diffusion_pretrained_H{cfg.image_height}.pt"
    if out_path.is_file():
        logger.info("Pretrained diffusion exists: %s", out_path)
        return out_path

    logger.info("=== Stage 2: pretrain latent diffusion (guided by pretrained iTrans) ===")

    # guidance from pretrained synthetic iTransformer — only used during this stage
    guidance = _build_ci_guidance(itrans_pretrained, device)
    model = LatentDiffusionTSF(cfg, vae, guidance_model=guidance).to(device)

    # 7-var synthetic data so the CI guidance wrapper can unflatten properly
    # bs=2 for smoke: each batch flattens to 2*7=14 univariate forward passes
    bs_real = 8 if not smoke_test else 2
    loader = get_synthetic_dataloader(
        num_samples=n_samples, lookback_length=LOOKBACK_LENGTH,
        forecast_length=FORECAST_LENGTH, batch_size=bs_real, num_workers=0,
        num_variables=N_VARIATES, pool_size=n_samples, cache_dir=cache_dir,
        lookback_overlap=LOOKBACK_OVERLAP, skip_cross_var_aug=True,
    )
    ds = loader.dataset
    n_val = min(len(ds) // 10, 1000)
    if smoke_test:
        n_train = min(6, len(ds) - n_val)
        n_val = min(2, n_val)
        train_dl = DataLoader(Subset(ds, list(range(n_train))), batch_size=bs_real, shuffle=True)
        val_dl = DataLoader(Subset(ds, list(range(len(ds) - n_val, len(ds)))), batch_size=bs_real)
    else:
        train_dl = DataLoader(Subset(ds, list(range(len(ds) - n_val))), batch_size=bs_real, shuffle=True)
        val_dl = DataLoader(Subset(ds, list(range(len(ds) - n_val, len(ds)))), batch_size=bs_real)

    epochs = 2 if smoke_test else PRETRAIN_EPOCHS
    patience = 1 if smoke_test else PRETRAIN_PATIENCE
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.learning_rate)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=cfg.learning_rate * 0.01)
    early = EarlyStopping(patience=patience)
    best_val = float("inf")

    for ep in range(epochs):
        tr = _ci_train_epoch(model, train_dl, opt, device, N_VARIATES)
        va = _ci_eval_loss(model, val_dl, device, N_VARIATES)
        sched.step()
        logger.info("[Diffusion pretrain] ep %d  train=%.4f  val=%.4f", ep + 1, tr, va)
        _wb_log({"s2_diff_pretrain/train_loss": tr, "s2_diff_pretrain/val_loss": va})
        if va < best_val:
            best_val = va
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": asdict(cfg),
                "vae_scale_factor": vae.scale_factor.detach().cpu(),
                "guidance_source": "pretrained_synth_itransformer",
            }, out_path)
        if early(va):
            break

    _wb_summary({"s2_diff_pretrain/best_val": best_val})
    logger.info("Pretrained diffusion → %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# stage 3: finetune iTransformer on ETTh2
# ---------------------------------------------------------------------------

def stage3_finetune_itransformer(
    itrans_pretrained: Path,
    device: torch.device,
    smoke_test: bool,
) -> Path:
    ckpt_path = CKPT_DIR / f"itransformer_finetuned_{DATASET}.pt"
    if ckpt_path.is_file():
        logger.info("Finetuned iTransformer exists: %s", ckpt_path)
        return ckpt_path

    logger.info("=== Stage 3: finetune iTransformer on %s ===", DATASET)

    train_ds, val_ds, _, _ = load_dataset(
        DATASET, variate_indices=None,
        lookback=LOOKBACK_LENGTH, horizon=FORECAST_LENGTH,
        stride=24 if not smoke_test else LOOKBACK_LENGTH,
        lookback_overlap=LOOKBACK_OVERLAP,
    )
    if smoke_test:
        train_ds = Subset(train_ds, list(range(min(8, len(train_ds)))))
        val_ds = Subset(val_ds, list(range(min(4, len(val_ds)))))

    bs = 32 if not smoke_test else 4
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0)

    # init from pretrained weights
    model = create_itransformer(num_vars=N_VARIATES, pred_len=FORECAST_LENGTH).to(device)
    ckpt = torch.load(itrans_pretrained, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=100 if not smoke_test else 3, eta_min=1e-6)
    criterion = nn.MSELoss()
    epochs = 3 if smoke_test else 100
    patience = 1 if smoke_test else 15
    early = EarlyStopping(patience=patience)
    best_val = float("inf")

    for ep in range(epochs):
        tr = train_itransformer_epoch(model, train_dl, opt, criterion, device, sched)
        va = validate_itransformer(model, val_dl, criterion, device)
        logger.info("[iTrans finetune %s] ep %d  train=%.4f  val=%.4f", DATASET, ep + 1, tr, va)
        _wb_log({"s3_itrans_finetune/train_loss": tr, "s3_itrans_finetune/val_loss": va})
        if va < best_val:
            best_val = va
            save_itransformer_checkpoint(
                model, opt, ep, tr, va,
                {"num_variates": N_VARIATES, "lr": 5e-5, "dataset": DATASET,
                 "init_from": str(itrans_pretrained)},
                ckpt_path,
            )
        if early(va):
            break

    _wb_summary({"s3_itrans_finetune/best_val": best_val})
    logger.info("Finetuned iTransformer → %s", ckpt_path)
    return ckpt_path


# ---------------------------------------------------------------------------
# stage 4: finetune diffusion on ETTh2 with finetuned iTrans + eval
# ---------------------------------------------------------------------------

def stage4_finetune_eval(
    cfg: LatentDiffusionConfig,
    vae: TimeSeriesVAE,
    itrans_finetuned: Path,
    diffusion_pretrained: Path,
    device: torch.device,
    smoke_test: bool,
) -> Dict:
    logger.info("=== Stage 4: finetune diffusion on %s (guidance = finetuned iTrans) ===", DATASET)

    train_ds, val_ds, test_ds, _ = load_dataset(
        DATASET, variate_indices=None,
        lookback=LOOKBACK_LENGTH, horizon=FORECAST_LENGTH,
        stride=24 if not smoke_test else LOOKBACK_LENGTH,
        lookback_overlap=LOOKBACK_OVERLAP,
    )
    if smoke_test:
        train_ds = Subset(train_ds, list(range(min(8, len(train_ds)))))
        val_ds = Subset(val_ds, list(range(min(4, len(val_ds)))))
        test_ds = Subset(test_ds, list(range(min(4, len(test_ds)))))

    train_dl = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=0)

    # finetuned guidance — this is the one used for everything from here on
    guidance = _build_ci_guidance(itrans_finetuned, device)
    model = LatentDiffusionTSF(cfg, vae, guidance_model=guidance).to(device)

    # load pretrained diffusion weights (guidance keys will mismatch — that's fine,
    # the pretrained ckpt had the synth iTrans; we're swapping in finetuned)
    ckpt = torch.load(diffusion_pretrained, map_location=device, weights_only=False)
    result = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if result.missing_keys:
        logger.info("Missing keys (expected — guidance model swap): %s",
                     [k for k in result.missing_keys if "guidance" in k][:5])

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    epochs = 2 if smoke_test else min(30, FINETUNE_EPOCHS)
    patience = 1 if smoke_test else min(10, FINETUNE_PATIENCE)
    early = EarlyStopping(patience=patience)

    for ep in range(epochs):
        tr = _ci_train_epoch(model, train_dl, opt, device, N_VARIATES)
        va = _ci_eval_loss(model, val_dl, device, N_VARIATES)
        logger.info("[Diffusion finetune %s] ep %d  train=%.4f  val=%.4f", DATASET, ep + 1, tr, va)
        _wb_log({"s4_diff_finetune/train_loss": tr, "s4_diff_finetune/val_loss": va})
        if early(va):
            break

    finetuned_path = CKPT_DIR / f"diffusion_finetuned_{DATASET}_H{cfg.image_height}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": asdict(cfg),
        "vae_scale_factor": vae.scale_factor.detach().cpu(),
        "guidance_source": str(itrans_finetuned),
        "dataset": DATASET,
    }, finetuned_path)
    logger.info("Finetuned diffusion → %s", finetuned_path)

    # ---- eval: CI diffusion (finetuned) ----
    model.eval()
    preds_ci, trues_ci = [], []
    ddim_steps = 10 if smoke_test else 50
    with torch.no_grad():
        for past, future in test_dl:
            B, V, L_past = past.shape
            past_flat = past.reshape(B * V, L_past).to(device)
            gen = model.generate(past_flat, num_ddim_steps=ddim_steps)
            pred = gen["prediction"].squeeze(1).reshape(B, V, -1)
            true = future[:, :, LOOKBACK_OVERLAP:]
            preds_ci.append(pred.cpu())
            trues_ci.append(true.cpu())
    y_hat_ci = torch.cat(preds_ci, 0)
    y_true = torch.cat(trues_ci, 0)
    ci_mse = F.mse_loss(y_hat_ci, y_true)
    ci_mae = F.l1_loss(y_hat_ci, y_true)
    logger.info("[%s] CI diffusion (finetuned)  MSE=%.6f  MAE=%.6f", DATASET, ci_mse.item(), ci_mae.item())

    per_var = {}
    for vi in range(N_VARIATES):
        m = compute_metrics(y_hat_ci[:, vi, :], y_true[:, vi, :])
        per_var[f"var{vi}_mse"] = _scalar(m["mse"])
        per_var[f"var{vi}_mae"] = _scalar(m["mae"])

    # ---- eval: finetuned iTransformer baseline (same checkpoint as guidance) ----
    itrans_model = create_itransformer(num_vars=N_VARIATES, pred_len=FORECAST_LENGTH).to(device)
    itc = torch.load(itrans_finetuned, map_location=device, weights_only=False)
    itrans_model.load_state_dict(itc["model_state_dict"])
    itrans_model.eval()
    preds_it, trues_it = [], []
    with torch.no_grad():
        for past, future in test_dl:
            x_enc = past.permute(0, 2, 1).to(device)
            y_pred = itrans_model(x_enc, None, None, None)
            if isinstance(y_pred, tuple):
                y_pred = y_pred[0]
            y_pred = y_pred.permute(0, 2, 1)
            true = future[:, :, LOOKBACK_OVERLAP:].to(device)
            preds_it.append(y_pred.cpu())
            trues_it.append(true.cpu())
    y_hat_it = torch.cat(preds_it, 0)
    y_true_it = torch.cat(trues_it, 0)
    it_mse = F.mse_loss(y_hat_it, y_true_it)
    it_mae = F.l1_loss(y_hat_it, y_true_it)
    logger.info("[%s] iTransformer (finetuned)  MSE=%.6f  MAE=%.6f", DATASET, it_mse.item(), it_mae.item())

    _wb_log({
        "eval/ci_diffusion_mse": _scalar(ci_mse),
        "eval/ci_diffusion_mae": _scalar(ci_mae),
        "eval/itransformer_finetuned_mse": _scalar(it_mse),
        "eval/itransformer_finetuned_mae": _scalar(it_mae),
    })
    _wb_summary({
        "eval/ci_diffusion_mse": _scalar(ci_mse),
        "eval/ci_diffusion_mae": _scalar(ci_mae),
        "eval/itransformer_finetuned_mse": _scalar(it_mse),
        "eval/itransformer_finetuned_mae": _scalar(it_mae),
        **{f"eval/{k}": v for k, v in per_var.items()},
    })

    return {
        "dataset": DATASET,
        "finetuned_diffusion_ckpt": str(finetuned_path),
        "finetuned_itrans_ckpt": str(itrans_finetuned),
        "ci_diffusion_mse": _scalar(ci_mse),
        "ci_diffusion_mae": _scalar(ci_mae),
        "itransformer_finetuned_mse": _scalar(it_mse),
        "itransformer_finetuned_mae": _scalar(it_mae),
        "num_variates": N_VARIATES,
        "image_height": cfg.image_height,
        "forecast_horizon": FORECAST_LENGTH,
        "test_stride": 24,
        "per_variate": per_var,
        "note": "Both diffusion and iTransformer baseline are finetuned on ETTh2. Fair comparison.",
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=f"CI latent diffusion full pipeline on {DATASET}")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--image-height", type=int, choices=[96, 128], default=128)
    p.add_argument("--stage", type=str, default="all",
                   choices=["0", "1", "2", "3", "4", "all"])
    p.add_argument("--cache-dir", type=str, default=None)
    args = p.parse_args()

    device = get_device()
    cfg = build_config(args.image_height)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    _wb_init(cfg, args)

    n_syn = 500 if args.smoke_test else SYNTHETIC_SAMPLES_FULL
    cache = args.cache_dir or str(CKPT_DIR / "synth_cache")
    stages = {"0", "1", "2", "3", "4"} if args.stage == "all" else {args.stage}

    logger.info(
        "CI latent %s: stages=%s smoke=%s H=%s device=%s",
        DATASET, ",".join(sorted(stages)), args.smoke_test, args.image_height, device,
    )

    itrans_pretrained = CKPT_DIR / "itransformer_pretrained_synth.pt"
    itrans_finetuned = CKPT_DIR / f"itransformer_finetuned_{DATASET}.pt"
    diff_pretrained = CKPT_DIR / f"diffusion_pretrained_H{args.image_height}.pt"
    results_path = RESULTS_DIR / f"ci_latent_{DATASET}_H{args.image_height}.json"

    vae = None

    # stage 0: VAE
    if stages & {"0", "2", "4"}:
        vae = stage0_vae(cfg, device, n_syn, cache, args.smoke_test)

    # stage 1: pretrain iTransformer on synthetic
    if "1" in stages:
        stage1_pretrain_itransformer(device, n_syn, cache, args.smoke_test)

    # stage 2: pretrain diffusion with pretrained iTrans guidance
    if "2" in stages:
        if vae is None:
            raise SystemExit("Need VAE (run stage 0)")
        if not itrans_pretrained.is_file():
            raise SystemExit(f"Missing {itrans_pretrained} (run stage 1)")
        stage2_pretrain_diffusion(cfg, vae, itrans_pretrained, device, n_syn, cache, args.smoke_test)

    # stage 3: finetune iTransformer on ETTh2
    if "3" in stages:
        if not itrans_pretrained.is_file():
            raise SystemExit(f"Missing {itrans_pretrained} (run stage 1)")
        stage3_finetune_itransformer(itrans_pretrained, device, args.smoke_test)

    # stage 4: finetune diffusion with FINETUNED iTrans + eval
    if "4" in stages:
        if vae is None:
            raise SystemExit("Need VAE (run stage 0)")
        if not itrans_finetuned.is_file():
            raise SystemExit(f"Missing {itrans_finetuned} (run stage 3)")
        if not diff_pretrained.is_file():
            raise SystemExit(f"Missing {diff_pretrained} (run stage 2)")
        metrics = stage4_finetune_eval(
            cfg, vae, itrans_finetuned, diff_pretrained, device, args.smoke_test,
        )
        with open(results_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Results → %s", results_path)
        logger.info("Summary:\n%s", json.dumps(
            {k: v for k, v in metrics.items() if k != "per_variate"}, indent=2,
        ))

    _wb_finish()


if __name__ == "__main__":
    main()
