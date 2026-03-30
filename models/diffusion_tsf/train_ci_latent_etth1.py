"""
CI (channel-independent) latent diffusion ablation on ETTh1 (7-var).

Two variants:
  guided   — iTransformer ghost conditioning per-variate
  unguided — pure latent diffusion, no guidance

Both evaluate against iTransformer-only baseline to see
whether guidance is still valuable in latent space.

Run from repo root:
  python -m models.diffusion_tsf.train_ci_latent_etth1 --smoke-test --stage all
  python -m models.diffusion_tsf.train_ci_latent_etth1 --no-guidance --smoke-test --stage all
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

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.diffusion_tsf.config import LatentDiffusionConfig
from models.diffusion_tsf.dataset import get_synthetic_dataloader
from models.diffusion_tsf.guidance import BaseGuidance
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

# shared with 1-var experiment
CKPT_LATENT = _SCRIPT_DIR / "checkpoints_latent"
# CI-specific
CKPT_CI = _SCRIPT_DIR / "checkpoints_ci_latent"
RESULTS_DIR = _SCRIPT_DIR / "results"

N_VARIATES = 7


# ---------------------------------------------------------------------------
# CI guidance wrapper — unflattens (B*V, L) for multivariate iTransformer
# ---------------------------------------------------------------------------

class CIiTransformerGuidance(BaseGuidance):
    """Wraps a multivariate iTransformer for channel-independent inference.

    The latent diffusion model processes each variate independently, so past
    context arrives as (B*V, L).  This wrapper unflattens it back to
    (B, V, L), runs the full multivariate iTransformer, then flattens the
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
        # (B*V, L) → (B, V, L) → (B, L, V) for iTransformer
        x_enc = past.reshape(B, V, L).permute(0, 2, 1)
        y_pred = self.model(x_enc, None, None, None)
        if isinstance(y_pred, tuple):
            y_pred = y_pred[0]
        # (B, H, V) → (B, V, H) → (B*V, H)
        return y_pred.permute(0, 2, 1).reshape(BV, -1)[:, :forecast_length]


# ---------------------------------------------------------------------------
# pixel encoder (same as train_latent_experiment)
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
    """Normalize per-sample → concat past+future → encode to 2D."""
    mean = past.mean(dim=-1, keepdim=True)
    std = past.std(dim=-1, keepdim=True) + 1e-8
    pn = (past - mean) / std
    fn = (future - mean) / std
    if pn.dim() == 2:
        pn, fn = pn.unsqueeze(1), fn.unsqueeze(1)
    return torch.cat([pixel_enc(pn), pixel_enc(fn)], dim=-1)


# ---------------------------------------------------------------------------
# config builders
# ---------------------------------------------------------------------------

def build_ci_config(image_height: int, use_guidance: bool) -> LatentDiffusionConfig:
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
        use_guidance_channel=use_guidance,
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
# stage 0: VAE (reuse from 1-var experiment if available)
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

    # train from scratch
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
            logger.info("VAE saved → %s  sf=%.6f", vae_path, vae.scale_factor.item())
        if early(va):
            break

    ckpt = torch.load(vae_path, map_location=device, weights_only=False)
    vae.load_state_dict(ckpt["vae_state_dict"])
    vae.scale_factor.copy_(ckpt["scale_factor"].to(device).view_as(vae.scale_factor))
    return vae


# ---------------------------------------------------------------------------
# stage 1: iTransformer on ETTh1 (7-var) — baseline + guidance source
# ---------------------------------------------------------------------------

def stage1_itransformer_etth1(
    device: torch.device,
    smoke_test: bool,
) -> Path:
    CKPT_CI.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_CI / "itransformer_etth1_7var.pt"
    if ckpt_path.is_file():
        logger.info("iTransformer checkpoint exists: %s — skipping training", ckpt_path)
        return ckpt_path

    train_ds, val_ds, _, _ = load_dataset(
        "ETTh1", variate_indices=None,
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

    model = create_itransformer(num_vars=N_VARIATES, pred_len=FORECAST_LENGTH).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    epochs = 3 if smoke_test else 100
    patience = 1 if smoke_test else 15
    early = EarlyStopping(patience=patience)
    best_val = float("inf")

    for ep in range(epochs):
        tr = train_itransformer_epoch(model, train_dl, opt, criterion, device)
        va = validate_itransformer(model, val_dl, criterion, device)
        logger.info("[iTransformer ETTh1 7var] ep %d  train=%.4f  val=%.4f", ep + 1, tr, va)
        if va < best_val:
            best_val = va
            save_itransformer_checkpoint(
                model, opt, ep, tr, va,
                {"num_variates": N_VARIATES, "lr": 1e-4, "bs": bs},
                ckpt_path,
            )
        if early(va):
            break

    logger.info("iTransformer ETTh1 saved → %s", ckpt_path)
    return ckpt_path


# ---------------------------------------------------------------------------
# stage 2: pretrain latent diffusion on synthetic univariate data
# ---------------------------------------------------------------------------

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


def stage2_pretrain(
    cfg: LatentDiffusionConfig,
    vae: TimeSeriesVAE,
    device: torch.device,
    n_samples: int,
    cache_dir: Optional[str],
    use_guidance: bool,
    smoke_test: bool,
) -> Path:
    tag = "guided" if use_guidance else "unguided"
    out_path = CKPT_CI / f"{tag}_pretrained_H{cfg.image_height}.pt"
    CKPT_CI.mkdir(parents=True, exist_ok=True)

    if out_path.is_file():
        logger.info("Pretrained %s checkpoint exists: %s", tag, out_path)
        return out_path

    # LinearRegression during pretraining — real iTransformer swapped in at finetune
    from models.diffusion_tsf.guidance import LinearRegressionGuidance
    guidance = LinearRegressionGuidance() if use_guidance else None

    model = LatentDiffusionTSF(cfg, vae, guidance_model=guidance).to(device)

    loader = get_synthetic_dataloader(
        num_samples=n_samples, lookback_length=LOOKBACK_LENGTH,
        forecast_length=FORECAST_LENGTH, batch_size=8, num_workers=0,
        num_variables=1, pool_size=n_samples, cache_dir=cache_dir,
        lookback_overlap=LOOKBACK_OVERLAP, skip_cross_var_aug=True,
    )
    ds = loader.dataset
    n_val = min(len(ds) // 10, 1000)
    train_dl = DataLoader(Subset(ds, list(range(len(ds) - n_val))), batch_size=8, shuffle=True)
    val_dl = DataLoader(Subset(ds, list(range(len(ds) - n_val, len(ds)))), batch_size=8)

    epochs = 2 if smoke_test else PRETRAIN_EPOCHS
    patience = 1 if smoke_test else PRETRAIN_PATIENCE
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.learning_rate)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=cfg.learning_rate * 0.01)
    early = EarlyStopping(patience=patience)
    best_val = float("inf")

    for ep in range(epochs):
        tr = _train_ld_epoch(model, train_dl, opt, device)
        va = _eval_ld(model, val_dl, device)
        sched.step()
        logger.info("[Pretrain %s] ep %d  train=%.4f  val=%.4f", tag, ep + 1, tr, va)
        if va < best_val:
            best_val = va
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": asdict(cfg),
                "vae_scale_factor": vae.scale_factor.detach().cpu(),
            }, out_path)
        if early(va):
            break

    logger.info("Pretrained %s → %s", tag, out_path)
    return out_path


# ---------------------------------------------------------------------------
# stage 3: fine-tune on ETTh1 (CI) + eval vs iTransformer baseline
# ---------------------------------------------------------------------------

def _ci_train_epoch(model, loader, optimizer, device, V):
    """Flatten (B, V, L) → (B*V, L), train as univariate."""
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


def _scalar(x):
    return float(x.detach().cpu().item()) if torch.is_tensor(x) else float(x)


def stage3_finetune_eval(
    cfg: LatentDiffusionConfig,
    vae: TimeSeriesVAE,
    itrans_path: Optional[Path],
    pretrained_path: Path,
    device: torch.device,
    smoke_test: bool,
    use_guidance: bool,
) -> Dict:
    tag = "guided" if use_guidance else "unguided"
    logger.info("=== Stage 3 [%s]: fine-tune + eval on ETTh1 ===", tag)

    # load ETTh1 (all 7 variates)
    train_ds, val_ds, test_ds, _ = load_dataset(
        "ETTh1", variate_indices=None,
        lookback=LOOKBACK_LENGTH, horizon=FORECAST_LENGTH,
        stride=24 if not smoke_test else LOOKBACK_LENGTH,
        lookback_overlap=LOOKBACK_OVERLAP,
    )
    if smoke_test:
        train_ds = Subset(train_ds, list(range(min(8, len(train_ds)))))
        val_ds = Subset(val_ds, list(range(min(4, len(val_ds)))))
        test_ds = Subset(test_ds, list(range(min(4, len(test_ds)))))

    # batch_size=4 → effective 4*7=28 univariate samples per step
    train_dl = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=0)

    # build model
    guidance = None
    if use_guidance:
        if itrans_path is None or not itrans_path.is_file():
            raise SystemExit(f"Guided mode needs iTransformer at {itrans_path}")
        guidance = _build_ci_guidance(itrans_path, device)

    model = LatentDiffusionTSF(cfg, vae, guidance_model=guidance).to(device)

    # load pretrained weights (strict=False because guidance_model params may differ)
    ckpt = torch.load(pretrained_path, map_location=device, weights_only=False)
    result = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if result.missing_keys:
        logger.info("State-dict missing keys (expected for CI guidance swap): %s",
                     [k for k in result.missing_keys if "guidance" in k][:5])

    # fine-tune
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    epochs = 2 if smoke_test else min(30, FINETUNE_EPOCHS)
    patience = 1 if smoke_test else min(10, FINETUNE_PATIENCE)
    early = EarlyStopping(patience=patience)

    for ep in range(epochs):
        tr = _ci_train_epoch(model, train_dl, opt, device, N_VARIATES)
        va = _ci_eval_loss(model, val_dl, device, N_VARIATES)
        logger.info("[Finetune %s] ep %d  train=%.4f  val=%.4f", tag, ep + 1, tr, va)
        if early(va):
            break

    finetuned_path = CKPT_CI / f"{tag}_finetuned_H{cfg.image_height}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(cfg),
            "vae_scale_factor": vae.scale_factor.detach().cpu(),
            "tag": tag,
            "use_guidance": use_guidance,
        },
        finetuned_path,
    )
    logger.info("Saved finetuned CI latent checkpoint → %s", finetuned_path)

    # ---- evaluate CI diffusion ----
    model.eval()
    preds_ci, trues_ci = [], []
    ddim_steps = 10 if smoke_test else 50
    with torch.no_grad():
        for past, future in test_dl:
            B, V, L_past = past.shape
            past_flat = past.reshape(B * V, L_past).to(device)
            gen = model.generate(past_flat, num_ddim_steps=ddim_steps)
            # (B*V, 1, H) → (B*V, H) → (B, V, H)
            pred = gen["prediction"].squeeze(1).reshape(B, V, -1)
            true = future[:, :, LOOKBACK_OVERLAP:]  # (B, V, H)
            preds_ci.append(pred.cpu())
            trues_ci.append(true.cpu())
    y_hat_ci = torch.cat(preds_ci, 0)  # (N, V, H)
    y_true = torch.cat(trues_ci, 0)
    ci_mse = F.mse_loss(y_hat_ci, y_true)
    ci_mae = F.l1_loss(y_hat_ci, y_true)
    logger.info("[%s] CI diffusion  MSE=%.6f  MAE=%.6f", tag, ci_mse.item(), ci_mae.item())

    # per-variate breakdown
    per_var = {}
    for vi in range(V):
        m = compute_metrics(y_hat_ci[:, vi, :], y_true[:, vi, :])
        per_var[f"var{vi}_mse"] = _scalar(m["mse"])
        per_var[f"var{vi}_mae"] = _scalar(m["mae"])

    # ---- iTransformer-only baseline ----
    it_mse, it_mae = None, None
    if itrans_path is not None and itrans_path.is_file():
        itrans_model = create_itransformer(num_vars=N_VARIATES, pred_len=FORECAST_LENGTH).to(device)
        itc = torch.load(itrans_path, map_location=device, weights_only=False)
        itrans_model.load_state_dict(itc["model_state_dict"])
        itrans_model.eval()
        preds_it, trues_it = [], []
        with torch.no_grad():
            for past, future in test_dl:
                x_enc = past.permute(0, 2, 1).to(device)  # (B, L, V)
                y_pred = itrans_model(x_enc, None, None, None)
                if isinstance(y_pred, tuple):
                    y_pred = y_pred[0]
                # (B, H, V) → (B, V, H)
                y_pred = y_pred.permute(0, 2, 1)
                true = future[:, :, LOOKBACK_OVERLAP:].to(device)
                preds_it.append(y_pred.cpu())
                trues_it.append(true.cpu())
        y_hat_it = torch.cat(preds_it, 0)
        y_true_it = torch.cat(trues_it, 0)
        it_mse = F.mse_loss(y_hat_it, y_true_it)
        it_mae = F.l1_loss(y_hat_it, y_true_it)
        logger.info("[baseline] iTransformer  MSE=%.6f  MAE=%.6f", it_mse.item(), it_mae.item())
    else:
        logger.info("No iTransformer checkpoint — skipping baseline eval")

    return {
        "mode": tag,
        "finetuned_checkpoint": str(finetuned_path),
        "ci_diffusion_mse": _scalar(ci_mse),
        "ci_diffusion_mae": _scalar(ci_mae),
        "itransformer_baseline_mse": _scalar(it_mse) if it_mse is not None else None,
        "itransformer_baseline_mae": _scalar(it_mae) if it_mae is not None else None,
        "num_variates": N_VARIATES,
        "image_height": cfg.image_height,
        "forecast_horizon": FORECAST_LENGTH,
        "per_variate": per_var,
    }


# ---------------------------------------------------------------------------
# smoke test validations
# ---------------------------------------------------------------------------

def run_ci_smoke_checks(
    cfg: LatentDiffusionConfig,
    vae: TimeSeriesVAE,
    device: torch.device,
    use_guidance: bool,
    itrans_path: Optional[Path] = None,
):
    logger.info("Running CI smoke checks (guidance=%s)...", use_guidance)

    guidance = None
    if use_guidance and itrans_path is not None:
        guidance = _build_ci_guidance(itrans_path, device)

    model = LatentDiffusionTSF(cfg, vae, guidance_model=guidance).to(device)

    # simulate a CI batch: (B=2, V=7, L)
    B, V = 2, N_VARIATES
    past = torch.randn(B, V, LOOKBACK_LENGTH, device=device)
    future = torch.randn(B, V, FORECAST_LENGTH + LOOKBACK_OVERLAP, device=device)

    past_flat = past.reshape(B * V, LOOKBACK_LENGTH)
    future_flat = future.reshape(B * V, FORECAST_LENGTH + LOOKBACK_OVERLAP)

    # forward pass
    model.train()
    out = model(past_flat, future_flat)
    out["loss"].backward()

    # check VAE is frozen
    for n, p in model.vae.named_parameters():
        g = p.grad
        assert g is None or g.abs().max().item() == 0.0, f"VAE grad leak: {n}"

    # check U-Net gets gradients
    unet_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.noise_predictor.parameters()
    )
    assert unet_grad, "U-Net must receive gradients"

    # generate
    model.eval()
    with torch.no_grad():
        gen = model.generate(past_flat[:V], num_ddim_steps=3)
    pred = gen["prediction"]  # (V, 1, H)
    assert pred.shape == (V, 1, FORECAST_LENGTH), f"bad pred shape: {pred.shape}"

    # unflatten check
    pred_unflat = pred.squeeze(1).reshape(1, V, FORECAST_LENGTH)
    assert pred_unflat.shape == (1, V, FORECAST_LENGTH)

    logger.info("CI smoke checks passed (guidance=%s)", use_guidance)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="CI latent diffusion ablation on ETTh1")
    p.add_argument("--no-guidance", action="store_true", help="disable iTransformer guidance (unguided variant)")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--image-height", type=int, choices=[96, 128], default=128)
    p.add_argument("--stage", type=str, default="all", choices=["0", "1", "2", "3", "all"])
    p.add_argument("--cache-dir", type=str, default=None)
    args = p.parse_args()

    device = get_device()
    use_guidance = not args.no_guidance
    tag = "guided" if use_guidance else "unguided"
    cfg = build_ci_config(args.image_height, use_guidance)
    CKPT_CI.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    n_syn = 500 if args.smoke_test else SYNTHETIC_SAMPLES_FULL
    cache = args.cache_dir or str(CKPT_CI / "synth_cache")
    stages = {"0", "1", "2", "3"} if args.stage == "all" else {args.stage}

    logger.info(
        "CI latent ETTh1: mode=%s stages=%s smoke=%s H=%s device=%s",
        tag, ",".join(sorted(stages)), args.smoke_test, args.image_height, device,
    )

    itrans_path = CKPT_CI / "itransformer_etth1_7var.pt"
    pretrained_path = CKPT_CI / f"{tag}_pretrained_H{args.image_height}.pt"
    results_path = RESULTS_DIR / f"ci_latent_etth1_{tag}_H{args.image_height}.json"

    vae = None

    # -- stage 0: VAE --
    if "0" in stages or "2" in stages or "3" in stages:
        vae = stage0_vae(cfg, device, n_syn, cache, args.smoke_test)

    # -- stage 1: iTransformer on ETTh1 (skip for unguided unless needed for baseline) --
    if "1" in stages:
        if use_guidance:
            stage1_itransformer_etth1(device, args.smoke_test)
        else:
            # still train for baseline comparison, but could skip if user only wants diffusion
            logger.info("Unguided mode: training iTransformer anyway (baseline comparison)")
            stage1_itransformer_etth1(device, args.smoke_test)

    # -- stage 2: pretrain latent diffusion --
    if "2" in stages and vae is not None:
        stage2_pretrain(cfg, vae, device, n_syn, cache, use_guidance, args.smoke_test)

    # -- stage 3: fine-tune + eval --
    if "3" in stages and vae is not None:
        if not pretrained_path.is_file():
            raise SystemExit(f"Missing pretrained checkpoint: {pretrained_path}")
        # ensure iTransformer is trained for baseline eval
        if not itrans_path.is_file():
            logger.info("Training iTransformer for baseline eval...")
            stage1_itransformer_etth1(device, args.smoke_test)
        metrics = stage3_finetune_eval(
            cfg, vae, itrans_path, pretrained_path,
            device, args.smoke_test, use_guidance,
        )
        with open(results_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Results → %s", results_path)
        logger.info("Summary: %s", json.dumps({
            k: v for k, v in metrics.items() if k != "per_variate"
        }, indent=2))

    # -- smoke checks --
    if args.smoke_test and vae is not None:
        run_ci_smoke_checks(
            cfg, vae, device, use_guidance,
            itrans_path=itrans_path if itrans_path.is_file() else None,
        )


if __name__ == "__main__":
    main()
