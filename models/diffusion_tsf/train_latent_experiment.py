"""
Latent diffusion experiment (1-variate ETTh1): VAE → iTransformer → latent diffusion → fine-tune.

Run from repo root:
  python -m models.diffusion_tsf.train_latent_experiment --smoke-test --stage all
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
from torch.utils.data import DataLoader, Subset

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.diffusion_tsf.config import LatentDiffusionConfig
from models.diffusion_tsf.dataset import get_synthetic_dataloader
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
    amp_context,
    create_itransformer,
    get_device,
    load_dataset,
    pretrain_itransformer_dim1,
    EarlyStopping,
    create_pixel_diffusion_baseline,
)
from models.diffusion_tsf.preprocessing import TimeSeriesTo2D, VerticalGaussianBlur
from models.diffusion_tsf.vae import TimeSeriesVAE, estimate_vae_scale_factor
from models.diffusion_tsf.guidance import iTransformerGuidance
from models.diffusion_tsf.metrics import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CKPT_LATENT = _SCRIPT_DIR / "checkpoints_latent"
RESULTS_DIR = _SCRIPT_DIR / "results"
DEFAULT_7VAR_CKPT = _SCRIPT_DIR / "checkpoints_7var"


def build_latent_config(image_height: int) -> LatentDiffusionConfig:
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


class PixelEncoder(nn.Module):
    """Same 2D scaling as DiffusionTSF.encode_to_2d (cdf path)."""

    def __init__(self, cfg: LatentDiffusionConfig):
        super().__init__()
        self.to_2d = TimeSeriesTo2D(
            height=cfg.image_height,
            max_scale=cfg.max_scale,
            representation_mode=cfg.representation_mode,
        )
        self.blur = VerticalGaussianBlur(
            kernel_size=cfg.blur_kernel_size,
            sigma=cfg.blur_sigma,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        im = self.to_2d(x)
        b = self.blur(im)
        if self.to_2d.representation_mode == "pdf":
            scaled = b * 30.0
            return scaled * 2.0 - 1.0
        return b.clamp(min=0.0, max=1.0) * 2.0 - 1.0


def unified_2d_from_batch(
    pixel_enc: PixelEncoder,
    past: torch.Tensor,
    future: torch.Tensor,
) -> torch.Tensor:
    """(B,1,L) + (B,1,F) normalized per-sample → (B,1,H,W) unified canvas."""
    mean = past.mean(dim=-1, keepdim=True)
    std = past.std(dim=-1, keepdim=True) + 1e-8
    pn = (past - mean) / std
    fn = (future - mean) / std
    if pn.dim() == 2:
        pn = pn.unsqueeze(1)
    if fn.dim() == 2:
        fn = fn.unsqueeze(1)
    p2 = pixel_enc(pn)
    f2 = pixel_enc(fn)
    return torch.cat([p2, f2], dim=-1)


def stage0_train_vae(
    cfg: LatentDiffusionConfig,
    device: torch.device,
    n_samples: int,
    epochs: int,
    patience: int,
    cache_dir: Optional[str],
    out_path: Path,
    skip_train: bool,
) -> TimeSeriesVAE:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if skip_train and out_path.is_file():
        ckpt = torch.load(out_path, map_location=device, weights_only=False)
        vae = TimeSeriesVAE(latent_channels=cfg.latent_channels).to(device)
        vae.load_state_dict(ckpt["vae_state_dict"])
        if "scale_factor" in ckpt:
            vae.scale_factor.copy_(ckpt["scale_factor"].to(device).view_as(vae.scale_factor))
        logger.info("Loaded VAE from %s", out_path)
        return vae

    loader = get_synthetic_dataloader(
        num_samples=n_samples,
        lookback_length=LOOKBACK_LENGTH,
        forecast_length=FORECAST_LENGTH,
        batch_size=16,
        num_workers=0,
        num_variables=1,
        pool_size=n_samples,
        cache_dir=cache_dir,
        lookback_overlap=LOOKBACK_OVERLAP,
        skip_cross_var_aug=True,
    )
    dataset = loader.dataset
    n_val = min(len(dataset) // 10, 2000)
    train_idx = list(range(len(dataset) - n_val))
    val_idx = list(range(len(dataset) - n_val, len(dataset)))
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=16, shuffle=False, num_workers=0)

    pixel_enc = PixelEncoder(cfg).to(device)
    pixel_enc.eval()
    vae = TimeSeriesVAE(latent_channels=cfg.latent_channels).to(device)
    opt = torch.optim.AdamW(vae.parameters(), lr=cfg.vae_lr)
    early = EarlyStopping(patience=patience)
    best_val = float("inf")

    for epoch in range(epochs):
        vae.train()
        tr = 0.0
        nb = 0
        for past, future in train_loader:
            past = past.to(device)
            future = future.to(device)
            with torch.no_grad():
                img = unified_2d_from_batch(pixel_enc, past, future)
            opt.zero_grad()
            out = vae(img, kl_weight=cfg.kl_weight)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            opt.step()
            tr += out["loss"].item()
            nb += 1
        vae.eval()
        va = 0.0
        nv = 0
        with torch.no_grad():
            for past, future in val_loader:
                past = past.to(device)
                future = future.to(device)
                img = unified_2d_from_batch(pixel_enc, past, future)
                out = vae(img, kl_weight=cfg.kl_weight)
                va += out["loss"].item()
                nv += 1
        tr /= max(nb, 1)
        va /= max(nv, 1)
        logger.info("VAE epoch %s train=%.4f val=%.4f", epoch + 1, tr, va)
        if va < best_val:
            best_val = va
            imgs = []
            with torch.no_grad():
                for past, future in val_loader:
                    past = past.to(device)
                    future = future.to(device)
                    imgs.append(unified_2d_from_batch(pixel_enc, past, future))
                    if len(imgs) >= 4:
                        break
            estimate_vae_scale_factor(vae, torch.cat(imgs, dim=0), max_batches=4, batch_size=16)
            torch.save(
                {
                    "vae_state_dict": vae.state_dict(),
                    "scale_factor": vae.scale_factor.detach().cpu(),
                    "config": {k: v for k, v in asdict(cfg).items() if isinstance(v, (int, float, str, bool))},
                    "latent_channels": cfg.latent_channels,
                    "image_height": cfg.image_height,
                },
                out_path,
            )
            logger.info("Saved VAE → %s scale_factor=%.6f", out_path, vae.scale_factor.item())
        if early(va):
            break

    ckpt = torch.load(out_path, map_location=device, weights_only=False)
    vae.load_state_dict(ckpt["vae_state_dict"])
    vae.scale_factor.copy_(ckpt["scale_factor"].to(device).view_as(vae.scale_factor))
    return vae


def stage1_itransformer(
    checkpoint_dir: Path,
    n_samples: int,
    epochs: int,
    patience: int,
    cache_dir: Optional[str],
    out_name: str = "itransformer_dim1.pt",
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = pretrain_itransformer_dim1(
        checkpoint_dir,
        n_samples=n_samples,
        epochs=epochs,
        patience=patience,
        cache_dir=cache_dir,
        smoke_test=(n_samples < 1000),
    )
    dest = checkpoint_dir / out_name
    if path.resolve() != dest.resolve():
        import shutil

        shutil.copy(path, dest)
    logger.info("iTransformer saved → %s", dest)
    return dest


def build_itrans_guidance(itrans_ckpt: Path, device: torch.device) -> iTransformerGuidance:
    model = create_itransformer(num_vars=1).to(device)
    ckpt = torch.load(itrans_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return iTransformerGuidance(
        model=model,
        use_norm=True,
        seq_len=LOOKBACK_LENGTH,
        pred_len=FORECAST_LENGTH,
    )


def train_latent_diffusion_epoch(
    model: LatentDiffusionTSF,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    n = 0
    for past, future in loader:
        past = past.to(device)
        future = future.to(device)
        optimizer.zero_grad()
        with amp_context():
            loss = model.get_loss(past, future)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
        optimizer.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def eval_latent_diffusion(model: LatentDiffusionTSF, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    n = 0
    for past, future in loader:
        past = past.to(device)
        future = future.to(device)
        with amp_context():
            loss = model.get_loss(past, future)
        total += loss.item()
        n += 1
    return total / max(n, 1)


def stage2_pretrain_latent(
    cfg: LatentDiffusionConfig,
    vae: TimeSeriesVAE,
    itrans_path: Path,
    device: torch.device,
    n_samples: int,
    epochs: int,
    patience: int,
    cache_dir: Optional[str],
    out_path: Path,
) -> None:
    guidance = build_itrans_guidance(itrans_path, device)
    model = LatentDiffusionTSF(cfg, vae, guidance_model=guidance).to(device)

    loader = get_synthetic_dataloader(
        num_samples=n_samples,
        lookback_length=LOOKBACK_LENGTH,
        forecast_length=FORECAST_LENGTH,
        batch_size=8,
        num_workers=0,
        num_variables=1,
        pool_size=n_samples,
        cache_dir=cache_dir,
        lookback_overlap=LOOKBACK_OVERLAP,
        skip_cross_var_aug=True,
    )
    dataset = loader.dataset
    n_val = min(len(dataset) // 10, 1000)
    train_loader = DataLoader(Subset(dataset, list(range(len(dataset) - n_val))), batch_size=8, shuffle=True)
    val_loader = DataLoader(Subset(dataset, list(range(len(dataset) - n_val, len(dataset)))), batch_size=8)

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=cfg.learning_rate * 0.01)
    early = EarlyStopping(patience=patience)
    best_val = float("inf")

    for epoch in range(epochs):
        tr = train_latent_diffusion_epoch(model, train_loader, opt, device)
        va = eval_latent_diffusion(model, val_loader, device)
        sched.step()
        logger.info("Latent diffusion epoch %s train=%.4f val=%.4f", epoch + 1, tr, va)
        if va < best_val:
            best_val = va
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(cfg),
                    "vae_scale_factor": vae.scale_factor.detach().cpu(),
                },
                out_path,
            )
        if early(va):
            break


def stage3_finetune_eval(
    cfg: LatentDiffusionConfig,
    vae: TimeSeriesVAE,
    itrans_path: Path,
    latent_ckpt: Path,
    image_height: int,
    device: torch.device,
    smoke_test: bool,
) -> Dict:
    variate_indices = [0]
    train_ds, val_ds, test_ds, _ = load_dataset(
        "ETTh1",
        variate_indices,
        lookback=LOOKBACK_LENGTH,
        horizon=FORECAST_LENGTH,
        stride=24 if not smoke_test else LOOKBACK_LENGTH,
        lookback_overlap=LOOKBACK_OVERLAP,
    )
    if smoke_test:
        train_ds = Subset(train_ds, list(range(min(4, len(train_ds)))))
        val_ds = Subset(val_ds, list(range(min(2, len(val_ds)))))
        test_ds = Subset(test_ds, list(range(min(2, len(test_ds)))))

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=0)

    guidance = build_itrans_guidance(itrans_path, device)
    model = LatentDiffusionTSF(cfg, vae, guidance_model=guidance).to(device)
    ckpt = torch.load(latent_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-5,
    )
    epochs = 2 if smoke_test else min(30, FINETUNE_EPOCHS)
    patience = 1 if smoke_test else min(10, FINETUNE_PATIENCE)
    early = EarlyStopping(patience=patience)

    for epoch in range(epochs):
        train_latent_diffusion_epoch(model, train_loader, opt, device)
        va = eval_latent_diffusion(model, val_loader, device)
        logger.info("Finetune epoch %s val=%.4f", epoch + 1, va)
        if early(va):
            break

    model.eval()
    preds_lat = []
    trues = []
    with torch.no_grad():
        for past, future in test_loader:
            past = past.to(device)
            future = future.to(device)
            gen = model.generate(past, num_ddim_steps=20 if smoke_test else 50, verbose=False)
            pred = gen["prediction"]
            true = future[..., LOOKBACK_OVERLAP:]
            preds_lat.append(pred.cpu())
            trues.append(true.cpu())
    y_hat_t = torch.cat(preds_lat, dim=0)
    y_t = torch.cat(trues, dim=0)
    m_lat = compute_metrics(y_hat_t.squeeze(1), y_t.squeeze(1))

    base = create_pixel_diffusion_baseline(image_height).to(device)
    base.set_guidance_model(guidance)

    baseline_metrics = None
    baseline_ckpt = DEFAULT_7VAR_CKPT / "pretrained_dim7" / "pretrained_diffusion.pt"
    if baseline_ckpt.is_file():
        try:
            bck = torch.load(baseline_ckpt, map_location=device, weights_only=False)
            base.load_state_dict(bck["model_state_dict"], strict=False)
            base.eval()
            preds_b = []
            with torch.no_grad():
                for past, future in test_loader:
                    past = past.to(device)
                    future = future.to(device)
                    gen = base.generate(past, num_ddim_steps=20 if smoke_test else 50, verbose=False)
                    preds_b.append(gen["prediction"].cpu())
            y_hat_b = torch.cat(preds_b, dim=0)
            baseline_metrics = compute_metrics(y_hat_b.squeeze(1), y_t.squeeze(1))
        except Exception as e:
            logger.warning("Baseline eval skipped: %s", e)
    else:
        logger.info("No baseline checkpoint at %s; skipping pixel baseline comparison", baseline_ckpt)

    def _scalar(x):
        return float(x.detach().cpu().item()) if torch.is_tensor(x) else float(x)

    return {
        "latent_mse": _scalar(m_lat["mse"]),
        "latent_mae": _scalar(m_lat["mae"]),
        "forecast_horizon_evaluated": int(y_t.shape[-1]),
        "baseline_mse": _scalar(baseline_metrics["mse"]) if baseline_metrics else None,
        "baseline_mae": _scalar(baseline_metrics["mae"]) if baseline_metrics else None,
        "note": "Metrics on trimmed test forecasts (OT variate).",
    }


def run_smoke_validations(
    cfg: LatentDiffusionConfig,
    vae: TimeSeriesVAE,
    itrans_path: Path,
    device: torch.device,
    strict_vae_mse: bool,
) -> None:
    pixel_enc = PixelEncoder(cfg).to(device)
    pixel_enc.eval()
    vae.eval()
    loader = get_synthetic_dataloader(
        num_samples=32,
        lookback_length=LOOKBACK_LENGTH,
        forecast_length=FORECAST_LENGTH,
        batch_size=4,
        num_variables=1,
        lookback_overlap=LOOKBACK_OVERLAP,
        skip_cross_var_aug=True,
    )
    past, future = next(iter(loader))
    past = past.to(device)
    future = future.to(device)
    with torch.no_grad():
        img = unified_2d_from_batch(pixel_enc, past, future)
    mu, _ = vae.encode_mu_logvar(img)
    cz = cfg.latent_channels
    h, w = cfg.image_height // 4, img.shape[-1] // 4
    assert mu.shape == (img.shape[0], cz, h, w), (mu.shape, (img.shape[0], cz, h, w))

    with torch.no_grad():
        recon = vae.decode(mu)
        mse_rt = float(nn.functional.mse_loss(recon, img).item())
    if strict_vae_mse:
        assert mse_rt < 0.01, f"VAE round-trip MSE {mse_rt} >= 0.01"

    guidance = build_itrans_guidance(itrans_path, device)
    model = LatentDiffusionTSF(cfg, vae, guidance_model=guidance).to(device)
    past_norm, future_norm, stats = model._normalize_sequence(past, future)
    future_2d = model.encode_to_2d(future_norm)
    fw = future_2d.shape[-1]
    model.eval()
    with torch.no_grad():
        g2d = model._generate_guidance_2d(past, past_norm, stats, fw)
        zg = model._pixels_to_latent_scaled(g2d)
        zf = model._pixels_to_latent_scaled(future_2d)
    assert zg.shape[-2:] == zf.shape[-2:], (zg.shape, zf.shape)

    model.train()
    past.requires_grad_(False)
    future.requires_grad_(False)
    out = model(past, future)
    out["loss"].backward()
    for n, p in model.vae.named_parameters():
        g = p.grad
        assert g is None or g.abs().max().item() == 0.0, f"VAE grad {n}"
    for n, p in guidance.model.named_parameters():
        g = p.grad
        assert g is None or g.abs().max().item() == 0.0, f"iTrans grad {n}"
    unet_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.noise_predictor.parameters()
    )
    assert unet_grad, "U-Net should receive gradients"

    model.eval()
    with torch.no_grad():
        gen = model.generate(past[:2], num_ddim_steps=5, verbose=False)
    H = FORECAST_LENGTH
    assert gen["prediction"].shape == (2, 1, H), gen["prediction"].shape

    logger.info(
        "Smoke OK: latent %s, VAE roundtrip MSE=%.5f%s",
        list(mu.shape[1:]),
        mse_rt,
        " (strict)" if strict_vae_mse else "",
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image-height", type=int, choices=[96, 128], default=128)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--stage", type=str, default="all", choices=["0", "1", "2", "3", "all"])
    p.add_argument("--skip-vae-train", action="store_true")
    p.add_argument("--cache-dir", type=str, default=None)
    args = p.parse_args()

    device = get_device()
    cfg = build_latent_config(args.image_height)
    CKPT_LATENT.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    n_syn = 500 if args.smoke_test else SYNTHETIC_SAMPLES_FULL
    ep_vae = 2 if args.smoke_test else cfg.vae_epochs
    pat_vae = 1 if args.smoke_test else 15
    ep_pre = 2 if args.smoke_test else PRETRAIN_EPOCHS
    pat_pre = 1 if args.smoke_test else PRETRAIN_PATIENCE
    cache = args.cache_dir or str(CKPT_LATENT / "synth_cache")

    vae_path = CKPT_LATENT / f"vae_dim1_H{args.image_height}.pt"
    itrans_path = CKPT_LATENT / "itransformer_dim1.pt"
    latent_path = CKPT_LATENT / f"latent_diffusion_dim1_H{args.image_height}.pt"
    results_path = RESULTS_DIR / f"latent_experiment_H{args.image_height}.json"

    stages = {"0", "1", "2", "3"} if args.stage == "all" else {args.stage}

    vae = None
    if "0" in stages or "2" in stages or "3" in stages:
        if "0" in stages:
            if args.skip_vae_train:
                if not vae_path.is_file():
                    raise SystemExit(f"--skip-vae-train but missing {vae_path}")
                skip_train = True
            else:
                skip_train = False
        else:
            skip_train = True
            if not vae_path.is_file():
                raise SystemExit(f"Need VAE at {vae_path} (run --stage 0 first)")
        vae = stage0_train_vae(
            cfg,
            device,
            n_samples=n_syn,
            epochs=ep_vae,
            patience=pat_vae,
            cache_dir=cache,
            out_path=vae_path,
            skip_train=skip_train,
        )

    if "1" in stages:
        stage1_itransformer(CKPT_LATENT, n_syn, ep_pre, pat_pre, cache)

    if "2" in stages and vae is not None:
        if not itrans_path.is_file():
            raise SystemExit(f"Missing {itrans_path}; run stage 1 first")
        stage2_pretrain_latent(
            cfg,
            vae,
            itrans_path,
            device,
            n_syn,
            ep_pre,
            pat_pre,
            cache,
            latent_path,
        )

    if "3" in stages and vae is not None:
        if not latent_path.is_file():
            raise SystemExit(f"Missing {latent_path}; run stage 2 first")
        metrics = stage3_finetune_eval(
            cfg, vae, itrans_path, latent_path, args.image_height, device, args.smoke_test
        )
        with open(results_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Wrote %s", results_path)

    if args.smoke_test and vae is not None and itrans_path.is_file():
        run_smoke_validations(
            cfg,
            vae,
            itrans_path,
            device,
            strict_vae_mse=False,
        )


if __name__ == "__main__":
    main()
