"""Optuna tuning for adversarial iTransformer PatchGAN regularization."""

from __future__ import annotations

import argparse
import gc
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import optuna
from optuna.samplers import TPESampler
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset

from models.diffusion_tsf.patchgan import (
    PatchGAN1D,
    PatchGAN2D,
    SoftCDFBinning,
    discriminator_loss,
    generator_loss,
    set_requires_grad,
)
from models.diffusion_tsf.train_multivariate_pipeline import create_itransformer, load_dataset


LOGGER = logging.getLogger("patchgan_tuning")


@dataclass(frozen=True)
class TrialParams:
    learning_rate_g: float
    learning_rate_d: float
    gan_loss_weight: float
    patchgan_receptive_field: int
    soft_binning_temperature: Optional[float] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=["all1d", "1d2d"], required=True)
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--n-variates", type=int, default=None)
    parser.add_argument("--variate-indices", default=None)
    parser.add_argument("--lookback-length", type=int, default=96)
    parser.add_argument("--forecast-length", type=int, default=96)
    parser.add_argument("--lookback-overlap", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--n-trials", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-train-batches", type=int, default=200)
    parser.add_argument("--max-val-batches", type=int, default=80)
    parser.add_argument("--num-bins", type=int, default=32)
    parser.add_argument("--value-min", type=float, default=-3.5)
    parser.add_argument("--value-max", type=float, default=3.5)
    parser.add_argument("--checkpoint-dir", default="results/patchgan_tune/ckpts")
    parser.add_argument("--storage", default="sqlite:///patchgan_tuning.db")
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_variate_indices(args: argparse.Namespace) -> Optional[list[int]]:
    if args.variate_indices:
        return [int(x.strip()) for x in args.variate_indices.split(",") if x.strip()]
    if args.n_variates is not None:
        return list(range(args.n_variates))
    return None


def maybe_subset(dataset, max_samples: Optional[int]):
    if max_samples is None or max_samples <= 0 or len(dataset) <= max_samples:
        return dataset
    return Subset(dataset, range(max_samples))


def ensure_sqlite_parent(storage: str) -> None:
    prefix = "sqlite:///"
    if not storage.startswith(prefix):
        return
    db_path = storage[len(prefix):]
    parent = os.path.dirname(db_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_patchgan_data(args: argparse.Namespace):
    variate_indices = parse_variate_indices(args)
    train_ds, val_ds, _, _ = load_dataset(
        args.dataset,
        variate_indices=variate_indices,
        lookback=args.lookback_length,
        horizon=args.forecast_length,
        stride=1,
        lookback_overlap=args.lookback_overlap,
    )
    if args.smoke_test:
        train_ds = maybe_subset(train_ds, args.batch_size)
        val_ds = maybe_subset(val_ds, args.batch_size)
    return train_ds, val_ds


def infer_num_variates(dataset) -> int:
    past, _ = dataset[0]
    return 1 if past.dim() == 1 else int(past.shape[0])


def make_loader(dataset, args: argparse.Namespace, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=generator if shuffle else None,
    )


def prepare_target(future: torch.Tensor, lookback_overlap: int) -> torch.Tensor:
    if future.dim() == 2:
        future = future.unsqueeze(1)
    if lookback_overlap > 0:
        future = future[..., lookback_overlap:]
    return future


def forecast_generator(generator: nn.Module, past: torch.Tensor) -> torch.Tensor:
    if past.dim() == 2:
        past = past.unsqueeze(1)
    x_enc = past.permute(0, 2, 1)
    seq_len = getattr(generator, "seq_len", x_enc.shape[1])
    if x_enc.shape[1] > seq_len:
        x_enc = x_enc[:, -seq_len:, :]
    forecast = generator(x_enc, None, None, None)
    return forecast.permute(0, 2, 1)


def suggest_trial_params(trial: optuna.Trial, architecture: str) -> TrialParams:
    params = TrialParams(
        learning_rate_g=trial.suggest_float("learning_rate_G", 1e-5, 1e-3, log=True),
        learning_rate_d=trial.suggest_float("learning_rate_D", 1e-5, 1e-3, log=True),
        gan_loss_weight=trial.suggest_float("gan_loss_weight", 1e-4, 1.0, log=True),
        patchgan_receptive_field=trial.suggest_categorical(
            "patchgan_receptive_field", [8, 16, 32]
        ),
        soft_binning_temperature=(
            trial.suggest_float("soft_binning_temperature", 0.01, 1.0, log=True)
            if architecture == "1d2d"
            else None
        ),
    )
    return params


def build_regularizer(
    args: argparse.Namespace,
    params: TrialParams,
    num_variates: int,
    device: torch.device,
) -> Tuple[nn.Module, Optional[SoftCDFBinning]]:
    if args.architecture == "all1d":
        discriminator = PatchGAN1D(
            in_channels=num_variates,
            receptive_field=params.patchgan_receptive_field,
        ).to(device)
        return discriminator, None

    binner = SoftCDFBinning(
        num_bins=args.num_bins,
        value_min=args.value_min,
        value_max=args.value_max,
        temperature=float(params.soft_binning_temperature),
        learnable_temperature=False,
    ).to(device)
    discriminator = PatchGAN2D(
        in_channels=num_variates,
        receptive_field=params.patchgan_receptive_field,
    ).to(device)
    return discriminator, binner


def encode_for_discriminator(
    series: torch.Tensor,
    binner: Optional[SoftCDFBinning],
) -> torch.Tensor:
    return series if binner is None else binner(series)


def train_one_epoch(
    generator: nn.Module,
    discriminator: nn.Module,
    binner: Optional[SoftCDFBinning],
    loader: DataLoader,
    optim_g: torch.optim.Optimizer,
    optim_d: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    params: TrialParams,
) -> Dict[str, float]:
    generator.train()
    discriminator.train()
    totals = {"mse": 0.0, "g_adv": 0.0, "d": 0.0, "g_total": 0.0}
    n_batches = 0

    for batch_idx, (past, future) in enumerate(loader):
        if args.max_train_batches and batch_idx >= args.max_train_batches:
            break

        past = past.to(device, non_blocking=True).float()
        target = prepare_target(future.to(device, non_blocking=True).float(), args.lookback_overlap)

        set_requires_grad(discriminator, True)
        optim_d.zero_grad(set_to_none=True)
        with torch.no_grad():
            fake = forecast_generator(generator, past)
        real_input = encode_for_discriminator(target, binner)
        fake_input = encode_for_discriminator(fake.detach(), binner)
        d_loss = discriminator_loss(discriminator(real_input), discriminator(fake_input))
        d_loss.backward()
        optim_d.step()

        set_requires_grad(discriminator, False)
        optim_g.zero_grad(set_to_none=True)
        forecast = forecast_generator(generator, past)
        mse_loss = F.mse_loss(forecast, target)
        fake_logits = discriminator(encode_for_discriminator(forecast, binner))
        g_adv = generator_loss(fake_logits)
        g_total = mse_loss + params.gan_loss_weight * g_adv
        g_total.backward()
        optim_g.step()
        set_requires_grad(discriminator, True)

        totals["mse"] += float(mse_loss.detach().item())
        totals["g_adv"] += float(g_adv.detach().item())
        totals["d"] += float(d_loss.detach().item())
        totals["g_total"] += float(g_total.detach().item())
        n_batches += 1

    return {key: value / max(n_batches, 1) for key, value in totals.items()}


@torch.no_grad()
def validate(
    generator: nn.Module,
    discriminator: nn.Module,
    binner: Optional[SoftCDFBinning],
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, float]:
    generator.eval()
    discriminator.eval()
    totals = {"mse": 0.0, "g_adv": 0.0}
    n_batches = 0

    for batch_idx, (past, future) in enumerate(loader):
        if args.max_val_batches and batch_idx >= args.max_val_batches:
            break

        past = past.to(device, non_blocking=True).float()
        target = prepare_target(future.to(device, non_blocking=True).float(), args.lookback_overlap)
        forecast = forecast_generator(generator, past)
        mse_loss = F.mse_loss(forecast, target)
        fake_logits = discriminator(encode_for_discriminator(forecast, binner))
        g_adv = generator_loss(fake_logits)
        totals["mse"] += float(mse_loss.item())
        totals["g_adv"] += float(g_adv.item())
        n_batches += 1

    return {key: value / max(n_batches, 1) for key, value in totals.items()}


def objective(
    trial: optuna.Trial,
    args: argparse.Namespace,
    train_ds,
    val_ds,
    num_variates: int,
    device: torch.device,
) -> float:
    params = suggest_trial_params(trial, args.architecture)
    trial_seed = args.seed + trial.number
    set_seed(trial_seed)

    generator = create_itransformer(
        seq_len=args.lookback_length,
        pred_len=args.forecast_length,
        num_vars=num_variates,
    ).to(device)
    discriminator, binner = build_regularizer(args, params, num_variates, device)

    optim_g = torch.optim.AdamW(generator.parameters(), lr=params.learning_rate_g)
    optim_d = torch.optim.AdamW(discriminator.parameters(), lr=params.learning_rate_d)
    train_loader = make_loader(train_ds, args, shuffle=True, seed=trial_seed)
    val_loader = make_loader(val_ds, args, shuffle=False, seed=trial_seed)

    best_val = float("inf")
    best_epoch = -1
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.checkpoint_dir) / (
        f"{args.dataset}_{args.architecture}_trial{trial.number}_best.pt"
    )

    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(
            generator,
            discriminator,
            binner,
            train_loader,
            optim_g,
            optim_d,
            device,
            args,
            params,
        )
        val_metrics = validate(generator, discriminator, binner, val_loader, device, args)
        val_mse = val_metrics["mse"]

        LOGGER.info(
            "trial=%s epoch=%s train_mse=%.6f train_g_adv=%.6f train_d=%.6f "
            "val_mse=%.6f val_g_adv=%.6f",
            trial.number,
            epoch,
            train_metrics["mse"],
            train_metrics["g_adv"],
            train_metrics["d"],
            val_metrics["mse"],
            val_metrics["g_adv"],
        )

        if val_mse < best_val:
            best_val = val_mse
            best_epoch = epoch
            torch.save(
                {
                    "generator_state_dict": generator.state_dict(),
                    "discriminator_state_dict": discriminator.state_dict(),
                    "architecture": args.architecture,
                    "num_variates": num_variates,
                    "trial_params": params.__dict__,
                    "val_mse": best_val,
                    "epoch": epoch,
                },
                ckpt_path,
            )

        trial.report(val_mse, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    trial.set_user_attr("best_epoch", best_epoch)
    trial.set_user_attr("best_val_mse", best_val)
    trial.set_user_attr("checkpoint", str(ckpt_path))

    del generator, discriminator, binner, optim_g, optim_d
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return best_val


def main() -> None:
    args = parse_args()
    setup_logging()

    if args.smoke_test:
        args.epochs = min(args.epochs, 1)
        args.max_train_batches = min(args.max_train_batches, 1)
        args.max_val_batches = min(args.max_val_batches, 1)
        args.n_trials = min(args.n_trials, 1)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("device=%s architecture=%s storage=%s", device, args.architecture, args.storage)

    train_ds, val_ds = load_patchgan_data(args)
    num_variates = infer_num_variates(train_ds)
    study_name = args.study_name or f"patchgan_{args.architecture}_{args.dataset}_v1"
    ensure_sqlite_parent(args.storage)

    study = optuna.create_study(
        study_name=study_name,
        storage=args.storage,
        direction="minimize",
        sampler=TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=2),
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(trial, args, train_ds, val_ds, num_variates, device),
        n_trials=args.n_trials,
        gc_after_trial=True,
    )

    LOGGER.info("best_value=%.6f best_params=%s", study.best_value, study.best_params)


if __name__ == "__main__":
    main()
