"""Optuna tuning for iTransformer with MSE + truncated signature loss."""

from __future__ import annotations

import argparse
import gc
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import optuna
from optuna.samplers import TPESampler
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from models.diffusion_tsf.signature_mse_loss import SignatureMSELoss, SignatureMSELossOutput
from models.diffusion_tsf.train_multivariate_pipeline import create_itransformer, load_dataset


LOGGER = logging.getLogger("signature_tuning")


@dataclass(frozen=True)
class TrialParams:
    learning_rate: float
    alpha: float
    beta: float
    depth: int
    use_cumsum: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--signature-depth", type=int, default=None, help="fixed depth; else tuned per trial")
    parser.add_argument("--checkpoint-dir", default="results/signature_tune/ckpts")
    parser.add_argument("--storage", default="sqlite:///signature_tuning.db")
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


def load_data(args: argparse.Namespace):
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
    return future.permute(0, 2, 1)


def forecast_generator(generator: nn.Module, past: torch.Tensor) -> torch.Tensor:
    if past.dim() == 2:
        past = past.unsqueeze(1)
    x_enc = past.permute(0, 2, 1)
    seq_len = getattr(generator, "seq_len", x_enc.shape[1])
    if x_enc.shape[1] > seq_len:
        x_enc = x_enc[:, -seq_len:, :]
    return generator(x_enc, None, None, None)


def suggest_trial_params(trial: optuna.Trial, args: argparse.Namespace) -> TrialParams:
    depth = args.signature_depth
    if depth is None:
        depth = trial.suggest_categorical("signature_depth", [3, 4])
    return TrialParams(
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        alpha=trial.suggest_float("alpha", 0.25, 2.0, log=True),
        beta=trial.suggest_float("beta", 1e-4, 1.0, log=True),
        depth=int(depth),
        use_cumsum=trial.suggest_categorical("use_cumsum", [False, True]),
    )


def train_one_epoch(
    generator: nn.Module,
    criterion: SignatureMSELoss,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, float]:
    generator.train()
    totals = {"loss": 0.0, "mse": 0.0, "sig": 0.0, "sig_raw": 0.0}
    n_batches = 0

    for batch_idx, (past, future) in enumerate(loader):
        if args.max_train_batches and batch_idx >= args.max_train_batches:
            break

        past = past.to(device, non_blocking=True).float()
        target = prepare_target(future.to(device, non_blocking=True).float(), args.lookback_overlap)

        optimizer.zero_grad(set_to_none=True)
        forecast = forecast_generator(generator, past)
        out = criterion(forecast, target, return_parts=True)
        assert isinstance(out, SignatureMSELossOutput)
        out.loss.backward()
        optimizer.step()

        totals["loss"] += float(out.loss.detach().item())
        totals["mse"] += float(out.loss_mse.detach().item())
        totals["sig"] += float(out.loss_sig.detach().item())
        totals["sig_raw"] += float(out.loss_sig_raw.detach().item())
        n_batches += 1

    return {key: value / max(n_batches, 1) for key, value in totals.items()}


@torch.no_grad()
def validate(
    generator: nn.Module,
    criterion: SignatureMSELoss,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, float]:
    generator.eval()
    totals = {"loss": 0.0, "mse": 0.0, "sig": 0.0}
    n_batches = 0

    for batch_idx, (past, future) in enumerate(loader):
        if args.max_val_batches and batch_idx >= args.max_val_batches:
            break

        past = past.to(device, non_blocking=True).float()
        target = prepare_target(future.to(device, non_blocking=True).float(), args.lookback_overlap)
        forecast = forecast_generator(generator, past)
        out = criterion(forecast, target, return_parts=True)
        assert isinstance(out, SignatureMSELossOutput)
        totals["loss"] += float(out.loss.item())
        totals["mse"] += float(out.loss_mse.item())
        totals["sig"] += float(out.loss_sig.item())
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
    params = suggest_trial_params(trial, args)
    trial_seed = args.seed + trial.number
    set_seed(trial_seed)

    generator = create_itransformer(
        seq_len=args.lookback_length,
        pred_len=args.forecast_length,
        num_vars=num_variates,
    ).to(device)
    criterion = SignatureMSELoss(
        alpha=params.alpha,
        beta=params.beta,
        depth=params.depth,
        use_cumsum=params.use_cumsum,
        normalize_sig=True,
    ).to(device)
    optimizer = torch.optim.AdamW(generator.parameters(), lr=params.learning_rate)
    train_loader = make_loader(train_ds, args, shuffle=True, seed=trial_seed)
    val_loader = make_loader(val_ds, args, shuffle=False, seed=trial_seed)

    best_val = float("inf")
    best_epoch = -1
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.checkpoint_dir) / f"{args.dataset}_trial{trial.number}_best.pt"

    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(generator, criterion, train_loader, optimizer, device, args)
        val_metrics = validate(generator, criterion, val_loader, device, args)
        val_loss = val_metrics["loss"]

        LOGGER.info(
            "trial=%s epoch=%s train_loss=%.6f train_mse=%.6f train_sig=%.6f train_sig_raw=%.6f "
            "val_loss=%.6f val_mse=%.6f val_sig=%.6f",
            trial.number,
            epoch,
            train_metrics["loss"],
            train_metrics["mse"],
            train_metrics["sig"],
            train_metrics["sig_raw"],
            val_metrics["loss"],
            val_metrics["mse"],
            val_metrics["sig"],
        )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "generator_state_dict": generator.state_dict(),
                    "num_variates": num_variates,
                    "trial_params": params.__dict__,
                    "val_loss": best_val,
                    "epoch": epoch,
                },
                ckpt_path,
            )

        trial.report(val_loss, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    trial.set_user_attr("best_epoch", best_epoch)
    trial.set_user_attr("best_val_loss", best_val)
    trial.set_user_attr("checkpoint", str(ckpt_path))

    del generator, criterion, optimizer
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
    LOGGER.info("device=%s dataset=%s storage=%s", device, args.dataset, args.storage)

    train_ds, val_ds = load_data(args)
    num_variates = infer_num_variates(train_ds)
    study_name = args.study_name or f"signature_mse_{args.dataset}_v1"
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
