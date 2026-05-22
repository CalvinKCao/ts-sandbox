"""Optuna tuning for iTransformer with MSE + truncated signature loss."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
from optuna.samplers import TPESampler
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset

from models.diffusion_tsf.metrics import compute_metrics
from models.diffusion_tsf.signature_mse_loss import SignatureMSELoss, SignatureMSELossOutput
from models.diffusion_tsf.train_multivariate_pipeline import create_itransformer, load_dataset


LOGGER = logging.getLogger("signature_tuning")

# Metrics reported on the held-out test split (finalize / best-checkpoint only).
TEST_METRIC_KEYS = (
    "mse",
    "mae",
    "rmse",
    "trend_accuracy",
    "gradient_mae",
    "gradient_correlation",
    "sign_agreement",
    "shape_score",
    "signature_l2_raw",
    "signature_l2_normalized",
)


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
    parser.add_argument(
        "--max-test-batches",
        type=int,
        default=None,
        help="cap test batches (None = full test split; only used in finalize)",
    )
    parser.add_argument("--signature-depth", type=int, default=None, help="fixed depth; else tuned per trial")
    parser.add_argument("--checkpoint-dir", default="results/signature_tune/ckpts")
    parser.add_argument("--results-dir", default="results/signature_tune")
    parser.add_argument("--storage", default="sqlite:///signature_tuning.db")
    parser.add_argument("--study-name", default=None)
    parser.add_argument(
        "--resume-study",
        action="store_true",
        help="alias for joining an existing study (tuning already uses load_if_exists=True)",
    )
    parser.add_argument(
        "--new-study",
        action="store_true",
        help="force create-only; fails if study name already exists (not for Slurm arrays)",
    )
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="load study, eval best trial on full test split, train MSE baseline, write report",
    )
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


def default_study_name(dataset: str, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    run_id = os.environ.get("SLURM_ARRAY_JOB_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"signature_mse_{dataset}_job{run_id}"


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


def load_data(args: argparse.Namespace, include_test: bool = True):
    variate_indices = parse_variate_indices(args)
    train_ds, val_ds, test_ds, _ = load_dataset(
        args.dataset,
        variate_indices=variate_indices,
        lookback=args.lookback_length,
        horizon=args.forecast_length,
        stride=1,
        lookback_overlap=args.lookback_overlap,
    )
    if args.smoke_test:
        cap = args.batch_size
        train_ds = maybe_subset(train_ds, cap)
        val_ds = maybe_subset(val_ds, cap)
        if include_test:
            test_ds = maybe_subset(test_ds, cap)
    if include_test:
        return train_ds, val_ds, test_ds
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
    out = generator(x_enc, None, None, None)
    if isinstance(out, tuple):
        out = out[0]
    return out


@torch.no_grad()
def collect_forecasts(
    generator: nn.Module,
    loader: DataLoader,
    device: torch.device,
    lookback_overlap: int,
    max_batches: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Return preds/targets as ``[N, C, T]`` and number of windows evaluated."""
    generator.eval()
    preds_list, targets_list = [], []
    n_windows = 0
    for batch_idx, (past, future) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        past = past.to(device, non_blocking=True).float()
        target = prepare_target(future.to(device, non_blocking=True).float(), lookback_overlap)
        forecast = forecast_generator(generator, past)
        preds_list.append(forecast.permute(0, 2, 1).cpu())
        targets_list.append(target.permute(0, 2, 1).cpu())
        n_windows += int(past.shape[0])
    return torch.cat(preds_list, dim=0), torch.cat(targets_list, dim=0), n_windows


def _tensor_metric(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float(value)


def full_test_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    *,
    sig_depth: Optional[int] = None,
    use_cumsum: bool = False,
) -> Dict[str, float]:
    """All test metrics on the full collected forecast tensors."""
    metrics: Dict[str, float] = {}
    metrics["mse"] = F.mse_loss(preds, targets).item()
    metrics["mae"] = F.l1_loss(preds, targets).item()
    metrics["rmse"] = float(metrics["mse"] ** 0.5)

    pred_diff = preds[:, :, 1:] - preds[:, :, :-1]
    tgt_diff = targets[:, :, 1:] - targets[:, :, :-1]
    metrics["trend_accuracy"] = ((pred_diff > 0) == (tgt_diff > 0)).float().mean().item()

    n_channels = preds.shape[1]
    shape_accum: Dict[str, float] = {}
    for ch in range(n_channels):
        ch_metrics = compute_metrics(preds[:, ch, :], targets[:, ch, :])
        for name, value in ch_metrics.items():
            shape_accum[name] = shape_accum.get(name, 0.0) + _tensor_metric(value)
    for name, total in shape_accum.items():
        if name in ("mse", "mae"):
            continue
        metrics[name] = total / n_channels

    if sig_depth is not None:
        criterion = SignatureMSELoss(
            alpha=0.0,
            beta=1.0,
            depth=sig_depth,
            use_cumsum=use_cumsum,
            normalize_sig=False,
        )
        y_hat = preds.permute(0, 2, 1)
        y = targets.permute(0, 2, 1)
        norm_sig, raw_sig = criterion._signature_l2(y_hat, y)
        metrics["signature_l2_raw"] = float(raw_sig.item())
        metrics["signature_l2_normalized"] = float(norm_sig.item())

    return metrics


@torch.no_grad()
def evaluate_on_test_split(
    generator: nn.Module,
    test_ds,
    args: argparse.Namespace,
    device: torch.device,
    *,
    sig_depth: Optional[int] = None,
    use_cumsum: bool = False,
) -> Dict[str, float]:
    test_loader = make_loader(test_ds, args, shuffle=False, seed=args.seed + 20_000)
    preds, targets, n_windows = collect_forecasts(
        generator,
        test_loader,
        device,
        args.lookback_overlap,
        max_batches=args.max_test_batches,
    )
    metrics = full_test_metrics(
        preds,
        targets,
        sig_depth=sig_depth,
        use_cumsum=use_cumsum,
    )
    metrics["n_test_windows"] = float(n_windows)
    return metrics


def params_from_best_trial(study: optuna.Study, best: optuna.trial.FrozenTrial) -> TrialParams:
    if "trial_params" in best.user_attrs:
        raw = best.user_attrs["trial_params"]
        return TrialParams(**raw)

    bp = study.best_params
    depth = bp.get("signature_depth")
    if depth is None:
        raise KeyError("best trial missing signature_depth in params and trial_params")
    return TrialParams(
        learning_rate=float(bp["learning_rate"]),
        alpha=float(bp["alpha"]),
        beta=float(bp["beta"]),
        depth=int(depth),
        use_cumsum=bool(bp["use_cumsum"]),
    )


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
    criterion: nn.Module,
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
        if isinstance(criterion, SignatureMSELoss):
            out = criterion(forecast, target, return_parts=True)
            assert isinstance(out, SignatureMSELossOutput)
            loss = out.loss
            totals["mse"] += float(out.loss_mse.detach().item())
            totals["sig"] += float(out.loss_sig.detach().item())
            totals["sig_raw"] += float(out.loss_sig_raw.detach().item())
        else:
            loss = criterion(forecast, target)
            totals["mse"] += float(loss.detach().item())

        loss.backward()
        optimizer.step()
        totals["loss"] += float(loss.detach().item())
        n_batches += 1

    return {key: value / max(n_batches, 1) for key, value in totals.items()}


@torch.no_grad()
def validate(
    generator: nn.Module,
    criterion: nn.Module,
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
        if isinstance(criterion, SignatureMSELoss):
            out = criterion(forecast, target, return_parts=True)
            assert isinstance(out, SignatureMSELossOutput)
            totals["loss"] += float(out.loss.item())
            totals["mse"] += float(out.loss_mse.item())
            totals["sig"] += float(out.loss_sig.item())
        else:
            mse = F.mse_loss(forecast, target)
            totals["loss"] += float(mse.item())
            totals["mse"] += float(mse.item())
        n_batches += 1

    return {key: value / max(n_batches, 1) for key, value in totals.items()}


def train_mse_baseline(
    args: argparse.Namespace,
    train_ds,
    val_ds,
    num_variates: int,
    learning_rate: float,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, float]]:
    """Train iTransformer with plain MSE (same schedule budget as signature trials)."""
    generator = create_itransformer(
        seq_len=args.lookback_length,
        pred_len=args.forecast_length,
        num_vars=num_variates,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(generator.parameters(), lr=learning_rate)
    train_loader = make_loader(train_ds, args, shuffle=True, seed=args.seed + 10_000)
    val_loader = make_loader(val_ds, args, shuffle=False, seed=args.seed + 10_001)

    best_val = float("inf")
    best_state = None
    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(generator, criterion, train_loader, optimizer, device, args)
        val_metrics = validate(generator, criterion, val_loader, device, args)
        LOGGER.info(
            "[mse_baseline] epoch=%s train_mse=%.6f val_mse=%.6f",
            epoch,
            train_metrics["mse"],
            val_metrics["mse"],
        )
        if val_metrics["mse"] < best_val:
            best_val = val_metrics["mse"]
            best_state = {k: v.detach().cpu().clone() for k, v in generator.state_dict().items()}

    if best_state is not None:
        generator.load_state_dict(best_state)
    return generator, {"best_val_mse": best_val}


def log_test_metrics(label: str, metrics: Dict[str, float]) -> None:
    parts = [f"{k}={metrics[k]:.6f}" for k in TEST_METRIC_KEYS if k in metrics]
    extra = [f"n_test_windows={int(metrics.get('n_test_windows', 0))}"]
    LOGGER.info("[%s] test metrics: %s | %s", label, " ".join(parts), " ".join(extra))


def compute_deltas(sig: Dict[str, float], base: Dict[str, float]) -> Dict[str, float]:
    deltas = {}
    for key in TEST_METRIC_KEYS:
        if key in sig and key in base:
            deltas[f"delta_{key}"] = sig[key] - base[key]
    return deltas


def save_test_report(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
    LOGGER.info("wrote test report: %s", path)


def finalize_study(
    study: optuna.Study,
    args: argparse.Namespace,
    test_ds,
    num_variates: int,
    device: torch.device,
) -> None:
    """Eval best Optuna trial + MSE-only baseline on the full test split."""
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        LOGGER.warning("no completed trials; skipping finalize")
        return

    best = study.best_trial
    ckpt_path = best.user_attrs.get("checkpoint")
    if not ckpt_path or not Path(ckpt_path).is_file():
        LOGGER.warning("best trial %s has no checkpoint at %s; skipping finalize", best.number, ckpt_path)
        return

    params = params_from_best_trial(study, best)
    LOGGER.info(
        "finalize dataset=%s study=%s best_trial=%s val_loss=%.6f params=%s",
        args.dataset,
        study.study_name,
        best.number,
        float(best.value),
        asdict(params),
    )

    generator = create_itransformer(
        seq_len=args.lookback_length,
        pred_len=args.forecast_length,
        num_vars=num_variates,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    generator.load_state_dict(ckpt["generator_state_dict"])

    sig_test = evaluate_on_test_split(
        generator,
        test_ds,
        args,
        device,
        sig_depth=params.depth,
        use_cumsum=params.use_cumsum,
    )
    log_test_metrics("best_signature", sig_test)

    train_ds, val_ds = load_data(args, include_test=False)
    baseline_model, baseline_val = train_mse_baseline(
        args, train_ds, val_ds, num_variates, params.learning_rate, device
    )
    baseline_path = Path(args.checkpoint_dir) / f"{args.dataset}_mse_baseline.pt"
    torch.save(
        {
            "generator_state_dict": baseline_model.state_dict(),
            "num_variates": num_variates,
            "learning_rate": params.learning_rate,
            "val_mse": baseline_val["best_val_mse"],
        },
        baseline_path,
    )

    baseline_test = evaluate_on_test_split(baseline_model, test_ds, args, device)
    log_test_metrics("mse_baseline", baseline_test)

    deltas = compute_deltas(sig_test, baseline_test)
    report = {
        "dataset": args.dataset,
        "study_name": study.study_name,
        "best_trial": best.number,
        "best_val_loss": float(best.value),
        "best_params": study.best_params,
        "trial_params": asdict(params),
        "signature_checkpoint": str(ckpt_path),
        "mse_baseline_checkpoint": str(baseline_path),
        "test_split": "held_out_test",
        "test_signature_model": sig_test,
        "test_mse_baseline": baseline_test,
        "test_deltas_signature_minus_baseline": deltas,
        "finalized_at": datetime.now().isoformat(),
    }

    out_path = Path(args.results_dir) / f"{args.dataset}_{study.study_name}_test_report.json"
    save_test_report(out_path, report)

    del generator, baseline_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


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
            train_metrics.get("sig", 0.0),
            train_metrics.get("sig_raw", 0.0),
            val_metrics["loss"],
            val_metrics["mse"],
            val_metrics.get("sig", 0.0),
        )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "generator_state_dict": generator.state_dict(),
                    "num_variates": num_variates,
                    "trial_params": asdict(params),
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
    trial.set_user_attr("trial_params", asdict(params))

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
        args.max_test_batches = 1
        args.n_trials = min(args.n_trials, 1)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    study_name = default_study_name(args.dataset, args.study_name)
    # Slurm array workers for the same dataset share one study name; they must join, not recreate.
    load_if_exists = not args.new_study
    ensure_sqlite_parent(args.storage)

    LOGGER.info(
        "device=%s dataset=%s study=%s storage=%s load_if_exists=%s mode=%s",
        device,
        args.dataset,
        study_name,
        args.storage,
        load_if_exists,
        "finalize-only" if args.finalize_only else "tune",
    )

    if args.finalize_only:
        _, _, test_ds = load_data(args, include_test=True)
        num_variates = infer_num_variates(test_ds)
        study = optuna.load_study(study_name=study_name, storage=args.storage)
        finalize_study(study, args, test_ds, num_variates, device)
        return

    train_ds, val_ds = load_data(args, include_test=False)
    num_variates = infer_num_variates(train_ds)

    study = optuna.create_study(
        study_name=study_name,
        storage=args.storage,
        direction="minimize",
        sampler=TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=2),
        load_if_exists=load_if_exists,
    )
    study.optimize(
        lambda trial: objective(trial, args, train_ds, val_ds, num_variates, device),
        n_trials=args.n_trials,
        gc_after_trial=True,
    )

    LOGGER.info(
        "tuning done dataset=%s best_val_loss=%.6f best_params=%s (run finalize-only for test eval)",
        args.dataset,
        study.best_value,
        study.best_params,
    )


if __name__ == "__main__":
    main()
