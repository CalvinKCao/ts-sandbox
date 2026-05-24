"""Optuna tuning for log-signature latent diffusion (SimDiff-style patch DDPM)."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
import torch
import torch.nn.functional as F
from optuna.samplers import TPESampler
from torch.utils.data import DataLoader, Subset

from models.diffusion_tsf.metrics import compute_metrics
from models.diffusion_tsf.signature_diffusion_model import (
    SignatureDiffusionConfig,
    SignatureDiffusionModel,
)
from models.diffusion_tsf.signature_latent import LatentConfig, select_channels
from models.diffusion_tsf.signature_mse_loss import SignatureMSELoss
from models.diffusion_tsf.train_multivariate_pipeline import load_dataset
from models.diffusion_tsf.variate_subsets import generate_variate_subsets

LOGGER = logging.getLogger("signature_diffusion_tuning")

TEST_METRIC_KEYS = (
    "mse",
    "mae",
    "rmse",
    "trend_accuracy",
    "gradient_mae",
    "gradient_correlation",
    "sign_agreement",
    "shape_score",
)


@dataclass(frozen=True)
class TrialParams:
    learning_rate: float
    depth: int
    use_cumsum: bool
    patch_size: int
    patch_stride: int
    normalize_logsig: bool
    diff_steps: int
    sample_steps: int
    d_model: int
    n_layers: int
    n_heads: int
    dropout: float
    decoder_hidden: int
    lambda_point: float
    lambda_logsig_consistency: float
    loss_type: str
    subset_scheme: str
    subset_size: int
    subset_stride: int
    max_branches: int
    cond_mode: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="ETTh1")
    p.add_argument("--n-variates", type=int, default=None)
    p.add_argument("--lookback-length", type=int, default=96)
    p.add_argument("--forecast-length", type=int, default=96)
    p.add_argument("--lookback-overlap", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--n-trials", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-train-batches", type=int, default=200)
    p.add_argument("--max-val-batches", type=int, default=80)
    p.add_argument("--max-test-batches", type=int, default=None)
    p.add_argument("--checkpoint-dir", default="results/signature_diffusion/ckpts")
    p.add_argument("--results-dir", default="results/signature_diffusion")
    p.add_argument("--storage", default="sqlite:///signature_diffusion_tuning.db")
    p.add_argument("--study-name", default=None)
    p.add_argument("--new-study", action="store_true")
    p.add_argument("--finalize-only", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


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
    return f"signature_diffusion_{dataset}_job{run_id}"


def parse_n_variates(args: argparse.Namespace) -> Optional[int]:
    if args.n_variates is not None:
        return args.n_variates
    return None


def maybe_subset(dataset, max_samples: Optional[int]):
    if max_samples is None or max_samples <= 0 or len(dataset) <= max_samples:
        return dataset
    return Subset(dataset, range(max_samples))


def ensure_sqlite_parent(storage: str) -> None:
    prefix = "sqlite:///"
    if storage.startswith(prefix):
        parent = os.path.dirname(storage[len(prefix):])
        if parent:
            os.makedirs(parent, exist_ok=True)


def load_data(args: argparse.Namespace, include_test: bool = True):
    train_ds, val_ds, test_ds, _ = load_dataset(
        args.dataset,
        variate_indices=None,
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
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=gen if shuffle else None,
    )


def prepare_future_btc(future: torch.Tensor, lookback_overlap: int) -> torch.Tensor:
    if future.dim() == 2:
        future = future.unsqueeze(0)
    if lookback_overlap > 0:
        future = future[..., lookback_overlap:]
    return future.permute(0, 2, 1)


def model_n_channels(num_variates: int, params: TrialParams) -> int:
    if params.subset_scheme == "all":
        return num_variates
    return min(params.subset_size, num_variates)


def build_model(
    args: argparse.Namespace,
    params: TrialParams,
    num_variates: int,
    device: torch.device,
) -> SignatureDiffusionModel:
    n_ch = model_n_channels(num_variates, params)
    # Local WSL signatory often segfaults on logsignature(); cluster jobs use logsignature.
    latent_rep = "signature" if args.smoke_test else "logsignature"
    latent = LatentConfig(
        depth=params.depth,
        use_cumsum=params.use_cumsum,
        patch_size=params.patch_size,
        patch_stride=params.patch_stride,
        normalize_logsig=params.normalize_logsig,
        latent_rep=latent_rep,
    )
    cfg = SignatureDiffusionConfig(
        n_channels=n_ch,
        lookback=args.lookback_length,
        horizon=args.forecast_length - args.lookback_overlap,
        diff_steps=params.diff_steps,
        d_model=params.d_model,
        n_layers=params.n_layers,
        n_heads=params.n_heads,
        dropout=params.dropout,
        decoder_hidden=params.decoder_hidden,
        lambda_point=params.lambda_point,
        lambda_logsig_consistency=params.lambda_logsig_consistency,
        loss_type=params.loss_type,
        latent=latent,
        subset_scheme=params.subset_scheme,
        subset_size=params.subset_size if params.subset_scheme != "all" else None,
        subset_stride=params.subset_stride,
        max_branches=params.max_branches,
        sample_steps=params.sample_steps,
    )
    return SignatureDiffusionModel(cfg).to(device)


def sample_subset(
    num_variates: int,
    params: TrialParams,
    rng: random.Random,
) -> Tuple[int, ...]:
    subsets = generate_variate_subsets(
        num_variates,
        scheme=params.subset_scheme,
        subset_size=params.subset_size,
        subset_stride=params.subset_stride,
        max_branches=params.max_branches,
        seed=rng.randint(0, 2**30),
    )
    n_ch = model_n_channels(num_variates, params)
    valid = [s for s in subsets if len(s) == n_ch]
    if not valid:
        return tuple(range(n_ch))
    return rng.choice(valid)


def train_batch(
    model: SignatureDiffusionModel,
    past: torch.Tensor,
    future_btc: torch.Tensor,
    params: TrialParams,
    num_variates: int,
    rng: random.Random,
) -> Tuple[torch.Tensor, dict]:
    n_ch = model_n_channels(num_variates, params)
    past_btc = past.permute(0, 2, 1)
    fut = future_btc
    if params.subset_scheme != "all":
        s = sample_subset(num_variates, params, rng)
        past_btc = select_channels(past_btc, s)
        fut = select_channels(fut, s)
    return model.forward_branch(past_btc, fut)


def train_one_epoch(
    model: SignatureDiffusionModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    params: TrialParams,
    num_variates: int,
    rng: random.Random,
) -> Dict[str, float]:
    model.train()
    totals: Dict[str, float] = {}
    n = 0
    for bi, (past, future) in enumerate(loader):
        if args.max_train_batches and bi >= args.max_train_batches:
            break
        past = past.to(device).float()
        future_btc = prepare_future_btc(future.to(device).float(), args.lookback_overlap)
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = train_batch(model, past, future_btc, params, num_variates, rng)
        loss.backward()
        optimizer.step()
        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + float(v.item())
        n += 1
    return {k: v / max(n, 1) for k, v in totals.items()}


@torch.no_grad()
def validate(
    model: SignatureDiffusionModel,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    params: TrialParams,
    num_variates: int,
    rng: random.Random,
) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = {}
    n = 0
    for bi, (past, future) in enumerate(loader):
        if args.max_val_batches and bi >= args.max_val_batches:
            break
        past = past.to(device).float()
        future_btc = prepare_future_btc(future.to(device).float(), args.lookback_overlap)
        _, metrics = train_batch(model, past, future_btc, params, num_variates, rng)
        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + float(v.item())
        n += 1
    return {k: v / max(n, 1) for k, v in totals.items()}


@torch.no_grad()
def collect_predictions(
    model: SignatureDiffusionModel,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    num_variates: int,
    max_batches: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    model.eval()
    preds, targets = [], []
    n_win = 0
    horizon = args.forecast_length - args.lookback_overlap
    for bi, (past, future) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        past = past.to(device).float()
        future_btc = prepare_future_btc(future.to(device).float(), args.lookback_overlap)
        pred_btc = model.predict(past, horizon=horizon, n_variates=num_variates)
        preds.append(pred_btc.permute(0, 2, 1).cpu())
        targets.append(future_btc.permute(0, 2, 1).cpu())
        n_win += past.size(0)
    return torch.cat(preds, 0), torch.cat(targets, 0), n_win


def full_test_metrics(preds: torch.Tensor, targets: torch.Tensor, *, sig_depth: int, use_cumsum: bool) -> Dict[str, float]:
    metrics: Dict[str, float] = {
        "mse": F.mse_loss(preds, targets).item(),
        "mae": F.l1_loss(preds, targets).item(),
    }
    metrics["rmse"] = float(metrics["mse"] ** 0.5)
    pred_diff = preds[:, :, 1:] - preds[:, :, :-1]
    tgt_diff = targets[:, :, 1:] - targets[:, :, :-1]
    metrics["trend_accuracy"] = ((pred_diff > 0) == (tgt_diff > 0)).float().mean().item()

    n_ch = preds.shape[1]
    shape_accum: Dict[str, float] = {}
    for ch in range(n_ch):
        ch_m = compute_metrics(preds[:, ch, :], targets[:, ch, :])
        for name, val in ch_m.items():
            if name in ("mse", "mae"):
                continue
            shape_accum[name] = shape_accum.get(name, 0.0) + float(val.item() if torch.is_tensor(val) else val)
    for name, total in shape_accum.items():
        metrics[name] = total / n_ch

    crit = SignatureMSELoss(alpha=0.0, beta=1.0, depth=sig_depth, use_cumsum=use_cumsum, normalize_sig=False)
    y_hat = preds.permute(0, 2, 1)
    y = targets.permute(0, 2, 1)
    norm_sig, raw_sig = crit._signature_l2(y_hat, y)
    metrics["signature_l2_raw"] = float(raw_sig.item())
    metrics["signature_l2_normalized"] = float(norm_sig.item())
    return metrics


def suggest_trial_params(trial: optuna.Trial, args: argparse.Namespace, num_variates: int) -> TrialParams:
    patch_size = trial.suggest_categorical("patch_size", [16, 24, 32])
    stride_choices = [max(4, patch_size // 2), max(4, patch_size // 4)]
    scheme = trial.suggest_categorical("subset_scheme", ["all", "sliding", "pairs"])
    if scheme == "all":
        subset_size = num_variates
    else:
        subset_size = trial.suggest_int("subset_size", 2, min(4, num_variates))

    return TrialParams(
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 3e-4, log=True),
        depth=trial.suggest_categorical("signature_depth", [3, 4]),
        use_cumsum=trial.suggest_categorical("use_cumsum", [False, True]),
        patch_size=patch_size,
        patch_stride=trial.suggest_categorical("patch_stride", stride_choices),
        normalize_logsig=trial.suggest_categorical("normalize_logsig", [True, False]),
        diff_steps=trial.suggest_categorical("diff_steps", [50, 100]),
        sample_steps=trial.suggest_categorical("sample_steps", [10, 20]),
        d_model=trial.suggest_categorical("d_model", [128, 256]),
        n_layers=trial.suggest_int("n_layers", 1, 2),
        n_heads=trial.suggest_categorical("num_heads", [4, 8]),
        dropout=trial.suggest_float("dropout", 0.0, 0.15),
        decoder_hidden=trial.suggest_categorical("decoder_hidden", [256, 512]),
        lambda_point=trial.suggest_float("lambda_point", 0.5, 2.0),
        lambda_logsig_consistency=trial.suggest_float("lambda_logsig_consistency", 0.1, 0.5),
        loss_type=trial.suggest_categorical("loss_type", ["l1", "mse"]),
        subset_scheme=scheme,
        subset_size=subset_size,
        subset_stride=trial.suggest_int("subset_stride", 1, 2),
        max_branches=trial.suggest_int("max_branches", 1, 5),
        cond_mode=trial.suggest_categorical("cond_mode", ["logsig_patches"]),
    )


def params_from_trial(trial: optuna.trial.FrozenTrial) -> TrialParams:
    raw = trial.user_attrs.get("trial_params")
    if raw:
        return TrialParams(**raw)
    bp = trial.params
    return TrialParams(
        learning_rate=float(bp["learning_rate"]),
        depth=int(bp["signature_depth"]),
        use_cumsum=bool(bp["use_cumsum"]),
        patch_size=int(bp["patch_size"]),
        patch_stride=int(bp["patch_stride"]),
        normalize_logsig=bool(bp["normalize_logsig"]),
        diff_steps=int(bp["diff_steps"]),
        sample_steps=int(bp["sample_steps"]),
        d_model=int(bp["d_model"]),
        n_layers=int(bp["n_layers"]),
        n_heads=int(bp["num_heads"]),
        dropout=float(bp["dropout"]),
        decoder_hidden=int(bp["decoder_hidden"]),
        lambda_point=float(bp["lambda_point"]),
        lambda_logsig_consistency=float(bp["lambda_logsig_consistency"]),
        loss_type=str(bp["loss_type"]),
        subset_scheme=str(bp["subset_scheme"]),
        subset_size=int(bp.get("subset_size", 7)),
        subset_stride=int(bp["subset_stride"]),
        max_branches=int(bp["max_branches"]),
        cond_mode=str(bp.get("cond_mode", "logsig_patches")),
    )


def finalize_study(
    study: optuna.Study,
    args: argparse.Namespace,
    test_ds,
    num_variates: int,
    device: torch.device,
) -> None:
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        LOGGER.warning("no completed trials")
        return
    best = study.best_trial
    ckpt_path = best.user_attrs.get("checkpoint")
    if not ckpt_path or not Path(ckpt_path).is_file():
        LOGGER.warning("missing checkpoint for best trial")
        return

    params = params_from_trial(best)
    model = build_model(args, params, num_variates, device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    loader = make_loader(test_ds, args, shuffle=False, seed=args.seed + 99)
    preds, targets, n_win = collect_predictions(
        model, loader, device, args, num_variates, args.max_test_batches
    )
    test_metrics = full_test_metrics(
        preds, targets, sig_depth=params.depth, use_cumsum=params.use_cumsum
    )
    test_metrics["n_test_windows"] = float(n_win)

    report = {
        "dataset": args.dataset,
        "study_name": study.study_name,
        "best_trial": best.number,
        "best_val_loss": float(best.value),
        "trial_params": asdict(params),
        "checkpoint": str(ckpt_path),
        "test_metrics": test_metrics,
        "finalized_at": datetime.now().isoformat(),
    }
    out = Path(args.results_dir) / f"{args.dataset}_{study.study_name}_test_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    LOGGER.info("wrote %s mse=%.6f", out, test_metrics["mse"])


def objective(
    trial: optuna.Trial,
    args: argparse.Namespace,
    train_ds,
    val_ds,
    num_variates: int,
    device: torch.device,
) -> float:
    params = suggest_trial_params(trial, args, num_variates)
    trial_seed = args.seed + trial.number
    set_seed(trial_seed)
    rng = random.Random(trial_seed)

    model = build_model(args, params, num_variates, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params.learning_rate)
    train_loader = make_loader(train_ds, args, shuffle=True, seed=trial_seed)
    val_loader = make_loader(val_ds, args, shuffle=False, seed=trial_seed)

    ckpt_path = Path(args.checkpoint_dir) / f"{args.dataset}_trial{trial.number}_best.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(args.epochs):
        train_m = train_one_epoch(
            model, train_loader, optimizer, device, args, params, num_variates, rng
        )
        val_m = validate(model, val_loader, device, args, params, num_variates, rng)
        val_loss = val_m["loss"]
        LOGGER.info(
            "trial=%s epoch=%s train_loss=%.4f val_loss=%.4f val_point=%.4f",
            trial.number,
            epoch,
            train_m["loss"],
            val_loss,
            val_m.get("loss_point", 0.0),
        )
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_variates": num_variates,
                    "trial_params": asdict(params),
                    "val_loss": best_val,
                },
                ckpt_path,
            )
        trial.report(val_loss, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    trial.set_user_attr("checkpoint", str(ckpt_path))
    trial.set_user_attr("trial_params", asdict(params))
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return best_val


def main() -> None:
    args = parse_args()
    setup_logging()
    if args.smoke_test:
        args.epochs = 1
        args.max_train_batches = 1
        args.max_val_batches = 1
        args.max_test_batches = 1
        args.n_trials = 1
        args.batch_size = min(args.batch_size, 4)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    study_name = default_study_name(args.dataset, args.study_name)
    ensure_sqlite_parent(args.storage)

    if args.finalize_only:
        train_ds, val_ds, test_ds = load_data(args)
        del train_ds, val_ds
        num_variates = parse_n_variates(args) or infer_num_variates(test_ds)
        study = optuna.load_study(study_name=study_name, storage=args.storage)
        finalize_study(study, args, test_ds, num_variates, device)
        return

    train_ds, val_ds, test_ds = load_data(args)
    del test_ds
    num_variates = parse_n_variates(args) or infer_num_variates(train_ds)
    if args.dataset == "exchange_rate" and args.n_variates is None:
        num_variates = 8

    load_kw = {"load_if_exists": True}
    if args.new_study:
        load_kw = {"load_if_exists": False}

    study = optuna.create_study(
        study_name=study_name,
        storage=args.storage,
        direction="minimize",
        sampler=TPESampler(seed=args.seed),
        **load_kw,
    )

    for _ in range(args.n_trials):
        study.optimize(
            lambda t: objective(t, args, train_ds, val_ds, num_variates, device),
            n_trials=1,
        )

    LOGGER.info("study=%s best=%.6f", study_name, study.best_value if study.trials else float("nan"))


if __name__ == "__main__":
    main()
