"""Optuna hyperparameter search for MMPD subset training."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import optuna
from optuna.samplers import TPESampler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.eval_mmpd_gaussian_anchor import (  # noqa: E402
    AnchorRun,
    build_mmpd_train_cmd,
    mmpd_env_for_run,
    run_cmd,
    stage_mmpd_dataset_for_run,
)
from utils.mmpd_paper_hparams import (  # noqa: E402
    DEFAULT_MMPD_HPARAMS,
    tuning_result_path,
)

_VALI_RE = re.compile(r"Vali Loss:\s*([0-9.eE+-]+)")

_DEFAULT_TUNE_SPEC: Dict[str, Any] = {
    "learning_rate": [3e-5, 3e-4],
    "point_weight": [0.005, 0.05],
    "dropout": 0.2,
    "ema_decay": [0.95, 0.99, 0.999],
}


def _tune_spec(args: Any) -> Dict[str, Any]:
    spec = getattr(args, "mmpd_tune_params", None)
    if isinstance(spec, dict) and spec:
        return dict(spec)
    return dict(_DEFAULT_TUNE_SPEC)


def _suggest_tune_hparams(trial: optuna.Trial, args: Any, run: AnchorRun) -> Dict[str, Any]:
    spec = _tune_spec(args)
    hparams: Dict[str, Any] = {}

    lr_spec = spec.get("learning_rate", _DEFAULT_TUNE_SPEC["learning_rate"])
    lo, hi = float(lr_spec[0]), float(lr_spec[1])
    hparams["learning_rate"] = trial.suggest_float("learning_rate", lo, hi, log=True)

    pw_spec = spec.get("point_weight", _DEFAULT_TUNE_SPEC["point_weight"])
    lo, hi = float(pw_spec[0]), float(pw_spec[1])
    hparams["point_weight"] = trial.suggest_float("point_weight", lo, hi, log=True)

    hparams["dropout"] = float(spec.get("dropout", 0.2))

    if "batch_size" in spec:
        bs_lo, bs_hi = int(spec["batch_size"][0]), int(spec["batch_size"][1])
        hparams["batch_size"] = trial.suggest_int("batch_size", bs_lo, bs_hi)

    ema_spec = spec.get("ema_decay")
    if ema_spec is not None:
        choices = [float(x) for x in ema_spec]
        hparams["ema_decay"] = trial.suggest_categorical("ema_decay", choices)

    if args.mmpd_backbone == "MaskAE":
        fl_spec = spec.get("finetune_layers", [1, 3])
        hparams["finetune_layers"] = trial.suggest_int(
            "finetune_layers", int(fl_spec[0]), int(fl_spec[1])
        )
        nn_spec = spec.get("neighbor_num")
        if nn_spec is not None:
            hparams["neighbor_num"] = trial.suggest_int(
                "neighbor_num", int(nn_spec[0]), int(nn_spec[1])
            )
        else:
            min_k, max_k = _maskae_neighbor_bounds(run)
            hparams["neighbor_num"] = trial.suggest_int("neighbor_num", min_k, max_k)

    return hparams


def _maskae_neighbor_bounds(run: AnchorRun) -> tuple[int, int]:
    n_variates = len(run.metadata.get("variate_indices", []))
    if n_variates < 1:
        raise ValueError(f"no variate_indices in anchor metadata for {run.dataset}")
    max_k = n_variates
    min_k = min(5, max_k)
    return min_k, max_k


def _trial_output_root(args: Any, dataset: str, trial_number: int) -> Path:
    return args.output_dir / "mmpd_tune_runs" / f"{dataset}_trial{trial_number:02d}"


def _parse_min_vali_loss(log_path: Path) -> float:
    if not log_path.is_file():
        raise FileNotFoundError(f"Missing tune log: {log_path}")
    best: Optional[float] = None
    with log_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            m = _VALI_RE.search(line)
            if not m:
                continue
            val = float(m.group(1))
            best = val if best is None else min(best, val)
    if best is None:
        raise RuntimeError(f"No validation loss found in {log_path}")
    return best


def run_mmpd_train_trial(
    args: Any,
    run: AnchorRun,
    hparams: Dict[str, Any],
    *,
    trial_number: int,
    train_epochs: int,
    patience: int,
) -> float:
    stage_mmpd_dataset_for_run(args.mmpd_data_dir, run)
    out_root = _trial_output_root(args, run.dataset, trial_number)
    log_path = args.output_dir / "logs" / f"mmpd_tune_{run.dataset}_trial{trial_number:02d}.log"
    cmd = build_mmpd_train_cmd(
        args,
        run,
        hparams=hparams,
        output_root=out_root,
        train_epochs=train_epochs,
        patience=patience,
    )
    run_cmd(
        cmd,
        cwd=args.mmpd_repo,
        env=mmpd_env_for_run(run, args),
        log_path=log_path,
    )
    return _parse_min_vali_loss(log_path)


def tune_mmpd_subset(args: Any, run: AnchorRun) -> Dict[str, Any]:
    dataset = run.dataset
    n_trials = int(args.mmpd_tune_trials)
    if n_trials < 1:
        raise ValueError("mmpd_tune_trials must be >= 1")

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=int(args.seed)),
        study_name=f"mmpd_{dataset}_{args.mmpd_backbone}",
    )

    def objective(trial: optuna.Trial) -> float:
        hparams = _suggest_tune_hparams(trial, args, run)
        print(
            f"[mmpd-tune] {dataset} trial {trial.number}: {hparams}",
            flush=True,
        )
        return run_mmpd_train_trial(
            args,
            run,
            hparams,
            trial_number=trial.number,
            train_epochs=int(args.mmpd_tune_epochs),
            patience=int(args.mmpd_tune_patience),
        )

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=False,
        catch=(RuntimeError, OSError, subprocess.CalledProcessError, ValueError),
    )
    completed = [t for t in study.trials if t.value is not None]
    if len(completed) >= 2:
        vals = [float(t.value) for t in completed]
        if max(vals) - min(vals) < 1e-12:
            raise RuntimeError(
                f"[mmpd-tune] {dataset}: all {len(completed)} completed trials "
                f"returned identical val loss ({vals[0]:.8f}); search is degenerate"
            )
    if study.best_trial is None:
        raise RuntimeError(f"[mmpd-tune] {dataset}: all {n_trials} trials failed")
    best = {**DEFAULT_MMPD_HPARAMS, **study.best_params}
    out_path = tuning_result_path(args.output_dir, dataset)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset,
        "backbone": args.mmpd_backbone,
        "n_trials": n_trials,
        "best_value": float(study.best_value),
        "hparams": best,
        "trials": [
            {
                "number": t.number,
                "value": float(t.value) if t.value is not None else None,
                "params": dict(t.params),
            }
            for t in study.trials
        ],
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(
        f"[mmpd-tune] {dataset}: best val={study.best_value:.6f} -> {out_path}",
        flush=True,
    )
    return best
