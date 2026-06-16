"""Optuna hyperparameter search for MMPD subset training."""

from __future__ import annotations

import json
import re
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
        env=mmpd_env_for_run(run),
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
        hparams: Dict[str, Any] = {
            "learning_rate": trial.suggest_float(
                "learning_rate", 3e-5, 3e-4, log=True
            ),
            "point_weight": trial.suggest_float(
                "point_weight", 0.005, 0.05, log=True
            ),
            "dropout": trial.suggest_float("dropout", 0.05, 0.35),
        }
        if args.mmpd_backbone == "MaskAE":
            hparams["finetune_layers"] = trial.suggest_int("finetune_layers", 1, 3)
            hparams["neighbor_num"] = trial.suggest_int("neighbor_num", 5, 15)
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

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
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
