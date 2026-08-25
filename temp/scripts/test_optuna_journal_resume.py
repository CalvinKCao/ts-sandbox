#!/usr/bin/env python3
"""CPU smoke: Optuna journals; n_trials is a finished-attempt budget."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from optuna.exceptions import TrialPruned
from optuna.trial import TrialState

from models.diffusion_tsf.pipeline.optuna_parallel import (
    remaining_trial_attempts,
    run_optuna_study,
)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        calls = {"n": 0}

        def builder(_worker_id: int):
            def objective(trial):
                calls["n"] += 1
                x = trial.suggest_float("x", 0.0, 1.0)
                if trial.number == 0:
                    raise TrialPruned()
                return float(x)

            return objective

        # Budget 3 attempts: 1 PRUNED + 2 COMPLETE, then stop (pruned counts).
        study = run_optuna_study(
            study_name="resume-smoke",
            checkpoint_dir=tmp,
            n_trials=3,
            parallel_workers=1,
            direction="minimize",
            objective_builder=builder,
        )
        assert remaining_trial_attempts(study, 3) == 0
        finished = sum(1 for t in study.trials if t.state.is_finished())
        assert finished == 3
        assert len(study.get_trials(states=(TrialState.PRUNED,))) == 1
        assert len(study.get_trials(states=(TrialState.COMPLETE,))) == 2
        first_calls = calls["n"]
        assert first_calls == 3

        # Second pass must not schedule more once the attempt budget is met.
        study2 = run_optuna_study(
            study_name="resume-smoke",
            checkpoint_dir=tmp,
            n_trials=3,
            parallel_workers=1,
            direction="minimize",
            objective_builder=builder,
        )
        assert calls["n"] == first_calls
        assert remaining_trial_attempts(study2, 3) == 0
        assert study2.best_trial is not None

    print("optuna attempt-budget resume smoke ok")


if __name__ == "__main__":
    main()
