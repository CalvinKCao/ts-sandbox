#!/usr/bin/env python3
"""CPU smoke: single-worker Optuna journals and resumes COMPLETE trials only."""

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
    remaining_complete_trials,
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

        study = run_optuna_study(
            study_name="resume-smoke",
            checkpoint_dir=tmp,
            n_trials=2,
            parallel_workers=1,
            direction="minimize",
            objective_builder=builder,
        )
        assert remaining_complete_trials(study, 2) == 0
        assert len(study.get_trials(states=(TrialState.COMPLETE,))) == 2
        first_calls = calls["n"]
        assert first_calls >= 3  # one prune + two complete

        # Second pass must not schedule more trials once the COMPLETE budget is met.
        study2 = run_optuna_study(
            study_name="resume-smoke",
            checkpoint_dir=tmp,
            n_trials=2,
            parallel_workers=1,
            direction="minimize",
            objective_builder=builder,
        )
        assert calls["n"] == first_calls
        assert remaining_complete_trials(study2, 2) == 0
        assert study2.best_trial is not None

    print("optuna resume journal smoke ok")


if __name__ == "__main__":
    main()
