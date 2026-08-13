"""Sequential Optuna hyperparameter study execution for single-worker/single-GPU runs."""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, List, Optional, Sequence

import optuna
from optuna.samplers import BaseSampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import TrialState

logger = logging.getLogger(__name__)


def is_optuna_child_worker() -> bool:
    """True when running inside a spawned Optuna worker process (always False here)."""
    return False


def _journal_storage(checkpoint_dir: str, study_name: str) -> JournalStorage:
    root = os.path.join(checkpoint_dir, "optuna", study_name)
    os.makedirs(root, exist_ok=True)
    journal_path = os.path.join(root, "journal.log")
    return JournalStorage(JournalFileBackend(file_path=journal_path))


def remaining_complete_trials(study: optuna.Study, n_trials: int) -> int:
    """How many new COMPLETE trials are still needed to hit ``n_trials``."""
    n_complete = len(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)))
    return max(0, int(n_trials) - int(n_complete))


def _finished_trial_count(study: optuna.Study) -> int:
    return sum(1 for t in study.trials if t.state.is_finished())


def _fail_stale_running_trials(study: optuna.Study, study_name: str) -> None:
    for trial in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
        logger.warning(
            "Optuna %s: marking stale RUNNING trial %d as FAIL for resume",
            study_name, trial.number,
        )
        try:
            study.tell(trial.number, state=TrialState.FAIL)
        except Exception as e:
            logger.warning(
                "Optuna %s: could not fail stale trial %d: %s",
                study_name, trial.number, e,
            )


def _enqueue_unique_trials(
    study: optuna.Study,
    trials: Optional[Sequence[dict[str, Any]]],
) -> None:
    """Queue configured control trials once, preserving resumable studies."""
    if not trials:
        return
    if not all(isinstance(params, dict) and params for params in trials):
        raise ValueError("enqueue_trials must be a sequence of non-empty parameter mappings")

    existing = [dict(trial.params) for trial in study.get_trials(deepcopy=False)]
    for params in trials:
        candidate = dict(params)
        if candidate in existing:
            continue
        study.enqueue_trial(candidate)
        existing.append(candidate)
        logger.info("Optuna %s: queued control trial %s", study.study_name, candidate)


def _optimize_until_complete(
    study: optuna.Study,
    *,
    study_name: str,
    n_trials: int,
    objective,
    callbacks: Optional[List[Callable]],
    show_progress_bar: bool,
    catch: Sequence[type[BaseException]],
) -> None:
    """Schedule attempts until ``n_trials`` COMPLETE results exist (or attempt cap)."""
    attempt_cap = max(int(n_trials) * 5, int(n_trials))
    while True:
        remaining = remaining_complete_trials(study, n_trials)
        finished = _finished_trial_count(study)
        if remaining == 0:
            return
        if finished >= attempt_cap:
            logger.warning(
                "Optuna %s: stopping with complete=%d/%d after %d finished attempts (cap=%d)",
                study_name,
                n_trials - remaining,
                n_trials,
                finished,
                attempt_cap,
            )
            return
        batch = min(remaining, attempt_cap - finished)
        logger.info(
            "Optuna %s: scheduling %d attempt(s) (complete=%d/%d finished=%d)",
            study_name,
            batch,
            n_trials - remaining,
            n_trials,
            finished,
        )
        study.optimize(
            objective,
            n_trials=batch,
            callbacks=callbacks,
            show_progress_bar=show_progress_bar,
            catch=tuple(catch) if catch else (),
        )


def run_optuna_study(
    *,
    study_name: str,
    checkpoint_dir: str,
    n_trials: int,
    direction: str,
    objective_builder: Callable[[int], Callable[[optuna.Trial], float]],
    sampler: Optional[BaseSampler] = None,
    pruner: Optional[Any] = None,
    callbacks: Optional[List[Callable]] = None,
    show_progress_bar: bool = False,
    catch: Sequence[type[BaseException]] = (),
    sampler_seed: Optional[int] = None,
    parallel_workers: int = 1,
    enqueue_trials: Optional[Sequence[dict[str, Any]]] = None,
) -> optuna.Study:
    """Run Optuna study sequentially on a single GPU worker.

    Always journals under ``{checkpoint_dir}/optuna/{study_name}/journal.log``
    and only schedules enough new trials to reach ``n_trials`` COMPLETE results.
    """
    del parallel_workers  # Retained for call-site compatibility; runs single-worker
    n_trials = max(0, int(n_trials))

    if n_trials == 0:
        raise ValueError("n_trials must be >= 1")

    storage = _journal_storage(checkpoint_dir, study_name)
    study_kwargs: dict = {
        "study_name": study_name,
        "storage": storage,
        "load_if_exists": True,
        "direction": direction,
    }
    if sampler is not None:
        study_kwargs["sampler"] = sampler
    if pruner is not None:
        study_kwargs["pruner"] = pruner
    study = optuna.create_study(**study_kwargs)
    _fail_stale_running_trials(study, study_name)
    _enqueue_unique_trials(study, enqueue_trials)

    remaining = remaining_complete_trials(study, n_trials)
    n_complete = n_trials - remaining
    logger.info(
        "Optuna %s: complete=%d target=%d remaining=%d journal=%s",
        study_name,
        n_complete,
        n_trials,
        remaining,
        os.path.join(checkpoint_dir, "optuna", study_name, "journal.log"),
    )
    if remaining == 0:
        return study

    _optimize_until_complete(
        study,
        study_name=study_name,
        n_trials=n_trials,
        objective=objective_builder(0),
        callbacks=callbacks,
        show_progress_bar=show_progress_bar,
        catch=catch,
    )
    return study
