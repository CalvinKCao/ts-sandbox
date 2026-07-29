"""Multi-GPU parallel Optuna within a single Slurm job."""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
from typing import Any, Callable, List, Optional, Sequence

import optuna
from optuna.samplers import BaseSampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import TrialState

logger = logging.getLogger(__name__)

_worker_env_flag = "OPTUNA_PARALLEL_CHILD"


def _trials_for_worker(n_trials: int, n_workers: int, worker_id: int) -> int:
    base = n_trials // n_workers
    extra = n_trials % n_workers
    return base + (1 if worker_id < extra else 0)


def _journal_storage(checkpoint_dir: str, study_name: str) -> JournalStorage:
    root = os.path.join(checkpoint_dir, "optuna", study_name)
    os.makedirs(root, exist_ok=True)
    journal_path = os.path.join(root, "journal.log")
    return JournalStorage(JournalFileBackend(file_path=journal_path))


def remaining_complete_trials(study: optuna.Study, n_trials: int) -> int:
    """How many new COMPLETE trials are still needed to hit ``n_trials``.

    Pruned/failed attempts do not consume the budget, so an all-OOM first pass
    can still schedule a full target after a config fix on ``--resume``.
    """
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
                "Optuna %s: stopping with complete=%d/%d after %d finished attempts "
                "(cap=%d)",
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


def _run_worker(
    worker_id: int,
    gpu_id: int,
    n_trials: int,
    checkpoint_dir: str,
    study_name: str,
    direction: str,
    sampler_seed: Optional[int],
    objective_builder: Callable[[int], Callable[[optuna.Trial], float]],
    catch: Sequence[type[BaseException]],
    callbacks: Optional[List[Callable]] = None,
) -> None:
    os.environ[_worker_env_flag] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch

    logging.getLogger().setLevel(logging.WARNING)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if n_trials <= 0:
        return

    storage = _journal_storage(checkpoint_dir, study_name)
    sampler: BaseSampler
    if sampler_seed is not None:
        from optuna.samplers import TPESampler
        sampler = TPESampler(seed=int(sampler_seed) + worker_id)
    else:
        from optuna.samplers import TPESampler
        sampler = TPESampler(seed=42 + worker_id)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction=direction,
        sampler=sampler,
    )
    objective = objective_builder(worker_id)
    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=callbacks,
        show_progress_bar=False,
        catch=tuple(catch) if catch else (),
    )
    del study
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_optuna_study(
    *,
    study_name: str,
    checkpoint_dir: str,
    n_trials: int,
    parallel_workers: int,
    direction: str,
    objective_builder: Callable[[int], Callable[[optuna.Trial], float]],
    sampler: Optional[BaseSampler] = None,
    pruner: Optional[Any] = None,
    callbacks: Optional[List[Callable]] = None,
    show_progress_bar: bool = False,
    catch: Sequence[type[BaseException]] = (),
    sampler_seed: Optional[int] = None,
) -> optuna.Study:
    """Run Optuna study sequentially or across multiple GPU worker processes.

    Always journals under ``{checkpoint_dir}/optuna/{study_name}/journal.log``
    and only schedules enough new trials to reach ``n_trials`` COMPLETE results,
    so ``--resume`` into the same checkpoint root continues an interrupted study.
    """
    parallel_workers = max(1, int(parallel_workers))
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

    if parallel_workers == 1:
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

    import torch

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpus < parallel_workers:
        raise RuntimeError(
            f"parallel_optuna_workers={parallel_workers} but only {n_gpus} GPU(s) visible"
        )

    # Parallel workers still split one scheduling wave of ``remaining`` attempts.
    # Follow-up ``--resume`` passes close any COMPLETE shortfall from prunes/OOMs.
    ctx = mp.get_context("spawn")
    processes = []
    for worker_id in range(1, parallel_workers):
        trials = _trials_for_worker(remaining, parallel_workers, worker_id)
        if trials <= 0:
            continue
        proc = ctx.Process(
            target=_run_worker,
            args=(
                worker_id,
                worker_id,
                trials,
                checkpoint_dir,
                study_name,
                direction,
                sampler_seed,
                objective_builder,
                catch,
                callbacks,
            ),
        )
        proc.start()
        processes.append(proc)

    parent_trials = _trials_for_worker(remaining, parallel_workers, 0)
    if parent_trials > 0:
        os.environ.pop(_worker_env_flag, None)
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        study.optimize(
            objective_builder(0),
            n_trials=parent_trials,
            callbacks=callbacks,
            show_progress_bar=show_progress_bar,
            catch=tuple(catch) if catch else (),
        )

    for proc in processes:
        proc.join()
        if proc.exitcode not in (0, None):
            raise RuntimeError(f"Optuna worker exited with code {proc.exitcode}")

    return optuna.load_study(study_name=study_name, storage=storage)


def is_optuna_child_worker() -> bool:
    return os.environ.get(_worker_env_flag) == "1"
