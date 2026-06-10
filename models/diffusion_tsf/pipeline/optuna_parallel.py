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
    """Run Optuna study sequentially or across multiple GPU worker processes."""
    parallel_workers = max(1, int(parallel_workers))
    n_trials = max(0, int(n_trials))

    if n_trials == 0:
        raise ValueError("n_trials must be >= 1")

    if parallel_workers == 1:
        study_kwargs: dict = {"direction": direction}
        if sampler is not None:
            study_kwargs["sampler"] = sampler
        if pruner is not None:
            study_kwargs["pruner"] = pruner
        study = optuna.create_study(**study_kwargs)
        study.optimize(
            objective_builder(0),
            n_trials=n_trials,
            callbacks=callbacks,
            show_progress_bar=show_progress_bar,
            catch=tuple(catch) if catch else (),
        )
        return study

    import torch

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpus < parallel_workers:
        raise RuntimeError(
            f"parallel_optuna_workers={parallel_workers} but only {n_gpus} GPU(s) visible"
        )

    storage = _journal_storage(checkpoint_dir, study_name)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
    )

    ctx = mp.get_context("spawn")
    processes = []
    for worker_id in range(1, parallel_workers):
        trials = _trials_for_worker(n_trials, parallel_workers, worker_id)
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
            ),
        )
        proc.start()
        processes.append(proc)

    parent_trials = _trials_for_worker(n_trials, parallel_workers, 0)
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
