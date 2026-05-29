"""Wandb helpers for grouped pipeline runs.

Each pipeline execution creates a wandb *group*. Each phase within that
pipeline creates its own wandb *run* inside the group. This gives a clean
dashboard where you can expand a group to see per-phase metrics.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None  # type: ignore[assignment]
    _WANDB_AVAILABLE = False


def _api_key_usable() -> bool:
    key = os.environ.get("WANDB_API_KEY", "").strip()
    return bool(key and re.fullmatch(r"[A-Za-z0-9_]+", key))


def make_group_name(experiment_name: str, dataset: str, seed: int) -> str:
    """Auto-generate a wandb group name."""
    date_str = datetime.now().strftime("%m-%d")
    return f"{experiment_name}-{dataset}-{date_str}-s{seed}"


def init_phase_run(
    phase_name: str,
    group: str,
    project: str,
    job_type: str,
    config: Dict[str, Any],
    tags: Optional[list] = None,
) -> Optional[Any]:
    """Start a new wandb run for one pipeline phase.

    Returns the run object (or None if wandb is unavailable/disabled).
    """
    if not _WANDB_AVAILABLE or not _api_key_usable():
        return None

    run_name = phase_name.replace("_", "-")
    try:
        run = wandb.init(
            project=project,
            group=group,
            job_type=job_type,
            name=run_name,
            config=config,
            reinit=True,
            tags=tags or [],
        )
        logger.info(f"wandb run started: {run.url}")
        return run
    except Exception as e:
        logger.warning(f"Failed to init wandb run for {phase_name}: {e}")
        return None


def finish_phase_run() -> None:
    """Finish the current wandb run (call at end of each phase)."""
    if _WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()


def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None, prefix: Optional[str] = None) -> None:
    """Log metrics to current wandb run."""
    if not _WANDB_AVAILABLE or wandb.run is None:
        return
    if prefix:
        metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
    wandb.log(metrics, step=step)


def log_summary(metrics: Dict[str, Any]) -> None:
    """Set summary metrics on the current run."""
    if not _WANDB_AVAILABLE or wandb.run is None:
        return
    for k, v in metrics.items():
        wandb.run.summary[k] = v
