"""Pipeline logging helpers."""

from __future__ import annotations

import logging
from typing import Iterable, Optional

# Slurm log friendly: day-hour:minute:second (matches sacct-style timestamps).
PIPELINE_LOG_DATEFMT = "%d-%H:%M:%S"
PIPELINE_LOG_FORMAT = "%(asctime)s %(message)s"


def configure_pipeline_logging(
    level: int = logging.INFO,
    handlers: Optional[Iterable[logging.Handler]] = None,
) -> None:
    root = logging.getLogger()
    formatter = logging.Formatter(PIPELINE_LOG_FORMAT, datefmt=PIPELINE_LOG_DATEFMT)
    if handlers:
        root.handlers.clear()
        for handler in handlers:
            handler.setFormatter(formatter)
            root.addHandler(handler)
    else:
        for handler in root.handlers:
            handler.setFormatter(formatter)
    root.setLevel(level)


def format_trial_params(trial) -> str:
    parts = [f"{k}={v!r}" for k, v in sorted(trial.params.items())]
    return "{" + ", ".join(parts) + "}"
