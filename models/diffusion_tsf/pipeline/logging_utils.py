"""Pipeline logging helpers."""

from __future__ import annotations

import logging
from typing import Iterable, Optional


def configure_pipeline_logging(
    level: int = logging.INFO,
    handlers: Optional[Iterable[logging.Handler]] = None,
) -> None:
    root = logging.getLogger()
    if handlers:
        root.handlers.clear()
        for handler in handlers:
            root.addHandler(handler)
    root.setLevel(level)


def format_trial_params(trial) -> str:
    parts = [f"{k}={v!r}" for k, v in sorted(trial.params.items())]
    return "{" + ", ".join(parts) + "}"
