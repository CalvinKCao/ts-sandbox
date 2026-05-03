"""CUDA-synced section timers for ConditionalUNet2D.forward.

Enabled only when train_multivariate_pipeline PROFILE.configure turns it on;
keeps log lines aligned with [PROFILE] and the same .epochN gating as TimingProfiler.
"""

from __future__ import annotations

import contextlib
import logging
import re
import time
from datetime import datetime
from typing import Iterator

logger = logging.getLogger(__name__)

_enabled = False
_max_logged_epoch = 1
_log_epoch = 1


def configure(*, enabled: bool, max_logged_epoch: int = 1, log_epoch: int = 1) -> None:
    global _enabled, _max_logged_epoch, _log_epoch
    _enabled = enabled
    _max_logged_epoch = max_logged_epoch
    _log_epoch = log_epoch


def _should_log(name: str) -> bool:
    if not _enabled:
        return False
    match = re.search(r"\.epoch(\d+)(?:\.|:|$)", name)
    if match and int(match.group(1)) > _max_logged_epoch:
        return False
    return True


def _scoped(section: str) -> str:
    return f"unet_forward.epoch{_log_epoch}.{section}"


@contextlib.contextmanager
def section(name: str) -> Iterator[None]:
    full = _scoped(name)
    if not _should_log(full):
        yield
        return
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ts = datetime.now().isoformat(timespec="milliseconds")
    start = time.perf_counter()
    logger.info("[PROFILE] %s | %s:start", ts, full)
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        ts_end = datetime.now().isoformat(timespec="milliseconds")
        logger.info(
            "[PROFILE] %s | %s:end | elapsed_ms=%.3f",
            ts_end,
            full,
            elapsed_ms,
        )
