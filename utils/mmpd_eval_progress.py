"""Progress logging for long-running MMPD / anchor eval loops."""

from __future__ import annotations

import time
from typing import Optional


def fmt_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


class EvalProgress:
    """Throttled batch progress with percent, elapsed, and ETA."""

    def __init__(
        self,
        tag: str,
        total: int,
        *,
        item_label: str = "batch",
        log_every: Optional[int] = None,
        min_interval_s: float = 30.0,
    ):
        self.tag = tag
        self.total = max(1, int(total))
        self.item_label = item_label
        self.log_every = log_every or max(1, self.total // 40)
        self.min_interval_s = min_interval_s
        self.start = time.time()
        self.last_log = 0.0

    def maybe_log(self, step: int, extra: str = "") -> None:
        step = max(1, min(step, self.total))
        now = time.time()
        due_step = step == 1 or step >= self.total or (step % self.log_every == 0)
        due_time = (now - self.last_log) >= self.min_interval_s
        if not (due_step or due_time):
            return

        elapsed = now - self.start
        rate = step / elapsed if elapsed > 0 else 0.0
        remaining = (self.total - step) / rate if rate > 0 else 0.0
        pct = 100.0 * step / self.total
        msg = (
            f"[{self.tag}] {self.item_label} {step}/{self.total} ({pct:.1f}%) "
            f"elapsed={fmt_duration(elapsed)} eta={fmt_duration(remaining)}"
        )
        if extra:
            msg = f"{msg} | {extra}"
        print(msg, flush=True)
        self.last_log = now

    def done(self, extra: str = "") -> None:
        elapsed = time.time() - self.start
        msg = f"[{self.tag}] finished {self.total} {self.item_label}s in {fmt_duration(elapsed)}"
        if extra:
            msg = f"{msg} | {extra}"
        print(msg, flush=True)
