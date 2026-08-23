"""Gated wall-clock timers for staged eval / generate.

Off unless ``TS_EVAL_BENCH=1`` or ``configure(True)``. CUDA runs synchronize
at span boundaries so numbers include GPU work plus CPU (layout, ``.item()``).
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch

_ENABLED: ContextVar[bool] = ContextVar("eval_bench_enabled", default=False)
_SESSION: ContextVar[Optional["BenchSession"]] = ContextVar("eval_bench_session", default=None)

_TRUTHY = {"1", "true", "yes", "on"}


def _env_on() -> bool:
    return os.environ.get("TS_EVAL_BENCH", "").strip().lower() in _TRUTHY


def configure(flag: bool = False) -> bool:
    on = bool(flag) or _env_on()
    _ENABLED.set(on)
    if on and _SESSION.get() is None:
        _SESSION.set(BenchSession())
    return on


def enabled() -> bool:
    return _ENABLED.get() or _env_on()


def reset() -> None:
    _SESSION.set(BenchSession() if enabled() else None)


def sync() -> None:
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()


def _session() -> Optional["BenchSession"]:
    sess = _SESSION.get()
    if sess is None and enabled():
        sess = BenchSession()
        _SESSION.set(sess)
    return sess


@dataclass
class _Span:
    name: str
    t0: float
    dt: float = 0.0
    children: List["_Span"] = field(default_factory=list)


@dataclass
class BenchSession:
    stack: List[_Span] = field(default_factory=list)
    roots: List[_Span] = field(default_factory=list)
    repeats: Dict[str, List[float]] = field(default_factory=dict)
    notes: Dict[str, Any] = field(default_factory=dict)


def note(key: str, value: Any) -> None:
    sess = _session()
    if sess is None:
        return
    sess.notes[key] = value


def repeat(name: str, dt: float) -> None:
    sess = _session()
    if sess is None:
        return
    sess.repeats.setdefault(name, []).append(float(dt))


@contextmanager
def span(name: str) -> Iterator[None]:
    if not enabled():
        yield
        return
    sess = _session()
    if sess is None:
        yield
        return
    sync()
    node = _Span(name=name, t0=time.perf_counter())
    parent = sess.stack[-1] if sess.stack else None
    if parent is None:
        sess.roots.append(node)
    else:
        parent.children.append(node)
    sess.stack.append(node)
    try:
        yield
    finally:
        sync()
        node.dt = time.perf_counter() - node.t0
        sess.stack.pop()


def _pct(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = min(len(s) - 1, max(0, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def _merge_children(children: List[_Span]) -> List[Tuple[str, float, int, List[_Span]]]:
    """Collapse same-named siblings into (name, sum_dt, n, nested_children)."""
    order: List[str] = []
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    nested: Dict[str, List[_Span]] = {}
    for ch in children:
        if ch.name not in sums:
            order.append(ch.name)
            sums[ch.name] = 0.0
            counts[ch.name] = 0
            nested[ch.name] = []
        sums[ch.name] += ch.dt
        counts[ch.name] += 1
        nested[ch.name].extend(ch.children)
    return [(n, sums[n], counts[n], nested[n]) for n in order]


def _fmt_ms(sec: float) -> str:
    if sec >= 10:
        return f"{sec:8.2f}s"
    if sec >= 1:
        return f"{sec:8.3f}s"
    return f"{sec * 1000:7.1f}ms"


def _walk(
    children: List[_Span],
    total: float,
    indent: int,
    lines: List[str],
) -> None:
    for name, dt, n, nested in _merge_children(children):
        pct = (100.0 * dt / total) if total > 0 else 0.0
        extra = f"  n={n} mean={_fmt_ms(dt / n).strip()}" if n > 1 else ""
        lines.append(
            f"{'  ' * indent}{name:<28} {_fmt_ms(dt)}  {pct:5.1f}%{extra}"
        )
        if nested:
            _walk(nested, total, indent + 1, lines)


def snapshot() -> Dict[str, Any]:
    sess = _session()
    if sess is None:
        return {"total": 0.0, "spans": [], "repeats": {}, "notes": {}}
    total = sum(r.dt for r in sess.roots)
    repeats = {}
    for name, xs in sess.repeats.items():
        repeats[name] = {
            "n": len(xs),
            "sum": float(sum(xs)),
            "mean": float(sum(xs) / len(xs)),
            "p50": _pct(xs, 50),
            "p95": _pct(xs, 95),
            "min": float(min(xs)),
            "max": float(max(xs)),
        }
    return {
        "total": total,
        "spans": [{"name": r.name, "dt": r.dt} for r in sess.roots],
        "repeats": repeats,
        "notes": dict(sess.notes),
    }


def report(title: str = "") -> str:
    sess = _session()
    if sess is None:
        return "eval-bench off"
    snap = snapshot()
    total = snap["total"]
    head = "eval-bench" if not title else f"eval-bench {title}"
    lines = [f"{head}  total={total:.3f}s"]
    if sess.notes:
        notes = " ".join(f"{k}={v}" for k, v in sess.notes.items())
        lines.append(f"  notes: {notes}")
    _walk(sess.roots, total if total > 0 else 1.0, 1, lines)
    if snap["repeats"]:
        lines.append("  step stats:")
        for name, st in snap["repeats"].items():
            lines.append(
                f"    {name:<26} n={st['n']:<5d} "
                f"mean={_fmt_ms(st['mean']).strip():>10} "
                f"p50={_fmt_ms(st['p50']).strip():>10} "
                f"p95={_fmt_ms(st['p95']).strip():>10} "
                f"sum={_fmt_ms(st['sum']).strip():>10}"
            )
    return "\n".join(lines)


def dump(logger: Optional[logging.Logger] = None, title: str = "") -> str:
    text = report(title)
    log = logger or logging.getLogger(__name__)
    for line in text.splitlines():
        log.info("%s", line)
    return text
