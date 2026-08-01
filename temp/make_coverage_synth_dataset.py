#!/usr/bin/env python3
"""Write datasets/coverage_synth/coverage_synth.csv (gitignored) if missing.

Stdlib only so the Killarney login-node preflight can run without a venv.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "datasets" / "coverage_synth" / "coverage_synth.csv"


def write_coverage_synth(path: Path, n: int = 5000, seed: int = 42) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    start = datetime(2000, 1, 1)
    x0 = 0.0
    x1 = 0.0
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "v0", "v1"])
        for t in range(n):
            x0 += rng.gauss(0.0, 0.3)
            x1 += 0.1 * rng.gauss(0.0, 0.2)
            v1 = math.sin(t / 37.0) + x1
            stamp = start + timedelta(hours=t)
            w.writerow([stamp.strftime("%Y-%m-%d %H:%M:%S"), f"{x0:.8f}", f"{v1:.8f}"])
    return path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    out = args.output.expanduser().resolve()
    if out.is_file() and not args.force:
        print(f"exists: {out}")
        return 0
    write_coverage_synth(out, n=args.n, seed=args.seed)
    print(f"wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
