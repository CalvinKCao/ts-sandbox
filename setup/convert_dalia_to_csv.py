#!/usr/bin/env python3
"""One-time: Forecast100 *.pt -> datasets/dalia/dalia.csv (same layout as other benchmarks)."""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "models", "diffusion_tsf"))

try:
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import torch  # noqa: F401
except ImportError as exc:
    print(
        "Missing dependency for convert_dalia_to_csv.py.\n"
        "On Killarney, run the shell wrapper (picks results/venv from grid jobs):\n"
        f"  {os.path.join(ROOT, 'setup', 'convert_dalia_to_csv.sh')} --pt-dir DALIA\n"
        f"Or: source {os.path.join(ROOT, 'results', 'venv', 'bin', 'activate')}\n"
        f"Original error: {exc}",
        file=sys.stderr,
    )
    sys.exit(1)

from dalia_data import convert_dalia_pt_to_csv, dalia_csv_path, resolve_dalia_dir  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--datasets-dir",
        default=os.path.join(ROOT, "datasets"),
        help="Parent of dalia/ (default: repo datasets/)",
    )
    p.add_argument(
        "--pt-dir",
        default=None,
        help="Folder with Forecast100X.pt / Forecast100Y.pt (repo-relative or absolute)",
    )
    p.add_argument("--x-path", default=None, help="Explicit path to Forecast100X.pt")
    p.add_argument("--y-path", default=None, help="Explicit path to Forecast100Y.pt")
    args = p.parse_args()
    if (args.x_path is None) ^ (args.y_path is None):
        p.error("--x-path and --y-path must be given together")
    out = dalia_csv_path(args.datasets_dir)
    convert_dalia_pt_to_csv(
        out,
        pt_dir=args.pt_dir,
        datasets_dir=args.datasets_dir,
        x_path=args.x_path,
        y_path=args.y_path,
    )
    print(f"Wrote {out}")
    print(f"Data dir: {resolve_dalia_dir(args.datasets_dir)}")
    print("You can remove legacy repo-root DALIA/ or datasets/DALIA/ after verifying.")


if __name__ == "__main__":
    main()
