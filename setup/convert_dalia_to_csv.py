#!/usr/bin/env python3
"""One-time: Forecast100 *.pt -> datasets/dalia/dalia.csv (same layout as other benchmarks)."""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "models", "diffusion_tsf"))

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
        help="Folder with Forecast100X.pt / Forecast100Y.pt (default: search dalia + legacy DALIA/)",
    )
    args = p.parse_args()
    out = dalia_csv_path(args.datasets_dir)
    convert_dalia_pt_to_csv(out, pt_dir=args.pt_dir, datasets_dir=args.datasets_dir)
    print(f"Wrote {out}")
    print(f"Data dir: {resolve_dalia_dir(args.datasets_dir)}")
    print("You can remove legacy repo-root DALIA/ or datasets/DALIA/ after verifying.")


if __name__ == "__main__":
    main()
