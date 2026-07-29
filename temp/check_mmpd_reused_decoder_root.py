#!/usr/bin/env python3
"""Validate the reused h720 paper Decoder checkpoints needed for discriminator eval."""

from __future__ import annotations

import argparse
from pathlib import Path


SUBSETS = {
    "ETTh1": "ETTh1",
    "traffic": "traffic_4v_s1",
    "exchange_rate": "exchange_rate",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mmpd_root", type=Path)
    parser.add_argument("--datasets", nargs="+", choices=sorted(SUBSETS), required=True)
    args = parser.parse_args()

    root = args.mmpd_root
    ckpt_root = root / "mmpd_out" / "checkpoints" / "Decoder-MMPD"
    if not ckpt_root.is_dir():
        raise FileNotFoundError(f"missing Decoder checkpoint root: {ckpt_root}")
    for dataset in args.datasets:
        subset = SUBSETS[dataset]
        matches = sorted(ckpt_root.glob(f"data{subset}_il336_ol720_backboneDecoder_*/model_checkpoint.pth"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"{ckpt_root}: expected exactly one h720 Decoder checkpoint for {subset}, found {len(matches)}"
            )
        print(f"[mmpd-reused] {dataset}: {matches[0]}", flush=True)


if __name__ == "__main__":
    main()
