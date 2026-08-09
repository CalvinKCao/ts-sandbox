#!/usr/bin/env python3
"""Standalone script to evaluate fixed-checkpoint non-ordinal patch-refine forecasts against GT.

Refactored from submit_binary.sh --eval-existing-patch-refine mode.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from temp.scripts.eval_univariate_patch_refine_vs_gt import run_eval, parse_args as parse_vs_gt_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate existing fixed-checkpoint patch-refine forecasts against GT."
    )
    parser.add_argument("--dataset", type=str, required=True, help="Target dataset (e.g. ETTh1)")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Path to fine-tuned checkpoint root")
    parser.add_argument("--output-dir", type=str, default="results/existing_patch_refine_eval", help="Output directory")
    parser.add_argument("--test-stride", type=int, default=4, help="Test window stride")
    parser.add_argument("--slice-lengths", type=int, nargs="+", default=[8, 16, 32], help="Slice lengths to evaluate")
    return parser.parse_args()


def main() -> None:
    cli_args = parse_args()
    eval_args = parse_vs_gt_args()
    eval_args.datasets = [cli_args.dataset]
    eval_args.checkpoint_dir = Path(cli_args.checkpoint_dir)
    eval_args.output_dir = Path(cli_args.output_dir)
    eval_args.test_stride = cli_args.test_stride
    eval_args.slice_lengths = cli_args.slice_lengths
    eval_args.force_raw_eval = True
    eval_args.force_train = True

    run_eval(eval_args)
    print(f"[eval_existing_patch_refine] Evaluation complete for {cli_args.dataset}.")


if __name__ == "__main__":
    main()
