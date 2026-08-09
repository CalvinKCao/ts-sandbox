#!/usr/bin/env python3
"""Standalone script to run lattice validation assertions on ordinal patch-refine forecasts.

Refactored from submit_binary.sh --ordinal-assert-only mode.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from temp.scripts.eval_univariate_patch_refine_ordinal_vs_mmpd import run_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run dataset lattice validation assertions on ordinal patch-refine forecasts."
    )
    parser.add_argument("--dataset", type=str, required=True, help="Target dataset name (e.g. ETTh1)")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Path to checkpoint root")
    parser.add_argument("--mmpd-root", type=str, required=True, help="Path to MMPD output root")
    parser.add_argument(
        "--binary-config",
        type=str,
        default="configs/binary_patch_refine_lb336_hz96_ordinal_tuned.yaml",
        help="Path to ordinal binary config",
    )
    parser.add_argument("--output-dir", type=str, default="results/assert_output", help="Output directory")
    parser.add_argument("--raw-eval-dir", type=str, default="results/assert_raw", help="Raw eval output directory")
    parser.add_argument("--assert-max-windows", type=int, default=8, help="Max windows to inspect during assertion")
    parser.add_argument("--slice-lengths", type=int, nargs="+", default=[8, 16, 32], help="Slice lengths to check")
    return parser.parse_args()


def main() -> None:
    cli_args = parse_args()
    eval_args = argparse.Namespace(
        datasets=[cli_args.dataset],
        checkpoint_dir=Path(cli_args.checkpoint_dir),
        mmpd_output_root=Path(cli_args.mmpd_root),
        binary_config=Path(cli_args.binary_config),
        output_dir=Path(cli_args.output_dir),
        raw_eval_dir=Path(cli_args.raw_eval_dir),
        pack_test_stride=4,
        test_stride=4,
        test_fraction=1.0,
        disc_index_stride=1,
        raw_binary_batch_size=8,
        slice_lengths=cli_args.slice_lengths,
        force_raw_eval=True,
        force_train=False,
        assert_only=True,
        assert_max_windows=cli_args.assert_max_windows,
        merge_partials_only=False,
        cpu=False,
        gpu=0,
    )
    run_eval(eval_args)
    print(f"[assert_ordinal_patch_refine] Successfully validated lattice for {cli_args.dataset}.")


if __name__ == "__main__":
    main()
