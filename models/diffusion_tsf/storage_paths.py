"""Checkpoint/results directory names under the diffusion_tsf package root."""
from __future__ import annotations

import os

CHECKPOINTS_DIR = "checkpoints_multivariate"
RESULTS_DIR = "results_multivariate"


def resolve_checkpoint_dir(script_dir: str) -> str:
    return os.path.join(script_dir, CHECKPOINTS_DIR)


def resolve_results_dir(script_dir: str) -> str:
    return os.path.join(script_dir, RESULTS_DIR)
