"""Checkpoint/results layout: new names by default, legacy dirs still discovered.

New runs write under ``checkpoints_multivariate`` / ``results_multivariate``. If those
do not exist but ``checkpoints_7var`` / ``results_7var`` do, those paths are used
and a warning is logged so old trees work without renaming.
"""
from __future__ import annotations

import logging
import os
from typing import List

logger = logging.getLogger(__name__)

CHECKPOINT_NEW = "checkpoints_multivariate"
CHECKPOINT_LEGACY = "checkpoints_7var"
RESULTS_NEW = "results_multivariate"
RESULTS_LEGACY = "results_7var"


def resolve_checkpoint_dir(script_dir: str) -> str:
    new_p = os.path.join(script_dir, CHECKPOINT_NEW)
    old_p = os.path.join(script_dir, CHECKPOINT_LEGACY)
    if os.path.isdir(new_p):
        return new_p
    if os.path.isdir(old_p):
        logger.warning(
            "Using legacy %s; new runs use %s — rename or migrate when ready.",
            CHECKPOINT_LEGACY,
            CHECKPOINT_NEW,
        )
        return old_p
    return new_p


def resolve_results_dir(script_dir: str) -> str:
    new_p = os.path.join(script_dir, RESULTS_NEW)
    old_p = os.path.join(script_dir, RESULTS_LEGACY)
    if os.path.isdir(new_p):
        return new_p
    if os.path.isdir(old_p):
        logger.warning(
            "Using legacy %s; new runs use %s — rename or migrate when ready.",
            RESULTS_LEGACY,
            RESULTS_NEW,
        )
        return old_p
    return new_p


def checkpoint_roots_ordered(script_dir: str) -> List[str]:
    """Existing checkpoint roots (new first), for discovery across both trees."""
    roots: List[str] = []
    new_p = os.path.join(script_dir, CHECKPOINT_NEW)
    old_p = os.path.join(script_dir, CHECKPOINT_LEGACY)
    if os.path.isdir(new_p):
        roots.append(new_p)
    if os.path.isdir(old_p) and old_p not in roots:
        roots.append(old_p)
    if not roots:
        roots.append(new_p)
    return roots
