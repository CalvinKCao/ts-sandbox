"""Tests for checkpoint/results path resolution (new names + legacy fallback)."""
import os
import tempfile

from models.diffusion_tsf.storage_paths import (
    CHECKPOINT_LEGACY,
    CHECKPOINT_NEW,
    RESULTS_LEGACY,
    RESULTS_NEW,
    checkpoint_roots_ordered,
    resolve_checkpoint_dir,
    resolve_results_dir,
)


def test_resolve_checkpoint_prefers_new_when_present():
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, CHECKPOINT_NEW))
        os.makedirs(os.path.join(td, CHECKPOINT_LEGACY))
        assert resolve_checkpoint_dir(td) == os.path.join(td, CHECKPOINT_NEW)


def test_resolve_checkpoint_falls_back_to_legacy():
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, CHECKPOINT_LEGACY))
        assert resolve_checkpoint_dir(td) == os.path.join(td, CHECKPOINT_LEGACY)


def test_resolve_checkpoint_default_new_when_missing():
    with tempfile.TemporaryDirectory() as td:
        assert resolve_checkpoint_dir(td) == os.path.join(td, CHECKPOINT_NEW)


def test_resolve_results_prefers_new_when_present():
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, RESULTS_NEW))
        os.makedirs(os.path.join(td, RESULTS_LEGACY))
        assert resolve_results_dir(td) == os.path.join(td, RESULTS_NEW)


def test_resolve_results_falls_back_to_legacy():
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, RESULTS_LEGACY))
        assert resolve_results_dir(td) == os.path.join(td, RESULTS_LEGACY)


def test_checkpoint_roots_ordered_both():
    with tempfile.TemporaryDirectory() as td:
        new_p = os.path.join(td, CHECKPOINT_NEW)
        old_p = os.path.join(td, CHECKPOINT_LEGACY)
        os.makedirs(new_p)
        os.makedirs(old_p)
        assert checkpoint_roots_ordered(td) == [new_p, old_p]


def test_checkpoint_roots_ordered_neither():
    with tempfile.TemporaryDirectory() as td:
        assert checkpoint_roots_ordered(td) == [os.path.join(td, CHECKPOINT_NEW)]
