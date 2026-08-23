#!/usr/bin/env python3
"""CPU smoke: NopPruner wiring and refit_from_pretrain train sources."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import torch
from optuna.pruners import HyperbandPruner, MedianPruner, NopPruner

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.diffusion_tsf.pipeline.phases.staged_diffusion_finetune_hp import (
    PatchRefineDiffusionFinetuneHPPhase,
)
from models.diffusion_tsf.pipeline.state import PipelineState


def _assert_pruners() -> None:
    none_p = PatchRefineDiffusionFinetuneHPPhase(pruner="none")._make_pruner(4)
    assert isinstance(none_p, NopPruner), type(none_p)
    off_p = PatchRefineDiffusionFinetuneHPPhase(pruner="off")._make_pruner(4)
    assert isinstance(off_p, NopPruner), type(off_p)
    med = PatchRefineDiffusionFinetuneHPPhase(
        pruner="median", pruner_n_startup_trials=2,
    )._make_pruner(4)
    assert isinstance(med, MedianPruner), type(med)
    hb = PatchRefineDiffusionFinetuneHPPhase(pruner="hyperband")._make_pruner(4)
    assert isinstance(hb, HyperbandPruner), type(hb)
    try:
        PatchRefineDiffusionFinetuneHPPhase(pruner="banana")._make_pruner(4)
    except ValueError as e:
        assert "banana" in str(e)
    else:
        raise AssertionError("unknown pruner should fail")


def _assert_refit_from_pretrain() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        pretrain = tmp_p / "pretrain.pt"
        pretrain.write_bytes(b"not-a-real-ckpt")
        subset_dir = tmp_p / "subset"
        subset_dir.mkdir()
        final_ckpt = str(subset_dir / "best.pt")

        phase = PatchRefineDiffusionFinetuneHPPhase(
            refit_best_max_epochs=20,
            refit_best_patience=5,
            refit_from_pretrain=True,
        )
        captured = {}

        def fake_train_once(**kwargs):
            captured.update(kwargs)
            return 0.12, 7

        phase._train_once = fake_train_once  # type: ignore[method-assign]
        state = PipelineState(experiment_name="smoke", dataset="ETTh1", smoke_test=False)
        mean = torch.zeros(7)
        std = torch.ones(7)
        params, val, epoch, done = phase._refit_best_if_configured(
            state=state,
            train_ds=MagicMock(),
            val_ds=MagicMock(),
            best_params={"learning_rate": 7.8e-4, "binary_length_g": 1.0},
            hp_best_val_loss=0.4,
            best_trial_num=3,
            diff_ckpt=str(pretrain),
            ft_guidance_ckpt="unused.pt",
            device=torch.device("cpu"),
            variate_indices=list(range(7)),
            final_ckpt=final_ckpt,
            search_space="lr_only",
            search_max_epochs=4,
            search_patience=5,
            subset_dir=str(subset_dir),
            subset_id="ETTh1_allv_randwin",
            subset_meta={"train_stride": 1},
            norm_stats={"mean": mean, "std": std},
        )
        assert done is True
        assert val == 0.12
        assert epoch == 7
        assert captured["pretrained_path"] == str(pretrain)
        assert captured["resume_ckpt"] is None
        assert captured["max_epochs"] == 20
        assert captured["patience"] == 5
        assert params["learning_rate"] == 7.8e-4

        phase_resume = PatchRefineDiffusionFinetuneHPPhase(
            refit_best_max_epochs=20,
            refit_from_pretrain=False,
        )
        captured.clear()
        phase_resume._train_once = fake_train_once  # type: ignore[method-assign]
        phase_resume._refit_best_if_configured(
            state=state,
            train_ds=MagicMock(),
            val_ds=MagicMock(),
            best_params={"learning_rate": 1e-3},
            hp_best_val_loss=0.4,
            best_trial_num=1,
            diff_ckpt=str(pretrain),
            ft_guidance_ckpt="unused.pt",
            device=torch.device("cpu"),
            variate_indices=list(range(7)),
            final_ckpt=final_ckpt,
            search_space="lr_only",
            search_max_epochs=4,
            search_patience=5,
            subset_dir=str(subset_dir),
            subset_id="ETTh1_allv_randwin",
            subset_meta={"train_stride": 1},
            norm_stats={"mean": mean, "std": std},
        )
        assert captured["pretrained_path"] is None
        assert captured["resume_ckpt"] == final_ckpt

        phase_missing = PatchRefineDiffusionFinetuneHPPhase(
            refit_best_max_epochs=20,
            refit_from_pretrain=True,
        )
        try:
            phase_missing._refit_best_if_configured(
                state=state,
                train_ds=MagicMock(),
                val_ds=MagicMock(),
                best_params={"learning_rate": 1e-3},
                hp_best_val_loss=0.4,
                best_trial_num=1,
                diff_ckpt=None,
                ft_guidance_ckpt="unused.pt",
                device=torch.device("cpu"),
                variate_indices=list(range(7)),
                final_ckpt=final_ckpt,
                search_space="lr_only",
                search_max_epochs=4,
                search_patience=5,
                subset_dir=str(subset_dir),
                subset_id="ETTh1_allv_randwin",
                subset_meta={"train_stride": 1},
                norm_stats={"mean": mean, "std": std},
            )
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("missing pretrain should fail")


def main() -> None:
    _assert_pruners()
    _assert_refit_from_pretrain()
    print("finetune hp pruner/refit smoke ok")


if __name__ == "__main__":
    main()
