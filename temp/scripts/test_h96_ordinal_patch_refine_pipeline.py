#!/usr/bin/env python3
"""Fast CPU contracts for the h96 ordinal patch-refine training leaf."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.diffusion_tsf.ordinal_window_norm import (
    build_global_ladder_from_training,
    ordinal_decode,
    ordinal_encode,
)
from models.diffusion_tsf.pipeline.config import load_experiment_config
from models.diffusion_tsf.pipeline.globals_bridge import patch_globals
from models.diffusion_tsf.pipeline.phases.staged_diffusion_finetune_hp import (
    _suggest_staged_params,
)
from models.diffusion_tsf.pipeline.phases.staged_diffusion_pretrain import (
    StagedDiffusionPretrainPhase,
)
from models.diffusion_tsf.pipeline.phases import staged_diffusion_pretrain as pretrain_mod
from models.diffusion_tsf.pipeline.state import PipelineState


class _Trial:
    """Deterministic Optuna-shaped probe for the focused search space."""

    def suggest_float(self, _name: str, lo: float, _hi: float, *, log: bool) -> float:
        assert log
        return float(lo)

    def suggest_categorical(self, _name: str, choices):
        return choices[-1]


def _test_config_and_search() -> None:
    cfg = load_experiment_config(
        str(REPO_ROOT / "configs" / "binary_patch_refine_lb336_hz96_ordinal_tuned.yaml")
    )
    state = PipelineState.from_config(cfg)
    assert state.use_ordinal_window_norm
    assert not state.use_window_normalization
    assert state.ordinal_ood_shift_causal_only
    assert state.use_patch_refine_stage
    assert state.patch_refine_canvas_height == 256
    assert state.patch_refine_patch_height == 32
    assert state.patch_refine_patch_width == 8
    assert state.patch_refine_col_stride == 6
    import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod

    patch_globals(pipeline_mod, state)
    assert pipeline_mod.ORDINAL_OOD_SHIFT_CAUSAL_ONLY is True

    phases = {p["phase"]: p for p in cfg["phases"]}
    pretrain = phases["staged_diffusion_pretrain"]
    assert pretrain["reuse_pretrain_from_config"] == "binary_patch_refine_lb336_hz96_full"
    assert pretrain["require_reuse_pretrain"] is True
    expected_u_grid = {
        "diffusion_coarse_finetune_hp": [512, 1024, 2048],
        # Crop expansion (~17x) forces a smaller ladder than coarse.
        "diffusion_patch_refine_finetune_hp": [64, 128, 256],
    }
    for stage, u_grid in expected_u_grid.items():
        phase = phases[stage]
        assert phase["n_trials"] == 4
        assert phase["search_space"] == "lr_eff_batch_univariate_ema"
        assert phase["effective_univariate_batch_grid"] == u_grid
        assert phase["ema_decay_grid"] == [0.0, 0.99, 0.995, 0.999]

        params = _suggest_staged_params(
            _Trial(), state, max_batch_size=16, smoke_test=False,
            search_space=phase["search_space"], phase_overrides=phase,
        )
        assert params["learning_rate"] == phase["hp_lr_min"]
        assert params["target_univariate_batch"] == u_grid[-1]
        assert params["ema_decay"] == 0.999
        assert params["prediction_target"] == "x0"
        assert params["loss_weighting"] == "min_snr"
        assert params["min_snr_gamma"] == 2.0
        assert params["binary_noise_schedule"] == "linear"

    fallback = load_experiment_config(
        str(REPO_ROOT / "configs" / "binary_patch_refine_lb336_hz96_ordinal_tuned_synth_fallback.yaml")
    )
    fallback_pretrain = next(p for p in fallback["phases"] if p["phase"] == "staged_diffusion_pretrain")
    assert fallback_pretrain["reuse_pretrain_from_config"] == "binary_patch_refine_lb336_hz96_full"
    assert fallback_pretrain["require_reuse_pretrain"] is False
    assert fallback_pretrain["epochs"] == 20
    assert fallback_pretrain["patch_refine_epochs"] == 1

    from_scratch = load_experiment_config(
        str(
            REPO_ROOT
            / "configs"
            / "binary_patch_refine_lb336_hz96_ordinal_tuned_from_scratch.yaml"
        )
    )
    fs_phases = {p["phase"]: p for p in from_scratch["phases"]}
    assert "staged_diffusion_pretrain" not in fs_phases
    assert from_scratch["experiment"]["experiment_name"].endswith("from_scratch")
    assert int(from_scratch["training"]["n_finetune_hp_trials"]) == 5
    fs_eval = fs_phases["staged_eval"]
    assert fs_eval["gmm_components"] == 10
    assert fs_eval["topk_max"] == 3
    assert fs_eval["probabilistic_sampler"] == "quad_t"
    assert fs_eval["tune_sampler"] is False
    for stage, u_grid in expected_u_grid.items():
        phase = fs_phases[stage]
        assert phase["from_random_init"] is True
        assert phase["n_trials"] == 5
        assert phase["search_space"] == "lr_eff_batch_univariate_ema"
        assert float(phase["hp_lr_max"]) == 1.5e-2
        assert phase["effective_univariate_batch_grid"] == u_grid
        assert phase["ema_decay_grid"] == [0.99]
        params = _suggest_staged_params(
            _Trial(), state, max_batch_size=16, smoke_test=False,
            search_space=phase["search_space"], phase_overrides=phase,
        )
        assert params["learning_rate"] == phase["hp_lr_min"]
        assert params["target_univariate_batch"] == u_grid[-1]
        assert params["ema_decay"] == 0.99


def _test_causal_ordinal_shift() -> None:
    train = np.linspace(-1.0, 1.0, 257, dtype=np.float32).reshape(-1, 1)
    ladder = build_global_ladder_from_training(train, tie_atol=1e-7)
    past = torch.full((1, 1, 12), 3.0)
    future_a = torch.full((1, 1, 8), 2.0)
    future_b = torch.full((1, 1, 8), 100.0)

    past_a, future_rank_a, ladder_a, shift_a = ordinal_encode(
        past, future_a, ladder=ladder, apply_ood_shift=True, causal_only=True,
    )
    past_b, future_rank_b, ladder_b, shift_b = ordinal_encode(
        past, future_b, ladder=ladder, apply_ood_shift=True, causal_only=True,
    )
    assert torch.equal(past_a, past_b)
    assert torch.equal(shift_a, shift_b)

    # A generated rank decoded under either true future's stats must stay in the
    # same dataset-normalized coordinate system because the shift is lookback-only.
    generated_rank = torch.full_like(future_rank_a, 128.0)
    _, decoded_a = ordinal_decode(past_a, generated_rank, ladder_a, ood_shift=shift_a)
    _, decoded_b = ordinal_decode(past_b, generated_rank, ladder_b, ood_shift=shift_b)
    assert torch.equal(decoded_a, decoded_b)
    assert not torch.equal(future_rank_a, future_rank_b)


def _test_required_reuse_fails_closed() -> None:
    cfg = load_experiment_config(
        str(REPO_ROOT / "configs" / "binary_patch_refine_lb336_hz96_ordinal_tuned.yaml")
    )
    state = PipelineState.from_config(cfg)
    state.dataset = "never_present_for_contract_test"
    state.checkpoint_dir = str(REPO_ROOT / "_missing_contract_ckpts")
    phase_cfg = next(p for p in cfg["phases"] if p["phase"] == "staged_diffusion_pretrain")
    phase = StagedDiffusionPretrainPhase(**phase_cfg)
    original = pretrain_mod.source_run_stage_pretrain_ckpt
    pretrain_mod.source_run_stage_pretrain_ckpt = lambda *_args, **_kwargs: None
    try:
        try:
            phase.should_skip(state)
        except FileNotFoundError as exc:
            assert "Refusing to train a replacement synthetic pretrain" in str(exc)
        else:
            raise AssertionError("strict synthetic donor contract did not fail closed")
    finally:
        pretrain_mod.source_run_stage_pretrain_ckpt = original


def main() -> None:
    _test_config_and_search()
    _test_causal_ordinal_shift()
    _test_required_reuse_fails_closed()
    print("h96 ordinal patch-refine pipeline contract ok")


if __name__ == "__main__":
    main()
