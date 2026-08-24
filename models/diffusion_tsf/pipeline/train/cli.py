"""CLI entry for the diffusion training pipeline."""

from __future__ import annotations

import argparse
import logging
import os
import sys

from models.diffusion_tsf.pipeline.config import (
    apply_cli_state_overrides,
    load_experiment_config,
    logging_settings,
)
from models.diffusion_tsf.pipeline.logging_utils import configure_diagnostic_logging
from models.diffusion_tsf.pipeline.orchestrator import Pipeline
from models.diffusion_tsf.pipeline.phases import PHASE_REGISTRY
from models.diffusion_tsf.pipeline.state import PipelineState
from models.diffusion_tsf.pipeline import wandb_utils


def main():
    # Lazy: this module is imported from train_multivariate_pipeline under __main__.
    from models.diffusion_tsf import train_multivariate_pipeline as pipeline_mod

    parser = argparse.ArgumentParser(description="Diffusion TSF Training Pipeline")
    parser.add_argument("--config", type=str, required=True, help="YAML experiment config")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset from YAML")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--smoke-test", action="store_true", help="Quick validation run")
    parser.add_argument(
        "--eval-bench",
        action="store_true",
        help="Time staged eval generate blocks; skip viz/diagnostics",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap staged eval windows (alias for eval_max_windows)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Cap denoise steps for staged eval (alias for eval_max_steps)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override random seed from YAML")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Override checkpoint directory")
    parser.add_argument("--results-dir", type=str, default=None, help="Override results directory")
    parser.add_argument("--datasets-dir", type=str, default=None, help="Benchmark CSV/NPZ root")
    parser.add_argument("--subset-id", type=str, default=None, help="Override data subset id")
    parser.add_argument("--synth-cache-dir", type=str, default=None, help="Shared synthetic pool cache")
    parser.add_argument("--fresh", action="store_true", help="Wipe manifest and checkpoints")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb-project", type=str, default=None, help="Override wandb project from YAML")
    args = parser.parse_args()

    logger = pipeline_mod.setup_logging()

    cli_overrides = {}
    if args.dataset:
        cli_overrides["dataset"] = args.dataset

    if args.seed is not None:
        cli_overrides["seed"] = args.seed
    if args.smoke_test:
        cli_overrides["smoke_test"] = True
    if args.eval_bench:
        cli_overrides["eval_bench"] = True
    if args.max_samples is not None:
        cli_overrides["eval_max_windows"] = int(args.max_samples)
    if args.max_steps is not None:
        cli_overrides["eval_max_steps"] = int(args.max_steps)
    if args.checkpoint_dir:
        cli_overrides["checkpoint_dir"] = args.checkpoint_dir
    if args.results_dir:
        cli_overrides["results_dir"] = args.results_dir
    if args.datasets_dir:
        cli_overrides["datasets_dir"] = os.path.abspath(args.datasets_dir)
    if args.synth_cache_dir:
        cli_overrides["synth_cache_dir"] = args.synth_cache_dir
    if args.fresh:
        cli_overrides["fresh"] = True
    if args.resume:
        cli_overrides["resume"] = True
    if args.subset_id:
        cli_overrides["subset_id"] = args.subset_id

    cfg = load_experiment_config(args.config, cli_overrides)
    state = PipelineState.from_config(cfg)
    apply_cli_state_overrides(state, cfg)
    if args.wandb:
        state.wandb_enabled = True
    if args.wandb_project:
        state.wandb_project = args.wandb_project

    configure_diagnostic_logging(bool(logging_settings(cfg).get("diagnostics_enabled", True)))

    subset_meta = pipeline_mod.resolve_pipeline_data_subset(state)
    logger.info(
        "Data subset resolved from YAML: %s -> %s vars, train_stride=%s, val_stride=%s, test_stride=%s",
        state.subset_id,
        subset_meta.get("n_variates"),
        subset_meta.get("train_stride"),
        subset_meta.get("val_stride"),
        subset_meta.get("test_stride"),
    )

    from models.diffusion_tsf.pipeline.config import visualization_settings
    from models.diffusion_tsf.pipeline.mmpd_viz_preflight import (
        REPO_ROOT,
        validate_mmpd_viz_requirements,
    )

    validate_mmpd_viz_requirements(
        visualization_settings(cfg),
        state.dataset,
        repo_root=REPO_ROOT,
    )

    phases = []
    for p in cfg["phases"]:
        p_class = PHASE_REGISTRY.get(p["phase"])
        if not p_class:
            logger.error("Unknown phase: %s", p["phase"])
            sys.exit(1)
        phases.append(p_class(**p))

    try:
        Pipeline(phases, state, merged_config=cfg).run()
    finally:
        if state.wandb_enabled:
            wandb_utils.finish_pipeline_run()


if __name__ == "__main__":
    main()
