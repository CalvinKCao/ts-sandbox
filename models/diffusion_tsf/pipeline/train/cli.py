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
    parser.add_argument("--n-variates", type=int, default=None, help="Override variate count")
    parser.add_argument("--variate-indices", type=str, default=None, help="Comma-separated variate indices")
    parser.add_argument("--subset-id", type=str, default=None, help="Optional subset id label")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--smoke-test", action="store_true", help="Quick validation run")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed from YAML")
    parser.add_argument("--parallel-optuna-workers", type=int, default=1, help="Parallel Optuna workers")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Override checkpoint directory")
    parser.add_argument("--results-dir", type=str, default=None, help="Override results directory")
    parser.add_argument("--datasets-dir", type=str, default=None, help="Benchmark CSV/NPZ root")
    parser.add_argument("--synth-cache-dir", type=str, default=None, help="Shared synthetic pool cache")
    parser.add_argument("--fresh", action="store_true", help="Wipe manifest and checkpoints")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb-project", type=str, default=None, help="Override wandb project from YAML")
    args = parser.parse_args()

    logger = pipeline_mod.setup_logging()

    cli_overrides = {}
    if args.dataset:
        cli_overrides["dataset"] = args.dataset

    nv = args.n_variates
    variate_indices = None
    if args.variate_indices:
        variate_indices = [int(x.strip()) for x in args.variate_indices.split(",") if x.strip()]
        cli_overrides["variate_indices"] = variate_indices
        if not nv:
            nv = len(variate_indices)

    if not nv and args.dataset:
        try:
            nv = pipeline_mod.get_dim_for_dataset(args.dataset)
        except Exception:
            pass
    if nv:
        cli_overrides["n_variates"] = nv

    if args.seed is not None:
        cli_overrides["seed"] = args.seed
    if args.smoke_test:
        cli_overrides["smoke_test"] = True
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

    parallel_workers = 1 if args.smoke_test else max(1, int(args.parallel_optuna_workers))
    cli_overrides["parallel_optuna_workers"] = parallel_workers

    cfg = load_experiment_config(args.config, cli_overrides)
    state = PipelineState.from_config(cfg)
    apply_cli_state_overrides(state, cfg)
    if args.wandb:
        state.wandb_enabled = True
    if args.wandb_project:
        state.wandb_project = args.wandb_project

    configure_diagnostic_logging(bool(logging_settings(cfg).get("diagnostics_enabled", True)))

    if args.checkpoint_dir:
        pipeline_mod.CHECKPOINT_DIR = args.checkpoint_dir
    if args.results_dir:
        pipeline_mod.RESULTS_DIR = args.results_dir
    if args.synth_cache_dir:
        pipeline_mod.SYNTH_CACHE_DIR = args.synth_cache_dir
    if args.datasets_dir:
        pipeline_mod.DATASETS_DIR = os.path.abspath(args.datasets_dir)
    if nv:
        pipeline_mod.N_VARIATES = nv

    subset_meta = pipeline_mod.resolve_pipeline_data_subset(state)
    if subset_meta.get("enabled"):
        logger.info(
            "Data subset resolved: %s -> %s vars, train_stride=%s, test_stride=%s, "
            "raw=%.2f MiB, reduced≈%.2f MiB",
            state.subset_id,
            subset_meta.get("n_variates"),
            subset_meta.get("train_stride"),
            subset_meta.get("test_stride"),
            float(subset_meta.get("raw_size_mb") or 0.0),
            float(subset_meta.get("reduced_size_mb") or 0.0),
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
