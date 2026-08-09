# Full walkthrough — binary staged diffusion (live path)

Read this top-to-bottom in **runtime order**. Code blocks are real excerpts from the repo (line ranges in the fence header). Sparse walk-through comments live in the source files themselves; this doc stitches the hot path so you can follow a computer through one campaign.

**Live DAG (ordinal leaf):** pretrain reuse → coarse HP → patch_refine HP → staged_eval. Fine is an alternate second stage, not stacked with patch_refine.

**Entrypoints:** `./submit_binary.sh` → `slurm_worker.sh` → `python -m models.diffusion_tsf.train_multivariate_pipeline`. MMPD / disc sit downstream.

Companion: `architecture.md`. Refactor smells collected at the end.

## 1. Login node: submit_binary.sh

Packs `--configs` / `--datasets` into env, then sbatch's the worker. Smoke vs full modes diverge here.

```bash
# submit_binary.sh:1-80
#!/bin/bash
# Login-node submitter for binary / patch-decoder diffusion jobs.
#
# Flow: this script parses args → sbatch slurm_worker.sh → worker runs
#   python -m models.diffusion_tsf.train_multivariate_pipeline
# (except discriminator modes, which call temp/eval_* scripts instead).
#
# Live training leaf: configs/binary_patch_refine_lb336_hz96_ordinal_tuned.yaml
# (ordinal window norm, coarse → patch_refine). Fine is an alternate second
# stage, not stacked with patch_refine.
#
# Each job gets isolated dirs:
#   ./results/ckpts/MM-DD-<jobid>-<dataset>-<config>/
#   ./results/datasets/MM-DD-<jobid>-<dataset>-<config>/
#
# Three modes (mutually exclusive after flags):
#   1) deferred ordinal disc vs MMPD (--eval-ordinal-patch-refine-vs-mmpd)
#   2) fixed-ckpt patch-refine disc (--eval-existing-patch-refine)
#   3) normal train/eval grid (default; --smoke / --resume / --parallel-optuna)
#
# USAGE (login node, repo root / $SCRATCH/ts-sandbox):
#   ./submit_binary.sh --configs binary_patch_refine_lb336_hz96_ordinal_tuned \
#       --datasets ETTh1,traffic --time 10:00:00
#   ./submit_binary.sh --smoke
#   ./submit_binary.sh --resume --configs binary_dual_scale_staged --datasets ETTh1
#
# --configs: comma paths, globs, or bare stems under configs/*.yaml.
# Do NOT add new submit_*.sh wrappers for minor YAML variants — edit a leaf YAML.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIGS=""
DATASETS="ETTh1,ETTh2,ETTm1,ETTm2,illness,exchange_rate,weather,electricity,traffic,PeMS,solar_Alabama"
SEEDS="42"
SMOKE=0
RESUME=0
CKPT_CONFIG=""
DEPENDENCY=""
WANDB_PROJECT=""
WANDB_PROJECT_EXPLICIT=0
WALL_OVERRIDE=""
PARALLEL_OPTUNA=""
EVAL_EXISTING_PATCH_REFINE=0
EXISTING_CKPT_ROOTS=""
DISC_RUN=""
ORDINAL_SLICE_LENGTHS="8;16;32"
SBATCH_EXCLUDE_NODES=""
RAW_RUN=""
JOB_MANIFEST=""
EVAL_ORDINAL_PATCH_REFINE_MMPD=0
MMPD_ROOT=""
ORDINAL_DISC_EVALUATOR="temp/eval_univariate_patch_refine_ordinal_vs_mmpd.py"
ORDINAL_BINARY_CONFIG="configs/binary_patch_refine_lb336_hz96_ordinal_tuned.yaml"
DEFER_CHECKPOINT_CHECK=0
ORDINAL_ASSERT_ONLY=0
if [[ "$(hostname)" == *"narval"* ]]; then
    ACCOUNT="def-boyuwang"
    GPU_TYPE="a100"
else
    ACCOUNT="aip-boyuwang"
    GPU_TYPE="l40s"
fi
while [[ $# -gt 0 ]]; do
    case "$1" in
        --configs|--config) CONFIGS="$2"; shift 2 ;;
        --datasets) DATASETS="$2"; shift 2 ;;
        --n-variates) N_VARIATES="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --smoke|--smoke-test) SMOKE=1; shift ;;
        --resume) RESUME=1; shift ;;
        --ckpt-config) CKPT_CONFIG="$2"; shift 2 ;;
        --dependency) DEPENDENCY="$2"; shift 2 ;;
        --wandb-project) WANDB_PROJECT="$2"; WANDB_PROJECT_EXPLICIT=1; shift 2 ;;
        --time) WALL_OVERRIDE="$2"; shift 2 ;;
        --parallel-optuna) PARALLEL_OPTUNA="$2"; shift 2 ;;
        --eval-existing-patch-refine) EVAL_EXISTING_PATCH_REFINE=1; shift ;;
        --existing-ckpt-roots) EXISTING_CKPT_ROOTS="$2"; shift 2 ;;
        --disc-run) DISC_RUN="$2"; shift 2 ;;
        --raw-run) RAW_RUN="$2"; shift 2 ;;
```

Mode selection and sbatch handoff (skip argparse boilerplate in between):


(sbatch / worker references around lines [4, 48, 89, 100, 183, 189, 199, 212]…)

```bash
# submit_binary.sh:183-238
    [[ -n "${SBATCH_EXCLUDE_NODES:-}" ]] && EXCLUDE_ARGS=(--exclude="$SBATCH_EXCLUDE_NODES")
    DISC_JOB_IDS=()
    for dataset_name in "${EVAL_DATASETS[@]}"; do
        checkpoint_root="${ROOT_BY_DATASET[$dataset_name]}"
        job_label="disc-opr96"
        [[ "$ORDINAL_ASSERT_ONLY" -eq 0 ]] || job_label="assert-opr96"
        job_id=$(sbatch --parsable \
            --job-name="${job_label}-${dataset_name}" \
            --account="$ACCOUNT" --time="$WALL" --nodes=1 "$GPU_ARG" \
            --cpus-per-task=8 --mem=50G \
            "${DEP_ARGS[@]}" \
            "${EXCLUDE_ARGS[@]}" \
            --output="$LOG_DIR/${job_label}-${dataset_name}-%j.log" \
            --error="$LOG_DIR/${job_label}-${dataset_name}-%j.log" \
            --mail-type=FAIL --mail-user="${USER_NAME}@uwo.ca" \
            --export=ALL,GRID_EVAL_ORDINAL_PATCH_REFINE_MMPD=1,GRID_ORDINAL_ASSERT_ONLY="$ORDINAL_ASSERT_ONLY",GRID_DATASET="$dataset_name",GRID_EXISTING_CKPT="$checkpoint_root",GRID_MMPD_ROOT="$MMPD_ROOT",GRID_DISC_OUTPUT="$DISC_OUTPUT",GRID_RAW_DISC_OUTPUT="$RAW_OUTPUT",GRID_ORDINAL_DISC_EVALUATOR="$ORDINAL_DISC_EVALUATOR",GRID_ORDINAL_BINARY_CONFIG="$ORDINAL_BINARY_CONFIG",GRID_SLICE_LENGTHS="$ORDINAL_SLICE_LENGTHS" \
            "$SCRIPT_DIR/slurm_worker.sh")
        manifest_role="ordinal_disc"
        [[ "$ORDINAL_ASSERT_ONLY" -eq 0 ]] || manifest_role="ordinal_assert"
        manifest_tool record --path "$MANIFEST_PATH" --role "$manifest_role" --dataset "$dataset_name" --job-id "$job_id" \
            --set "checkpoint_root=$checkpoint_root"
        DISC_JOB_IDS+=("$job_id")
    done
    if [[ "$ORDINAL_ASSERT_ONLY" -eq 1 ]]; then
        echo "ordinal patch-refine assertion manifest: $MANIFEST_PATH"
        exit 0
    fi
    disc_dep="afterok:${DISC_JOB_IDS[0]}"
    for job_id in "${DISC_JOB_IDS[@]:1}"; do disc_dep+=":$job_id"; done
    merge_id=$(sbatch --parsable \
        --job-name="disc-opr96-merge" --account="$ACCOUNT" --nodes=1 \
        --cpus-per-task=2 --mem=8G --time=0:30:00 --dependency="$disc_dep" \
        --output="$LOG_DIR/disc-opr96-merge-%j.log" --error="$LOG_DIR/disc-opr96-merge-%j.log" \
        --mail-type=FAIL --mail-user="${USER_NAME}@uwo.ca" \
        --export=ALL,GRID_EVAL_ORDINAL_PATCH_REFINE_MMPD=1,GRID_ORDINAL_DISC_MERGE=1,GRID_DISC_OUTPUT="$DISC_OUTPUT",GRID_RAW_DISC_OUTPUT="$RAW_OUTPUT",GRID_ORDINAL_DISC_EVALUATOR="$ORDINAL_DISC_EVALUATOR",GRID_ORDINAL_BINARY_CONFIG="$ORDINAL_BINARY_CONFIG" \
        "$SCRIPT_DIR/slurm_worker.sh")
    manifest_tool record --path "$MANIFEST_PATH" --role ordinal_disc_merge --job-id "$merge_id"
    echo "ordinal patch-refine discriminator manifest: $MANIFEST_PATH"
    exit 0
fi

# Mode 2: fixed-ckpt h96 disc. Skips training; only fits the discriminator.
# Login node checks coarse/patch_refine best.pt before sbatch.
if [[ "$EVAL_EXISTING_PATCH_REFINE" -eq 1 ]]; then
    [[ -n "$EXISTING_CKPT_ROOTS" ]] || {
        echo "ERROR: --eval-existing-patch-refine requires --existing-ckpt-roots dataset=/absolute/or/relative/run,..." >&2
        exit 1
    }
    [[ -z "$CONFIGS" ]] || {
        echo "ERROR: --configs is not used with --eval-existing-patch-refine" >&2
        exit 1
    }
    [[ "$RESUME" -eq 0 && "$SMOKE" -eq 0 && -z "$PARALLEL_OPTUNA" ]] || {
        echo "ERROR: --resume, --smoke, and --parallel-optuna are not valid with --eval-existing-patch-refine" >&2
        exit 1
    }
```

## 2. Compute node: slurm_worker.sh

Activates cluster env, cds to repo, runs the Python module with the YAML leaf.

```bash
# slurm_worker.sh:1-60
#!/bin/bash
# Compute-node worker for binary diffusion jobs. Submitted by submit_binary.sh
# (do not sbatch this by hand unless debugging).
#
# Modes (via GRID_* env from the submitter):
#   GRID_EVAL_ORDINAL_PATCH_REFINE_MMPD=1 → ordinal disc / assert / merge
#   GRID_EVAL_PATCH_REFINE=1             → fixed-ckpt h96 disc vs GT
#   else                                 → train_multivariate_pipeline (main path)
#
# Always rebuilds a fresh venv on $SLURM_TMPDIR from
# setup/requirements-killarney.txt (wheelhouse / no network on compute).

set -euo pipefail

PY_ARGS=("$@")
ORDINAL_DISC_MODE="${GRID_EVAL_ORDINAL_PATCH_REFINE_MMPD:-0}"
ORDINAL_DISC_MERGE="${GRID_ORDINAL_DISC_MERGE:-0}"
ORDINAL_ASSERT_ONLY="${GRID_ORDINAL_ASSERT_ONLY:-0}"

ts() { date +'%d-%H:%M:%S'; }

echo "$(ts) =========================================="
echo "$(ts) Job: $SLURM_JOB_NAME  ID: $SLURM_JOB_ID  Node: ${SLURMD_NODENAME:-unknown}"
echo "$(ts) =========================================="

STORE="${GRID_STORE:-$SCRATCH/ts-sandbox/results}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
REQ="$REPO/setup/requirements-killarney.txt"

# Fail fast before the slow venv build if the requested YAML is missing on this checkout.
CONFIG_REL=""
for ((i = 1; i < $#; i++)); do
    if [[ "${!i}" == "--config" ]]; then
        j=$((i + 1))
        CONFIG_REL="${!j}"
        break
    fi
done
if [[ -n "$CONFIG_REL" ]]; then
    CONFIG_PATH="$REPO/$CONFIG_REL"
    if [[ ! -f "$CONFIG_PATH" ]]; then
        BRANCH="$(git -C "$REPO" branch --show-current 2>/dev/null || echo unknown)"
        echo "ERROR: config not found: $CONFIG_PATH" >&2
        # Hard-coded branch hint — often stale; treat as a nudge, not truth.
        echo "ERROR: repo branch=$BRANCH — git checkout feat/patch-decoder-cross-variate-ctx && git pull" >&2
        exit 1
    fi
fi

[[ -f "$REQ" ]] || { echo "ERROR: missing $REQ — run ./setup/killarney_freeze_requirements.sh on login node" >&2; exit 1; }
[[ -n "${SLURM_TMPDIR:-}" ]] || { echo "ERROR: SLURM_TMPDIR is not set." >&2; exit 1; }

module purge 2>/dev/null || true
module load StdEnv/2023 python/3.11 2>/dev/null || true
if [[ "$ORDINAL_DISC_MERGE" -ne 1 ]]; then
    module load cuda/12.2 cudnn/8.9 2>/dev/null || true
fi
command -v virtualenv >/dev/null || { echo "ERROR: virtualenv not available after module load." >&2; exit 1; }

echo "$(ts) [setup] Building node-local venv on \$SLURM_TMPDIR from $REQ"
```
```bash
# slurm_worker.sh:1-32
#!/bin/bash
# Compute-node worker for binary diffusion jobs. Submitted by submit_binary.sh
# (do not sbatch this by hand unless debugging).
#
# Modes (via GRID_* env from the submitter):
#   GRID_EVAL_ORDINAL_PATCH_REFINE_MMPD=1 → ordinal disc / assert / merge
#   GRID_EVAL_PATCH_REFINE=1             → fixed-ckpt h96 disc vs GT
#   else                                 → train_multivariate_pipeline (main path)
#
# Always rebuilds a fresh venv on $SLURM_TMPDIR from
# setup/requirements-killarney.txt (wheelhouse / no network on compute).

set -euo pipefail

PY_ARGS=("$@")
ORDINAL_DISC_MODE="${GRID_EVAL_ORDINAL_PATCH_REFINE_MMPD:-0}"
ORDINAL_DISC_MERGE="${GRID_ORDINAL_DISC_MERGE:-0}"
ORDINAL_ASSERT_ONLY="${GRID_ORDINAL_ASSERT_ONLY:-0}"

ts() { date +'%d-%H:%M:%S'; }

echo "$(ts) =========================================="
echo "$(ts) Job: $SLURM_JOB_NAME  ID: $SLURM_JOB_ID  Node: ${SLURMD_NODENAME:-unknown}"
echo "$(ts) =========================================="

STORE="${GRID_STORE:-$SCRATCH/ts-sandbox/results}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
REQ="$REPO/setup/requirements-killarney.txt"

# Fail fast before the slow venv build if the requested YAML is missing on this checkout.
CONFIG_REL=""
for ((i = 1; i < $#; i++)); do
```

## 3. Module entry → CLI

`python -m …train_multivariate_pipeline` hits `__main__`, which delegates to `pipeline.train.cli.main`.

```python
# models/diffusion_tsf/train_multivariate_pipeline.py:1-80

"""Legacy training library + CLI entry for the binary staged diffusion pipeline.

Live path (cluster): ./submit_binary.sh → slurm_worker.sh → this module as
``python -m models.diffusion_tsf.train_multivariate_pipeline``, which hands off
to ``pipeline.train.cli.main``. CLI builds PipelineState, registers phases from
YAML, and runs ``pipeline.orchestrator.Pipeline``.

This file still holds most of the training implementation that phases call into
(model factory, dataloaders, Optuna objectives, guidance helpers). Module-level
globals below are placeholders; ``pipeline.globals_bridge.patch_globals``
overwrites them from YAML before each phase runs.

Typical leaf phases: staged pretrain → coarse HP → patch_refine HP → staged_eval.
Fine and patch_refine are alternative second stages, not stacked.

Usage:
    python -m models.diffusion_tsf.train_multivariate_pipeline --config configs/foo.yaml --dataset ETTh1
    python -m models.diffusion_tsf.train_multivariate_pipeline --smoke-test ...
"""

import argparse
import errno
import gc
import importlib.util
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import optuna
from optuna.samplers import TPESampler
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

# Setup path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.diffusion_tsf.config import DiffusionTSFConfig
from models.diffusion_tsf.diffusion_model import DiffusionTSF
from models.diffusion_tsf.realts import get_synthetic_dataloader
from models.diffusion_tsf.guidance import PatchDecoderGuidance
from models.diffusion_tsf.patch_guidance_stack import PatchGuidanceStack, PatchGuidanceStackConfig
from models.diffusion_tsf.ordinal_window_norm import (
    build_global_ladder_from_training,
    ordinal_encode,
    ranks_to_unit,
)
from models.diffusion_tsf.storage_paths import resolve_checkpoint_dir, resolve_results_dir
from models.diffusion_tsf.pipeline.data_subset import resolve_data_subset

DATASETS_DIR = os.path.join(project_root, "datasets")
CHECKPOINT_DIR = resolve_checkpoint_dir(script_dir)
RESULTS_DIR = resolve_results_dir(script_dir)
SYNTH_CACHE_DIR: Optional[str] = None

def is_main_process() -> bool:
    """True on the coordinator process (not an Optuna child worker)."""
    from models.diffusion_tsf.pipeline.optuna_parallel import is_optuna_child_worker
    return not is_optuna_child_worker()


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def unwrap_model(model: nn.Module) -> nn.Module:
    # Leftover from a DDP era — identity today. Call sites still go through it.
    return model
```
```python
# models/diffusion_tsf/train_multivariate_pipeline.py:1490-1495
if __name__ == "__main__":
    # Import here: cli imports this module, so a top-level import would cycle.
    from models.diffusion_tsf.pipeline.train.cli import main

    main()
```

CLI: load YAML → PipelineState → PHASE_REGISTRY → Pipeline.run().

```python
# models/diffusion_tsf/pipeline/train/cli.py:1-153
"""CLI entry for the diffusion training pipeline.

Called from ``train_multivariate_pipeline`` under ``__main__`` (Slurm path:
submit_binary → slurm_worker → ``python -m ...train_multivariate_pipeline``).

Responsibility: parse args → load/merge YAML → build PipelineState → instantiate
phases from PHASE_REGISTRY → ``Pipeline(...).run()``. Actual training lives in
phase classes + helpers still on the train_multivariate_pipeline module.
"""

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
    # Lazy import: cli is loaded from train_multivariate_pipeline under __main__,
    # so a top-level import of that module would cycle.
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
            # Fail-soft: leave nv unset and let YAML / later resolve_data_subset decide.
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

    # Also poke legacy module globals so older helpers that still read them see
    # CLI path overrides before patch_globals runs inside phases.
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

    # YAML phase list → concrete classes. Unknown names fail here, not mid-run.
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
```

## 4. Config merge + phase registry

Leaf YAML `extends` base; `normalize_guidance_phases` keeps fine XOR patch_refine.

```python
# models/diffusion_tsf/pipeline/phases/__init__.py:1-40
"""Phase registry — maps YAML ``phase:`` names to concrete classes.

cli.py looks up each entry in the merged config's ``phases:`` list here.
Live ordinal leaf: staged_diffusion_pretrain → diffusion_coarse_finetune_hp →
diffusion_patch_refine_finetune_hp → staged_eval.

``diffusion_fine_finetune_hp`` remains registered for coarse→fine campaigns;
``normalize_guidance_phases`` drops fine when patch_refine is also listed.
``itrans_finetune_hp`` is a scrubbed alias — same class as patch_guidance, no
iTransformer package.
"""

from models.diffusion_tsf.pipeline.phases.patch_guidance_finetune_hp import PatchGuidanceFinetuneHPPhase
from models.diffusion_tsf.pipeline.phases.staged_diffusion_pretrain import StagedDiffusionPretrainPhase
from models.diffusion_tsf.pipeline.phases.staged_diffusion_finetune_hp import (
    CoarseDiffusionFinetuneHPPhase,
    FineDiffusionFinetuneHPPhase,
    PatchRefineDiffusionFinetuneHPPhase,
)
from models.diffusion_tsf.pipeline.phases.staged_eval import StagedEvalPhase

# itrans_finetune_hp is a scrubbed YAML alias for patch-decoder guidance (no iTransformer).
PHASE_REGISTRY = {
    "patch_guidance_finetune_hp": PatchGuidanceFinetuneHPPhase,
    "itrans_finetune_hp": PatchGuidanceFinetuneHPPhase,
    "staged_diffusion_pretrain": StagedDiffusionPretrainPhase,
    "diffusion_coarse_finetune_hp": CoarseDiffusionFinetuneHPPhase,
    "diffusion_fine_finetune_hp": FineDiffusionFinetuneHPPhase,
    "diffusion_patch_refine_finetune_hp": PatchRefineDiffusionFinetuneHPPhase,
    "staged_eval": StagedEvalPhase,
}

__all__ = [
    "PHASE_REGISTRY",
    "PatchGuidanceFinetuneHPPhase",
    "StagedDiffusionPretrainPhase",
    "CoarseDiffusionFinetuneHPPhase",
    "FineDiffusionFinetuneHPPhase",
    "PatchRefineDiffusionFinetuneHPPhase",
    "StagedEvalPhase",
```
```python
# models/diffusion_tsf/pipeline/config.py:312-401
def normalize_guidance_phases(
    phases: list,
    guidance_type: str,
    *,
    experiment: Optional[Dict[str, Any]] = None,
) -> list:
    """Normalize merged phase lists for guidance / patch-refine variants.

    ``itrans_finetune_hp`` is a scrubbed YAML alias for patch-decoder guidance.
    If ``diffusion_patch_refine_finetune_hp`` is present, fine is dropped —
    fine and patch_refine are alternative second stages, not stacked.
    Removed phases (finer / vertical_dual / channel_dual) fail fast.
    """
    removed = {
        "diffusion_finer_finetune_hp",
        "diffusion_vertical_dual_finetune_hp",
        "diffusion_channel_dual_finetune_hp",
    }
    by_name: Dict[str, Dict[str, Any]] = {}
    for entry in phases:
        name = str(entry["phase"])
        if name in removed:
            raise ValueError(
                f"phase {name!r} was removed; use coarse+fine or coarse+patch_refine"
            )
        # Scrubbed alias: historical iTransformer slot → patch_decoder guidance.
        if name == "itrans_finetune_hp":
            if guidance_type not in ("patch_decoder", "", "none"):
                raise ValueError(
                    "itrans_finetune_hp no longer loads iTransformer weights; "
                    "use patch_guidance_finetune_hp / guidance_type=patch_decoder"
                )
            entry = dict(entry)
            entry["phase"] = "patch_guidance_finetune_hp"
            name = "patch_guidance_finetune_hp"
        by_name[name] = dict(entry)
    if "diffusion_patch_refine_finetune_hp" in by_name:
        by_name.pop("diffusion_fine_finetune_hp", None)
    exp = experiment or {}
    # Match DiffusionTSFConfig / live YAML: guidance channel defaults off; XA may still need tokens.
    needs_guidance = bool(exp.get("use_guidance_channel", False)) or not bool(
        exp.get("disable_cross_attention", False)
    )
    if not needs_guidance:
        by_name.pop("patch_guidance_finetune_hp", None)
    preferred = (
        "staged_diffusion_pretrain",
        "patch_guidance_finetune_hp",
        "diffusion_coarse_finetune_hp",
        "diffusion_fine_finetune_hp",
        "diffusion_patch_refine_finetune_hp",
        "staged_eval",
    )
    ordered = [by_name[n] for n in preferred if n in by_name]
    seen = {str(p["phase"]) for p in ordered}
    for entry in phases:
        name = str(entry["phase"])
        if name == "itrans_finetune_hp":
            name = "patch_guidance_finetune_hp"
        if name in removed:
            continue
        if name not in seen and name in by_name:
            ordered.append(dict(by_name[name]))
            seen.add(name)
    return ordered

def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k == "phases" and _is_phase_list(out.get(k)) and _is_phase_list(v):
            out[k] = _merge_phase_lists(out[k], v)
        elif k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def _resolve_config_path(path: str, relative_to: str) -> str:
    if os.path.isabs(path):
        return os.path.abspath(path)
    for base in (os.path.dirname(relative_to), os.path.join(_REPO_ROOT, "configs"), _REPO_ROOT):
        candidate = os.path.normpath(os.path.join(base, path))
        if os.path.isfile(candidate):
            return candidate
    return os.path.normpath(os.path.join(os.path.dirname(relative_to), path))

def _load_yaml_tree(path: str, seen: Optional[Set[str]] = None) -> Dict[str, Any]:
    """Recursively load YAML with ``extends`` (parents first, leaf wins).

    ``replace_phases: true`` on a leaf swaps the inherited phase list wholesale
```
```python
# models/diffusion_tsf/pipeline/config.py:495-584
def load_experiment_config(
    yaml_path: str,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load experiment config from YAML (with extends), validate, apply CLI overrides.

    After validation, hard-rejects unsupported guidance / noise / stage combos
    so bad leaves fail before any GPU work. Returns the merged dict that
    ``PipelineState.from_config`` and phase constructors consume.
    """
    if not yaml_path:
        raise ValueError("--config is required")

    cfg = _load_yaml_tree(os.path.abspath(yaml_path))
    cfg["_yaml_path"] = os.path.abspath(yaml_path)

    if cli_overrides:
        exp = dict(cfg.get("experiment", {}))
        state_overrides: Dict[str, Any] = {}
        for key, value in cli_overrides.items():
            if key in CLI_EXPERIMENT_KEYS:
                exp[key] = value
            elif key in CLI_STATE_KEYS:
                state_overrides[key] = value
            else:
                raise ValueError(
                    f"unsupported CLI override {key!r}; wandb settings belong in the YAML wandb: section"
                )
        cfg["experiment"] = exp
        if state_overrides:
            cfg["_cli_state_overrides"] = state_overrides

    validate_config(cfg)
    exp = cfg.get("experiment") or {}
    guidance_type = str(exp.get("guidance_type", "patch_decoder"))
    if guidance_type != "patch_decoder":
        raise ValueError(
            f"Only guidance_type='patch_decoder' is supported; got {guidance_type!r}"
        )
    guidance_placement = str(exp.get("guidance_placement", "canvas"))
    if guidance_placement != "canvas":
        raise ValueError(
            f"Only guidance_placement='canvas' is supported; got {guidance_placement!r}"
        )
    binary_noise_schedule = str(exp.get("binary_noise_schedule", "linear"))
    if binary_noise_schedule != "linear":
        raise ValueError(
            f"Only binary_noise_schedule='linear' is supported; got {binary_noise_schedule!r}"
        )
    anchor_mode = str(exp.get("binary_anchor_input_mode", "stationary_flat"))
    if anchor_mode != "stationary_flat":
        raise ValueError(
            f"Only binary_anchor_input_mode='stationary_flat' is supported; got {anchor_mode!r}"
        )
    diffusion_stage = str(exp.get("diffusion_stage", "joint"))
    if diffusion_stage not in {"joint", "coarse", "fine", "patch_refine"}:
        raise ValueError(
            "diffusion_stage must be one of {'joint', 'coarse', 'fine', 'patch_refine'}, "
            f"got {diffusion_stage!r}."
        )
    cfg["phases"] = normalize_guidance_phases(
        cfg["phases"],
        guidance_type,
        experiment=cfg.get("experiment") or {},
    )
    return cfg

def apply_cli_state_overrides(state: Any, cfg: Dict[str, Any]) -> None:
    """Apply runtime CLI flags onto PipelineState after YAML load."""
    from dataclasses import fields

    overrides = cfg.get("_cli_state_overrides") or {}
    known_fields = {f.name for f in fields(state)}
    for key, value in overrides.items():
        if key not in known_fields:
            raise ValueError(f"unsupported CLI state override: {key!r}")
        setattr(state, key, value)

def apply_training_config_to_module(mod: Any, cfg: Optional[Dict[str, Any]], state: Any = None) -> None:
    """Push training section from merged YAML onto pipeline module globals.

    Called from ``globals_bridge.patch_globals``. ``state`` is currently unused
    (kept for call-site compatibility).
    """
    training = (cfg or {}).get("training")
    if not isinstance(training, dict):
        raise ValueError("merged config missing training section")
    for yaml_key, attr in _TRAINING_GLOBAL_MAP.items():
        if yaml_key not in training:
            raise KeyError(f"training.{yaml_key} required")
```

Base vs live leaf (phases graph):

```yaml
# configs/base/binary_staged.yaml:1-80
# Shared defaults for staged binary DiT pipeline runs.
# Leaf configs: `extends: base/binary_staged.yaml` (or a fuller leaf) then
# override name / arch / phases. Loaded by pipeline.config.load_experiment_config.
#
# Default phase graph below is coarse → fine (generic dual-scale). Live ordinal
# patch-refine leaves replace fine with diffusion_patch_refine_finetune_hp —
# fine and patch_refine are alternatives, not stacked.

training:
  pretrain_epochs: 10
  pretrain_diffusion_epochs: 20
  pretrain_diffusion_max_epochs: 20
  force_retrain_synthetic: true
  skip_synthetic_tuning: true
  use_hardcoded_synthetic_hp: true
  pretrain_synthetic_override: null
  synthetic_samples_full_cap: 50000
  synthetic_samples_hp_tune: 20000
  synthetic_samples_diff_tune: 10000
  synthetic_samples_min: 4096
  n_diffusion_hp_trials: 8
  n_finetune_hp_trials: 5
  diffusion_hp_patience: 4
  hp_tune_epochs: 20
  hp_tune_patience: 15
  diffusion_batch_size: 32
  diffusion_batch_sizes: [16]
  finetune_batch_sizes: [4, 8, 16]
  finetune_max_micro_batch: 2048
  finetune_hp_lr_min: 0.000003
  finetune_hp_lr_max: 0.0002
  use_amp: true
  use_gradient_checkpointing: true
  unet_max_chunk_size: 128
  eval_num_samples: 30
  diffusion_ema_decay: 0.99

logging:
  diagnostics_enabled: true

visualization:
  enabled: true
  n_samples: 3
  n_dual_scale_vars: 3
  jpeg_dpi: 100
  dual_scale_sampler: anchor
  dual_scale_inference_steps: 20

wandb:
  enabled: true
  project: ts-sandbox-leaderboard
  group: null
  tags: []

experiment:
  dataset: ETTh1
  n_variates: 7
  seed: 42
  diffusion_type: binary
  model_type: dit
  image_height: 16
  coarse_image_height: 16
  fine_image_height: 16
  staged_representation: value_precision
  # Per-dataset max_scale overrides the scalar max_scale when the key matches.
  max_scale: 3.5
  max_scale_by_dataset:
    ETTh1: 5.2
    ETTh2: 9.0
    ETTm1: 7.7
    ETTm2: 9.0
    illness: 6.3
    exchange_rate: 10.6
    weather: 9.3
    electricity: 5.4
    traffic: 6.0
    PeMS: 11.8
    solar_Alabama: 13.0
    dynamic: 6.0
  window_norm_std_floor: 0.1
```
```yaml
# configs/binary_patch_refine_lb336_hz96_ordinal_tuned.yaml:1-70
# Live ordinal h96 patch-refine leaf (coarse → patch_refine, not coarse → fine).
# Transfers synthetic coarse/patch-refine weights from the non-ordinal donor;
# every real-data stage gets a fresh HP study under ordinal window norm.
# Submit: ./submit_binary.sh --configs binary_patch_refine_lb336_hz96_ordinal_tuned --datasets ...
extends: configs/binary_patch_refine_lb336_hz96_full.yaml

training:
  n_finetune_hp_trials: 4
  finetune_max_micro_batch: 2048

wandb:
  tags: [patch_refine, ordinal_norm, lb336_hz96, tuned]

experiment:
  name: binary-patch-refine-lb336-hz96-ordinal-tuned
  experiment_name: binary_patch_refine_lb336_hz96_ordinal_tuned
  # Ordinal ranks replace z-score window norm for this campaign.
  use_window_normalization: false
  use_ordinal_window_norm: true
  ordinal_ood_shift_causal_only: true
  patch_refine_unique_segments: true
  patch_refine_prev_cond_dropout: 0.5

phases:
  - phase: staged_diffusion_pretrain
    # Reuse synthetic geometry-compatible weights only.  A missing donor is a
    # submission error, never a request to silently synthesize a replacement.
    reuse_pretrain_from_config: binary_patch_refine_lb336_hz96_full
    require_reuse_pretrain: true

  - phase: diffusion_coarse_finetune_hp
    n_trials: 4
    max_epochs: 20
    patience: 8
    search_space: lr_eff_batch_univariate_ema
    hp_lr_min: 5.0e-5
    hp_lr_max: 1.5e-3
    effective_univariate_batch_grid: [512, 1024, 2048]
    ema_decay_grid: [0.0, 0.99, 0.995, 0.999]
    prediction_target: x0
    loss_weighting: min_snr
    min_snr_gamma: 2.0
    binary_noise_schedule: linear

  # Second stage: local tall-canvas refine (alternative to diffusion_fine_finetune_hp).
  - phase: diffusion_patch_refine_finetune_hp
    n_trials: 4
    max_epochs: 20
    patience: 8
    search_space: lr_eff_batch_univariate_ema
    hp_lr_min: 5.0e-5
    hp_lr_max: 1.5e-3
    # Patch-refine materializes ~17 local 32x8 crops per univariate window
    # (W=8, stride=6 on hz=96). The coarse-stage {512,1024,2048} grid OOMs
    # an L40S (~48 GiB) before a single trial completes; keep LR/EMA the same
    # but drop the univariate batch ladder by ~8x so crop count fits.
    effective_univariate_batch_grid: [64, 128, 256]
    ema_decay_grid: [0.0, 0.99, 0.995, 0.999]
    prediction_target: x0
    loss_weighting: min_snr
    min_snr_gamma: 2.0
    binary_noise_schedule: linear

  # Unique-seg AR: thin the window grid, but keep 20-step quad_t. Parallelize
  # the 4 independent prob samples in one generate (see staged_eval).
  - phase: staged_eval
    eval_test_fraction: 0.25
    test_stride: 16
    probabilistic_n_samples: 4
    probabilistic_num_inference_steps: 20
```

## 5. Orchestrator

Sequential phase loop; state carries ckpt paths / best HP between stages.

```python
# models/diffusion_tsf/pipeline/orchestrator.py:1-129
"""Pipeline orchestrator — runs an ordered list of phases.

Next hop for each phase: ``phase.execute(state)`` (or ``on_skip`` when cached).
Phases live under ``pipeline.phases.*`` and are registered in
``pipeline.phases.PHASE_REGISTRY``. State mutations (ckpt paths, best HP) are
the only contract between stages.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from models.diffusion_tsf.pipeline.phase import PipelinePhase
from models.diffusion_tsf.pipeline.state import PipelineState
from models.diffusion_tsf.pipeline.config import build_wandb_config
from models.diffusion_tsf.pipeline import wandb_utils

logger = logging.getLogger(__name__)


class Pipeline:
    """Runs phases sequentially, managing wandb groups and error handling."""

    def __init__(
        self,
        phases: List[PipelinePhase],
        state: PipelineState,
        merged_config: Optional[dict] = None,
    ):
        self.phases = phases
        self.state = state
        self.merged_config = merged_config or state.merged_config or {}

    def run(self) -> PipelineState:
        self.state.seed_everything()
        self.state.resolve_device()
        self.state.ensure_dirs()

        wandb_manifest: dict = {}
        phase_names = [p.name for p in self.phases]
        if self.state.wandb_enabled:
            wandb_manifest = wandb_utils.resolve_wandb_settings(
                self.state, self.merged_config,
            )

        logger.info("=" * 60)
        logger.info(f"Pipeline: {self.state.experiment_name}")
        logger.info(f"Dataset: {self.state.dataset} | Variates: {self.state.n_variates}")
        logger.info(f"Phases: {phase_names}")
        logger.info(f"Device: {self.state.device}")
        if self.state.wandb_group:
            logger.info(f"wandb group: {self.state.wandb_group}")
        if self.state.wandb_enabled:
            logger.info(f"wandb project: {self.state.wandb_project}")
        logger.info("=" * 60)

        run = None
        if self.state.wandb_enabled:
            pipeline_config = build_wandb_config(
                self.merged_config,
                self.state,
                phase_name=None,
                phase_overrides=None,
            )
            run = wandb_utils.init_pipeline_run(
                group=self.state.wandb_group or "",
                project=self.state.wandb_project,
                config=pipeline_config,
                tags=wandb_utils.build_pipeline_tags(
                    dataset=self.state.dataset,
                    phase_names=phase_names,
                    extra_tags=self.state.wandb_tags,
                ),
                yaml_path=self.merged_config.get("_yaml_path"),
                run_id=self.state.wandb_run_id,
            )
            if run is not None and getattr(run, "id", None):
                wandb_utils.record_pipeline_run_id(
                    self.state.checkpoint_dir,
                    run.id,
                    wandb_manifest,
                )
                self.state.wandb_run_id = run.id

        try:
            # Sequential phases. Skip = artifact already on disk (resume / cache).
            for i, phase in enumerate(self.phases):
                phase_label = f"[{i+1}/{len(self.phases)}] {phase.name}"

                if phase.should_skip(self.state):
                    logger.info(f"{phase_label}: SKIPPED (cached)")
                    try:
                        self.state = phase.on_skip(self.state)
                    except Exception:
                        # Fail-soft: skip hook errors must not kill a successful cache hit.
                        logger.exception(f"{phase_label}: on_skip failed (non-fatal)")
                    continue

                logger.info(f"{phase_label}: STARTING")
                logger.info(f"  overrides: {phase.overrides}")

                if self.state.wandb_enabled and run is not None:
                    phase_config = build_wandb_config(
                        self.merged_config,
                        self.state,
                        phase_name=phase.name,
                        phase_overrides=phase.overrides,
                    )
                    wandb_utils.begin_phase(phase_config)

                try:
                    self.state = phase.execute(self.state)
                    logger.info(f"{phase_label}: DONE")
                except KeyboardInterrupt:
                    logger.info(f"\nInterrupted during {phase.name}")
                    raise
                except Exception:
                    logger.exception(f"{phase_label}: FAILED")
                    raise
        finally:
            if run is not None:
                wandb_utils.finish_pipeline_run()

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)
        return self.state
```

`PipelineState` is the blackboard (device, dirs, stage ckpts, training knobs):

```python
# models/diffusion_tsf/pipeline/state.py:1-120
"""Shared mutable state passed between pipeline phases.

Built once in ``cli.main`` via ``PipelineState.from_config``, then threaded
through ``Pipeline.run``. Phases read knobs / prior ckpt paths and write back
artifacts (finetune ckpts, best HP dicts).

Still dual-writes onto ``train_multivariate_pipeline`` module globals through
``globals_bridge.patch_globals`` so legacy helpers keep working — new code
should prefer reading from this dataclass.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from models.diffusion_tsf.pipeline.config import (
    REQUIRED_EXPERIMENT_KEYS,
    apply_training_section_to_state,
    apply_wandb_section_to_state,
)


@dataclass
class PipelineState:
    """Everything that flows between phases."""

    # -- Experiment identity (frozen after init) --
    experiment_name: str = "experiment"
    dataset: str = "ETTh1"
    n_variates: int = 7
    seed: int = 42
    smoke_test: bool = False
    parallel_optuna_workers: int = 1

    # -- Model / diffusion knobs --
    diffusion_type: str = "binary"
    use_ordinal_window_norm: bool = False
    ordinal_ood_shift_causal_only: bool = False
    ordinal_tie_atol: float = 1e-6
    model_type: str = "dit"
    image_height: int = 32
    coarse_image_height: int = 16
    fine_image_height: int = 16
    max_scale: float = 3.5
    max_scale_by_dataset: Dict[str, float] = field(default_factory=dict)
    staged_representation: str = "value_precision"
    dit_patch_size: Tuple[int, int] = (8, 8)
    dit_embed_dim: int = 384
    dit_depth: int = 8
    dit_num_heads: int = 6
    dit_mlp_ratio: float = 4.0
    dit_dropout: float = 0.0
    use_patch_refine_stage: bool = False
    diffusion_stage: str = "joint"
    patch_refine_canvas_height: int = 256
    patch_refine_patch_height: int = 32
    patch_refine_patch_width: int = 8
    patch_refine_col_stride: int = 6
    patch_refine_unique_segments: bool = False
    patch_refine_prev_cond_dropout: float = 0.5
    dit_cond_patch_size: Optional[Tuple[int, int]] = None
    use_guidance_channel: bool = True
    guidance_placement: str = "canvas"
    guidance_type: str = "patch_decoder"
    mmpd_patch_size: int = 12
    cfg_dropout: float = 0.1
    deterministic_anchor_loss: bool = False
    deterministic_anchor_lambda: float = 0.99
    binary_anchor_input_mode: str = "stationary_flat"
    anchor_mse_proxy_lambda: float = 0.5
    eval_sampler: str = "quad_t"
    disable_cross_attention: bool = False
    use_window_normalization: bool = True
    window_norm_center: str = "mean"  # mean-only; "last" removed
    window_norm_std_floor: float = 1e-8
    window_norm_low_var_threshold: float = 0.0
    window_norm_low_var_unit_std: float = 1.0
    window_norm_low_var_unit_std_by_variate: Dict[str, List[float]] = field(default_factory=dict)
    window_norm_low_var_unit_std_by_dataset: Dict[str, float] = field(default_factory=dict)
    lookback_overlap_center_shift: bool = False
    use_raw_lookback_cond_channel: bool = False

    # -- Sequence geometry --
    lookback_length: int = 96
    forecast_length: int = 96
    lookback_overlap: int = 8
    diffusion_lookback_cap: int = 0
    diffusion_chunk_horizon: int = 0
    representation_time_stride: int = 1
    past_cond_resize_to_horizon: bool = True
    binary_noise_schedule: str = "linear"
    prediction_target: str = "x0"
    loss_weighting: str = "min_snr"
    min_snr_gamma: float = 2.0
    window_stride: int = 1
    binary_num_steps: int = 1000
    binary_beta_start: float = 1e-5
    binary_beta_end: float = 0.5
    n_diffusion_hp_trials: int = 10
    n_finetune_hp_trials: int = 10

    # -- Paths --
    checkpoint_dir: str = "./results/ckpts"
    results_dir: str = "./results/datasets"
    synth_cache_dir: Optional[str] = None
    datasets_dir: str = "./datasets"

    # -- Variate selection --
    variate_indices: Optional[List[int]] = None
    subset_id: Optional[str] = None
    data_subset: Dict[str, Any] = field(default_factory=dict)
    data_subset_resolved: Dict[str, Any] = field(default_factory=dict)

    # -- Device --
```

## 6. Pretrain phase

`staged_diffusion_stages` picks `("coarse","patch_refine")` or `("coarse","fine")`. Live ordinal leaves often **reuse** donor pretrain via `reused_paths` and skip synthetic training.

```python
# models/diffusion_tsf/pipeline/phases/staged_diffusion_pretrain.py:40-70
def _assert_supported_staged_pretrain_config(state: PipelineState) -> None:
    # Removed dual-concat / triple-scale layouts; keep the raise so old YAMLs scream.
    if getattr(state, "use_channel_dual_concat", False):
        raise ValueError(
            "channel_dual staged pretrain was removed; use coarse+fine or coarse+patch_refine"
        )
    if getattr(state, "use_vertical_dual_concat", False):
        raise ValueError(
            "vertical_dual staged pretrain was removed; use coarse+fine or coarse+patch_refine"
        )
    if getattr(state, "use_triple_scale", False):
        raise ValueError(
            "triple-scale (finer) staged pretrain was removed; use coarse+fine or coarse+patch_refine"
        )


def staged_diffusion_stages(state: PipelineState) -> tuple[str, ...]:
    """Which second stage runs after coarse: fine (alt) or patch_refine (live)."""
    _assert_supported_staged_pretrain_config(state)
    if getattr(state, "use_patch_refine_stage", False):
        return ("coarse", "patch_refine")
    return ("coarse", "fine")


def _stage_pretrain_cache_enabled(phase: PipelinePhase, state: PipelineState) -> bool:
    if state.extra.get("force_retrain_synthetic", False):
        return False
    if phase.get("reuse_pretrain_from_config"):
        return False
    if state.smoke_test:
        return False
```
```python
# models/diffusion_tsf/pipeline/phases/staged_diffusion_pretrain.py:824-960
class StagedDiffusionPretrainPhase(PipelinePhase):
    name = "staged_diffusion_pretrain"

    def _requires_reused_pretrain(self) -> bool:
        return bool(self.get("require_reuse_pretrain", False))

    def _config_name(self, state: PipelineState) -> str:
        if "phase1_config_name" in self.overrides:
            return str(self.require("phase1_config_name"))
        if state.extra.get("phase1_config_name"):
            return str(state.extra["phase1_config_name"])
        raise KeyError(f"phase {self.name!r} missing required key 'phase1_config_name'")

    def _cached_stage_ckpt(self, state: PipelineState, config_name: str, stage: str) -> Optional[str]:
        # Prefer explicit reuse donor, then local, then shared signature cache.
        reuse_from = self.get("reuse_pretrain_from_config")
        if reuse_from:
            reused = source_run_stage_pretrain_ckpt(state, str(reuse_from), stage)
            if reused:
                logger.info(
                    "  [%s] %s reused pretrain from *-%s-%s: %s",
                    self.name,
                    stage,
                    state.dataset,
                    reuse_from,
                    reused,
                )
                return reused
            return None

        local_ckpt = _stage_pretrain_ckpt(state, stage)
        if os.path.exists(local_ckpt):
            if _stage_pretrain_cache_enabled(self, state):
                logger.info("  [%s] %s local cached: %s", self.name, stage, local_ckpt)
                return local_ckpt
            sig_path = os.path.join(_stage_pretrain_dir(state, stage), ".signature")
            expected = _stage_pretrain_signature(state, config_name)
            if os.path.isfile(sig_path):
                with open(sig_path, encoding="utf-8") as f:
                    if f.read().strip() == expected:
                        logger.info(
                            "  [%s] %s local cached (signature match): %s",
                            self.name,
                            stage,
                            local_ckpt,
                        )
                        return local_ckpt
            logger.info(
                "  [%s] %s ignoring stale local pretrain (shared_cache=false): %s",
                self.name,
                stage,
                local_ckpt,
            )
        if _stage_pretrain_cache_enabled(self, state):
            shared_ckpt = _shared_stage_pretrain_ckpt(state, config_name, stage)
            if os.path.exists(shared_ckpt):
                logger.info("  [%s] %s shared cached: %s", self.name, stage, shared_ckpt)
                return shared_ckpt
            discovered = _discover_existing_stage_pretrain(state, stage)
            if discovered:
                logger.info("  [%s] %s discovered cached: %s", self.name, stage, discovered)
                return discovered
        return None

    def should_skip(self, state: PipelineState) -> bool:
        config_name = self._config_name(state)
        reuse_from = self.get("reuse_pretrain_from_config")
        if self._requires_reused_pretrain() and not reuse_from:
            raise ValueError(
                "require_reuse_pretrain=true requires reuse_pretrain_from_config"
            )
        ckpts = {
            stage: self._cached_stage_ckpt(state, config_name, stage)
            for stage in staged_diffusion_stages(state)
        }
        if self._requires_reused_pretrain():
            missing = [stage for stage, ckpt in ckpts.items() if not ckpt]
            if missing:
                required = ", ".join(
                    f"pretrained_{stage}/pretrained_diffusion.pt" for stage in missing
                )
                raise FileNotFoundError(
                    "Required synthetic staged-pretrain donor missing for "
                    f"dataset={state.dataset!r}, config={reuse_from!r}: {required}. "
                    "Refusing to train a replacement synthetic pretrain."
                )
        if all(ckpts.values()):
            state.diffusion_coarse_pretrain_ckpt = ckpts["coarse"]
            if "patch_refine" in ckpts:
                state.diffusion_patch_refine_pretrain_ckpt = ckpts["patch_refine"]
            elif "fine" in ckpts:
                state.diffusion_fine_pretrain_ckpt = ckpts["fine"]
            return True
        return False

    def on_skip(self, state: PipelineState) -> PipelineState:
        """Cached/reused pretrain still gets a couple RealTS pred panels + wandb."""
        try:
            config_name = self._config_name(state)
            source_dir = None
            try:
                source_dir = _phase1_source_dir(
                    state,
                    self.get("phase1_source_dir"),
                    config_name=config_name,
                )
            except FileNotFoundError:
                source_dir = None
            best_params: Dict[str, Any] = {}
            try:
                best_params = _resolve_diff_hp(state, source_dir)
            except FileNotFoundError:
                pass
            guidance_ckpt = _find_existing_synthetic_patch_guidance(state, source_dir)
            _log_synthetic_pretrain_visualizations(
                state,
                guidance_ckpt=guidance_ckpt,
                best_params=best_params or None,
            )
        except Exception as e:
            logger.warning(
                "[%s] cached-pretrain viz failed: %s", self.name, e, exc_info=True,
            )
        return state

    def execute(self, state: PipelineState) -> PipelineState:
        from models.diffusion_tsf.pipeline.train.pretrain import pretrain_diffusion
        import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod

        config_name = self._config_name(state)
        reuse_from = self.get("reuse_pretrain_from_config")
        if reuse_from:
            missing = []
            for stage in staged_diffusion_stages(state):
                ckpt = self._cached_stage_ckpt(state, config_name, stage)
                if ckpt:
                    if stage == "coarse":
```
```python
# models/diffusion_tsf/pipeline/phases/staged_diffusion_pretrain.py:949-1140
    def execute(self, state: PipelineState) -> PipelineState:
        from models.diffusion_tsf.pipeline.train.pretrain import pretrain_diffusion
        import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod

        config_name = self._config_name(state)
        reuse_from = self.get("reuse_pretrain_from_config")
        if reuse_from:
            missing = []
            for stage in staged_diffusion_stages(state):
                ckpt = self._cached_stage_ckpt(state, config_name, stage)
                if ckpt:
                    if stage == "coarse":
                        state.diffusion_coarse_pretrain_ckpt = ckpt
                    elif stage == "fine":
                        state.diffusion_fine_pretrain_ckpt = ckpt
                    elif stage == "patch_refine":
                        state.diffusion_patch_refine_pretrain_ckpt = ckpt
                    else:
                        raise ValueError(f"Unknown staged diffusion stage: {stage!r}")
                else:
                    missing.append(stage)
            if not missing:
                source_dir = _phase1_source_dir(
                    state,
                    self.get("phase1_source_dir"),
                    config_name=config_name,
                )
                best_params = _resolve_diff_hp(state, source_dir)
                guidance_ckpt, guidance_meta = _resolve_synthetic_patch_guidance(
                    state,
                    source_dir,
                    retrain_synthetic_patch_guidance=bool(
                        self.get("retrain_synthetic_patch_guidance")
                        or self.get("retrain_synthetic_itrans", False)
                    ),
                )
                n_samples = int(self.require("n_samples"))
                if state.smoke_test:
                    n_samples = min(n_samples, 4)
                _log_staged_pretrain_diagnostics(
                    state,
                    guidance_ckpt=guidance_ckpt,
                    guidance_meta=guidance_meta,
                    best_params=best_params,
                    n_samples=n_samples,
                )
                _log_synthetic_pretrain_visualizations(
                    state,
                    guidance_ckpt=guidance_ckpt,
                    best_params=best_params,
                )
                return state
            if self._requires_reused_pretrain():
                required = ", ".join(
                    f"pretrained_{stage}/pretrained_diffusion.pt" for stage in missing
                )
                raise FileNotFoundError(
                    "Required synthetic staged-pretrain donor missing for "
                    f"dataset={state.dataset!r}, config={reuse_from!r}: {required}. "
                    "Refusing to train a replacement synthetic pretrain."
                )
            # Legacy configs retain the soft fallback when a donor has been purged.
            logger.warning(
                "  [%s] reuse_pretrain_from_config=%r missing pretrained_%s under "
                "*-%s-%s (incl. cross-dataset fallback); training synthetic pretrain instead",
                self.name,
                reuse_from,
                "/pretrained_".join(missing),
                state.dataset,
                reuse_from,
            )
        source_dir = _phase1_source_dir(
            state,
            self.get("phase1_source_dir"),
            config_name=config_name,
        )
        best_params = _resolve_diff_hp(state, source_dir)
        needs_guidance = state.needs_guidance
        if needs_guidance:
            guidance_ckpt, guidance_meta = _resolve_synthetic_patch_guidance(
                state,
                source_dir,
                retrain_synthetic_patch_guidance=bool(
                    self.get("retrain_synthetic_patch_guidance")
                    or self.get("retrain_synthetic_itrans", False)
                ),
            )
        else:
            guidance_ckpt, guidance_meta = "", {}

        n_samples = int(self.require("n_samples"))
        epochs = int(self.require("epochs"))
        patience = int(self.require("patience"))
        if state.smoke_test:
            n_samples = min(n_samples, 4)
            epochs = 1
            patience = 1

        shared_cache = _stage_pretrain_cache_enabled(self, state)
        shared_wait_seconds = float(self.get("shared_cache_wait_seconds", 6 * 60 * 60))

        for stage in staged_diffusion_stages(state):
            stage_epochs = 1 if state.smoke_test else int(self.get(f"{stage}_epochs", epochs))
            stage_patience = min(patience, stage_epochs)
            ckpt = self._cached_stage_ckpt(state, config_name, stage)
            if ckpt is None and shared_cache:
                shared_ckpt = _wait_for_shared_stage_ckpt(
                    state,
                    config_name,
                    stage,
                    wait_seconds=shared_wait_seconds,
                )
                if shared_ckpt:
                    ckpt = shared_ckpt
                else:
                    ckpt = _shared_stage_pretrain_ckpt(state, config_name, stage)
                    stage_dir = os.path.dirname(ckpt)
                    os.makedirs(stage_dir, exist_ok=True)
                    try:
                        with _synthetic_pretrain_globals(pipeline_mod, state, stage):
                            # Stamp signature before training so a mid-run kill can resume
                            # from pretrained_diffusion.pt (best-so-far) without retraining.
                            sig_path = os.path.join(stage_dir, ".signature")
                            with open(sig_path, "w", encoding="utf-8") as f:
                                f.write(_stage_pretrain_signature(state, config_name))
                            ckpt, _best_val = pretrain_diffusion(
                                best_params=best_params,
                                guidance_checkpoint=guidance_ckpt,
                                n_samples=n_samples,
                                epochs=stage_epochs,
                                patience=stage_patience,
                                checkpoint_dir=stage_dir,
                                smoke_test=state.smoke_test,
                            )
                    finally:
                        _release_shared_lock(ckpt)

                    meta_path = os.path.join(stage_dir, "shared_pretrain_metadata.json")
                    with open(meta_path, "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "dataset": state.dataset,
                                "n_variates": state.n_variates,
                                "config_name": config_name,
                                "stage": stage,
                                "signature": _stage_pretrain_signature(state, config_name),
                                "checkpoint": ckpt,
                            },
                            f,
                            indent=2,
                            sort_keys=True,
                        )
            elif ckpt is None:
                stage_dir = _stage_pretrain_dir(state, stage)
                os.makedirs(stage_dir, exist_ok=True)
                # Stamp before training so mid-run kills can reuse best-so-far weights.
                sig_path = os.path.join(stage_dir, ".signature")
                with open(sig_path, "w", encoding="utf-8") as f:
                    f.write(_stage_pretrain_signature(state, config_name))
                with _synthetic_pretrain_globals(pipeline_mod, state, stage):
                    ckpt, _best_val = pretrain_diffusion(
                        best_params=best_params,
                        guidance_checkpoint=guidance_ckpt,
                        n_samples=n_samples,
                        epochs=stage_epochs,
                        patience=stage_patience,
                        checkpoint_dir=stage_dir,
                        smoke_test=state.smoke_test,
                    )
            if stage == "coarse":
                state.diffusion_coarse_pretrain_ckpt = ckpt
            elif stage == "fine":
                state.diffusion_fine_pretrain_ckpt = ckpt
            elif stage == "patch_refine":
                state.diffusion_patch_refine_pretrain_ckpt = ckpt
            else:
                raise ValueError(f"Unknown staged diffusion stage: {stage!r}")

        _log_staged_pretrain_diagnostics(
            state,
            guidance_ckpt=guidance_ckpt,
            guidance_meta=guidance_meta,
            best_params=best_params,
            n_samples=n_samples,
        )
        _log_synthetic_pretrain_visualizations(
            state,
            guidance_ckpt=guidance_ckpt,
            best_params=best_params,
        )

        return state
```

Synthetic pool + train loop helpers:

```python
# models/diffusion_tsf/pipeline/train/pretrain.py:1-163
"""Staged synthetic diffusion pretrain loop.

Called per stage from StagedDiffusionPretrainPhase. Builds RealTS loader
(realts.get_synthetic_dataloader), DiffusionTSF via create_diffusion_model, then
train_diffusion_epoch. Writes pretrained_diffusion.pt under the stage dir.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict

import torch
from torch.utils.data import DataLoader, Subset

from models.diffusion_tsf.pipeline.train.checkpointing import EarlyStopping, save_checkpoint
from models.diffusion_tsf.pipeline.train.diffusion_loop import (
    train_diffusion_epoch,
    validate_diffusion_epoch,
)
from models.diffusion_tsf.realts import get_synthetic_dataloader

logger = logging.getLogger(__name__)


def pretrain_diffusion(
    best_params: Dict,
    guidance_checkpoint: str,
    n_samples: int,
    epochs: int,
    patience: int,
    checkpoint_dir: str,
    smoke_test: bool = False,
) -> str:
    """Train one staged diffusion checkpoint on synthetic data (not post-HP retrain)."""
    # Lazy: train_multivariate_pipeline re-exports this module at import time.
    from models.diffusion_tsf import train_multivariate_pipeline as m

    logger.info("=" * 60)
    logger.info("Staged synthetic diffusion pretrain (with patch_decoder guidance)")
    logger.info("Samples: %s, Epochs: %s, Patience: %s", n_samples, epochs, patience)
    logger.info("Params: %s", best_params)
    logger.info("=" * 60)

    device = m.get_device()

    lr = m.require_tuned_param(best_params, "learning_rate", "Diffusion pretraining")
    tuned_batch_size = m.require_tuned_param(best_params, "batch_size", "Diffusion pretraining")
    batch_size = tuned_batch_size

    needs_guidance = bool(m.USE_GUIDANCE_CHANNEL) or not bool(m.DISABLE_CROSS_ATTENTION)
    guidance = None
    if needs_guidance:
        if not guidance_checkpoint:
            raise ValueError("guidance_checkpoint is required when guidance/cross-attn is enabled")
        guidance = m.load_wrapped_guidance(
            guidance_checkpoint,
            m.N_VARIATES,
            device,
            guidance_type="patch_decoder",
        )

    synth_cache = m.get_synth_cache_dir(checkpoint_dir=checkpoint_dir, smoke_test=smoke_test)
    n_val = 0 if smoke_test else min(n_samples // 10, 5000)
    epoch_cap = 1 if smoke_test else m.synthetic_epoch_capacity_pretrain_diffusion()
    synthetic_loader = get_synthetic_dataloader(
        batch_size=min(16, max(2, tuned_batch_size)),
        lookback_length=m.LOOKBACK_LENGTH,
        forecast_length=m.FORECAST_LENGTH,
        num_variables=m.N_VARIATES,
        num_samples=n_samples,
        num_workers=0 if smoke_test else 4,
        lookback_overlap=m.LOOKBACK_OVERLAP,
        cache_dir=synth_cache,
        skip_cross_var_aug=(m.N_VARIATES > 32),
        val_tail_n=n_val,
        synthetic_epoch_capacity=epoch_cap,
    )

    dataset = synthetic_loader.dataset
    train_subset = Subset(dataset, list(range(len(dataset) - n_val)))
    val_subset = Subset(dataset, list(range(len(dataset) - n_val, len(dataset))))
    batch_size = tuned_batch_size or (
        min(4, m.DIFFUSION_BATCH_SIZE) if smoke_test else m.DIFFUSION_BATCH_SIZE
    )
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0 if smoke_test else 4,
    )
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)

    model_kwargs = m.anchor_kwargs_from_params(best_params)
    for key in (
        "max_scale",
        "dit_dropout",
        "prediction_target",
        "loss_weighting",
        "use_ordinal_window_norm",
        "ordinal_tie_atol",
    ):
        if key in best_params:
            model_kwargs[key] = best_params[key]
    model = m.create_diffusion_model(
        guidance_model=guidance,
        **model_kwargs,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01,
    )

    early_stop = EarlyStopping(patience=patience)
    best_val_loss = float("inf")
    ckpt_path = os.path.join(checkpoint_dir, "pretrained_diffusion.pt")

    for epoch in range(epochs):
        t0 = time.time()
        train_loss = train_diffusion_epoch(
            model,
            train_loader,
            device,
            optimizer,
            set_loader_mode=m._set_ordinal_loader_mode,
            set_training_epoch=m.set_realts_training_epoch,
            epoch=epoch,
        )
        val_loss = validate_diffusion_epoch(
            model,
            val_loader,
            device,
            set_loader_mode=m._set_ordinal_loader_mode,
        )

        scheduler.step()
        logger.info(
            "[Diffusion] Epoch %d/%d | Train: %.4f | Val: %.4f | LR: %.2e | Time: %.1fs",
            epoch + 1,
            epochs,
            train_loss,
            val_loss,
            scheduler.get_last_lr()[0],
            time.time() - t0,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model,
                optimizer,
                epoch,
                train_loss,
                val_loss,
                {"diffusion_params": best_params, "guidance_checkpoint": guidance_checkpoint},
                ckpt_path,
            )
            logger.info("  -> New best! Saved to %s", ckpt_path)

        if early_stop(val_loss):
```
```python
# models/diffusion_tsf/pipeline/reused_paths.py:1-80
"""Canonical reused checkpoint layout under $SCRATCH/ts-sandbox/reused/.

Pretrain / guidance / tuned_params donors so new Slurm runs skip synth pretrain
or copy HP. find_reused_* returns a path only if the file exists.
find_reused_binary_staged_root still accepts channel_dual/vertical_dual donors
even though those train phases were removed (compat for old trees).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence


def reused_root() -> str:
    scratch = os.environ.get("SCRATCH")
    if scratch:
        return os.path.join(scratch, "ts-sandbox", "reused")
    return os.path.join(os.getcwd(), "reused")


def reused_pretrain_ckpt(config_suffix: str, stage: str) -> str:
    return os.path.join(
        reused_root(),
        "pretrain",
        config_suffix,
        f"pretrained_{stage}",
        "pretrained_diffusion.pt",
    )


def reused_guidance_ckpt(config_suffix: str, subset_id: str) -> str:
    return os.path.join(
        reused_root(),
        "guidance",
        config_suffix,
        f"{subset_id}_patch_guidance.pt",
    )


def reused_tuned_params_meta(config_suffix: str, subset_id: str, stage: str) -> str:
    return os.path.join(
        reused_root(),
        "tuned_params",
        config_suffix,
        subset_id,
        stage,
        "metadata.json",
    )


def reused_stage_best_ckpt(config_suffix: str, subset_id: str, stage: str) -> str:
    return os.path.join(
        reused_root(),
        "tuned_params",
        config_suffix,
        subset_id,
        stage,
        "best.pt",
    )


def reused_binary_staged_root(config_stem: str) -> str:
    return os.path.join(reused_root(), "binary", config_stem)


def reused_mmpd_campaign_root(config_suffix: str) -> str:
    return os.path.join(reused_root(), "mmpd", config_suffix)


def find_reused_pretrain_ckpt(config_suffix: str, stage: str) -> Optional[str]:
    path = reused_pretrain_ckpt(config_suffix, stage)
    return path if os.path.isfile(path) else None


def find_reused_guidance_ckpt(config_suffix: str, subset_id: str) -> Optional[str]:
    path = reused_guidance_ckpt(config_suffix, subset_id)
    return path if os.path.isfile(path) else None
```
```python
# models/diffusion_tsf/realts.py:1-60
"""RealTS synthetic series for staged pretrain (ViTime-style generators).

get_synthetic_dataloader is the pipeline entry. Generators (RWB/PWB/...) build
diverse shapes so coarse/fine/patch_refine learn structure before real finetune.
Pool files under cache_dir are flock-locked for parallel Slurm jobs.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Tuple
import logging
import os
import uuid
import fcntl
from contextlib import contextmanager

try:
    from .augmentation import generate_multivariate_synthetic_data
except ImportError:
    from augmentation import generate_multivariate_synthetic_data

logger = logging.getLogger(__name__)


@contextmanager
def _synth_pool_file_lock(cache_path: str):
    """Exclusive lock so parallel Slurm jobs do not corrupt a shared pool file."""
    lock_path = f"{cache_path}.lock"
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as lockf:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)


# Irregular Periodicity Helpers

def _choose_irregularity() -> Optional[str]:
    """Randomly select an irregularity level for periodic generators.
    
    Returns None (regular) 50% of the time, otherwise one of three levels:
    - 'mild': 1-2 periods randomly stretched (20%)
    - 'medium': period length slowly oscillates over time (15%)
    - 'extreme': every period has independently random length (15%)
    """
    r = np.random.random()
    if r < 0.50:
        return None
    elif r < 0.70:
        return 'mild'
    elif r < 0.85:
        return 'medium'
    else:
        return 'extreme'


def _irregular_phase(length: int, base_periods: float, level: str) -> np.ndarray:
```

## 7. Coarse / Fine / PatchRefine HP finetune

Three thin subclasses share `_BaseStagedDiffusionFinetuneHPPhase`. `execute` runs Optuna (or fixed HP), `_train_once` calls `model.forward` with the stage set.

```python
# models/diffusion_tsf/pipeline/phases/staged_diffusion_finetune_hp.py:1-35
"""Optuna HP finetune for one staged model (coarse, fine, or patch_refine).

Thin subclasses at the bottom only set ``stage``. Shared logic:
  should_skip / on_skip cache best.pt+metadata
  run() probes batch, Optuna trials, optional long refit, writes stage best.pt
Loads synth pretrain via _stage_pretrain_ckpt, then real data. Next: staged_eval.
"""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import time
from typing import Any, Dict, Optional, Tuple

import torch
from optuna.exceptions import TrialPruned
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from torch.utils.data import DataLoader, Subset

from models.diffusion_tsf.pipeline.phase import PipelinePhase
from models.diffusion_tsf.pipeline.state import PipelineState
from models.diffusion_tsf.pipeline import wandb_utils
from models.diffusion_tsf.pipeline.phases.staged_diffusion_pretrain import (
    _stage_pretrain_ckpt,
    discover_dataset_run_ckpt_dir,
    patch_stage_globals,
)

logger = logging.getLogger(__name__)
```
```python
# models/diffusion_tsf/pipeline/phases/staged_diffusion_finetune_hp.py:660-735
class _BaseStagedDiffusionFinetuneHPPhase(PipelinePhase):
    stage = ""

    def should_skip(self, state: PipelineState) -> bool:
        # retrain=true forces a fresh train on a new run, but --resume must still
        # honor local best.pt+metadata so we can finish eval after quota crashes.
        if self.get("retrain", False) and not bool(getattr(state, "resume", False)):
            return False
        best_pt = _stage_best_ckpt(state, self.stage)
        meta = os.path.join(_stage_subset_dir(state, self.stage), "metadata.json")
        if os.path.exists(best_pt) and os.path.exists(meta):
            try:
                # Corrupt/truncated saves (disk quota) look like a cache hit otherwise.
                torch.load(best_pt, map_location="cpu", weights_only=False)
            except Exception as e:
                logger.warning(
                    "  [%s] ignoring unreadable cache %s: %s", self.name, best_pt, e,
                )
                return False
            # Search@N + refit@M: only skip once the long refit has finished.
            if self.get("refit_best_max_epochs") is not None:
                try:
                    with open(meta, encoding="utf-8") as f:
                        meta_obj = json.load(f)
                except Exception as e:
                    logger.warning("  [%s] ignoring unreadable meta %s: %s", self.name, meta, e)
                    return False
                if not meta_obj.get("refit_completed"):
                    logger.info(
                        "  [%s] search ckpt present but refit_best_max_epochs pending; not skipping",
                        self.name,
                    )
                    return False
            logger.info("  [%s] cached: %s", self.name, best_pt)
            params = None
            try:
                with open(meta) as f:
                    params = json.load(f).get("tuned_params")
            except Exception as e:
                logger.warning("Failed to load tuned params from %s: %s", meta, e)
            # Stash ckpt paths on state so staged_eval / later stages find them.
            if self.stage == "coarse":
                state.diffusion_coarse_finetune_ckpt = best_pt
                state.coarse_finetune_best_params = params
            elif self.stage == "fine":
                state.diffusion_fine_finetune_ckpt = best_pt
                state.fine_finetune_best_params = params
            elif self.stage == "patch_refine":
                state.diffusion_patch_refine_finetune_ckpt = best_pt
                state.patch_refine_finetune_best_params = params
            else:
                raise ValueError(f"Unknown diffusion stage {self.stage!r}")
            return True
        return False

    def on_skip(self, state: PipelineState) -> PipelineState:
        best_pt = _stage_best_ckpt(state, self.stage)
        if not os.path.exists(best_pt):
            return state
        meta_path = os.path.join(_stage_subset_dir(state, self.stage), "metadata.json")
        best_params: Dict[str, Any] = {}
        try:
            with open(meta_path, encoding="utf-8") as f:
                best_params = dict(json.load(f).get("tuned_params") or {})
        except Exception as e:
            logger.warning("Failed to load tuned params from %s: %s", meta_path, e)
        try:
            self._log_post_finetune_viz_and_diagnostics(
                state,
                final_ckpt=best_pt,
                best_params=best_params,
            )
        except Exception as e:
            logger.warning("[%s] cached-phase viz/wandb log failed: %s", self.name, e, exc_info=True)
        return state
```
```python
# models/diffusion_tsf/pipeline/phases/staged_diffusion_finetune_hp.py:834-980
    def _pretrained_ckpt(self, state: PipelineState) -> Optional[str]:
        if bool(self.get("from_random_init", False)):
            logger.info(
                "  [%s] from_random_init=true; skipping staged pretrain load",
                self.name,
            )
            return None
        attr = {
            "coarse": state.diffusion_coarse_pretrain_ckpt,
            "fine": state.diffusion_fine_pretrain_ckpt,
            "patch_refine": state.diffusion_patch_refine_pretrain_ckpt,
        }[self.stage]
        candidates = [
            self.get("pretrained_ckpt"),
            attr,
            _stage_pretrain_ckpt(state, self.stage),
        ]
        for path in candidates:
            if path and os.path.exists(path):
                return path
        raise FileNotFoundError(
            f"{self.name} requires a staged {self.stage} pretrain checkpoint. "
            f"Expected one of: {', '.join(str(p) for p in candidates if p)}"
        )

    def _build_model(
        self,
        *,
        state: PipelineState,
        n_iv: int,
        itrans_guidance,
        device: torch.device,
        params: Dict[str, Any],
    ):
        from models.diffusion_tsf.train_multivariate_pipeline import (
            anchor_kwargs_from_params,
            create_diffusion_model,
            dataset_window_lengths,
        )

        ds_lb, ds_hz = dataset_window_lengths(state.dataset)
        model_kwargs = anchor_kwargs_from_params(params)
        model_kwargs.update(_state_anchor_kwargs(state))
        model_kwargs.update(_model_kwargs_from_tuned(params))
        return create_diffusion_model(
            n_variates=n_iv,
            lookback=ds_lb,
            horizon=ds_hz,
            guidance_model=itrans_guidance,
            diffusion_stage=self.stage,
            use_guidance_channel=state.use_guidance_channel,
            **model_kwargs,
        ).to(device)



    def _refit_best_if_configured(
        self,
        *,
        state: PipelineState,
        train_ds,
        val_ds,
        best_params: Dict[str, Any],
        diff_ckpt: Optional[str],
        ft_guidance_ckpt: str,
        device: torch.device,
        variate_indices,
        final_ckpt: str,
        hp_best_val_loss: float,
        best_trial_num: int,
        search_space: str,
        search_max_epochs: int,
        search_patience: int,
        subset_dir: str,
        subset_id: str,
        subset_meta: Dict[str, Any],
        norm_stats: Dict[str, Any],
        selection_metric: str = "diffusion_val",
        anchor_val_ds=None,
    ) -> Tuple[Dict[str, Any], float, int, bool]:
        """Optionally retrain the Optuna winner from pretrain for more epochs.

        Returns (best_params, final_val, final_epoch, refit_completed).
        """
        refit_epochs = self.get("refit_best_max_epochs")
        if refit_epochs is None:
            return best_params, float(hp_best_val_loss), 0, False
        refit_epochs = int(refit_epochs)
        if refit_epochs < 1:
            raise ValueError(f"refit_best_max_epochs must be >= 1, got {refit_epochs}")
        if state.smoke_test:
            refit_epochs = 1
        refit_patience = int(self.get("refit_best_patience", refit_epochs))
        if state.smoke_test:
            refit_patience = 1
        best_params = _with_state_anchor_params(
            best_params, state,
        )
        selection_metric = str(selection_metric)
        logger.info(
            "  [%s] refit_best: search_epochs=%d -> refit_epochs=%d patience=%d "
            "lr=%.2e g=%s selection=%s",
            self.name,
            search_max_epochs,
            refit_epochs,
            refit_patience,
            float(best_params.get("learning_rate", 0.0)),
            selection_metric,
        )
        # Persist search winner before long refit so --resume can skip Optuna.
        meta_pending: Dict[str, Any] = {
            "subset_id": subset_id,
            "dataset_name": state.dataset,
            "variate_indices": list(variate_indices),
            "data_subset": subset_meta,
            "norm_mean": norm_stats["mean"].tolist(),
            "norm_std": norm_stats["std"].tolist(),
            "tuned_params": best_params,
            "best_trial": best_trial_num,
            "hp_best_val_loss": float(hp_best_val_loss),
            "best_val_loss": float(hp_best_val_loss),
            "diffusion_stage": self.stage,
            "staged_representation": state.staged_representation,
            "search_space": search_space,
            "selection_metric": selection_metric,
            "max_epochs": search_max_epochs,
            "patience": search_patience,
            "refit_best_max_epochs": refit_epochs,
            "refit_completed": False,
        }
        with open(os.path.join(subset_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta_pending, f, indent=2, sort_keys=True)

        final_val, final_epoch = self._train_once(
            state=state,
            train_ds=train_ds,
            val_ds=val_ds,
            params=best_params,
            pretrained_path=diff_ckpt,
            guidance_checkpoint=ft_guidance_ckpt,
            device=device,
            variate_indices=variate_indices,
            ckpt_path=final_ckpt,
            max_epochs=refit_epochs,
            patience=refit_patience,
            trial=None,
            selection_metric=selection_metric,
```

`_train_once` (core train loop for a trial) — abbreviated mid-helpers, keep start + loss step:

```python
# models/diffusion_tsf/pipeline/phases/staged_diffusion_finetune_hp.py:985-1200
    def _train_once(
        self,
        *,
        state: PipelineState,
        train_ds,
        val_ds,
        params: Dict[str, Any],
        pretrained_path: Optional[str],
        guidance_checkpoint: str,
        device: torch.device,
        variate_indices,
        ckpt_path: Optional[str],
        max_epochs: int,
        patience: int,
        trial=None,
        guidance=None,
        pretrained_state_dict: Optional[Dict[str, Any]] = None,
        selection_metric: str = "diffusion_val",
        anchor_val_ds=None,
    ) -> Tuple[float, int]:
        from models.diffusion_tsf.train_multivariate_pipeline import (
            EarlyStopping,
            amp_context,
            dataset_window_lengths,
            load_diffusion_state_keep_attached_guidance,
            load_wrapped_guidance,
            save_checkpoint,
            unwrap_model,
        )

        params = _with_state_anchor_params(params, state)
        _decoded = {"anchor_mse"}
        if selection_metric not in {"diffusion_val"} | _decoded:
            raise ValueError(
                f"selection_metric must be diffusion_val or anchor_mse, "
                f"got {selection_metric!r}"
            )
        use_decoded = selection_metric in _decoded
        if use_decoded and anchor_val_ds is None:
            raise ValueError(
                f"selection_metric={selection_metric} requires anchor_val_ds"
            )
        n_iv = len(variate_indices)
        batch_size = int(params["batch_size"])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        anchor_loader = None
        if use_decoded:
            # Cap micro-batch for one-shot generate; still covers the full subset.
            anchor_bs = max(1, min(batch_size, 8))
            anchor_loader = DataLoader(
                anchor_val_ds, batch_size=anchor_bs, shuffle=False, num_workers=0,
            )
        n_train_batches = len(train_loader)
        n_val_batches = len(val_loader)
        trial_label = (
            f"trial={trial.number}" if trial is not None else "trial=single"
        )
        logger.info(
            "  [%s/%s] %s START epochs=%d patience=%d lr=%.2e bs=%d accum=%d "
            "train_batches=%d val_batches=%d selection=%s anchor_batches=%s g=%s",
            self.name,
            self.stage,
            trial_label,
            max_epochs,
            patience,
            float(params["learning_rate"]),
            batch_size,
            int(params.get("gradient_accumulation_steps", 1)),
            n_train_batches,
            n_val_batches,
            selection_metric,
            len(anchor_loader) if anchor_loader is not None else "-",
        )
        _log_gpu_mem(f"{self.stage}/{trial_label}/start")

        ds_lb, ds_hz = dataset_window_lengths(state.dataset)
        if guidance is None and guidance_checkpoint:
            guidance = load_wrapped_guidance(
                guidance_checkpoint,
                n_iv,
                device,
                guidance_type=state.guidance_type,
                dataset_lookback=ds_lb,
                dataset_horizon=ds_hz,
            )
        model = self._build_model(
            state=state,
            n_iv=n_iv,
            itrans_guidance=guidance,
            device=device,
            params=params,
        )
        try:
            if pretrained_path or pretrained_state_dict is not None:
                if pretrained_state_dict is None:
                    ckpt = torch.load(pretrained_path, map_location=device, weights_only=False)
                    pretrained_state_dict = ckpt["model_state_dict"]
                load_diffusion_state_keep_attached_guidance(model, pretrained_state_dict)
            else:
                logger.info(
                    "  [%s] random init (no pretrain ckpt)",
                    self.name,
                )

            optimizer = torch.optim.AdamW(model.parameters(), lr=float(params["learning_rate"]))

            early_stop = EarlyStopping(patience=patience)
            ema = _Ema(model, float(params.get("ema_decay", 0.0))) if params.get("ema_decay", 0.0) else None
            accum_steps = max(1, int(params.get("gradient_accumulation_steps", 1)))
            best_val = float("inf")
            best_epoch = 0
            saved_ckpt = False
            train_log_stride = max(1, n_train_batches // 4)
            val_log_stride = max(1, n_val_batches // 2)
            epoch_t0 = time.perf_counter()
            epoch_history: list[Dict[str, Any]] = []

            for epoch in range(max_epochs):
                epoch_start = time.perf_counter()
                # UniquePatchSegmentDataset.set_epoch may sit under Subset.
                _epoch_ds = train_ds
                while hasattr(_epoch_ds, "dataset") and not hasattr(_epoch_ds, "set_epoch"):
                    _epoch_ds = _epoch_ds.dataset
                if hasattr(_epoch_ds, "set_epoch"):
                    _epoch_ds.set_epoch(epoch)
                logger.info(
                    "  [%s/%s] %s epoch %d/%d train_start",
                    self.name, self.stage, trial_label, epoch + 1, max_epochs,
                )
                model.train()
                from models.diffusion_tsf.train_multivariate_pipeline import _set_ordinal_loader_mode

                _set_ordinal_loader_mode(model, train_loader, eval_mode=False)
                train_loss = 0.0
                n_train = 0
                optimizer.zero_grad(set_to_none=True)
                for batch_idx, batch in enumerate(train_loader):
                    if batch_idx == 0 or (batch_idx + 1) % train_log_stride == 0 or batch_idx + 1 == n_train_batches:
                        logger.info(
                            "  [%s/%s] %s epoch %d/%d train_batch %d/%d",
                            self.name, self.stage, trial_label,
                            epoch + 1, max_epochs, batch_idx + 1, n_train_batches,
                        )
                    if len(batch) == 3:
                        past, future, patch_col0 = batch
                        patch_col0 = patch_col0.to(device)
                    else:
                        past, future = batch
                        patch_col0 = None
                    past, future = past.to(device), future.to(device)
                    with amp_context():
                        loss = model.get_loss(past, future, patch_col0=patch_col0) / accum_steps
                    loss.backward()
                    if (batch_idx + 1) % accum_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        if ema is not None:
                            ema.update(model)
                    train_loss += float(loss.item()) * accum_steps
                    n_train += 1
                if accum_steps > 1 and len(train_loader) % accum_steps != 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if ema is not None:
                        ema.update(model)

                train_loss_avg = train_loss / max(n_train, 1)
                train_elapsed = time.perf_counter() - epoch_start
                logger.info(
                    "  [%s/%s] %s epoch %d/%d train_done loss=%.4f time=%.1fs",
                    self.name, self.stage, trial_label,
                    epoch + 1, max_epochs, train_loss_avg, train_elapsed,
                )

                backup = ema.swap_in(model) if ema is not None else None
                model.eval()
                _set_ordinal_loader_mode(model, val_loader, eval_mode=True)
                val_loss = 0.0
                n_val = 0
                val_start = time.perf_counter()
                logger.info(
                    "  [%s/%s] %s epoch %d/%d val_start",
                    self.name, self.stage, trial_label, epoch + 1, max_epochs,
                )
                with torch.no_grad():
                    for val_idx, batch in enumerate(val_loader):
                        if val_idx == 0 or (val_idx + 1) % val_log_stride == 0 or val_idx + 1 == n_val_batches:
                            logger.info(
                                "  [%s/%s] %s epoch %d/%d val_batch %d/%d",
                                self.name, self.stage, trial_label,
                                epoch + 1, max_epochs, val_idx + 1, n_val_batches,
                            )
                        if len(batch) == 3:
                            past, future, patch_col0 = batch
                            patch_col0 = patch_col0.to(device)
                        else:
                            past, future = batch
                            patch_col0 = None
                        past, future = past.to(device), future.to(device)
                        with amp_context():
                            loss = model.get_loss(past, future, patch_col0=patch_col0)
                        val_loss += float(loss.item())
                        n_val += 1
                val_loss /= max(n_val, 1)
                val_elapsed = time.perf_counter() - val_start
                anchor_mse = None
                if use_decoded:
                    anchor_start = time.perf_counter()
                    anchor_mse = _anchor_mse_on_loader(model, anchor_loader, device)
                    selection_score = float(anchor_mse)
                    logger.info(
                        "  [%s/%s] %s epoch %d/%d anchor_mse=%.6f time=%.1fs",
                        self.name, self.stage, trial_label,
```
```python
# models/diffusion_tsf/pipeline/phases/staged_diffusion_finetune_hp.py:1312-1550
    def execute(self, state: PipelineState) -> PipelineState:
        # Optuna (or fixed HP) finetune for self.stage; writes best.pt under stage subset dir.
        from models.diffusion_tsf.train_multivariate_pipeline import (
            dataset_window_lengths,
            generate_dataset_job,
            load_dataset,
            load_wrapped_guidance,
        )
        from models.diffusion_tsf.pipeline.train.batch_config import (
            configured_finetune_micro_batch,
            configured_max_diffusion_batch,
        )
        import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod

        patch_stage_globals(pipeline_mod, state, self.stage, honor_dataset_windows=True)

        subset_id = state.subset_id or state.dataset
        variate_indices = state.variate_indices
        if variate_indices is None:
            variate_indices = generate_dataset_job(state.dataset)["variate_indices"]
        subset_meta = state.data_subset_resolved or {}
        train_stride = int(subset_meta.get("train_stride", state.window_stride))
        test_stride = int(subset_meta.get("test_stride", 1))

        ft_guidance_ckpt = state.guidance_finetune_ckpt
        if not ft_guidance_ckpt or not os.path.exists(ft_guidance_ckpt):
            ft_guidance_ckpt = state.default_guidance_finetune_ckpt_path()
        needs_guidance = state.needs_guidance
        if needs_guidance and not os.path.exists(ft_guidance_ckpt):
            raise RuntimeError(
                f"{self.name} requires finetuned guidance ({state.guidance_type}), got: {ft_guidance_ckpt}"
            )
        if not needs_guidance:
            ft_guidance_ckpt = ""
        if self.stage == "fine" and not state.diffusion_coarse_finetune_ckpt:
            raise RuntimeError("fine staged tuning requires completed coarse best model first")
        if self.stage == "patch_refine" and not state.diffusion_coarse_finetune_ckpt:
            raise RuntimeError("patch_refine staged tuning requires completed coarse best model first")
        diff_ckpt = self._pretrained_ckpt(state)

        device = state.resolve_device()
        n_iv = len(variate_indices)
        train_ds, val_ds, _, norm_stats = load_dataset(
            state.dataset,
            variate_indices,
            stride=train_stride,
            test_stride=test_stride,
            ordinal_tie_atol=float(state.ordinal_tie_atol),
            use_ordinal_window_norm=state.use_ordinal_window_norm,
        )
        if (
            self.stage == "patch_refine"
            and bool(getattr(state, "patch_refine_unique_segments", False))
        ):
            from models.diffusion_tsf.patch_refine_segments import (
                wrap_timeseries_as_unique_segments,
            )

            seg_stride = max(1, int(train_stride))
            train_ds = wrap_timeseries_as_unique_segments(
                train_ds,
                patch_width=int(getattr(state, "patch_refine_patch_width", 8)),
                segment_stride=seg_stride,
                series_id=0,
            )
            val_ds = wrap_timeseries_as_unique_segments(
                val_ds,
                patch_width=int(getattr(state, "patch_refine_patch_width", 8)),
                segment_stride=seg_stride,
                series_id=1,
            )
            logger.info(
                "  [%s] unique patch segments enabled "
                "(segment_stride=%d train=%d val=%d prev_dropout=%.2f)",
                self.name,
                seg_stride,
                len(train_ds),
                len(val_ds),
                float(getattr(state, "patch_refine_prev_cond_dropout", 0.5)),
            )
        if norm_stats.get("ordinal_ladder") is not None:
            state.extra["global_ordinal_ladder"] = norm_stats["ordinal_ladder"]
        if state.smoke_test:
            train_ds = Subset(train_ds, list(range(min(4, len(train_ds)))))
            val_ds = Subset(val_ds, list(range(min(2, len(val_ds)))))
        logger.info(
            "  [%s] train/val windows=%d/%d",
            self.name, len(train_ds), len(val_ds),
        )

        selection_metric = str(self.get("hp_objective") or "diffusion_val").lower()
        _decoded = {"anchor_mse"}
        if selection_metric not in {"diffusion_val"} | _decoded:
            raise ValueError(
                f"hp_objective/selection_metric must be diffusion_val or anchor_mse, "
                f"got {selection_metric!r}"
            )
        anchor_val_ds = None
        if selection_metric in _decoded:
            frac = float(self.get("hp_anchor_eval_val_fraction", 0.5))
            if state.smoke_test:
                frac = 1.0
            anchor_val_ds = _fraction_subset(val_ds, frac, int(state.seed))
            logger.info(
                "  [%s] selection_metric=%s on %d/%d val windows (fraction=%.3f)",
                self.name, selection_metric, len(anchor_val_ds), len(val_ds), frac,
            )

        from models.diffusion_tsf.pipeline.phase_diagnostics import run_phase_start_diagnostics
        from models.diffusion_tsf.pipeline.visualize_utils import _load_staged_diffusion_from_ckpt

        if not diff_ckpt:
            logger.info(
                "  [%s] phase-start diagnostics skipped (from_random_init / no pretrain ckpt)",
                self.name,
            )
        else:
            try:
                probe_model, _ = _load_staged_diffusion_from_ckpt(
                    ckpt_path=diff_ckpt,
                    stage=self.stage,
                    itrans_ckpt_path=ft_guidance_ckpt,
                    n_vars=n_iv,
                    device=device,
                    guidance_type=state.guidance_type,
                )
                ckpt_info = []
                if ft_guidance_ckpt and os.path.exists(ft_guidance_ckpt):
                    ckpt_info.append(
                        {
                            "kind": state.guidance_type,
                            "path": ft_guidance_ckpt,
                            "n_variates": n_iv,
                            "lookback": int(state.lookback_length),
                            "horizon": int(state.forecast_length),
                        }
                    )
                ckpt_info.append(
                    {
                        "kind": f"diffusion_pretrain_{self.stage}",
                        "path": diff_ckpt,
                        "n_variates": n_iv,
                        "lookback": int(state.lookback_length),
                        "horizon": int(state.forecast_length),
                    }
                )
                phase_start = run_phase_start_diagnostics(
                    state,
                    phase_name=self.name,
                    models=[probe_model],
                    model_labels=[f"diffusion_{self.stage}"],
                    datasets=[train_ds],
                    dataset_prefixes=["dataset"],
                    ckpt_info=ckpt_info,
                )
                wandb_utils.log_phase_diagnostics_result({"summary": phase_start})
                del probe_model
            except Exception as e:
                logger.warning("[%s] phase-start diagnostics failed: %s", self.name, e, exc_info=True)
        # Diagnostics may mutate module globals (e.g. DIT_PATCH_SIZE); restore from state.
        patch_stage_globals(pipeline_mod, state, self.stage, honor_dataset_windows=True)

        ds_lb, ds_hz = dataset_window_lengths(state.dataset)
        micro_ceiling = configured_max_diffusion_batch(state, state.smoke_test)
        default_micro = configured_finetune_micro_batch(state, state.smoke_test)
        logger.info(
            "  [%s] finetune micro-batch default=%d ceiling=%d (YAML; no GPU probe)",
            self.name,
            default_micro,
            micro_ceiling,
        )

        accum_mult = state.extra.get("diffusion_effective_batch_multiplier")
        if accum_mult is not None and float(accum_mult) > 1.0:
            batch_plan = resolve_diffusion_batch_and_accum(default_micro, accum_mult)
            logger.info(
                "  [%s] grad accum: base_micro=%d multiplier=%s -> micro=%d accum=%d effective=%d",
                self.name,
                default_micro,
                accum_mult,
                batch_plan["batch_size"],
                batch_plan["gradient_accumulation_steps"],
                batch_plan["effective_batch_size"],
            )

        reuse_from = self.get("reuse_tuned_params_from")
        by_dataset = self.get("reuse_tuned_params_from_by_dataset") or {}
        if isinstance(by_dataset, dict) and state.dataset in by_dataset:
            reuse_from = by_dataset[state.dataset]
        retrain_reused = bool(reuse_from) and bool(self.get("retrain", False))

        max_epochs = int(self.require("max_epochs"))
        patience = int(self.require("patience"))
        if state.smoke_test:
            max_epochs = patience = 1

        subset_dir = _stage_subset_dir(state, self.stage)
        from models.diffusion_tsf.train_multivariate_pipeline import ensure_checkpoint_dir

        ensure_checkpoint_dir(final_ckpt := _stage_best_ckpt(state, self.stage))
        trials_dir = os.path.join(subset_dir, "_trials")
        ensure_checkpoint_dir(os.path.join(trials_dir, "_trial.pt"))

        reuse_meta: Dict[str, Any] = {}
        hp_best_val_loss: Optional[float] = None
        best_trial_num = -1
        final_val = float("nan")
        final_epoch = 0
        search_space = "lr_only"
        refit_completed = False
        pending_refit = False
        best_params: Dict[str, Any] = {}
        meta_path = os.path.join(subset_dir, "metadata.json")

        if (
            not reuse_from
            and self.get("refit_best_max_epochs") is not None
            and os.path.isfile(final_ckpt)
            and os.path.isfile(meta_path)
        ):
            try:
                with open(meta_path, encoding="utf-8") as f:
                    prev_meta = json.load(f)
            except Exception as e:
                prev_meta = {}
                logger.warning("  [%s] could not read %s for pending refit: %s", self.name, meta_path, e)
            if prev_meta.get("tuned_params") and not prev_meta.get("refit_completed"):
                best_params = dict(prev_meta["tuned_params"])
                hp_best_val_loss = float(
                    prev_meta.get("hp_best_val_loss")
                    or prev_meta.get("best_val_loss")
                    or float("nan")
                )
                best_trial_num = int(prev_meta.get("best_trial", -1))
                search_space = str(
                    prev_meta.get("search_space") or self.get("search_space") or "lr_only"
                ).lower()
                pending_refit = True
                logger.info(
```
```python
# models/diffusion_tsf/pipeline/phases/staged_diffusion_finetune_hp.py:1550-1750
                logger.info(
                    "  [%s] resuming pending refit (search already done, trial=%d)",
                    self.name,
                    best_trial_num,
                )

        if reuse_from:
            best_params, source_dir, reuse_meta = _load_reused_stage_params(
                state, stage=self.stage, subset_id=subset_id, source_config=str(reuse_from),
            )
            search_space = str(reuse_meta.get("search_space") or self.get("search_space") or "lr_only").lower()
            tuned_bs = int(best_params.get("batch_size", default_micro))
            best_params["batch_size"] = min(tuned_bs, micro_ceiling)
            best_params = _with_state_anchor_params(best_params, state)
            if retrain_reused:
                final_val, final_epoch = self._train_once(
                    state=state,
                    train_ds=train_ds,
                    val_ds=val_ds,
                    params=best_params,
                    pretrained_path=diff_ckpt,
                    guidance_checkpoint=ft_guidance_ckpt,
                    device=device,
                    variate_indices=variate_indices,
                    ckpt_path=final_ckpt,
                    max_epochs=max_epochs,
                    patience=patience,
                    trial=None,
                    selection_metric=selection_metric,
                    anchor_val_ds=anchor_val_ds,
                )
                hp_best_val_loss = float(final_val)
                logger.info(
                    "  [%s] retrained %s with reused HP from %s (lr=%s)",
                    self.name,
                    self.stage,
                    source_dir,
                    best_params.get("learning_rate"),
                )
            else:
                src_best = os.path.join(source_dir, subset_id, self.stage, "best.pt")
                if not os.path.exists(src_best):
                    raise FileNotFoundError(f"Missing reused staged checkpoint: {src_best}")
                if not os.path.exists(final_ckpt):
                    import shutil
                    shutil.copy2(src_best, final_ckpt)
                hp_best_val_loss = float(
                    reuse_meta.get("best_val_loss")
                    or reuse_meta.get("hp_best_val_loss")
                    or float("nan")
                )
                final_val = hp_best_val_loss
                final_epoch = int(reuse_meta.get("best_epoch", 0))
                logger.info("  [%s] reused %s from %s", self.name, self.stage, source_dir)
        elif pending_refit:
            logger.info("  [%s] skipping Optuna; using cached search winner for refit", self.name)
        else:
            n_trials = int(self.require("n_trials"))
            if state.smoke_test:
                n_trials = 1
            search_space = str(self.require("search_space")).lower()
            if search_space not in {
                "lr_only",
                "lr_eff_batch_univariate",
                "lr_eff_batch_univariate_ema",
                "fixed",
            }:
                raise ValueError(f"Unknown staged diffusion search_space={search_space!r}")
            if search_space in {
                "lr_eff_batch_univariate",
                "lr_eff_batch_univariate_ema",
            }:
                required = ["hp_lr_min", "hp_lr_max", "effective_univariate_batch_grid"]
                if search_space == "lr_eff_batch_univariate_ema":
                    required.append("ema_decay_grid")
                for key in required:
                    if self.get(key) is None:
                        raise ValueError(
                            f"search_space={search_space} requires phase {key}"
                        )
            if search_space == "fixed" and not (
                self.get("fixed_tuned_params") or self.get("fixed_tuned_params_by_dataset")
            ):
                raise ValueError(
                    "search_space=fixed requires fixed_tuned_params "
                    "and/or fixed_tuned_params_by_dataset in phase YAML"
                )

            if search_space == "fixed":
                best_params = _build_fixed_hp_params(
                    state, default_micro, state.smoke_test, self.overrides,
                )
                best_params = _with_state_anchor_params(best_params, state)
                final_val, final_epoch = self._train_once(
                    state=state,
                    train_ds=train_ds,
                    val_ds=val_ds,
                    params=best_params,
                    pretrained_path=diff_ckpt,
                    guidance_checkpoint=ft_guidance_ckpt,
                    device=device,
                    variate_indices=variate_indices,
                    ckpt_path=final_ckpt,
                    max_epochs=max_epochs,
                    patience=patience,
                    trial=None,
                    selection_metric=selection_metric,
                    anchor_val_ds=anchor_val_ds,
                )
                hp_best_val_loss = float(final_val)
                best_trial_num = 0
                logger.info(
                    "  [%s] fixed HP train done: val=%.4f epoch=%d lr=%.2e micro_bs=%d",
                    self.name,
                    hp_best_val_loss,
                    final_epoch,
                    float(best_params.get("learning_rate", 0.0)),
                    int(best_params.get("batch_size", 1)),
                )
            else:
                from models.diffusion_tsf.pipeline.optuna_parallel import run_optuna_study

                phase = self

                def objective_builder(_worker_id: int):
                    dev = state.resolve_device()
                    worker_guidance = None
                    if ft_guidance_ckpt:
                        worker_guidance = load_wrapped_guidance(
                            ft_guidance_ckpt,
                            n_iv,
                            dev,
                            guidance_type=state.guidance_type,
                            dataset_lookback=ds_lb,
                            dataset_horizon=ds_hz,
                        )
                    if diff_ckpt:
                        worker_pretrained = torch.load(
                            diff_ckpt, map_location=dev, weights_only=False,
                        )["model_state_dict"]
                    else:
                        worker_pretrained = None

                    def objective(trial):
                        plan_batch = (
                            default_micro if search_space == "lr_only" else micro_ceiling
                        )
                        params = _suggest_staged_params(
                            trial,
                            state,
                            plan_batch,
                            state.smoke_test,
                            search_space=search_space,
                            phase_overrides=phase.overrides,
                        )
                        trial.set_user_attr("full_params", dict(params))
                        if search_space in {
                            "lr_eff_batch_univariate",
                            "lr_eff_batch_univariate_ema",
                        } and not state.smoke_test:
                            micro = int(params["batch_size"])
                            accum = int(params.get("gradient_accumulation_steps", 1))
                            if micro < 1 or accum > 2048:
                                raise RuntimeError(
                                    f"Degenerate batch plan micro_bs={micro} accum={accum} "
                                    f"(effective={micro * accum}); stale Optuna journal or planner bug"
                                )
                        logger.info(
                            "  [%s] Optuna trial %d/%d suggested lr=%.2e micro_bs=%d "
                            "accum=%d effective_bs=%d univariate_U=%s (target_U=%s) g=%s",
                            phase.name,
                            trial.number + 1,
                            n_trials,
                            float(params["learning_rate"]),
                            int(params["batch_size"]),
                            int(params.get("gradient_accumulation_steps", 1)),
                            int(params.get("effective_batch_size", params["batch_size"])),
                            params.get("effective_univariate_batch", "-"),
                            params.get("target_univariate_batch", "-"),
                        )
                        trial_ckpt = os.path.join(
                            trials_dir, f"trial_{trial.number}_best.pt",
                        )
                        trial_t0 = time.perf_counter()
                        try:
                            best_val, best_ep = phase._train_once(
                                state=state,
                                train_ds=train_ds,
                                val_ds=val_ds,
                                params=params,
                                pretrained_path=diff_ckpt,
                                guidance_checkpoint=ft_guidance_ckpt,
                                device=dev,
                                variate_indices=variate_indices,
                                ckpt_path=trial_ckpt,
                                max_epochs=max_epochs,
                                patience=patience,
                                trial=trial,
                                guidance=worker_guidance,
                                pretrained_state_dict=worker_pretrained,
                                selection_metric=selection_metric,
```
```python
# models/diffusion_tsf/pipeline/phases/staged_diffusion_finetune_hp.py:1750-1998
                                selection_metric=selection_metric,
                                anchor_val_ds=anchor_val_ds,
                            )
                        except torch.cuda.OutOfMemoryError:
                            logger.warning(
                                "  [%s] trial %d OOM (batch=%s), pruning",
                                phase.name, trial.number, params.get("batch_size"),
                            )
                            # _train_once's finally deletes the model, but the
                            # CUDA caching allocator only releases after the
                            # frame unwinds — empty here or the next trial
                            # starts already near the L40S ceiling.
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            raise TrialPruned() from None
                        except TrialPruned:
                            logger.info(
                                "  [%s] Optuna trial %d pruned after %.1fs",
                                phase.name, trial.number, time.perf_counter() - trial_t0,
                            )
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            raise
                        trial.set_user_attr("best_epoch", best_ep)
                        logger.info(
                            "  [%s] Optuna trial %d finished best_val=%.4f best_epoch=%d time=%.1fs",
                            phase.name,
                            trial.number,
                            best_val,
                            best_ep,
                            time.perf_counter() - trial_t0,
                        )
                        return best_val

                    return objective

                def _retain_complete_trial_ckpts(study, _trial) -> None:
                    # Keep every COMPLETE trial weight so --resume can still
                    # promote whichever trial wins after more trials land.
                    # Drop only pruned/failed mid-run checkpoints.
                    from optuna.trial import TrialState

                    keep_nums = {
                        int(t.number)
                        for t in study.get_trials(
                            deepcopy=False, states=(TrialState.COMPLETE,),
                        )
                    }
                    keep_names = {f"trial_{n}_best.pt" for n in keep_nums}
                    keep_names |= {f"_diff_ft_trial_{n}_best.pt" for n in keep_nums}
                    for trial_dir in (trials_dir, subset_dir):
                        if not os.path.isdir(trial_dir):
                            continue
                        for fn in os.listdir(trial_dir):
                            if not fn.endswith("_best.pt"):
                                continue
                            if not (
                                fn.startswith("trial_")
                                or fn.startswith("_diff_ft_trial_")
                            ):
                                continue
                            if fn in keep_names:
                                continue
                            path = os.path.join(trial_dir, fn)
                            try:
                                os.remove(path)
                            except OSError:
                                pass

                logger.info(
                    "  [%s] Optuna study start: n_trials=%d max_epochs=%d patience=%d",
                    self.name, n_trials, max_epochs, patience,
                )
                study_t0 = time.perf_counter()
                study = run_optuna_study(
                    study_name=f"{state.experiment_name}-{self.stage}-hp",
                    checkpoint_dir=subset_dir,
                    n_trials=n_trials,
                    parallel_workers=state.parallel_optuna_workers,
                    direction="minimize",
                    objective_builder=objective_builder,
                    sampler=TPESampler(seed=state.seed, multivariate=True, group=True),
                    pruner=HyperbandPruner(
                        min_resource=1, max_resource=max_epochs, reduction_factor=3,
                    ),
                    sampler_seed=state.seed,
                    callbacks=[_retain_complete_trial_ckpts],
                )
                try:
                    best_trial = study.best_trial
                except ValueError as e:
                    raise RuntimeError(
                        f"All {self.stage} diffusion HP trials failed for {subset_id}"
                    ) from e

                best_params = dict(best_trial.user_attrs.get("full_params") or best_trial.params)
                best_params.setdefault("min_snr_gamma", 5.0)
                best_params.setdefault(
                    "max_scale",
                    float(state.max_scale_by_dataset.get(state.dataset, state.max_scale)),
                )
                best_params = _with_state_anchor_params(best_params, state)
                hp_best_val_loss = float(study.best_value)
                best_trial_num = int(best_trial.number)
                final_epoch = int(best_trial.user_attrs.get("best_epoch", 0))
                logger.info(
                    "  [%s] Optuna study done in %.1fs: best_trial=%d best_val=%.4f best_epoch=%d lr=%.2e",
                    self.name,
                    time.perf_counter() - study_t0,
                    best_trial_num,
                    hp_best_val_loss,
                    final_epoch,
                    float(best_params.get("learning_rate", 0.0)),
                )

                import shutil
                src = _resolve_best_trial_ckpt(
                    study, trials_dir, subset_dir, best_trial_num,
                )
                shutil.copy2(src, final_ckpt)
                if not os.path.isfile(final_ckpt):
                    raise RuntimeError(f"Failed to promote best trial checkpoint to {final_ckpt}")
                final_val = hp_best_val_loss
                _cleanup_trial_ckpts(trials_dir, subset_dir, keep=src)

        if not reuse_from and self.get("refit_best_max_epochs") is not None:
            if hp_best_val_loss is None:
                raise RuntimeError(f"{self.name}: refit_best_max_epochs set but no HP winner available")
            best_params, final_val, final_epoch, refit_completed = self._refit_best_if_configured(
                state=state,
                train_ds=train_ds,
                val_ds=val_ds,
                best_params=best_params,
                diff_ckpt=diff_ckpt,
                ft_guidance_ckpt=ft_guidance_ckpt,
                device=device,
                variate_indices=variate_indices,
                final_ckpt=final_ckpt,
                hp_best_val_loss=float(hp_best_val_loss),
                best_trial_num=best_trial_num,
                search_space=search_space,
                search_max_epochs=max_epochs,
                search_patience=patience,
                subset_dir=subset_dir,
                subset_id=subset_id,
                subset_meta=subset_meta,
                norm_stats=norm_stats,
                selection_metric=selection_metric,
                anchor_val_ds=anchor_val_ds,
            )

        meta_out: Dict[str, Any] = {
            "subset_id": subset_id,
            "dataset_name": state.dataset,
            "variate_indices": variate_indices,
            "data_subset": subset_meta,
            "norm_mean": norm_stats["mean"].tolist(),
            "norm_std": norm_stats["std"].tolist(),
            "tuned_params": best_params,
            "best_trial": best_trial_num,
            "hp_best_val_loss": hp_best_val_loss,
            "best_val_loss": float(final_val),
            "best_selection_score": float(final_val),
            "best_epoch": int(final_epoch),
            "diffusion_stage": self.stage,
            "staged_representation": state.staged_representation,
            "search_space": search_space,
            "selection_metric": selection_metric,
            "max_epochs": (
                int(self.get("refit_best_max_epochs"))
                if refit_completed
                else max_epochs
            ),
            "patience": (
                int(self.get("refit_best_patience", self.get("refit_best_max_epochs")))
                if refit_completed
                else patience
            ),
            "search_max_epochs": max_epochs,
            "search_patience": patience,
        }
        if self.get("refit_best_max_epochs") is not None:
            meta_out["refit_best_max_epochs"] = int(self.get("refit_best_max_epochs"))
            meta_out["refit_completed"] = bool(refit_completed)
        if reuse_from:
            meta_out.update({
                "reuse_tuned_params_from": str(reuse_from),
                "retrain_reused_params": bool(self.get("retrain", False)),
                "reused_max_scale_policy": best_params.get("max_scale"),
                "reused_max_scale_previous": reuse_meta.get("reused_max_scale_previous"),
            })
        with open(os.path.join(subset_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta_out, f, indent=2, sort_keys=True)

        if self.stage == "coarse":
            state.diffusion_coarse_finetune_ckpt = final_ckpt
            state.coarse_finetune_best_params = best_params
        elif self.stage == "fine":
            state.diffusion_fine_finetune_ckpt = final_ckpt
            state.fine_finetune_best_params = best_params
        elif self.stage == "patch_refine":
            state.diffusion_patch_refine_finetune_ckpt = final_ckpt
            state.patch_refine_finetune_best_params = best_params
        else:
            raise ValueError(f"Unknown diffusion stage {self.stage!r}")

        wandb_utils.log_summary({
            f"hp/{self.stage}_diff_ft_best_val_loss": final_val,
            f"hp/{self.stage}_diff_ft_hp_best_val_loss": hp_best_val_loss,
            f"hp/{self.stage}_diff_ft_best_trial": best_trial_num,
            f"hp/{self.stage}_diff_ft_best_lr": best_params.get("learning_rate"),
            f"hp/{self.stage}_diff_ft_batch_size": best_params.get("batch_size"),
            f"hp/{self.stage}_diff_ft_effective_univariate_batch": best_params.get(
                "effective_univariate_batch"
            ),
            f"hp/{self.stage}_diff_ft_target_univariate_batch": best_params.get(
                "target_univariate_batch"
            ),
            f"hp/{self.stage}_diff_ft_max_scale": best_params.get("max_scale"),
            f"hp/{self.stage}_diff_ft_refit_completed": bool(refit_completed),
        })

        self._log_post_finetune_viz_and_diagnostics(
            state,
            final_ckpt=final_ckpt,
            best_params=best_params,
            train_ds=train_ds,
        )

        return state


class CoarseDiffusionFinetuneHPPhase(_BaseStagedDiffusionFinetuneHPPhase):
    """Full-horizon coarse CDF stage."""
    name = "diffusion_coarse_finetune_hp"
    stage = "coarse"


class FineDiffusionFinetuneHPPhase(_BaseStagedDiffusionFinetuneHPPhase):
    """Residual fine stage (alt second stage; not used when patch_refine is on)."""
    name = "diffusion_fine_finetune_hp"
    stage = "fine"


class PatchRefineDiffusionFinetuneHPPhase(_BaseStagedDiffusionFinetuneHPPhase):
    """Tall-canvas boundary-patch second stage (live path with use_patch_refine_stage)."""
    name = "diffusion_patch_refine_finetune_hp"
    stage = "patch_refine"
```

Shared micro-loop / AMP / ckpt helpers used by pretrain + finetune:

```python
# models/diffusion_tsf/pipeline/train/diffusion_loop.py:1-84
"""Shared diffusion train/val epoch loops.

Used by HP / pretrain phases after they build a DiffusionTSF + loaders.
Loss comes from ``model.get_loss(past, future)``; AMP context is the module
global USE_AMP via checkpointing.amp_context.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch.utils.data import DataLoader

from models.diffusion_tsf.pipeline.train.checkpointing import amp_context


def train_diffusion_epoch(
    model,
    train_loader: DataLoader,
    device: torch.device,
    optimizer,
    *,
    accum_steps: int = 1,
    clip_grad: float = 1.0,
    set_loader_mode: Optional[Callable] = None,
    set_training_epoch: Optional[Callable] = None,
    epoch: Optional[int] = None,
    ema=None,
) -> float:
    if set_training_epoch is not None and epoch is not None:
        set_training_epoch(train_loader, epoch)

    model.train()
    if set_loader_mode is not None:
        set_loader_mode(model, train_loader, eval_mode=False)

    total_loss = 0.0
    n_batches = 0
    optimizer.zero_grad(set_to_none=True)
    accum_steps = max(1, int(accum_steps))

    for batch_idx, (past, future) in enumerate(train_loader):
        past, future = past.to(device), future.to(device)
        with amp_context():
            loss = model.get_loss(past, future) / accum_steps
        loss.backward()
        if (batch_idx + 1) % accum_steps == 0:
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)
        total_loss += float(loss.item()) * accum_steps
        n_batches += 1

    # Flush a partial accumulation window at epoch end (common Optuna microbatch pattern).
    if accum_steps > 1 and len(train_loader) % accum_steps != 0:
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if ema is not None:
            ema.update(model)

    return total_loss / max(n_batches, 1)


def validate_diffusion_epoch(
    model,
    val_loader: DataLoader,
    device: torch.device,
    *,
    set_loader_mode: Optional[Callable] = None,
) -> float:
    model.eval()
    if set_loader_mode is not None:
        set_loader_mode(model, val_loader, eval_mode=True)

    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for past, future in val_loader:
```
```python
# models/diffusion_tsf/pipeline/train/checkpointing.py:1-103
"""Checkpoint I/O and early stopping for diffusion train loops.

``save_checkpoint`` strips frozen ``guidance_model.*`` weights — guidance is
loaded separately at runtime. Atomic write via ``.tmp`` + ``os.replace``.
"""

from __future__ import annotations

import os

import torch


class EarlyStopping:
    def __init__(self, patience: int = 25, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def amp_context():
    # Reads USE_AMP from the legacy module globals (set by patch_globals).
    from models.diffusion_tsf import train_multivariate_pipeline as m

    if m.USE_AMP and torch.cuda.is_available():
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    from contextlib import nullcontext

    return nullcontext()


def ensure_checkpoint_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if not parent or os.path.isdir(parent):
        return
    if os.path.isfile(parent):
        raise FileExistsError(
            f"checkpoint parent exists as a file, not a directory: {parent}"
        )
    try:
        os.makedirs(parent, exist_ok=True)
    except FileExistsError:
        if not os.path.isdir(parent):
            raise


def _diffusion_state_dict_without_guidance(model) -> dict:
    """Drop frozen guidance weights — guidance is loaded separately at runtime."""
    return {
        k: v
        for k, v in model.state_dict().items()
        if not k.startswith("guidance_model.")
    }


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, config, path, extra=None):
    ensure_checkpoint_dir(path)
    ckpt = {
        "epoch": epoch,
        "model_state_dict": _diffusion_state_dict_without_guidance(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "config": config,
    }
    if extra:
        ckpt.update(extra)
    tmp_path = f"{path}.tmp"
    try:
        torch.save(ckpt, tmp_path)
        os.replace(tmp_path, path)
    except OSError as e:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        err = getattr(e, "errno", None)
        if err in (28, 122) or "quota" in str(e).lower() or "no space" in str(e).lower():
            raise RuntimeError(
                f"Disk quota/space exhausted while saving {path}. "
                "Free scratch (old results/ckpts, wandb, trial_*.pt) then --resume."
            ) from e
        raise
    except RuntimeError as e:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        if "quota" in str(e).lower() or "no space" in str(e).lower():
```
```python
# models/diffusion_tsf/pipeline/training_helpers.py:1-54
"""Pure helpers for synthetic pool sizing.

Used by staged pretrain / synth HP to clamp RealTS pool size under smoke and
``synthetic_samples_*`` caps from YAML training section. Thin wrappers on
``train_multivariate_pipeline`` just bind module globals into these functions.
"""

from __future__ import annotations

from typing import Optional, Tuple


def resolve_synthetic_params(
    requested_n: int,
    requested_cap: int,
    smoke_test: bool,
    *,
    samples_cap: Optional[int],
    samples_min: int,
) -> Tuple[int, int]:
    if smoke_test:
        return 4, 1

    n = requested_n
    cap = requested_cap

    if samples_cap is not None:
        total = n * cap
        if total > samples_cap:
            n = max(samples_min, samples_cap // cap)
            if n * cap > samples_cap:
                cap = max(1, samples_cap // n)

    return int(n), int(cap)


def resolve_pretrain_virtual_dataset_size(
    smoke_test: bool,
    *,
    pretrain_epochs: int,
    pretrain_diffusion_max_epochs: int,
    pretrain_synthetic_override: Optional[int],
    samples_cap: Optional[int],
    samples_min: int,
) -> int:
    if smoke_test:
        return 4
    if pretrain_synthetic_override is not None:
        return max(4, int(pretrain_synthetic_override))

    steps = 32 + 48 * pretrain_epochs
    steps = max(64, steps)
    ref_bs = 8
    requested_n = steps * ref_bs
```

## 8. Model core — DiffusionTSF

Normalize → hard CDF encode → DiT binary diffusion → decode. Dispatch on `diffusion_stage`.

```python
# models/diffusion_tsf/diffusion_model.py:1-128
"""Binary CDF diffusion TSF: the model object training and eval call into.

Walkthrough order for staged runs:
  forward/generate dispatch on config.diffusion_stage
    -> coarse/fine: _forward_binary_staged / _generate_binary_staged
    -> patch_refine: _forward_binary_patch_refine / _generate_binary_patch_refine
    -> joint (legacy single-map): _forward_binary_factorized / _generate_binary_factorized

Coarse is full-horizon low-res CDF. Fine is the residual second stage (alt).
Patch refine is the live second stage: tall canvas, boundary-centered crops.
Hard CDF maps, XOR/BCE bit-flip noise, FactorizedDiT (one variate per BV row).
Optional stationary-flat anchor at max noise. Decode hops live in preprocessing.py
and patch_refine.py; the DiT itself is dit.py; the bit-flip loop is diffusion.py.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from .config import DiffusionTSFConfig
from .preprocessing import TimeSeriesTo2D
from .diffusion import BinaryDiffusionScheduler
from .ordinal_window_norm import OrdinalLadder, ordinal_decode, ordinal_encode
from .guidance import GuidanceModel
from .dit import FactorizedDiT

logger = logging.getLogger(__name__)


class DiffusionTSF(nn.Module):
    """Owns encode/noise/DiT/decode for one stage (or joint). Pipeline phases swap stages."""

    def __init__(
        self,
        config: DiffusionTSFConfig,
        guidance_model: Optional[Union[GuidanceModel, nn.Module]] = None,
    ):
        # Builds TimeSeriesTo2D + FactorizedDiT + BinaryDiffusionScheduler.
        # Channel counts come from config properties (backbone_in_channels, visual_cond_channels).
        super().__init__()
        self.config = config

        needs_guidance_model = config.use_guidance_channel or not config.disable_cross_attention
        if needs_guidance_model and guidance_model is None:
            raise ValueError(
                "A guidance model is required for forecast channels or cross-variate "
                "encoder tokens; none was provided."
            )

        self.to_2d = TimeSeriesTo2D(
            height=config.image_height,
            max_scale=config.max_scale,
        )
        self.guidance_model = guidance_model if needs_guidance_model else None

        backbone_in_channels = config.backbone_in_channels
        is_patch_refine = config.diffusion_stage == "patch_refine"
        dit_patch = config.dit_patch_size
        cond_patch = config.dit_cond_patch_size
        if is_patch_refine and cond_patch is None:
            cond_patch = (8, 8)
        # use_scale_embedding / enable_cross_scale_attention stay hard-off; see dit.py.
        self.noise_predictor = FactorizedDiT(
            in_channels=backbone_in_channels,
            cond_channels=config.visual_cond_channels,
            out_channels=config.dit_out_channels,
            image_height=config.image_height,
            patch_size=dit_patch,
            embed_dim=config.dit_embed_dim,
            depth=config.dit_depth,
            num_heads=config.dit_num_heads,
            mlp_ratio=config.dit_mlp_ratio,
            dropout=config.dit_dropout,
            context_dim=config.context_embedding_dim,
            gradient_checkpointing=config.use_gradient_checkpointing,
            cond_patch_size=cond_patch,
            use_scale_embedding=False,
            enable_cross_scale_attention=False,
            use_variate_embedding=(
                config.use_variate_embedding
                and config.variate_factorized
                and config.num_variables > 1
            ),
            max_variates=max(config.num_variables, 512),
            use_patch_abs_embedding=is_patch_refine,
            max_coarse_bins=max(16, int(config.coarse_image_height)),
            max_horizon_steps=max(
                1024,
                int(config.dataset_forecast_length or 0),
                int(config.forecast_length),
            ),
        )

        self._ctx_token_variate_ids: Optional[torch.Tensor] = None
        if config.guidance_type != "patch_decoder":
            raise ValueError(
                f"Only guidance_type='patch_decoder' is supported; got {config.guidance_type!r}"
            )
        self.context_encoder = None

        self.binary_scheduler = BinaryDiffusionScheduler(
            num_steps=config.binary_num_steps,
            beta_start=config.binary_beta_start,
            beta_end=config.binary_beta_end,
            schedule_type=config.binary_noise_schedule,
        )

        logger.debug("DiffusionTSF initialized:")
        logger.debug(
            "  Variables: %d (%s)",
            config.num_variables,
            "multivariate" if config.num_variables > 1 else "univariate",
        )
        logger.debug(
            "  Lookback: %d, Forecast: %d",
            config.lookback_length,
            config.forecast_length,
        )
        logger.debug(
            "  Image size: %d x %d (H x W)",
            config.image_height,
            config.forecast_length,
        )
```
```python
# models/diffusion_tsf/diffusion_model.py:299-366
    def _normalize_sequence(
        self,
        past: torch.Tensor,
        future: Optional[torch.Tensor] = None,
        *,
        apply_ood_shift: Optional[bool] = None,
        data_is_ranked: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Optional[OrdinalLadder]]]:
        """First hop in every forward/generate: ordinal ranks or lookback mean/std window norm."""
        if apply_ood_shift is None:
            apply_ood_shift = bool(getattr(self, "_ordinal_apply_ood_shift", False))
        if data_is_ranked is None:
            data_is_ranked = bool(getattr(self, "_ordinal_input_is_ranked", False))
        if self.config.use_ordinal_window_norm:
            ladder = self.config.ordinal_ladder
            if ladder is None:
                raise ValueError("ordinal_ladder is required when use_ordinal_window_norm=True")
            if data_is_ranked:
                batch_size = past.shape[0] if past.dim() == 3 else 1
                ladder_b = ladder.expand_batch(batch_size)
                center = torch.zeros_like(past[..., :1])
                std = torch.ones_like(past[..., :1])
                return past, future, (center, std, ladder_b)
            past_ord, future_ord, ladder_b, ood_shift = ordinal_encode(
                past,
                future,
                ladder=ladder,
                apply_ood_shift=apply_ood_shift,
                causal_only=bool(self.config.ordinal_ood_shift_causal_only),
            )
            center = torch.zeros_like(past[..., :1])
            std = torch.ones_like(past[..., :1])
            return past_ord, future_ord, (center, std, ladder_b, ood_shift)

        if not self.config.use_window_normalization:
            mean = torch.zeros_like(past[..., :1])
            std = torch.ones_like(past[..., :1])
            return past, future, (mean, std, None)
        center = self._window_norm_center(past)
        past_std = past.std(dim=-1, keepdim=True)
        threshold = float(self.config.window_norm_low_var_threshold)
        if threshold > 0.0:
            std_floor = past_std.clamp_min(self.config.window_norm_std_floor)
            per_v = self.config.window_norm_low_var_unit_std_per_variate
            default_unit = float(self.config.window_norm_low_var_unit_std)
            if per_v is not None:
                if len(per_v) != past.shape[1]:
                    raise ValueError(
                        "window_norm_low_var_unit_std_per_variate length "
                        f"{len(per_v)} != num_variables {past.shape[1]}"
                    )
                unit = torch.tensor(
                    per_v, device=past.device, dtype=past.dtype,
                ).view(1, -1, 1).expand_as(past_std)
            else:
                unit = torch.full_like(past_std, default_unit)
            low_var = past_std < threshold
            flat = past_std <= self.config.window_norm_std_floor
            std = torch.where(flat | low_var, unit, std_floor)
        else:
            std = past_std.clamp_min(self.config.window_norm_std_floor)
        past_norm = (past - center) / std
        if future is not None:
            future_norm = (future - center) / std
        else:
            future_norm = None
        return past_norm, future_norm, (center, std, None)
```
```python
# models/diffusion_tsf/diffusion_model.py:636-735
    def _encode_staged_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Coarse = full-range CDF; fine = within-bin residual CDF. See preprocessing.encode_dual_*.
        x = self._subsample_repr_time(x)
        coarse_h, fine_h = self._staged_image_heights()
        if self._uses_global_ordinal_encoding():
            vmax = self._ordinal_rank_max_tensor(x.device, dtype=x.dtype)
            coarse, fine = self.to_2d.encode_dual_heights_bounded(
                x,
                coarse_height=coarse_h,
                fine_height=fine_h,
                value_min=0.0,
                value_max_per_variate=vmax,
            )
            return {"coarse": coarse, "fine": fine}
        coarse, fine = self.to_2d.encode_dual_heights(
            x,
            coarse_height=coarse_h,
            fine_height=fine_h,
        )
        return {"coarse": coarse, "fine": fine}








    def _binary_anchor_canvas_like(self, like: torch.Tensor) -> torch.Tensor:
        """One-shot anchor input: flat Bernoulli(0.5) mean, no XOR sample.

        Training mixes this path when use_deterministic_anchor_loss; eval often
        uses sampler='anchor' which runs a single forward at t=T-1 on this canvas.
        """
        mode = self.config.binary_anchor_input_mode
        if mode != "stationary_flat":
            raise ValueError(
                f"binary_anchor_input_mode must be 'stationary_flat', got {mode!r}"
            )
        return torch.full_like(like, 0.5)

    def _binary_anchor_canvas_shape(
        self,
        shape: Tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        template = torch.empty(shape, device=device, dtype=dtype)
        return self._binary_anchor_canvas_like(template)






    def _resize_cdf_height(self, image: torch.Tensor, target_height: int) -> torch.Tensor:
        if image.shape[2] == target_height:
            return image
        flat = image.reshape(-1, 1, image.shape[2], image.shape[3])
        resized = F.interpolate(flat, size=(target_height, image.shape[3]), mode="bilinear", align_corners=False)
        return resized.reshape(image.shape[0], image.shape[1], target_height, image.shape[3])

    def _coarse_cdf_to_height(self, coarse_map: torch.Tensor, target_height: int) -> torch.Tensor:
        if coarse_map.shape[2] == target_height:
            return coarse_map
        coarse_value = self.to_2d._decode_occupancy_in_range(
            coarse_map,
            value_range=self.config.max_scale,
            cdf_decoder="mean",
        )
        return self.to_2d._encode_values_in_range(
            coarse_value,
            value_range=self.config.max_scale,
            height=target_height,
        )

    def decode_dual_from_2d(
        self,
        coarse_map: torch.Tensor,
        fine_map: torch.Tensor,
        from_diffusion: bool = False,
        decoder_method: str = "mean",
        past_seed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fine-stage decode: coarse + residual fine -> 1D (still in window/ordinal space)."""
        del past_seed  # call-site leftover; unused
        if decoder_method != "mean":
            raise ValueError(f"decoder_method must be 'mean', got {decoder_method!r}")
        if from_diffusion:
            coarse_map = (coarse_map + 1.0) / 2.0
            fine_map = (fine_map + 1.0) / 2.0
        if self._uses_global_ordinal_encoding():
            return self._decode_staged_combined_1d(coarse_map, fine_map, cdf_decoder="mean")
        return self.to_2d.decode_dual(
            coarse_map,
            fine_map,
            cdf_decoder="mean",
            squeeze_univariate=(coarse_map.shape[1] == 1),
        )
```
```python
# models/diffusion_tsf/diffusion_model.py:788-854
    def forward(
        self,
        past: torch.Tensor,
        future: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        *,
        patch_col0: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Train step entry. Stage is fixed on this model instance via config.diffusion_stage."""
        if self.config.diffusion_stage in {"coarse", "fine"}:
            return self._forward_binary_staged(past, future, t)
        if self.config.diffusion_stage == "patch_refine":
            # Training loops sample one timestep per window; expand onto crops.
            return self._forward_binary_patch_refine(
                past, future, t,
                expand_t_per_window=t is not None,
                patch_col0=patch_col0,
            )
        return self._forward_binary_factorized(past, future, t)

    @torch.no_grad()
    def generate(
        self,
        past: torch.Tensor,
        verbose: bool = False,
        decoder_method: str = "mean",
        sampler: str = "quad_t",
        num_inference_steps: Optional[int] = None,
        yield_intermediates: bool = False,
        reverse_step_indices: Optional[torch.Tensor] = None,
        snapshot_timesteps: Optional[Tuple[int, ...]] = None,
        future_coarse_2d: Optional[torch.Tensor] = None,
        future_fine_2d: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Inference entry. Fine/patch_refine need future_coarse_2d from the coarse model.

        sampler: 'quad_t' (alias 'ddim_quad') for quadratic timestep spacing,
        or 'anchor' / 'deterministic_anchor' for one-shot anchor decode.
        Plain 'ddim' is rejected (linear spacing removed).
        num_inference_steps overrides binary_sample_steps when set.
        """
        if decoder_method != "mean":
            raise ValueError(f"decoder_method must be 'mean', got {decoder_method!r}")
        if str(sampler).lower() == "ddim":
            raise ValueError(
                "sampler='ddim' was removed; use sampler='quad_t' (prob) or "
                "sampler='anchor' (point)."
            )
        steps = num_inference_steps if num_inference_steps is not None else self.config.binary_sample_steps
        gen_common = dict(
            num_steps=steps,
            verbose=verbose,
            decoder_method=decoder_method,
            sampler=sampler,
            yield_intermediates=yield_intermediates,
            reverse_step_indices=reverse_step_indices,
            snapshot_timesteps=snapshot_timesteps,
            future_coarse_2d=future_coarse_2d,
            future_fine_2d=future_fine_2d,
        )
        if self.config.diffusion_stage in {"coarse", "fine"}:
            return self._generate_binary_staged(past, **gen_common)
        if self.config.diffusion_stage == "patch_refine":
            return self._generate_binary_patch_refine(past, **gen_common)
        return self._generate_binary_factorized(past, **gen_common)
```

### 8a. Patch refine forward / generate (live second stage)

```python
# models/diffusion_tsf/diffusion_model.py:1219-1271
    def _patch_refine_geometry_knobs(self) -> Tuple[int, int, int, int]:
        return (
            int(self.config.patch_refine_canvas_height),
            int(self.config.patch_refine_patch_height),
            int(self.config.patch_refine_patch_width),
            int(self.config.patch_refine_col_stride),
        )

    def _encode_absolute_future_hir(
        self,
        future_norm: torch.Tensor,
        canvas_height: int,
    ) -> torch.Tensor:
        from .patch_refine import encode_absolute_hir_cdf

        ordinal_max = None
        if self._uses_global_ordinal_encoding():
            ordinal_max = self._ordinal_rank_max_tensor(future_norm.device, dtype=future_norm.dtype)
        return encode_absolute_hir_cdf(
            future_norm,
            canvas_height=canvas_height,
            max_scale=float(self.config.max_scale),
            ordinal_rank_max=ordinal_max,
        )

    def _decode_absolute_future_hir(self, hir_cdf: torch.Tensor) -> torch.Tensor:
        from .patch_refine import decode_absolute_hir_cdf

        ordinal_max = None
        if self._uses_global_ordinal_encoding():
            ordinal_max = self._ordinal_rank_max_tensor(hir_cdf.device, dtype=hir_cdf.dtype)
        return decode_absolute_hir_cdf(
            hir_cdf,
            max_scale=float(self.config.max_scale),
            ordinal_rank_max=ordinal_max,
        )

    def _patch_refine_lookback_cond(
        self,
        past_norm: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Full native-width stacked past coarse∥fine, never resized to the crop."""
        from .patch_refine import stack_past_coarse_fine

        past_tail_len = int(past_norm.shape[-1])
        cap = int(self.config.diffusion_lookback_cap or 0)
        if cap > 0:
            past_tail_len = min(past_tail_len, cap)
        past_tail = past_norm[..., -past_tail_len:]
        past_maps = self._encode_staged_maps(past_tail)
        cond = stack_past_coarse_fine(past_maps["coarse"], past_maps["fine"])
        return cond, past_maps
```
```python
# models/diffusion_tsf/diffusion_model.py:1272-1507
    def _forward_binary_patch_refine(
        self,
        past: torch.Tensor,
        future: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        *,
        expand_t_per_window: bool = False,
        patch_col0: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Train second stage: XOR-noise absolute hi-res CDF crops around coarse edges.

        Flow: normalize -> coarse edges + tall GT CDF -> pick crops (geometry/segments)
        -> aux channels (patch_refine.build_patch_aux_channels) -> DiT -> BCE (+ optional anchor).
        """
        from .patch_refine import (
            build_patch_aux_channels,
            expand_ctx_for_patches,
            expand_lookback_cond_for_patches,
            expand_variate_indices_for_patches,
            naive_upscale_coarse_cdf,
        )
        from .patch_refine_geometry import (
            coarse_edges_from_cdf,
            extract_patch_batch,
            select_patch_locations,
        )
        from .patch_refine_segments import (
            compress_prev_refine_32_to_16,
            extract_prev_refine_crops,
            locations_for_fixed_col0,
        )

        assert self.binary_scheduler is not None
        B = past.shape[0]
        V = self.config.num_variables
        device = past.device
        canvas_h, patch_h, patch_w, col_stride = self._patch_refine_geometry_knobs()
        coarse_h = int(self.config.coarse_image_height)
        unique = bool(getattr(self.config, "patch_refine_unique_segments", False))

        past_norm, future_norm, _stats = self._normalize_sequence(past, future)
        future_maps = self._encode_staged_maps(future_norm)
        hir_gt = self._encode_absolute_future_hir(future_norm, canvas_h)
        naive = naive_upscale_coarse_cdf(future_maps["coarse"], canvas_h)
        edges = coarse_edges_from_cdf(future_maps["coarse"], canvas_height=canvas_h)
        if unique:
            if patch_col0 is None:
                # Synth / legacy loaders: one random stride-1 crop per window.
                max_c0 = int(edges.shape[-1]) - patch_w
                patch_col0 = torch.randint(0, max_c0 + 1, (B,), device=device)
            else:
                patch_col0 = patch_col0.to(device=device, dtype=torch.long).view(B)
            locations = locations_for_fixed_col0(
                edges,
                patch_col0,
                canvas_height=canvas_h,
                patch_height=patch_h,
                patch_width=patch_w,
                hir_canvas=hir_gt,
            )
        else:
            locations = select_patch_locations(
                edges,
                canvas_height=canvas_h,
                patch_height=patch_h,
                patch_width=patch_w,
                col_stride=col_stride,
            )
        if not locations:
            raise RuntimeError("patch_refine produced zero training crops")

        target_patches = extract_patch_batch(
            hir_gt, locations, patch_height=patch_h, patch_width=patch_w,
        )
        # Full/empty crop columns have their GT transition outside this patch.
        # Mask them so training does not turn out-of-view into a boundary cue.
        target_occupancy = target_patches.sum(dim=-2, keepdim=True)
        target_visible = (target_occupancy > 0) & (target_occupancy < patch_h)
        target_visible_mask = target_visible.expand_as(target_patches).to(target_patches.dtype)
        if not bool(target_visible.any()):
            raise RuntimeError("patch_refine batch has no visible GT transitions")
        n_patches = target_patches.shape[0]
        if t is None:
            t = torch.randint(0, self.config.binary_num_steps, (n_patches,), device=device)
        elif expand_t_per_window:
            if t.numel() != B:
                raise ValueError(
                    f"expand_t_per_window requires one timestep per window "
                    f"(got {t.numel()}, B={B})"
                )
            t = torch.tensor(
                [int(t[loc.batch_index].item()) for loc in locations],
                device=device,
                dtype=torch.long,
            )
        elif t.numel() != n_patches:
            raise ValueError(
                f"timestep batch {t.numel()} incompatible with {n_patches} patches "
                "(pass expand_t_per_window=True to broadcast per-window timesteps)"
            )

        xt, zt = self.binary_scheduler.add_noise(target_patches, t)
        lookback_cond, past_maps = self._patch_refine_lookback_cond(past_norm)
        cond = expand_lookback_cond_for_patches(lookback_cond, locations)

        prev_refine_16 = None
        if unique:
            # Prev GT in the previous primary's row frame (matches AR infer).
            prev_32 = extract_prev_refine_crops(
                hir_gt,
                locations,
                patch_height=patch_h,
                patch_width=patch_w,
                col_stride=col_stride,
                coarse_edges=edges,
                canvas_height=canvas_h,
            )
            prev_refine_16 = compress_prev_refine_32_to_16(prev_32)
            drop_p = float(getattr(self.config, "patch_refine_prev_cond_dropout", 0.5))
            if self.training and drop_p > 0.0:
                keep = torch.rand(n_patches, device=device) >= drop_p
                prev_refine_16 = prev_refine_16 * keep.view(n_patches, 1, 1, 1).to(
                    dtype=prev_refine_16.dtype
                )

        aux, patch_coarse_bin, patch_time0 = build_patch_aux_channels(
            naive,
            edges,
            locations,
            patch_height=patch_h,
            patch_width=patch_w,
            canvas_height=canvas_h,
            coarse_height=coarse_h,
            horizon_width=int(hir_gt.shape[-1]),
            prev_refine_16=prev_refine_16,
        )

        ctx = None if getattr(self.config, "disable_cross_attention", False) else self._get_cross_variate_context(past, past_norm)
        ctx_flat = self._flatten_ctx_for_factorized_dit(ctx, B, V)
        ctx_patches = expand_ctx_for_patches(ctx_flat, locations)
        variate_indices = expand_variate_indices_for_patches(locations, device)

        canvas = self._inject_coordinate_channel(xt.float())
        canvas = torch.cat([canvas, aux], dim=1)
        base_cond = cond

        if self.training and self.config.cfg_dropout > 0.0 and ctx_patches is not None:
            drop = torch.rand(n_patches, device=device) < self.config.cfg_dropout
            ctx_patches = torch.where(
                drop.view(n_patches, 1, 1),
                torch.zeros_like(ctx_patches),
                ctx_patches,
            )

        out = self._predict_noise_chunked(
            canvas,
            t,
            cond,
            ctx_patches,
            variate_indices=variate_indices,
            token_variate_ids=self._ctx_token_variate_ids,
            patch_coarse_bin=patch_coarse_bin,
            patch_time0=patch_time0,
        )
        primary_logits, zt_logits = self._split_binary_heads(out)
        x0_logits = self._x0_logits_from_prediction(primary_logits, xt)
        if self.config.prediction_target == "epsilon":
            loss_x0 = self._binary_weighted_bce_loss(
                primary_logits, zt, t, weight_source=target_patches,
                element_mask=target_visible_mask,
            )
            loss_zt = self._binary_weighted_bce_loss(
                zt_logits, target_patches, t, weight_source=target_patches,
                element_mask=target_visible_mask,
            )
        else:
            loss_x0 = self._binary_weighted_bce_loss(
                primary_logits, target_patches, t, weight_source=target_patches,
                element_mask=target_visible_mask,
            )
            loss_zt = self._binary_weighted_bce_loss(
                zt_logits, zt, t, weight_source=target_patches,
                element_mask=target_visible_mask,
            )
        regular_loss = loss_x0 + loss_zt

        anchor_loss = torch.tensor(0.0, device=device)
        combined_loss = regular_loss
        if self.config.use_deterministic_anchor_loss:
            anchor_t = torch.full((n_patches,), self.config.binary_num_steps - 1, device=device, dtype=t.dtype)
            neutral = self._binary_anchor_canvas_like(target_patches)
            anchor_canvas = self._inject_coordinate_channel(neutral)
            anchor_canvas = torch.cat([anchor_canvas, aux], dim=1)
            anchor_out = self._predict_noise_chunked(
                anchor_canvas,
                anchor_t,
                base_cond,
                ctx_patches,
                variate_indices=variate_indices,
                token_variate_ids=self._ctx_token_variate_ids,
                patch_coarse_bin=patch_coarse_bin,
                patch_time0=patch_time0,
            )
            anchor_primary, _ = self._split_binary_heads(anchor_out)
            anchor_x0 = self._x0_logits_from_prediction(anchor_primary, neutral)
            anchor_loss = self._binary_plain_bce_loss(
                anchor_x0, target_patches, weight_source=target_patches,
                element_mask=target_visible_mask,
            )
            lam = self.config.deterministic_anchor_lambda
            combined_loss = lam * regular_loss + (1.0 - lam) * anchor_loss

        x0_pred = torch.sigmoid(x0_logits)
        return {
            "loss": combined_loss,
            "noise_loss": regular_loss,
            "combined_mse_loss": combined_loss,
            "anchor_loss": anchor_loss,
            "loss_x0": loss_x0,
            "loss_zt": loss_zt,
            "emd_loss": torch.tensor(0.0, device=device),
            "guidance_loss": torch.tensor(0.0, device=device),
            "noise_pred": x0_pred,
            "x0_pred": x0_pred,
            "future_2d": hir_gt,
            "future_2d_coarse": future_maps["coarse"],
            "future_2d_fine": future_maps["fine"],
            "past_2d_coarse": past_maps["coarse"],
            "past_2d_fine": past_maps["fine"],
            "t": t,
            "diffusion_stage": "patch_refine",
            "n_patches": torch.tensor(float(n_patches), device=device),
            "patch_visible_column_fraction": target_visible.float().mean(),
        }

    @torch.no_grad()
```
```python
# models/diffusion_tsf/diffusion_model.py:1508-1782
    def _generate_binary_patch_refine(
        self,
        past: torch.Tensor,
        num_steps: int = 20,
        verbose: bool = False,
        decoder_method: str = "mean",
        sampler: str = "quad_t",
        yield_intermediates: bool = False,
        reverse_step_indices: Optional[torch.Tensor] = None,
        snapshot_timesteps: Optional[Tuple[int, ...]] = None,
        future_coarse_2d: Optional[torch.Tensor] = None,
        future_fine_2d: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Infer second stage: sample patches on the coarse scaffold, blend, decode tall CDF.

        Needs future_coarse_2d from a coarse generate. Unique-segment mode AR-chains
        col0 groups (patch_refine_segments); else stride crops + blend_patch_bins.
        """
        from .patch_refine import (
            build_patch_aux_channels,
            expand_ctx_for_patches,
            expand_lookback_cond_for_patches,
            expand_variate_indices_for_patches,
            naive_upscale_coarse_cdf,
        )
        from .patch_refine_geometry import (
            blend_patch_bins,
            coarse_edges_from_cdf,
            select_patch_locations,
        )

        assert self.binary_scheduler is not None
        if future_coarse_2d is None:
            raise ValueError("patch_refine generation requires future_coarse_2d from the coarse model")

        B = past.shape[0]
        V = self.config.num_variables
        device = past.device
        canvas_h, patch_h, patch_w, col_stride = self._patch_refine_geometry_knobs()
        coarse_h = int(self.config.coarse_image_height)
        raw_hz_w = int(self.config.forecast_length)
        W_fut = self._repr_forecast_width(raw_hz_w)

        past_norm, _, stats = self._normalize_sequence(past)
        coarse = future_coarse_2d.to(device)
        if coarse.shape[:2] != (B, V) or coarse.shape[3] != W_fut:
            raise ValueError(
                "future_coarse_2d must have shape "
                f"(B={B}, V={V}, Hc, W={W_fut}), got {tuple(coarse.shape)}"
            )

        naive = naive_upscale_coarse_cdf(coarse, canvas_h)
        edges = coarse_edges_from_cdf(coarse, canvas_height=canvas_h)
        unique = bool(getattr(self.config, "patch_refine_unique_segments", False))
        lookback_cond, past_maps = self._patch_refine_lookback_cond(past_norm)
        ctx = None if getattr(self.config, "disable_cross_attention", False) else self._get_cross_variate_context(past, past_norm)
        ctx_flat = self._flatten_ctx_for_factorized_dit(ctx, B, V)

        def _sample_locations(
            locs: List,
            prev_refine_16: Optional[torch.Tensor],
        ) -> torch.Tensor:
            if not locs:
                return torch.zeros(0, 1, patch_h, patch_w, device=device)
            cond_l = expand_lookback_cond_for_patches(lookback_cond, locs)
            aux_l, patch_coarse_bin_l, patch_time0_l = build_patch_aux_channels(
                naive,
                edges,
                locs,
                patch_height=patch_h,
                patch_width=patch_w,
                canvas_height=canvas_h,
                coarse_height=coarse_h,
                horizon_width=W_fut,
                prev_refine_16=prev_refine_16,
            )
            ctx_l = expand_ctx_for_patches(ctx_flat, locs)
            var_l = expand_variate_indices_for_patches(locs, device)
            n_l = len(locs)

            def _build_canvas(xt: torch.Tensor) -> torch.Tensor:
                canvas = self._inject_coordinate_channel(xt)
                return torch.cat([canvas, aux_l], dim=1)

            def _chunked_model_fn(xt: torch.Tensor, t_batch: torch.Tensor):
                out = self._predict_noise_chunked(
                    _build_canvas(xt),
                    t_batch,
                    cond_l,
                    ctx_l,
                    variate_indices=var_l,
                    token_variate_ids=self._ctx_token_variate_ids,
                    patch_coarse_bin=patch_coarse_bin_l,
                    patch_time0=patch_time0_l,
                )
                primary, zt = self._split_binary_heads(out)
                x0_logits = self._x0_logits_from_prediction(primary, xt)
                return x0_logits, zt

            sample_shape = (n_l, 1, patch_h, patch_w)
            if sampler in ("anchor", "deterministic_anchor"):
                t_batch = torch.full(
                    (n_l,),
                    self.config.binary_num_steps - 1,
                    device=device,
                    dtype=torch.long,
                )
                neutral = self._binary_anchor_canvas_shape(sample_shape, device=device)
                x0_logits, _ = _chunked_model_fn(neutral, t_batch)
                return (torch.sigmoid(x0_logits) > 0.5).float()
            return self.binary_scheduler.sample(
                model_fn=_chunked_model_fn,
                shape=sample_shape,
                num_steps=num_steps,
                device=device,
                verbose=verbose,
                sampler=sampler,
                reverse_step_indices=reverse_step_indices,
                snapshot_timesteps=snapshot_timesteps,
            )

        if unique:
            from .patch_refine_segments import (
                compress_prev_refine_32_to_16,
                group_locations_by_col0,
                select_coverage_gap_locations,
                select_primary_ar_locations,
            )
            # AR is the main predictor: unique stride-6 col0 chain, then
            # blanked-prev fills only where primary coverage is incomplete.
            primary_locs = select_primary_ar_locations(
                edges,
                canvas_height=canvas_h,
                patch_height=patch_h,
                patch_width=patch_w,
                col_stride=col_stride,
            )
            last_pred: Dict[Tuple[int, int], torch.Tensor] = {}
            primary_pred_by_key: Dict[Tuple[int, int, int, int], torch.Tensor] = {}
            for _col0, col_locs in group_locations_by_col0(primary_locs):
                # Batch all (B,V) at this col0 together.
                prev_chunks = []
                for loc in col_locs:
                    key = (loc.batch_index, loc.variate_index)
                    if key in last_pred:
                        prev_chunks.append(
                            compress_prev_refine_32_to_16(last_pred[key].unsqueeze(0))
                        )
                    else:
                        prev_chunks.append(
                            torch.zeros(1, 1, 16, patch_w, device=device)
                        )
                prev_16 = torch.cat(prev_chunks, dim=0)
                pred = _sample_locations(col_locs, prev_16)
                for j, loc in enumerate(col_locs):
                    last_pred[(loc.batch_index, loc.variate_index)] = pred[j]
                    primary_pred_by_key[
                        (loc.batch_index, loc.variate_index, loc.col0, loc.row0)
                    ] = pred[j]

            gap_locs = select_coverage_gap_locations(
                edges,
                primary_locs,
                canvas_height=canvas_h,
                patch_height=patch_h,
                patch_width=patch_w,
            )
            locations = list(primary_locs) + list(gap_locs)
            n_patches = len(locations)
            patch_cdf = torch.zeros(n_patches, 1, patch_h, patch_w, device=device)
            for i, loc in enumerate(primary_locs):
                patch_cdf[i] = primary_pred_by_key[
                    (loc.batch_index, loc.variate_index, loc.col0, loc.row0)
                ]
            if gap_locs:
                gap_prev = torch.zeros(len(gap_locs), 1, 16, patch_w, device=device)
                gap_pred = _sample_locations(gap_locs, gap_prev)
                patch_cdf[len(primary_locs) :] = gap_pred
        else:
            locations = select_patch_locations(
                edges,
                canvas_height=canvas_h,
                patch_height=patch_h,
                patch_width=patch_w,
                col_stride=col_stride,
            )
            n_patches = len(locations)
            cond = expand_lookback_cond_for_patches(lookback_cond, locations)
            aux, patch_coarse_bin, patch_time0 = build_patch_aux_channels(
                naive,
                edges,
                locations,
                patch_height=patch_h,
                patch_width=patch_w,
                canvas_height=canvas_h,
                coarse_height=coarse_h,
                horizon_width=W_fut,
            )
            ctx_patches = expand_ctx_for_patches(ctx_flat, locations)
            variate_indices = expand_variate_indices_for_patches(locations, device)

            def _build_canvas(xt: torch.Tensor) -> torch.Tensor:
                canvas = self._inject_coordinate_channel(xt)
                return torch.cat([canvas, aux], dim=1)

            def _chunked_model_fn(xt: torch.Tensor, t_batch: torch.Tensor):
                out = self._predict_noise_chunked(
                    _build_canvas(xt),
                    t_batch,
                    cond,
                    ctx_patches,
                    variate_indices=variate_indices,
                    token_variate_ids=self._ctx_token_variate_ids,
                    patch_coarse_bin=patch_coarse_bin,
                    patch_time0=patch_time0,
                )
                primary, zt = self._split_binary_heads(out)
                x0_logits = self._x0_logits_from_prediction(primary, xt)
                return x0_logits, zt

            sample_shape = (n_patches, 1, patch_h, patch_w)
            if sampler in ("anchor", "deterministic_anchor"):
                t_batch = torch.full(
                    (n_patches,),
                    self.config.binary_num_steps - 1,
                    device=device,
                    dtype=torch.long,
                )
                neutral = self._binary_anchor_canvas_shape(sample_shape, device=device)
                x0_logits, _ = _chunked_model_fn(neutral, t_batch)
                patch_cdf = (torch.sigmoid(x0_logits) > 0.5).float()
            else:
                patch_cdf = self.binary_scheduler.sample(
                    model_fn=_chunked_model_fn,
                    shape=sample_shape,
                    num_steps=num_steps,
                    device=device,
                    verbose=verbose,
                    sampler=sampler,
                    reverse_step_indices=reverse_step_indices,
                    snapshot_timesteps=snapshot_timesteps,
                )

        hir_cdf, patch_vote_counts = blend_patch_bins(
            patch_cdf,
            locations,
            edges,
            canvas_height=canvas_h,
            patch_height=patch_h,
            patch_width=patch_w,
        )
        future_norm = self._decode_absolute_future_hir(hir_cdf)
        future_with_overlap = self._denormalize_future(
            future_norm, past, stats, trim_overlap=False,
        )
        future = future_with_overlap[..., int(self.config.lookback_overlap):]
        return {
            "prediction": future,
            "prediction_norm": future_norm,
            "prediction_global_norm": future,
            "prediction_with_overlap": future_with_overlap,
            "future_2d": hir_cdf,
            "future_2d_coarse": coarse,
            "future_2d_fine": hir_cdf,
            "past_2d_coarse": past_maps["coarse"],
            "past_2d_fine": past_maps["fine"],
            # Keep the pre-blend crops for diagnostics which must not average
            # the stride-overlapping patch predictions.
            "patch_cdf_unblended": patch_cdf,
            "patch_locations": locations,
            "patch_vote_counts": patch_vote_counts,
            "diffusion_stage": "patch_refine",
        }
```

### 8b. Coarse / fine staged forward / generate

```python
# models/diffusion_tsf/diffusion_model.py:1783-1957
    def _forward_binary_staged(
        self,
        past: torch.Tensor,
        future: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Train one full-horizon stage: coarse CDF or fine residual CDF.

        Fine always sees GT coarse on the cond path during train (CFG drops tokens only).
        Loss is dual-head BCE (x0 + zt); optional stationary-flat anchor mix.
        """
        assert self.binary_scheduler is not None, "binary scheduler is not initialized"
        stage = self.config.diffusion_stage
        if stage not in {"coarse", "fine"}:
            raise ValueError(f"_forward_binary_staged called for stage={stage!r}")

        B = past.shape[0]
        V = self.config.num_variables
        H = self.config.image_height
        device = past.device
        BV = B * V

        past_norm, future_norm, _stats = self._normalize_sequence(past, future)
        future_maps = self._encode_staged_maps(future_norm)
        target_2d = future_maps[stage]
        W_fut = target_2d.shape[3]
        H = target_2d.shape[2]
        target_flat = target_2d.reshape(BV, 1, H, W_fut)

        if t is None:
            t = torch.randint(0, self.config.binary_num_steps, (B,), device=device)
        t_flat = t.unsqueeze(1).expand(-1, V).reshape(BV)
        variate_indices = None
        if self.config.use_variate_embedding and self.config.variate_factorized and V > 1:
            variate_indices = self._flat_variate_indices(BV, V, device)

        xt_flat, zt_flat = self.binary_scheduler.add_noise(target_flat, t_flat)

        ctx = None if getattr(self.config, 'disable_cross_attention', False) else self._get_cross_variate_context(past, past_norm)
        ctx_flat = self._flatten_ctx_for_factorized_dit(ctx, B, V)

        cond_for_unet, past_maps = self._staged_past_condition(past_norm, W_fut, past_raw=past)
        if stage == "fine":
            # Teacher-force coarse map as an extra horizon cond channel.
            future_coarse_cond = self._coarse_cdf_to_height(future_maps["coarse"], H)
            future_coarse_flat = future_coarse_cond.reshape(BV, 1, H, W_fut)
            cond_for_unet = self._cat_past_and_horizon_cond(cond_for_unet, future_coarse_flat)

        guidance_flat = None
        n_guidance_cond_ch = 0
        if self.config.use_guidance_channel:
            raw_hz_w = int(future_norm.shape[-1])
            guidance_forecast_norm = self._get_guidance_forecast_norm(past, past_norm, _stats, raw_hz_w)
            if self._uses_canvas_guidance():
                guidance_maps = self._encode_staged_maps(guidance_forecast_norm)
                if stage == "coarse":
                    guidance_flat = self._resize_cdf_height(guidance_maps["coarse"], H).reshape(BV, 1, H, W_fut)
                else:
                    guidance_flat = self._resize_cdf_height(guidance_maps["fine"], H).reshape(BV, 1, H, W_fut)
            elif self._uses_cond_chunk_guidance():
                guidance_cond = self._encode_guidance_cond_chunks(
                    guidance_forecast_norm, stage, H, BV,
                )
                n_guidance_cond_ch = int(guidance_cond.shape[1])
                guidance_cond = self._align_guidance_cond_width(guidance_cond, cond_for_unet.shape[-1])
                cond_for_unet = torch.cat([cond_for_unet, guidance_cond], dim=1)

        base_cond_for_unet = cond_for_unet

        canvas = self._inject_coordinate_channel(xt_flat.float())

        # Staged visual conditioning is always GT during training. CFG dropout is
        # restricted to context tokens so the fine stage never sees predicted coarse.
        ctx_anchor = ctx_flat
        if self.training and self.config.cfg_dropout > 0.0:
            drop_mask = torch.rand(B, device=device) < self.config.cfg_dropout
            drop_mask_flat = drop_mask.unsqueeze(1).expand(-1, V).reshape(BV)
            if ctx_flat is not None:
                ctx_flat = torch.where(
                    drop_mask_flat.view(BV, 1, 1),
                    torch.zeros_like(ctx_flat),
                    ctx_flat,
                )
            if guidance_flat is not None:
                guidance_for_unet = torch.where(
                    drop_mask_flat.view(BV, 1, 1, 1),
                    torch.zeros_like(guidance_flat),
                    guidance_flat,
                )
                canvas = torch.cat([canvas, guidance_for_unet], dim=1)
            elif n_guidance_cond_ch > 0:
                cond_for_unet = self._zero_guidance_cond_tail(
                    cond_for_unet, n_guidance_cond_ch, drop_mask_flat,
                )
        elif guidance_flat is not None:
            canvas = torch.cat([canvas, guidance_flat], dim=1)

        out_flat = self._predict_noise_chunked(
            canvas, t_flat, cond_for_unet, ctx_flat, variate_indices=variate_indices, token_variate_ids=self._ctx_token_variate_ids,
        )
        primary_logits, zt_logits = self._split_binary_heads(out_flat)
        x0_logits = self._x0_logits_from_prediction(primary_logits, xt_flat)
        if self.config.prediction_target == "epsilon":
            loss_x0 = self._binary_weighted_bce_loss(
                primary_logits, zt_flat, t_flat, weight_source=target_flat,
            )
            loss_zt = self._binary_weighted_bce_loss(
                zt_logits, target_flat, t_flat, weight_source=target_flat,
            )
        else:
            loss_x0 = self._binary_weighted_bce_loss(
                primary_logits, target_flat, t_flat, weight_source=target_flat,
            )
            loss_zt = self._binary_weighted_bce_loss(
                zt_logits, zt_flat, t_flat, weight_source=target_flat,
            )
        regular_loss = loss_x0 + loss_zt

        anchor_loss = torch.tensor(0.0, device=device)
        combined_loss = regular_loss
        if self.config.use_deterministic_anchor_loss:
            anchor_t_flat = torch.full(
                (BV,),
                self.config.binary_num_steps - 1,
                device=device,
                dtype=t_flat.dtype,
            )
            neutral_future_flat = self._binary_anchor_canvas_like(target_flat)
            anchor_canvas = self._inject_coordinate_channel(neutral_future_flat)
            if guidance_flat is not None:
                anchor_canvas = torch.cat([anchor_canvas, guidance_flat], dim=1)
            anchor_out_flat = self._predict_noise_chunked(
                anchor_canvas,
                anchor_t_flat,
                base_cond_for_unet,
                ctx_anchor,
                variate_indices=variate_indices, token_variate_ids=self._ctx_token_variate_ids,
            )
            anchor_primary, _ = self._split_binary_heads(anchor_out_flat)
            anchor_x0_logits = self._x0_logits_from_prediction(anchor_primary, neutral_future_flat)
            anchor_bce = self._binary_plain_bce_loss(
                anchor_x0_logits, target_flat, weight_source=target_flat,
            )
            anchor_loss = anchor_bce
            lam = self.config.deterministic_anchor_lambda
            combined_loss = lam * regular_loss + (1.0 - lam) * anchor_loss

        x0_pred = torch.sigmoid(x0_logits).reshape(B, V, H, W_fut)
        # combined_mse_loss / emd_loss / guidance_loss names are logging leftovers (BCE here; zeros).
        result = {
            'loss': combined_loss,
            'noise_loss': regular_loss,
            'combined_mse_loss': combined_loss,
            'anchor_loss': anchor_loss,
            'loss_x0': loss_x0,
            'loss_zt': loss_zt,
            'emd_loss': torch.tensor(0.0, device=device),
            'guidance_loss': torch.tensor(0.0, device=device),
            'noise_pred': x0_pred,
            'x0_pred': x0_pred,
            'future_2d': target_2d,
            'future_2d_coarse': future_maps["coarse"],
            'future_2d_fine': future_maps["fine"],
            'past_2d_coarse': past_maps["coarse"],
            'past_2d_fine': past_maps["fine"],
            't': t,
            'diffusion_stage': stage,
        }
        if stage == "coarse":
            result['x0_pred_coarse'] = x0_pred
        else:
            result['x0_pred_fine'] = x0_pred
        return result

    @torch.no_grad()
```
```python
# models/diffusion_tsf/diffusion_model.py:2038-2210
    def _generate_binary_staged(
        self,
        past: torch.Tensor,
        num_steps: int = 20,
        verbose: bool = False,
        decoder_method: str = "mean",
        sampler: str = "quad_t",
        yield_intermediates: bool = False,
        reverse_step_indices: Optional[torch.Tensor] = None,
        snapshot_timesteps: Optional[Tuple[int, ...]] = None,
        future_coarse_2d: Optional[torch.Tensor] = None,
        future_fine_2d: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Sample one full-horizon stage. Fine conditions on predicted coarse, then dual-decodes.

        Reverse loop is BinaryDiffusionScheduler.sample (or one-shot anchor).
        Eval chains this via staged_eval._staged_generate_once.
        """
        assert self.binary_scheduler is not None, "binary scheduler is not initialized"
        stage = self.config.diffusion_stage
        if stage not in {"coarse", "fine"}:
            raise ValueError(f"_generate_binary_staged called for stage={stage!r}")

        B = past.shape[0]
        V = self.config.num_variables
        H = self.config.image_height
        device = past.device
        BV = B * V
        raw_hz_w = int(self.config.forecast_length)
        W_fut = self._repr_forecast_width(raw_hz_w)

        past_norm, _, stats = self._normalize_sequence(past)
        cond_for_unet, past_maps = self._staged_past_condition(past_norm, W_fut, past_raw=past)
        coarse_for_decode = future_coarse_2d
        if stage == "fine":
            if future_coarse_2d is None:
                raise ValueError("fine-stage generation requires future_coarse_2d from the coarse model.")
            if future_coarse_2d.shape[:2] != (B, V) or future_coarse_2d.shape[3] != W_fut:
                raise ValueError(
                    "future_coarse_2d must have shape "
                    f"(B={B}, V={V}, Hc, W={W_fut}), got {tuple(future_coarse_2d.shape)}"
                )
            future_coarse_cond = self._coarse_cdf_to_height(future_coarse_2d.to(device), H)
            future_coarse_flat = future_coarse_cond.reshape(BV, 1, H, W_fut)
            cond_for_unet = self._cat_past_and_horizon_cond(cond_for_unet, future_coarse_flat)

        ctx = None if getattr(self.config, 'disable_cross_attention', False) else self._get_cross_variate_context(past, past_norm)
        ctx_flat = self._flatten_ctx_for_factorized_dit(ctx, B, V)
        variate_indices = None
        if self.config.use_variate_embedding and self.config.variate_factorized and V > 1:
            variate_indices = self._flat_variate_indices(BV, V, device)

        guidance_flat = None
        if self.config.use_guidance_channel:
            guidance_forecast_norm = self._get_guidance_forecast_norm(past, past_norm, stats, raw_hz_w)
            if self._uses_canvas_guidance():
                guidance_maps = self._encode_staged_maps(guidance_forecast_norm)
                if stage == "coarse":
                    guidance_flat = self._resize_cdf_height(guidance_maps["coarse"], H).reshape(BV, 1, H, W_fut)
                else:
                    guidance_flat = self._resize_cdf_height(guidance_maps["fine"], H).reshape(BV, 1, H, W_fut)
            elif self._uses_cond_chunk_guidance():
                guidance_cond = self._encode_guidance_cond_chunks(
                    guidance_forecast_norm, stage, H, BV,
                )
                guidance_cond = self._align_guidance_cond_width(
                    guidance_cond, cond_for_unet.shape[-1],
                )
                cond_for_unet = torch.cat([cond_for_unet, guidance_cond], dim=1)

        def _build_canvas(xt: torch.Tensor) -> torch.Tensor:
            canvas = self._inject_coordinate_channel(xt)
            if guidance_flat is not None:
                canvas = torch.cat([canvas, guidance_flat], dim=1)
            return canvas

        def _chunked_model_fn(xt: torch.Tensor, t_batch: torch.Tensor):
            out = self._predict_noise_chunked(
                _build_canvas(xt), t_batch, cond_for_unet, ctx_flat,
                variate_indices=variate_indices, token_variate_ids=self._ctx_token_variate_ids,
            )
            primary, zt = self._split_binary_heads(out)
            x0_logits = self._x0_logits_from_prediction(primary, xt)
            return x0_logits, zt

        intermediates = None
        sample_shape = (BV, self._occupancy_channels(), H, W_fut)
        if sampler in ("anchor", "deterministic_anchor"):
            t_batch = torch.full(
                (BV,),
                self.config.binary_num_steps - 1,
                device=device,
                dtype=torch.long,
            )
            neutral_future_flat = self._binary_anchor_canvas_shape(
                sample_shape, device=device,
            )
            x0_logits, _zt_logits = _chunked_model_fn(neutral_future_flat, t_batch)
            future_2d_flat = (torch.sigmoid(x0_logits) > 0.5).float()
            if yield_intermediates:
                intermediates = [(999, neutral_future_flat.clone()), (0, future_2d_flat.clone())]
        else:
            sample_kwargs = dict(
                model_fn=_chunked_model_fn,
                shape=sample_shape,
                num_steps=num_steps,
                device=device,
                verbose=verbose,
                sampler=sampler,
                reverse_step_indices=reverse_step_indices,
                snapshot_timesteps=snapshot_timesteps,
            )
            if yield_intermediates:
                future_2d_flat, intermediates = self.binary_scheduler.sample(
                    yield_intermediates=True,
                    **sample_kwargs,
                )
            else:
                future_2d_flat = self.binary_scheduler.sample(**sample_kwargs)

        generated_2d = future_2d_flat.reshape(B, V, H, W_fut)
        if stage == "coarse":
            future_2d_coarse = generated_2d
            if decoder_method != "mean":
                raise ValueError(f"decoder_method must be 'mean', got {decoder_method!r}")
            future_norm = self._decode_coarse_1d_from_map(
                future_2d_coarse,
                cdf_decoder="mean",
            )
            future_2d_fine = None
        else:
            future_2d_coarse = coarse_for_decode.to(device)
            future_2d_fine = generated_2d
            k = int(self.config.lookback_overlap)
            if k > 0:
                past_seed = past_norm[..., k - 1]
            else:
                past_seed = past_norm[..., -1]
            future_norm = self.decode_dual_from_2d(
                future_2d_coarse,
                future_2d_fine,
                from_diffusion=False,
                decoder_method=decoder_method,
                past_seed=past_seed,
            )
        future_with_overlap = self._denormalize_future(
            future_norm, past, stats, trim_overlap=False,
        )
        future = future_with_overlap[..., int(self.config.lookback_overlap):]

        result = {
            'prediction': future,
            'prediction_norm': future_norm,
            'prediction_global_norm': future,
            # Retain the K lookback-overlap predictions for diagnostic plots.
            # Metrics continue to consume the forecast-only tensors above.
            'prediction_with_overlap': future_with_overlap,
            'future_2d': generated_2d,
            'future_2d_coarse': future_2d_coarse,
            'past_2d_coarse': past_maps["coarse"],
            'past_2d_fine': past_maps["fine"],
            'diffusion_stage': stage,
        }
        if future_2d_fine is not None:
            result['future_2d_fine'] = future_2d_fine
        if intermediates is not None:
            reshaped_intermediates = []
            for (t_idx, i_tensor) in intermediates:
                reshaped_intermediates.append((t_idx, i_tensor.reshape(B, V, H, W_fut)))
            result['intermediates'] = reshaped_intermediates
        return result
```

## 9. Binary scheduler + FactorizedDiT + preprocessing

```python
# models/diffusion_tsf/diffusion.py:1-162
"""Binary bit-flip scheduler for hard CDF images.

Forward: XOR each bit with Bernoulli(beta_t). Reverse: model predicts x0 logits,
draw Bernoulli(sigmoid), reflip toward lower t. Called from DiffusionTSF.generate
via BinaryDiffusionScheduler.sample. Anchor decode bypasses this loop.
"""

import logging
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def _build_transition_schedule(
    num_steps: int,
    transition_min: float,
    transition_max: float,
    schedule_type: str,
    device: str,
) -> torch.Tensor:
    """Build β_t in [transition_min, transition_max] (linear schedule only)."""
    if schedule_type != "linear":
        raise ValueError(
            f"binary_noise_schedule must be 'linear', got {schedule_type!r}"
        )
    t = torch.linspace(0.0, 1.0, num_steps, device=device)
    betas = transition_min + t * (transition_max - transition_min)
    return betas.clamp(1e-8, 1.0 - 1e-8)


class BinaryDiffusionScheduler:
    """Bit-flip diffusion scheduler for hard binary CDF images.

    The forward process flips each bit with probability beta_t. The reverse
    sampler predicts a clean x0 image, then re-noises it at the next lower
    timestep until reaching a clean binary map.
    """

    def __init__(
        self,
        num_steps: int = 1000,
        beta_start: float = 1e-5,
        beta_end: float = 0.5,
        schedule_type: str = "linear",
        device: str = "cpu",
    ):
        self.num_steps = num_steps
        self.device = device

        self.betas = _build_transition_schedule(
            num_steps,
            beta_start,
            beta_end,
            schedule_type,
            device,
        )
        self.schedule_type = schedule_type
        logger.debug(
            "BinaryDiffusionScheduler initialized: T=%d, schedule=%s, beta=[%.2e, %.3f]",
            self.num_steps,
            self.schedule_type,
            self.betas[0].item(),
            self.betas[-1].item(),
        )

    def to(self, device: str) -> "BinaryDiffusionScheduler":
        self.device = device
        self.betas = self.betas.to(device)
        return self

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Training corruption: xt = x0 XOR zt, zt ~ Bern(beta_t). Returns (xt, zt)."""
        beta_t = self.betas[t]
        shape = (-1,) + (1,) * (x0.dim() - 1)
        beta_t = beta_t.view(shape).expand_as(x0)
        zt = torch.bernoulli(beta_t)
        xt = (x0.bool() ^ zt.bool()).float()
        return xt, zt

    @torch.no_grad()
    def sample(
        self,
        model_fn,
        shape: Tuple[int, ...],
        num_steps: int = 20,
        device: str = "cpu",
        verbose: bool = False,
        sampler: str = "quad_t",
        yield_intermediates: bool = False,
        reverse_step_indices: Optional[torch.Tensor] = None,
        snapshot_timesteps: Optional[Tuple[int, ...]] = None,
    ):
        """Reduced reverse schedule. model_fn(xt, t) -> (x0_logits, zt_logits).

        ``sampler`` only selects the discrete timestep grid (not a continuous
        DPM++ ODE solver):
          - ``quad_t`` / ``ddim_quad``: quadratic spacing (more steps near high noise)
        Legacy names ``ddim`` (linear spacing) and ``dpmpp`` are rejected.

        Each step draws ``x0 ~ Bernoulli(sigmoid(logits))`` (not hard threshold)
        and the final step keeps that Bernoulli draw (no silent freeze). Mid-loop
        steps still reflip with ``Bernoulli(β_next)``. Anchor one-shot decode is
        a separate path in ``generate`` and stays hard-thresholded.
        """
        if reverse_step_indices is not None:
            step_indices = reverse_step_indices.to(device=device, dtype=torch.long)
        else:
            name = str(sampler).lower()
            if name == "ddim":
                raise ValueError(
                    "sampler='ddim' (linear timestep spacing) was removed. "
                    "Use sampler='quad_t' (alias: 'ddim_quad')."
                )
            if name == "dpmpp":
                raise ValueError(
                    "sampler='dpmpp' was only quadratic timestep spacing, not DPM++. "
                    "Use sampler='quad_t' (alias: 'ddim_quad')."
                )
            if name in {"quad_t", "ddim_quad"}:
                ramp = torch.linspace(1.0, 0.0, num_steps, device=device)
                step_indices = torch.round((ramp ** 2) * (self.num_steps - 1)).long()
            else:
                raise ValueError(
                    f"Unknown binary sampler {sampler!r}; expected quad_t or ddim_quad"
                )
        snapshot_set = None
        if snapshot_timesteps is not None:
            snapshot_set = {int(min(max(0, t), self.num_steps - 1)) for t in snapshot_timesteps}
        xt = torch.bernoulli(torch.full(shape, 0.5, device=device))

        intermediates = []

        for i, t_val in enumerate(step_indices):
            t_idx = int(t_val.item())
            if yield_intermediates and (
                snapshot_set is None or t_idx in snapshot_set
            ):
                intermediates.append((t_idx, xt.clone()))

            t_batch = torch.full((shape[0],), t_idx, device=device, dtype=torch.long)
            x0_logits, _zt_logits = model_fn(xt, t_batch)
            # A1+A2: Bernoulli x0 every step, including the last (no hard threshold / freeze).
            x0_hat = torch.bernoulli(torch.sigmoid(x0_logits))

            if i < len(step_indices) - 1:
                t_next = int(step_indices[i + 1].item())
                beta_next = self.betas[t_next].item()
                zt_new = torch.bernoulli(torch.full_like(x0_hat, beta_next))
                xt = (x0_hat.bool() ^ zt_new.bool()).float()
            else:
                xt = x0_hat

            if verbose and i % 5 == 0:
                logger.debug(f"  binary step {i + 1}/{num_steps} (t={t_idx})")

        if yield_intermediates:
            if snapshot_set is None or 0 in snapshot_set:
                intermediates.append((0, xt.clone()))
            return xt, intermediates
        return xt
```
```python
# models/diffusion_tsf/dit.py:229-489
class FactorizedDiT(nn.Module):
    """Per-variate DiT backbone with bottleneck cross-attention to iTrans tokens.

    Inputs:
        x: (BV, in_channels, H, W_fut) noisy future canvas + aux + guidance ghost
        t: (BV,) diffusion timestep
        cond: (BV, cond_channels, H, W_fut) visual conditioning (past 2D resized)
        encoder_hidden_states: (BV, M, ctx_dim) or None — guidance token memory

    Returns:
        (BV, out_channels, H, W_fut) noise prediction
    """

    def __init__(
        self,
        in_channels: int,
        cond_channels: int,
        out_channels: int = 1,
        image_height: int = 32,
        patch_size: Tuple[int, int] = (8, 8),
        embed_dim: int = 384,
        depth: int = 8,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        context_dim: int = 256,
        max_pos_tokens: int = 8192,
        gradient_checkpointing: bool = False,
        cond_patch_size: Optional[Tuple[int, int]] = None,
        use_scale_embedding: bool = False,
        enable_cross_scale_attention: bool = False,
        use_variate_embedding: bool = False,
        max_variates: int = 512,
        use_patch_abs_embedding: bool = False,
        max_coarse_bins: int = 16,
        max_horizon_steps: int = 1024,
    ):
        super().__init__()
        pH, pW = patch_size
        if image_height % pH != 0:
            raise ValueError(f"image_height={image_height} not divisible by patch_height={pH}")
        self.image_height = image_height
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.out_channels = out_channels
        self.gradient_checkpointing = gradient_checkpointing
        self.use_scale_embedding = use_scale_embedding
        self.enable_cross_scale_attention = enable_cross_scale_attention
        self.use_variate_embedding = use_variate_embedding
        self.use_patch_abs_embedding = use_patch_abs_embedding
        self.cond_patch_size = cond_patch_size or patch_size

        self.x_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cond_embed = nn.Conv2d(
            cond_channels,
            embed_dim,
            kernel_size=self.cond_patch_size,
            stride=self.cond_patch_size,
        )

        # Separate learned positional embeddings for cond vs x slots so the model
        # can distinguish them even though they share the same sequence axis.
        self.pos_x = nn.Parameter(torch.zeros(1, max_pos_tokens, embed_dim))
        self.pos_cond = nn.Parameter(torch.zeros(1, max_pos_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_x, std=0.02)
        nn.init.trunc_normal_(self.pos_cond, std=0.02)

        self.t_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        # Patch-refine absolute position tables; unused unless use_patch_abs_embedding.
        if use_scale_embedding:
            self.scale_embed = nn.Embedding(2, embed_dim)
        else:
            self.scale_embed = None
        if use_variate_embedding:
            self.variate_embed = nn.Embedding(max_variates, embed_dim)
        else:
            self.variate_embed = None
        if use_patch_abs_embedding:
            self.coarse_bin_embed = nn.Embedding(max_coarse_bins, embed_dim)
            self.horizon_time_embed = nn.Embedding(max_horizon_steps, embed_dim)
        else:
            self.coarse_bin_embed = None
            self.horizon_time_embed = None

        self.ctx_proj = nn.Linear(context_dim, embed_dim)
        self.ctx_norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # Bottleneck position: middle of the stack. One cross-attn block.
        self.bottleneck_idx = depth // 2
        self.blocks = nn.ModuleList()
        for i in range(depth):
            if i == self.bottleneck_idx:
                self.blocks.append(
                    _DiTCrossAttnBlock(
                        embed_dim,
                        num_heads,
                        mlp_ratio,
                        dropout,
                        enable_cross_scale_attention=enable_cross_scale_attention,
                    )
                )
            else:
                self.blocks.append(_DiTBlock(embed_dim, num_heads, mlp_ratio, dropout))

        self.final_norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, 2 * embed_dim))
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)

        self.head = nn.Linear(embed_dim, out_channels * pH * pW)
        # zero-init head: model starts as identity (noise in -> noise out)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _patchify(self, img: torch.Tensor, proj: nn.Conv2d) -> Tuple[torch.Tensor, int, int]:
        """(B, C, H, W) -> (B, gh*gw, D), with (gh, gw) returned for unpatchify."""
        h = proj(img)  # (B, D, gh, gw)
        gh, gw = h.shape[-2], h.shape[-1]
        return h.flatten(2).transpose(1, 2), gh, gw

    def _unpatchify(self, tokens: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
        B = tokens.shape[0]
        pH, pW = self.patch_size
        h = self.head(tokens).view(B, gh, gw, self.out_channels, pH, pW)
        h = h.permute(0, 3, 1, 4, 2, 5).contiguous()
        return h.view(B, self.out_channels, gh * pH, gw * pW)

    @staticmethod
    def _pad_to_patch(
        img: torch.Tensor,
        patch_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, int, int]:
        pH, pW = patch_size
        H, W = img.shape[-2], img.shape[-1]
        pad_h = (pH - H % pH) % pH
        pad_w = (pW - W % pW) % pW
        if pad_h or pad_w:
            img = F.pad(img, (0, pad_w, 0, pad_h), mode="reflect")
        return img, pad_h, pad_w

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        scale_indices: Optional[torch.Tensor] = None,
        variate_indices: Optional[torch.Tensor] = None,
        token_variate_ids: Optional[torch.Tensor] = None,
        patch_coarse_bin: Optional[torch.Tensor] = None,
        patch_time0: Optional[torch.Tensor] = None,
        return_cross_attn_weights: bool = False,
    ):
        # x = noisy canvas (+ coord + aux); cond = past 2D visual; tokens = guidance.
        BV, _, H, W = x.shape
        self._diag_cross_attn_weights = None

        x_p, pad_h, pad_w = self._pad_to_patch(x, self.patch_size)
        cond_p, _, _ = self._pad_to_patch(cond, self.cond_patch_size)

        x_tok, gh, gw = self._patchify(x_p, self.x_embed)
        c_tok, _, _ = self._patchify(cond_p, self.cond_embed)

        Nx, Nc = x_tok.shape[1], c_tok.shape[1]
        if Nx > self.pos_x.shape[1] or Nc > self.pos_cond.shape[1]:
            raise RuntimeError(
                f"DiT pos table too small: need Nx={Nx}, Nc={Nc}, "
                f"have {self.pos_x.shape[1]}. Increase max_pos_tokens."
            )
        x_tok = x_tok + self.pos_x[:, :Nx]
        c_tok = c_tok + self.pos_cond[:, :Nc]

        if self.use_patch_abs_embedding:
            if patch_coarse_bin is None or patch_time0 is None:
                raise ValueError(
                    "patch_coarse_bin and patch_time0 are required when "
                    "use_patch_abs_embedding=True"
                )
            if patch_coarse_bin.shape[0] != BV or patch_time0.shape[0] != BV:
                raise ValueError(
                    f"patch location batch mismatch: bins={tuple(patch_coarse_bin.shape)} "
                    f"time0={tuple(patch_time0.shape)} BV={BV}"
                )
            # Crop-level absolute ids broadcast over all target tokens.
            abs_emb = (
                self.coarse_bin_embed(patch_coarse_bin.long())
                + self.horizon_time_embed(patch_time0.long())
            ).unsqueeze(1)
            x_tok = x_tok + abs_emb

        tokens = torch.cat([c_tok, x_tok], dim=1)  # (BV, Nc + Nx, D)

        if self.variate_embed is not None:
            if variate_indices is None:
                raise ValueError("variate_indices are required when variate embeddings are enabled.")
            if variate_indices.shape[0] != BV:
                raise ValueError(f"variate_indices batch {variate_indices.shape[0]} != BV {BV}")
            v_emb = self.variate_embed(variate_indices.long()).unsqueeze(1)
            tokens = tokens + v_emb

        if t.shape[0] != BV:
            raise ValueError(f"timestep batch {t.shape[0]} != BV {BV}")
        t_emb = self.t_embed(_timestep_embedding(t, self.embed_dim))  # (BV, D)
        if self.scale_embed is not None:
            if scale_indices is None:
                raise ValueError("scale_indices are required when scale embeddings are enabled.")
            if scale_indices.shape[0] != BV:
                raise ValueError(f"scale_indices batch {scale_indices.shape[0]} != BV {BV}")
            t_emb = t_emb + self.scale_embed(scale_indices.long())

        ctx_proj: Optional[torch.Tensor] = None
        if encoder_hidden_states is not None:
            if encoder_hidden_states.shape[0] != BV:
                raise ValueError(
                    f"encoder_hidden_states batch {encoder_hidden_states.shape[0]} != BV {BV}"
                )
            ctx_proj = self.ctx_norm(self.ctx_proj(encoder_hidden_states))  # (BV, V, D)

        for i, block in enumerate(self.blocks):
            if i == self.bottleneck_idx:
                if self.gradient_checkpointing and self.training:
                    tokens = checkpoint(
                        block,
                        tokens,
                        t_emb,
                        ctx_proj,
                        scale_indices,
                        variate_indices,
                        use_reentrant=False,
                    )
                elif return_cross_attn_weights:
                    tokens, attn_w = block(
                        tokens, t_emb, ctx_proj, scale_indices, variate_indices,
                        token_variate_ids=token_variate_ids,
                        return_attn_weights=True,
                    )
                    self._diag_cross_attn_weights = attn_w
                else:
                    tokens = block(
                        tokens, t_emb, ctx_proj, scale_indices, variate_indices,
                        token_variate_ids=token_variate_ids,
                    )
            else:
                if self.gradient_checkpointing and self.training:
                    tokens = checkpoint(block, tokens, t_emb, use_reentrant=False)
                else:
                    tokens = block(tokens, t_emb)

        x_out = tokens[:, Nc:]  # (BV, Nx, D), drop cond slots
        shift, scale = self.final_adaLN(t_emb).chunk(2, dim=-1)
        x_out = _modulate(self.final_norm(x_out), shift, scale)
        out = self._unpatchify(x_out, gh, gw)

        if pad_h or pad_w:
            out = out[:, :, :H, :W]
        return out
```
```python
# models/diffusion_tsf/preprocessing.py:1-120
"""1D <-> hard CDF image encode/decode for DiffusionTSF.

TimeSeriesTo2D.forward: single-map CDF (joint stage).
encode_dual_heights / decode_dual*: coarse full-range + fine residual (staged).
Bounded variants serve ordinal ranks. Called from diffusion_model encode/decode helpers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class TimeSeriesTo2D(nn.Module):
    """Maps 1D values to hard CDF occupancy along the value axis (bottom-filled bars).

    Soft blur was removed; maps are binary 0/1 per row. inverse / _decode_* recover
    1D via mid-bin or column expectation.
    """

    def __init__(self, height: int = 32, max_scale: float = 3.5):
        """
        Args:
            height: Height H of the 2D representation (number of bins)
            max_scale: MS parameter - values beyond [-MS, MS] are clipped
        """
        super().__init__()
        self.height = height
        self.max_scale = max_scale
        
        # Precompute bin centers for inverse mapping
        # Centers: (j + 0.5) * (2*MS/H) - MS for j in [0, H-1]
        bin_width = (2 * max_scale) / height
        bin_centers = torch.tensor([
            (j + 0.5) * bin_width - max_scale 
            for j in range(height)
        ])
        self.register_buffer('bin_centers', bin_centers)
        
        logger.debug("TimeSeriesTo2D initialized: H=%s, MS=%s", height, max_scale)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """1D normalized series → hard CDF map in {0,1}.

        Univariate: (batch, seq_len) -> (batch, 1, height, seq_len)
        Multivariate: (batch, num_vars, seq_len) -> (batch, num_vars, height, seq_len)
        """
        # Handle univariate case: (batch, seq_len) -> (batch, 1, seq_len)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        batch_size, num_vars, seq_len = x.shape
        
        # Clip values to [-MS, MS] range
        x_clipped = torch.clamp(x, -self.max_scale, self.max_scale)
        
        # Calculate bin indices: (batch, num_vars, seq_len)
        # Formula: y = (x + MS) / (2*MS) * H, then clip to [0, H-1]
        bin_indices = ((x_clipped + self.max_scale) / (2 * self.max_scale) * self.height)
        bin_indices = torch.clamp(bin_indices.long(), 0, self.height - 1)

        height_range = torch.arange(self.height, device=x.device).view(1, 1, self.height, 1)
        filled = (height_range <= bin_indices.unsqueeze(2)).float()
        image = filled

        logger.debug(f"TimeSeriesTo2D: input {x.shape} -> output {image.shape}")
        return image

    def _cdf_from_bin_indices(self, bin_indices: torch.Tensor, height: Optional[int] = None) -> torch.Tensor:
        height = int(height or self.height)
        height_range = torch.arange(height, device=bin_indices.device).view(1, 1, height, 1)
        return (height_range <= bin_indices.unsqueeze(2)).float()

    def _encode_values_in_range(
        self,
        x: torch.Tensor,
        *,
        value_range: float,
        height: int,
    ) -> torch.Tensor:
        x_clipped = torch.clamp(x, -value_range, value_range)
        pos = (x_clipped + value_range) / (2 * value_range) * height
        bin_indices = torch.clamp(pos.long(), 0, height - 1)
        return self._cdf_from_bin_indices(bin_indices, height=height)

    def encode_dual(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode values as full-range coarse CDF plus within-bin residual CDF."""
        return self.encode_dual_heights(x, coarse_height=self.height, fine_height=self.height)

    def encode_dual_heights(
        self,
        x: torch.Tensor,
        *,
        coarse_height: int,
        fine_height: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Staged maps: coarse bins full [-MS,MS]; fine bins residual inside that coarse cell."""
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x_clipped = torch.clamp(x, -self.max_scale, self.max_scale)
        coarse_pos = (x_clipped + self.max_scale) / (2 * self.max_scale) * coarse_height
        coarse_bin = torch.clamp(coarse_pos.long(), 0, coarse_height - 1)
        coarse = self._cdf_from_bin_indices(coarse_bin, height=coarse_height)

        coarse_width = (2 * self.max_scale) / coarse_height
        coarse_center = (coarse_bin.to(x_clipped.dtype) + 0.5) * coarse_width - self.max_scale
        residual = x_clipped - coarse_center
        residual_range = self.max_scale / coarse_height
        residual = torch.clamp(residual, -residual_range, residual_range)
        fine_pos = (residual + residual_range) / (2 * residual_range) * fine_height
        fine_bin = torch.clamp(fine_pos.long(), 0, fine_height - 1)
        fine = self._cdf_from_bin_indices(fine_bin, height=fine_height)
        return coarse, fine

    def encode_dual_heights_bounded(
        self,
```
```python
# models/diffusion_tsf/preprocessing.py:168-310
    def _decode_occupancy_bounded(
        self,
        cdf_map: torch.Tensor,
        *,
        value_min: float,
        value_max: float,
        cdf_decoder: str = "mean",
        expectation_sharpen_temp: Optional[float] = None,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        cdf_map = torch.clamp(cdf_map, 0.0, 1.0)
        height = cdf_map.shape[2]
        span = float(value_max) - float(value_min)
        if span <= 0.0:
            return torch.full(
                (cdf_map.shape[0], cdf_map.shape[1], cdf_map.shape[3]),
                float(value_min),
                device=cdf_map.device,
                dtype=cdf_map.dtype,
            )
        del expectation_sharpen_temp, eps
        if cdf_decoder != "mean":
            raise ValueError(f"cdf_decoder must be 'mean', got {cdf_decoder!r}")

        column_sum = cdf_map.sum(dim=2).clamp(1.0, float(height))
        bin_idx = (column_sum - 1.0).clamp(0.0, float(height - 1))
        return (bin_idx + 0.5) / float(height) * span + float(value_min)

    def decode_dual_heights_bounded(
        self,
        coarse_map: torch.Tensor,
        fine_map: torch.Tensor,
        *,
        value_min: float = 0.0,
        value_max_per_variate: torch.Tensor | list[float] | float,
        cdf_decoder: str = "mean",
        expectation_sharpen_temp: Optional[float] = None,
        squeeze_univariate: bool = True,
    ) -> torch.Tensor:
        if coarse_map.shape[:2] != fine_map.shape[:2] or coarse_map.shape[3] != fine_map.shape[3]:
            raise ValueError(f"coarse/fine shapes differ: {coarse_map.shape} vs {fine_map.shape}")
        _batch_size, num_vars, coarse_height, _seq_len = coarse_map.shape
        if isinstance(value_max_per_variate, (int, float)):
            vmax = torch.full((num_vars,), float(value_max_per_variate), device=coarse_map.device, dtype=coarse_map.dtype)
        elif isinstance(value_max_per_variate, list):
            vmax = torch.tensor(value_max_per_variate, device=coarse_map.device, dtype=coarse_map.dtype)
        else:
            vmax = value_max_per_variate.to(device=coarse_map.device, dtype=coarse_map.dtype).reshape(-1)

        coarse_vals = []
        fine_vals = []
        vmin = float(value_min)
        for vi in range(num_vars):
            span = float(vmax[vi].item()) - vmin
            fine_range = span / float(coarse_height) * 0.5 if span > 0.0 else 0.0
            coarse_vals.append(
                self._decode_occupancy_bounded(
                    coarse_map[:, vi : vi + 1],
                    value_min=vmin,
                    value_max=vmin + span,
                    cdf_decoder=cdf_decoder,
                    expectation_sharpen_temp=expectation_sharpen_temp,
                )
            )
            fine_vals.append(
                self._decode_occupancy_bounded(
                    fine_map[:, vi : vi + 1],
                    value_min=-fine_range,
                    value_max=fine_range,
                    cdf_decoder=cdf_decoder,
                    expectation_sharpen_temp=expectation_sharpen_temp,
                )
            )
        coarse_value = torch.cat(coarse_vals, dim=1)
        fine_value = torch.cat(fine_vals, dim=1)
        x = coarse_value + fine_value
        if squeeze_univariate and num_vars == 1:
            x = x.squeeze(1)
        return x



    def _decode_occupancy_in_range(
        self,
        cdf_map: torch.Tensor,
        value_range: float,
        cdf_decoder: str = "mean",
        expectation_sharpen_temp: Optional[float] = None,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        cdf_map = torch.clamp(cdf_map, 0.0, 1.0)
        height = cdf_map.shape[2]
        del expectation_sharpen_temp, eps
        if cdf_decoder != "mean":
            raise ValueError(f"cdf_decoder must be 'mean', got {cdf_decoder!r}")

        column_sum = cdf_map.sum(dim=2).clamp(1.0, float(height))
        bin_idx = (column_sum - 1.0).clamp(0.0, float(height - 1))
        return ((bin_idx + 0.5) / float(height) * (2 * value_range)) - value_range

    def decode_dual(
        self,
        coarse_map: torch.Tensor,
        fine_map: torch.Tensor,
        cdf_decoder: str = "mean",
        expectation_sharpen_temp: Optional[float] = None,
        squeeze_univariate: bool = True,
    ) -> torch.Tensor:
        """Decode full-range coarse CDF plus residual CDF back to normalized values."""
        if coarse_map.shape[:2] != fine_map.shape[:2] or coarse_map.shape[3] != fine_map.shape[3]:
            raise ValueError(f"coarse/fine shapes differ: {coarse_map.shape} vs {fine_map.shape}")
        batch_size, num_vars, coarse_height, seq_len = coarse_map.shape

        coarse_value = self._decode_occupancy_in_range(
            coarse_map,
            value_range=self.max_scale,
            cdf_decoder=cdf_decoder,
            expectation_sharpen_temp=expectation_sharpen_temp,
        )
        fine_value = self._decode_occupancy_in_range(
            fine_map,
            value_range=self.max_scale / coarse_height,
            cdf_decoder=cdf_decoder,
            expectation_sharpen_temp=expectation_sharpen_temp,
        )
        x = coarse_value + fine_value
        x = torch.clamp(x, -self.max_scale, self.max_scale)
        if squeeze_univariate and num_vars == 1:
            x = x.squeeze(1)
        logger.debug(
            "TimeSeriesTo2D.decode_dual: input %s/%s -> output %s",
            coarse_map.shape,
            fine_map.shape,
            x.shape,
        )
        return x

    @staticmethod




    def bin_indices_from_cdf(cdf_map: torch.Tensor) -> torch.Tensor:
```
```python
# models/diffusion_tsf/ordinal_window_norm.py:1-100
"""Global training-set ordinal encoding (replaces window/instance norm when enabled).

Build ladder once from train split (build_global_ladder_from_training), stash on
config.ordinal_ladder. DiffusionTSF._normalize_sequence calls ordinal_encode /
ordinal_decode. Ranks are floats 0..K-1 per variate; CDF maps use bounded encode.

_value_to_rank_slow is a dead reference impl; live path is _value_to_rank.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch


@dataclass
class OrdinalLadder:
    """Global per-variate uniquified value ladder, padded to n_unique_max."""

    values: torch.Tensor  # (B, V, K)
    n_unique: torch.Tensor  # (B, V) int64
    tie_atol: float
    precomputed_ranks: Optional[torch.Tensor] = None  # (T, V) float32, optional

    def rank_max_per_variate(self) -> torch.Tensor:
        """Inclusive max rank index per variate (0 when only one unique value).

        Always uses row 0 so batch-expanded ladders (expand_batch) still broadcast
        as (V,) / (1, V, 1), not (B*V,).
        """
        return (self.n_unique[0] - 1).clamp_min(0)

    def z_envelope(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-variate train ladder min/max z-scores, shape (V,)."""
        v = self.values.shape[1]
        k_max = int(self.n_unique[0].max().item())
        mins = []
        maxs = []
        for vi in range(v):
            k = int(self.n_unique[0, vi].item())
            uniq = self.values[0, vi, :k]
            mins.append(uniq[0])
            maxs.append(uniq[k - 1])
        return torch.stack(mins), torch.stack(maxs)

    def expand_batch(self, batch_size: int) -> "OrdinalLadder":
        if self.values.shape[0] == batch_size:
            return self
        if self.values.shape[0] != 1:
            raise ValueError(f"cannot expand ladder batch {self.values.shape[0]} -> {batch_size}")
        return OrdinalLadder(
            values=self.values.expand(batch_size, -1, -1),
            n_unique=self.n_unique.expand(batch_size, -1),
            tie_atol=self.tie_atol,
            precomputed_ranks=self.precomputed_ranks,
        )


def _unique_sorted_1d_np(x: np.ndarray, tie_atol: float) -> Tuple[np.ndarray, int]:
    if x.size == 0:
        return np.zeros(0, dtype=np.float64), 0
    xs = np.sort(x.reshape(-1))
    groups = [float(xs[0])]
    for v in xs[1:]:
        if abs(float(v) - groups[-1]) > tie_atol:
            groups.append(float(v))
    uniq = np.asarray(groups, dtype=np.float64)
    return uniq, int(uniq.size)


def _value_to_rank_slow(values: torch.Tensor, x: torch.Tensor, tie_atol: float) -> torch.Tensor:
    """Unused reference: full (N,K) distance matrix. Prefer _value_to_rank."""
    k = values.shape[0]
    if k <= 1:
        return torch.zeros_like(x, dtype=torch.long)
    flat = x.reshape(-1)
    dist = (flat.unsqueeze(-1) - values.unsqueeze(0)).abs()
    tie_hit = dist <= tie_atol
    has_tie = tie_hit.any(dim=-1)
    tie_rank = tie_hit.int().argmax(dim=-1)
    nearest = dist.argmin(dim=-1)
    ranks = torch.where(has_tie, tie_rank, nearest)
    return ranks.reshape(x.shape).clamp(0, k - 1)


def _value_to_rank(values: torch.Tensor, x: torch.Tensor, tie_atol: float) -> torch.Tensor:
    """Map x to ladder rung index via nearest uniquified value (no semi-infinite outer bins)."""
    k = values.shape[0]
    if k <= 1:
        return torch.zeros_like(x, dtype=torch.long)
    flat = x.reshape(-1).contiguous()
    idx_r = torch.searchsorted(values, flat)
    left_i = (idx_r - 1).clamp(0, k - 1)
    right_i = idx_r.clamp(0, k - 1)

    left_d = (flat - values[left_i]).abs()
    right_d = (flat - values[right_i]).abs()
```
```python
# models/diffusion_tsf/config.py:1-160
"""Settings bag for DiffusionTSF. Pipeline YAML lands here via create_diffusion_model.

diffusion_stage picks the train/generate path: joint | coarse | fine | patch_refine.
Channel counts (backbone_in_channels, visual_cond_channels) must match what the
forward path cats; mismatch fails at DiT construct or first forward.
__post_init__ fails fast on removed options (non-binary, non-canvas guidance, etc.).

Dead / misleading fields still validated: anchor_mse_proxy_lambda (unused in loss),
emd_lambda / use_monotonicity_loss (unused), representation_mode allows 'pdf' then
decode rejects it, unified_time_axis unused.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple


@dataclass
class DiffusionTSFConfig:
    """Settings for binary CDF diffusion with FactorizedDiT."""

    # seq lens
    lookback_length: int = 512
    forecast_length: int = 96
    # YAML horizon (before overlap); used for AR when > diffusion_chunk_horizon.
    dataset_forecast_length: int = 0
    # Cap 2D past conditioning width; 0 = legacy min(past_len, target_width).
    diffusion_lookback_cap: int = 0
    # Fixed denoiser chunk width; 0 = use full dataset_forecast_length.
    diffusion_chunk_horizon: int = 0
    # Subsample timesteps before 2D encode (x[..., ::stride]); decode upsamples linearly.
    representation_time_stride: int = 1
    # When False, past 2D cond keeps native lookback width (e.g. 336); DiT cond tokens
    # are separate from the wider horizon canvas. Fine-stage horizon cond channels are
    # zero-padded to horizon width before channel concat, never bilinearly stretched.
    past_cond_resize_to_horizon: bool = True
    # Lookback overlap: predict the last K observed timesteps alongside the
    # future horizon to smooth the past/future boundary.
    lookback_overlap: int = 8

    # multivariate support
    num_variables: int = 1
    variate_factorized: bool = True
    use_variate_embedding: bool = True
    disable_cross_attention: bool = False

    # 2d mapping (hard binary CDF, no vertical blur)
    image_height: int = 32
    coarse_image_height: int = 16
    fine_image_height: int = 16
    patch_refine_canvas_height: int = 256
    patch_refine_patch_height: int = 32
    patch_refine_patch_width: int = 8
    patch_refine_col_stride: int = 6
    # Unique absolute 8-step segments + AR prev-refine cond (see patch_refine_segments).
    patch_refine_unique_segments: bool = False
    patch_refine_prev_cond_dropout: float = 0.5
    max_scale: float = 3.5
    representation_mode: str = "cdf"  # pdf rejected at decode; assert still allows it
    staged_representation: str = "value_precision"

    # unused leftover
    unified_time_axis: bool = False

    diffusion_type: str = "binary"
    use_ordinal_window_norm: bool = False
    # Derive any ordinal OOD envelope shift from the lookback alone so an
    # unseen future cannot change the forecast coordinate system.
    ordinal_ood_shift_causal_only: bool = False
    ordinal_tie_atol: float = 1e-6
    ordinal_ladder: Optional[Any] = None
    binary_num_steps: int = 1000
    binary_sample_steps: int = 20
    binary_beta_start: float = 1e-5
    binary_beta_end: float = 0.5
    binary_noise_schedule: str = "linear"
    diffusion_stage: str = "joint"  # joint, coarse, fine, patch_refine
    # Soft-decode MSE mix inside deterministic anchor: λ*BCE + (1-λ)*MSE.
    # MSE is unit-rank (or already O(1) window-norm) so it stays BCE-scale.
    # Note: λ is the BCE weight (λ=1 → BCE-only), despite the "mse" name.
    # Plumbed through pipeline but never read in the loss path today.
    anchor_mse_proxy_lambda: float = 0.5

    # classifier-free guidance (training dropout only; inference is always conditional)
    cfg_dropout: float = 0.1

    # 2d augs (cutout)
    cutout_prob: float = 0.5
    cutout_min_masks: int = 1
    cutout_max_masks: int = 3
    cutout_shapes: List[Tuple[int, int]] = field(
        default_factory=lambda: [(16, 16), (32, 5)]
    )

    # unused in diffusion_model loss
    emd_lambda: float = 0.2
    use_monotonicity_loss: bool = False
    monotonicity_weight: float = 1.0
    guidance_penalty_weight: float = 0.0

    # Deterministic anchor loss at max-noise stationary Bernoulli(0.5) state.
    use_deterministic_anchor_loss: bool = False
    deterministic_anchor_lambda: float = 0.99
    binary_anchor_input_mode: str = "stationary_flat"
    use_window_normalization: bool = True
    # Center for per-window norm: lookback mean only.
    window_norm_center: str = "mean"
    window_norm_std_floor: float = 1e-8
    # When past_std < threshold (z-score units), divide by unit_std instead of std_floor.
    window_norm_low_var_threshold: float = 0.0
    window_norm_low_var_unit_std: float = 1.0
    # Per local variate index (batch dim V); falls back to window_norm_low_var_unit_std.
    window_norm_low_var_unit_std_per_variate: Optional[List[float]] = None
    # Shift window center at decode so overlap preds align with past tail (quantization fix).
    lookback_overlap_center_shift: bool = False
    prediction_target: str = "x0"  # x0 or epsilon (bit-flip mask)
    loss_weighting: str = "none"  # none or min_snr
    min_snr_gamma: float = 5.0

    model_type: str = "dit"

    # DiT backbone
    dit_patch_size: Tuple[int, int] = (8, 8)
    dit_cond_patch_size: Optional[Tuple[int, int]] = None
    dit_embed_dim: int = 384
    dit_depth: int = 8
    dit_num_heads: int = 6
    dit_mlp_ratio: float = 4.0
    dit_dropout: float = 0.0

    # memory optimization
    use_gradient_checkpointing: bool = False
    unet_max_chunk_size: int = 128  # chunks BV through FactorizedDiT
    use_amp: bool = False

    # Extra visual cond: coarse CDF of lookback tail in dataset z-score space.
    use_raw_lookback_cond_channel: bool = False

    # Stage 1 guidance (ghost image + encoder tokens)
    use_guidance_channel: bool = False
    guidance_placement: str = "canvas"  # only canvas allowed; cond_chunks code is dead
    context_embedding_dim: int = 256
    guidance_type: str = "patch_decoder"
    mmpd_patch_size: int = 12

    # train
    learning_rate: float = 2e-4
    batch_size: int = 8

    def __post_init__(self):
        assert self.image_height > 0
        assert self.max_scale > 0
        if self.diffusion_type != "binary":
            raise ValueError(f"diffusion_type must be 'binary', got {self.diffusion_type!r}")
        if self.use_ordinal_window_norm and self.use_window_normalization:
            raise ValueError(
                "use_ordinal_window_norm replaces window normalization; set use_window_normalization=false"
            )
        if self.binary_anchor_input_mode != "stationary_flat":
            raise ValueError(
                "binary_anchor_input_mode must be 'stationary_flat', "
```

Patch geometry / segments / helpers (crops, naive upscale, blend):

```python
# models/diffusion_tsf/patch_refine_geometry.py:1-120
"""Boundary-centered crop geometry for high-resolution CDF refinement.

coarse_edges_from_cdf -> select_patch_locations / extract_patch_batch /
blend_patch_bins. Called from DiffusionTSF._forward/_generate_binary_patch_refine.
Crops sit vertically on the coarse transition row so the DiT sees the boundary.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from .preprocessing import TimeSeriesTo2D

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PatchLocation:
    """One crop in a flattened ``B * V`` future canvas."""

    flat_index: int
    batch_index: int
    variate_index: int
    row0: int
    col0: int


def coarse_edges_from_cdf(
    coarse_cdf: torch.Tensor,
    *,
    canvas_height: int,
) -> torch.Tensor:
    """Return nearest-neighbour-upscaled coarse boundary rows, shape ``(B,V,W)``."""
    if coarse_cdf.ndim != 4:
        raise ValueError(f"coarse_cdf must be (B,V,H,W), got {tuple(coarse_cdf.shape)}")
    coarse_height = int(coarse_cdf.shape[-2])
    if canvas_height % coarse_height:
        raise ValueError(
            f"canvas_height={canvas_height} must be divisible by coarse height={coarse_height}"
        )
    scale = canvas_height // coarse_height
    bins = TimeSeriesTo2D.bin_indices_from_cdf(coarse_cdf).round().long()
    return ((bins + 1) * scale - 1).clamp(0, canvas_height - 1)


def select_patch_locations(
    coarse_edges: torch.Tensor,
    *,
    canvas_height: int,
    patch_height: int,
    patch_width: int,
    col_stride: int,
    max_patches_per_variate: Optional[int] = None,
) -> list[PatchLocation]:
    """Place stride crops, then add fill-ins until every timestep boundary is covered."""
    if coarse_edges.ndim != 3:
        raise ValueError(f"coarse_edges must be (B,V,W), got {tuple(coarse_edges.shape)}")
    if patch_height > canvas_height:
        raise ValueError("patch height exceeds canvas height")
    width = int(coarse_edges.shape[-1])
    if patch_width > width:
        raise ValueError(f"patch width {patch_width} exceeds future width {width}")
    if col_stride <= 0:
        raise ValueError("col_stride must be positive")

    batch_size, n_variates, _ = coarse_edges.shape
    max_row0 = canvas_height - patch_height
    max_col0 = width - patch_width
    primary_starts = list(range(0, max_col0 + 1, col_stride))
    # Primary stride crops + at most one fill-in per uncovered timestep.
    soft_cap = len(primary_starts) + width
    hard_cap = int(max_patches_per_variate) if max_patches_per_variate is not None else soft_cap
    if hard_cap < len(primary_starts):
        raise ValueError(
            f"max_patches_per_variate={hard_cap} < primary crop count {len(primary_starts)}"
        )
    locations: list[PatchLocation] = []
    n_primary_total = 0
    n_fallback_total = 0

    for bi in range(batch_size):
        for vi in range(n_variates):
            flat_index = bi * n_variates + vi
            edges = coarse_edges[bi, vi]
            covered = torch.zeros(width, device=edges.device, dtype=torch.bool)
            seen: set[tuple[int, int]] = set()
            n_before = len(locations)

            def add_crop(col0: int, anchor_t: int) -> None:
                row0 = max(
                    0,
                    min(int(edges[anchor_t].item()) - patch_height // 2, max_row0),
                )
                key = (row0, col0)
                if key not in seen:
                    if len(seen) >= hard_cap:
                        raise RuntimeError(
                            f"patch_refine crop cap exceeded at B={bi} V={vi}: "
                            f"cap={hard_cap} (primary={len(primary_starts)}, width={width})"
                        )
                    seen.add(key)
                    locations.append(
                        PatchLocation(
                            flat_index=flat_index,
                            batch_index=bi,
                            variate_index=vi,
                            row0=row0,
                            col0=col0,
                        )
                    )
                cols = torch.arange(
                    col0,
                    col0 + patch_width,
                    device=edges.device,
                )
                in_rows = (edges[cols] >= row0) & (edges[cols] < row0 + patch_height)
```
```python
# models/diffusion_tsf/patch_refine_segments.py:1-100
"""Unique absolute patch-refine segments + previous-stride teacher force.

When patch_refine_unique_segments is on, training samples absolute col0 crops
(and parent windows) instead of dense stride crops. Infer AR-chains primary
col0 groups via select_primary_ar_locations, then fills gaps with blanked prev.
Wired from DiffusionTSF._forward/_generate_binary_patch_refine.
"""

from __future__ import annotations

import hashlib
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .patch_refine_geometry import PatchLocation


def iter_unique_segment_starts(
    series_len: int,
    *,
    lookback: int,
    horizon: int,
    overlap: int,
    patch_width: int,
    segment_stride: int = 1,
) -> List[int]:
    """Absolute left edges ``t`` of every unique patch that fits in some parent.

    ``segment_stride`` should match the data-subset sample stride (e.g. dynamic
    480) so sparse series do not explode into dense absolute indices.
    """
    segment_stride = max(1, int(segment_stride))
    patch_width = int(patch_width)
    fut_w = int(horizon) + int(overlap)
    if patch_width > fut_w:
        raise ValueError(f"patch_width {patch_width} > future width {fut_w}")
    # Absolute future of parent S covers [S+lb-K, S+lb+hz).
    t_min = int(lookback) - int(overlap)
    t_max = int(series_len) - patch_width
    # Need at least one parent: S in [0, series_len-lb-hz] with patch inside future.
    max_S = int(series_len) - int(lookback) - int(horizon)
    if max_S < 0 or t_max < t_min:
        return []
    starts: List[int] = []
    for t in range(t_min, t_max + 1, segment_stride):
        # Parent S must satisfy: S+lb-K <= t <= S+lb+hz-patch_width
        # => t - (lb+hz-pw) <= S <= t - (lb-K)
        s_lo = t - (int(lookback) + int(horizon) - patch_width)
        s_hi = t - (int(lookback) - int(overlap))
        s_lo = max(0, s_lo)
        s_hi = min(max_S, s_hi)
        if s_lo <= s_hi:
            starts.append(int(t))
    return starts


def parent_starts_for_segment(
    t: int,
    *,
    lookback: int,
    horizon: int,
    overlap: int,
    patch_width: int,
    series_len: int,
) -> List[int]:
    max_S = int(series_len) - int(lookback) - int(horizon)
    if max_S < 0:
        return []
    s_lo = t - (int(lookback) + int(horizon) - int(patch_width))
    s_hi = t - (int(lookback) - int(overlap))
    s_lo = max(0, s_lo)
    s_hi = min(max_S, s_hi)
    if s_lo > s_hi:
        return []
    return list(range(s_lo, s_hi + 1))


def sample_parent_start(
    t: int,
    *,
    epoch: int,
    series_id: int,
    lookback: int,
    horizon: int,
    overlap: int,
    patch_width: int,
    series_len: int,
) -> int:
    parents = parent_starts_for_segment(
        t,
        lookback=lookback,
        horizon=horizon,
        overlap=overlap,
        patch_width=patch_width,
        series_len=series_len,
    )
```
```python
# models/diffusion_tsf/patch_refine.py:1-181
"""Boundary-patch CDF refinement helpers used by DiffusionTSF patch_refine paths.

Geometry (where to crop) lives in patch_refine_geometry; unique AR segments in
patch_refine_segments. This file: tall CDF encode/decode, aux channel pack,
and expand_* to turn BV tensors into per-crop batches for the DiT.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .patch_refine_geometry import (
    PatchLocation,
    blend_patch_bins,
    coarse_edges_from_cdf,
    extract_patch_batch,
    select_patch_locations,
)
from .preprocessing import TimeSeriesTo2D


def naive_upscale_coarse_cdf(coarse_cdf: torch.Tensor, canvas_height: int) -> torch.Tensor:
    """Nearest-neighbor vertical upscale of coarse CDF to ``canvas_height`` (scaffold)."""
    if coarse_cdf.ndim != 4:
        raise ValueError(f"coarse_cdf must be (B,V,H,W), got {tuple(coarse_cdf.shape)}")
    if coarse_cdf.shape[2] == canvas_height:
        return coarse_cdf
    flat = coarse_cdf.reshape(-1, 1, coarse_cdf.shape[2], coarse_cdf.shape[3])
    up = F.interpolate(flat, size=(canvas_height, coarse_cdf.shape[3]), mode="nearest")
    return up.reshape(coarse_cdf.shape[0], coarse_cdf.shape[1], canvas_height, coarse_cdf.shape[3])


def stack_past_coarse_fine(
    past_coarse: torch.Tensor,
    past_fine: torch.Tensor,
) -> torch.Tensor:
    """Lossless stack to ``(B,V,Hc+Hf,W)`` then flatten to ``(BV,1,H,W)``."""
    if past_coarse.shape[:2] != past_fine.shape[:2] or past_coarse.shape[-1] != past_fine.shape[-1]:
        raise ValueError(
            f"past coarse/fine shape mismatch: {tuple(past_coarse.shape)} vs {tuple(past_fine.shape)}"
        )
    stacked = torch.cat([past_coarse, past_fine], dim=2)
    b, v, h, w = stacked.shape
    return stacked.reshape(b * v, 1, h, w)


def encode_absolute_hir_cdf(
    values: torch.Tensor,
    *,
    canvas_height: int,
    max_scale: float,
    ordinal_rank_max: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Encode absolute hi-res CDF ``(B,V,H_hi,W)`` from 1D values."""
    if values.dim() == 2:
        values = values.unsqueeze(1)
    if ordinal_rank_max is not None:
        vmax = ordinal_rank_max.to(device=values.device, dtype=values.dtype).reshape(-1)
        maps = []
        for vi in range(values.shape[1]):
            span = float(vmax[vi].item())
            xi = values[:, vi : vi + 1]
            if span <= 0.0:
                bins = torch.zeros_like(xi, dtype=torch.long)
            else:
                pos = (xi.clamp(0.0, span) / span) * canvas_height
                bins = pos.long().clamp(0, canvas_height - 1)
            rows = torch.arange(canvas_height, device=values.device).view(1, 1, canvas_height, 1)
            maps.append((rows <= bins.unsqueeze(2)).to(values.dtype))
        return torch.cat(maps, dim=1)

    x_clipped = values.clamp(-max_scale, max_scale)
    pos = (x_clipped + max_scale) / (2 * max_scale) * canvas_height
    bins = pos.long().clamp(0, canvas_height - 1)
    rows = torch.arange(canvas_height, device=values.device).view(1, 1, canvas_height, 1)
    return (rows <= bins.unsqueeze(2)).to(values.dtype)


def decode_absolute_hir_cdf(
    hir_cdf: torch.Tensor,
    *,
    max_scale: float,
    ordinal_rank_max: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mid-bin decode of absolute hi-res CDF to 1D values ``(B,V,W)``."""
    bins = TimeSeriesTo2D.bin_indices_from_cdf(hir_cdf)
    height = float(hir_cdf.shape[-2])
    mid = (bins + 0.5) / height
    if ordinal_rank_max is not None:
        vmax = ordinal_rank_max.to(device=hir_cdf.device, dtype=hir_cdf.dtype).reshape(1, -1, 1)
        return mid * vmax
    return mid * (2 * max_scale) - max_scale


def build_patch_aux_channels(
    naive_canvas: torch.Tensor,
    coarse_edges: torch.Tensor,
    locations: Sequence[PatchLocation],
    *,
    patch_height: int,
    patch_width: int,
    canvas_height: int,
    coarse_height: int,
    horizon_width: int,
    prev_refine_16: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(aux, patch_coarse_bin, patch_time0)``.

    ``aux`` is ``(N,3,ph,pw)`` = naive crop, coarse-cell id map, absolute-time map.

    When ``prev_refine_16`` is provided ``(N,1,16,pw)``, it is written into the
    **top 16 rows** of the coarse-cell channel (which is otherwise H-constant).
    The bottom 16 rows keep the real per-column coarse-cell values.
    """
    naive_patches = extract_patch_batch(
        naive_canvas, locations, patch_height=patch_height, patch_width=patch_width,
    )
    n = len(locations)
    device = naive_canvas.device
    dtype = naive_canvas.dtype
    coarse_cell = torch.zeros(n, 1, patch_height, patch_width, device=device, dtype=dtype)
    time_map = torch.zeros_like(coarse_cell)
    patch_coarse_bin = torch.zeros(n, device=device, dtype=torch.long)
    patch_time0 = torch.zeros(n, device=device, dtype=torch.long)
    denom_bin = max(1, coarse_height - 1)
    denom_t = max(1, horizon_width - 1)

    for i, loc in enumerate(locations):
        cols = torch.arange(loc.col0, loc.col0 + patch_width, device=device)
        edges = coarse_edges[loc.batch_index, loc.variate_index, cols]
        # Invert NN-upscale edge formula: edge = (bin+1)*scale - 1.
        scale = canvas_height // coarse_height
        bins = ((edges + 1) // scale - 1).clamp(0, coarse_height - 1)
        cell = (bins.float() / float(denom_bin)).view(1, 1, 1, patch_width)
        coarse_cell[i] = cell.expand(1, 1, patch_height, patch_width)
        tnorm = (cols.float() / float(denom_t)).view(1, 1, 1, patch_width)
        time_map[i] = tnorm.expand(1, 1, patch_height, patch_width)
        mid = patch_width // 2
        patch_coarse_bin[i] = bins[mid]
        patch_time0[i] = loc.col0

    if prev_refine_16 is not None:
        if prev_refine_16.shape != (n, 1, 16, patch_width):
            raise ValueError(
                f"prev_refine_16 must be {(n, 1, 16, patch_width)}, "
                f"got {tuple(prev_refine_16.shape)}"
            )
        if patch_height < 16:
            raise ValueError("prev_refine_16 stuffing requires patch_height >= 16")
        # Reclaim H-padding of the H-constant coarse-cell map.
        coarse_cell = coarse_cell.clone()
        coarse_cell[:, :, :16, :] = prev_refine_16.to(device=device, dtype=dtype)

    aux = torch.cat([naive_patches, coarse_cell, time_map], dim=1)
    return aux, patch_coarse_bin, patch_time0


def expand_lookback_cond_for_patches(
    lookback_cond: torch.Tensor,
    locations: Sequence[PatchLocation],
) -> torch.Tensor:
    """``(BV,1,H,Lb)`` → ``(N,1,H,Lb)`` by indexing each crop's flat variate row."""
    return torch.stack([lookback_cond[loc.flat_index] for loc in locations], dim=0)


def expand_ctx_for_patches(
    ctx_flat: Optional[torch.Tensor],
    locations: Sequence[PatchLocation],
) -> Optional[torch.Tensor]:
    if ctx_flat is None:
        return None
    return torch.stack([ctx_flat[loc.flat_index] for loc in locations], dim=0)


def expand_variate_indices_for_patches(
    locations: Sequence[PatchLocation],
    device: torch.device,
) -> torch.Tensor:
```

## 10. Staged accuracy eval

Loads coarse + second-stage ckpts, generates once (or AR), logs `eval/staged_*`.

```python
# models/diffusion_tsf/pipeline/phases/staged_eval.py:1-50
"""Chained coarse → fine/patch_refine eval for leaderboard metrics.

Loads both stage ckpts, runs anchor (+ optional quad_t samples), writes
partials + raw npz + wandb eval/staged_*. Core chain is _staged_generate_once.
AR chunking is optional when diffusion_chunk_horizon < dataset horizon.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from models.diffusion_tsf.pipeline.phase import PipelinePhase
from models.diffusion_tsf.pipeline.state import PipelineState
from models.diffusion_tsf.pipeline import wandb_utils
from models.diffusion_tsf.pipeline.config import visualization_settings
from models.diffusion_tsf.pipeline.visualize_utils import (
    decode_staged_anchor_components,
    per_window_anchor_mse,
    per_window_crps,
    run_eval_probabilistic_sample_visualizations,
    run_eval_full_dataset_visualization,
    run_eval_worst_window_visualizations,
    run_ordinal_roundtrip_visualization,
    run_ordinal_coarse_fine_2d_visualization,
    run_real_dataset_phase_diagnostics,
    run_staged_finetune_visualizations,
)
from models.diffusion_tsf.pipeline.phases.staged_diffusion_finetune_hp import (
    _model_kwargs_from_tuned,
    _stage_best_ckpt,
)
from models.diffusion_tsf.pipeline.phases.staged_diffusion_pretrain import patch_stage_globals

logger = logging.getLogger(__name__)


def _reshape_parallel_samples(t: torch.Tensor, batch: int, n_samples: int) -> torch.Tensor:
    """``(B*S, V, ...)`` → ``(B, V, S, ...)``."""
    if t.shape[0] != batch * n_samples:
        raise ValueError(
            f"parallel sample reshape expected leading {batch * n_samples}, got {tuple(t.shape)}"
```
```python
# models/diffusion_tsf/pipeline/phases/staged_eval.py:236-301
def _staged_generate_once(
    *,
    coarse_model,
    fine_model,
    past: torch.Tensor,
    gen_kwargs: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    """One-window chain: coarse.generate -> second_stage.generate(future_coarse_2d=...)."""
    coarse_out = coarse_model.generate(past, **gen_kwargs)
    fine_out = fine_model.generate(
        past,
        future_coarse_2d=coarse_out["future_2d_coarse"],
        **gen_kwargs,
    )
    pred = _staged_anchor_global_norm(fine_model, coarse_out, fine_out)
    pred_t = torch.from_numpy(pred).to(past.device)
    return {"coarse": coarse_out, "fine": fine_out, "prediction": pred_t}


def _staged_generate_autoregressive(
    *,
    coarse_model,
    fine_model,
    past: torch.Tensor,
    gen_kwargs: Dict[str, Any],
) -> torch.Tensor:
    """Roll out staged coarse/fine in AR chunks; return global-norm forecast (B,V,H)."""
    K = int(getattr(coarse_model.config, "lookback_overlap", 0))
    dataset_h = int(getattr(coarse_model.config, "dataset_forecast_length", 0) or 0)
    n_chunks = coarse_model._ar_num_chunks(dataset_h)
    pieces = []
    remaining = dataset_h
    for c in range(n_chunks):
        if c == 0:
            past_c = past
        else:
            hist = torch.cat(pieces, dim=-1)
            past_c = torch.cat([past, hist], dim=-1)
        out = _staged_generate_once(
            coarse_model=coarse_model,
            fine_model=fine_model,
            past=past_c,
            gen_kwargs=gen_kwargs,
        )
        chunk = out["prediction"]
        if isinstance(chunk, np.ndarray):
            chunk = torch.from_numpy(chunk).to(past.device)
        if c > 0:
            chunk = chunk[..., K:]
        if chunk.shape[-1] > remaining:
            chunk = chunk[..., :remaining]
        pieces.append(chunk)
        remaining -= chunk.shape[-1]
        if remaining <= 0:
            break
    return torch.cat(pieces, dim=-1)


def _staged_det_gen_kwargs(state: PipelineState, default_steps: int) -> Dict[str, Any]:
    sampler = str(getattr(state, "eval_sampler", "anchor"))
    if sampler in ("anchor", "deterministic_anchor"):
        return {"sampler": sampler}
    steps = 5 if state.smoke_test else int(default_steps)
    return {"sampler": sampler, "num_inference_steps": steps}
```
```python
# models/diffusion_tsf/pipeline/phases/staged_eval.py:346-549
    def _load_model(self, state: PipelineState, stage: str, itrans_guidance, n_iv: int, device: torch.device):
        from models.diffusion_tsf.train_multivariate_pipeline import (
            anchor_kwargs_from_params,
            create_diffusion_model,
            dataset_window_lengths,
            load_diffusion_state_keep_attached_guidance,
        )
        import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod

        patch_stage_globals(pipeline_mod, state, stage, honor_dataset_windows=True)
        ds_lb, ds_hz = dataset_window_lengths(state.dataset)
        meta = _load_stage_metadata(state, stage)
        tuned = meta.get("tuned_params") or {}
        model_kwargs = anchor_kwargs_from_params(tuned)
        model_kwargs.update(_model_kwargs_from_tuned(tuned))
        model = create_diffusion_model(
            n_variates=n_iv,
            lookback=ds_lb,
            horizon=ds_hz,
            guidance_model=itrans_guidance,
            diffusion_stage=stage,
            use_guidance_channel=state.use_guidance_channel,
            ordinal_ladder=pipeline_mod.GLOBAL_ORDINAL_LADDER,
            **model_kwargs,
        ).to(device)
        ckpt = torch.load(_stage_finetune_ckpt(state, stage), map_location=device, weights_only=False)
        load_diffusion_state_keep_attached_guidance(model, ckpt["model_state_dict"])
        model.eval()
        return model

    def _run_eval(
        self,
        *,
        state: PipelineState,
        subset_id: str,
        loader: DataLoader,
        device: torch.device,
        coarse_model,
        fine_model,
        prob_sampler: str,
        prob_steps: int,
        prob_samples: int,
        gmm_components: int,
        topk_max: int,
        window_indices: Sequence[int],
        test_stride: int,
    ) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        if prob_sampler in {"anchor", "deterministic_anchor"}:
            raise ValueError("staged probabilistic eval must use a regular sampler, not anchor.")
        prob_kwargs = {"sampler": prob_sampler, "num_inference_steps": prob_steps}
        det_kwargs = _staged_det_gen_kwargs(state, prob_steps)
        y_true_all = []
        y_true_with_overlap_all = []
        det_all = []
        det_with_overlap_all = []
        coarse_all = []
        fine_all = []
        sample_all = []
        samples_with_overlap_all = []
        window_idx_all = []
        t0 = time.perf_counter()
        ranked = getattr(loader.dataset, "yields_ordinal_ranks", False)
        if isinstance(loader.dataset, Subset):
            ranked = getattr(loader.dataset.dataset, "yields_ordinal_ranks", False)
        for m in (coarse_model, fine_model):
            m._ordinal_input_is_ranked = ranked
            m._ordinal_apply_ood_shift = not ranked
        logger.info(
            "[%s] staged eval start: windows=%d batches=%d prob_samples=%d sampler=%s steps=%d",
            subset_id,
            len(loader.dataset),
            len(loader),
            prob_samples,
            prob_sampler,
            prob_steps,
        )
        with torch.no_grad():
            for batch_idx, (past, future) in enumerate(loader):
                past = past.to(device)
                future = future.to(device)
                batch_n = past.shape[0]
                batch_start = batch_idx * loader.batch_size
                batch_window_indices = window_indices[batch_start:batch_start + batch_n]
                window_idx_all.extend(batch_window_indices)
                K = getattr(coarse_model.config, "lookback_overlap", 0)
                y_true_with_overlap_all.append(future.cpu().numpy())
                if K > 0:
                    future = future[..., K:]
                y_true_all.append(future.cpu().numpy())

                torch.manual_seed(state.seed + batch_idx)
                batch_t0 = time.perf_counter()
                if _ar_eval_enabled(coarse_model):
                    det_t = _staged_generate_autoregressive(
                        coarse_model=coarse_model,
                        fine_model=fine_model,
                        past=past,
                        gen_kwargs=det_kwargs,
                    )
                    det_all.append(det_t.detach().cpu().numpy())
                    coarse_det = coarse_model.generate(past, **det_kwargs)
                    fine_det = fine_model.generate(
                        past,
                        future_coarse_2d=coarse_det["future_2d_coarse"],
                        **det_kwargs,
                    )
                    coarse_np, fine_np, _ = decode_staged_anchor_components(
                        fine_model, coarse_det, fine_det,
                    )
                    coarse_all.append(coarse_np)
                    fine_all.append(fine_np)
                else:
                    coarse_det = coarse_model.generate(past, **det_kwargs)
                    fine_det = fine_model.generate(
                        past,
                        future_coarse_2d=coarse_det["future_2d_coarse"],
                        **det_kwargs,
                    )
                    coarse_np, fine_np, _final_np = decode_staged_anchor_components(
                        fine_model, coarse_det, fine_det,
                    )
                    det_all.append(fine_det["prediction_global_norm"].detach().cpu().numpy())
                    det_with_overlap_all.append(
                        fine_det["prediction_with_overlap"].detach().cpu().numpy()
                    )
                    coarse_all.append(coarse_np)
                    fine_all.append(fine_np)

                det_s = time.perf_counter() - batch_t0
                prob_t0 = time.perf_counter()
                # Expand window batch across independent MC samples so unique-seg
                # AR (and other generate paths) fill the GPU in one forward chain.
                torch.manual_seed(state.seed + batch_idx * 1009)
                past_exp = past.repeat_interleave(prob_samples, dim=0)
                if _ar_eval_enabled(coarse_model):
                    sample_t = _staged_generate_autoregressive(
                        coarse_model=coarse_model,
                        fine_model=fine_model,
                        past=past_exp,
                        gen_kwargs=prob_kwargs,
                    )
                    samples_bvs = _reshape_parallel_samples(sample_t, batch_n, prob_samples)
                    sample_all.append(samples_bvs.detach().cpu().numpy())
                else:
                    coarse_sample = coarse_model.generate(past_exp, **prob_kwargs)
                    fine_sample = fine_model.generate(
                        past_exp,
                        future_coarse_2d=coarse_sample["future_2d_coarse"],
                        **prob_kwargs,
                    )
                    pred = fine_sample["prediction_global_norm"]
                    overlap = fine_sample["prediction_with_overlap"]
                    samples_bvs = _reshape_parallel_samples(pred, batch_n, prob_samples)
                    overlap_bvs = _reshape_parallel_samples(overlap, batch_n, prob_samples)
                    sample_all.append(samples_bvs.detach().cpu().numpy())
                    samples_with_overlap_all.append(overlap_bvs.detach().cpu().numpy())

                prob_s = time.perf_counter() - prob_t0
                batch_s = time.perf_counter() - batch_t0
                done = batch_idx + 1
                elapsed = time.perf_counter() - t0
                eta_s = (elapsed / done) * (len(loader) - done) if done else 0.0
                logger.info(
                    "[%s] staged eval batch %d/%d n=%d "
                    "det=%.1fs prob=%.1fs (n_samp=%d parallel) batch=%.1fs "
                    "elapsed=%.1fs eta=%.1fs",
                    subset_id,
                    done,
                    len(loader),
                    batch_n,
                    det_s,
                    prob_s,
                    prob_samples,
                    batch_s,
                    elapsed,
                    eta_s,
                )

        pack = {
            "y_true": np.concatenate(y_true_all, axis=0),
            "deterministic": np.concatenate(det_all, axis=0),
            "coarse_anchor": np.concatenate(coarse_all, axis=0),
            "fine_anchor": np.concatenate(fine_all, axis=0),
            "final_anchor": np.concatenate(det_all, axis=0),
            "samples": np.concatenate(sample_all, axis=0),
            "window_indices": np.array(window_idx_all, dtype=np.int64),
            "series_starts": np.array(window_idx_all, dtype=np.int64) * int(test_stride),
        }
        if det_with_overlap_all and len(det_with_overlap_all) == len(det_all):
            pack["y_true_with_overlap"] = np.concatenate(y_true_with_overlap_all, axis=0)
            pack["deterministic_with_overlap"] = np.concatenate(det_with_overlap_all, axis=0)
            pack["final_anchor_with_overlap"] = pack["deterministic_with_overlap"]
        if samples_with_overlap_all and len(samples_with_overlap_all) == len(sample_all):
            pack["samples_with_overlap"] = np.concatenate(samples_with_overlap_all, axis=0)
            pack["sample_mean_with_overlap"] = pack["samples_with_overlap"].mean(axis=2)
        pack["sample_mean"] = pack["samples"].mean(axis=2)
        metrics = _summarize_staged_eval_metrics(
            pack,
            gmm_components=gmm_components,
            seed=state.seed,
            topk_max=topk_max,
        )
        return metrics, pack
```
```python
# models/diffusion_tsf/pipeline/phases/staged_eval.py:550-964
    def execute(self, state: PipelineState) -> PipelineState:
        # Load coarse + second-stage ckpts, score test set, write partials/raw/wandb.
        from models.diffusion_tsf.train_multivariate_pipeline import (
            generate_dataset_job,
            load_dataset,
            load_wrapped_guidance,
            dataset_window_lengths,
        )
        import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod
        from models.diffusion_tsf.pipeline.globals_bridge import patch_globals

        if getattr(state, "use_vertical_dual_concat", False):
            raise ValueError("staged_eval no longer supports vertical_dual")
        if getattr(state, "use_channel_dual_concat", False):
            raise ValueError("staged_eval no longer supports channel_dual")
        if getattr(state, "use_triple_scale", False):
            raise ValueError("staged_eval no longer supports finer/triple_scale")
        if bool(self.get("tune_sampler", False)):
            raise ValueError(
                "staged_eval no longer supports tune_sampler; set probabilistic_sampler explicitly"
            )

        device = state.resolve_device()
        gmm_components = int(self.require("gmm_components"))
        topk_max = int(self.require("topk_max"))
        subset_id = state.subset_id or state.dataset
        variate_indices = state.variate_indices
        if variate_indices is None:
            variate_indices = generate_dataset_job(state.dataset)["variate_indices"]
        subset_meta = state.data_subset_resolved or {}
        train_stride = int(subset_meta.get("train_stride", state.window_stride))
        phase_test_stride = int(self.require("test_stride"))
        subset_test_stride = int(subset_meta.get("test_stride", 1))
        # Never evaluate denser than the subset policy (e.g. dynamic sample_stride=480).
        test_stride = max(phase_test_stride, subset_test_stride)
        if test_stride != phase_test_stride:
            logger.info(
                "[%s] eval test_stride=%d (phase=%d, subset=%d)",
                subset_id,
                test_stride,
                phase_test_stride,
                subset_test_stride,
            )
        n_iv = len(variate_indices)

        patch_globals(pipeline_mod, state, honor_dataset_windows=True)
        full_train_ds, full_val_ds, full_test_ds, norm_stats = load_dataset(
            state.dataset,
            variate_indices,
            stride=train_stride,
            test_stride=test_stride,
            ordinal_tie_atol=float(state.ordinal_tie_atol),
            use_ordinal_window_norm=state.use_ordinal_window_norm,
        )
        if norm_stats.get("ordinal_ladder") is not None:
            state.extra["global_ordinal_ladder"] = norm_stats["ordinal_ladder"]

        ft_guidance_ckpt = state.guidance_finetune_ckpt
        if not ft_guidance_ckpt or not os.path.exists(ft_guidance_ckpt):
            ft_guidance_ckpt = state.default_guidance_finetune_ckpt_path()
        needs_guidance = state.needs_guidance
        if needs_guidance and not os.path.exists(ft_guidance_ckpt):
            raise FileNotFoundError(f"Missing finetuned guidance checkpoint: {ft_guidance_ckpt}")
        if not needs_guidance:
            ft_guidance_ckpt = ""

        ds_lb, ds_hz = dataset_window_lengths(state.dataset)
        guidance = None
        if needs_guidance:
            guidance = load_wrapped_guidance(
                ft_guidance_ckpt,
                n_iv,
                device,
                guidance_type=state.guidance_type,
                dataset_lookback=ds_lb,
                dataset_horizon=ds_hz,
            )
        patch_refine = bool(getattr(state, "use_patch_refine_stage", False))
        coarse_model = self._load_model(state, "coarse", guidance, n_iv, device)
        refine_stage = "patch_refine" if patch_refine else "fine"
        fine_model = self._load_model(state, refine_stage, guidance, n_iv, device)

        batch_size = int(self.require("batch_size"))
        if state.smoke_test:
            final_ds = Subset(full_test_ds, list(range(min(2, len(full_test_ds)))))
            prob_samples = 1
            default_steps = 5
        else:
            eval_fraction = _resolve_eval_test_fraction(self, state)
            final_ds = _fraction_subset(full_test_ds, eval_fraction, state.seed) if eval_fraction < 1.0 else full_test_ds
            if eval_fraction < 1.0:
                logger.info(
                    "[%s] eval subset: %d/%d windows (eval_test_fraction=%.3f)",
                    subset_id,
                    len(final_ds),
                    len(full_test_ds),
                    eval_fraction,
                )
            prob_samples = int(self.require("probabilistic_n_samples"))
            default_steps = int(self.require("probabilistic_num_inference_steps"))

        # Probe peak generate batch on this GPU; dataloader batch is smaller so
        # that B_windows * n_prob_samples still fits the parallel MC expand.
        if (
            bool(self.get("probe_eval_batch_size", False))
            and not state.smoke_test
            and device.type == "cuda"
        ):
            probe_kwargs = dict(_staged_det_gen_kwargs(state, default_steps))
            probe_kwargs["num_inference_steps"] = 1
            max_fit = _probe_max_staged_eval_batch_size(
                coarse_model=coarse_model,
                fine_model=fine_model,
                lookback=int(ds_lb),
                n_variates=n_iv,
                device=device,
                det_kwargs=probe_kwargs,
                min_bs=1,
                max_bs=int(self.get("probe_eval_batch_size_max", 64)),
            )
            # Parallel samples expand leading dim by prob_samples.
            usable = max(1, max_fit // max(1, int(prob_samples)))
            if usable != batch_size:
                logger.info(
                    "[%s] staged_eval probe: config batch_size=%d -> probed=%d "
                    "(max_fit=%d / n_samples=%d)",
                    subset_id,
                    batch_size,
                    usable,
                    max_fit,
                    prob_samples,
                )
            batch_size = usable

        if isinstance(final_ds, Subset):
            eval_window_indices = [int(i) for i in final_ds.indices]
        else:
            eval_window_indices = list(range(len(final_ds)))

        try:
            from models.diffusion_tsf.pipeline.phase_diagnostics import run_phase_start_diagnostics

            diagnostic_stages = [
                ("coarse", coarse_model, _stage_finetune_ckpt(state, "coarse"), None),
                (
                    refine_stage,
                    fine_model,
                    _stage_finetune_ckpt(state, refine_stage),
                    _stage_finetune_ckpt(state, "coarse"),
                ),
            ]
            ckpt_info = []
            if ft_guidance_ckpt and os.path.exists(ft_guidance_ckpt):
                ckpt_info.append(
                    {
                        "kind": state.guidance_type,
                        "path": ft_guidance_ckpt,
                        "n_variates": n_iv,
                        "lookback": int(ds_lb),
                        "horizon": int(ds_hz),
                    }
                )
            ckpt_info.extend(
                {
                    "kind": f"diffusion_{stage}",
                    "path": ckpt,
                    "n_variates": n_iv,
                    "lookback": int(ds_lb),
                    "horizon": int(ds_hz),
                }
                for stage, _model, ckpt, _coarse_ckpt in diagnostic_stages
            )
            run_phase_start_diagnostics(
                state,
                phase_name=self.name,
                models=[item[1] for item in diagnostic_stages],
                model_labels=[f"diffusion_{item[0]}" for item in diagnostic_stages],
                ckpt_info=ckpt_info,
            )
            _, _, test_ds, _ = load_dataset(
                state.dataset,
                variate_indices,
                stride=train_stride,
                test_stride=test_stride,
                ordinal_tie_atol=float(state.ordinal_tie_atol),
                use_ordinal_window_norm=state.use_ordinal_window_norm,
            )
            for eval_stage, eval_model, eval_ckpt, eval_coarse in diagnostic_stages:
                diag = run_real_dataset_phase_diagnostics(
                    state,
                    train_ds=test_ds,
                    model=eval_model,
                    itrans_ckpt_path=ft_guidance_ckpt,
                    stage=eval_stage,
                    diffusion_ckpt_path=eval_ckpt,
                    coarse_ckpt_path=eval_coarse,
                    tag=f"staged_eval/{eval_stage}",
                    include_phase_start=(eval_stage == "coarse"),
                )
                wandb_utils.log_phase_diagnostics_result(diag)
        except Exception as e:
            logger.warning("[%s] eval diagnostics failed: %s", self.name, e, exc_info=True)

        selected_sampler = str(self.require("probabilistic_sampler"))
        if selected_sampler in {"anchor", "deterministic_anchor", "ddim"}:
            raise ValueError(
                "staged probabilistic_sampler must be quad_t or ddim_quad, not "
                f"{selected_sampler!r}."
            )
        selected_steps = default_steps

        loader = DataLoader(final_ds, batch_size=batch_size, shuffle=False)
        metrics, pack = self._run_eval(
            state=state,
            subset_id=subset_id,
            loader=loader,
            device=device,
            coarse_model=coarse_model,
            fine_model=fine_model,
            prob_sampler=selected_sampler,
            prob_steps=selected_steps,
            prob_samples=prob_samples,
            gmm_components=gmm_components,
            topk_max=topk_max,
            window_indices=eval_window_indices,
            test_stride=test_stride,
        )
        metrics.update({
            "selected_probabilistic_sampler": selected_sampler,
            "selected_probabilistic_num_inference_steps": selected_steps,
        })

        from models.diffusion_tsf.pipeline.phase_diagnostics import select_spaced_top_k

        crps_scores = per_window_crps(pack["y_true"], pack["samples"])
        anchor_scores = per_window_anchor_mse(pack["y_true"], pack["final_anchor"])
        series_starts = pack["series_starts"]
        window_indices_arr = pack["window_indices"]
        worst_manifest: List[Dict[str, Any]] = []
        for metric_name, scores in (("crps", crps_scores), ("anchor_mse", anchor_scores)):
            top_idx = select_spaced_top_k(scores, series_starts, k=10, min_spacing=48)
            for rank, wi in enumerate(top_idx, start=1):
                worst_manifest.append({
                    "metric": metric_name,
                    "rank": rank,
                    "window_index": int(window_indices_arr[wi]),
                    "series_start": int(series_starts[wi]),
                    "score": float(scores[wi]),
                })

        partial_dir = os.path.join(state.results_dir, "partials")
        raw_dir = os.path.join(state.results_dir, "raw")
        nested_dir = os.path.join(state.results_dir, subset_id)
        os.makedirs(partial_dir, exist_ok=True)
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(nested_dir, exist_ok=True)
        with open(os.path.join(nested_dir, "worst_windows.json"), "w") as f:
            json.dump(worst_manifest, f, indent=2)
        with open(os.path.join(partial_dir, f"{state.dataset}_staged_anchor.json"), "w") as f:
            payload = dict(metrics)
            payload["seed"] = int(state.seed)
            json.dump(payload, f, indent=2, sort_keys=True)
        np.savez_compressed(os.path.join(raw_dir, f"staged_anchor_{state.dataset}.npz"), **pack)
        np.savez_compressed(
            os.path.join(raw_dir, f"staged_anchor_samples_{state.dataset}.npz"),
            y_true=pack["y_true"],
            anchor=pack["deterministic"],
        )
        np.savez_compressed(
            os.path.join(raw_dir, f"staged_quad_t_samples_{state.dataset}.npz"),
            y_true=pack["y_true"],
            samples=pack["samples"],
            sample_mean=pack["sample_mean"],
        )
        with open(os.path.join(nested_dir, "staged_results.json"), "w") as f:
            json.dump({
                "dataset": state.dataset,
                "subset_id": subset_id,
                "seed": int(state.seed),
                "variate_indices": variate_indices,
                "data_subset": subset_meta,
                "eval_metrics": {"staged_anchor": metrics},
            }, f, indent=2, sort_keys=True)

        wandb_utils.log_eval_metrics({
            "eval/staged_prob_mse": metrics.get("mse"),
            "eval/staged_prob_mae": metrics.get("mae"),
            "eval/staged_sample_mean_mse": metrics.get("sample_mean_mse"),
            "eval/staged_sample_mean_mae": metrics.get("sample_mean_mae"),
            "eval/staged_anchor_mse": metrics.get("anchor_mse"),
            "eval/staged_anchor_mae": metrics.get("anchor_mae"),
            "eval/staged_crps": metrics.get("crps"),
            "eval/staged_top1_mse": metrics.get("top1_mse"),
            "eval/staged_top3_mse": metrics.get("top3_mse"),
            "eval/selected_sampler": selected_sampler,
            "eval/selected_steps": selected_steps,
        })

        skip_viz = bool(
            self.get("skip_eval_visualizations", False)
            or state.extra.get("skip_eval_visualizations", False)
        )
        viz_cfg = visualization_settings(state.merged_config)
        coarse_ft = state.diffusion_coarse_finetune_ckpt or _stage_finetune_ckpt(state, "coarse")
        fine_ft = (
            state.diffusion_patch_refine_finetune_ckpt
            if patch_refine
            else state.diffusion_fine_finetune_ckpt
        ) or _stage_finetune_ckpt(state, refine_stage)
        if not skip_viz and viz_cfg.get("enabled", True) and not state.smoke_test:
            try:
                tuned = state.fine_finetune_best_params or state.coarse_finetune_best_params
                viz_paths = run_staged_finetune_visualizations(
                    state,
                    coarse_ckpt_path=coarse_ft,
                    fine_ckpt_path=fine_ft,
                    itrans_ckpt_path=ft_guidance_ckpt,
                    tuned_params=tuned,
                    tag="eval_staged_dual_scale",
                )
                wandb_utils.log_visualization_paths(
                    viz_paths, wandb_key="eval/dual_scale_visualizations",
                )
            except Exception as e:
                logger.warning("Staged eval visualizations failed: %s", e, exc_info=True)

        if not skip_viz and viz_cfg.get("enabled", True):
            try:
                worst_viz = run_eval_worst_window_visualizations(
                    state,
                    test_ds=full_test_ds,
                    pack=pack,
                    worst_manifest=worst_manifest,
                    coarse_model=coarse_model,
                    fine_model=fine_model,
                    device=device,
                )
                wandb_utils.log_visualization_paths(worst_viz, wandb_key="eval/worst_samples")
            except Exception as e:
                logger.warning("Worst-window eval viz failed: %s", e, exc_info=True)

            try:
                prob_viz = run_eval_probabilistic_sample_visualizations(
                    state,
                    test_ds=full_test_ds,
                    pack=pack,
                    worst_manifest=worst_manifest,
                    coarse_model=coarse_model,
                    fine_model=fine_model,
                    device=device,
                    sampler=selected_sampler,
                    num_inference_steps=selected_steps,
                )
                wandb_utils.log_visualization_paths(
                    prob_viz, wandb_key="eval/probabilistic_samples",
                )
            except Exception as e:
                logger.warning("Probabilistic sample eval viz failed: %s", e, exc_info=True)

            try:
                dataset_viz = run_eval_full_dataset_visualization(
                    state, splits={"train": full_train_ds, "val": full_val_ds, "test": full_test_ds},
                )
                wandb_utils.log_visualization_paths(dataset_viz, wandb_key="eval/full_dataset_splits")
            except Exception as e:
                logger.warning("Full-dataset eval viz failed: %s", e, exc_info=True)

            if state.use_ordinal_window_norm:
                try:
                    ord_paths = run_ordinal_roundtrip_visualization(state, split="test")
                    wandb_utils.log_visualization_paths(
                        ord_paths, wandb_key="eval/ordinal_roundtrip",
                    )
                except Exception as e:
                    logger.warning("Ordinal roundtrip viz failed: %s", e, exc_info=True)
                try:
                    repr_paths = run_ordinal_coarse_fine_2d_visualization(state, variate=0)
                    wandb_utils.log_visualization_paths(
                        repr_paths, wandb_key="eval/ordinal_coarse_fine_2d",
                    )
                except Exception as e:
                    logger.warning("Ordinal coarse/fine 2D viz failed: %s", e, exc_info=True)

            # Pack-native anchor + probabilistic mean/band panels (same shape as MMPD viz).
            try:
                from utils.mmpd_sample_viz import generate_mmpd_sample_visualizations

                pack_viz_dir = os.path.join(state.results_dir, "viz", "binary_anchor_prob", state.dataset)
                pack_viz = generate_mmpd_sample_visualizations(
                    pack,
                    dataset=state.dataset,
                    out_dir=Path(pack_viz_dir),
                    model_label="binary",
                    n_windows=4 if not state.smoke_test else 1,
                    seed=int(state.seed),
                )
                wandb_utils.log_visualization_paths(
                    pack_viz, wandb_key="eval/binary_anchor_prob_samples",
                )
            except Exception as e:
                logger.warning("Binary pack anchor+prob viz failed: %s", e, exc_info=True)

        logger.info(
            "[%s] staged eval done: sampler=%s steps=%d "
            "prob_mse=%.4f prob_mae=%.4f anchor_mse=%.4f anchor_mae=%.4f crps=%.4f",
            subset_id,
            selected_sampler,
            selected_steps,
            metrics.get("mse", float("nan")),
            metrics.get("mae", float("nan")),
            metrics.get("anchor_mse", float("nan")),
            metrics.get("anchor_mae", float("nan")),
            metrics.get("crps", float("nan")),
        )
        return state
```
```python
# models/diffusion_tsf/metrics.py:1-120
"""Forecast metrics used by staged_eval and viz helpers.

Point: mse/mae. Probabilistic: crps_ensemble, topk_from_modes, probabilistic_forecast_metrics.
Texture suite (haar/rqa/variogram/...) is optional diagnostics; staged leaderboard
rows mostly care about eval/staged_* from staged_eval summaries.
"""

import math
import torch
import torch.nn.functional as F
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean Squared Error.
    
    Args:
        pred: Predictions of shape (batch, seq_len)
        target: Ground truth of shape (batch, seq_len)
        
    Returns:
        Scalar MSE value
    """
    return F.mse_loss(pred, target)


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean Absolute Error.
    
    Args:
        pred: Predictions of shape (batch, seq_len)
        target: Ground truth of shape (batch, seq_len)
        
    Returns:
        Scalar MAE value
    """
    return F.l1_loss(pred, target)


def first_order_gradient(x: torch.Tensor) -> torch.Tensor:
    """Compute first-order differences (discrete derivative).
    
    Args:
        x: Time series of shape (batch, seq_len)
        
    Returns:
        Gradients of shape (batch, seq_len - 1)
    """
    return x[:, 1:] - x[:, :-1]


def shape_preservation_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_bins: int = 50
) -> Dict[str, torch.Tensor]:
    """Shape-Preservation Metric.
    
    Compares the distribution of first-order derivatives (gradients) between
    predictions and ground truth. This captures whether high-frequency textures
    (jagged edges, W/V shapes) are preserved.
    
    The metric computes:
    1. Gradient MAE: Direct comparison of derivatives
    2. Gradient Distribution Divergence: KL divergence between histogram distributions
    
    Args:
        pred: Predictions of shape (batch, seq_len)
        target: Ground truth of shape (batch, seq_len)
        num_bins: Number of bins for histogram
        
    Returns:
        Dictionary with:
        - 'gradient_mae': MAE of first-order derivatives
        - 'gradient_correlation': Correlation between gradients
        - 'shape_score': Combined shape preservation score (lower is better)
    """
    # Compute first-order gradients
    pred_grad = first_order_gradient(pred)
    target_grad = first_order_gradient(target)
    
    # 1. Direct gradient comparison (MAE)
    gradient_mae = F.l1_loss(pred_grad, target_grad)
    
    # 2. Gradient correlation (Pearson correlation)
    pred_grad_flat = pred_grad.flatten()
    target_grad_flat = target_grad.flatten()
    
    # Center the data
    pred_centered = pred_grad_flat - pred_grad_flat.mean()
    target_centered = target_grad_flat - target_grad_flat.mean()
    
    # Compute correlation
    numerator = (pred_centered * target_centered).sum()
    denominator = torch.sqrt((pred_centered ** 2).sum() * (target_centered ** 2).sum()) + 1e-8
    gradient_corr = numerator / denominator
    
    # 3. Sign agreement (captures direction of changes)
    pred_sign = torch.sign(pred_grad)
    target_sign = torch.sign(target_grad)
    sign_agreement = (pred_sign == target_sign).float().mean()
    
    # Combined shape score (lower is better)
    # Weight: MAE is penalized, correlation and sign agreement are rewarded
    shape_score = gradient_mae - 0.1 * gradient_corr - 0.1 * sign_agreement + 0.2
    
    logger.debug(f"Shape metrics: grad_mae={gradient_mae:.4f}, grad_corr={gradient_corr:.4f}, "
                 f"sign_agree={sign_agreement:.4f}")
    
    return {
        'gradient_mae': gradient_mae,
        'gradient_correlation': gradient_corr,
        'sign_agreement': sign_agreement,
        'shape_score': shape_score
    }
```

## 11. Standalone staged forecast helper (disc / MMPD match packs)

```python
# utils/staged_binary_forecast.py:1-90
#!/usr/bin/env python3
"""Materialize binary staged forecast packs for disc / compare evaluators.

Callers: utils.disc_shared.ensure_raw_packs, temp ordinal/vs-gt evaluators.
Resolves ckpt dirs (reused/binary preferred), loads coarse+fine or vertical_dual,
runs generate_staged_forecast, writes raw/binary_staged_{dataset}.npz with
y_true / deterministic / samples (N,V,S,H).

MMPD packs are produced separately by eval_mmpd_gaussian_anchor.run_mmpd_eval.
Pack reduction for the disc (sample0 vs mean) lives in forecast_pack_reduce.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.diffusion_tsf.pipeline.globals_bridge import patch_globals
from models.diffusion_tsf.pipeline.reused_paths import find_reused_binary_staged_root
from models.diffusion_tsf.train_multivariate_pipeline import (
    load_dataset,
    load_wrapped_guidance,
    resolve_pipeline_data_subset,
)
from utils.eval_mmpd_gaussian_anchor import (
    AnchorRun,
    DEFAULT_MMPD_DATA,
    DEFAULT_MMPD_REPO,
    ensure_mmpd_repo,
    load_tsf_pack_pool,
    load_tsf_test_subset,
    make_eval_indices,
    make_pack_pool_indices,
    parse_pack_splits,
    run_mmpd_eval,
    run_subset_id,
    run_test_stride,
    run_train_stride,
    run_variate_indices,
    stable_dataset_seed,
    summarize_prediction_pack,
)
from utils.mmpd_eval_progress import EvalProgress, fmt_duration
from utils.visualize_staged_eval_2d_preds import (
    _build_state,
    _load_stage_model,
    _load_staged_bundle,
    _resolve_guidance_ckpt,
    _window_lengths,
)


ROBUST_TEXTURE_KEYS = [
    "texture_increment_wasserstein",
    "texture_curvature_wasserstein",
    "texture_haar_detail_jsd",
    "texture_jump_plateau_distance",
    "texture_derivative_motif_jsd",
]

DEFAULT_SUBSET_DATASETS = (
    "ETTh1",
    "ETTh2",
    "exchange_rate",
    "weather",
    "electricity",
    "traffic",
    "solar_Alabama",
)

DEFAULT_ANCHOR_CONFIG = "binary_patch_refine_lb336_hz96"
DEFAULT_CKPT_BASE = REPO_ROOT / "results" / "ckpts"
DEFAULT_MMPD_OUTPUT_ROOT = REPO_ROOT / "results" / "datasets" / "06-13-binary-mmpd-subset-compare"


def resolve_staged_ckpt_dir(ckpt_base: Path, dataset: str, anchor_config: str) -> Path:
    """Prefer reused/binary/<stem>, else newest results/ckpts/*-{dataset}-{stem}."""
```
```python
# utils/staged_binary_forecast.py:241-438
def generate_staged_forecast(
    coarse_model: Any,
    fine_model: Any,
    past: torch.Tensor,
    *,
    vertical_dual: bool,
    fine_seed: Optional[int] = None,
    **generate_kwargs: Any,
) -> Dict[str, torch.Tensor]:
    """One window through coarse→fine, or a single vertical_dual generate()."""
    if vertical_dual:
        # Vertical dual owns the stacked canvas + decode; do not chain a second model.
        return coarse_model.generate(past, **generate_kwargs)
    coarse_out = coarse_model.generate(past, **generate_kwargs)
    if fine_seed is not None:
        torch.manual_seed(int(fine_seed))
    return fine_model.generate(
        past,
        future_coarse_2d=coarse_out["future_2d_coarse"],
        **generate_kwargs,
    )


def evaluate_staged_binary(
    args: argparse.Namespace,
    run: AnchorRun,
    sub: Dict[str, Any],
    indices: Sequence[int],
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Full pack loop: load models → det + S samples → cache npz under output_dir/raw/."""
    from torch.utils.data import Subset

    raw_path = args.output_dir / "raw" / f"binary_staged_{run.dataset}.npz"
    if raw_path.exists() and not args.force_binary_eval:
        with np.load(raw_path) as data:
            return {key: data[key] for key in data.files}

    lookback, horizon = dataset_window_lengths_for_run(args, run)
    pack_splits = parse_pack_splits(getattr(args, "pack_splits", None))
    pool, series_starts_full, splits, _part_lengths, norm_stats = load_tsf_pack_pool(
        run.dataset,
        run_variate_indices(run),
        lookback=lookback,
        horizon=horizon,
        train_stride=run_train_stride(run),
        test_stride=run_test_stride(run),
        pack_splits=pack_splits,
        ordinal_tie_atol=1e-6,
        use_ordinal_window_norm=None,
    )
    subset = Subset(pool, list(indices))
    # Keep micro-batches small for lb336/hz720 maps.
    batch_size = min(int(args.binary_batch_size), 2)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    config_path = _binary_config_path(args, run.dataset)
    subset_id = run_subset_id(run)
    state = _build_state(run.root, run.dataset, subset_id, config_path)
    resolve_pipeline_data_subset(state)
    import models.diffusion_tsf.train_multivariate_pipeline as pipeline_mod

    patch_globals(pipeline_mod, state, honor_dataset_windows=True)
    # Rebuild ladder under the same binary config knobs as training.
    _, _, _test_ds, norm_stats = load_dataset(
        run.dataset,
        run_variate_indices(run),
        stride=run_train_stride(run),
        test_stride=run_test_stride(run),
        lookback=lookback,
        horizon=horizon,
        ordinal_tie_atol=float(state.ordinal_tie_atol),
        use_ordinal_window_norm=state.use_ordinal_window_norm,
    )
    if norm_stats.get("ordinal_ladder") is not None:
        state.extra["global_ordinal_ladder"] = norm_stats["ordinal_ladder"]
        pipeline_mod.GLOBAL_ORDINAL_LADDER = norm_stats["ordinal_ladder"]

    guidance_type = str(run.metadata.get("guidance_type") or "auto")
    guidance_path, guidance_type = _resolve_guidance_ckpt(run.root, subset_id, guidance_type)
    guidance_model = load_wrapped_guidance(
        str(guidance_path),
        len(run_variate_indices(run)),
        device,
        guidance_type=guidance_type,
        dataset_lookback=lookback,
        dataset_horizon=horizon,
    )
    # A vertical-dual run has one H=(Hc+Hf) model, not independent Hc/Hf
    # checkpoints.  _load_staged_bundle deliberately aliases coarse_pt/fine_pt
    # to that one file for metadata compatibility, so dispatch on ``stage``
    # before constructing the model.  Loading it as ``coarse`` or ``fine``
    # silently drops the 32-row decoder parameters and invalidates the sample.
    vertical_dual = (
        str(sub.get("stage") or "") == "vertical_dual"
        or bool(getattr(state, "use_vertical_dual_concat", False))
    )
    if vertical_dual:
        coarse_model = _load_stage_model(
            state,
            "vertical_dual",
            Path(sub["coarse_pt"]),
            guidance_model,
            len(run_variate_indices(run)),
            device,
            strict_non_guidance_shapes=True,
        )
        fine_model = coarse_model
    else:
        coarse_model = _load_stage_model(
            state, "coarse", Path(sub["coarse_pt"]), guidance_model, len(run_variate_indices(run)), device,
            strict_non_guidance_shapes=True,
        )
        fine_model = _load_stage_model(
            state, "fine", Path(sub["fine_pt"]), guidance_model, len(run_variate_indices(run)), device,
            strict_non_guidance_shapes=True,
        )
    # Pool windows are z-score series (not pre-ranked) even under ordinal configs.
    for m in (coarse_model, fine_model):
        m._ordinal_input_is_ranked = False
        m._ordinal_apply_ood_shift = bool(state.use_ordinal_window_norm)

    prob_kwargs = {"sampler": args.probabilistic_sampler, "num_inference_steps": args.num_sampling_steps}
    y_true_all: List[np.ndarray] = []
    det_all: List[np.ndarray] = []
    samples_all: List[np.ndarray] = []
    progress = EvalProgress(f"binary-staged/{run.dataset}", len(loader))
    print(
        f"[binary-staged] {run.dataset}: windows={len(indices)} batches={len(loader)} "
        f"samples={args.sample_num} pack_splits={splits} stride_train={run_train_stride(run)} "
        f"config={config_path}",
        flush=True,
    )
    t0 = time.time()
    with torch.no_grad():
        for batch_idx, (past, future) in enumerate(loader):
            t_batch = time.time()
            past = past.to(device)
            future = future.to(device)
            K = int(getattr(coarse_model.config, "lookback_overlap", 0) or 0)
            if K > 0:
                future = future[..., K:]
            y_true_all.append(future.cpu().numpy())

            torch.manual_seed(args.seed + batch_idx)
            fine_det = generate_staged_forecast(
                coarse_model,
                fine_model,
                past,
                vertical_dual=vertical_dual,
                sampler="anchor",
            )
            det_all.append(_prediction_tensor(fine_det).cpu().numpy())

            batch_samples = []
            for sample_idx in range(args.sample_num):
                seed = args.seed + batch_idx * 1009 + sample_idx * 17
                torch.manual_seed(seed)
                fine_sample = generate_staged_forecast(
                    coarse_model,
                    fine_model,
                    past,
                    vertical_dual=vertical_dual,
                    fine_seed=seed,
                    **prob_kwargs,
                )
                batch_samples.append(_prediction_tensor(fine_sample).cpu().numpy())
            samples_all.append(np.stack(batch_samples, axis=2))

            progress.maybe_log(
                batch_idx + 1,
                extra=(
                    f"last_batch={fmt_duration(time.time() - t_batch)} "
                    f"elapsed={fmt_duration(time.time() - t0)}"
                ),
            )
    progress.done(extra=f"writing {raw_path}")

    idx_arr = np.asarray(indices, dtype=np.int64)
    pack = {
        "y_true": np.concatenate(y_true_all, axis=0),
        "deterministic": np.concatenate(det_all, axis=0),
        "samples": np.concatenate(samples_all, axis=0),
        "indices": idx_arr,
        "series_starts": series_starts_full[idx_arr],
        "pack_splits": np.asarray(splits),
    }
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(raw_path, **pack)
    return pack
```

## 12. MMPD path

`./submit_mmpd.sh` → `utils/eval_mmpd_gaussian_anchor.py`. Train Decoder, gaussian-anchor eval, optional leaderboard.

```bash
# submit_mmpd.sh:1-80
#!/bin/bash
# Login-node submitter for MMPD train+eval campaigns (flat subsets / paper Decoder).
# Sibling of submit_binary.sh — does NOT go through slurm_worker.sh; writes its
# own per-job worker scripts that call utils/eval_mmpd_gaussian_anchor.py.
#
# DAG: optional init → per-dataset mmpd workers → merge. With mmpd.leaderboard
# in the run YAML, each worker can push eval/staged_* rows to ts-sandbox-leaderboard.
#
# USAGE (Killarney or Narval login, from $SCRATCH/ts-sandbox):
#   ./submit_mmpd.sh --smoke-test
#   ./submit_mmpd.sh --mmpd-run-config mmpd_decoder_flat_subsets_paper_lb336_hz720 \
#       --output-dir results/datasets/$(date +%m-%d)-mmpd-paper-lb336-hz720 --time 24:00:00
#
# Cluster auto-detect: Killarney → l40s (aip-boyuwang); Narval → a100 (def-boyuwang).
# Clones temp/MMPD on the login node (compute has no GitHub egress).
# --mmpd-run-config: path or bare stem under configs/*.yaml.
# Default subset source: --subset-config (no binary ckpts). Legacy: --use-anchor-ckpts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE=0
RESUME=0
FORCE=0
FORCE_INIT=0
SKIP_MMPD_TRAIN=0
OUTPUT_DIR=""
SUBSET_CONFIG="configs/binary_anchor_ar.yaml"
USE_ANCHOR_CKPTS=0
ANCHOR_CONFIG="binary_patch_refine_lb336_hz96"
DATASETS_CSV="ETTh1,ETTh2,exchange_rate,weather,electricity,traffic,solar_Alabama"
DEPENDENCY=""
SEED=2026
LOOKBACK=96
HORIZON=96
WALL_MMPD="3:00:00"
WALL_INIT="0:45:00"
WALL_MERGE="0:30:00"
MMPD_RUN_CONFIG=""
MMPD_BACKBONE="Decoder"
MMPD_TUNE_TRIALS=0
MMPD_TUNE_EPOCHS=10
MMPD_TUNE_PATIENCE=3
ORDINAL_UPSCALE=0
DATASETS_EXPLICIT=0
GPU_TYPE=""
MMPD_INSTANCE_NORM=1
EVAL_EXISTING_DISCRIMINATOR=0
EXISTING_MMPD_ROOT=""
DISC_RUN=""
RAW_RUN=""
JOB_MANIFEST=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-test|--smoke) SMOKE=1; shift ;;
        --resume) RESUME=1; shift ;;
        --force-init) FORCE_INIT=1; shift ;;
        --force) FORCE=1; shift ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --subset-config) SUBSET_CONFIG="$2"; shift 2 ;;
        --use-anchor-ckpts) USE_ANCHOR_CKPTS=1; shift ;;
        --anchor-config) ANCHOR_CONFIG="$2"; USE_ANCHOR_CKPTS=1; shift 2 ;;
        --datasets) DATASETS_CSV="$2"; DATASETS_EXPLICIT=1; shift 2 ;;
        --dependency) DEPENDENCY="$2"; shift 2 ;;
        --lookback) LOOKBACK="$2"; shift 2 ;;
        --horizon) HORIZON="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --time) WALL_MMPD="$2"; shift 2 ;;
        --mmpd-backbone) MMPD_BACKBONE="$2"; shift 2 ;;
        --mmpd-run-config) MMPD_RUN_CONFIG="$2"; shift 2 ;;
        --mmpd-tune-trials) MMPD_TUNE_TRIALS="$2"; shift 2 ;;
        --mmpd-tune-epochs) MMPD_TUNE_EPOCHS="$2"; shift 2 ;;
        --mmpd-tune-patience) MMPD_TUNE_PATIENCE="$2"; shift 2 ;;
        --force-mmpd-tune) FORCE=1; shift ;;
        --skip-mmpd-train) SKIP_MMPD_TRAIN=1; shift ;;
        --gpu) GPU_TYPE="$2"; shift 2 ;;
        --mmpd-instance-norm) MMPD_INSTANCE_NORM=1; shift ;;
        --no-mmpd-instance-norm) MMPD_INSTANCE_NORM=0; shift ;;
        --eval-existing-discriminator) EVAL_EXISTING_DISCRIMINATOR=1; shift ;;
```
```python
# utils/mmpd_run_config.py:1-75
"""Apply YAML ``mmpd:`` onto eval_mmpd_gaussian_anchor argparse.

Called from eval_mmpd_gaussian_anchor.main() when --mmpd-run-config is set
(submit_mmpd.sh). Mutates the Namespace in place; does not train or eval.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_subset_config_path(subset_config: str, *, repo_root: Path = REPO_ROOT) -> Path:
    p = Path(subset_config)
    if p.is_file():
        return p.resolve()
    candidate = repo_root / "configs" / subset_config
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(f"subset config not found: {subset_config}")


def apply_mmpd_run_config(args: Any, block: Dict[str, Any], *, repo_root: Path = REPO_ROOT) -> None:
    """Map known ``mmpd`` YAML keys onto CLI flags; unknown keys are ignored."""
    # Ordinal MMPD campaigns turn off instance norm (mutually exclusive paths).
    if block.get("use_ordinal_window_norm"):
        args.use_ordinal_window_norm = True
        args.mmpd_instance_norm = False
    if "ordinal_tie_atol" in block:
        args.ordinal_tie_atol = float(block["ordinal_tie_atol"])
    if backbone := block.get("backbone"):
        args.mmpd_backbone = str(backbone)
    if subset_config := block.get("subset_config"):
        args.subset_config = resolve_subset_config_path(str(subset_config), repo_root=repo_root)
        # Subset YAML implies MMPD-only; binary anchor eval is skipped.
        args.mmpd_only = True
    if lookback := block.get("lookback"):
        args.lookback = int(lookback)
    if horizon := block.get("horizon"):
        args.horizon = int(horizon)
    if train_epochs := block.get("train_epochs"):
        args.mmpd_train_epochs = int(train_epochs)
    if patience := block.get("patience"):
        args.mmpd_patience = int(patience)
    if batch_size := block.get("batch_size"):
        args.mmpd_batch_size = int(batch_size)
    if sample_num := block.get("sample_num"):
        args.sample_num = int(sample_num)
    if num_sampling_steps := block.get("num_sampling_steps"):
        args.num_sampling_steps = int(num_sampling_steps)
    if gmm_components := block.get("gmm_components"):
        args.gmm_components = int(gmm_components)
    if gmm_iterations := block.get("gmm_iterations"):
        args.gmm_iterations = int(gmm_iterations)
    if "tune_trials" in block:
        args.mmpd_tune_trials = int(block["tune_trials"])
    if tune_epochs := block.get("tune_epochs"):
        args.mmpd_tune_epochs = int(tune_epochs)
    if tune_patience := block.get("tune_patience"):
        args.mmpd_tune_patience = int(tune_patience)
    tune_params = block.get("tune_params")
    if isinstance(tune_params, dict):
        args.mmpd_tune_params = dict(tune_params)
    if lradj := block.get("lradj"):
        args.mmpd_lradj = str(lradj)
    fixed = dict(getattr(args, "mmpd_fixed_hparams", None) or {})
    for key in ("learning_rate", "point_weight", "dropout", "ema_decay"):
        if key in block:
            fixed[key] = block[key]
    if fixed:
        args.mmpd_fixed_hparams = fixed
    if block.get("leaderboard", False):
        args.mmpd_log_leaderboard = True
```
```python
# utils/mmpd_paper_hparams.py:1-66
"""MMPD appendix D.3 patch size + default / tuned hparams.

Used by eval_mmpd_gaussian_anchor (train cmd + eval helper) and Optuna tune.
``ECL``/``Traffic`` aliases coexist with repo keys ``electricity``/``traffic``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

# Paper names + repo dataset keys that always take the wide patch.
WIDE_PATCH_DATASETS = frozenset({"electricity", "traffic", "ECL", "Traffic"})

DEFAULT_MMPD_HPARAMS: Dict[str, Any] = {
    "learning_rate": 1e-4,
    "point_weight": 0.01,
    "dropout": 0.2,
    "finetune_layers": 0,
    "neighbor_num": 0,
    "ema_decay": 0.99,
}


def mmpd_patch_size(dataset: str, horizon: int) -> int:
    """Paper D.3: P=12 default; P=24 for tau in {336,720} or ECL/Traffic."""
    if dataset in WIDE_PATCH_DATASETS:
        return 24
    if horizon in (336, 720):
        return 24
    return 12


def resolve_mmpd_patch_size(
    dataset: str,
    horizon: int,
    override: int | None = None,
) -> int:
    if override is not None:
        return int(override)
    return mmpd_patch_size(dataset, horizon)


def tuning_result_path(output_dir: Path, dataset: str) -> Path:
    return output_dir / "tuning" / f"{dataset}_best.json"


def load_tuned_hparams(output_dir: Path, dataset: str) -> Optional[Dict[str, Any]]:
    path = tuning_result_path(output_dir, dataset)
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    hparams = payload.get("hparams")
    return dict(hparams) if isinstance(hparams, dict) else None


def resolved_mmpd_hparams(
    output_dir: Path,
    dataset: str,
    *,
    fallback: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Prefer tuning/<dataset>_best.json, else YAML/CLI fallback, else DEFAULT_MMPD_HPARAMS."""
    tuned = load_tuned_hparams(output_dir, dataset)
```
```python
# utils/eval_mmpd_gaussian_anchor.py:1-50
#!/usr/bin/env python3
"""MMPD train + gaussian-anchor eval entrypoint (submit_mmpd.sh).

Live path: ./submit_mmpd.sh → this file with --mmpd-run-config <YAML>.
YAML ``mmpd:`` is applied in main() via utils.mmpd_run_config; paper D.3
patch/hparams come from utils.mmpd_paper_hparams.

Phases (--phase): init → mmpd (train+eval+optional leaderboard) → optional
anchor → merge. Slurm workers usually run one dataset at --phase mmpd.

Next hops after packs land: disc evaluators under temp/ and utils/disc_shared.py
consume raw/mmpd_*.npz; staged binary packs come from staged_binary_forecast.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
from contextlib import contextmanager
import shutil
import subprocess
import sys
import textwrap
import time
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from utils.mmpd_eval_progress import EvalProgress, fmt_duration
from utils.mmpd_paper_hparams import resolve_mmpd_patch_size


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def pipeline_python() -> str:
    """Venv interpreter; module load after activate can leave sys.executable on CVMFS."""
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
```
```python
# utils/eval_mmpd_gaussian_anchor.py:1068-1099
def train_mmpd(args: argparse.Namespace, runs: Sequence[AnchorRun]) -> None:
    """Stage CSV → optional Optuna tune → upstream main_mmpd.py (or reuse ckpt)."""
    from utils.mmpd_paper_hparams import load_tuned_hparams, tuning_result_path
    from utils.mmpd_subset_tune import tune_mmpd_subset

    for run in runs:
        dataset = run.dataset
        stage_mmpd_dataset_for_run(args.mmpd_data_dir, run)
        if args.mmpd_tune_trials > 0:
            tuned = load_tuned_hparams(args.output_dir, dataset)
            if tuned is None or args.force_mmpd_tune:
                tune_mmpd_subset(args, run)
            else:
                print(f"[mmpd-tune] {dataset}: reusing {tuning_result_path(args.output_dir, dataset)}")
        ckpt, _ = resolve_mmpd_checkpoint(args, run)
        if ckpt.exists() and not args.force_mmpd_train:
            print(f"[mmpd] Reusing checkpoint for {dataset}: {ckpt}")
            continue
        if args.skip_mmpd_train:
            if args.skip_mmpd_eval:
                print(f"[mmpd] Skipping train/eval for {dataset}; checkpoint not required.")
                continue
            raise FileNotFoundError(f"--skip-mmpd-train set but missing {ckpt}")
        log_path = args.output_dir / "logs" / f"mmpd_train_{dataset}.log"
        run_cmd(
            build_mmpd_train_cmd(args, run),
            cwd=args.mmpd_repo,
            env=mmpd_env_for_run(run, args),
            log_path=log_path,
        )
```
```python
# utils/eval_mmpd_gaussian_anchor.py:1845-1945
def run_mmpd_eval(
    args: argparse.Namespace,
    run: AnchorRun,
    indices: Sequence[int],
) -> Dict[str, np.ndarray]:
    """Materialize raw/mmpd_{dataset}.npz via the helper subprocess (cache unless --force)."""
    dataset = run.dataset
    out_npz = args.output_dir / "raw" / f"mmpd_{dataset}.npz"
    indices_json = args.output_dir / "raw" / f"indices_{dataset}_mmpd_eval.json"
    indices_json.parent.mkdir(parents=True, exist_ok=True)
    pack_splits = parse_pack_splits(getattr(args, "pack_splits", None))
    eval_ds = build_mmpd_pack_pool(args, run, pack_splits)
    indices = filter_valid_mmpd_indices(dataset, eval_ds, indices)
    write_json_atomic(indices_json, list(indices))

    if not out_npz.exists() or args.force_mmpd_eval:
        stage_mmpd_dataset_for_run(args.mmpd_data_dir, run)
        ckpt_path, mmpd_data = resolve_mmpd_checkpoint(args, run)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"MMPD checkpoint missing for {dataset}: {ckpt_path}")
        helper = write_mmpd_eval_helper(args.mmpd_repo)
        lookback, horizon = dataset_window_lengths(args, dataset)
        data_dim = len(run_variate_indices(run))
        batch_size = mmpd_eval_batch_size(args, dataset, data_dim=data_dim)
        patch_size = dataset_mmpd_patch_size(args, dataset)
        from utils.mmpd_paper_hparams import resolved_mmpd_hparams

        hp = resolved_mmpd_hparams(
            mmpd_hparams_root(args), dataset, fallback=mmpd_run_fallback_hparams(args)
        )
        cmd = [
            pipeline_python(),
            "-u",
            str(helper),
            "--dataset",
            mmpd_data,
            "--root-path",
            str(args.mmpd_data_dir),
            "--data-path",
            mmpd_staged_filename_for_run(run),
            "--data-split",
            mmpd_data_split(run, args.mmpd_data_dir),
            "--output-root",
            str(mmpd_output_root(args) / "mmpd_out"),
            "--out-npz",
            str(out_npz),
            "--indices-json",
            str(indices_json),
            "--eval-splits",
            ",".join(pack_splits),
            "--lookback",
            str(lookback),
            "--horizon",
            str(horizon),
            "--patch-size",
            str(patch_size),
            "--data-dim",
            str(data_dim),
            "--mmpd-backbone",
            args.mmpd_backbone,
            "--point-weight",
            str(float(hp["point_weight"])),
            "--sample-num",
            str(args.sample_num),
            "--num-sampling-steps",
            str(args.num_sampling_steps),
            "--gmm-components",
            str(args.gmm_components),
            "--gmm-iterations",
            str(args.gmm_iterations),
            "--batch-size",
            str(batch_size),
            "--num-workers",
            str(args.num_workers),
            "--gpu",
            str(args.gpu),
        ]
        if args.mmpd_backbone != "Decoder":
            raise ValueError(f"Only Decoder backbone is supported; got {args.mmpd_backbone!r}")
        if args.cpu:
            cmd.append("--cpu")
        env = mmpd_env_for_run(run, args, for_eval=True)
        print(
            f"[mmpd-eval] {dataset}: launching helper "
            f"(windows={len(indices)}, batch={batch_size}, variates={data_dim}, "
            f"pack_splits={pack_splits}, "
            f"eval_test_stride={eval_test_stride(args, run)})",
            flush=True,
        )
        run_cmd(
            cmd,
            cwd=args.mmpd_repo,
            env=env,
            log_path=args.output_dir / "logs" / f"mmpd_eval_{dataset}.log",
        )
        print(f"[mmpd-eval] {dataset}: helper finished -> {out_npz}", flush=True)

    with np.load(out_npz) as data:
        return {key: data[key] for key in data.files}
```
```python
# utils/eval_mmpd_gaussian_anchor.py:2515-2621
def run_phase_init(args: argparse.Namespace, commit: str) -> None:
    anchors = discover_anchors_by_variant(args, args.datasets)
    indices_by_dataset: Dict[str, List[int]] = {}
    for dataset in args.datasets:
        run = anchors["binary"][dataset]
        stage_mmpd_dataset_for_run(args.mmpd_data_dir, run)
        indices_by_dataset[dataset] = get_or_create_indices(args, run)

    manifest = {
        "args": jsonable_args(args),
        "mmpd_commit": commit,
        "anchor_runs": anchors_to_manifest(anchors),
        "indices_by_dataset": indices_by_dataset,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"[init] Wrote {args.output_dir / 'run_manifest.json'}")


def generate_mmpd_phase_viz(
    args: argparse.Namespace, dataset: str, mmpd_pack: Dict[str, np.ndarray]
) -> List[str]:
    """Anchor + probabilistic sample panels for one dataset's MMPD eval pack.

    Fail-soft: viz errors print and return [] so metrics/leaderboard still land.
    """
    if getattr(args, "skip_mmpd_sample_viz", False) or getattr(args, "smoke_test", False):
        return []
    try:
        from utils.mmpd_sample_viz import generate_mmpd_sample_visualizations

        return generate_mmpd_sample_visualizations(
            mmpd_pack,
            dataset=dataset,
            out_dir=args.output_dir / "viz" / "mmpd_samples" / dataset,
            n_windows=int(getattr(args, "mmpd_sample_viz_windows", 4)),
            seed=int(args.seed),
        )
    except Exception as exc:  # fail-soft: never block mmpd phase on plots
        print(f"[mmpd-viz] {dataset}: skipped ({exc})")
        return []


def run_phase_mmpd(
    args: argparse.Namespace,
    dataset: str,
    anchors_by_variant: Dict[str, Dict[str, AnchorRun]],
) -> None:
    """One-dataset Slurm worker: train (optional) → eval pack → metrics → leaderboard."""
    binary_run = anchors_by_variant["binary"][dataset]
    indices = get_or_create_indices(args, binary_run)
    indices = subsample_eval_indices(
        indices,
        args.test_max_items,
        seed=args.seed,
        dataset=dataset,
    )
    if not args.skip_mmpd_train:
        train_mmpd(args, [binary_run])
    elif not args.skip_mmpd_eval:
        stage_mmpd_dataset_for_run(args.mmpd_data_dir, binary_run)
        ckpt, _ = resolve_mmpd_checkpoint(args, binary_run)
        if not ckpt.exists():
            raise FileNotFoundError(
                f"--skip-mmpd-train but missing MMPD checkpoint: {ckpt}"
            )
    if args.skip_mmpd_eval:
        return
    print(f"[mmpd] {dataset}: eval phase ({len(indices)} windows)", flush=True)
    mmpd_pack = run_mmpd_eval(args, binary_run, indices)
    print(f"[mmpd] {dataset}: summarizing metrics", flush=True)
    metrics = summarize_for_profile(mmpd_pack, args, dataset)
    write_partial_metrics(args.output_dir, dataset, "mmpd", metrics)
    sample_viz_paths = generate_mmpd_phase_viz(args, dataset, mmpd_pack)
    from utils.log_mmpd_eval_leaderboard import maybe_log_mmpd_eval_leaderboard

    maybe_log_mmpd_eval_leaderboard(args, dataset, metrics, extra_viz_paths=sample_viz_paths)


def run_phase_anchor(
    args: argparse.Namespace,
    dataset: str,
    variant: str,
    anchors_by_variant: Dict[str, Dict[str, AnchorRun]],
    device: torch.device,
) -> None:
    if variant not in ANCHOR_VARIANTS:
        raise ValueError(f"Unknown anchor variant: {variant}")
    run = anchors_by_variant[variant][dataset]
    model_name = ANCHOR_VARIANTS[variant]["model_name"]
    indices = get_or_create_indices(args, run)
    print(f"[anchor] {dataset}: eval phase ({len(indices)} windows)", flush=True)
    raw_dir = args.output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    anchor_raw_path = raw_dir / f"{variant}_anchor_{dataset}.npz"
    if anchor_raw_path.exists() and not args.force_anchor_eval:
        with np.load(anchor_raw_path) as data:
            anchor_pack = {key: data[key] for key in data.files}
    else:
        anchor_pack = evaluate_anchor(args, run, indices, device)
        np.savez_compressed(anchor_raw_path, **anchor_pack)
    metrics = summarize_for_profile(anchor_pack, args, dataset)
    write_partial_metrics(args.output_dir, dataset, model_name, metrics)
```
```python
# utils/eval_mmpd_gaussian_anchor.py:2964-3096
def run_phase_all(args: argparse.Namespace, commit: str) -> None:
    anchors_by_variant = discover_anchors_by_variant(args, args.datasets)
    binary_runs = [anchors_by_variant["binary"][dataset] for dataset in args.datasets]
    train_mmpd(args, binary_runs)

    indices_by_dataset: Dict[str, List[int]] = {}
    for dataset in args.datasets:
        run = anchors_by_variant["binary"][dataset]
        indices_by_dataset[dataset] = get_or_create_indices(args, run)

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else f"cuda:{args.gpu}"
    )
    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for dataset in args.datasets:
        results[dataset] = {}
        indices = indices_by_dataset[dataset]

        if not args.skip_mmpd_eval:
            mmpd_pack = run_mmpd_eval(args, anchors_by_variant["binary"][dataset], indices)
            results[dataset]["mmpd"] = summarize_for_profile(mmpd_pack, args, dataset)

        if args.mmpd_only:
            continue

        for variant, anchors in anchors_by_variant.items():
            raw_dir = args.output_dir / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            anchor_raw_path = raw_dir / f"{variant}_anchor_{dataset}.npz"
            if anchor_raw_path.exists() and not args.force_anchor_eval:
                with np.load(anchor_raw_path) as data:
                    anchor_pack = {key: data[key] for key in data.files}
            else:
                anchor_pack = evaluate_anchor(args, anchors[dataset], indices, device)
                np.savez_compressed(anchor_raw_path, **anchor_pack)
            results[dataset][ANCHOR_VARIANTS[variant]["model_name"]] = summarize_for_profile(
                anchor_pack, args, dataset
            )

    manifest = {
        "args": jsonable_args(args),
        "mmpd_commit": commit,
        "anchor_runs": anchors_to_manifest(anchors_by_variant),
        "indices_by_dataset": indices_by_dataset,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_outputs(args, manifest, results)
    print_summary(results, profile=args.metrics_profile)
    print(f"\nWrote metrics to {args.output_dir / 'metrics.json'}")
    print(f"Wrote CSV to {args.output_dir / 'metrics.csv'}")


def main() -> None:
    # submit_mmpd.sh → parse → apply YAML → smoke clamps → phase dispatch.
    args = parse_args()
    args.mmpd_log_leaderboard = False
    if args.mmpd_run_config is not None:
        from utils.mmpd_run_config import apply_mmpd_run_config

        cfg_path = args.mmpd_run_config.resolve()
        with cfg_path.open(encoding="utf-8") as f:
            full_cfg = yaml.safe_load(f) or {}
        mmpd_block = full_cfg.get("mmpd")
        if not isinstance(mmpd_block, dict):
            raise ValueError(f"{cfg_path} missing top-level mmpd: mapping")
        apply_mmpd_run_config(args, mmpd_block)
        exp = full_cfg.get("experiment") or {}
        if exp.get("experiment_name"):
            args.mmpd_config_suffix = str(exp["experiment_name"])
        elif exp.get("name"):
            args.mmpd_config_suffix = str(exp["name"]).replace("-", "_")
        if exp.get("use_ordinal_window_norm"):
            args.use_ordinal_window_norm = True
            args.mmpd_instance_norm = False
        if "ordinal_tie_atol" in exp:
            args.ordinal_tie_atol = float(exp["ordinal_tie_atol"])
    if args.mmpd_leaderboard:
        args.mmpd_log_leaderboard = True
    if args.no_mmpd_leaderboard:
        args.mmpd_log_leaderboard = False
    if args.mmpd_tune_spec_file is not None:
        import json

        with args.mmpd_tune_spec_file.open(encoding="utf-8") as f:
            args.mmpd_tune_params = json.load(f)
    apply_mmpd_smoke_defaults(args)
    args.datasets = list(dict.fromkeys(args.datasets))
    unknown = sorted(set(args.datasets) - set(DATASET_FILES))
    if unknown:
        raise ValueError(f"Unsupported dataset(s): {unknown}")
    validate_phase_args(args)
    args.output_dir = args.output_dir.resolve()
    args.mmpd_repo = args.mmpd_repo.resolve()
    args.mmpd_data_dir = args.mmpd_data_dir.resolve()
    args.ckpt_base = args.ckpt_base.resolve()
    if args.indices_dir is not None:
        args.indices_dir = args.indices_dir.resolve()
    if args.mmpd_output_root is not None:
        args.mmpd_output_root = args.mmpd_output_root.resolve()
    if args.subset_config is not None:
        args.subset_config = args.subset_config.resolve()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    commit = ensure_mmpd_repo(args.mmpd_repo, update=not args.no_update_mmpd)

    if args.phase == "all":
        run_phase_all(args, commit)
        return

    if args.phase == "init":
        run_phase_init(args, commit)
        return

    if args.phase == "merge":
        run_phase_merge(args, commit)
        return

    anchors_by_variant = discover_anchors_by_variant(args, args.datasets)
    dataset = args.datasets[0]

    if args.phase == "mmpd":
        run_phase_mmpd(args, dataset, anchors_by_variant)
        return

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else f"cuda:{args.gpu}"
    )
    run_phase_anchor(args, dataset, args.anchor_variant, anchors_by_variant, device)


if __name__ == "__main__":
    main()
```

## 13. Texture discriminator eval (ordinal live)

Default evaluator from deferred DAG on submit_binary: ordinal pack → ladder snap → uni classifier vs MMPD.

```python
# temp/eval_univariate_patch_refine_ordinal_vs_mmpd.py:1-80
#!/usr/bin/env python3
"""Live ordinal h96 patch-refine vs non-ordinal MMPD univariate disc.

Entry for the ordinal disc campaign. Reads completed MMPD raw packs, generates
ordinal patch-refine forecasts from coarse+patch_refine ckpts, snaps GT/MMPD
(and off-lattice binary) onto the causal 256-row support
(utils.patch_refine_ordinal_ladder), then trains via
eval_discriminator_binary_vs_mmpd_univariate.train_classifier.

Sibling: temp/eval_univariate_patch_refine_vs_gt.py (non-ordinal, binary vs GT only).
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from copy import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.diffusion_tsf.pipeline.globals_bridge import patch_globals
from models.diffusion_tsf.train_multivariate_pipeline import (
    load_dataset,
    load_wrapped_guidance,
    resolve_pipeline_data_subset,
)
from temp.eval_univariate_patch_refine_vs_gt import load_patch_refine_run
from utils.dual_scale_bin_filter import (
    align_mmpd_to_binary_dataset_norm,
)
from utils.eval_discriminator_binary_vs_mmpd_univariate import train_classifier
from utils.disc_shared import (
    apply_smoke_defaults as apply_base_smoke_defaults,
    binary_mmpd_train_scaler_map,
    collect_partials,
    parse_args as parse_base_args,
    split_windows,
    write_json,
)
from utils.eval_mmpd_gaussian_anchor import (
    load_tsf_pack_pool,
    parse_pack_splits,
    run_subset_id,
    run_test_stride,
    run_train_stride,
    run_variate_indices,
)
from utils.staged_binary_forecast import generate_staged_forecast
from utils.forecast_pack_reduce import assert_not_anchor_agg, reduce_pack_forecast
from utils.patch_refine_ordinal_ladder import (
    assert_on_patch_refine_levels,
    assert_support_is_causal,
    legal_patch_refine_levels_dataset_z,
    snap_to_patch_refine_levels,
)
from utils.disc_bin_center_shift import bin_center_shift  # noqa: E402
from utils.visualize_staged_eval_2d_preds import _build_state, _load_stage_model, _resolve_guidance_ckpt
from utils.visualize_discriminator_univariate_confusions import visualize_univariate_combo
from utils.binary_mmpd_sample_panels import generate_binary_vs_mmpd_anchor_prob_panels


DEFAULT_OUTPUT = REPO_ROOT / "results" / "datasets" / "disc-ordinal-patch-refine-h96-vs-mmpd"
```
```python
# temp/eval_univariate_patch_refine_ordinal_vs_mmpd.py:531-710
def _materialize_binary(
    args: argparse.Namespace,
    dataset: str,
    root: Path,
    indices: Sequence[int],
    device: torch.device,
) -> tuple[Mapping[str, np.ndarray], Any, Any]:
    """Generate/cache ordinal patch-refine pack aligned to MMPD window indices."""
    cache = args.raw_eval_dir / f"binary_ordinal_patch_refine_{dataset}.npz"
    run, coarse, refine, ladder = _load_binary_models(args, dataset, root, device)
    required_cached = {
        "y_true", "samples", "indices", "past",
        "unblended_nonoverlap_patch_pred", "unblended_nonoverlap_patch_gt",
        "unblended_nonoverlap_patch_past", "unblended_nonoverlap_patch_parent", "patch_vote_counts",
    }
    if cache.is_file() and not args.force_raw_eval:
        with np.load(cache) as data:
            pack = {key: data[key] for key in data.files}
        if required_cached.issubset(pack) and np.array_equal(pack.get("indices"), np.asarray(indices, dtype=np.int64)):
            return pack, run, ladder

    pool, starts, splits, _, _ = load_tsf_pack_pool(
        dataset,
        run_variate_indices(run),
        lookback=args.lookback,
        horizon=args.horizon,
        train_stride=run_train_stride(run),
        test_stride=_pack_test_stride(args),
        pack_splits=parse_pack_splits(args.pack_splits),
        use_ordinal_window_norm=False,
    )
    if not indices or min(indices) < 0 or max(indices) >= len(pool):
        raise ValueError(
            f"{dataset}: MMPD indices are outside the shared TSF pool "
            f"(n_indices={len(indices)}, index_range="
            f"[{min(indices) if indices else 'n/a'}, {max(indices) if indices else 'n/a'}], "
            f"pool_len={len(pool)}, train_stride={run_train_stride(run)}, "
            f"pack_test_stride={_pack_test_stride(args)}, "
            f"binary_meta_test_stride={run_test_stride(run)}, "
            f"pack_splits={parse_pack_splits(args.pack_splits)}). "
            f"MMPD matched-binary packs require pack_test_stride=4 "
            f"(got {_pack_test_stride(args)})."
        )
    print(
        f"[{dataset}] pack pool: len={len(pool)} pack_test_stride={_pack_test_stride(args)} "
        f"n_indices={len(indices)} train_stride={run_train_stride(run)} "
        f"binary_meta_test_stride={run_test_stride(run)}",
        flush=True,
    )

    batch_size = max(1, int(args.raw_binary_batch_size))
    if bool(getattr(args, "probe_binary_batch_size", False)) and device.type == "cuda":
        from models.diffusion_tsf.pipeline.phases.staged_eval import (
            _probe_max_staged_eval_batch_size,
        )

        sample_past, _sample_future = pool[int(indices[0])]
        max_fit = _probe_max_staged_eval_batch_size(
            coarse_model=coarse,
            fine_model=refine,
            lookback=int(sample_past.shape[-1]),
            n_variates=int(sample_past.shape[0]),
            device=device,
            det_kwargs={
                "sampler": args.probabilistic_sampler,
                "num_inference_steps": 1,
            },
            joint_dual=False,
            min_bs=1,
            max_bs=int(getattr(args, "probe_binary_batch_size_max", 64)),
        )
        if max_fit != batch_size:
            print(
                f"[{dataset}] binary generate probe: config batch={batch_size} -> probed={max_fit}",
                flush=True,
            )
        batch_size = max(1, int(max_fit))

    loader = DataLoader(
        Subset(pool, list(indices)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    true_chunks: List[np.ndarray] = []
    pred_chunks: List[np.ndarray] = []
    past_chunks: List[np.ndarray] = []
    patch_pred_chunks: List[np.ndarray] = []
    patch_gt_chunks: List[np.ndarray] = []
    patch_past_chunks: List[np.ndarray] = []
    patch_parent_chunks: List[np.ndarray] = []
    patch_start_chunks: List[np.ndarray] = []
    patch_variate_chunks: List[np.ndarray] = []
    vote_count_chunks: List[np.ndarray] = []
    patch_diag = {"candidates": 0, "rejected_invalid_or_out_of_bounds": 0, "selected": 0}
    windows_seen = 0
    n_batches = len(loader)
    print(
        f"[{dataset}] materializing binary packs: windows={len(indices)} "
        f"batches={n_batches} batch_size={batch_size} "
        f"sampler={args.probabilistic_sampler} steps={args.num_sampling_steps}",
        flush=True,
    )
    with torch.no_grad():
        for batch_i, (past, future) in enumerate(loader):
            past = past.to(device)
            overlap = int(refine.config.lookback_overlap)
            target = future.to(device)[..., overlap:] if overlap else future.to(device)
            torch.manual_seed(int(args.seed) + batch_i * 1009)
            result = generate_staged_forecast(
                coarse,
                refine,
                past,
                vertical_dual=False,
                sampler=args.probabilistic_sampler,
                num_inference_steps=args.num_sampling_steps,
            )
            prediction = result["prediction_global_norm"]
            if prediction.shape != target.shape:
                raise RuntimeError(
                    f"{dataset}: binary prediction/target mismatch {tuple(prediction.shape)} vs {tuple(target.shape)}"
                )
            levels = legal_patch_refine_levels_dataset_z(
                past.detach().cpu().numpy(), ladder=ladder, device=device,
            )
            patch_values = _unblended_nonoverlap_patch_batch(
                result=result,
                target=target,
                past=past,
                legal_levels=levels,
                canvas_height=int(refine.config.patch_refine_canvas_height),
                patch_height=int(refine.config.patch_refine_patch_height),
                patch_width=int(refine.config.patch_refine_patch_width),
            )
            true_chunks.append(target.cpu().numpy())
            pred_chunks.append(prediction.cpu().numpy())
            past_chunks.append(past.cpu().numpy())
            patch_pred_chunks.append(patch_values[0])
            patch_gt_chunks.append(patch_values[1])
            patch_past_chunks.append(patch_values[2])
            patch_parent_chunks.append(patch_values[3] + windows_seen)
            patch_start_chunks.append(patch_values[4])
            patch_variate_chunks.append(patch_values[5])
            vote_count_chunks.append(result["patch_vote_counts"].detach().cpu().numpy())
            for key, value in patch_values[6].items():
                patch_diag[key] += int(value)
            windows_seen += int(past.shape[0])
            if (batch_i + 1) == n_batches or (batch_i + 1) % max(1, n_batches // 10) == 0:
                print(
                    f"[{dataset}] binary generate {batch_i + 1}/{n_batches} "
                    f"(windows_done={windows_seen}/{len(indices)})",
                    flush=True,
                )
    patch_pred = _concat_patch_chunks(patch_pred_chunks, width=int(refine.config.patch_refine_patch_width))
    patch_gt = _concat_patch_chunks(patch_gt_chunks, width=int(refine.config.patch_refine_patch_width))
    patch_past = _concat_patch_chunks(patch_past_chunks, width=int(args.lookback))
    patch_parent = _concat_int_chunks(patch_parent_chunks)
    patch_start = _concat_int_chunks(patch_start_chunks)
    patch_variate = _concat_int_chunks(patch_variate_chunks)
    for parent, variate in set(zip(patch_parent.tolist(), patch_variate.tolist())):
        starts_for_parent = np.sort(patch_start[(patch_parent == parent) & (patch_variate == variate)])
        if np.any(starts_for_parent[1:] < starts_for_parent[:-1] + int(refine.config.patch_refine_patch_width)):
            raise RuntimeError(f"{dataset}: coherent raw patch examples overlap in parent row {parent}")
    pack = {
        "y_true": np.concatenate(true_chunks).astype(np.float32),
        "samples": np.concatenate(pred_chunks).astype(np.float32)[:, :, None, :],
        "past": np.concatenate(past_chunks).astype(np.float32),
        "indices": np.asarray(indices, dtype=np.int64),
        "series_starts": starts[np.asarray(indices, dtype=np.int64)],
        "pack_splits": np.asarray(splits),
        "unblended_nonoverlap_patch_pred": patch_pred,
        "unblended_nonoverlap_patch_gt": patch_gt,
        "unblended_nonoverlap_patch_past": patch_past,
        "unblended_nonoverlap_patch_parent": patch_parent,
        "unblended_nonoverlap_patch_start": patch_start,
        "unblended_nonoverlap_patch_variate": patch_variate,
        "unblended_patch_candidates": np.asarray(patch_diag["candidates"], dtype=np.int64),
        "unblended_patch_rejected_invalid_or_out_of_bounds": np.asarray(
            patch_diag["rejected_invalid_or_out_of_bounds"], dtype=np.int64,
```
```python
# temp/eval_univariate_patch_refine_ordinal_vs_mmpd.py:840-1019
def run_eval(args: argparse.Namespace) -> None:
    """Per dataset: load MMPD pack → materialize binary → lattice snap → train discs."""
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else f"cuda:{args.gpu}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.raw_eval_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "partials").mkdir(exist_ok=True)
    for dataset in args.datasets:
        mmpd_pack = _mmpd_pack(args.mmpd_output_root, dataset)
        indices = [int(value) for value in mmpd_pack["indices"].tolist()]
        n_full = len(indices)
        if args.assert_only:
            cap = int(args.assert_max_windows or 8)
            if n_full > cap:
                rng = np.random.default_rng(
                    int(args.seed) + (sum(ord(c) for c in dataset) % 10_007)
                )
                pick = np.sort(rng.choice(n_full, size=cap, replace=False))
                indices, mmpd_pack = _subset_mmpd_aligned(indices, mmpd_pack, pick=pick)
                print(
                    f"[{dataset}] assert-only: sampling {cap}/{n_full} windows for lattice gate",
                    flush=True,
                )
        else:
            disc_stride = args.disc_index_stride
            if disc_stride is None:
                # pack is already eval_test_stride=4; stride 1 keeps the full MMPD-aligned pool.
                disc_stride = 1
            indices, mmpd_pack = _thin_disc_windows(
                indices,
                mmpd_pack,
                dataset=dataset,
                seed=int(args.seed),
                test_fraction=float(args.test_fraction),
                disc_index_stride=int(disc_stride),
            )
        binary_pack, run, ladder = _materialize_binary(
            args, dataset, args.checkpoint_dir, indices, device,
        )
        binary_gt = binary_pack["y_true"].astype(np.float32)
        binary_pred = reduce_pack_forecast(binary_pack, agg=args.fake_agg)
        mmpd_gt = mmpd_pack["y_true"].astype(np.float32)
        mmpd_pred = reduce_pack_forecast(mmpd_pack, agg=args.fake_agg)
        print(
            f"[{dataset}] disc forecasts via fake_agg={args.fake_agg} "
            f"(binary S={binary_pack['samples'].shape[2]}, mmpd S={mmpd_pack['samples'].shape[2]})",
            flush=True,
        )
        if not np.array_equal(binary_pack["indices"], mmpd_pack["indices"]):
            raise RuntimeError(f"{dataset}: binary/MMPD indices differ")
        scalers = binary_mmpd_train_scaler_map(args, run)
        mmpd_binary_z, align = align_mmpd_to_binary_dataset_norm(
            binary_y_true=binary_gt,
            mmpd_y_true=mmpd_gt,
            mmpd_fakes=mmpd_pred,
            **scalers,
        )
        past_pool, _, _, _, _ = load_tsf_pack_pool(
            dataset,
            run_variate_indices(run),
            lookback=args.lookback,
            horizon=args.horizon,
            train_stride=run_train_stride(run),
            test_stride=_pack_test_stride(args),
            pack_splits=parse_pack_splits(args.pack_splits),
            use_ordinal_window_norm=False,
        )
        past = np.stack([past_pool[index][0].detach().cpu().numpy() for index in indices]).astype(np.float32)
        legal_levels = legal_patch_refine_levels_dataset_z(past, ladder=ladder, device=device)
        # Real-checkpoint counterpart of the synthetic causal contract.  The
        # legal support must not change if only the future fixture changes.
        assert_support_is_causal(
            past,
            binary_gt,
            binary_gt + np.float32(123.456),
            ladder=ladder,
            canvas_height=256,
            device=device,
        )
        gt, gt_snap = snap_to_patch_refine_levels(binary_gt, legal_levels)
        mmpd, mmpd_snap = snap_to_patch_refine_levels(mmpd_binary_z, legal_levels)
        mmpd_window_mean, mmpd_window_std, mmpd_inverse_residual = _mmpd_instance_summary(
            binary_past=past, mmpd_prediction=mmpd_binary_z, scalers=scalers,
        )
        # Patch-refine decode should land on the 256-row ladder; allow small fp slack
        # from unique-seg blending (elec disc hit max_error≈6e-3 vs atol 1e-6).
        # Fail-soft for the live disc path: snap binary onto the ladder so training
        # finishes even when sample0 sits off support. assert-only still hard-fails.
        binary_atol = _binary_lattice_atol(legal_levels)
        binary_raw = np.asarray(binary_pred, dtype=np.float32)
        if args.assert_only:
            binary = binary_raw
            binary_staged_stats = assert_on_patch_refine_levels(
                binary, legal_levels, atol=binary_atol,
            )
            binary_staged_stats.update({
                "raw_binary_retained": 1.0,
                "support_atol": float(binary_atol),
            })
        else:
            binary, binary_snap = snap_to_patch_refine_levels(binary_raw, legal_levels)
            raw_err = float(np.abs(binary_raw - binary).max(initial=0.0))
            binary_staged_stats = assert_on_patch_refine_levels(binary, legal_levels)
            binary_staged_stats.update(binary_snap)
            binary_staged_stats.update({
                "raw_binary_retained": 0.0 if raw_err > float(binary_atol) else 1.0,
                "raw_max_support_error": raw_err,
                "support_atol": float(binary_atol),
            })
            if raw_err > float(binary_atol):
                print(
                    f"[{dataset}] binary off lattice max_error={raw_err:.6g} "
                    f"atol={binary_atol:.6g}; snapping for disc "
                    f"(mean_abs_snap_delta={binary_snap['mean_abs_snap_delta']:.6g})",
                    flush=True,
                )
        lattice = {
            "gt": assert_on_patch_refine_levels(gt, legal_levels),
            "binary_staged": binary_staged_stats,
            "mmpd": assert_on_patch_refine_levels(mmpd, legal_levels),
        }
        lattice["gt"].update(gt_snap)
        lattice["mmpd"].update(mmpd_snap)
        lattice["mmpd_alignment"] = align
        lattice["causal_support_real_checkpoint_asserted"] = 1.0
        write_json(args.raw_eval_dir / f"lattice_assertion_{dataset}.json", lattice)
        if args.assert_only:
            print(f"[{dataset}] real checkpoint snapping/assertion gate passed", flush=True)
            continue
        # Bin-center shift runs per L-slice inside UnivariateRealVsFakeDataset
        # (replaces zscore_time). Do not pre-shift full-H packs here.
        if bool(getattr(args, "disc_bin_center_shift", False)):
            print(
                f"[{dataset}] disc_bin_center_shift=ON (per L-slice in dataset; "
                f"reduce={getattr(args, 'disc_bin_center_reduce', 'per_variate')}; "
                f"zscore_time disabled)",
                flush=True,
            )
        bundle = SimpleNamespace(
            fakes={"binary_staged": binary, "mmpd": mmpd},
            y_true_by_source={"binary_staged": gt, "mmpd": gt.copy()},
            past=past,
            legal_levels=np.asarray(legal_levels, dtype=np.float32),
            indices=np.asarray(indices, dtype=np.int64),
            series_starts=binary_pack["series_starts"],
            run=run,
            pack_splits=[str(x) for x in binary_pack["pack_splits"].tolist()],
        )
        splits = split_windows(
            len(gt), args, dataset, indices=bundle.indices, lookback=args.lookback,
            horizon=args.horizon, test_stride=_pack_test_stride(args), series_starts=bundle.series_starts,
        )
        by_source: Dict[str, Dict[str, float]] = {}
        for source in ("binary_staged", "mmpd"):
            per_length: Dict[str, float] = {}
            for length in args.slice_lengths:
                if int(length) <= args.horizon:
                    per_length[str(int(length))] = train_classifier(
                        args, dataset, source, int(length), bundle, splits, device,
                    )
            write_json(args.output_dir / "partials" / f"{dataset}__{source}.json", per_length)
            by_source[source] = per_length
            nonoverlap_args = copy(args)
            nonoverlap_args.nonoverlapping_patches = True
            nonoverlap_source = f"{source}_candidate_nonoverlap"
            nonoverlap_bundle = SimpleNamespace(
                fakes={nonoverlap_source: bundle.fakes[source]},
                y_true_by_source={nonoverlap_source: bundle.y_true_by_source[source]},
                past=bundle.past,
                legal_levels=bundle.legal_levels,
                indices=bundle.indices,
                series_starts=bundle.series_starts,
                run=bundle.run,
                pack_splits=bundle.pack_splits,
            )
            nonoverlap_per_length: Dict[str, float] = {}
            for length in args.slice_lengths:
                if int(length) <= args.horizon:
                    nonoverlap_per_length[str(int(length))] = train_classifier(
                        nonoverlap_args, dataset, nonoverlap_source, int(length), nonoverlap_bundle, splits, device,
                    )
```
```python
# temp/eval_univariate_patch_refine_ordinal_vs_mmpd.py:1214-1228
def main() -> None:
    args = parse_args()
    apply_smoke_defaults(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.merge_partials_only:
        run_merge_only(args)
        return
    # Shard jobs own partials; merge is a separate pass.
    args.merge_metrics = False
    run_eval(args)


if __name__ == "__main__":
    main()
```
```python
# utils/patch_refine_ordinal_ladder.py:1-136
"""Map h96 ordinal patch-refine absolute rows ↔ dataset-z support.

Live consumer: temp/eval_univariate_patch_refine_ordinal_vs_mmpd.py snaps GT +
MMPD (+ binary if off-lattice) onto the same causal 256-row support before disc
training. Not the legacy 16×16 dual-scale bin path.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

from models.diffusion_tsf.ordinal_window_norm import (
    OrdinalLadder,
    ordinal_decode,
    ordinal_encode,
)


def legal_patch_refine_levels_dataset_z(
    past: np.ndarray,
    *,
    ladder: OrdinalLadder,
    canvas_height: int = 256,
    device: torch.device,
) -> np.ndarray:
    """(N,V,256) dataset-z midpoints for each absolute row; OOD from lookback only."""
    past_np = np.asarray(past, dtype=np.float32)
    if past_np.ndim != 3:
        raise ValueError(f"past must be (N,V,L), got {past_np.shape}")
    if canvas_height <= 0:
        raise ValueError(f"canvas_height must be positive, got {canvas_height}")
    past_t = torch.from_numpy(past_np).to(device)
    with torch.no_grad():
        past_rank, _future_rank, ladder_b, ood_shift = ordinal_encode(
            past_t,
            None,
            ladder=ladder,
            apply_ood_shift=True,
            causal_only=True,
        )
        rank_max = ladder_b.rank_max_per_variate().to(device=device, dtype=past_t.dtype)
        rows = torch.arange(canvas_height, device=device, dtype=past_t.dtype)
        # This is decode_absolute_hir_cdf's midpoint decode for each row.
        rank_centers = ((rows + 0.5) / float(canvas_height)).view(1, 1, -1)
        rank_centers = rank_centers * rank_max.view(1, -1, 1)
        rank_centers = rank_centers.expand(past_t.shape[0], -1, -1)
        _past_z, levels = ordinal_decode(
            past_rank[..., :1], rank_centers, ladder_b, ood_shift=ood_shift,
        )
    if levels is None or not torch.isfinite(levels).all():
        raise ValueError("ordinal patch-refine support contains non-finite values")
    return levels.detach().cpu().numpy().astype(np.float32)


def snap_to_patch_refine_levels(
    values: np.ndarray,
    legal_levels: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Nearest-row snap in binary dataset-z coordinates, with endpoint clamp."""
    vals = np.asarray(values, dtype=np.float32)
    levels = np.asarray(legal_levels, dtype=np.float32)
    if vals.ndim != 3 or levels.ndim != 3:
        raise ValueError(f"expected values/levels (N,V,T)/(N,V,H), got {vals.shape}/{levels.shape}")
    if vals.shape[:2] != levels.shape[:2]:
        raise ValueError(f"values/levels N,V mismatch: {vals.shape}/{levels.shape}")
    if not (np.isfinite(vals).all() and np.isfinite(levels).all()):
        raise ValueError("cannot snap non-finite values")
    delta = np.abs(vals[..., None] - levels[:, :, None, :])
    rows = np.argmin(delta, axis=-1)
    snapped = np.take_along_axis(levels[:, :, None, :], rows[..., None], axis=-1)[..., 0]
    residual = np.abs(vals - snapped)
    return snapped.astype(np.float32), {
        "mean_abs_snap_delta": float(residual.mean()),
        "max_abs_snap_delta": float(residual.max(initial=0.0)),
        "n_rows": float(levels.shape[-1]),
        "n_unique_levels_min": float(min(np.unique(levels[i, j]).size for i in range(levels.shape[0]) for j in range(levels.shape[1]))),
    }


def assert_on_patch_refine_levels(
    values: np.ndarray,
    legal_levels: np.ndarray,
    *,
    atol: float = 1e-6,
) -> Dict[str, float]:
    """Fail unless all values are exactly on their window-specific 256-row support."""
    snapped, stats = snap_to_patch_refine_levels(values, legal_levels)
    err = float(np.abs(np.asarray(values, dtype=np.float32) - snapped).max(initial=0.0))
    if err > float(atol):
        raise AssertionError(
            f"values are off the patch-refine ordinal support: max_error={err:.6g}, atol={atol:.6g}"
        )
    stats["max_support_error"] = err
    return stats


def assert_support_is_causal(
    past: np.ndarray,
    future_a: np.ndarray,
    future_b: np.ndarray,
    *,
    ladder: OrdinalLadder,
    canvas_height: int,
    device: torch.device,
) -> None:
    """Fail if legal rows change when only the future fixture changes."""
    if np.asarray(future_a).shape != np.asarray(future_b).shape:
        raise ValueError("future fixtures must have equal shape")
    past_t = torch.from_numpy(np.asarray(past, dtype=np.float32)).to(device)
    future_a_t = torch.from_numpy(np.asarray(future_a, dtype=np.float32)).to(device)
    future_b_t = torch.from_numpy(np.asarray(future_b, dtype=np.float32)).to(device)
    with torch.no_grad():
        _pa, _fa, _la, shift_a = ordinal_encode(
            past_t, future_a_t, ladder=ladder, apply_ood_shift=True, causal_only=True,
        )
        _pb, _fb, _lb, shift_b = ordinal_encode(
            past_t, future_b_t, ladder=ladder, apply_ood_shift=True, causal_only=True,
        )
    if not torch.equal(shift_a, shift_b):
        raise AssertionError("causal ordinal OOD shift changed when only future changed")
    levels_a = legal_patch_refine_levels_dataset_z(
        past, ladder=ladder, canvas_height=canvas_height, device=device,
    )
    levels_b = legal_patch_refine_levels_dataset_z(
        past, ladder=ladder, canvas_height=canvas_height, device=device,
    )
    if not np.array_equal(levels_a, levels_b):
        raise AssertionError("ordinal patch-refine support changed when only future changed")
```
```python
# utils/forecast_pack_reduce.py:1-109
"""Collapse (N,V,S,H) forecast packs to one (N,V,H) trajectory.

Disc paths default to sample0 (first draw). prob_mean is opt-in averaging.
deterministic/anchor is refused for probabilistic discs via assert_not_anchor_agg.
Callers: disc_shared.build_raw_bundle, temp ordinal vs-mmpd evaluator.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

FAKE_AGG_CHOICES = ("prob_mean", "sample0", "deterministic")
DEFAULT_FAKE_AGG = "sample0"


def pack_index_key(pack: Mapping[str, np.ndarray]) -> str:
    # Over-compat: older packs used window_indices; live code writes indices.
    if "indices" in pack:
        return "indices"
    if "window_indices" in pack:
        return "window_indices"
    raise KeyError("pack has neither 'indices' nor 'window_indices'")


def reduce_pack_forecast(
    pack: Mapping[str, np.ndarray],
    *,
    agg: str = DEFAULT_FAKE_AGG,
) -> np.ndarray:
    """Return ``(N, V, H)`` forecast for the requested aggregation.

    ``sample0`` — first stochastic draw (disc default).
    ``prob_mean`` — mean over sample axis (requires ``samples`` with S>=1).
    ``deterministic`` — anchor / point forecast key.
    """
    mode = str(agg).strip().lower()
    if mode not in FAKE_AGG_CHOICES:
        raise ValueError(f"agg must be one of {FAKE_AGG_CHOICES}, got {agg!r}")

    if mode == "deterministic":
        for key in ("deterministic", "final_anchor", "anchor"):
            if key in pack:
                out = np.asarray(pack[key], dtype=np.float32)
                if out.ndim != 3:
                    raise ValueError(f"{key} must be (N,V,H), got {out.shape}")
                return out
        raise KeyError(
            "deterministic aggregation requested but pack has no "
            "deterministic/final_anchor/anchor array"
        )

    if "samples" not in pack:
        raise KeyError(f"{mode} aggregation requires pack['samples']")
    samples = np.asarray(pack["samples"], dtype=np.float32)
    if samples.ndim != 4 or samples.shape[2] < 1:
        raise ValueError(f"samples must be (N,V,S,H) with S>=1, got {samples.shape}")
    if mode == "sample0":
        return samples[:, :, 0, :].astype(np.float32, copy=False)
    # prob_mean
    if "sample_mean" in pack:
        mean = np.asarray(pack["sample_mean"], dtype=np.float32)
        if mean.shape == samples[:, :, 0, :].shape:
            return mean
    return samples.mean(axis=2).astype(np.float32)


def subset_pack_by_pool_indices(
    pack: Mapping[str, np.ndarray],
    pool_indices: np.ndarray,
    *,
    allow_missing: bool = False,
) -> dict:
    """Keep rows whose pool index is in ``pool_indices`` (order preserved)."""
    key = pack_index_key(pack)
    idx = np.asarray(pack[key], dtype=np.int64)
    want = np.asarray(pool_indices, dtype=np.int64)
    pos = {int(i): j for j, i in enumerate(idx.tolist())}
    rows = []
    missing = []
    for i in want.tolist():
        j = pos.get(int(i))
        if j is None:
            missing.append(int(i))
        else:
            rows.append(j)
    if missing and not allow_missing:
        raise KeyError(f"{len(missing)} pool indices missing from pack (e.g. {missing[:5]})")
    if not rows:
        raise ValueError("no overlapping pool indices between pack and request")
    rows_arr = np.asarray(rows, dtype=np.int64)
    n = int(idx.shape[0])
    out = {}
    for k, v in pack.items():
        if isinstance(v, np.ndarray) and v.shape[:1] == (n,):
            out[k] = v[rows_arr]
        else:
            out[k] = v
    return out


def assert_not_anchor_agg(agg: str) -> None:
    if str(agg).strip().lower() == "deterministic":
        raise ValueError(
            "refusing deterministic/anchor aggregation for a probabilistic disc path; "
            "use agg='sample0' (or 'prob_mean' only if mean-over-S is intentional)"
        )
```
```python
# utils/disc_shared.py:335-454
def ensure_raw_packs(
    args: argparse.Namespace,
    dataset: str,
    device: torch.device,
) -> Tuple[Any, Dict[str, Any], List[int], Dict[str, Dict[str, np.ndarray]], np.ndarray, List[str]]:
    """Resolve ckpt + indices; materialize binary_staged / mmpd npz if missing."""
    anchor_config = _anchor_config_for(args, dataset)
    ckpt_dir = resolve_staged_ckpt_dir(args.ckpt_base, dataset, anchor_config)
    run, sub = staged_anchor_run(dataset, ckpt_dir, args.test_stride)
    indices = saved_indices(args.raw_eval_dir, dataset)
    eval_args = raw_eval_args(args, dataset)
    if indices is None:
        indices = make_indices(eval_args, run)

    if "binary_staged" in args.fake_sources:
        binary_path = pack_path(args.raw_eval_dir, "binary_staged", dataset)
        if args.force_raw_eval or not binary_path.is_file():
            print(f"[raw] materializing binary_staged/{dataset} -> {binary_path}", flush=True)
            evaluate_staged_binary(eval_args, run, sub, indices, device)

    if "mmpd" in args.fake_sources:
        mmpd_path = pack_path(args.raw_eval_dir, "mmpd", dataset)
        if args.force_raw_eval or not mmpd_path.is_file():
            print(f"[raw] materializing mmpd/{dataset} -> {mmpd_path}", flush=True)
            ensure_mmpd_repo(args.mmpd_repo, update=not args.no_update_mmpd)
            run_mmpd_eval(eval_args, run, indices)

    packs: Dict[str, Dict[str, np.ndarray]] = {}
    for fake_source in args.fake_sources:
        path = pack_path(args.raw_eval_dir, fake_source, dataset)
        if not path.is_file():
            raise FileNotFoundError(f"raw pack missing after materialization: {path}")
        pack = load_npz(path)
        validate_stochastic_pack(path, pack)
        packs[fake_source] = pack

    series_starts, pack_splits = _resolve_pack_series_meta(args, run, indices, packs)
    # Persist meta onto MMPD packs that predate series_starts.
    for fake_source, pack in packs.items():
        if "series_starts" not in pack or "pack_splits" not in pack:
            path = pack_path(args.raw_eval_dir, fake_source, dataset)
            merged = dict(pack)
            merged["series_starts"] = series_starts
            merged["pack_splits"] = np.asarray(pack_splits)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **merged)
            packs[fake_source] = merged

    return run, sub, indices, packs, series_starts, pack_splits


def _resolve_pack_series_meta(
    args: argparse.Namespace,
    run: Any,
    indices: Sequence[int],
    packs: Mapping[str, Mapping[str, np.ndarray]],
) -> Tuple[np.ndarray, List[str]]:
    for pack in packs.values():
        if "series_starts" in pack:
            starts = np.asarray(pack["series_starts"], dtype=np.int64)
            if starts.shape[0] != len(indices):
                raise ValueError(
                    f"{run.dataset}: series_starts length {starts.shape[0]} != n_indices {len(indices)}"
                )
            splits = (
                [str(x) for x in np.asarray(pack["pack_splits"]).tolist()]
                if "pack_splits" in pack
                else parse_pack_splits(getattr(args, "pack_splits", None))
            )
            return starts, splits

    lookback, horizon = dataset_window_lengths_for_run(args, run)
    pack_splits = parse_pack_splits(getattr(args, "pack_splits", None))
    _pool, series_starts_full, splits, _lens, _stats = load_tsf_pack_pool(
        run.dataset,
        run_variate_indices(run),
        lookback=lookback,
        horizon=horizon,
        train_stride=run_train_stride(run),
        test_stride=run_test_stride(run),
        pack_splits=pack_splits,
    )
    idx_arr = np.asarray(indices, dtype=np.int64)
    return series_starts_full[idx_arr], splits


def load_past_windows(
    args: argparse.Namespace,
    run: Any,
    indices: Sequence[int],
    device: torch.device,
) -> np.ndarray:
    from torch.utils.data import Subset

    lookback, horizon = dataset_window_lengths_for_run(args, run)
    pack_splits = parse_pack_splits(getattr(args, "pack_splits", None))
    pool, _starts, _splits, _lens, _stats = load_tsf_pack_pool(
        run.dataset,
        run_variate_indices(run),
        lookback=lookback,
        horizon=horizon,
        train_stride=run_train_stride(run),
        test_stride=run_test_stride(run),
        pack_splits=pack_splits,
    )
    subset = Subset(pool, list(indices))
    loader = DataLoader(
        subset,
        batch_size=args.raw_load_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    past_all: List[np.ndarray] = []
    for past, _future in loader:
        past_all.append(past.numpy())
    return np.concatenate(past_all, axis=0)


def binary_mmpd_train_scaler_map(args: argparse.Namespace, run: Any) -> Dict[str, np.ndarray]:
```
```python
# utils/disc_shared.py:506-625
def build_raw_bundle(
    args: argparse.Namespace,
    dataset: str,
    device: torch.device,
) -> RawBundle:
    """Load packs, reduce fakes, optionally map MMPD→binary-z / ladder / bin-match."""
    run, sub, indices, packs, series_starts, pack_splits = ensure_raw_packs(args, dataset, device)
    past = load_past_windows(args, run, indices, device)
    y_true_by_source: Dict[str, np.ndarray] = {}
    fakes: Dict[str, np.ndarray] = {}
    ref_shape: Optional[Tuple[int, ...]] = None
    from utils.forecast_pack_reduce import assert_not_anchor_agg, reduce_pack_forecast

    fake_agg = str(getattr(args, "fake_agg", "sample0") or "sample0")
    assert_not_anchor_agg(fake_agg)
    for fake_source, pack in packs.items():
        y_true = pack["y_true"].astype(np.float32)
        # Default: first stochastic draw (sample0). No mean-over-S; anchor
        # rejected via assert_not_anchor_agg. Existing S>1 packs still work.
        fake = reduce_pack_forecast(pack, agg=fake_agg)
        if ref_shape is None:
            ref_shape = y_true.shape
        elif y_true.shape != ref_shape:
            raise ValueError(f"{dataset}/{fake_source}: y_true shape differs from first pack")
        if fake.shape != ref_shape:
            raise ValueError(f"{dataset}/{fake_source}: fake shape differs from y_true")
        if not np.array_equal(pack["indices"], np.asarray(indices, dtype=pack["indices"].dtype)):
            raise ValueError(f"{dataset}/{fake_source}: raw pack indices do not match discriminator indices")
        y_true_by_source[fake_source] = y_true
        fakes[fake_source] = fake
    print(f"[{dataset}] disc fake aggregation: {fake_agg}", flush=True)

    if past.shape[0] != ref_shape[0]:
        raise ValueError(f"{dataset}: past/y_true window mismatch {past.shape[0]} vs {ref_shape[0]}")
    validate_variate_alignment(dataset, run, sub, past, y_true_by_source, fakes)

    if (
        bool(getattr(args, "mmpd_to_binary_dataset_norm", False))
        and "binary_staged" in y_true_by_source
        and "mmpd" in y_true_by_source
    ):
        scalers = binary_mmpd_train_scaler_map(args, run)
        aligned_mmpd, align_stats = align_mmpd_to_binary_dataset_norm(
            binary_y_true=y_true_by_source["binary_staged"],
            mmpd_y_true=y_true_by_source["mmpd"],
            mmpd_fakes=fakes["mmpd"],
            **scalers,
        )
        # Labels must use exactly one GT tensor.  This is deliberate rather
        # than an approximate post-hoc equality check: both model forecasts
        # are scored against binary's train-split dataset-z target values.
        y_true_by_source["mmpd"] = y_true_by_source["binary_staged"].copy()
        fakes["mmpd"] = aligned_mmpd
        print(
            f"[{dataset}] MMPD→binary dataset-norm map: "
            f"scale=[{align_stats['scale_min']:.8f},{align_stats['scale_max']:.8f}] "
            f"offset=[{align_stats['offset_min']:.8f},{align_stats['offset_max']:.8f}] "
            f"target_rmse_max={align_stats['target_rmse_max']:.2e} "
            f"target_max_abs={align_stats['target_max_abs']:.2e}",
            flush=True,
        )

    if len(y_true_by_source) > 1:
        sources = list(y_true_by_source)
        ref = y_true_by_source[sources[0]]
        for src in sources[1:]:
            other = y_true_by_source[src]
            mse = float(np.mean((ref - other) ** 2))
            if mse > 1e-6:
                msg = (
                    f"{dataset}: y_true differs between {sources[0]} and {src} "
                    f"(mse={mse:.6f}); packs are not in the same coordinate space"
                )
                if getattr(args, "ordinal_ladder_quantize", False) and "mmpd" in fakes:
                    raise ValueError(
                        msg + "; refusing --ordinal-ladder-quantize onto a mismatched ladder"
                    )
                # Fail-soft: texture disc can train each source on its own GT;
                # ordinal ladder path above hard-fails instead.
                print(f"[warn] {msg}; each discriminator uses its own pack GT.", flush=True)

    if args.bin_match_filter:
        # Same path binary ordinal_norm uses: train z-score → ordinal ranks
        # (+ OOD constant shift) → [optional stride subsample] → bounded coarse/fine
        # → upsample → ordinal decode. No instance norm.
        ladder = load_ordinal_ladder_for_run(args, run)
        _ms, coarse_h, fine_h = resolve_dual_scale_bin_params(
            dataset,
            sub,
            fallback_max_scale=args.bin_max_scale,
            coarse_height=args.bin_coarse_height or args.bin_image_height,
            fine_height=args.bin_fine_height or args.bin_image_height,
        )
        from models.diffusion_tsf.pipeline.config import load_experiment_config

        cfg = load_experiment_config(str(_binary_config_path(args, run.dataset)))
        repr_stride = int(
            (cfg.get("experiment") or {}).get("representation_time_stride", 1) or 1
        )
        print(
            f"[{dataset}] applying ordinal dual-scale bin-match filter={args.bin_match_filter} "
            f"(coarse={coarse_h}, fine={fine_h}, repr_stride={repr_stride}, "
            f"decoder={args.bin_decoder}, ood_shift=on, no instance-norm)",
            flush=True,
        )
        args._resolved_bin_repr_time_stride = repr_stride
        if args.binary_debias_quantization:
            raise ValueError(
                "--bin-match-filter already canonicalizes all selected sources onto the "
                "binary ordinal lattice; refuse combining with --binary-debias-quantization "
                "(would jitter only binary_staged after the shared round-trip)"
            )
        y_true_by_source, fakes = apply_bin_match_to_bundle(
            mode=args.bin_match_filter,
            past=past.astype(np.float32),
            y_true_by_source=y_true_by_source,
            fakes=fakes,
            ladder=ladder,
            coarse_height=coarse_h,
            fine_height=fine_h,
```
```python
# utils/disc_shared.py:773-892
def split_windows(
    n_windows: int,
    args: argparse.Namespace,
    dataset: str,
    *,
    indices: Optional[Sequence[int]] = None,
    lookback: Optional[int] = None,
    horizon: Optional[int] = None,
    test_stride: Optional[int] = None,
    series_starts: Optional[Sequence[int]] = None,
) -> Dict[str, np.ndarray]:
    """Temporal train/val/test over absolute series starts (no random shuffle leak)."""
    if args.max_windows is not None:
        n_windows = min(n_windows, int(args.max_windows))
    if indices is None or lookback is None or horizon is None or test_stride is None:
        raise ValueError("split_windows requires indices, lookback, horizon, and test_stride")
    raw_indices = [int(i) for i in list(indices)[:n_windows]]
    if len(raw_indices) != n_windows:
        raise ValueError(f"{dataset}: got {len(raw_indices)} split indices for {n_windows} windows")
    starts_all = None if series_starts is None else list(series_starts)[:n_windows]

    starts, ends = window_time_bounds(
        dataset,
        raw_indices,
        int(lookback),
        int(horizon),
        int(test_stride),
        series_starts=starts_all,
    )
    order = np.argsort(starts, kind="mergesort")
    n_train_target = max(1, int(round(len(order) * args.train_fraction)))
    n_val_target = max(1, int(round(len(order) * args.val_fraction)))
    if n_train_target + n_val_target >= len(order):
        n_val_target = max(1, len(order) - n_train_target - 1)
    n_test = len(order) - n_train_target - n_val_target
    if n_test < 1:
        raise ValueError(f"not enough windows for train/val/test split: {len(order)}")

    test = order[-n_test:]
    test_start = int(starts[test].min())
    # Hold out anything whose span reaches into the test region. No silent fallback.
    train_val_pool = np.asarray(
        [idx for idx in order[:-n_test] if int(ends[idx]) <= test_start],
        dtype=np.int64,
    )
    if len(train_val_pool) < 2:
        raise ValueError(
            f"{dataset}: hard temporal purge left {len(train_val_pool)} train/val windows "
            f"(need >=2) before test_start={test_start}. "
            f"windows={len(order)} test={n_test} lookback={lookback} horizon={horizon}. "
            f"Raise --pack-fraction / enlarge pack_splits — overlapping fallback is disabled."
        )

    # Allow overlap *within* train and within val (needed at lb336/hz720). The leak we
    # kill is train/val ↔ test absolute-time overlap, not within-split density.
    val_ratio = args.val_fraction / max(args.train_fraction + args.val_fraction, 1e-8)
    n_val = max(1, int(round(len(train_val_pool) * val_ratio)))
    if n_val >= len(train_val_pool):
        n_val = len(train_val_pool) - 1
    # Chronological train then val within the purged pool.
    tv_order = train_val_pool[np.argsort(starts[train_val_pool], kind="mergesort")]
    train = tv_order[:-n_val]
    val = tv_order[-n_val:]

    print(
        f"[split] {dataset}: pack={len(order)} -> train/val/test="
        f"{len(train)}/{len(val)}/{len(test)} "
        f"(raw targets {n_train_target}/{n_val_target}/{n_test}; "
        f"test_start={test_start}; train/val purged vs test only)",
        flush=True,
    )
    return {
        "train": np.sort(train),
        "val": np.sort(val),
        "test": np.sort(test),
    }


def stable_hash(text: str) -> int:
    value = 0
    for ch in text:
        value = (value * 131 + ord(ch)) % 1_000_003
    return value


def zscore_time(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mean) / np.maximum(std, 1e-5)


class HorizonSliceDataset(Dataset):
    def __init__(
        self,
        past: np.ndarray,
        real: np.ndarray,
        fake: np.ndarray,
        windows: np.ndarray,
        slice_len: int,
        *,
        seed: int,
        offset_stride: int = 1,
        max_examples: Optional[int] = None,
        include_past: bool = True,
        apply_zscore: bool = True,
    ) -> None:
        if real.shape != fake.shape:
            raise ValueError(f"real/fake shape mismatch: {real.shape} vs {fake.shape}")
        if real.shape[0] != past.shape[0]:
            raise ValueError(f"past/real window mismatch: {past.shape[0]} vs {real.shape[0]}")
        if slice_len > real.shape[-1]:
            raise ValueError(f"slice_len={slice_len} exceeds horizon={real.shape[-1]}")

        self.past = past
        self.real = real
        self.fake = fake
        self.slice_len = int(slice_len)
        self.include_past = bool(include_past)
        self.apply_zscore = bool(apply_zscore)
        offsets = list(range(0, real.shape[-1] - slice_len + 1, max(1, int(offset_stride))))
```
```python
# utils/disc_shared.py:1-60
#!/usr/bin/env python3
"""Shared disc protocol: packs, coordinate align, temporal splits, train loop.

Not a standalone entrypoint. Live ordinal campaign:
  temp/eval_univariate_patch_refine_ordinal_vs_mmpd.py
Sibling non-ordinal vs-GT: temp/eval_univariate_patch_refine_vs_gt.py
Univariate train_classifier lives in eval_discriminator_binary_vs_mmpd_univariate
(this file's train_classifier is the multivariate / texture path).

Flow: ensure_raw_packs → build_raw_bundle (norm align / bin-match / ladder) →
split_windows → train_classifier → partials merge.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.dual_scale_bin_filter import (
    BIN_MATCH_CHOICES,
    align_mmpd_to_binary_dataset_norm,
    apply_bin_match_to_bundle,
)
from utils.binary_disc_debias import (
    debias_binary_staged_fakes,
    quantize_to_ordinal_ladder,
    resolve_dual_scale_bin_params,
)
from utils.eval_mmpd_gaussian_anchor import (
    DEFAULT_MMPD_DATA,
    DEFAULT_MMPD_REPO,
    ensure_mmpd_repo,
    load_tsf_pack_pool,
    mmpd_data_split,
    mmpd_staged_filename_for_run,
    load_tsf_test_subset,
    parse_pack_splits,
    run_mmpd_eval,
    run_subset_id,
    run_test_stride,
```
```python
# utils/eval_discriminator_binary_vs_mmpd_univariate.py:1-80
#!/usr/bin/env python3
"""Univariate real-vs-fake disc (binary vs GT, MMPD vs GT).

Shared lib for temp/eval_univariate_patch_refine_ordinal_vs_mmpd.py and
temp/eval_univariate_patch_refine_vs_gt.py (train_classifier + dataset).
Protocol matches the multivariate texture disc: label 1=fake, 0=GT, one model
per (dataset, fake_source, L), patches pooled across variates.

Packs/splits come from disc_shared; this file owns the univariate Dataset +
train loop. Prefer the temp/ ordinal evaluator as the live campaign entry.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.disc_bin_center_shift import bin_center_shift  # noqa: E402
from utils.disc_shared import (  # noqa: E402
    DEFAULT_DISC_OUTPUT,
    FAKE_SOURCES,
    InvertedSliceDiscriminator,
    LOG2,
    apply_smoke_defaults as _apply_smoke_defaults_base,
    binary_auroc,
    build_raw_bundle,
    parse_args as disc_parse_args,
    split_windows,
    stable_hash,
    window_level_metrics,
    write_json,
    zscore_time,
)
from utils.eval_mmpd_gaussian_anchor import run_test_stride  # noqa: E402
from utils.mmpd_eval_progress import EvalProgress, fmt_duration  # noqa: E402
from typing import Literal

ReduceMode = Literal["per_variate", "joint"]

DEFAULT_OUTPUT = (
    DEFAULT_DISC_OUTPUT.parent
    / "disc-lb336-hz720-ordinal-four-patch-only-fair-univariate-bin16"
)


class UnivariateRealVsFakeDataset(Dataset):
    """Balanced univariate patches: label 1=fake, 0=GT. Pools all variates."""

    def __init__(
        self,
        real: np.ndarray,
        fake: np.ndarray,
        past: np.ndarray,
        windows: np.ndarray,
        slice_len: int,
        *,
        seed: int,
        offset_stride: int = 1,
        max_examples: Optional[int] = None,
        include_past: bool = False,
        apply_zscore: bool = True,
        apply_bin_center_shift: bool = False,
        legal_levels: Optional[np.ndarray] = None,
        bin_center_reduce: ReduceMode = "per_variate",
    ) -> None:
```
```python
# utils/eval_discriminator_binary_vs_mmpd_univariate.py:231-380
def train_classifier(
    args: argparse.Namespace,
    dataset: str,
    fake_source: str,
    slice_len: int,
    bundle: Any,
    splits: Mapping[str, np.ndarray],
    device: torch.device,
) -> Dict[str, float]:
    """Univariate disc train; reused by temp ordinal vs-mmpd and vs-gt evaluators."""
    fake = bundle.fakes[fake_source]
    y_true = bundle.y_true_by_source[fake_source]
    horizon = int(y_true.shape[-1])
    n_variates = int(y_true.shape[1])
    max_offset = horizon - slice_len
    seed_base = args.seed + stable_hash(f"{dataset}:{fake_source}:uni:{slice_len}")
    include_past = not bool(getattr(args, "candidate_only", False))
    offset_stride = int(getattr(args, "offset_stride", 1) or 1)
    if bool(getattr(args, "nonoverlapping_patches", False)):
        offset_stride = int(slice_len)
    use_offset_embedding = not bool(getattr(args, "no_offset_embedding", False))
    apply_bin_center = bool(getattr(args, "disc_bin_center_shift", False))
    apply_zscore = not apply_bin_center
    legal_levels = getattr(bundle, "legal_levels", None)
    if apply_bin_center and legal_levels is None:
        raise ValueError(
            "disc_bin_center_shift requires bundle.legal_levels (N,V,H) for per-slice centering"
        )
    reduce_mode = str(getattr(args, "disc_bin_center_reduce", "per_variate"))
    if reduce_mode not in ("per_variate", "joint"):
        raise ValueError(f"invalid disc_bin_center_reduce={reduce_mode!r}")
    ds_kwargs = dict(
        offset_stride=offset_stride,
        include_past=include_past,
        apply_zscore=apply_zscore,
        apply_bin_center_shift=apply_bin_center,
        legal_levels=legal_levels,
        bin_center_reduce=reduce_mode,  # type: ignore[arg-type]
    )

    ds_train = UnivariateRealVsFakeDataset(
        y_true, fake, bundle.past, splits["train"], slice_len,
        seed=seed_base, max_examples=args.max_train_examples, **ds_kwargs,
    )
    ds_val = UnivariateRealVsFakeDataset(
        y_true, fake, bundle.past, splits["val"], slice_len,
        seed=seed_base + 1, max_examples=args.max_eval_examples, **ds_kwargs,
    )
    ds_test = UnivariateRealVsFakeDataset(
        y_true, fake, bundle.past, splits["test"], slice_len,
        seed=seed_base + 2, max_examples=args.max_eval_examples, **ds_kwargs,
    )

    generator = torch.Generator()
    generator.manual_seed(seed_base)
    train_loader = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"), generator=generator,
    )
    val_loader = DataLoader(
        ds_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        ds_test, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    seq_len = int(slice_len if not include_past else bundle.past.shape[-1] + slice_len)
    print(
        f"[disc-uni] {dataset}/{fake_source}/L{slice_len}: real-vs-fake univariate "
        f"candidate_only={not include_past} offset_stride={offset_stride} "
        f"offset_emb={use_offset_embedding} seq_len={seq_len} n_variates={n_variates} "
        f"n_train={len(ds_train)} n_val={len(ds_val)} n_test={len(ds_test)}",
        flush=True,
    )
    model = InvertedSliceDiscriminator(
        seq_len=seq_len,
        max_offset=max_offset,
        d_model=args.d_model,
        n_heads=args.n_heads,
        depth=args.depth,
        d_ff=args.d_ff,
        dropout=args.dropout,
        use_offset_embedding=use_offset_embedding,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val = float("inf")
    best_epoch = -1
    stale = 0
    progress = EvalProgress(f"disc-uni/{dataset}/{fake_source}/L{slice_len}", args.epochs)
    t0 = time.time()
    epoch = -1
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_count = 0
        for batch_idx, batch in enumerate(train_loader):
            x, offsets, labels = batch[0], batch[1], batch[2]
            x = x.to(device)
            offsets = offsets.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, offsets)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_loss += float(loss.item()) * int(labels.numel())
            train_count += int(labels.numel())
            if args.max_batches_per_epoch and batch_idx + 1 >= args.max_batches_per_epoch:
                break

        val_metrics = evaluate_classifier(model, val_loader, device)
        train_bce = train_loss / max(1, train_count)
        if val_metrics["disc_bce"] < best_val:
            best_val = val_metrics["disc_bce"]
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        progress.maybe_log(
            epoch + 1,
            extra=(
                f"train_bce={train_bce:.4f} val_bce={val_metrics['disc_bce']:.4f} "
                f"val_auc={val_metrics['disc_auroc']:.3f} "
                f"val_auc_win={val_metrics.get('disc_auroc_window', float('nan')):.3f} "
                f"elapsed={fmt_duration(time.time() - t0)}"
            ),
        )
        if stale >= args.patience:
            break

    progress.done(extra=f"best_epoch={best_epoch} best_val_bce={best_val:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate_classifier(model, test_loader, device)

    if bool(getattr(args, "save_classification_scores", False)):
        score_path = (
            args.output_dir / "scores" / f"{dataset}_{fake_source}_L{slice_len}_test_scores.npz"
        )
        score_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(score_path, **collect_classifier_scores(model, test_loader, device))
        print(f"[disc-uni] wrote classifier scores {score_path}", flush=True)
```

vs_gt sibling (binary vs snapped GT, no MMPD) — entry only:

```python
# temp/eval_univariate_patch_refine_vs_gt.py:1-80
#!/usr/bin/env python3
"""Non-ordinal h96 patch-refine univariate disc vs locally snapped GT.

Sibling of the live ordinal vs-MMPD evaluator. Loads coarse+patch_refine,
snaps GT to the unbounded per-window midpoint grid (not ordinal ladder),
trains binary-vs-GT plus an unblended non-overlap L8 patch disc. Does not
consume MMPD packs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.diffusion_tsf.pipeline.globals_bridge import patch_globals
from models.diffusion_tsf.train_multivariate_pipeline import (
    load_dataset,
    load_wrapped_guidance,
    resolve_pipeline_data_subset,
)
from utils.eval_discriminator_binary_vs_mmpd_univariate import train_classifier
from utils.disc_shared import (
    DEFAULT_DISC_OUTPUT,
    apply_smoke_defaults as apply_base_smoke_defaults,
    parse_args as parse_base_args,
    split_windows,
    write_json,
)
from utils.eval_mmpd_gaussian_anchor import (
    AnchorRun,
    load_tsf_pack_pool,
    parse_pack_splits,
    run_subset_id,
    run_test_stride,
    run_train_stride,
    run_variate_indices,
)
from utils.mmpd_eval_progress import EvalProgress, fmt_duration
from utils.staged_binary_forecast import (
    generate_staged_forecast,
    make_indices,
)
from utils.patch_refine_value_grid import (
    assert_on_patch_refine_grid,
    grid_coordinates,
    normalized_grid_step,
    snap_to_unbounded_patch_refine_grid,
    window_normalization_stats,
)
from utils.visualize_staged_eval_2d_preds import (
    _build_state,
    _load_stage_model,
    _resolve_guidance_ckpt,
)


DEFAULT_ANCHOR_CONFIG = "binary_patch_refine_lb336_hz96_full"
DEFAULT_BINARY_CONFIG = REPO_ROOT / "configs" / f"{DEFAULT_ANCHOR_CONFIG}.yaml"
DEFAULT_OUTPUT = DEFAULT_DISC_OUTPUT.parent / "disc-univariate-patch-refine-lb336-hz96-vs-gt"


def _unblended_nonoverlap_patch_batch(
    result: Dict[str, Any],
    target: torch.Tensor,
    past: torch.Tensor,
    config: Any,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, int]]:
```

## Later refactors (smells spotted while commenting)

Do not fix these in this pass unless you are explicitly cleaning — track for a dedicated PR.

### Bugs / landmines
- `train_multivariate_pipeline.py` `__main__` had been deleted (iTransformer scrub); restored so Slurm module entry works.
- `eval_mmpd_gaussian_anchor.py` Decoder-only edit had truncated `run_cmd` after building `cmd`; restored.
- `config.py` allows `representation_type=pdf` but decode hard-requires CDF.
- Misleading losses: `combined_mse_loss` is BCE; `emd_loss` / `guidance_loss` often always 0.

### Dead / unused
- `unwrap_model` identity; `MAX_SCALE_TUNING*` globals; `itrans_*` fields on `PipelineState`.
- `anchor_mse_proxy_lambda`, `emd_lambda`, `use_monotonicity_loss`, `guidance_penalty_weight`, `unified_time_axis`.
- `cond_chunks` guidance path (`guidance_placement` forced to `canvas`).
- DiT scale / cross-scale flags always False.
- `_value_to_rank_slow` in ordinal_window_norm.
- `DEFAULT_STAGED_CKPTS` / dual_scale fallback in `staged_binary_forecast`.
- `submit_mmpd.sh` still parses `--eval-existing-discriminator` then errors.
- Texture/RQA/signature helpers mostly unused under `metrics_profile=anchor-compat`.

### Redundant / over-compat
- `window_norm_center` duplicated in `diffusion_arch_config_dict`.
- Double `finish_pipeline_run` (orchestrator + cli finally).
- `honor_dataset_windows` unused kwarg; `phase.get` alias of `optional`.
- `getattr` over-compat for real dataclass fields (`globals_bridge`, diffusion_model heights).
- `reused_paths` still accepts `channel_dual` / `vertical_dual` donors.
- `forecast_pack_reduce.window_indices` alias; unknown YAML keys silently ignored in `mmpd_run_config`.
- Duplicated `train_classifier` (disc_shared vs uni evaluator); duplicated unblended patch batch in temp evals.
- Embedded ~300-line MMPD eval helper as a string — should be a real module.

### Fail-soft (prefer fail-fast later)
- `cli.py` swallows `get_dim_for_dataset` errors.
- Orchestrator `on_skip` failures non-fatal.
- Staged eval bare `except` on skip-check JSON.
- MMPD fail-soft `git pull` / viz exceptions.
- Ordinal disc snaps off-lattice binary by default; `assert-only` hard-fails.
- Disc y_true coord mismatch warn+continue unless ladder quantize on.

### Onboarding traps
- Non-smoke default `--configs` in `submit_binary.sh` is still `binary_dual_scale_staged`, not the live ordinal leaf.
- `slurm_worker.sh` checkout hint hard-codes an old branch name.
- Architecture debt: `PipelineState` + `globals_bridge` dual-write to module globals.
