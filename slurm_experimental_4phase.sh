#!/bin/bash
# =============================================================================
# Slurm script for 4-phase experimental pipeline (A, B, A+B)
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ===========================================================================
# If NOT inside a Slurm job → submit ourselves
# ===========================================================================
if [ -z "${SLURM_JOB_ID:-}" ]; then
    mkdir -p "$SCRIPT_DIR/results_experimental/bootstrap"
    SB_OUT='results_experimental/bootstrap/%x-%j.out'
    SB_ERR='results_experimental/bootstrap/%x-%j.err'

    IS_SMOKE=0
    if [ "${1:-}" = "--smoke-test" ]; then
        IS_SMOKE=1
        shift
    fi

    if [ "$#" -gt 0 ]; then
        DATASETS=("$@")
    else
        DATASETS=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "weather" "exchange_rate")
    fi

    # The experiments requested: A, B, A+B (and baseline)
    SCENARIOS=("baseline" "A" "B" "A+B")

    for ds in "${DATASETS[@]}"; do
        DS_TAG="${ds//_/-}"
        WALLTIME="12:00:00"
        if [ "$IS_SMOKE" -eq 1 ]; then WALLTIME="00:15:00"; fi

        for scenario in "${SCENARIOS[@]}"; do
            # Format the scenario string for job name (e.g. A+B -> A-B)
            SAFE_SCENARIO="${scenario//+/_}"
            JOB_NAME="exp_${SAFE_SCENARIO}_${DS_TAG}"
            [ "$IS_SMOKE" -eq 1 ] && JOB_NAME="${JOB_NAME}_smoke"

            echo "Submitting $JOB_NAME ..."
            sbatch \
                --job-name="$JOB_NAME" \
                --account=aip-boyuwang \
                --time="$WALLTIME" \
                --nodes=1 \
                --gres=gpu:l40s:1 \
                --cpus-per-task=8 \
                --mem=50G \
                --chdir="$SCRIPT_DIR" \
                --output="$SB_OUT" \
                --error="$SB_ERR" \
                --export="ALL,SCENARIO=$scenario,DATASET=$ds,SMOKE=$IS_SMOKE" \
                "$SCRIPT_DIR/slurm_experimental_4phase.sh"
        done
    done
    echo "All jobs submitted!"
    exit 0
fi

# ===========================================================================
# Inside Slurm Job
# ===========================================================================
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
ALLIANCE_RUN_STEM="$(date +%m-%d)-${SLURM_JOB_ID}-${SLURM_JOB_NAME}"

RUN_RESULTS_ROOT="$SLURM_SUBMIT_DIR/results_experimental/runs/$ALLIANCE_RUN_STEM"
RUN_LOG_DIR="$RUN_RESULTS_ROOT/logs"
RUN_CKPT_DIR="$RUN_RESULTS_ROOT/ckpts"
RUN_DATA_DIR="$RUN_RESULTS_ROOT/datasets"
mkdir -p "$RUN_LOG_DIR" "$RUN_CKPT_DIR" "$RUN_DATA_DIR"

LOG_FILENAME="$(basename "$ALLIANCE_RUN_STEM").log"
ALLIANCE_JOB_LOG="$RUN_LOG_DIR/$LOG_FILENAME"

exec >>"$ALLIANCE_JOB_LOG" 2>&1

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Experiment: $SCENARIO"
echo "Dataset: $DATASET"
echo "Started: $(date '+%m-%d %H:%M:%S')"
echo "=========================================="

module purge || true
module load StdEnv/2023 python/3.11 cuda/12.2 cudnn/8.9

# Assuming fast venv is preferred
if [ -n "${SLURM_TMPDIR:-}" ]; then
    echo "Building fast venv on \$SLURM_TMPDIR..."
    python -m venv "$SLURM_TMPDIR/env"
    source "$SLURM_TMPDIR/env/bin/activate"
    pip install --no-index --upgrade pip
    pip install --no-index torch numpy scipy pandas scikit-learn wandb optuna tqdm matplotlib einops
    pip install reformer-pytorch --index-url https://pypi.org/simple
else
    source .venv/bin/activate
fi

SYNTH_CACHE_ROOT="$SLURM_SUBMIT_DIR/synth_data"
mkdir -p "$SYNTH_CACHE_ROOT"

SMOKE_FLAG=""
if [ "$SMOKE" -eq 1 ]; then
    SMOKE_FLAG="--smoke-test"
fi

# We use the unet backbone for these experiments as they modify visual channels and normalization
COMMON_ARGS=(
    "--dataset" "$DATASET"
    "--model-type" "unet"
    "--checkpoint-dir" "$RUN_CKPT_DIR"
    "--results-dir" "$RUN_DATA_DIR"
    "--synth-cache-dir" "$SYNTH_CACHE_ROOT"
    "--experiment" "$SCENARIO"
    "--subset-id" "exp_$SCENARIO"
    "--fresh"
)
if [ -n "$SMOKE_FLAG" ]; then
    COMMON_ARGS+=("$SMOKE_FLAG")
fi

echo "Running Phase 1 (Pretrain)..."
python models/diffusion_tsf/train_multivariate_pipeline.py \
    --mode pretrain \
    "${COMMON_ARGS[@]}"

echo "Running Phase 2 (Finetune)..."
python models/diffusion_tsf/train_multivariate_pipeline.py \
    --mode finetune \
    "${COMMON_ARGS[@]}"

echo "Pipeline complete."
