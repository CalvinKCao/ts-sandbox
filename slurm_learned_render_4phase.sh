#!/bin/bash
# =============================================================================
# Slurm: 4-phase pipeline for learned-render hybrid (ETTh1 + exchange_rate)
# L=96, H=96, image_height=64, K=8 overlap
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "${SLURM_JOB_ID:-}" ]; then
    mkdir -p "$SCRIPT_DIR/results_learned_render/bootstrap"
    SB_OUT='results_learned_render/bootstrap/%x-%j.out'
    SB_ERR='results_learned_render/bootstrap/%x-%j.err'

    IS_SMOKE=0
    while [ $# -gt 0 ]; do
        case "${1:-}" in
            --smoke-test)
                IS_SMOKE=1
                shift
                ;;
            *)
                break
                ;;
        esac
    done

    if [ "$#" -gt 0 ]; then
        DATASETS=("$@")
    else
        DATASETS=("ETTh1" "exchange_rate")
    fi

    for ds in "${DATASETS[@]}"; do
        DS_TAG="${ds//_/-}"
        WALLTIME="12:00:00"
        if [ "$IS_SMOKE" -eq 1 ]; then WALLTIME="00:15:00"; fi

        JOB_NAME="learned_render_${DS_TAG}"
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
            --export="ALL,DATASET=$ds,SMOKE=$IS_SMOKE" \
            "$SCRIPT_DIR/slurm_learned_render_4phase.sh"
    done
    echo "All learned-render jobs submitted!"
    exit 0
fi

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
ALLIANCE_RUN_STEM="$(date +%m-%d)-${SLURM_JOB_ID}-${SLURM_JOB_NAME}"

RUN_RESULTS_ROOT="$SLURM_SUBMIT_DIR/results_learned_render/runs/$ALLIANCE_RUN_STEM"
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
echo "Experiment: learned_render (hybrid 1D->2D->1D)"
echo "Dataset: $DATASET"
echo "Geometry: L=96 H=96 K=8 image_height=64"
echo "Started: $(date '+%m-%d %H:%M:%S')"
echo "=========================================="

module purge || true
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9

if [ -n "${SLURM_TMPDIR:-}" ]; then
    echo "Building fast venv on \$SLURM_TMPDIR..."
    python3 -m venv "$SLURM_TMPDIR/env"
    source "$SLURM_TMPDIR/env/bin/activate"
    pip install --no-index --upgrade pip
    pip install --no-index torch numpy scipy pandas scikit-learn wandb optuna tqdm matplotlib einops
    pip install reformer-pytorch --index-url https://pypi.org/simple
    pip install -r "$SLURM_SUBMIT_DIR/requirements.txt" || true
else
    source "$SLURM_SUBMIT_DIR/.venv/bin/activate"
fi

SYNTH_CACHE_ROOT="$SLURM_SUBMIT_DIR/synth_data"
mkdir -p "$SYNTH_CACHE_ROOT"

SMOKE_FLAG=""
if [ "$SMOKE" -eq 1 ]; then
    SMOKE_FLAG="--smoke-test"
fi

COMMON_ARGS=(
    "--dataset" "$DATASET"
    "--model-type" "unet"
    "--checkpoint-dir" "$RUN_CKPT_DIR"
    "--results-dir" "$RUN_DATA_DIR"
    "--synth-cache-dir" "$SYNTH_CACHE_ROOT"
    "--experiment" "learned_render"
    "--subset-id" "exp_learned_render"
    "--lookback-length" "96"
    "--forecast-length" "96"
    "--image-height" "64"
    "--fresh"
)
if [ -n "$SMOKE_FLAG" ]; then
    COMMON_ARGS+=("$SMOKE_FLAG")
fi

TARGET_DIM=7
if [ "$DATASET" = "exchange_rate" ]; then TARGET_DIM=8; fi

echo "Running Phase 1 (Pretrain)..."
python3 -m models.diffusion_tsf.train_multivariate_pipeline \
    --mode pretrain \
    --n-variates "$TARGET_DIM" \
    "${COMMON_ARGS[@]}"

echo "Running Phase 2 (Finetune)..."
python3 -m models.diffusion_tsf.train_multivariate_pipeline \
    --mode finetune \
    --n-variates "$TARGET_DIM" \
    "${COMMON_ARGS[@]}"

echo "Pipeline complete."
