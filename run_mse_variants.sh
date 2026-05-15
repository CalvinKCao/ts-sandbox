#!/bin/bash
# =============================================================================
# MSE loss weight ablation — iterates datasets x weight values.
#
# USAGE:
#   ./run_mse_variants.sh                 # Submit to Slurm (login node)
#   ./run_mse_variants.sh --smoke-test    # Submit smoke test
#
# This script handles its own Slurm submission if run from a login node,
# ensuring the job name and results directory are descriptive.
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ===========================================================================
# If NOT inside a Slurm job → submit ourselves with the right resources
# ===========================================================================
if [ -z "${SLURM_JOB_ID:-}" ]; then
    mkdir -p "$SCRIPT_DIR/results/bootstrap"
    SB_OUT='results/bootstrap/mse-ablation-%j.out'
    SB_ERR='results/bootstrap/mse-ablation-%j.err'

    IS_SMOKE=0
    WALLTIME="12:00:00"
    [ "${1:-}" = "--smoke-test" ] && IS_SMOKE=1 && WALLTIME="00:20:00"

    echo "Submitting MSE ablation sweep (Job Name: mse-ablation)..."
    sbatch \
        --job-name="mse-ablation" \
        --account=aip-boyuwang \
        --time="$WALLTIME" \
        --nodes=1 \
        --gres=gpu:l40s:1 \
        --cpus-per-task=8 \
        --mem=50G \
        --chdir="$SCRIPT_DIR" \
        --output="$SB_OUT" \
        --error="$SB_ERR" \
        --mail-type=END,FAIL \
        --mail-user=ccao87@uwo.ca \
        --export="ALL,IS_MSE_EXP=1" \
        "$SCRIPT_DIR/run_mse_variants.sh" "$@"
    exit 0
fi

# ===========================================================================
# Inside Slurm Job — Do the actual training
# ===========================================================================

# ---- Resolve project root ----
if [ "${TS_SANDBOX_PROJECT_ROOT_SUBMIT_DIR:-}" = "1" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
elif [ -n "${SCRATCH:-}" ] && [ -d "$SCRATCH/ts-sandbox" ]; then
    PROJECT_ROOT="$SCRATCH/ts-sandbox"
elif [ -d "$HOME/ts-sandbox" ]; then
    PROJECT_ROOT="$HOME/ts-sandbox"
else
    PROJECT_ROOT="$SCRIPT_DIR"
fi
export PROJECT_ROOT
cd "$PROJECT_ROOT"

# Source cluster preamble if available
if [ -n "${STORE:-}" ] && [ -f "$STORE/job_preamble.sh" ]; then
    echo "Sourcing $STORE/job_preamble.sh"
    source "$STORE/job_preamble.sh"
fi

SMOKE_FLAG=""
if [ "${1:-}" = "--smoke-test" ]; then
    SMOKE_FLAG="--smoke-test"
fi

# Setup results directory
ALLIANCE_RUN_STEM="$(date +%m-%d)-${SLURM_JOB_ID}-mse-ablation"
RUN_RESULTS_ROOT="$PROJECT_ROOT/results/$ALLIANCE_RUN_STEM"
RUN_LOG_DIR="$RUN_RESULTS_ROOT/logs"
RUN_CKPT_DIR="$RUN_RESULTS_ROOT/ckpts"
RUN_DATA_DIR="$RUN_RESULTS_ROOT/datasets"
mkdir -p "$RUN_LOG_DIR" "$RUN_CKPT_DIR" "$RUN_DATA_DIR"

# Redirect stdout/stderr to a log file inside the descriptive results folder
LOG_FILENAME="mse_ablation_sweep.log"
exec >>"$RUN_LOG_DIR/$LOG_FILENAME" 2>&1

echo "=========================================="
echo "MSE Ablation Sweep Started"
echo "Job ID: $SLURM_JOB_ID"
echo "Results: $RUN_RESULTS_ROOT"
echo "Started: $(date '+%m-%d %H:%M:%S')"
echo "=========================================="

PYTHON="python -u -m models.diffusion_tsf.train_multivariate_pipeline"

# ---- Experiment matrix ----
DATASETS=("ETTh1" "ETTm1" "exchange_rate")
WEIGHTS=(0.05 0.2 0.5)

declare -A DIM_MAP
DIM_MAP["ETTh1"]=7
DIM_MAP["ETTm1"]=7
DIM_MAP["exchange_rate"]=8

for ds in "${DATASETS[@]}"; do
    dim="${DIM_MAP[$ds]}"
    for w in "${WEIGHTS[@]}"; do
        tag="${ds}_mse${w}"
        
        # Set descriptive wandb name for each run in the matrix
        export WANDB_NAME="mse-ablation-${tag}"
        
        echo ""
        echo "============================================================"
        echo "  RUN: $tag  (mse_loss_weight=$w)"
        echo "============================================================"

        $PYTHON \
            --mode finetune \
            --dataset "$ds" \
            --n-variates "$dim" \
            --mse-loss-weight "$w" \
            --model-type dit \
            --guidance-penalty-weight 0.2 \
            --checkpoint-dir "$RUN_CKPT_DIR" \
            --results-dir "$RUN_DATA_DIR" \
            --subset-id "$tag" \
            --fresh \
            --wandb \
            $SMOKE_FLAG \
            || echo "[WARN] $tag failed with exit code $?"
    done
done

echo ""
echo "============================================================"
echo "  MSE variant sweep complete"
echo "============================================================"

