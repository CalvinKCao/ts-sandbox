#!/bin/bash
# =============================================================================
# Submits a grid of experiments using the new YAML pipeline configs.
#
# USAGE (run from login node):
#   ./submit_grid.sh --configs configs/binary_anchor.yaml --datasets ETTh1,exchange_rate
#   ./submit_grid.sh --smoke  # runs configs/smoke_test.yaml
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIGS=""
DATASETS="ETTh1,ETTh2,ETTm1,ETTm2,illness,exchange_rate,weather,electricity,traffic,PeMS,solar_Alabama,dalia"
SEEDS="42"
SMOKE=0
DEPENDENCY=""
WANDB_PROJECT="${WANDB_PROJECT:-ts-sandbox-binary-anchor-92d3}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --configs) CONFIGS="$2"; shift 2 ;;
        --datasets) DATASETS="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --smoke|--smoke-test) SMOKE=1; shift ;;
        --dependency) DEPENDENCY="$2"; shift 2 ;;
        --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
        -*) echo "Unknown flag: $1" >&2; exit 1 ;;
        *) CONFIGS="$1"; shift ;;
    esac
done

if [[ "$SMOKE" -eq 1 ]]; then
    CONFIGS="${CONFIGS:-configs/smoke_test.yaml}"
    WALL="0:30:00"
    MEM="24G"
    CPUS=4
    JOB_PREFIX="smoke"
else
    CONFIGS="${CONFIGS:-configs/binary_anchor.yaml}"
    WALL="1-00:00:00"
    MEM="60G"
    CPUS=8
    JOB_PREFIX="grid"
fi

IFS=',' read -ra CONF_ARR <<< "$CONFIGS"
IFS=',' read -ra DATA_ARR <<< "$DATASETS"
IFS=',' read -ra SEED_ARR <<< "$SEEDS"

# Setup scratch results paths
USER=$(whoami)
STORE="/scratch/$USER/results"
LOG_DIR="$STORE/logs"
CKPT_DIR="$STORE/ckpts"
DATA_DIR="$STORE/datasets"
mkdir -p "$LOG_DIR" "$CKPT_DIR" "$DATA_DIR"

echo "Submitting grid... (Storage: $STORE)"
printf "%-10s %-15s %-25s %-8s %s\n" "JOB ID" "DATASET" "CONFIG" "SEED" "LOG"
echo "--------------------------------------------------------------------------------"

for CFG in "${CONF_ARR[@]}"; do
    CFG_NAME=$(basename "$CFG" .yaml)
    for DS in "${DATA_ARR[@]}"; do
        for SD in "${SEED_ARR[@]}"; do
            
            JOB_NAME="${JOB_PREFIX}-${DS}-${CFG_NAME}"
            DATE_STR=$(date +%m-%d)
            
            # Submitting the job using --parsable to capture JOB_ID early for log name formatting
            # Note: We append %j to log file, Slurm replaces it. But we don't know the exact %j before submission unless we capture it.
            # We'll use a wrapper script approach or just let Slurm fill %j.
            LOG_FILE="$LOG_DIR/${DATE_STR}-%j-${DS}-${CFG_NAME}.log"
            
            S_ARGS=(
                --parsable
                --job-name="$JOB_NAME"
                --account=aip-boyuwang
                --time="$WALL"
                --nodes=1
                --gres=gpu:l40s:1
                --cpus-per-task="$CPUS"
                --mem="$MEM"
                --output="$LOG_FILE"
                --error="$LOG_FILE"
                --mail-type=FAIL
                --mail-user="${USER}@uwo.ca"
            )
            
            if [[ -n "$DEPENDENCY" ]]; then
                S_ARGS+=(--dependency="$DEPENDENCY")
            fi
            
            # The args passed to the python script
            PY_ARGS=(
                --config "$CFG"
                --checkpoint-dir "$CKPT_DIR"
                --results-dir "$DATA_DIR"
                --dataset "$DS"
                --seed "$SD"
            )
            
            if [[ -n "${WANDB_API_KEY:-}" ]]; then
                PY_ARGS+=(--wandb --wandb-project "$WANDB_PROJECT")
            fi
            
            if [[ "$SMOKE" -eq 1 ]]; then
                PY_ARGS+=(--smoke-test)
            fi

            JOB_ID=$(sbatch "${S_ARGS[@]}" slurm_worker.sh "${PY_ARGS[@]}")
            
            ACTUAL_LOG="$LOG_DIR/${DATE_STR}-${JOB_ID}-${DS}-${CFG_NAME}.log"
            printf "%-10s %-15s %-25s %-8s %s\n" "$JOB_ID" "$DS" "$CFG_NAME" "$SD" "$ACTUAL_LOG"
            
        done
    done
done
echo "--------------------------------------------------------------------------------"
echo "Monitor with: squeue -u $USER"
