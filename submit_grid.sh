#!/bin/bash
# =============================================================================
# Submits a grid of experiments using the new YAML pipeline configs.
#
# Each job gets isolated checkpoint/results dirs:
#   ./results/ckpts/MM-DD-<jobid>-<dataset>-<config>/
#
# USAGE (run from login node):
#   ./submit_grid.sh --configs configs/binary_anchor.yaml --datasets ETTh1,exchange_rate
#   ./submit_grid.sh --smoke
#   ./submit_grid.sh --resume --configs configs/binary_dual_scale.yaml --datasets ETTh1
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIGS=""
DATASETS="ETTh1,ETTh2,ETTm1,ETTm2,illness,exchange_rate,weather,electricity,traffic,PeMS,solar_Alabama,dalia"
SEEDS="42"
SMOKE=0
RESUME=0
DEPENDENCY=""
WANDB_PROJECT="${WANDB_PROJECT:-ts-sandbox-binary-anchor-92d3}"
WALL_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --configs) CONFIGS="$2"; shift 2 ;;
        --datasets) DATASETS="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --smoke|--smoke-test) SMOKE=1; shift ;;
        --resume) RESUME=1; shift ;;
        --dependency) DEPENDENCY="$2"; shift 2 ;;
        --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
        --time) WALL_OVERRIDE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ "$SMOKE" -eq 1 ]]; then
    CONFIGS="${CONFIGS:-configs/smoke_test.yaml}"
    WALL_DEFAULT="0:30:00"
    MEM="24G"
    CPUS=4
    JOB_PREFIX="smoke"
else
    CONFIGS="${CONFIGS:-configs/binary_anchor.yaml}"
    WALL_DEFAULT="5:00:00"
    MEM="60G"
    CPUS=8
    JOB_PREFIX="grid"
fi

IFS=',' read -ra CONF_ARR <<< "$CONFIGS"
IFS=',' read -ra DATA_ARR <<< "$DATASETS"
IFS=',' read -ra SEED_ARR <<< "$SEEDS"

USER=$(whoami)
STORE="${RESULTS_ROOT:-$SCRIPT_DIR/results}"
LOG_DIR="$STORE/logs"
CKPT_ROOT="$STORE/ckpts"
DATA_ROOT="$STORE/datasets"
mkdir -p "$LOG_DIR" "$CKPT_ROOT" "$DATA_ROOT"

pick_resume_stem() {
    local ds="$1" cfg="$2"
    # Legacy shared-dir layout (one dataset folder directly under ckpts/)
    if [[ -f "$CKPT_ROOT/${ds}/metadata.json" ]]; then
        echo "$ds"
        return
    fi
    local best="" best_mtime=0 d m
    shopt -s nullglob
    for d in "$CKPT_ROOT"/*-"${ds}"-"${cfg}"; do
        [[ -d "$d" ]] || continue
        m=$(stat -c %Y "$d" 2>/dev/null || echo 0)
        if [[ "$m" -gt "$best_mtime" ]]; then
            best_mtime="$m"
            best="$(basename "$d")"
        fi
    done
    shopt -u nullglob
    echo "$best"
}

echo "Submitting grid... (Storage: $STORE)"
[[ "$RESUME" -eq 1 ]] && echo "Resume: reusing newest *-<dataset>-<config> checkpoint dir when present."
printf "%-10s %-15s %-25s %-8s %s\n" "JOB ID" "DATASET" "CONFIG" "SEED" "LOG"
echo "--------------------------------------------------------------------------------"

for CFG in "${CONF_ARR[@]}"; do
    CFG_NAME=$(basename "$CFG" .yaml)
    for DS in "${DATA_ARR[@]}"; do
        for SD in "${SEED_ARR[@]}"; do

            JOB_NAME="${JOB_PREFIX}-${DS}-${CFG_NAME}"
            DATE_STR=$(date +%m-%d)

            if [[ -n "$WALL_OVERRIDE" ]]; then
                WALL="$WALL_OVERRIDE"
            else
                WALL="$WALL_DEFAULT"
            fi

            RUN_STEM=""
            if [[ "$RESUME" -eq 1 ]]; then
                RUN_STEM=$(pick_resume_stem "$DS" "$CFG_NAME")
                if [[ -z "$RUN_STEM" ]]; then
                    echo "WARN: no prior run for ${DS}/${CFG_NAME}; new isolated dir after submit." >&2
                fi
            fi

            if [[ -n "$RUN_STEM" ]]; then
                LOG_FILE="$LOG_DIR/${RUN_STEM}.log"
            else
                LOG_FILE="$LOG_DIR/${DATE_STR}-%j-${DS}-${CFG_NAME}.log"
            fi

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
                --export=ALL,GRID_DATE_STR="$DATE_STR",GRID_DATASET="$DS",GRID_CFG_NAME="$CFG_NAME",GRID_STORE="$STORE",GRID_RESUME="$RESUME",GRID_RUN_STEM="$RUN_STEM"
            )

            if [[ -n "$DEPENDENCY" ]]; then
                S_ARGS+=(--dependency="$DEPENDENCY")
            fi

            PY_ARGS=(
                --config "$CFG"
                --dataset "$DS"
                --seed "$SD"
            )

            if [[ -n "${WANDB_API_KEY:-}" ]]; then
                PY_ARGS+=(--wandb --wandb-project "$WANDB_PROJECT")
            fi

            if [[ "$SMOKE" -eq 1 ]]; then
                PY_ARGS+=(--smoke-test)
            fi

            if [[ "$RESUME" -eq 1 && -n "$RUN_STEM" ]]; then
                PY_ARGS+=(--resume)
            fi

            JOB_ID=$(sbatch "${S_ARGS[@]}" "$SCRIPT_DIR/slurm_worker.sh" "${PY_ARGS[@]}")

            if [[ -z "$RUN_STEM" ]]; then
                RUN_STEM="${DATE_STR}-${JOB_ID}-${DS}-${CFG_NAME}"
            fi
            ACTUAL_LOG="$LOG_DIR/${RUN_STEM}.log"
            if [[ "$LOG_FILE" == *'%j'* ]]; then
                ACTUAL_LOG="$LOG_DIR/${DATE_STR}-${JOB_ID}-${DS}-${CFG_NAME}.log"
            fi
            printf "%-10s %-15s %-25s %-8s %s\n" "$JOB_ID" "$DS" "$CFG_NAME" "$SD" "$ACTUAL_LOG"

        done
    done
done
echo "--------------------------------------------------------------------------------"
echo "Monitor with: squeue -u $USER"
