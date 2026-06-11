#!/bin/bash
# =============================================================================
# Submits a grid of experiments using the new YAML pipeline configs. *
#
# Each job gets isolated checkpoint/results dirs:
#   ./results/ckpts/MM-DD-<jobid>-<dataset>-<config>/
#
# USAGE (run from login node):
#   ./submit_grid.sh --configs configs/binary_anchor.yaml --datasets ETTh1,exchange_rate
#   ./submit_grid.sh --smoke
#   ./submit_grid.sh --resume --configs configs/binary_dual_scale.yaml --datasets ETTh1
#   ./submit_grid.sh --parallel-optuna 4 --configs configs/binary_dual_scale_staged.yaml --datasets ETTh1
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIGS=""
DATASETS="ETTh1,ETTh2,ETTm1,ETTm2,illness,exchange_rate,weather,electricity,traffic,PeMS,solar_Alabama,dalia"
SEEDS="42"
SMOKE=0
RESUME=0
CKPT_CONFIG=""
DEPENDENCY=""
WANDB_PROJECT="${WANDB_PROJECT:-ts-sandbox-binary-anchor-92d3}"
WALL_OVERRIDE=""
PARALLEL_OPTUNA=""
if [[ "$(hostname)" == *"narval"* ]]; then
    ACCOUNT="def-boyuwang"
else
    ACCOUNT="aip-boyuwang"
fi
GPU_TYPE="l40s"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --configs) CONFIGS="$2"; shift 2 ;;
        --datasets) DATASETS="$2"; shift 2 ;;
        --n-variates) N_VARIATES="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --smoke|--smoke-test) SMOKE=1; shift ;;
        --resume) RESUME=1; shift ;;
        --ckpt-config) CKPT_CONFIG="$2"; shift 2 ;;
        --dependency) DEPENDENCY="$2"; shift 2 ;;
        --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
        --time) WALL_OVERRIDE="$2"; shift 2 ;;
        --parallel-optuna) PARALLEL_OPTUNA="$2"; shift 2 ;;
        --gpu) GPU_TYPE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ "$SMOKE" -eq 1 ]]; then
    CONFIGS="${CONFIGS:-configs/smoke_test.yaml}"
    WALL_DEFAULT="0:30:00"
    MEM="24G"
    CPUS=4
    GPUS=1
    JOB_PREFIX="smoke"
else
    CONFIGS="${CONFIGS:-configs/binary_dual_scale_staged.yaml}"
    WALL_DEFAULT="3:00:00"
    MEM="60G"
    CPUS=8
    GPUS=1
    JOB_PREFIX="grid"
fi

if [[ -n "$PARALLEL_OPTUNA" ]]; then
    if [[ "$SMOKE" -eq 1 ]]; then
        echo "ERROR: --parallel-optuna is not supported with --smoke" >&2
        exit 1
    fi
    if ! [[ "$PARALLEL_OPTUNA" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: --parallel-optuna requires a positive integer (got: $PARALLEL_OPTUNA)" >&2
        exit 1
    fi
    GPUS="$PARALLEL_OPTUNA"
    MEM="$((PARALLEL_OPTUNA * 20))G"
    CPUS="$((PARALLEL_OPTUNA * 2))"
fi

IFS=',' read -ra CONF_ARR <<< "$CONFIGS"
IFS=',' read -ra DATA_ARR <<< "$DATASETS"
IFS=',' read -ra SEED_ARR <<< "$SEEDS"

expand_config_globs() {
    local expanded=() CFG matches
    for CFG in "${CONF_ARR[@]}"; do
        if [[ "$CFG" == *"*"* || "$CFG" == *"?"* || "$CFG" == *"["* ]]; then
            shopt -s nullglob
            matches=( "$SCRIPT_DIR"/$CFG )
            shopt -u nullglob
            if [[ ${#matches[@]} -eq 0 ]]; then
                echo "ERROR: no config yml files matched glob: $CFG" >&2
                exit 1
            fi
            for m in "${matches[@]}"; do
                expanded+=( "${m#$SCRIPT_DIR/}" )
            done
        else
            expanded+=( "$CFG" )
        fi
    done
    CONF_ARR=( "${expanded[@]}" )
}
expand_config_globs

USER=$(whoami)
STORE="${RESULTS_ROOT:-$SCRIPT_DIR/results}"
LOG_DIR="$STORE/logs"
CKPT_ROOT="$STORE/ckpts"
DATA_ROOT="$STORE/datasets"
mkdir -p "$LOG_DIR" "$CKPT_ROOT" "$DATA_ROOT"

# Resume: find newest ckpts/*-<dataset>-<config> dir; stem is reused for logs/ckpts/datasets.
pick_resume_stem() {
    local ds="$1" cfg="$2"
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

for CFG in "${CONF_ARR[@]}"; do
    if [[ ! -f "$SCRIPT_DIR/$CFG" ]]; then
        echo "ERROR: config yml file that was specified was not found: $SCRIPT_DIR/$CFG" >&2
        exit 1
    fi
done

echo "Submitting grid... (Storage: $STORE)"
[[ "$RESUME" -eq 1 ]] && echo "Resume: requires existing *-<dataset>-<config> checkpoint dir (newest mtime wins)."
[[ -n "$CKPT_CONFIG" ]] && echo "Checkpoint stem matcher: *-<dataset>-${CKPT_CONFIG}"
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
            CKPT_MATCH="${CKPT_CONFIG:-$CFG_NAME}"
            if [[ "$RESUME" -eq 1 ]]; then
                RUN_STEM=$(pick_resume_stem "$DS" "$CKPT_MATCH")
                if [[ -z "$RUN_STEM" ]]; then
                    echo "ERROR: --resume but no checkpoint dir matching ${CKPT_ROOT}/*-${DS}-${CKPT_MATCH}" >&2
                    exit 1
                fi
            fi

            if [[ -n "$RUN_STEM" ]]; then
                LOG_FILE="$LOG_DIR/${RUN_STEM}.log"
            else
                LOG_FILE="$LOG_DIR/${DATE_STR}-%j-${DS}-${CFG_NAME}.log"
            fi

            if [[ "$GPU_TYPE" == a100* || "$GPU_TYPE" == h100* ]]; then
                GPU_ARG="--gpus=${GPU_TYPE}:${GPUS}"
            else
                GPU_ARG="--gres=gpu:${GPU_TYPE}:${GPUS}"
            fi

            S_ARGS=(
                --parsable
                --job-name="$JOB_NAME"
                --account="$ACCOUNT"
                --time="$WALL"
                --nodes=1
                "$GPU_ARG"
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

            if [[ -n "${N_VARIATES:-}" ]]; then
                PY_ARGS+=(--n-variates "$N_VARIATES")
            fi

            if [[ -n "${WANDB_API_KEY:-}" ]]; then
                PY_ARGS+=(--wandb --wandb-project "$WANDB_PROJECT")
            fi

            if [[ "$SMOKE" -eq 1 ]]; then
                PY_ARGS+=(--smoke-test)
            fi

            if [[ "$RESUME" -eq 1 && -n "$RUN_STEM" ]]; then
                PY_ARGS+=(--resume)
            fi

            if [[ -n "$PARALLEL_OPTUNA" ]]; then
                PY_ARGS+=(--parallel-optuna-workers "$PARALLEL_OPTUNA")
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
