#!/bin/bash
# =============================================================================
# CFG scale ablation — eval-only on finished binary_dual_scale checkpoints.
#
# Reuses training artifacts from submit_grid + configs/binary_dual_scale.yaml
# (e.g. Killarney jobs 3828089–3828100). Does NOT retrain.
#
# USAGE (login node, $SCRATCH/ts-sandbox):
#   ./submit_cfg_ablation.sh --smoke-test
#   ./submit_cfg_ablation.sh
#   ./submit_cfg_ablation.sh --datasets ETTm2,ETTh1 --cfg-scales 2,4,7,10
#
# CFG 4/8/12 on finished binary_dual_scale ckpts (skip ETTm*, illness, electricity, solar):
#   ./submit_cfg_ablation.sh \
#     --datasets ETTh1,ETTh2,exchange_rate,weather,traffic,PeMS,dalia \
#     --cfg-scales 4,8,12
#
# Cancel mistaken full-retrain grid jobs first:
#   scancel -u $USER -n grid-ETTm2-binary_dual_scale_cfg   # etc.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-configs/binary_dual_scale_cfg_eval.yaml}"
CKPT_CONFIG="${CKPT_CONFIG:-binary_dual_scale}"
CFG_SCALES="${CFG_SCALES:-1,2,4,7,10,15}"
DATASETS="${DATASETS:-ETTh1,ETTh2,ETTm1,ETTm2,illness,exchange_rate,weather,electricity,traffic,PeMS,solar_Alabama,dalia}"
SEED=42
SMOKE=0
WANDB_PROJECT="${WANDB_PROJECT:-ts-sandbox-binary-anchor-92d3}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-test|--smoke) SMOKE=1; shift ;;
        --datasets) DATASETS="$2"; shift 2 ;;
        --cfg-scales) CFG_SCALES="$2"; shift 2 ;;
        --ckpt-config) CKPT_CONFIG="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

IFS=',' read -ra DATA_ARR <<< "$DATASETS"
IFS=',' read -ra SCALE_ARR <<< "$CFG_SCALES"

USER=$(whoami)
STORE="${RESULTS_ROOT:-$SCRIPT_DIR/results}"
if [[ -n "${SCRATCH:-}" ]]; then
    if [[ -d "$SCRATCH/${USER}/ts-sandbox/results" ]]; then
        STORE="$SCRATCH/${USER}/ts-sandbox/results"
    elif [[ -d "$SCRATCH/ts-sandbox/results" ]]; then
        STORE="$SCRATCH/ts-sandbox/results"
    fi
fi
LOG_DIR="$STORE/logs/cfg_ablation"
CKPT_ROOT="$STORE/ckpts"
DATA_ROOT="$STORE/datasets"
mkdir -p "$LOG_DIR"

pick_ckpt_stem() {
    local ds="$1"
    if [[ -f "$CKPT_ROOT/${ds}/metadata.json" ]]; then
        echo "$ds"
        return
    fi
    local best="" best_mtime=0 d m
    shopt -s nullglob
    for d in "$CKPT_ROOT"/*-"${ds}"-"${CKPT_CONFIG}"; do
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

if [[ "$SMOKE" -eq 1 ]]; then
    DATA_ARR=(ETTh1)
    SCALE_ARR=(2)
    WALL="0:45:00"
    MEM="24G"
    CPUS=4
    SMOKE_FLAG=(--smoke-test)
else
    WALL="2:30:00"
    MEM="60G"
    CPUS=8
    SMOKE_FLAG=()
fi

CFG_NAME=$(basename "$CONFIG" .yaml)
echo "CFG ablation (eval-only)"
echo "  config=$CONFIG  ckpt_config=$CKPT_CONFIG  scales=${SCALE_ARR[*]}"
echo "  storage=$STORE"
printf "%-10s %-12s %-8s %-6s %s\n" "JOB" "DATASET" "CFG" "SEED" "LOG"
echo "--------------------------------------------------------------------------------"

for DS in "${DATA_ARR[@]}"; do
    RUN_STEM=$(pick_ckpt_stem "$DS")
    if [[ -z "$RUN_STEM" ]]; then
        echo "ERROR: no checkpoint dir *-${DS}-${CKPT_CONFIG} under $CKPT_ROOT" >&2
        exit 1
    fi
    CKPT_DIR="$CKPT_ROOT/$RUN_STEM"
    if ! compgen -G "$CKPT_DIR"/*/best.pt >/dev/null; then
        echo "ERROR: missing */best.pt under $CKPT_DIR (need finished binary_dual_scale train)" >&2
        exit 1
    fi

    for SCALE in "${SCALE_ARR[@]}"; do
        RESULTS_STEM="${RUN_STEM}-cfg${SCALE}"
        RESULTS_DIR="$DATA_ROOT/$RESULTS_STEM"
        LOG_FILE="$LOG_DIR/${RESULTS_STEM}.log"
        JOB_NAME="cfg-${DS}-w${SCALE}"

        PY_ARGS=(
            --config "$CONFIG"
            --dataset "$DS"
            --seed "$SEED"
            --checkpoint-dir "$CKPT_DIR"
            --results-dir "$RESULTS_DIR"
            --cfg-scale "$SCALE"
        )
        if [[ "$(python3 -c "print(float('$SCALE') > 1.0)")" == "True" ]]; then
            PY_ARGS+=(--use-cfg-inference)
        fi
        if [[ -n "${WANDB_API_KEY:-}" ]]; then
            PY_ARGS+=(--wandb --wandb-project "$WANDB_PROJECT")
        fi
        PY_ARGS+=("${SMOKE_FLAG[@]}")

        EXPORT_LIST="GRID_STORE=${STORE},SLURM_SUBMIT_DIR=${SCRIPT_DIR}"
        [[ -n "${SCRATCH:-}" ]] && EXPORT_LIST+=",SCRATCH=${SCRATCH}"
        [[ -n "${WANDB_API_KEY:-}" ]] && EXPORT_LIST+=",WANDB_API_KEY=${WANDB_API_KEY}"

        JOB_ID=$(sbatch --parsable \
            --job-name="$JOB_NAME" \
            --account=aip-boyuwang \
            --time="$WALL" \
            --nodes=1 \
            --gres=gpu:l40s:1 \
            --cpus-per-task="$CPUS" \
            --mem="$MEM" \
            --output="$LOG_FILE" \
            --error="$LOG_FILE" \
            --mail-type=FAIL \
            --mail-user="${USER}@uwo.ca" \
            --export="$EXPORT_LIST" \
            "$SCRIPT_DIR/slurm_worker.sh" "${PY_ARGS[@]}")

        printf "%-10s %-12s %-8s %-6s %s\n" "$JOB_ID" "$DS" "$SCALE" "$SEED" "$LOG_FILE"
    done
done

echo "--------------------------------------------------------------------------------"
echo "Monitor: squeue -u $USER | grep '^cfg-'"
