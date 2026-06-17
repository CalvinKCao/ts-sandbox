#!/bin/bash
# Resume unfinished binary grad_accum 1.5× LR-lo runs (5 datasets killed mid-finetune).
#
# USAGE (Killarney login node, from $SCRATCH/ts-sandbox):
#   ./submit_resume_binary_grad_accum_150_eval.sh
#   ./submit_resume_binary_grad_accum_150_eval.sh --time 6:00:00
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="configs/binary_anchor_stationary_flat_subsets_grad_accum_150_lr_lo.yaml"
DATASETS="ETTm1,ETTm2,PeMS,dalia,dynamic"
WALL="4:00:00"
JOB_IDS_OUT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --time) WALL="$2"; shift 2 ;;
        --datasets) DATASETS="$2"; shift 2 ;;
        --job-ids-out) JOB_IDS_OUT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

ARGS=(--resume --configs "$CONFIG" --datasets "$DATASETS" --time "$WALL")
if [[ -n "$JOB_IDS_OUT" ]]; then
    ARGS+=(--job-ids-out "$JOB_IDS_OUT")
fi

exec ./submit_grid.sh "${ARGS[@]}"
