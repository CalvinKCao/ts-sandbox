#!/bin/bash
# =============================================================================
# Patch (4×8) train + dpmpp vs ddim eval (CFG-ablation metrics, no CFG at inference).
#
# Train: ETTm1, ETTm2, dalia, electricity, exchange_rate, solar_Alabama, traffic, weather
# Eval:  same 8 datasets × {dpmpp, ddim} on finished *-binary_dual_scale_patch48 ckpts
#
# USAGE (login node, $SCRATCH/ts-sandbox):
#   ./submit_patch48_sampler_study.sh --train-only
#   ./submit_patch48_sampler_study.sh --eval-only    # after all train jobs finish
#   ./submit_patch48_sampler_study.sh --smoke-test
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DATASETS="${TRAIN_DATASETS:-ETTm1,ETTm2,dalia,electricity,exchange_rate,solar_Alabama,traffic,weather}"
CFG="${CFG:-configs/binary_dual_scale_patch48.yaml}"
CKPT_SUFFIX="binary_dual_scale_patch48"
RUN_STEM="$(date +%m-%d)-patch48-sampler"
MODE="both"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --train-only) MODE="train"; shift ;;
        --eval-only) MODE="eval"; shift ;;
        --smoke-test|--smoke) MODE="smoke"; shift ;;
        --run-stem) RUN_STEM="$2"; shift 2 ;;
        --datasets) TRAIN_DATASETS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

run_train() {
    local wall="10:00:00"
    local smoke_flag=()
    [[ "$1" == smoke ]] && smoke_flag=(--smoke-test) && wall="0:45:00"
    echo "=== Train patch48 (${TRAIN_DATASETS}) ==="
    "$SCRIPT_DIR/submit_grid.sh" \
        --configs "$CFG" \
        --datasets "$TRAIN_DATASETS" \
        --time "$wall" \
        "${smoke_flag[@]}"
}

run_eval() {
    local smoke_flag=()
    [[ "$1" == smoke ]] && smoke_flag=(--smoke-test)
    echo "=== Eval dpmpp vs ddim (ckpt *-${CKPT_SUFFIX}) ==="
    CKPT_SUFFIX="$CKPT_SUFFIX" \
    RUN_STEM="$RUN_STEM" \
    SAMPLERS="${SAMPLERS:-dpmpp,ddim}" \
    DATASETS="$TRAIN_DATASETS" \
    "$SCRIPT_DIR/submit_sampler_ablation.sh" \
        --ckpt-suffix "$CKPT_SUFFIX" \
        --run-stem "$RUN_STEM" \
        "${smoke_flag[@]}"
}

case "$MODE" in
    train) run_train ;;
    eval) run_eval ;;
    smoke)
        TRAIN_DATASETS="ETTm1"
        run_train smoke
        echo ""
        echo "After smoke train completes, run:"
        echo "  RUN_STEM=$RUN_STEM TRAIN_DATASETS=ETTm1 ./submit_patch48_sampler_study.sh --eval-only --smoke-test"
        ;;
    both)
        run_train
        echo ""
        echo "After ALL train jobs finish:"
        echo "  RUN_STEM=$RUN_STEM ./submit_patch48_sampler_study.sh --eval-only"
        ;;
esac
