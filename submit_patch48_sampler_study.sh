#!/bin/bash
# =============================================================================
# Patch 4×4 train + dpmpp vs ddim eval (CFG-ablation metrics, no CFG at inference).
#
# Train: configs/binary_dual_scale_patch48.yaml (NO pipeline eval phase).
# Eval:  submit_sampler_ablation.sh / submit_patch48_eval_redo.sh
#        → eval_mmpd_gaussian_anchor.py (same as submit_cfg_ablation.sh).
#
# USAGE (login node, $SCRATCH/ts-sandbox):
#   ./submit_patch48_sampler_study.sh --train-only
#   ./submit_patch48_eval_redo.sh              # after train (3844450–3844457 ckpts)
#   ./submit_patch48_sampler_study.sh --eval-only
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
        --eval-redo) MODE="eval"; shift ;;
        --smoke-test|--smoke) MODE="smoke"; shift ;;
        --run-stem) RUN_STEM="$2"; shift 2 ;;
        --datasets) TRAIN_DATASETS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

run_train() {
    local wall="8:00:00"
    local smoke_flag=()
    if [[ "${1:-}" == smoke ]]; then
        smoke_flag=(--smoke-test)
        wall="0:45:00"
    fi
    echo "=== Train patch48 (${TRAIN_DATASETS}) — no pipeline eval ==="
    "$SCRIPT_DIR/submit_grid.sh" \
        --configs "$CFG" \
        --datasets "$TRAIN_DATASETS" \
        --time "$wall" \
        "${smoke_flag[@]}"
}

run_eval() {
    local smoke_flag=()
    if [[ "${1:-}" == smoke ]]; then
        smoke_flag=(--smoke-test)
        export CKPT_STEM_PREFIX=""
    else
        export CKPT_STEM_PREFIX="${CKPT_STEM_PREFIX:-06-02-384445}"
    fi
    echo "=== Eval (CFG-ablation path, ckpt *-${CKPT_SUFFIX}) ==="
    export CKPT_SUFFIX RUN_STEM
    export DATASETS="$TRAIN_DATASETS"
    export SAMPLERS="${SAMPLERS:-dpmpp,ddim}"
    "$SCRIPT_DIR/submit_sampler_ablation.sh" \
        --ckpt-suffix "$CKPT_SUFFIX" \
        --ckpt-stem-prefix "$CKPT_STEM_PREFIX" \
        --run-stem "$RUN_STEM" \
        --datasets "$TRAIN_DATASETS" \
        --samplers "$SAMPLERS" \
        "${smoke_flag[@]}"
}

case "$MODE" in
    train) run_train ;;
    eval) run_eval ;;
    smoke)
        TRAIN_DATASETS="ETTm1"
        run_train smoke
        echo ""
        echo "After smoke train completes:"
        echo "  TRAIN_DATASETS=ETTm1 RUN_STEM=$RUN_STEM ./submit_patch48_sampler_study.sh --eval-only --smoke-test"
        ;;
    both)
        run_train
        echo ""
        echo "After ALL train jobs finish (ckpts 06-02-384445*-*):"
        echo "  ./submit_patch48_eval_redo.sh"
        echo "  # or: RUN_STEM=$RUN_STEM ./submit_patch48_sampler_study.sh --eval-only"
        ;;
esac
