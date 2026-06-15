#!/bin/bash
# Window AR sweep: LB336/H96 + LB96/H720 at grad-accum 1.5×; LR Optuna (5 trials/stage).
# Reuses ema099 synthetic pretrain only; iTrans + diffusion finetune run fresh per config.
#
# USAGE (Killarney login node, from $SCRATCH/ts-sandbox):
#   ./submit_binary_anchor_ar_windows.sh
#   ./submit_binary_anchor_ar_windows.sh --smoke
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATASETS="ETTh1,ETTh2,exchange_rate,weather,electricity,traffic,solar_Alabama"
CONFIGS="configs/binary_anchor_ar_lb336_hz96_grad_accum_150.yaml,configs/binary_anchor_ar_lb96_hz720_grad_accum_150.yaml"
WALL_TIME="16:00:00"

if [[ "${1:-}" == "--smoke" ]]; then
    ./submit_grid.sh --smoke \
        --configs configs/binary_anchor_ar_lb96_hz720_grad_accum_150.yaml \
        --datasets ETTh1 \
        --time "$WALL_TIME"
    exit 0
fi

./submit_grid.sh \
    --configs "$CONFIGS" \
    --datasets "$DATASETS" \
    --time "$WALL_TIME"
