#!/bin/bash
# Gradient-accum sweep on flat-subset binary runs: reuse synthetic pretrain + iTrans,
# probe max micro-batch per GPU, then scale effective batch via
# training.diffusion_effective_batch_multiplier (1.25 / 1.5 / 2.0).
#
# USAGE (Killarney login node, from $SCRATCH/ts-sandbox):
#   ./submit_binary_flat_subsets_grad_accum_sweep.sh
#   ./submit_binary_flat_subsets_grad_accum_sweep.sh --smoke
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATASETS="ETTh1,ETTh2,exchange_rate,weather,electricity,traffic,solar_Alabama"
CONFIGS="configs/binary_anchor_stationary_flat_subsets_grad_accum_125.yaml,configs/binary_anchor_stationary_flat_subsets_grad_accum_150.yaml,configs/binary_anchor_stationary_flat_subsets_grad_accum_200.yaml"
WANDB_PROJECT="ts-sandbox-flat-subsets-grad-accum"
WALL_TIME="3:00:00"

if [[ "${1:-}" == "--smoke" ]]; then
    ./submit_grid.sh --smoke \
        --configs configs/binary_anchor_stationary_flat_subsets_grad_accum_150.yaml \
        --datasets ETTh1 \
        --wandb-project "${WANDB_PROJECT}-smoke"
    exit 0
fi

./submit_grid.sh \
    --configs "$CONFIGS" \
    --datasets "$DATASETS" \
    --wandb-project "$WANDB_PROJECT" \
    --time "$WALL_TIME"
