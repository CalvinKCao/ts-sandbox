#!/bin/bash
# Grad-accum flat-subset sweep: reuse pretrain + iTrans, Optuna LR per diffusion stage
# (log-uniform 3e-6–2e-4), effective batch = probed_max × multiplier (1.5 / 2.0).
#
# USAGE (Killarney login node, from $SCRATCH/ts-sandbox):
#   ./submit_binary_flat_subsets_grad_accum_sweep.sh
#   ./submit_binary_flat_subsets_grad_accum_sweep.sh --smoke
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATASETS="ETTh1,ETTh2,exchange_rate,weather,electricity,traffic,solar_Alabama"
CONFIGS="configs/binary_anchor_stationary_flat_subsets_grad_accum_150.yaml,configs/binary_anchor_stationary_flat_subsets_grad_accum_200.yaml"
WANDB_PROJECT="ts-sandbox-flat-subsets-grad-accum-lr-tune"
WALL_TIME="8:00:00"

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
