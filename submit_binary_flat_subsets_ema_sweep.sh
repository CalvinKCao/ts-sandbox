#!/bin/bash
# EMA decay sweep on flat-subset binary runs: reuse synthetic pretrain + iTrans,
# re-run diffusion coarse/fine/finer finetune with fixed diffusion_ema_decay.
#
# USAGE (Killarney login node, from $SCRATCH/ts-sandbox):
#   ./submit_binary_flat_subsets_ema_sweep.sh
#   ./submit_binary_flat_subsets_ema_sweep.sh --smoke
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATASETS="ETTh1,ETTh2,exchange_rate,weather,electricity,traffic,solar_Alabama"
CONFIGS="configs/binary_anchor_stationary_flat_subsets_ema_sweep_090.yaml,configs/binary_anchor_stationary_flat_subsets_ema_sweep_095.yaml,configs/binary_anchor_stationary_flat_subsets_ema_sweep_098.yaml,configs/binary_anchor_stationary_flat_subsets_ema_sweep_0995.yaml,configs/binary_anchor_stationary_flat_subsets_ema_sweep_0999.yaml"
WANDB_PROJECT="ts-sandbox-flat-subsets-ema-sweep"
WALL_TIME="3:00:00"

if [[ "${1:-}" == "--smoke" ]]; then
    ./submit_grid.sh --smoke \
        --configs configs/binary_anchor_stationary_flat_subsets_ema_sweep_098.yaml \
        --datasets ETTh1 \
        --wandb-project "${WANDB_PROJECT}-smoke"
    exit 0
fi

./submit_grid.sh \
    --configs "$CONFIGS" \
    --datasets "$DATASETS" \
    --wandb-project "$WANDB_PROJECT" \
    --time "$WALL_TIME"
