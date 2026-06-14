#!/bin/bash
# Optuna max_scale sweep on sweep_baseline-fixed HPs (lr_only search_space).
#
# USAGE (Killarney login node, from $SCRATCH/ts-sandbox):
#   ./submit_hp_max_scale_tuning.sh
#   ./submit_hp_max_scale_tuning.sh --smoke
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="configs/tuning_sweep/hp_max_scale_tuning.yaml"
DATASETS="ETTh1,ETTm1,exchange_rate,weather"

if [[ "${1:-}" == "--smoke" ]]; then
    ./submit_grid.sh --smoke \
        --configs "$CONFIG" \
        --datasets ETTh1
    exit 0
fi

./submit_grid.sh \
    --configs "$CONFIG" \
    --datasets "$DATASETS"
