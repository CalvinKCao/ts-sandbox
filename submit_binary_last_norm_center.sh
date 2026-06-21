#!/bin/bash
# Window-norm center ablation: subtract last lookback value (not mean) before /std.
# Off grad_accum_150_lr_lo baseline; exchange_rate + weather.
#
# USAGE (Killarney login node, from $SCRATCH/ts-sandbox):
#   ./submit_binary_last_norm_center.sh
#   ./submit_binary_last_norm_center.sh --smoke-test
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="configs/binary_anchor_stationary_flat_subsets_grad_accum_150_lr_lo_last_norm.yaml"
DATASETS="exchange_rate,weather"
WALL_TIME="8:00:00"
EXCLUDE_NODES="kn001"
SMOKE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-test|--smoke) SMOKE=1; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

ARGS=(--configs "$CONFIG" --datasets "$DATASETS" --time "$WALL_TIME" --exclude-nodes "$EXCLUDE_NODES")
if [[ "$SMOKE" -eq 1 ]]; then
    ARGS=(--smoke --configs "$CONFIG" --datasets ETTh1 --time "0:45:00" --exclude-nodes "$EXCLUDE_NODES")
fi

./submit_grid.sh "${ARGS[@]}"
