#!/bin/bash
# lb336/hz720 patch-decoder retrain with stride-2 repr subsample and fixed diffusion HP.
#
# USAGE (Killarney login node, from $SCRATCH/ts-sandbox):
#   ./submit_patch_decoder_lb336_hz720_fixed_killarney.sh --smoke-test --datasets ETTh1
#   ./submit_patch_decoder_lb336_hz720_fixed_killarney.sh --datasets ETTh1,electricity,exchange_rate
#   ./submit_patch_decoder_lb336_hz720_fixed_killarney.sh --resume
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="configs/binary_anchor_ar_patch_decoder_ctx_lb336_hz720_fixed.yaml"
DATASETS="ETTh1,electricity,exchange_rate"
WALL_TIME="24:00:00"
SMOKE=0
RESUME=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-test|--smoke) SMOKE=1; shift ;;
        --datasets) DATASETS="$2"; shift 2 ;;
        --time) WALL_TIME="$2"; shift 2 ;;
        --resume) RESUME=1; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

ARGS=(--configs "$CONFIG" --gpu l40s --time "$WALL_TIME" --wandb-project ts-sandbox-leaderboard)
if [[ "$SMOKE" -eq 1 ]]; then
    ARGS=(--smoke --configs "$CONFIG" --datasets ETTh1 --gpu l40s --time "0:45:00" --wandb-project ts-sandbox-leaderboard)
else
    ARGS+=(--datasets "$DATASETS")
fi

if [[ "$RESUME" -eq 1 ]]; then
    ARGS+=(--resume)
fi

exec ./test_submit.sh "${ARGS[@]}" "${EXTRA_ARGS[@]}"
