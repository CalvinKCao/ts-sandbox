#!/bin/bash
# Fourier flatline-blur fine target + patch-decoder guidance on Narval.
#
# USAGE (Narval login node, from $SCRATCH/ts-sandbox):
#   ./submit_fourier_flatline_blur_patch_decoder_narval.sh --smoke-test
#   ./submit_fourier_flatline_blur_patch_decoder_narval.sh
#   ./submit_fourier_flatline_blur_patch_decoder_narval.sh --datasets ETTh1,exchange_rate
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f "$SCRIPT_DIR/test_submit.sh" ]]; then
    echo "ERROR: missing test_submit.sh in $SCRIPT_DIR" >&2
    exit 1
fi

if [[ ! -f "$SCRIPT_DIR/submit_grid.sh" ]]; then
    ln -sf test_submit.sh submit_grid.sh
fi

CONFIG="configs/binary_anchor_stationary_flat_subsets_grad_accum_150_fourier_flatline_blur_patch_decoder_ctx.yaml"
DATASETS="ETTh1,ETTm1,exchange_rate,weather"
WALL_TIME="8:00:00"
SMOKE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-test|--smoke) SMOKE=1; shift ;;
        --datasets) DATASETS="$2"; shift 2 ;;
        --time) WALL_TIME="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

ARGS=(--configs "$CONFIG" --gpu a100 --time "$WALL_TIME")
if [[ "$SMOKE" -eq 1 ]]; then
    ARGS=(--smoke --configs "$CONFIG" --datasets ETTh1 --gpu a100 --time "0:45:00")
else
    ARGS+=(--datasets "$DATASETS")
fi

exec ./test_submit.sh "${ARGS[@]}"
