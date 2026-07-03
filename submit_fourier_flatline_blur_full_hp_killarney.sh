#!/bin/bash
# Fourier flatline-blur + full coarse/fine HP (effective batch 0.25x–4x) on Killarney (L40S).
#
# USAGE (Killarney login node, from $SCRATCH/ts-sandbox):
#   ./submit_fourier_flatline_blur_full_hp_killarney.sh --smoke-test
#   ./submit_fourier_flatline_blur_full_hp_killarney.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="configs/binary_anchor_stationary_flat_subsets_grad_accum_150_fourier_flatline_blur_full_hp.yaml"
DATASETS="ETTh1,ETTh2,ETTm1,electricity,exchange_rate"
WALL_TIME="8:00:00"
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

ARGS=(--configs "$CONFIG" --gpu l40s --time "$WALL_TIME")
if [[ "$SMOKE" -eq 1 ]]; then
    ARGS=(--smoke --configs "$CONFIG" --datasets ETTh1 --gpu l40s --time "0:45:00")
else
    ARGS+=(--datasets "$DATASETS")
fi

if [[ "$RESUME" -eq 1 ]]; then
    ARGS+=(--resume)
fi

exec ./test_submit.sh "${ARGS[@]}" "${EXTRA_ARGS[@]}"
