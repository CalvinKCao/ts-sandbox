#!/bin/bash
# =============================================================================
# Submit the Fourier frequency staged config across the core benchmark datasets.
#
# USAGE (Narval login node, from repo root):
#   ./submit_fourier_freq_grid.sh
#   ./submit_fourier_freq_grid.sh --datasets ETTh1,exchange_rate
#   ./submit_fourier_freq_grid.sh --resume
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="configs/binary_anchor_stationary_flat_subsets_grad_accum_150_fourier_freq.yaml"
DATASETS="ETTh1,ETTh2,ETTm1,ETTm2,exchange_rate,weather"
WALL_TIME="2:00:00"
GPU="a100"
RESUME=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --datasets) DATASETS="$2"; shift 2 ;;
        --time) WALL_TIME="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --resume) RESUME=1; shift ;;
        --parallel-optuna) EXTRA_ARGS+=(--parallel-optuna "$2"); shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

ARGS=(
    --configs "$CONFIG"
    --datasets "$DATASETS"
    --time "$WALL_TIME"
    --gpu "$GPU"
)

if [[ "$RESUME" -eq 1 ]]; then
    ARGS+=(--resume)
fi

./test_submit.sh "${ARGS[@]}" "${EXTRA_ARGS[@]}"
