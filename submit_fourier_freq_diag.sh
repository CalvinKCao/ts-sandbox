#!/bin/bash
# =============================================================================
# Quick 30-min resume diagnostic for the Fourier freq pipeline.
# Reuses existing full-run checkpoint dirs; only reruns fine finetune (1 trial,
# 12 epochs) with verbose per-epoch logging to catch hangs past epoch ~7.
#
# USAGE (Narval login node, from repo root):
#   ./submit_fourier_freq_diag.sh
#   ./submit_fourier_freq_diag.sh --datasets exchange_rate
#   ./submit_fourier_freq_diag.sh --datasets ETTh1,exchange_rate,weather
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="configs/binary_anchor_stationary_flat_subsets_grad_accum_150_fourier_freq_diag.yaml"
# Match stems from the full grid config, not this diag yaml basename.
CKPT_CONFIG="binary_anchor_stationary_flat_subsets_grad_accum_150_fourier_freq"
DATASETS=""
WALL_TIME="0:30:00"
GPU="a100"
RESUME=1
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --datasets) DATASETS="$2"; shift 2 ;;
        --time) WALL_TIME="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --no-resume) RESUME=0; shift ;;
        --parallel-optuna) EXTRA_ARGS+=(--parallel-optuna "$2"); shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$DATASETS" ]]; then
    DATASETS=$(
        for ds in ETTh1 ETTh2 ETTm1 ETTm2 exchange_rate weather; do
            if compgen -G "results/ckpts/*-${ds}-${CKPT_CONFIG}" > /dev/null; then
                echo "$ds"
            fi
        done | paste -sd, -
    )
    if [[ -z "$DATASETS" ]]; then
        echo "ERROR: no checkpoint dirs matching results/ckpts/*-{dataset}-${CKPT_CONFIG}" >&2
        exit 1
    fi
fi

echo "Diag datasets: $DATASETS"
echo "Wall time: $WALL_TIME | fine HP: 1 trial x 12 epochs"

for ds in ${DATASETS//,/ }; do
    CKPT=$(ls -dt results/ckpts/*-"${ds}"-"${CKPT_CONFIG}" 2>/dev/null | head -1)
    [[ -n "$CKPT" ]] || continue
    if [[ ! -f "$CKPT/${ds}/fine/best.pt" ]]; then
        rm -f "$CKPT/${ds}/fine/_diff_ft_trial_"*
        echo "cleared partial fine trials: $ds"
    fi
done

ARGS=(
    --configs "$CONFIG"
    --ckpt-config "$CKPT_CONFIG"
    --datasets "$DATASETS"
    --time "$WALL_TIME"
    --gpu "$GPU"
)

if [[ "$RESUME" -eq 1 ]]; then
    ARGS+=(--resume)
fi

./test_submit.sh "${ARGS[@]}" "${EXTRA_ARGS[@]}"
