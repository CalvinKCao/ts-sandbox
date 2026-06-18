#!/bin/bash
# Flatline-to-pulse + strong-trend ablations off grad_accum_150_lr_lo baseline.
#
# Problem 1 (flatline→pulse): ETTh1, ETTm1, electricity
#   1A: use_window_normalization=false
#   1B: use_guidance_channel=true (reuses hi_guidance pretrain/iTrans)
#
# Problem 2 (upward bias on trends): exchange_rate, weather (flat subsets as usual)
#   2A: disable_cross_attention=true
#   2B: max_scale_tuning=true
#
# USAGE (Killarney login node, from $SCRATCH/ts-sandbox):
#   ./submit_flatline_pulse_trend_ablations.sh
#   ./submit_flatline_pulse_trend_ablations.sh --smoke
#   ./submit_flatline_pulse_trend_ablations.sh --only 1a
#   ./submit_flatline_pulse_trend_ablations.sh --only 1b,2a
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

FLATLINE_DATASETS="ETTh1,ETTm1,electricity"
TREND_DATASETS="exchange_rate,weather"
WALL_TIME="8:00:00"
EXCLUDE_NODES="kn001"
ONLY="all"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke|--smoke-test) SMOKE=1; shift ;;
        --only) ONLY="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

SMOKE="${SMOKE:-0}"
want() {
    local key="$1"
    [[ "$ONLY" == "all" ]] && return 0
    IFS=',' read -ra PARTS <<< "$ONLY"
    for p in "${PARTS[@]}"; do
        [[ "$p" == "$key" ]] && return 0
    done
    return 1
}

run_grid() {
    local cfg="$1" datasets="$2"
    local args=(--configs "$cfg" --datasets "$datasets" --time "$WALL_TIME" --exclude-nodes "$EXCLUDE_NODES")
    if [[ "$SMOKE" -eq 1 ]]; then
        args=(--smoke --configs "$cfg" --datasets ETTh1 --time "0:45:00" --exclude-nodes "$EXCLUDE_NODES")
    fi
    ./submit_grid.sh "${args[@]}"
}

if want 1a; then
    echo "=== 1A: no window normalization (${FLATLINE_DATASETS}) ==="
    run_grid configs/binary_anchor_stationary_flat_subsets_grad_accum_150_lr_lo_no_win_norm.yaml "$FLATLINE_DATASETS"
fi

if want 1b; then
    echo "=== 1B: guidance channel (${FLATLINE_DATASETS}) ==="
    run_grid configs/binary_anchor_stationary_flat_subsets_grad_accum_150_lr_lo_guidance.yaml "$FLATLINE_DATASETS"
fi

if want 2a; then
    echo "=== 2A: disable cross-attention (${TREND_DATASETS}) ==="
    run_grid configs/binary_anchor_stationary_flat_subsets_grad_accum_150_lr_lo_no_cross_attn.yaml "$TREND_DATASETS"
fi

if want 2b; then
    echo "=== 2B: max_scale tuning (${TREND_DATASETS}) ==="
    run_grid configs/binary_anchor_stationary_flat_subsets_grad_accum_150_lr_lo_max_scale_tune.yaml "$TREND_DATASETS"
fi
