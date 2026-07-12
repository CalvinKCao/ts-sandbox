#!/bin/bash
# Full 20-epoch past-native stride-2 train+eval at CRPS-calibrated per-dataset g.
#
# From reports/noise_sched_past_native_crps_grid (4-epoch CRPS search):
#   ETTh1         → g=1.0  (identity; base past_native config)
#   traffic       → g=1.5
#   electricity   → g=4.0
#   exchange_rate → g=6.0
#
# USAGE (Killarney login, from $SCRATCH/ts-sandbox):
#   ./submit_past_native_crps_g_full_killarney.sh --smoke-test
#   ./submit_past_native_crps_g_full_killarney.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

WALL_TIME="1-00:00:00"
SMOKE=0
RESUME=0

CFG_G1="configs/binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_past_native.yaml"
CFG_G1P5="configs/binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_past_native_g1p5.yaml"
CFG_G4="configs/binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_past_native_g4p0.yaml"
CFG_G6="configs/binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_past_native_g6p0.yaml"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-test|--smoke) SMOKE=1; shift ;;
        --time) WALL_TIME="$2"; shift 2 ;;
        --resume) RESUME=1; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

submit_train() {
    local cfg="$1"
    local datasets="$2"
    local label="$3"
    local args=(
        --configs "$cfg"
        --datasets "$datasets"
        --gpu l40s
        --time "$WALL_TIME"
        --wandb-project ts-sandbox-leaderboard
    )
    if [[ "$SMOKE" -eq 1 ]]; then
        args=(
            --configs "$cfg"
            --datasets "$datasets"
            --gpu l40s
            --time "0:45:00"
            --smoke-test
            --wandb-project ts-sandbox-leaderboard
        )
    fi
    if [[ "$RESUME" -eq 1 ]]; then
        args+=(--resume)
    fi
    echo "=== train [$label]: datasets=$datasets config=$cfg ==="
    ./test_submit.sh "${args[@]}"
}

submit_train "$CFG_G1"   "ETTh1"         "past-native full @ g=1"
submit_train "$CFG_G1P5" "traffic"       "past-native full @ g=1.5"
submit_train "$CFG_G4"   "electricity"   "past-native full @ g=4"
submit_train "$CFG_G6"   "exchange_rate" "past-native full @ g=6"
