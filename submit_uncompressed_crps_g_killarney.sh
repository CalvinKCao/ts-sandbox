#!/bin/bash
# Full 336/720_uncompressed train+eval at CRPS-calibrated per-dataset g.
# Also submits endpoint noise-schedule diagnosis for datasets not yet diagnosed
# (excludes illness + dalia).
#
# Recs from reports/noise_sched_crps_grid (seed=42 coarse grid; refine/confirm
# may still be pending on cluster):
#   ETTh1, exchange_rate → g=7
#   traffic, electricity → g=3
#
# USAGE (Killarney login, from $SCRATCH/ts-sandbox):
#   ./submit_uncompressed_crps_g_killarney.sh --smoke-test
#   ./submit_uncompressed_crps_g_killarney.sh --mode train
#   ./submit_uncompressed_crps_g_killarney.sh --mode diagnose
#   ./submit_uncompressed_crps_g_killarney.sh --mode all
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="all"
WALL_TIME="1-00:00:00"
SMOKE=0
RESUME=0

CFG_G3="configs/binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_uncompressed_g3p0.yaml"
CFG_G7="configs/binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_uncompressed_g7p0.yaml"

# Datasets with CRPS-calibrated g (only these get full train at optimal g)
DS_G7="ETTh1,exchange_rate"
DS_G3="traffic,electricity"

# Endpoint diagnosis: repo datasets minus illness/dalia and already-diagnosed
# (ETTh1, weather, electricity, exchange_rate, traffic).
DIAG_REMAINING="ETTh2,ETTm1,ETTm2,PeMS,solar_Alabama"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        --smoke-test|--smoke) SMOKE=1; shift ;;
        --time) WALL_TIME="$2"; shift 2 ;;
        --resume) RESUME=1; shift ;;
        --diag-datasets) DIAG_REMAINING="$2"; shift 2 ;;
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

submit_diagnose() {
    local ds_csv="$1"
    echo "=== endpoint diagnosis for: $ds_csv ==="
    IFS=',' read -ra ARR <<< "$ds_csv"
    for d in "${ARR[@]}"; do
        [[ "$d" == "illness" || "$d" == "dalia" ]] && continue
        ./submit_diagnose_noise_schedule_killarney.sh --datasets "$d" --time "0:15:00"
    done
}

case "$MODE" in
    train)
        submit_train "$CFG_G7" "$DS_G7" "full unc @ g=7"
        submit_train "$CFG_G3" "$DS_G3" "full unc @ g=3"
        ;;
    diagnose)
        submit_diagnose "$DIAG_REMAINING"
        ;;
    all)
        submit_train "$CFG_G7" "$DS_G7" "full unc @ g=7"
        submit_train "$CFG_G3" "$DS_G3" "full unc @ g=3"
        submit_diagnose "$DIAG_REMAINING"
        ;;
    *)
        echo "Unknown --mode $MODE (expected: train|diagnose|all)" >&2
        exit 1
        ;;
esac
