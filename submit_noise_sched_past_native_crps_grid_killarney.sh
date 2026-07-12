#!/bin/bash
# CRPS-targeted noise-schedule g calibration for past-native stride-2 compression.
# Geometry: representation_time_stride=2, past_cond_resize_to_horizon=false
#   (lookback encoded at 2 timesteps→1 col, same as horizon; no bilinear past resize).
#
# Datasets: ETTh1, traffic, exchange_rate, electricity
#
# Modes:
#   --mode grid      full g∈{1,1.5,3,4,5,6,7,8,9,10} + g=1 seed floor (default)
#   --mode coarse    g∈{1,1.5,3,4,5,7,10} only
#   --mode fine      g∈{6,8,9} refine neighbors
#   --mode seeds     g=1.0 s43/s44 only
#
# USAGE (Killarney login, from $SCRATCH/ts-sandbox):
#   ./submit_noise_sched_past_native_crps_grid_killarney.sh --smoke-test
#   ./submit_noise_sched_past_native_crps_grid_killarney.sh --mode grid
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="grid"
WALL_TIME="3:00:00"
SMOKE=0
RESUME=0
DATASETS="ETTh1,traffic,exchange_rate,electricity"

G_COARSE=(
    configs/binary_noise_sched_ablation_past_native_g1p0.yaml
    configs/binary_noise_sched_ablation_past_native_g1p5.yaml
    configs/binary_noise_sched_ablation_past_native_g3p0.yaml
    configs/binary_noise_sched_ablation_past_native_g4p0.yaml
    configs/binary_noise_sched_ablation_past_native_g5p0.yaml
    configs/binary_noise_sched_ablation_past_native_g7p0.yaml
    configs/binary_noise_sched_ablation_past_native_g10p0.yaml
)
G_FINE=(
    configs/binary_noise_sched_ablation_past_native_g6p0.yaml
    configs/binary_noise_sched_ablation_past_native_g8p0.yaml
    configs/binary_noise_sched_ablation_past_native_g9p0.yaml
)
G_FULL=("${G_COARSE[@]}" "${G_FINE[@]}")
G_SEEDS=(
    configs/binary_noise_sched_ablation_past_native_g1p0_s43.yaml
    configs/binary_noise_sched_ablation_past_native_g1p0_s44.yaml
)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        --smoke-test|--smoke) SMOKE=1; shift ;;
        --time) WALL_TIME="$2"; shift 2 ;;
        --resume) RESUME=1; shift ;;
        --datasets) DATASETS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

submit_one() {
    local config_csv="$1"
    local datasets="$2"
    local label="$3"
    local args=(
        --configs "$config_csv"
        --datasets "$datasets"
        --gpu l40s
        --time "$WALL_TIME"
        --wandb-project ts-sandbox-leaderboard
    )
    if [[ "$SMOKE" -eq 1 ]]; then
        args=(
            --configs "$config_csv"
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
    echo "=== submit [$label]: datasets=$datasets configs=$config_csv ==="
    ./test_submit.sh "${args[@]}"
}

case "$MODE" in
    coarse)
        submit_one "$(IFS=,; echo "${G_COARSE[*]}")" "$DATASETS" "past-native coarse g"
        ;;
    fine)
        submit_one "$(IFS=,; echo "${G_FINE[*]}")" "$DATASETS" "past-native fine g=6/8/9"
        ;;
    seeds)
        submit_one "$(IFS=,; echo "${G_SEEDS[*]}")" "$DATASETS" "past-native g=1 seed floor"
        ;;
    grid)
        submit_one "$(IFS=,; echo "${G_FULL[*]}")" "$DATASETS" "past-native full g grid"
        submit_one "$(IFS=,; echo "${G_SEEDS[*]}")" "$DATASETS" "past-native g=1 seed floor"
        ;;
    *)
        echo "Unknown --mode $MODE (expected: grid|coarse|fine|seeds)" >&2
        exit 1
        ;;
esac

echo ""
echo "After results land:"
echo "  python utils/analyze_noise_sched_crps_grid.py --out-dir reports/noise_sched_past_native_crps_grid \\"
echo "    --datasets ETTh1 traffic exchange_rate electricity"
echo "(analyzer stem regex must include past_native — see utils/analyze_noise_sched_crps_grid.py)"
