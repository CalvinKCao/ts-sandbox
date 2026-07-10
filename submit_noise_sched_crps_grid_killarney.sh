#!/bin/bash
# CRPS-targeted noise-schedule g grid (336/720_uncompressed).
# Reuses synthetic pretrain + patch guidance; short coarse+fine (4 epochs) + staged_eval.
#
# Modes:
#   --mode extended     g∈{4,5,7,10} on ETTh1,traffic,exchange_rate,electricity
#   --mode electricity  full g∈{1,1.5,3,4,5,7,10} on electricity only
#   --mode seeds        g=1.0 seed replicates (s43,s44) on all four datasets
#   --mode all          electricity full + extended on other three + seeds  (default)
#
# USAGE (Killarney login, from $SCRATCH/ts-sandbox):
#   ./submit_noise_sched_crps_grid_killarney.sh --smoke-test --mode electricity
#   ./submit_noise_sched_crps_grid_killarney.sh --mode all
#   ./submit_noise_sched_crps_grid_killarney.sh --mode extended --datasets ETTh1,traffic
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="all"
WALL_TIME="3:00:00"
SMOKE=0
RESUME=0
DATASETS_OVERRIDE=""

G_BASE=(
    configs/binary_noise_sched_ablation_elec_unc_g1p0.yaml
    configs/binary_noise_sched_ablation_elec_unc_g1p5.yaml
    configs/binary_noise_sched_ablation_elec_unc_g3p0.yaml
)
G_EXT=(
    configs/binary_noise_sched_ablation_elec_unc_g4p0.yaml
    configs/binary_noise_sched_ablation_elec_unc_g5p0.yaml
    configs/binary_noise_sched_ablation_elec_unc_g7p0.yaml
    configs/binary_noise_sched_ablation_elec_unc_g10p0.yaml
)
G_FULL=("${G_BASE[@]}" "${G_EXT[@]}")
G_SEEDS=(
    configs/binary_noise_sched_ablation_elec_unc_g1p0_s43.yaml
    configs/binary_noise_sched_ablation_elec_unc_g1p0_s44.yaml
)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        --smoke-test|--smoke) SMOKE=1; shift ;;
        --time) WALL_TIME="$2"; shift 2 ;;
        --resume) RESUME=1; shift ;;
        --datasets) DATASETS_OVERRIDE="$2"; shift 2 ;;
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
    extended)
        DS="${DATASETS_OVERRIDE:-ETTh1,traffic,exchange_rate,electricity}"
        submit_one "$(IFS=,; echo "${G_EXT[*]}")" "$DS" "extended g=4/5/7/10"
        ;;
    electricity)
        DS="${DATASETS_OVERRIDE:-electricity}"
        submit_one "$(IFS=,; echo "${G_FULL[*]}")" "$DS" "electricity full g grid"
        ;;
    seeds)
        DS="${DATASETS_OVERRIDE:-ETTh1,traffic,exchange_rate,electricity}"
        submit_one "$(IFS=,; echo "${G_SEEDS[*]}")" "$DS" "g=1.0 seed replicates"
        ;;
    all)
        # 1) electricity missing baseline+extended
        submit_one "$(IFS=,; echo "${G_FULL[*]}")" \
            "${DATASETS_OVERRIDE:-electricity}" \
            "electricity full g grid"
        # 2) extended g on the three datasets that already have 1.0/1.5/3.0
        #    (also include electricity in extended only if override forces it — default skip dup)
        if [[ -n "$DATASETS_OVERRIDE" ]]; then
            submit_one "$(IFS=,; echo "${G_EXT[*]}")" "$DATASETS_OVERRIDE" "extended g"
        else
            submit_one "$(IFS=,; echo "${G_EXT[*]}")" \
                "ETTh1,traffic,exchange_rate" \
                "extended g on prior datasets"
        fi
        # 3) seed noise floor
        submit_one "$(IFS=,; echo "${G_SEEDS[*]}")" \
            "${DATASETS_OVERRIDE:-ETTh1,traffic,exchange_rate,electricity}" \
            "g=1.0 seed replicates"
        ;;
    *)
        echo "Unknown --mode $MODE (expected: extended|electricity|seeds|all)" >&2
        exit 1
        ;;
esac
