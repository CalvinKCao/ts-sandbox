#!/bin/bash
# Electricity 336/720_uncompressed noise-schedule ablation:
# three short coarse+fine retrains (g=1.0 / 1.5 / 3.0), reuse synthetic + patch guidance.
#
# USAGE (Killarney login, from $SCRATCH/ts-sandbox):
#   ./submit_noise_sched_ablation_elec_unc_killarney.sh
#   ./submit_noise_sched_ablation_elec_unc_killarney.sh --time 4:00:00
#   ./submit_noise_sched_ablation_elec_unc_killarney.sh --smoke-test
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIGS=(
    configs/binary_noise_sched_ablation_elec_unc_g1p0.yaml
    configs/binary_noise_sched_ablation_elec_unc_g1p5.yaml
    configs/binary_noise_sched_ablation_elec_unc_g3p0.yaml
)
DATASETS="electricity"
WALL_TIME="3:00:00"
SMOKE=0
RESUME=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-test|--smoke) SMOKE=1; shift ;;
        --time) WALL_TIME="$2"; shift 2 ;;
        --resume) RESUME=1; shift ;;
        --datasets) DATASETS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

CONFIG_CSV="$(IFS=,; echo "${CONFIGS[*]}")"

ARGS=(
    --configs "$CONFIG_CSV"
    --datasets "$DATASETS"
    --gpu l40s
    --time "$WALL_TIME"
    --wandb-project ts-sandbox-leaderboard
)
if [[ "$SMOKE" -eq 1 ]]; then
    # Pass --smoke-test through so pipeline forces 1 epoch / tiny batch / light eval.
    ARGS=(
        --configs "$CONFIG_CSV"
        --datasets "$DATASETS"
        --gpu l40s
        --time "0:45:00"
        --smoke-test
        --wandb-project ts-sandbox-leaderboard
    )
fi
if [[ "$RESUME" -eq 1 ]]; then
    ARGS+=(--resume)
fi

echo "Submitting noise-sched ablation: configs=$CONFIG_CSV datasets=$DATASETS"
exec ./test_submit.sh "${ARGS[@]}"
