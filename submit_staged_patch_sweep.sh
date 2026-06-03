#!/bin/bash
# Quick patch-size smoke sweep: ETTh1 + exchange_rate × 4 DiT patch grids.
#
# dit_patch_size is [value_height, time_width] on the 16×96 CDF canvas:
#   p8w4  = 8 high × 4 wide  → [8, 4]
#   p16w4 = 16 high × 4 wide → [16, 4]
#   p16w8 = 16 high × 8 wide → [16, 8]
#   p2w4  = 2 high × 4 wide  → [2, 4]
#
# Killarney:
#   ./submit_staged_patch_sweep.sh
#   WALL=1:30:00 ./submit_staged_patch_sweep.sh
#
# Local GPU (WSL):
#   ./submit_staged_patch_sweep.sh --local

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_CFGS=(
    configs/binary_dual_scale_staged_smoke_p8w4.yaml
    configs/binary_dual_scale_staged_smoke_p16w4.yaml
    configs/binary_dual_scale_staged_smoke_p16w8.yaml
    configs/binary_dual_scale_staged_smoke_p2w4.yaml
)
DATASETS="${DATASETS:-ETTh1,exchange_rate}"
WALL="${WALL:-1:00:00}"
WANDB_PROJECT="${WANDB_PROJECT:-ts-sandbox-staged-patch-sweep}"

LOCAL=0
EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --local) LOCAL=1; shift ;;
        --datasets) DATASETS="$2"; shift 2 ;;
        --time) WALL="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ "$LOCAL" -eq 1 ]]; then
    cd "$SCRIPT_DIR"
    source .venv/bin/activate
    for cfg in "${PATCH_CFGS[@]}"; do
        IFS=',' read -ra DS_ARR <<< "$DATASETS"
        for ds in "${DS_ARR[@]}"; do
            echo "=== LOCAL smoke: $cfg dataset=$ds ==="
            python -u -m models.diffusion_tsf.train_multivariate_pipeline \
                --config "$cfg" \
                --dataset "$ds" \
                --smoke-test \
                --wandb \
                --wandb-project "$WANDB_PROJECT"
        done
    done
    exit 0
fi

CFG_CSV=$(IFS=,; echo "${PATCH_CFGS[*]}")
exec "$SCRIPT_DIR/submit_grid.sh" \
    --smoke \
    --configs "$CFG_CSV" \
    --datasets "$DATASETS" \
    --time "$WALL" \
    --wandb-project "$WANDB_PROJECT"
