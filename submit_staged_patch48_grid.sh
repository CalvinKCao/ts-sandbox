#!/bin/bash
# =============================================================================
# All datasets: 4×4 patch + 2-stage (coarse→fine) in one job per dataset.
#
# Config: configs/binary_dual_scale_staged_patch48.yaml
#   - Phase 1: itrans + joint dual-scale diffusion HP (dit_patch_size [4,4])
#   - Phase 2+: staged coarse/fine pretrain, finetune, aligned staged_eval
#
# USAGE (Killarney login node, repo root):
#   ./submit_staged_patch48_grid.sh
#   ./submit_staged_patch48_grid.sh --smoke-test
#   DATASETS=ETTh1,PeMS ./submit_staged_patch48_grid.sh
#
# Warm-start Phase 1 from existing patch48-only train ckpts (skip in-job HP pretrain):
#   ./submit_staged_patch48_grid.sh --warmstart
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG="${CFG:-configs/binary_dual_scale_staged_patch48.yaml}"
DATASETS="${DATASETS:-ETTh1,ETTh2,ETTm1,ETTm2,illness,exchange_rate,weather,electricity,traffic,PeMS,solar_Alabama,dalia}"
WALL="${WALL:-12:00:00}"

EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-test|--smoke) EXTRA+=(--smoke-test); WALL="0:45:00"; shift ;;
        --config) CFG="$2"; shift 2 ;;
        --datasets) DATASETS="$2"; shift 2 ;;
        --time) WALL="$2"; shift 2 ;;
        --warmstart)
            CFG="configs/binary_dual_scale_staged_patch48_warmstart.yaml"
            WALL="${WALL:-8:00:00}"
            shift
            ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

echo "=== 4×4 + 2-stage grid (${DATASETS}) ==="
echo "  config: $CFG"
echo "  wall:   $WALL"
echo ""

exec "$SCRIPT_DIR/submit_grid.sh" \
    --configs "$CFG" \
    --datasets "$DATASETS" \
    --time "$WALL" \
    "${EXTRA[@]}"
