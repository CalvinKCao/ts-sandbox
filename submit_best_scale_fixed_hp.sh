#!/bin/bash
# =============================================================================
# Fast best-scale grid: policy max_scale + reused diffusion HP from the
# exhaustive binary_dual_scale_staged grid (no Optuna, no sampler sweep).
#
# Config: configs/binary_dual_scale_staged_best_scale_fixed_hp.yaml
#   Ckpt stem: *-<dataset>-binary_dual_scale_staged_best_scale_fixed_hp
#   Partials:  results/datasets/partials/<dataset>_staged_anchor.json
#
# USAGE (Killarney login node):
#   ./submit_best_scale_fixed_hp.sh
#   ./submit_best_scale_fixed_hp.sh --time 2:00:00
#   ./submit_best_scale_fixed_hp.sh --smoke-test --datasets ETTh2
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG="${CFG:-configs/binary_dual_scale_staged_best_scale_fixed_hp.yaml}"
DATASETS="${DATASETS:-ETTh1,ETTh2,ETTm1,ETTm2,illness,exchange_rate,weather,electricity,traffic,PeMS,solar_Alabama,dalia}"
WALL="${WALL:-2:00:00}"

EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-test|--smoke) EXTRA+=(--smoke-test); WALL="0:30:00"; shift ;;
        --config) CFG="$2"; shift 2 ;;
        --datasets) DATASETS="$2"; shift 2 ;;
        --time) WALL="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

echo "=== Best-scale fixed-HP retrain (${DATASETS}) ==="
echo "  config: $CFG"
echo "  wall:   $WALL"
echo "  reuse pretrain + HP from: *-<dataset>-binary_dual_scale_staged (no synthetic phase 1)"
echo "  policy max_scale: configs/binary_dual_scale_staged_best_scale_fixed_hp.yaml"
echo ""

exec "$SCRIPT_DIR/submit_grid.sh" \
    --configs "$CFG" \
    --datasets "$DATASETS" \
    --time "$WALL" \
    "${EXTRA[@]}"
