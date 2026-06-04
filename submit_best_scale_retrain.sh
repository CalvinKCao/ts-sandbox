#!/bin/bash
# =============================================================================
# Policy max_scale staged retrains (cap-rate decision tree, std floor 0.1).
#
# Config: configs/binary_dual_scale_staged_best_scale.yaml
#   Isolated ckpt/results stems: *-<dataset>-binary_dual_scale_staged_best_scale
#   (does NOT resume q99.5 binary_dual_scale_staged runs)
#
# Default datasets (policy MS differs from q99.5 grid): ETTh2, ETTm1/2, illness,
# solar_Alabama, weather, electricity. Skip ETTh1/dalia/traffic (legacy 2-stage)
# and exchange_rate/PeMS (reuse q99.5 grid in report).
#
# USAGE (Killarney login node, repo root):
#   ./submit_best_scale_retrain.sh
#   ./submit_best_scale_retrain.sh --smoke-test --datasets ETTh2
#   DATASETS=illness,weather ./submit_best_scale_retrain.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG="${CFG:-configs/binary_dual_scale_staged_best_scale.yaml}"
DATASETS="${DATASETS:-ETTh2,ETTm1,ETTm2,illness,solar_Alabama,weather,electricity}"
WALL="${WALL:-4:00:00}"

EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-test|--smoke) EXTRA+=(--smoke-test); WALL="0:45:00"; shift ;;
        --config) CFG="$2"; shift 2 ;;
        --datasets) DATASETS="$2"; shift 2 ;;
        --time) WALL="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

echo "=== Best-scale staged retrain (${DATASETS}) ==="
echo "  config: $CFG"
echo "  wall:   $WALL"
echo "  ckpt stem: *-<dataset>-binary_dual_scale_staged_best_scale"
echo "  (no --resume; new dirs per job)"
echo ""

exec "$SCRIPT_DIR/submit_grid.sh" \
    --configs "$CFG" \
    --datasets "$DATASETS" \
    --time "$WALL" \
    "${EXTRA[@]}"
