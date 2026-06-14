#!/bin/bash
# Submit both reuse-based flat-subset sweeps (EMA decay + grad accum), 3h wall each.
#
# USAGE (Killarney login node, from $SCRATCH/ts-sandbox):
#   ./submit_binary_flat_subsets_reuse_sweeps.sh
#   ./submit_binary_flat_subsets_reuse_sweeps.sh ema
#   ./submit_binary_flat_subsets_reuse_sweeps.sh grad-accum
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

run_ema() { ./submit_binary_flat_subsets_ema_sweep.sh; }
run_grad() { ./submit_binary_flat_subsets_grad_accum_sweep.sh; }

case "${1:-all}" in
    all)
        run_ema
        echo ""
        run_grad
        ;;
    ema|ema-sweep)
        run_ema
        ;;
    grad-accum|grad|accum)
        run_grad
        ;;
    *)
        echo "Usage: $0 {all|ema|grad-accum}" >&2
        exit 1
        ;;
esac
