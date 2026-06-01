#!/bin/bash
# =============================================================================
# Full re-eval: MMPD probabilistic baseline + binary dual-scale deterministic
# anchor and probabilistic sampling metrics.
#
# Uses existing MMPD checkpoints/indices from SOURCE_MATRIX_DIR; does not train.
# Binary dual-scale checkpoint roots are pinned below to the finished 05-31 grid.
# Submit from the Killarney login node, repo root ($SCRATCH/ts-sandbox).
#
# Usage:
#   ./slurm_mmpd_dualscale_reeval.sh
#   SOURCE_MATRIX_DIR=results/datasets/<matrix> ./slurm_mmpd_dualscale_reeval.sh
#
# Cancel a stuck matrix and resubmit (login node):
#   scancel -u "$USER" -n mmpd-mx-init -n mmpd-mx-merge
#   scancel -u "$USER" -n 'mmpd-mx-*' -n 'mmpd-mx-b-*' 2>/dev/null || true
#   cd "$SCRATCH/ts-sandbox" && git pull && ./slurm_mmpd_dualscale_reeval.sh --reeval-only
#
# Retry failed datasets only (same output dir, no merge until all partials exist):
#   MATRIX_OUTPUT_DIR=results/datasets/06-01-mmpd-binary-aligned \
#   MATRIX_DATASETS="ETTm2 illness" \
#   ./slurm_mmpd_dualscale_reeval.sh --reeval-only --skip-merge
# Then merge all 12:
#   MATRIX_OUTPUT_DIR=results/datasets/06-01-mmpd-binary-aligned \
#   ./slurm_mmpd_dualscale_reeval.sh --merge-only
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_MATRIX_DIR="${SOURCE_MATRIX_DIR:-results/datasets/06-01-mmpd-binary-aligned}"
RUN_STEM="${RUN_STEM:-$(date +%m-%d)-mmpd-dualscale-reeval}"
# Re-eval reuses indices/checkpoints from SOURCE unless MATRIX_OUTPUT_DIR is set.
MATRIX_OUTPUT_DIR="${MATRIX_OUTPUT_DIR:-$SOURCE_MATRIX_DIR}"

BINARY_ROOTS=(
  "results/ckpts/05-31-3828089-ETTh1-binary_dual_scale"
  "results/ckpts/05-31-3828090-ETTh2-binary_dual_scale"
  "results/ckpts/05-31-3828091-ETTm1-binary_dual_scale"
  "results/ckpts/05-31-3828092-ETTm2-binary_dual_scale"
  "results/ckpts/05-31-3828093-illness-binary_dual_scale"
  "results/ckpts/05-31-3828094-exchange_rate-binary_dual_scale"
  "results/ckpts/05-31-3828095-weather-binary_dual_scale"
  "results/ckpts/05-31-3828096-electricity-binary_dual_scale"
  "results/ckpts/05-31-3828097-traffic-binary_dual_scale"
  "results/ckpts/05-31-3828098-PeMS-binary_dual_scale"
  "results/ckpts/05-31-3828099-solar_Alabama-binary_dual_scale"
  "results/ckpts/05-31-3828100-dalia-binary_dual_scale"
)

export SOURCE_MATRIX_DIR
export MATRIX_OUTPUT_DIR
export MATRIX_INDICES_DIR="${MATRIX_INDICES_DIR:-$SOURCE_MATRIX_DIR}"
export MMPD_REUSE_DIR="${MMPD_REUSE_DIR:-$SOURCE_MATRIX_DIR}"
export BINARY_ANCHOR_ROOTS="${BINARY_ANCHOR_ROOTS:-${BINARY_ROOTS[*]}}"
export MATRIX_DATASETS="${MATRIX_DATASETS:-}"
export MATRIX_MERGE_DATASETS="${MATRIX_MERGE_DATASETS:-ETTh1 ETTh2 ETTm1 ETTm2 illness exchange_rate weather electricity traffic PeMS solar_Alabama dalia}"

if [[ $# -gt 0 ]]; then
  exec "$SCRIPT_DIR/slurm_mmpd_gaussian_anchor_eval.sh" "$@"
fi
exec "$SCRIPT_DIR/slurm_mmpd_gaussian_anchor_eval.sh" --reeval-only
