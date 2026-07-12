#!/bin/bash
# Binary vs MMPD diag for the lb336/hz720 ordinal-norm grid four (jobs 4208596–4208599).
#
# Prereq: run ./temp/submit_migrate_grid_lb336_hz720_ordinal_four_killarney.sh --apply
#
# USAGE (Killarney login, $SCRATCH/ts-sandbox):
#   ./temp/submit_binary_mmpd_diag_lb336_hz720_ordinal_four_killarney.sh --smoke-test
#   ./temp/submit_binary_mmpd_diag_lb336_hz720_ordinal_four_killarney.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

exec "$REPO_ROOT/submit_binary_mmpd_staged_diag_killarney.sh" \
    --binary-config configs/binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm.yaml \
    --binary-ckpt-stem binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm \
    --mmpd-config configs/mmpd_decoder_flat_subsets_paper_lb336_hz720_ordinal_norm.yaml \
    --mmpd-config-suffix mmpd_decoder_flat_subsets_paper_lb336_hz720_ordinal_norm \
    --mmpd-dir results/datasets/07-08-mmpd-decoder-ordinal-norm-lb336-hz720 \
    --datasets ETTh1,traffic,electricity,exchange_rate \
    --output-dir reports/binary_vs_mmpd_lb336_hz720_ordinal_four \
    --time 4:00:00 \
    "$@"
