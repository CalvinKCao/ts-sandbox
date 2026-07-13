#!/bin/bash
# Migrate binary grid jobs 4208596–4208599 + paper (non-ordinal) MMPD lb336/hz720 ckpts.
#
# Binary: ETTh1, traffic, electricity, exchange_rate @ past_native per-dataset stems
# MMPD:  same four datasets @ mmpd_decoder_flat_subsets_paper_lb336_hz720
#        (campaign 07-10-mmpd-decoder-paper-lb336-hz720-subset)
#
# USAGE (Killarney login, $SCRATCH/ts-sandbox):
#   ./temp/submit_migrate_grid_lb336_hz720_ordinal_four_killarney.sh --dry-run
#   ./temp/submit_migrate_grid_lb336_hz720_ordinal_four_killarney.sh --apply
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/submit_migrate_reused_checkpoints_killarney.sh" \
    --migrate-grid-lb336-hz720-ordinal-four \
    "$@"
