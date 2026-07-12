#!/bin/bash
# Migrate lb336/hz96 MMPD paper-subset checkpoints into $SCRATCH/ts-sandbox/reused/mmpd/.
#
# USAGE (Killarney login, $SCRATCH/ts-sandbox):
#   ./temp/submit_migrate_mmpd_lb336_hz96_killarney.sh --dry-run
#   ./temp/submit_migrate_mmpd_lb336_hz96_killarney.sh --apply
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/submit_migrate_reused_checkpoints_killarney.sh" \
    --migrate-mmpd-lb336-hz96 \
    "$@"
