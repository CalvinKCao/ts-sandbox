#!/bin/bash
# Migrate compatible checkpoints from results/ckpts into $SCRATCH/ts-sandbox/reused/.
# CPU-only I/O job (no GPU). Run from Killarney login node in $SCRATCH/ts-sandbox.
#
# USAGE:
#   ./temp/submit_migrate_reused_checkpoints_killarney.sh \
#     --config-suffix binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm \
#     --dataset ETTh1 --dry-run
#
#   ./temp/submit_migrate_reused_checkpoints_killarney.sh \
#     --config-suffix binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm \
#     --dataset ETTh1 --apply
#
#   ./temp/submit_migrate_reused_checkpoints_killarney.sh \
#     --migrate-mmpd-lb336-hz96 --dry-run
#
#   ./temp/submit_migrate_mmpd_lb336_hz96_killarney.sh --apply
#
#   ./temp/submit_migrate_grid_lb336_hz720_ordinal_four_killarney.sh --apply
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    WALL="0:30:00"
    APPLY=0
    MIGRATE_ARGS=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --time) WALL="$2"; shift 2 ;;
            --apply) APPLY=1; shift ;;
            --dry-run) APPLY=0; shift ;;
            *) MIGRATE_ARGS+=("$1"); shift ;;
        esac
    done

    if [[ ${#MIGRATE_ARGS[@]} -eq 0 ]]; then
        echo "ERROR: pass --config-suffix <stem> and/or --migrate-mmpd-lb336-hz96 or --migrate-grid-lb336-hz720-ordinal-four" >&2
        exit 1
    fi

    CONFIG_SUFFIX=""
    DATASET="ETTh1"
    JOB_NAME="migrate-reused"
    for ((i = 0; i < ${#MIGRATE_ARGS[@]}; i++)); do
        if [[ "${MIGRATE_ARGS[$i]}" == "--config-suffix" && $((i + 1)) -lt ${#MIGRATE_ARGS[@]} ]]; then
            CONFIG_SUFFIX="${MIGRATE_ARGS[$((i + 1))]}"
        fi
        if [[ "${MIGRATE_ARGS[$i]}" == "--dataset" && $((i + 1)) -lt ${#MIGRATE_ARGS[@]} ]]; then
            DATASET="${MIGRATE_ARGS[$((i + 1))]}"
        fi
        if [[ "${MIGRATE_ARGS[$i]}" == "--migrate-mmpd-lb336-hz96" ]]; then
            JOB_NAME="migrate-mmpd-lb336-hz96"
            DATASET="all"
        fi
        if [[ "${MIGRATE_ARGS[$i]}" == "--migrate-grid-lb336-hz720-ordinal-four" ]]; then
            JOB_NAME="migrate-grid-hz720-ord4"
            DATASET="four"
        fi
    done
    if [[ -n "$CONFIG_SUFFIX" ]]; then
        JOB_NAME="migrate-${CONFIG_SUFFIX##*_}"
        JOB_NAME="${JOB_NAME:0:40}-${DATASET}"
    fi

    LOG_DIR="${REPO_ROOT}/results/logs"
    mkdir -p "$LOG_DIR"
    DATE_STR="$(date +%m-%d)"
    LOG_FILE="$LOG_DIR/${DATE_STR}-migrate-reused-%j.log"

    echo "Submitting $JOB_NAME (CPU, wall=$WALL, apply=$APPLY)..."
    exec sbatch \
        --parsable \
        --job-name="$JOB_NAME" \
        --account=aip-boyuwang \
        --nodes=1 \
        --cpus-per-task=4 \
        --mem=16G \
        --time="$WALL" \
        --output="$LOG_FILE" \
        --error="$LOG_FILE" \
        --mail-type=FAIL \
        --mail-user="${USER}@uwo.ca" \
        "$SCRIPT_DIR/submit_migrate_reused_checkpoints_killarney.sh" \
        ${APPLY:+--apply} \
        "${MIGRATE_ARGS[@]}"
fi

REPO="${SLURM_SUBMIT_DIR:-$REPO_ROOT}"
REQ="$REPO/setup/requirements-killarney.txt"
[[ -d "$REPO" ]] || { echo "ERROR: repo missing at $REPO" >&2; exit 1; }
[[ -f "$REQ" ]] || { echo "ERROR: missing $REQ" >&2; exit 1; }

cd "$REPO"
module purge
module load StdEnv/2023 gcc python/3.12 scipy-stack/2025a
source setup/activate_killarney_venv.sh

APPLY=0
MIGRATE_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --apply) APPLY=1; shift ;;
        --dry-run) APPLY=0; shift ;;
        *) MIGRATE_ARGS+=("$1"); shift ;;
    esac
done

if [[ ${#MIGRATE_ARGS[@]} -eq 0 ]]; then
    echo "ERROR: missing migrate args" >&2
    exit 1
fi

if [[ "$APPLY" -eq 0 ]]; then
    MIGRATE_ARGS+=(--dry-run)
fi

echo "SCRATCH=${SCRATCH:-unset}"
echo "reused root: ${SCRATCH:+$SCRATCH/ts-sandbox/reused}"
echo "migrate args: ${MIGRATE_ARGS[*]}"

python utils/migrate_reused_checkpoints.py "${MIGRATE_ARGS[@]}"

echo "Done: $(date)"
