#!/bin/bash
# Slurm wrapper: top-20 per-window anchor_mse / CRPS delta panels (fair MMPD vs binary).
#
# USAGE (Killarney login node, from $SCRATCH/ts-sandbox):
#   ./submit_fair_mmpd_vs_binary_delta_viz.sh
#   ./submit_fair_mmpd_vs_binary_delta_viz.sh --smoke-test
#   ./submit_fair_mmpd_vs_binary_delta_viz.sh --dependency afterok:12345
#   ./submit_fair_mmpd_vs_binary_delta_viz.sh --allow-fallback-binary
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    IS_SMOKE=0
    DEPENDENCY=""
    EXTRA=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --smoke-test|--smoke) IS_SMOKE=1; shift ;;
            --dependency) DEPENDENCY="$2"; shift 2 ;;
            --allow-fallback-binary) EXTRA+=(--allow-fallback-binary); shift ;;
            --datasets) EXTRA+=(--datasets "$2"); shift 2 ;;
            --mmpd-run) EXTRA+=(--mmpd-run "$2"); shift 2 ;;
            *) echo "Unknown arg: $1" >&2; exit 1 ;;
        esac
    done

    if [[ -d "${SCRATCH:-}/ts-sandbox" ]]; then
        REPO="${SCRATCH}/ts-sandbox"
    else
        REPO="$SCRIPT_DIR"
    fi
    cd "$REPO"

    if [[ "$IS_SMOKE" -eq 1 ]]; then
        TIME="0:20:00"
        MEM="8G"
        JOB_NAME="fair-mmpd-delta-viz-smoke"
        EXTRA+=(--datasets ETTh1 --top-k 2)
    else
        TIME="1:00:00"
        MEM="16G"
        JOB_NAME="fair-mmpd-delta-viz"
    fi

    S_ARGS=(
        --job-name="$JOB_NAME"
        --account=aip-boyuwang
        --time="$TIME"
        --nodes=1
        --cpus-per-task=4
        --mem="$MEM"
        --output="$REPO/logs/${JOB_NAME}-%j.log"
        --error="$REPO/logs/${JOB_NAME}-%j.log"
        --mail-type=FAIL
        --mail-user="${USER:-ccao87}@uwo.ca"
    )
    if [[ -n "$DEPENDENCY" ]]; then
        S_ARGS+=(--dependency="$DEPENDENCY")
    fi

    mkdir -p "$REPO/logs"
    sbatch "${S_ARGS[@]}" "$SCRIPT_DIR/submit_fair_mmpd_vs_binary_delta_viz.sh" "${EXTRA[@]}"
    exit 0
fi

# --- compute node ---
REPO="${SLURM_SUBMIT_DIR:-$SCRIPT_DIR}"
cd "$REPO"

module purge 2>/dev/null || true
module load StdEnv/2023 python/3.11 2>/dev/null || true

if [[ -f "$REPO/.venv/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "$REPO/.venv/bin/activate"
elif [[ -f "${SCRATCH:-}/ts-sandbox/.venv/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${SCRATCH}/ts-sandbox/.venv/bin/activate"
fi

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

echo "Job $SLURM_JOB_ID on ${SLURMD_NODENAME:-?} — fair MMPD vs binary delta viz"
python -u utils/visualize_fair_mmpd_vs_binary_delta.py "$@"
