#!/bin/bash
# Slurm wrapper: instance-norm horizon trend histograms (binary vs MMPD).
#
# Reads saved eval npz only — no GPU inference. Ensure these exist first:
#   results/datasets/06-16-mmpd-maskae-fair-13d/raw/mmpd_{exchange_rate,weather}.npz
#   results/datasets/*-{ds}-binary_anchor_stationary_flat_subsets_grad_accum_150_lr_lo/raw/staged_anchor_{ds}.npz
#
# USAGE (Killarney login node, from $SCRATCH/ts-sandbox):
#   ./submit_horizon_trend_histogram.sh
#   ./submit_horizon_trend_histogram.sh --smoke-test
#   ./submit_horizon_trend_histogram.sh --dependency afterok:12345
#   ./submit_horizon_trend_histogram.sh --datasets exchange_rate,weather
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
            --datasets) EXTRA+=(--datasets "$2"); shift 2 ;;
            --mmpd-run) EXTRA+=(--mmpd-run "$2"); shift 2 ;;
            --output-dir) EXTRA+=(--output-dir "$2"); shift 2 ;;
            --allow-fallback-binary) EXTRA+=(--allow-fallback-binary); shift ;;
            --binary-only) EXTRA+=(--binary-only); shift ;;
            --binary-results-dir) EXTRA+=(--binary-results-dir "$2"); shift 2 ;;
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
        TIME="0:15:00"
        MEM="8G"
        JOB_NAME="horizon-trend-hist-smoke"
        EXTRA+=(--datasets illness --binary-only)
    else
        TIME="0:30:00"
        MEM="16G"
        JOB_NAME="horizon-trend-hist"
        HAS_DATASETS=0
        for ((i = 0; i < ${#EXTRA[@]}; i++)); do
            [[ "${EXTRA[i]}" == "--datasets" ]] && HAS_DATASETS=1
        done
        [[ "$HAS_DATASETS" -eq 0 ]] && EXTRA+=(--datasets exchange_rate,weather)
        HAS_MMPD=0
        for ((i = 0; i < ${#EXTRA[@]}; i++)); do
            [[ "${EXTRA[i]}" == "--mmpd-run" ]] && HAS_MMPD=1
        done
        [[ "$HAS_MMPD" -eq 0 ]] && EXTRA+=(--mmpd-run results/datasets/06-16-mmpd-maskae-fair-13d)
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
    sbatch "${S_ARGS[@]}" "$SCRIPT_DIR/submit_horizon_trend_histogram.sh" "${EXTRA[@]}"
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

echo "Job $SLURM_JOB_ID on ${SLURMD_NODENAME:-?} — horizon trend histograms"
python -u utils/analyze_horizon_trend_distribution.py "$@"
