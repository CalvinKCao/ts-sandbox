#!/bin/bash
# Slurm: GT train/test + binary/MMPD test horizon-trend 4-panel compare (all datasets).
#
# CPU-only; reads eval npz + local CSV data. Outputs:
#   reports/forecast_horizon_trend/<dataset>/horizon_trend_four_panel.png
#
# USAGE (Killarney login node, from $SCRATCH/ts-sandbox):
#   ./submit_horizon_trend_full_compare.sh
#   ./submit_horizon_trend_full_compare.sh --smoke-test
#   ./submit_horizon_trend_full_compare.sh --datasets exchange_rate,weather
#   ./submit_horizon_trend_full_compare.sh --write-gt-splits
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ALL_DATASETS="ETTh1,ETTh2,ETTm1,ETTm2,illness,exchange_rate,weather,electricity,traffic,PeMS,solar_Alabama,dalia,dynamic"

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
            --write-gt-splits) EXTRA+=(--write-gt-splits); shift ;;
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
        CPUS=2
        JOB_NAME="horizon-trend-compare-smoke"
        EXTRA+=(--datasets illness)
    else
        TIME="2:00:00"
        MEM="32G"
        CPUS=8
        JOB_NAME="horizon-trend-compare-all"
        HAS_DATASETS=0
        for ((i = 0; i < ${#EXTRA[@]}; i++)); do
            [[ "${EXTRA[i]}" == "--datasets" ]] && HAS_DATASETS=1
        done
        [[ "$HAS_DATASETS" -eq 0 ]] && EXTRA+=(--datasets "$ALL_DATASETS")
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
        --cpus-per-task="$CPUS"
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
    sbatch "${S_ARGS[@]}" "$SCRIPT_DIR/submit_horizon_trend_full_compare.sh" "${EXTRA[@]}"
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

echo "Job $SLURM_JOB_ID on ${SLURMD_NODENAME:-?} — horizon trend 4-panel compare"
python -u utils/analyze_horizon_trend_full_compare.py "$@"
