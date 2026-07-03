#!/bin/bash
# Slurm: classical statistical baselines (statsforecast + statsmodels) for all
# repo datasets except dalia. CPU-bound on L40S nodes; logs to ts-sandbox-leaderboard.
#
# USAGE (Killarney login node, from $SCRATCH/ts-sandbox):
#   ./submit_classical_baselines.sh
#   ./submit_classical_baselines.sh --smoke-test
#   ./submit_classical_baselines.sh --datasets ETTh1,illness
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ALL_DATASETS="ETTh1,ETTh2,ETTm1,ETTm2,illness,exchange_rate,weather,electricity,traffic,PeMS,solar_Alabama,dynamic"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    IS_SMOKE=0
    EXTRA=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --smoke-test|--smoke) IS_SMOKE=1; shift ;;
            --datasets) EXTRA+=(--datasets "$2"); shift 2 ;;
            --config) EXTRA+=(--config "$2"); shift 2 ;;
            --dry-run) EXTRA+=(--dry-run); shift ;;
            --no-wandb) EXTRA+=(--no-wandb); shift ;;
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
        TIME="0:45:00"
        MEM="16G"
        CPUS=4
        JOB_NAME="classical-baselines-smoke"
        EXTRA+=(--smoke-test)
    else
        TIME="8:00:00"
        MEM="64G"
        CPUS=16
        JOB_NAME="classical-baselines-all"
        HAS_DATASETS=0
        for ((i = 0; i < ${#EXTRA[@]}; i++)); do
            [[ "${EXTRA[i]}" == "--datasets" ]] && HAS_DATASETS=1
        done
        [[ "$HAS_DATASETS" -eq 0 ]] && EXTRA+=(--datasets "$ALL_DATASETS")
    fi

    mkdir -p "$REPO/results/logs" "$REPO/logs"
    LOG="$REPO/results/logs/$(date +%m-%d)-classical-baselines-submit.log"

    S_ARGS=(
        --job-name="$JOB_NAME"
        --account=aip-boyuwang
        --time="$TIME"
        --nodes=1
        --cpus-per-task="$CPUS"
        --mem="$MEM"
        --gres=gpu:l40s:1
        --output="$REPO/logs/${JOB_NAME}-%j.log"
        --error="$REPO/logs/${JOB_NAME}-%j.log"
        --mail-type=FAIL
        --mail-user="${USER:-ccao87}@uwo.ca"
    )

    sbatch "${S_ARGS[@]}" "$SCRIPT_DIR/submit_classical_baselines.sh" "${EXTRA[@]}" | tee -a "$LOG"
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

# statsforecast / statsmodels are not in the Alliance wheelhouse — install once per job.
pip install -q statsforecast statsmodels

export PYTHONUNBUFFERED=1

STEM="$(date +%m-%d)-${SLURM_JOB_ID: -3}-classical-baselines"
LOG="$REPO/results/logs/${STEM}.log"
mkdir -p "$REPO/results/logs" "$REPO/results/datasets"
exec >>"$LOG" 2>&1

echo "Job $SLURM_JOB_ID on ${SLURMD_NODENAME:-?} — classical baselines"
echo "CPUs=$SLURM_CPUS_PER_TASK mem=${SLURM_MEM_PER_NODE:-?} stem=$STEM"

python -u utils/run_classical_baselines.py \
    --output-dir "$REPO/results/datasets/$STEM" \
    --n-jobs "${SLURM_CPUS_PER_TASK:-1}" \
    "$@"

echo "Done. Log: $LOG"
