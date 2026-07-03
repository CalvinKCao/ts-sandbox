#!/bin/bash
# Slurm: classical statistical baselines (statsforecast + statsmodels) for all
# repo datasets except dalia. CPU-bound on L40S nodes; logs to ts-sandbox-leaderboard.
#
# USAGE (Killarney login node, from $SCRATCH/ts-sandbox):
#   ./setup/killarney_bootstrap_classical_wheels.sh   # one-time: cache PyPI wheels on PROJECT
#   ./submit_classical_baselines.sh --smoke-test
#   ./submit_classical_baselines.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ALL_DATASETS="ETTh1,ETTh2,ETTm1,ETTm2,illness,exchange_rate,weather,electricity,traffic,PeMS,solar_Alabama,dynamic"

_resolve_project() {
    if [[ -n "${PROJECT:-}" ]]; then
        return 0
    fi
    shopt -s nullglob
    _m=("$HOME"/projects/aip-* "$HOME"/projects/def-*)
    shopt -u nullglob
    if [[ "${#_m[@]}" -gt 0 ]]; then
        export PROJECT
        PROJECT="$(readlink -f "${_m[0]}")"
    fi
}

_classical_wheel_dir() {
    _resolve_project
    if [[ -z "${PROJECT:-}" ]]; then
        return 1
    fi
    echo "$PROJECT/$USER/ts-sandbox/wheels-classical"
}

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

    WHEEL_DIR="$(_classical_wheel_dir || true)"
    if [[ -z "$WHEEL_DIR" || ! -d "$WHEEL_DIR" ]] || ! compgen -G "$WHEEL_DIR/*.whl" >/dev/null; then
        echo "ERROR: classical wheel cache missing at ${WHEEL_DIR:-\$PROJECT/\$USER/ts-sandbox/wheels-classical}" >&2
        echo "Run once on the login node:" >&2
        echo "  cd \"\$SCRATCH/ts-sandbox\" && ./setup/killarney_bootstrap_classical_wheels.sh" >&2
        exit 1
    fi

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

STEM="$(date +%m-%d)-${SLURM_JOB_ID: -3}-classical-baselines"
LOG="$REPO/results/logs/${STEM}.log"
mkdir -p "$REPO/results/logs" "$REPO/results/datasets"
exec >>"$LOG" 2>&1

echo "Job $SLURM_JOB_ID on ${SLURMD_NODENAME:-?} — classical baselines"
echo "CPUs=$SLURM_CPUS_PER_TASK mem=${SLURM_MEM_PER_NODE:-?} stem=$STEM"
echo "Slurm log mirror: $REPO/logs/${SLURM_JOB_NAME:-classical-baselines}-${SLURM_JOB_ID}.log"

REQ="$REPO/setup/requirements-killarney.txt"
WHEEL_DIR="$(_classical_wheel_dir || true)"

[[ -f "$REQ" ]] || {
    echo "ERROR: missing $REQ — run ./setup/killarney_freeze_requirements.sh on login node" >&2
    exit 1
}
[[ -n "$WHEEL_DIR" && -d "$WHEEL_DIR" ]] || {
    echo "ERROR: classical wheel cache missing — run ./setup/killarney_bootstrap_classical_wheels.sh on login node" >&2
    exit 1
}
[[ -n "${SLURM_TMPDIR:-}" ]] || { echo "ERROR: SLURM_TMPDIR is not set." >&2; exit 1; }

# Arrow module must load before venv (Alliance pyarrow is not on PyPI).
module purge 2>/dev/null || true
module load StdEnv/2023 python/3.11 2>/dev/null || true
module load gcc arrow 2>/dev/null || {
    echo "ERROR: could not load gcc/arrow (required for pyarrow / statsforecast)." >&2
    exit 1
}
command -v virtualenv >/dev/null || { echo "ERROR: virtualenv not available after module load." >&2; exit 1; }

echo "[setup] Building node-local venv on \$SLURM_TMPDIR from $REQ"
virtualenv --no-download "$SLURM_TMPDIR/env"
# shellcheck source=/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index -r "$REQ" -q
echo "[setup] pyarrow from Alliance wheelhouse (arrow module loaded)..."
pip install --no-index pyarrow -q
echo "[setup] statsforecast stack from $WHEEL_DIR ..."
pip install --no-index --find-links "$WHEEL_DIR" --no-deps \
    cloudpickle threadpoolctl coreforecast utilsforecast fugue statsmodels statsforecast -q
pip install --no-index numba -q 2>/dev/null || true

python -c "
import statsforecast, statsmodels, torch, wandb, yaml
print('venv ok: torch', torch.__version__, '| statsforecast', statsforecast.__version__)
"

export PYTHONUNBUFFERED=1

python -u utils/run_classical_baselines.py \
    --output-dir "$REPO/results/datasets/$STEM" \
    --n-jobs "${SLURM_CPUS_PER_TASK:-1}" \
    "$@"

echo "Done. Log: $LOG"
