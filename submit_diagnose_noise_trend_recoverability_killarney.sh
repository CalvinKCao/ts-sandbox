#!/bin/bash
# Trend/ACF recoverability under bit-flip noise: 96/96 vs 336/720_uncompressed.
#
# USAGE (Killarney login, from $SCRATCH/ts-sandbox):
#   ./submit_diagnose_noise_trend_recoverability_killarney.sh --datasets ETTh1
#   for ds in ETTh1 weather electricity exchange_rate traffic; do
#     ./submit_diagnose_noise_trend_recoverability_killarney.sh --datasets "$ds"
#   done
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    WALL="0:15:00"
    EXTRA=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --time) WALL="$2"; shift 2 ;;
            *) EXTRA+=("$1"); shift ;;
        esac
    done

    JOB_NAME="noise-trend-diag"
    for ((i = 0; i < ${#EXTRA[@]}; i++)); do
        if [[ "${EXTRA[$i]}" == "--datasets" && $((i + 1)) -lt ${#EXTRA[@]} ]]; then
            JOB_NAME="noise-trend-${EXTRA[$((i + 1))]%%,*}"
            break
        fi
    done

    LOG_DIR="${SCRIPT_DIR}/results/logs"
    mkdir -p "$LOG_DIR"
    DATE_STR="$(date +%m-%d)"
    LOG_FILE="$LOG_DIR/${DATE_STR}-noise-trend-diag-%j.log"

    echo "Submitting $JOB_NAME (L40S, wall=$WALL)..."
    exec sbatch \
        --parsable \
        --job-name="$JOB_NAME" \
        --account=aip-boyuwang \
        --nodes=1 \
        --gres=gpu:l40s:1 \
        --cpus-per-task=4 \
        --mem=32G \
        --time="$WALL" \
        --output="$LOG_FILE" \
        --error="$LOG_FILE" \
        --mail-type=FAIL \
        --mail-user="${USER}@uwo.ca" \
        "$SCRIPT_DIR/submit_diagnose_noise_trend_recoverability_killarney.sh" \
        "${EXTRA[@]}"
fi

REPO="${SLURM_SUBMIT_DIR:-$SCRIPT_DIR}"
REQ="$REPO/setup/requirements-killarney.txt"
[[ -d "$REPO" ]] || { echo "ERROR: repo missing at $REPO" >&2; exit 1; }
[[ -f "$REQ" ]] || { echo "ERROR: missing $REQ" >&2; exit 1; }
[[ -n "${SLURM_TMPDIR:-}" ]] || { echo "ERROR: SLURM_TMPDIR unset" >&2; exit 1; }

DATASETS="ETTh1,weather,electricity,exchange_rate,traffic"
GEOMETRIES="96/96,336/720_uncompressed"
N_SAMPLES=24
STAGE="coarse"
EXTRA_PY=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --datasets) DATASETS="$2"; shift 2 ;;
        --geometries) GEOMETRIES="$2"; shift 2 ;;
        --n-samples) N_SAMPLES="$2"; shift 2 ;;
        --stage) STAGE="$2"; shift 2 ;;
        *) EXTRA_PY+=("$1"); shift ;;
    esac
done

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID   Node: ${SLURMD_NODENAME:-unknown}"
echo "GPU:    $(nvidia-smi -L 2>/dev/null | head -1 || echo unknown)"
echo "Started: $(date)"
echo "Repo:   $REPO"
echo "=========================================="

module purge 2>/dev/null || true
module load StdEnv/2023 python/3.11 cuda/12.2 cudnn/8.9 2>/dev/null || true
command -v virtualenv >/dev/null || { echo "ERROR: virtualenv missing" >&2; exit 1; }

echo "[setup] node-local venv from $REQ"
virtualenv --no-download "$SLURM_TMPDIR/env"
# shellcheck source=/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index -r "$REQ" -q
python -c "import torch, yaml, einops; print('torch', torch.__version__)"

export PYTHONUNBUFFERED=1
export TS_SANDBOX_REPO="$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"

OUT_DIR="$REPO/reports/noise_trend_recoverability_${SLURM_JOB_ID}"
PY_ARGS=(
    utils/diagnose_binary_noise_trend_recoverability.py
    --datasets "$DATASETS"
    --geometries "$GEOMETRIES"
    --n-samples "$N_SAMPLES"
    --stage "$STAGE"
    --output-dir "$OUT_DIR"
)
[[ ${#EXTRA_PY[@]} -gt 0 ]] && PY_ARGS+=("${EXTRA_PY[@]}")

echo "[run] ${PY_ARGS[*]}"
python -u "${PY_ARGS[@]}"

echo "=========================================="
echo "Done: $(date)"
echo "Outputs: $OUT_DIR"
echo "=========================================="
