#!/bin/bash
# Probe true L40S max diffusion micro-batch (no 512/V AutoBS ceiling).
# CSV best_fit / safe_80pct are univariate counts U=B*C (one variate = one batch item).
#
# USAGE (Killarney login node, from $SCRATCH/ts-sandbox):
#   ./submit_probe_diffusion_max_batch_killarney.sh --smoke-test
#   ./submit_probe_diffusion_max_batch_killarney.sh
#   ./submit_probe_diffusion_max_batch_killarney.sh --datasets ETTh1,weather --max-candidate 768
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    SMOKE=0
    WALL="1:00:00"
    EXTRA=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --smoke-test|--smoke) SMOKE=1; shift ;;
            --time) WALL="$2"; shift 2 ;;
            *) EXTRA+=("$1"); shift ;;
        esac
    done

    if [[ "$SMOKE" -eq 1 ]]; then
        WALL="0:20:00"
        JOB_NAME="probe-bs-smoke"
    else
        JOB_NAME="probe-bs"
    fi

    LOG_DIR="${SCRIPT_DIR}/results/logs"
    mkdir -p "$LOG_DIR"
    DATE_STR="$(date +%m-%d)"
    LOG_FILE="$LOG_DIR/${DATE_STR}-probe-diffusion-max-batch-%j.log"

    SBATCH_ARGS=()
    [[ "$SMOKE" -eq 1 ]] && SBATCH_ARGS+=(--smoke-test)
    [[ ${#EXTRA[@]} -gt 0 ]] && SBATCH_ARGS+=("${EXTRA[@]}")

    echo "Submitting $JOB_NAME (L40S, wall=$WALL)..."
    exec sbatch \
        --parsable \
        --job-name="$JOB_NAME" \
        --account=aip-boyuwang \
        --nodes=1 \
        --gres=gpu:l40s:1 \
        --cpus-per-task=8 \
        --mem=60G \
        --time="$WALL" \
        --output="$LOG_FILE" \
        --error="$LOG_FILE" \
        --mail-type=FAIL \
        --mail-user="${USER}@uwo.ca" \
        "$SCRIPT_DIR/submit_probe_diffusion_max_batch_killarney.sh" \
        "${SBATCH_ARGS[@]}"
fi

# ---------------------------------------------------------------------------
# Compute node
# ---------------------------------------------------------------------------
REPO="${SLURM_SUBMIT_DIR:-$SCRIPT_DIR}"
REQ="$REPO/setup/requirements-killarney.txt"
[[ -d "$REPO" ]] || { echo "ERROR: repo missing at $REPO" >&2; exit 1; }
[[ -f "$REQ" ]] || { echo "ERROR: missing $REQ" >&2; exit 1; }
[[ -n "${SLURM_TMPDIR:-}" ]] || { echo "ERROR: SLURM_TMPDIR unset" >&2; exit 1; }

DATASETS="ETTh1,weather,electricity,exchange_rate,traffic"
GEOMETRIES="96/96,336/720_uncompressed"
STAGES="coarse,fine"
MAX_CANDIDATE=512
SMOKE=0
EXTRA_PY=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-test|--smoke) SMOKE=1; shift ;;
        --datasets) DATASETS="$2"; shift 2 ;;
        --geometries) GEOMETRIES="$2"; shift 2 ;;
        --stages) STAGES="$2"; shift 2 ;;
        --max-candidate) MAX_CANDIDATE="$2"; shift 2 ;;
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
command -v virtualenv >/dev/null || { echo "ERROR: virtualenv missing after module load" >&2; exit 1; }

echo "[setup] node-local venv from $REQ"
virtualenv --no-download "$SLURM_TMPDIR/env"
# shellcheck source=/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index -r "$REQ" -q
python -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, torch.cuda.get_device_name(0))"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TS_SANDBOX_REPO="$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"

OUT_CSV="$REPO/reports/probe_diffusion_max_batch_${SLURM_JOB_ID}.csv"
PY_ARGS=(
    utils/probe_diffusion_max_batch.py
    --datasets "$DATASETS"
    --geometries "$GEOMETRIES"
    --stages "$STAGES"
    --max-candidate "$MAX_CANDIDATE"
    --device cuda
    --output-csv "$OUT_CSV"
)
[[ "$SMOKE" -eq 1 ]] && PY_ARGS+=(--smoke-test)
[[ ${#EXTRA_PY[@]} -gt 0 ]] && PY_ARGS+=("${EXTRA_PY[@]}")

echo "[run] ${PY_ARGS[*]}"
python -u "${PY_ARGS[@]}"

echo "=========================================="
echo "Done: $(date)"
echo "CSV:  $OUT_CSV"
echo "=========================================="
