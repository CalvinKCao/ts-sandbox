#!/bin/bash
# =============================================================================
# Synthetic DiT capacity probe — self-resubmitting Killarney job (L40S default).
#
# Trains four DiT variants on on-the-fly linear + periodic univariate series only.
# No RealTS, no real datasets.
#
# USAGE (login node, repo root):
#   ./slurm_synthetic_dit_capacity.sh --smoke-test
#   ./slurm_synthetic_dit_capacity.sh
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "${SLURM_JOB_ID:-}" ]; then
    IS_SMOKE=0
    for arg in "$@"; do [ "$arg" = "--smoke-test" ] && IS_SMOKE=1; done

    mkdir -p "$SCRIPT_DIR/results/bootstrap"
    SB_OUT='results/bootstrap/%x-%j.out'
    SB_ERR='results/bootstrap/%x-%j.err'
    EXPORT_ARG="ALL,TS_SANDBOX_PROJECT_ROOT_SUBMIT_DIR=1"

    if [ "$IS_SMOKE" -eq 1 ]; then
        echo "Submitting SMOKE (L40S, 8G, 30 min)..."
        sbatch \
            --job-name=synth-dit-cap-smoke \
            --account=aip-boyuwang \
            --time=0:30:00 \
            --nodes=1 \
            --gres=gpu:l40s:1 \
            --cpus-per-task=4 \
            --mem=8G \
            --chdir="$SCRIPT_DIR" \
            --output="$SB_OUT" \
            --error="$SB_ERR" \
            --mail-type=END,FAIL \
            --mail-user=ccao87@uwo.ca \
            --export="$EXPORT_ARG" \
            "$SCRIPT_DIR/slurm_synthetic_dit_capacity.sh" "$@"
    else
        echo "Submitting FULL (L40S, 50G, 12h)..."
        sbatch \
            --job-name=synth-dit-cap \
            --account=aip-boyuwang \
            --time=12:00:00 \
            --nodes=1 \
            --gres=gpu:l40s:1 \
            --cpus-per-task=8 \
            --mem=50G \
            --chdir="$SCRIPT_DIR" \
            --output="$SB_OUT" \
            --error="$SB_ERR" \
            --mail-type=BEGIN,END,FAIL \
            --mail-user=ccao87@uwo.ca \
            --export="$EXPORT_ARG" \
            "$SCRIPT_DIR/slurm_synthetic_dit_capacity.sh" "$@"
    fi
    exit 0
fi

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$SCRIPT_DIR}"

STEM="$(date +%m-%d)-${SLURM_JOB_ID: -3}-synth-dit-capacity"
RUN_ROOT="$SLURM_SUBMIT_DIR/results/$STEM"
LOG_DIR="$RUN_ROOT/logs"
CKPT_DIR="$RUN_ROOT/ckpts"
DATA_DIR="$RUN_ROOT/datasets"
mkdir -p "$LOG_DIR" "$CKPT_DIR" "$DATA_DIR"
LOG_FILE="$LOG_DIR/${STEM}.log"
touch "$LOG_FILE"
exec >>"$LOG_FILE" 2>&1

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID   Node: ${SLURMD_NODENAME:-unknown}"
echo "GPU:    $(nvidia-smi -L 2>/dev/null | head -1 || echo unknown)"
echo "Started: $(date)"
echo "Log: $LOG_FILE"
echo "=========================================="

module purge || true
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9

if [ "${TS_SANDBOX_PROJECT_ROOT_SUBMIT_DIR:-}" = "1" ] && [ -d "$SLURM_SUBMIT_DIR/models/diffusion_tsf" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
elif [ -d "${SCRATCH:-}/ts-sandbox" ]; then
    PROJECT_ROOT="${SCRATCH}/ts-sandbox"
elif [ -d "$SLURM_SUBMIT_DIR/models/diffusion_tsf" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
elif [ -d "$HOME/ts-sandbox" ]; then
    PROJECT_ROOT="$HOME/ts-sandbox"
else
    echo "ERROR: ts-sandbox not found"
    exit 1
fi
cd "$PROJECT_ROOT"
echo "PROJECT_ROOT=$PROJECT_ROOT"

echo "[setup] venv on SLURM_TMPDIR..."
virtualenv --no-download "$SLURM_TMPDIR/env"
# shellcheck source=/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index \
    'torch==2.11.0+computecanada' \
    numpy pandas scipy scikit-learn tqdm einops -q

python -c "import torch; assert torch.cuda.is_available(), 'CUDA required'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"

export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONUNBUFFERED=1

SMOKE_FLAG=""
PY_ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--smoke-test" ]; then
        SMOKE_FLAG="--smoke-test"
    else
        PY_ARGS+=("$arg")
    fi
done

python -u -m models.diffusion_tsf.train_synthetic_dit_capacity \
    --results-dir "$LOG_DIR" \
    $SMOKE_FLAG \
    "${PY_ARGS[@]}"

echo "=========================================="
echo "Done: $(date)"
echo "Metrics: $LOG_DIR"
echo "Log: $LOG_FILE"
echo "=========================================="
