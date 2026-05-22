#!/bin/bash
# PatchGAN iTransformer Optuna tuning on Killarney.
#
# Usage from the repo checkout on Killarney:
#   bash slurm_patchgan_tune.sh
#   WORKERS=16 MAX_CONCURRENT=8 DATASET=ETTh1 bash slurm_patchgan_tune.sh
#   ARCHITECTURE=all1d N_TRIALS_PER_WORKER=2 bash slurm_patchgan_tune.sh
#   SMOKE_TEST=1 bash slurm_patchgan_tune.sh
#
# The default Optuna storage is intentionally relative to the repo root:
#   sqlite:///patchgan_tuning.db
# so independent array workers join the same SQLite-backed study.

#SBATCH --job-name=patchgan-tune
#SBATCH --account=aip-boyuwang
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=1-00:00:00
#SBATCH --array=1-8%8
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ccao87@uwo.ca

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "${SLURM_JOB_ID:-}" ]; then
    WORKERS="${WORKERS:-8}"
    MAX_CONCURRENT="${MAX_CONCURRENT:-$WORKERS}"
    mkdir -p "$SCRIPT_DIR/results/patchgan_tune/logs"
    echo "Submitting PatchGAN Optuna array: workers=$WORKERS max_concurrent=$MAX_CONCURRENT"
    sbatch \
        --array="1-${WORKERS}%${MAX_CONCURRENT}" \
        --output=/dev/null \
        --error=/dev/null \
        "$SCRIPT_DIR/slurm_patchgan_tune.sh"
    exit 0
fi

ARRAY_ID="${SLURM_ARRAY_TASK_ID:-1}"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$SCRIPT_DIR}"
RUN_ROOT="$SUBMIT_DIR/results/patchgan_tune"
LOG_DIR="$RUN_ROOT/logs"
CKPT_DIR="$RUN_ROOT/ckpts"
mkdir -p "$LOG_DIR" "$CKPT_DIR"

STEM="$(date +%m-%d)-${SLURM_JOB_ID: -3}-a${ARRAY_ID}-patchgan"
LOG_FILE="$LOG_DIR/${STEM}.log"
exec >>"$LOG_FILE" 2>&1

echo "=========================================="
echo "PatchGAN Optuna worker"
echo "Job: $SLURM_JOB_ID  Array: $ARRAY_ID  Node: ${SLURMD_NODENAME:-unknown}"
echo "Started: $(date)"
echo "Submit dir: $SUBMIT_DIR"
echo "Log: $LOG_FILE"
echo "=========================================="

module purge || true
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9

if [ -d "${SCRATCH:-}/ts-sandbox" ]; then
    PROJECT_ROOT="$SCRATCH/ts-sandbox"
elif [ -d "$SUBMIT_DIR/models/diffusion_tsf" ]; then
    PROJECT_ROOT="$SUBMIT_DIR"
elif [ -d "$HOME/ts-sandbox" ]; then
    PROJECT_ROOT="$HOME/ts-sandbox"
else
    echo "ERROR: could not find ts-sandbox checkout. Clone it to \$SCRATCH/ts-sandbox."
    exit 1
fi

if [ -z "${PROJECT:-}" ] && [ -d "$HOME/projects" ]; then
    shopt -s nullglob
    matches=("$HOME"/projects/aip-* "$HOME"/projects/def-*)
    shopt -u nullglob
    if [ "${#matches[@]}" -gt 0 ]; then
        export PROJECT="$(readlink -f "${matches[0]}")"
    fi
fi

STORE_BASE="${PROJECT:-${SCRATCH:-}}"
if [ -z "${STORE:-}" ] && [ -z "$STORE_BASE" ]; then
    echo "ERROR: neither STORE nor PROJECT/SCRATCH is set."
    exit 1
fi
STORE="${STORE:-$STORE_BASE/$USER/ts-sandbox-patchgan}"
mkdir -p "$STORE"

if [ "${PATCHGAN_USE_SHARED_VENV:-0}" = "1" ] && [ -d "$STORE/venv" ]; then
    echo "[setup] Activating shared venv: $STORE/venv"
    source "$STORE/venv/bin/activate"
else
    echo "[setup] Building node-local venv on $SLURM_TMPDIR"
    virtualenv --no-download "$SLURM_TMPDIR/env"
    source "$SLURM_TMPDIR/env/bin/activate"
    pip install --no-index --upgrade pip -q
    pip install --no-index 'torch==2.11.0+computecanada' numpy pandas scipy scikit-learn tqdm -q
    pip install --no-index optuna -q || pip install optuna -q
fi

python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA is required for this Slurm job"
print("torch", torch.__version__, "gpu", torch.cuda.get_device_name(0))
PY

cd "$PROJECT_ROOT"

DATASET="${DATASET:-ETTh1}"
ARCHITECTURE="${ARCHITECTURE:-both}"
N_TRIALS_PER_WORKER="${N_TRIALS_PER_WORKER:-1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-8}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-200}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-80}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-2}"
LOOKBACK_LENGTH="${LOOKBACK_LENGTH:-96}"
FORECAST_LENGTH="${FORECAST_LENGTH:-96}"
LOOKBACK_OVERLAP="${LOOKBACK_OVERLAP:-8}"
OPTUNA_STORAGE="${OPTUNA_STORAGE:-sqlite:///patchgan_tuning.db}"

if [ "$ARCHITECTURE" = "both" ]; then
    if [ $((ARRAY_ID % 2)) -eq 1 ]; then
        ARCH="all1d"
    else
        ARCH="1d2d"
    fi
else
    ARCH="$ARCHITECTURE"
fi

STUDY_NAME="patchgan_${ARCH}_${DATASET}_v1"
PY_ARGS=(
    -m models.diffusion_tsf.train_patchgan_tuning
    --architecture "$ARCH"
    --dataset "$DATASET"
    --lookback-length "$LOOKBACK_LENGTH"
    --forecast-length "$FORECAST_LENGTH"
    --lookback-overlap "$LOOKBACK_OVERLAP"
    --batch-size "$BATCH_SIZE"
    --epochs "$EPOCHS"
    --n-trials "$N_TRIALS_PER_WORKER"
    --num-workers "$DATALOADER_WORKERS"
    --max-train-batches "$MAX_TRAIN_BATCHES"
    --max-val-batches "$MAX_VAL_BATCHES"
    --checkpoint-dir "$CKPT_DIR"
    --storage "$OPTUNA_STORAGE"
    --study-name "$STUDY_NAME"
    --seed "$((42 + ARRAY_ID))"
)

if [ -n "${N_VARIATES:-}" ]; then
    PY_ARGS+=(--n-variates "$N_VARIATES")
fi
if [ -n "${VARIATE_INDICES:-}" ]; then
    PY_ARGS+=(--variate-indices "$VARIATE_INDICES")
fi
if [ "${SMOKE_TEST:-0}" = "1" ]; then
    PY_ARGS+=(--smoke-test)
fi

echo "Project root: $PROJECT_ROOT"
echo "Store: $STORE"
echo "Architecture: $ARCH"
echo "Dataset: $DATASET"
echo "Study: $STUDY_NAME"
echo "Storage: $OPTUNA_STORAGE"
echo "Command: python -u ${PY_ARGS[*]}"

python -u "${PY_ARGS[@]}"

echo "=========================================="
echo "Completed: $(date)"
echo "=========================================="
