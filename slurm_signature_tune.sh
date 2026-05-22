#!/bin/bash
# iTransformer MSE + truncated signature Optuna tuning on Killarney (parallel array).
#
# Usage from repo root on Killarney login node:
#   bash slurm_signature_tune.sh
#   WORKERS=12 MAX_CONCURRENT=12 bash slurm_signature_tune.sh
#   SMOKE_TEST=1 bash slurm_signature_tune.sh
#
# Finalize (full test eval on best trial + MSE baseline) is auto-submitted with
# --dependency=afterok on the tuning array. Set SKIP_FINALIZE=1 to disable.
#
# Array task id maps to dataset (ETTh1, ETTh2, exchange_rate) round-robin.
# Workers sharing a dataset join the same Optuna study via sqlite storage.

#SBATCH --job-name=sig-mse-tune
#SBATCH --account=aip-boyuwang
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=10:00:00
#SBATCH --array=1-12%12
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ccao87@uwo.ca

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "${SLURM_JOB_ID:-}" ]; then
    WORKERS="${WORKERS:-12}"
    MAX_CONCURRENT="${MAX_CONCURRENT:-$WORKERS}"
    mkdir -p "$SCRIPT_DIR/results/signature_tune/logs"
    echo "Submitting signature+MSE Optuna array: workers=$WORKERS max_concurrent=$MAX_CONCURRENT"
    TUNE_SBATCH=(
        --parsable
        --array="1-${WORKERS}%${MAX_CONCURRENT}"
        --output=/dev/null
        --error=/dev/null
    )
    if [ "${SMOKE_TEST:-0}" = "1" ]; then
        TUNE_SBATCH+=(--time=0:30:00 --export=ALL,SMOKE_TEST=1)
    fi
    ARRAY_JOB_ID="$(sbatch "${TUNE_SBATCH[@]}" "$SCRIPT_DIR/slurm_signature_tune.sh")"
    echo "Tuning array job: $ARRAY_JOB_ID"

    FINALIZE_JOB_ID=""
    if [ "${SKIP_FINALIZE:-0}" != "1" ]; then
        EXPORT_LIST="ALL,ARRAY_JOB_ID=${ARRAY_JOB_ID}"
        if [ "${SMOKE_TEST:-0}" = "1" ]; then
            EXPORT_LIST="${EXPORT_LIST},SMOKE_TEST=1"
        fi
        FINALIZE_SBATCH=(
            --parsable
            --dependency="afterok:${ARRAY_JOB_ID}"
            --export="${EXPORT_LIST}"
            --output=/dev/null
            --error=/dev/null
        )
        if [ "${SMOKE_TEST:-0}" = "1" ]; then
            FINALIZE_SBATCH+=(--time=0:20:00)
        fi
        FINALIZE_JOB_ID="$(sbatch "${FINALIZE_SBATCH[@]}" "$SCRIPT_DIR/slurm_signature_finalize.sh")"
        echo "Finalize array job: $FINALIZE_JOB_ID  [afterok:${ARRAY_JOB_ID}]"
    else
        echo "SKIP_FINALIZE=1: test eval not submitted"
    fi

    MANIFEST="$SCRIPT_DIR/results/signature_tune/last_submission.json"
    python - <<PY
import json
import os
from datetime import datetime, timezone
from pathlib import Path

manifest = {
    "array_job_id": int("${ARRAY_JOB_ID}"),
    "finalize_job_id": int("${FINALIZE_JOB_ID}") if "${FINALIZE_JOB_ID}" else None,
    "workers": int("${WORKERS}"),
    "submitted_at": datetime.now(timezone.utc).isoformat(),
    "study_name_pattern": "signature_mse_{dataset}_job${ARRAY_JOB_ID}",
    "datasets": ["ETTh1", "ETTh2", "exchange_rate"],
    "dependency": "afterok:${ARRAY_JOB_ID}",
}
path = Path("${MANIFEST}")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Wrote {path}")
if manifest["finalize_job_id"]:
    print(f"Monitor: squeue -u $USER")
    print(f"Cancel tune+finalize: scancel {manifest['array_job_id']} {manifest['finalize_job_id']}")
PY
    exit 0
fi

ARRAY_ID="${SLURM_ARRAY_TASK_ID:-1}"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$SCRIPT_DIR}"
RUN_ROOT="$SUBMIT_DIR/results/signature_tune"
LOG_DIR="$RUN_ROOT/logs"
CKPT_DIR="$RUN_ROOT/ckpts"
mkdir -p "$LOG_DIR" "$CKPT_DIR"

DATASETS=(ETTh1 ETTh2 exchange_rate)
DATASET="${DATASETS[$(( (ARRAY_ID - 1) % ${#DATASETS[@]} ))]}"

STEM="$(date +%m-%d)-${SLURM_JOB_ID: -3}-a${ARRAY_ID}-${DATASET}-sigmse"
LOG_FILE="$LOG_DIR/${STEM}.log"
exec >>"$LOG_FILE" 2>&1

echo "=========================================="
echo "Signature+MSE Optuna worker"
echo "Job: $SLURM_JOB_ID  Array: $ARRAY_ID  Node: ${SLURMD_NODENAME:-unknown}"
echo "Dataset: $DATASET"
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
STORE="${STORE:-$STORE_BASE/$USER/ts-sandbox-signature}"
mkdir -p "$STORE"

echo "[setup] Building node-local venv on $SLURM_TMPDIR"
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index 'torch==2.11.0+computecanada' numpy pandas scipy scikit-learn tqdm matplotlib einops -q
pip install --no-index optuna -q || pip install optuna -q
pip install reformer-pytorch -q
pip install signatory --no-build-isolation -q

python - <<'PY'
import torch
import signatory
from reformer_pytorch import LSHSelfAttention
assert torch.cuda.is_available(), "CUDA is required for this Slurm job"
print("torch", torch.__version__, "signatory ok", "gpu", torch.cuda.get_device_name(0))
PY

cd "$PROJECT_ROOT"

N_TRIALS_PER_WORKER="${N_TRIALS_PER_WORKER:-2}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-8}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-200}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-80}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-2}"
LOOKBACK_LENGTH="${LOOKBACK_LENGTH:-96}"
FORECAST_LENGTH="${FORECAST_LENGTH:-96}"
LOOKBACK_OVERLAP="${LOOKBACK_OVERLAP:-8}"
OPTUNA_STORAGE="${OPTUNA_STORAGE:-sqlite:///$RUN_ROOT/signature_tuning.db}"
RESULTS_DIR="${RESULTS_DIR:-$RUN_ROOT}"

# One Optuna study per dataset per sbatch submission; array workers join via load_if_exists=True.
if [ "${RESUME_STUDY:-0}" = "1" ]; then
    STUDY_NAME="${RESUME_STUDY_NAME:-signature_mse_${DATASET}_v1}"
else
    STUDY_NAME="${STUDY_NAME:-signature_mse_${DATASET}_job${SLURM_ARRAY_JOB_ID:-local}}"
fi

PY_ARGS=(
    -m models.diffusion_tsf.train_signature_tuning
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
    --results-dir "$RESULTS_DIR"
    --storage "$OPTUNA_STORAGE"
    --study-name "$STUDY_NAME"
    --seed "$((42 + ARRAY_ID))"
)

if [ "$DATASET" = "exchange_rate" ]; then
    PY_ARGS+=(--n-variates 8)
fi

if [ "${SMOKE_TEST:-0}" = "1" ]; then
    PY_ARGS+=(--smoke-test)
fi

echo "Project root: $PROJECT_ROOT"
echo "Store: $STORE"
echo "Study: $STUDY_NAME"
echo "Storage: $OPTUNA_STORAGE"
echo "Command: python -u ${PY_ARGS[*]}"

python -u "${PY_ARGS[@]}"

echo "=========================================="
echo "Completed: $(date)"
echo "=========================================="
