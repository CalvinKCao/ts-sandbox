#!/bin/bash
# Finalize signature diffusion studies: full held-out test eval + JSON/TXT report.
#
# Normally submitted automatically by slurm_signature_diffusion_tune.sh with afterok dependency.
# Manual re-run:
#   ARRAY_JOB_ID=3712500 bash slurm_signature_diffusion_finalize.sh

#SBATCH --job-name=sig-diff-final
#SBATCH --account=aip-boyuwang
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=1:00:00
#SBATCH --array=1-3%3
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ccao87@uwo.ca

set -euo pipefail

SCRIPT_PATH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$SCRIPT_PATH_DIR}"
if [ ! -f "$SCRIPT_DIR/slurm_signature_diffusion_finalize.sh" ]; then
    SCRIPT_DIR="$SCRIPT_PATH_DIR"
fi
VENV_HELPER="$SCRIPT_DIR/cluster/setup_signature_cluster_venv.sh"
DATASETS=(ETTh1 ETTh2 exchange_rate)

if [ -z "${SLURM_JOB_ID:-}" ]; then
    MANIFEST="${MANIFEST:-$SCRIPT_DIR/results/signature_diffusion/last_submission.json}"
    if [ -z "${ARRAY_JOB_ID:-}" ] && [ -f "$MANIFEST" ]; then
        ARRAY_JOB_ID="$(python - <<PY
import json
from pathlib import Path
data = json.loads(Path("$MANIFEST").read_text())
print(data["array_job_id"])
PY
)"
        echo "Using ARRAY_JOB_ID=$ARRAY_JOB_ID from $MANIFEST"
    fi
    if [ -z "${ARRAY_JOB_ID:-}" ]; then
        echo "ERROR: set ARRAY_JOB_ID=<tuning array job id> or run slurm_signature_diffusion_tune.sh first."
        exit 1
    fi
    mkdir -p "$SCRIPT_DIR/results/signature_diffusion/logs"
    EXPORT_LIST="ALL,ARRAY_JOB_ID=${ARRAY_JOB_ID}"
    SBATCH_ARGS=(--parsable --export="${EXPORT_LIST}")
    if [ "${SMOKE_TEST:-0}" = "1" ]; then
        EXPORT_LIST="${EXPORT_LIST},SMOKE_TEST=1"
        SBATCH_ARGS=(--parsable --export="${EXPORT_LIST}" --time=0:20:00)
    fi
    FINALIZE_JOB_ID="$(sbatch "${SBATCH_ARGS[@]}" "$SCRIPT_DIR/slurm_signature_diffusion_finalize.sh")"
    echo "Submitted finalize array: $FINALIZE_JOB_ID"
    exit 0
fi

ARRAY_ID="${SLURM_ARRAY_TASK_ID:-1}"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$SCRIPT_DIR}"
RUN_ROOT="$SUBMIT_DIR/results/signature_diffusion"
LOG_DIR="$RUN_ROOT/logs"
CKPT_DIR="$RUN_ROOT/ckpts"
mkdir -p "$LOG_DIR" "$CKPT_DIR"

DATASET="${DATASETS[$(( ARRAY_ID - 1 ))]}"
STEM="$(date +%m-%d)-${SLURM_JOB_ID: -3}-fin${ARRAY_ID}-${DATASET}"
LOG_FILE="$LOG_DIR/${STEM}.log"
exec >>"$LOG_FILE" 2>&1

echo "=========================================="
echo "Signature diffusion finalize (test split)"
echo "Job: $SLURM_JOB_ID  Array: $ARRAY_ID  Dataset: $DATASET"
echo "ARRAY_JOB_ID (tuning): ${ARRAY_JOB_ID:?set ARRAY_JOB_ID}"
echo "Started: $(date)"
echo "=========================================="

if [ ! -f "$VENV_HELPER" ]; then
    echo "ERROR: missing helper: $VENV_HELPER"
    echo "Submit from the repo checkout, e.g. cd \$SCRATCH/ts-sandbox && ARRAY_JOB_ID=... bash slurm_signature_diffusion_finalize.sh"
    exit 1
fi
# shellcheck source=cluster/setup_signature_cluster_venv.sh
source "$VENV_HELPER"

module purge || true
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9

if [ -d "${SCRATCH:-}/ts-sandbox" ]; then
    PROJECT_ROOT="$SCRATCH/ts-sandbox"
elif [ -d "$SUBMIT_DIR/models/diffusion_tsf" ]; then
    PROJECT_ROOT="$SUBMIT_DIR"
else
    PROJECT_ROOT="$HOME/ts-sandbox"
fi

cd "$PROJECT_ROOT"

if [ -z "${PROJECT:-}" ] && [ -d "$HOME/projects" ]; then
    shopt -s nullglob
    matches=("$HOME"/projects/aip-* "$HOME"/projects/def-*)
    shopt -u nullglob
    if [ "${#matches[@]}" -gt 0 ]; then
        export PROJECT="$(readlink -f "${matches[0]}")"
    fi
fi
STORE="${STORE:-${PROJECT:-${SCRATCH:-}}/$USER/ts-sandbox-sigdiff}"
mkdir -p "$STORE"

if ! signature_cluster_venv; then
    echo "ERROR: venv setup failed (rebuild: BUILD_SHARED_VENV=1 bash slurm_signature_diffusion_tune.sh)"
    exit 1
fi

STUDY_NAME="signature_diffusion_${DATASET}_job${ARRAY_JOB_ID}"
OPTUNA_STORAGE="${OPTUNA_STORAGE:-sqlite:///$RUN_ROOT/signature_diffusion_tuning.db}"
RESULTS_DIR="${RESULTS_DIR:-$RUN_ROOT}"

PY_ARGS=(
    -m models.diffusion_tsf.train_signature_diffusion_tuning
    --dataset "$DATASET"
    --finalize-only
    --study-name "$STUDY_NAME"
    --storage "$OPTUNA_STORAGE"
    --checkpoint-dir "$CKPT_DIR"
    --results-dir "$RESULTS_DIR"
    --batch-size "${BATCH_SIZE:-32}"
    --epochs "${EPOCHS:-8}"
    --num-workers "${DATALOADER_WORKERS:-2}"
    --max-train-batches "${MAX_TRAIN_BATCHES:-200}"
    --max-val-batches "${MAX_VAL_BATCHES:-80}"
    --seed 42
)

if [ "$DATASET" = "exchange_rate" ]; then
    PY_ARGS+=(--n-variates 8)
fi
if [ "${SMOKE_TEST:-0}" = "1" ]; then
    PY_ARGS+=(--smoke-test)
fi
if [ -n "${MAX_TEST_BATCHES:-}" ]; then
    PY_ARGS+=(--max-test-batches "$MAX_TEST_BATCHES")
fi

echo "Study: $STUDY_NAME"
echo "Test eval: full held-out split (unless --max-test-batches set)"
python -u "${PY_ARGS[@]}"
echo "Done: $(date)"
