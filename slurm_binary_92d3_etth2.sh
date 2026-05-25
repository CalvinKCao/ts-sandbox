#!/bin/bash
# =============================================================================
# Binary CDF diffusion on ETTh2 from the 92d3 pipeline.
#
# USAGE (from repo root on Killarney login node, preferably $SCRATCH/ts-sandbox):
#   ./slurm_binary_92d3_etth2.sh
#   ./slurm_binary_92d3_etth2.sh --smoke-test
#   ./slurm_binary_92d3_etth2.sh --fresh --seed 23
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="ETTh2"
SEED="42"
FRESH=0
SMOKE=0
VARS_TO_PLOT=3
ENSEMBLE=3
WANDB_PROJECT="${WANDB_PROJECT:-ts-sandbox-binary-92d3}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --fresh) FRESH=1; shift ;;
        --smoke-test|--smoke) SMOKE=1; shift ;;
        --vars) VARS_TO_PLOT="$2"; shift 2 ;;
        --ensemble) ENSEMBLE="$2"; shift 2 ;;
        --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    if [[ "$SMOKE" -eq 1 ]]; then
        WALL="0:30:00"
        MEM="24G"
        CPUS=4
        JOB_NAME="binary92d3-smoke"
    else
        WALL="2-00:00:00"
        MEM="60G"
        CPUS=8
        JOB_NAME="binary92d3-etth2"
    fi

    echo "Submitting ${JOB_NAME} on Killarney L40S..."
    sbatch \
        --job-name="$JOB_NAME" \
        --account=aip-boyuwang \
        --time="$WALL" \
        --nodes=1 \
        --gres=gpu:l40s:1 \
        --cpus-per-task="$CPUS" \
        --mem="$MEM" \
        --output=/dev/null \
        --error=/dev/null \
        --mail-type=END,FAIL \
        --mail-user=ccao87@uwo.ca \
        "$SCRIPT_DIR/slurm_binary_92d3_etth2.sh" \
        --dataset "$DATASET" --seed "$SEED" --vars "$VARS_TO_PLOT" --ensemble "$ENSEMBLE" \
        $([[ "$FRESH" -eq 1 ]] && echo --fresh) \
        $([[ "$SMOKE" -eq 1 ]] && echo --smoke-test) \
        --wandb-project "$WANDB_PROJECT"
    exit 0
fi

cd "$SLURM_SUBMIT_DIR"
PROJECT_ROOT="$SLURM_SUBMIT_DIR"
if [[ ! -f "$PROJECT_ROOT/models/diffusion_tsf/train_multivariate_pipeline.py" ]]; then
    echo "ERROR: submit from the ts-sandbox repo root." >&2
    exit 1
fi
if [[ "$PROJECT_ROOT" == /home/* ]]; then
    echo "ERROR: Killarney GPU jobs should run from a scratch/project checkout, not /home." >&2
    exit 1
fi

RUN_STEM="$(date +%m-%d)-${SLURM_JOB_ID: -3}-binary-92d3-${DATASET,,}"
RUN_DIR="./results/${RUN_STEM}"
LOG_DIR="${RUN_DIR}/logs"
CKPT_DIR="${RUN_DIR}/ckpts"
DATA_DIR="${RUN_DIR}/datasets"
mkdir -p "$LOG_DIR" "$CKPT_DIR" "$DATA_DIR"
LOG_FILE="${LOG_DIR}/${RUN_STEM}.log"
exec >>"$LOG_FILE" 2>&1

echo "=========================================="
echo "Job: $SLURM_JOB_NAME  ID: $SLURM_JOB_ID  Node: ${SLURMD_NODENAME:-unknown}"
echo "Repo: $PROJECT_ROOT"
echo "Run:  $RUN_DIR"
echo "GPU:  $(nvidia-smi -L 2>/dev/null | head -1 || echo none)"
echo "Started: $(date)"
echo "=========================================="

module purge || true
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9

echo "[setup] Building venv on \$SLURM_TMPDIR..."
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index 'torch==2.11.0+computecanada' numpy pandas scipy scikit-learn tqdm matplotlib optuna wandb einops -q
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA required; check torch wheel/modules"
print("torch", torch.__version__, "gpu", torch.cuda.get_device_name(0))
PY

export PYTHONUNBUFFERED=1
export WANDB_PROJECT

TRAIN_ARGS=(
    --mode full
    --dataset "$DATASET"
    --n-variates 7
    --binary-diffusion
    --checkpoint-dir "$CKPT_DIR"
    --results-dir "$DATA_DIR"
    --synth-cache-dir "$DATA_DIR"
    --seed "$SEED"
)
if [[ "$FRESH" -eq 1 ]]; then
    TRAIN_ARGS+=(--fresh)
fi
if [[ "$SMOKE" -eq 1 ]]; then
    TRAIN_ARGS+=(--smoke-test)
fi
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    TRAIN_ARGS+=(--wandb --wandb-project "$WANDB_PROJECT")
else
    echo "[wandb] WANDB_API_KEY not set; training will run without wandb."
fi

echo "[train] Binary full training/eval..."
python -u -m models.diffusion_tsf.train_multivariate_pipeline "${TRAIN_ARGS[@]}"

echo "[viz] Rendering comparison..."
python -u -m models.diffusion_tsf.visualize_comparison \
    --checkpoint-dir "$CKPT_DIR" \
    --output-dir "$DATA_DIR" \
    --dataset "$DATASET" \
    --num-samples 3 \
    --vars "$VARS_TO_PLOT" \
    --ensemble "$ENSEMBLE" \
    --num-extra-windows 2 \
    --diffusion-type binary

if [[ -n "${WANDB_API_KEY:-}" ]]; then
    echo "[wandb] Uploading combined log artifact..."
    python - <<PY
import os
import wandb

run = wandb.init(
    project=os.environ.get("WANDB_PROJECT", "ts-sandbox-binary-92d3"),
    name="${RUN_STEM}-logs",
    job_type="slurm-log-upload",
    reinit=True,
)
artifact = wandb.Artifact("${RUN_STEM}-logs", type="logs")
artifact.add_file("${LOG_FILE}")
run.log_artifact(artifact)
run.finish()
PY
fi

echo "=========================================="
echo "Done: $(date)"
echo "Log: $LOG_FILE"
echo "Checkpoints: $CKPT_DIR"
echo "Results/Viz: $DATA_DIR"
echo "=========================================="
