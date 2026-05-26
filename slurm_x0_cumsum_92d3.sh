#!/bin/bash
# =============================================================================
# Gaussian DiT diffusion with x0_cumsum (softmax + reverse CDF mold) for 92d3.
#
# USAGE (from repo root on Killarney login node, preferably $SCRATCH/ts-sandbox):
#   ./slurm_x0_cumsum_92d3.sh --dataset ETTh1
#   ./slurm_x0_cumsum_92d3.sh --dataset ETTh2
#   ./slurm_x0_cumsum_92d3.sh --dataset exchange_rate
#   ./slurm_x0_cumsum_92d3.sh --dataset ETTh1 --smoke-test
#   ./slurm_x0_cumsum_92d3.sh --dataset ETTh1 --resume
#   ./slurm_x0_cumsum_92d3.sh --dataset ETTh1 --run-stem 05-25-4211-x0cumsum-etth1
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="ETTh1"
N_VARIATES=""
SEED="42"
FRESH=0
RESUME=1
RUN_STEM=""
SMOKE=0
VARS_TO_PLOT=3
ENSEMBLE=1
WALL="${WALL:-}"
EVAL_SAMPLER="ddim"
WANDB_PROJECT="${WANDB_PROJECT:-ts-sandbox-x0-cumsum-92d3}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            case "$2" in
                exchange) DATASET="exchange_rate" ;;
                *) DATASET="$2" ;;
            esac
            shift 2
            ;;
        --n-variates) N_VARIATES="$2"; shift 2 ;;
        --walltime|--time) WALL="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --fresh) FRESH=1; RESUME=0; shift ;;
        --resume) RESUME=1; shift ;;
        --no-resume) RESUME=0; shift ;;
        --run-stem) RUN_STEM="$2"; shift 2 ;;
        --smoke-test|--smoke) SMOKE=1; shift ;;
        --vars) VARS_TO_PLOT="$2"; shift 2 ;;
        --ensemble) ENSEMBLE="$2"; shift 2 ;;
        --eval-sampler) EVAL_SAMPLER="$2"; shift 2 ;;
        --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    if [[ "$SMOKE" -eq 1 ]]; then
        WALL="0:30:00"
        MEM="24G"
        CPUS=4
        JOB_NAME="x0cumsum-smoke"
    else
        [[ -z "$WALL" ]] && WALL="1-00:00:00"
        MEM="60G"
        CPUS=8
        JOB_NAME="x0cumsum-${DATASET,,}"
    fi

    echo "Submitting ${JOB_NAME} (${DATASET}) wall=${WALL} on Killarney L40S..."
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
        "$SCRIPT_DIR/slurm_x0_cumsum_92d3.sh" \
        --dataset "$DATASET" --seed "$SEED" \
        --vars "$VARS_TO_PLOT" --ensemble "$ENSEMBLE" \
        --eval-sampler "$EVAL_SAMPLER" \
        $([[ -n "$N_VARIATES" ]] && echo --n-variates "$N_VARIATES") \
        $([[ -n "$WALL" && "$SMOKE" -eq 0 ]] && echo --walltime "$WALL") \
        $([[ "$FRESH" -eq 1 ]] && echo --fresh) \
        $([[ "$RESUME" -eq 1 ]] && echo --resume) \
        $([[ -n "$RUN_STEM" ]] && echo --run-stem "$RUN_STEM") \
        $([[ "$SMOKE" -eq 1 ]] && echo --smoke-test) \
        --wandb-project "$WANDB_PROJECT"
    exit 0
fi

cd "$SLURM_SUBMIT_DIR"
# BASH_SOURCE points at Slurm spool copy; repo paths must use submit dir.
SCRIPT_DIR="$SLURM_SUBMIT_DIR"
PROJECT_ROOT="$SLURM_SUBMIT_DIR"
if [[ ! -f "$PROJECT_ROOT/models/diffusion_tsf/train_multivariate_pipeline.py" ]]; then
    echo "ERROR: submit from the ts-sandbox repo root." >&2
    exit 1
fi
if [[ "$PROJECT_ROOT" == /home/* ]]; then
    echo "ERROR: Killarney GPU jobs should run from a scratch/project checkout, not /home." >&2
    exit 1
fi

# shellcheck source=slurm/lib_92d3_resume.sh
source "$SCRIPT_DIR/slurm/lib_92d3_resume.sh"
mkdir -p ./results/logs ./results/ckpts ./results/datasets
resolve_92d3_run_dirs ckpts_flat "x0cumsum-${DATASET,,}" "${DATASET,,}" 4
mkdir -p "$(dirname "$LOG_FILE")" "$CKPT_DIR" "$DATA_DIR"
exec >>"$LOG_FILE" 2>&1

echo "=========================================="
echo "Job: $SLURM_JOB_NAME  ID: $SLURM_JOB_ID  Node: ${SLURMD_NODENAME:-unknown}"
echo "Repo: $PROJECT_ROOT"
echo "Checkpoints: $CKPT_DIR"
echo "Results: $DATA_DIR"
echo "GPU: $(nvidia-smi -L 2>/dev/null | head -1 || echo none)"
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
# Training stack only (no matplotlib — fonttools wheel can hit transient CVMFS I/O errors).
pip install --no-index \
    'torch==2.11.0+computecanada' numpy pandas scipy scikit-learn tqdm optuna wandb einops \
    -q
pip install --no-index reformer_pytorch -q 2>/dev/null \
    || pip install --no-index reformer-pytorch -q 2>/dev/null \
    || pip install reformer-pytorch -q 2>/dev/null \
    || echo "[setup] reformer-pytorch not installed (OK unless using Reformer attention)"
VIZ_WHEELS_OK=0
for _try in 1 2 3; do
    if pip install --no-index matplotlib -q; then
        VIZ_WHEELS_OK=1
        break
    fi
    echo "[setup] matplotlib install attempt ${_try}/3 failed (CVMFS I/O?); sleeping 30s..."
    sleep 30
done
if [[ "$VIZ_WHEELS_OK" -ne 1 ]]; then
    echo "[setup] WARNING: matplotlib not installed; training will run, viz may be skipped."
fi
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA required; check torch/modules"
print("torch", torch.__version__, "gpu", torch.cuda.get_device_name(0))
PY

export PYTHONUNBUFFERED=1
export WANDB_PROJECT

TRAIN_ARGS=(
    --mode full
    --dataset "$DATASET"
    --checkpoint-dir "$CKPT_DIR"
    --results-dir "$DATA_DIR"
    --synth-cache-dir "$DATA_DIR"
    --seed "$SEED"
    --model-type dit
    --prediction-mode x0_cumsum
    --eval-sampler "$EVAL_SAMPLER"
    --pretrained-diff-ckpt "$CKPT_DIR/diff_hp_best.pt"
)
if [[ -n "$N_VARIATES" ]]; then
    TRAIN_ARGS+=(--n-variates "$N_VARIATES")
fi
if [[ "$FRESH" -eq 1 ]]; then
    TRAIN_ARGS+=(--fresh)
elif [[ "$RESUME" -eq 1 ]]; then
    TRAIN_ARGS+=(--resume)
fi
if [[ "$SMOKE" -eq 1 ]]; then
    TRAIN_ARGS+=(--smoke-test)
fi
if [[ -n "${WANDB_API_KEY:-}" ]] && [[ "$WANDB_API_KEY" =~ ^[A-Za-z0-9_]+$ ]]; then
    TRAIN_ARGS+=(--wandb --wandb-project "$WANDB_PROJECT")
elif [[ -n "${WANDB_API_KEY:-}" ]]; then
    echo "[wandb] WANDB_API_KEY has invalid characters; training without wandb."
    unset WANDB_API_KEY
else
    echo "[wandb] WANDB_API_KEY not set; training will run without wandb."
fi

echo "[train] Gaussian DiT full training/eval with prediction_mode=x0_cumsum..."
python -u -m models.diffusion_tsf.train_multivariate_pipeline "${TRAIN_ARGS[@]}"

if [[ "$VIZ_WHEELS_OK" -eq 1 ]] || python -c "import matplotlib" 2>/dev/null; then
    echo "[viz] Rendering comparison..."
    python -u -m models.diffusion_tsf.visualize_comparison \
        --checkpoint-dir "$CKPT_DIR" \
        --output-dir "$DATA_DIR" \
        --dataset "$DATASET" \
        --num-samples 3 \
        --vars "$VARS_TO_PLOT" \
        --ensemble "$ENSEMBLE" \
        --num-extra-windows 2 \
        --diffusion-type gaussian \
        --model-type dit \
        --prediction-mode x0_cumsum \
        --diffusion-sampler "$EVAL_SAMPLER"
else
    echo "[viz] Skipped (matplotlib unavailable after wheel install retries)."
fi

if [[ -n "${WANDB_API_KEY:-}" ]] && [[ "$WANDB_API_KEY" =~ ^[A-Za-z0-9_]+$ ]]; then
    echo "[wandb] Uploading combined log artifact..."
    python - <<PY
import os
import wandb

run = wandb.init(
    project=os.environ.get("WANDB_PROJECT", "ts-sandbox-x0-cumsum-92d3"),
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
