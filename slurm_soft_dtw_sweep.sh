#!/bin/bash
# =============================================================================
# Killarney L40S sweep for decoded 1D MSE + Soft-DTW diffusion losses.
#
# USAGE (from repo root on the Killarney login node):
#   ./slurm_soft_dtw_sweep.sh
#   ./slurm_soft_dtw_sweep.sh --smoke-test
#   ./slurm_soft_dtw_sweep.sh ETTh1 weather
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "${SLURM_JOB_ID:-}" ]; then
    IS_SMOKE=0
    if [ "${1:-}" = "--smoke-test" ]; then
        IS_SMOKE=1
        shift
    fi

    if [ "$#" -gt 0 ]; then
        DATASETS=("$@")
    else
        DATASETS=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "illness" "exchange_rate" "weather")
    fi

    walltime_for_dataset() {
        local ds="$1"
        if [ "$IS_SMOKE" -eq 1 ]; then
            printf '%s' "00:30:00"
        elif [ "$ds" = "weather" ]; then
            printf '%s' "18:00:00"
        else
            printf '%s' "12:00:00"
        fi
    }

    mkdir -p "$SCRIPT_DIR/results/bootstrap"
    for ds in "${DATASETS[@]}"; do
        case "$ds" in
            ETTh1|ETTh2|ETTm1|ETTm2|illness|exchange_rate|weather) ;;
            *)
                echo "ERROR: unsupported dataset '$ds' for this sweep"
                echo "Allowed: ETTh1 ETTh2 ETTm1 ETTm2 illness exchange_rate weather"
                exit 1
                ;;
        esac

        tag="${ds//_/-}"
        job_name="softdtw-${tag}"
        [ "$IS_SMOKE" -eq 1 ] && job_name="${job_name}-smoke"
        echo "Submitting $job_name ..."
        sbatch \
            --job-name="$job_name" \
            --account=aip-boyuwang \
            --time="$(walltime_for_dataset "$ds")" \
            --nodes=1 \
            --gres=gpu:l40s:1 \
            --cpus-per-task=8 \
            --mem=50G \
            --chdir="$SCRIPT_DIR" \
            --output=/dev/null \
            --error=/dev/null \
            --mail-type=END,FAIL \
            --mail-user=ccao87@uwo.ca \
            --export="ALL,DATASET=$ds,SMOKE=$IS_SMOKE" \
            "$SCRIPT_DIR/slurm_soft_dtw_sweep.sh"
    done
    echo "All Soft-DTW sweep jobs submitted."
    exit 0
fi

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

RUN_STEM="$(date +%m-%d)-${SLURM_JOB_ID}-${SLURM_JOB_NAME}"
RUN_ROOT="$SLURM_SUBMIT_DIR/results/$RUN_STEM"
RUN_LOG_DIR="$RUN_ROOT/logs"
RUN_CKPT_DIR="$RUN_ROOT/ckpts"
RUN_DATA_DIR="$RUN_ROOT/datasets"
mkdir -p "$RUN_LOG_DIR" "$RUN_CKPT_DIR" "$RUN_DATA_DIR"

LOG="$RUN_LOG_DIR/${RUN_STEM}.log"
export WANDB_NAME="$RUN_STEM"
export WANDB_DIR="$RUN_LOG_DIR/wandb"
mkdir -p "$WANDB_DIR"
exec >>"$LOG" 2>&1

echo "=========================================="
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "Dataset:  $DATASET"
echo "GPU:      $(nvidia-smi -L 2>/dev/null | head -1 || echo unknown)"
echo "Started:  $(date)"
echo "=========================================="

module purge || true
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9

if [ "${TS_SANDBOX_PROJECT_ROOT_SUBMIT_DIR:-}" = "1" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
elif [ -d "${SCRATCH:-}/ts-sandbox" ]; then
    PROJECT_ROOT="$SCRATCH/ts-sandbox"
elif [ -d "$HOME/ts-sandbox" ]; then
    PROJECT_ROOT="$HOME/ts-sandbox"
else
    echo "ERROR: clone repo to \$SCRATCH/ts-sandbox or submit from the repo root"
    exit 1
fi
export PROJECT_ROOT

if [ -z "${PROJECT:-}" ] && [ -d "$HOME/projects" ]; then
    shopt -s nullglob
    _matches=("$HOME"/projects/aip-* "$HOME"/projects/def-*)
    shopt -u nullglob
    if [ "${#_matches[@]}" -gt 0 ]; then
        export PROJECT=$(readlink -f "${_matches[0]}")
    fi
fi

echo "[setup] Building venv on \$SLURM_TMPDIR ..."
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index 'torch==2.11.0+computecanada' numpy scipy pandas scikit-learn optuna wandb tqdm matplotlib einops -q
pip install reformer-pytorch pysdtw -q

python - <<'PY'
import torch
import pysdtw

assert torch.cuda.is_available(), "CUDA required for this Slurm job"
print("torch", torch.__version__, "gpu", torch.cuda.get_device_name(0))
print("pysdtw", pysdtw.SoftDTW)
PY

cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export WANDB_PROJECT="${WANDB_PROJECT:-diffusion-tsf-softdtw}"
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "[wandb] WANDB_API_KEY not set; using offline mode."
    export WANDB_MODE=offline
fi

SYNTH_CACHE_ROOT="$PROJECT_ROOT/synth_data"
mkdir -p "$SYNTH_CACHE_ROOT"
if [ ! -e "$RUN_DATA_DIR/repo" ]; then
    ln -s "$PROJECT_ROOT/datasets" "$RUN_DATA_DIR/repo"
fi

SMOKE_FLAG=()
if [ "$SMOKE" -eq 1 ]; then
    SMOKE_FLAG=(--smoke-test)
fi

TARGET_DIM=7
if [ "$DATASET" = "weather" ]; then TARGET_DIM=21; fi
if [ "$DATASET" = "exchange_rate" ]; then TARGET_DIM=8; fi

COMMON_ARGS=(
    --dataset "$DATASET"
    --n-variates "$TARGET_DIM"
    --checkpoint-dir "$RUN_CKPT_DIR"
    --results-dir "$RUN_DATA_DIR"
    --synth-cache-dir "$SYNTH_CACHE_ROOT"
    --model-type dit
    --subset-id "softdtw-dit"
    --guidance-penalty-weight 0
    --fresh
    --wandb
    "${SMOKE_FLAG[@]}"
)

echo "Running Phase 1 (pretrain)..."
python -u -m models.diffusion_tsf.train_multivariate_pipeline \
    --mode pretrain \
    "${COMMON_ARGS[@]}"

echo "Running Phase 2 (finetune/eval)..."
python -u -m models.diffusion_tsf.train_multivariate_pipeline \
    --mode finetune \
    "${COMMON_ARGS[@]}"

upload_logs_to_wandb() {
    local run_id_file="$RUN_CKPT_DIR/wandb_run_id.txt"
    if [ ! -f "$run_id_file" ] || [ "${WANDB_MODE:-online}" = "offline" ]; then
        return 0
    fi
    python - "$run_id_file" "$LOG" <<'PY' || true
import os
import sys
import wandb

run_id_path, log_path = sys.argv[1:3]
with open(run_id_path) as f:
    run_id = f.read().strip()
if not run_id or not os.path.isfile(log_path):
    raise SystemExit(0)
run = wandb.init(project=os.environ.get("WANDB_PROJECT", "diffusion-tsf-softdtw"),
                 id=run_id, resume="allow", reinit=True)
artifact = wandb.Artifact(f"slurm-log-{os.environ.get('SLURM_JOB_ID', 'unknown')}", type="logs")
artifact.add_file(log_path)
run.log_artifact(artifact)
run.finish()
PY
}
upload_logs_to_wandb

echo "=========================================="
echo "Job complete: $(date)"
echo "Results: $RUN_ROOT"
echo "=========================================="
