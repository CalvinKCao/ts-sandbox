#!/bin/bash
# =============================================================================
# U-Net full-variate — self-resubmitting Slurm script for Killarney
#
# When run from the login node, it picks partition + wall time and sbatch's itself.
# When run inside a Slurm job (SLURM_JOB_ID is set), it runs full-variate training
# (same train_multivariate_pipeline as the old run_unet_fullvar.sh: bf16, H=96, no splitting).
#
# USAGE (from login node, repo root):
#   ./run.sh --smoke-test                     # L40S smoke
#   ./run.sh                                  # L40S full run
#   ./run.sh --dataset electricity
#   ./run.sh --no-wandb                         # metrics stay local only (no wandb.init)
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ===========================================================================
# If NOT inside a Slurm job → submit ourselves with the right resources
# ===========================================================================

if [ -z "$SLURM_JOB_ID" ]; then
    # Slurm captures stdout/stderr here until `exec >>…` inside the job redirects fds.
    mkdir -p "$SCRIPT_DIR/results/bootstrap"
    # Paths are relative to --chdir; %x = job name, %j = Slurm job id.
    SB_OUT='results/bootstrap/%x-%j.out'
    SB_ERR='results/bootstrap/%x-%j.err'

    IS_SMOKE=0
    VARIANT="default"
    WALLTIME="1-00:00:00"
    H_VAL=24
    USE_H100=0
    ARGS=("$@")
    for ((i=0; i<${#ARGS[@]}; i++)); do
        arg="${ARGS[$i]}"
        [ "$arg" = "--smoke-test" ] && IS_SMOKE=1
        if [ "$arg" = "--variant" ] && [ $((i + 1)) -lt ${#ARGS[@]} ]; then
            VARIANT="${ARGS[$((i + 1))]}"
        fi
        if [ "$arg" = "--resume" ] && [ $((i + 1)) -lt ${#ARGS[@]} ]; then
            RESUME="${ARGS[$((i + 1))]}"
        fi
        if [ "$arg" = "--h100" ]; then
            USE_H100=1
        fi
        if [ "$arg" = "--hours" ] && [ $((i + 1)) -lt ${#ARGS[@]} ]; then
            h="${ARGS[$((i + 1))]}"
            H_VAL=$h
            # Convert hours to DD-HH:MM:SS
            days=$(( h / 24 ))
            rem_h=$(( h % 24 ))
            WALLTIME=$(printf "%d-%02d:00:00" $days $rem_h)
        fi
    done

    if [ "$IS_SMOKE" -eq 1 ]; then
        echo "Submitting SMOKE TEST (L40S, 8GB, 15 min) [variant=$VARIANT]..."
        sbatch \
            --job-name="unet-fullvar-${VARIANT}-smoke" \
            --account=aip-boyuwang \
            --time=0:15:00 \
            --nodes=1 \
            --gres=gpu:l40s:1 \
            --cpus-per-task=2 \
            --mem=8G \
            --chdir="$SCRIPT_DIR" \
            --output="$SB_OUT" \
            --error="$SB_ERR" \
            --mail-type=END,FAIL \
            --mail-user=ccao87@uwo.ca \
            "$SCRIPT_DIR/run.sh" "$@"
    elif [ "$USE_H100" -eq 1 ]; then
        # Selection logic for H100 partition based on hours
        PARTITION="gpubase_h100_b2" # up to 24h
        if [ "$H_VAL" -ge 24 ]; then PARTITION="gpubase_h100_b3"; fi # 24h to 3 days
        if [ "$H_VAL" -ge 72 ]; then PARTITION="gpubase_h100_b4"; fi # 3 days to 7 days
        
        echo "Submitting H100 FULL RUN (64GB, $WALLTIME wall, $PARTITION) [variant=$VARIANT]..."
        
        EXPORT_ARGS="ALL"
        if [ -n "$RESUME" ]; then
            EXPORT_ARGS="ALL,RESUME_STEM=$RESUME"
        fi

        sbatch \
            --job-name="unet-fullvar-${VARIANT}-h100" \
            --account=aip-boyuwang \
            --partition="$PARTITION" \
            --gpus-per-node=h100:1 \
            --cpus-per-task=16 \
            --mem=64G \
            --time="$WALLTIME" \
            --chdir="$SCRIPT_DIR" \
            --output="$SB_OUT" \
            --error="$SB_ERR" \
            --mail-type=BEGIN,END,FAIL \
            --mail-user=ccao87@uwo.ca \
            --export="$EXPORT_ARGS" \
            "$SCRIPT_DIR/run.sh" "$@"
    else
        echo "Submitting FULL RUN (L40S, 50GB, $WALLTIME wall) [variant=$VARIANT]..."
        
        # If resuming, pass RESUME_STEM to the job's environment
        EXPORT_ARGS="ALL"
        if [ -n "$RESUME" ]; then
            EXPORT_ARGS="ALL,RESUME_STEM=$RESUME"
        fi

        sbatch \
            --job-name="unet-fullvar-${VARIANT}" \
            --account=aip-boyuwang \
            --time="$WALLTIME" \
            --nodes=1 \
            --gres=gpu:l40s:1 \
            --cpus-per-task=8 \
            --mem=50G \
            --chdir="$SCRIPT_DIR" \
            --output="$SB_OUT" \
            --error="$SB_ERR" \
            --mail-type=BEGIN,END,FAIL \
            --mail-user=ccao87@uwo.ca \
            --export="$EXPORT_ARGS" \
            "$SCRIPT_DIR/run.sh" "$@"
    fi
    exit 0
fi

# ===========================================================================
# We're inside a Slurm job — do the actual work
# ===========================================================================

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
_slug="${SLURM_JOB_NAME}"

# Read RESUME from environment if we passed it via --export
if [ -n "${RESUME_STEM:-}" ]; then
    # Strip leading "results/" if the user accidentally included it
    ALLIANCE_RUN_STEM="${RESUME_STEM#results/}"
    echo "Resuming from existing job directory: $ALLIANCE_RUN_STEM"
else
    ALLIANCE_RUN_STEM="$(date +%m-%d)-${SLURM_JOB_ID}-${_slug}"
fi

RUN_RESULTS_ROOT="$SLURM_SUBMIT_DIR/results/$ALLIANCE_RUN_STEM"
RUN_LOG_DIR="$RUN_RESULTS_ROOT/logs"
RUN_CKPT_DIR="$RUN_RESULTS_ROOT/ckpts"
RUN_DATA_DIR="$RUN_RESULTS_ROOT/datasets"
mkdir -p "$RUN_LOG_DIR" "$RUN_CKPT_DIR" "$RUN_DATA_DIR"
# Use basename so we don't try to create a log file with slashes in the name
LOG_FILENAME="$(basename "$ALLIANCE_RUN_STEM").log"
ALLIANCE_JOB_LOG="$RUN_LOG_DIR/$LOG_FILENAME"
touch "$ALLIANCE_JOB_LOG"
exec >>"$ALLIANCE_JOB_LOG" 2>&1

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi -L 2>/dev/null | head -1 || echo 'unknown')"
echo "Started: $(date '+%m-%d %H:%M:%S')"
echo "Log: $ALLIANCE_JOB_LOG"
echo "=========================================="

CKPT_ROOT="$RUN_CKPT_DIR"
RES_ROOT="$RUN_DATA_DIR"
export WANDB_DIR="$RUN_LOG_DIR/wandb"
mkdir -p "$WANDB_DIR"

# ---- Environment ----

module purge || true
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9

if [ -d "$SCRATCH/ts-sandbox" ]; then
    export PROJECT_ROOT="$SCRATCH/ts-sandbox"
elif [ -d "$HOME/ts-sandbox" ]; then
    export PROJECT_ROOT="$HOME/ts-sandbox"
else
    echo "ERROR: ts-sandbox not found in SCRATCH or HOME"
    exit 1
fi

# Auto-detect PROJECT (nullglob — bare ls + pipefail kills the job if globs miss)
if [ -z "${PROJECT:-}" ] && [ -d "$HOME/projects" ]; then
    shopt -s nullglob
    _m=("$HOME"/projects/def-* "$HOME"/projects/aip-*)
    shopt -u nullglob
    if [ "${#_m[@]}" -gt 0 ]; then
        export PROJECT=$(readlink -f "${_m[0]}")
    fi
fi

if [ -z "${PROJECT:-}" ]; then
    echo "ERROR: PROJECT not found"
    exit 1
fi

echo "CKPT_ROOT: $CKPT_ROOT"
echo "RES_ROOT:  $RES_ROOT"

SYNTH_CACHE_ROOT="$PROJECT_ROOT/synth_data"
mkdir -p "$SYNTH_CACHE_ROOT"

# Venv — reuse main pipeline venv if it exists; else a persistent fullvar venv under PROJECT
VENV_PATH="$PROJECT/$USER/diffusion-tsf/venv"
if [ ! -d "$VENV_PATH" ]; then
    VENV_PATH="$PROJECT/$USER/diffusion-tsf-fullvar/venv"
    if [ ! -d "$VENV_PATH" ]; then
        echo "Creating virtual environment at $VENV_PATH ..."
        mkdir -p "$(dirname "$VENV_PATH")"
        python -m venv "$VENV_PATH"
        export PATH="$VENV_PATH/bin:$PATH"
        pip install --upgrade pip
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
        pip install numpy pandas scipy scikit-learn optuna wandb tqdm matplotlib einops reformer-pytorch
        [ -f "$PROJECT_ROOT/requirements.txt" ] && pip install -r "$PROJECT_ROOT/requirements.txt"
    else
        export PATH="$VENV_PATH/bin:$PATH"
    fi
else
    export PATH="$VENV_PATH/bin:$PATH"
    echo "Reusing existing venv: $VENV_PATH"
fi

if [ ! -e "$RUN_DATA_DIR/repo" ]; then
    ln -s "$PROJECT_ROOT/datasets" "$RUN_DATA_DIR/repo"
fi

wandb_upload_job_logs() {
    local checkpoint_dir="$1"
    shift
    local run_id_file="${checkpoint_dir}/wandb_run_id.txt"
    if [ ! -f "$run_id_file" ]; then
        echo "[wandb] WARN: no run id file at $run_id_file; skipping log upload."
        return 0
    fi
    local run_id
    run_id="$(tr -d '[:space:]' < "$run_id_file")"
    [ -z "$run_id" ] && echo "[wandb] WARN: empty run id in $run_id_file; skipping." && return 0

    local files=()
    local f
    for f in "$@"; do
        [ -f "$f" ] && files+=("$f")
    done
    [ "${#files[@]}" -eq 0 ] && echo "[wandb] WARN: no log files found to upload." && return 0

    python - "$run_id" "${files[@]}" <<'PY' || true
import os
import sys
import wandb

run_id = sys.argv[1]
files = sys.argv[2:]
project = os.environ.get("WANDB_PROJECT", "diffusion-tsf")
job_id = os.environ.get("SLURM_JOB_ID", "unknown")
job_name = os.environ.get("SLURM_JOB_NAME", "unknown")

run = wandb.init(project=project, id=run_id, resume="must", reinit=True)
artifact = wandb.Artifact(f"slurm-job-logs-{job_id}", type="logs")
artifact.metadata.update({"slurm_job_id": job_id, "slurm_job_name": job_name})
for path in files:
    if os.path.isfile(path):
        artifact.add_file(path)
run.log_artifact(artifact)
run.finish()
print(f"[wandb] Uploaded {len(files)} log file(s) for job {job_id}.")
PY
}

# ---- Cleanup ----

cleanup() {
    trap '' EXIT ERR SIGTERM SIGINT SIGUSR1
    local code=${1:-$?}
    [ "$code" -ne 0 ] && echo "[CLEANUP] $(date '+%m-%d %H:%M:%S') — killing child processes..."
    kill -- -$$ 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT ERR SIGTERM SIGINT SIGUSR1

# ---- Args: checkpoint/results under ./results/, then pass-through (strip --hours) ----

PIPELINE_ARGS=(--checkpoint-dir "$CKPT_ROOT" --results-dir "$RES_ROOT")
while [[ $# -gt 0 ]]; do
    case $1 in
        --hours) shift 2 ;;
        --h100)  shift ;;
        *)       PIPELINE_ARGS+=("$1"); shift ;;
    esac
done

cd "$PROJECT_ROOT"

# ---- Inlined full-variate U-Net driver (former run_unet_fullvar.sh) ----

set -- "${PIPELINE_ARGS[@]}"

# All training-behavior knobs (epochs, trials, patience, U-Net topology,
# image height, AMP, sequence lengths, etc.) live in:
#     models/diffusion_tsf/pipeline_config.py
# Edit that file to change them. This script only handles run-level dispatch.

SEED=42
VARIANT="default"   # label-only; only affects Slurm job name

SMOKE_TEST=""
PRETRAIN_ONLY=""
SINGLE_DATASET=""
RESUME=""
EXTRA_PY_ARGS=""
SUBSET_VARIATE_INDICES=""
SUBSET_ID=""
ENABLE_WANDB=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --smoke-test)     SMOKE_TEST="--smoke-test"; shift ;;
        --dataset)        SINGLE_DATASET="$2"; shift 2 ;;
        --pretrain-only)  PRETRAIN_ONLY=1; shift ;;
        --resume)         RESUME="$2"; shift 2 ;;
        --seed)           SEED="$2"; shift 2 ;;
        --variant)        VARIANT="$2"; shift 2 ;;
        --no-wandb)       ENABLE_WANDB=0; shift ;;
        --wandb)          ENABLE_WANDB=1; shift ;;
        --checkpoint-dir) EXTRA_PY_ARGS="$EXTRA_PY_ARGS --checkpoint-dir $2"; shift 2 ;;
        --results-dir)    EXTRA_PY_ARGS="$EXTRA_PY_ARGS --results-dir $2"; shift 2 ;;
        --hours)          shift 2 ;;   # consumed by login-side submit logic only
        --h100)           shift ;;     # consumed by login-side submit logic only
        *)
            echo "Unknown option: $1"
            echo "Note: training-behavior flags (epochs/trials/patience/topology) were removed."
            echo "      Edit models/diffusion_tsf/pipeline_config.py instead."
            exit 1
            ;;
    esac
done

if [ "$ENABLE_WANDB" -eq 1 ]; then
    EXTRA_PY_ARGS="$EXTRA_PY_ARGS --wandb"
fi

if [ "$ENABLE_WANDB" -eq 1 ] && [ -z "${WANDB_API_KEY:-}" ]; then
    echo "[wandb] ERROR: WANDB_API_KEY is not set."
    echo "[wandb] Export WANDB_API_KEY from https://wandb.ai/authorize and re-submit."
    exit 2
fi
if [ "$ENABLE_WANDB" -eq 1 ]; then
    echo "[wandb] Using WANDB_API_KEY from environment."
fi

if [ -z "$SINGLE_DATASET" ]; then
    SINGLE_DATASET="electricity"
fi

PYTHON="python -m models.diffusion_tsf.train_multivariate_pipeline"
BASE_ARGS="--seed $SEED $SMOKE_TEST $EXTRA_PY_ARGS"
BASE_ARGS="$BASE_ARGS --synth-cache-dir $SYNTH_CACHE_ROOT"
# Keep the same wandb run id as the original job when resuming into an existing results stem.
if [ -n "${RESUME_STEM:-}" ]; then
    BASE_ARGS="$BASE_ARGS --resume"
fi

# Pull lookback/forecast/overlap from pipeline_config so dataset filtering
# uses the same values the pipeline trains with. (Single source of truth.)
read LOOKBACK_LENGTH FORECAST_LENGTH LOOKBACK_OVERLAP < <(python -c "
from models.diffusion_tsf.pipeline_config import (
    LOOKBACK_LENGTH, FORECAST_LENGTH, LOOKBACK_OVERLAP,
)
print(LOOKBACK_LENGTH, FORECAST_LENGTH, LOOKBACK_OVERLAP)
")

echo ""
echo "============================================================"
echo "  U-Net Full-Variate Training (Slurm)"
echo "============================================================"
echo "  Backbone:     U-Net (config in models/diffusion_tsf/pipeline_config.py)"
echo "  Variant tag:  $VARIANT  (Slurm job name only)"
echo "  Dataset:      $SINGLE_DATASET"
echo "  Smoke test:   ${SMOKE_TEST:-no}"
echo "  wandb:        $([ "$ENABLE_WANDB" -eq 1 ] && echo yes || echo no)"
echo "============================================================"
echo ""

if [ ! -d "datasets" ] && [ -d "$RUN_DATA_DIR/repo" ]; then
    echo "[INFO] Symlinking datasets from $RUN_DATA_DIR/repo"
    ln -sfn "$RUN_DATA_DIR/repo" datasets
fi

TRAFFIC_DIR="datasets/traffic"
TRAFFIC_CSV="$TRAFFIC_DIR/traffic.csv"
if [ ! -f "$TRAFFIC_CSV" ]; then
    if [ -f "$TRAFFIC_DIR/traffic_part1.csv" ] && [ -f "$TRAFFIC_DIR/traffic_part2.csv" ]; then
        echo "[INFO] Recombining traffic CSV..."
        head -1 "$TRAFFIC_DIR/traffic_part1.csv" > "$TRAFFIC_CSV"
        tail -n +2 "$TRAFFIC_DIR/traffic_part1.csv" >> "$TRAFFIC_CSV"
        tail -n +2 "$TRAFFIC_DIR/traffic_part2.csv" >> "$TRAFFIC_CSV"
        echo "[INFO] traffic.csv created ($(wc -l < "$TRAFFIC_CSV") rows)"
    fi
fi

declare -A DATASET_DIM

discover_dims() {
    python -c "
import pandas as pd, os

registry = {
    'ETTh1': 'datasets/ETT-small/ETTh1.csv',
    'ETTh2': 'datasets/ETT-small/ETTh2.csv',
    'ETTm1': 'datasets/ETT-small/ETTm1.csv',
    'ETTm2': 'datasets/ETT-small/ETTm2.csv',
    'illness': 'datasets/illness/national_illness.csv',
    'exchange_rate': 'datasets/exchange_rate/exchange_rate.csv',
    'weather': 'datasets/weather/weather.csv',
    'electricity': 'datasets/electricity/electricity.csv',
    'traffic': 'datasets/traffic/traffic.csv',
}

min_rows = $LOOKBACK_LENGTH + $FORECAST_LENGTH + $LOOKBACK_OVERLAP

for name, path in sorted(registry.items()):
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    if len(df) < min_rows:
        import sys
        print(f'[SKIP] {name}: only {len(df)} rows (need {min_rows})', file=sys.stderr)
        continue
    n_cols = sum(1 for c in df.columns if c.lower() != 'date')
    print(f'{name} {n_cols}')
"
}

while IFS=' ' read -r ds ncols; do
    DATASET_DIM[$ds]=$ncols
done < <(discover_dims)

target_dim="${DATASET_DIM[$SINGLE_DATASET]}"
if [ -z "$target_dim" ]; then
    echo "[ERROR] Unknown or missing dataset: $SINGLE_DATASET"
    exit 1
fi
if [ -n "$SUBSET_VARIATE_INDICES" ]; then
    target_dim=$(awk -F',' '{print NF}' <<< "$SUBSET_VARIATE_INDICES")
fi

if [ -n "$SUBSET_VARIATE_INDICES" ]; then
    echo "[INFO] $SINGLE_DATASET subset: indices=[$SUBSET_VARIATE_INDICES] (dim=$target_dim)"
else
    echo "[INFO] $SINGLE_DATASET: $target_dim variates (native, no splitting)"
fi
echo ""

echo "============================================================"
echo "  PHASE 1: Synthetic Pretraining (dim=$target_dim)"
echo "============================================================"

$PYTHON --mode pretrain --n-variates "$target_dim" $BASE_ARGS

if [ -n "$PRETRAIN_ONLY" ]; then
    echo ""
    echo "[INFO] --pretrain-only: stopping after Phase 1"
    exit 0
fi

echo ""
echo "============================================================"
echo "  PHASE 2: Fine-tuning $SINGLE_DATASET (dim=$target_dim)"
echo "============================================================"

$PYTHON --mode finetune --dataset "$SINGLE_DATASET" --n-variates "$target_dim" --variate-indices "$SUBSET_VARIATE_INDICES" --subset-id "$SUBSET_ID" $BASE_ARGS

echo ""
echo "============================================================"
echo "  PIPELINE COMPLETE"
echo "============================================================"

SUMMARY_CSV=$(find . -name "summary.csv" -path "*/results*" 2>/dev/null | head -1)
if [ -n "$SUMMARY_CSV" ]; then
    echo ""
    echo "Results summary:"
    head -20 "$SUMMARY_CSV"
    echo ""
    echo "Full results: $SUMMARY_CSV"
fi

echo ""
echo "=========================================="
echo "Job completed: $(date '+%m-%d %H:%M:%S')"
echo "Results: $RES_ROOT"
echo "Checkpoints: $CKPT_ROOT"
echo "=========================================="

wandb_upload_job_logs "$CKPT_ROOT" "$ALLIANCE_JOB_LOG"
