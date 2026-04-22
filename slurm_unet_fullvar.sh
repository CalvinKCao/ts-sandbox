#!/bin/bash
# =============================================================================
# U-Net full-variate — self-resubmitting Slurm script for Killarney
#
# When run from the login node, it picks partition + wall time and sbatch's itself.
# When run inside a Slurm job (SLURM_JOB_ID is set), it runs full-variate training
# (same train_multivariate_pipeline as the old run_unet_fullvar.sh: bf16, H=96, no splitting).
#
# USAGE (from login node, repo root):
#   ./slurm_unet_fullvar.sh --smoke-test                     # L40S smoke
#   ./slurm_unet_fullvar.sh                                  # H100 full run
#   ./slurm_unet_fullvar.sh --dataset electricity
#   ./slurm_unet_fullvar.sh --resume --dataset traffic
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ===========================================================================
# If NOT inside a Slurm job → submit ourselves with the right resources
# ===========================================================================

if [ -z "$SLURM_JOB_ID" ]; then
    IS_SMOKE=0
    for arg in "$@"; do
        [ "$arg" = "--smoke-test" ] && IS_SMOKE=1
    done

    if [ "$IS_SMOKE" -eq 1 ]; then
        echo "Submitting SMOKE TEST (L40S, 8GB, 15 min)..."
        sbatch \
            --job-name=unet-fullvar-smoke \
            --account=aip-boyuwang \
            --time=0:15:00 \
            --nodes=1 \
            --gres=gpu:l40s:1 \
            --cpus-per-task=2 \
            --mem=8G \
            --output=unet-fullvar-smoke-%j.out \
            --error=unet-fullvar-smoke-%j.err \
            --mail-type=END,FAIL \
            --mail-user=ccao87@uwo.ca \
            "$SCRIPT_DIR/slurm_unet_fullvar.sh" "$@"
    else
        echo "Submitting FULL RUN (H100, 60GB, b4 = 3 days)..."
        sbatch \
            --job-name=unet-fullvar \
            --account=aip-boyuwang \
            --partition=gpubase_h100_b4 \
            --time=1-12:00:00 \
            --nodes=1 \
            --gpus-per-node=h100:1 \
            --cpus-per-task=6 \
            --mem=60G \
            --output=unet-fullvar-%j.out \
            --error=unet-fullvar-%j.err \
            --mail-type=BEGIN,END,FAIL \
            --mail-user=ccao87@uwo.ca \
            "$SCRIPT_DIR/slurm_unet_fullvar.sh" "$@"
    fi
    exit 0
fi

# ===========================================================================
# We're inside a Slurm job — do the actual work
# ===========================================================================

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi -L 2>/dev/null | head -1 || echo 'unknown')"
echo "Started: $(date)"
echo "=========================================="

# ---- Environment ----

module purge
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

# Auto-detect PROJECT
if [ -z "$PROJECT" ]; then
    if [ -d "$HOME/projects" ]; then
        FIRST_PROJECT=$(ls -d $HOME/projects/def-* $HOME/projects/aip-* 2>/dev/null | head -1)
        [ -n "$FIRST_PROJECT" ] && export PROJECT=$(readlink -f "$FIRST_PROJECT")
    fi
fi

if [ -z "$PROJECT" ]; then
    echo "ERROR: PROJECT not found"
    exit 1
fi

# Separate storage root so it doesn't conflict with the main multivariate pipeline
export STORAGE_ROOT="$PROJECT/$USER/diffusion-tsf-fullvar"
echo "STORAGE_ROOT: $STORAGE_ROOT"

mkdir -p "$STORAGE_ROOT/checkpoints"
mkdir -p "$STORAGE_ROOT/results"

# Copy datasets to PROJECT if needed
if [ ! -d "$STORAGE_ROOT/datasets" ]; then
    echo "Copying datasets to PROJECT storage..."
    cp -r "$PROJECT_ROOT/datasets" "$STORAGE_ROOT/datasets"
fi

# Venv — reuse main pipeline venv if it exists
VENV_PATH="$PROJECT/$USER/diffusion-tsf/venv"
if [ ! -d "$VENV_PATH" ]; then
    VENV_PATH="$STORAGE_ROOT/venv"
    if [ ! -d "$VENV_PATH" ]; then
        echo "Creating virtual environment..."
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

export WANDB_MODE=offline

# ---- Cleanup ----

cleanup() {
    trap '' EXIT ERR SIGTERM SIGINT SIGUSR1
    local code=${1:-$?}
    [ "$code" -ne 0 ] && echo "[CLEANUP] $(date) — killing child processes..."
    kill -- -$$ 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT ERR SIGTERM SIGINT SIGUSR1

# ---- Args: checkpoint/results from STORAGE_ROOT, then pass-through (strip --hours) ----

PIPELINE_ARGS=(--checkpoint-dir "$STORAGE_ROOT/checkpoints" --results-dir "$STORAGE_ROOT/results")
while [[ $# -gt 0 ]]; do
    case $1 in
        --hours) shift 2 ;;
        *)       PIPELINE_ARGS+=("$1"); shift ;;
    esac
done

cd "$PROJECT_ROOT"

# ---- Inlined full-variate U-Net driver (former run_unet_fullvar.sh) ----

set -- "${PIPELINE_ARGS[@]}"

AMP_FLAG="--amp"
IMAGE_HEIGHT=96
SUBSET_THRESHOLD=999999
SYNTHETIC_SAMPLES=75000
ITRANSFORMER_TRIALS=12
SEED=42

SMOKE_TEST=""
PRETRAIN_ONLY=""
SINGLE_DATASET=""
RESUME=""
EXTRA_PY_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --smoke-test)     SMOKE_TEST="--smoke-test"; shift ;;
        --dataset)        SINGLE_DATASET="$2"; shift 2 ;;
        --pretrain-only)  PRETRAIN_ONLY=1; shift ;;
        --resume)         RESUME=1; shift ;;
        --seed)           SEED="$2"; shift 2 ;;
        --wandb)          EXTRA_PY_ARGS="$EXTRA_PY_ARGS --wandb"; shift ;;
        --checkpoint-dir) EXTRA_PY_ARGS="$EXTRA_PY_ARGS --checkpoint-dir $2"; shift 2 ;;
        --results-dir)    EXTRA_PY_ARGS="$EXTRA_PY_ARGS --results-dir $2"; shift 2 ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$SINGLE_DATASET" ]; then
    SINGLE_DATASET="traffic"
fi

PYTHON="python -m models.diffusion_tsf.train_multivariate_pipeline"
BASE_ARGS="--seed $SEED $SMOKE_TEST $EXTRA_PY_ARGS"
BASE_ARGS="$BASE_ARGS $AMP_FLAG --image-height $IMAGE_HEIGHT"
BASE_ARGS="$BASE_ARGS --synthetic-samples $SYNTHETIC_SAMPLES"
BASE_ARGS="$BASE_ARGS --itransformer-trials $ITRANSFORMER_TRIALS"
BASE_ARGS="$BASE_ARGS --subset-threshold $SUBSET_THRESHOLD"

LOOKBACK_LENGTH=1024
FORECAST_LENGTH=192
LOOKBACK_OVERLAP=8

echo ""
echo "============================================================"
echo "  U-Net Full-Variate Training (Slurm)"
echo "============================================================"
echo "  Backbone:     U-Net (bf16)"
echo "  Image height: $IMAGE_HEIGHT"
echo "  Synth pool:   $SYNTHETIC_SAMPLES"
echo "  iTransformer trials: $ITRANSFORMER_TRIALS"
echo "  Subset threshold: $SUBSET_THRESHOLD (no splitting)"
echo "  Dataset:      $SINGLE_DATASET"
echo "  Smoke test:   ${SMOKE_TEST:-no}"
echo "============================================================"
echo ""

if [ ! -d "datasets" ] && [ -n "$STORAGE_ROOT" ] && [ -d "$STORAGE_ROOT/datasets" ]; then
    echo "[INFO] Symlinking datasets from $STORAGE_ROOT/datasets"
    ln -sf "$STORAGE_ROOT/datasets" datasets
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

echo "[INFO] $SINGLE_DATASET: $target_dim variates (native, no splitting)"
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

$PYTHON --mode finetune --dataset "$SINGLE_DATASET" --n-variates "$target_dim" $BASE_ARGS

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
echo "Job completed: $(date)"
echo "Results: $STORAGE_ROOT/results"
echo "Checkpoints: $STORAGE_ROOT/checkpoints"
echo "=========================================="
