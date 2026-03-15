#!/bin/bash
# run_unet_fullvar.sh — train U-Net directly on full-variate datasets
#
# Trains with bf16, image height=96, no subset splitting. This is a parallel
# training path alongside pipeline.sh for testing U-Net scalability.
#
# Usage:
#   ./run_unet_fullvar.sh                          # default: traffic (861-var)
#   ./run_unet_fullvar.sh --smoke-test             # quick validation
#   ./run_unet_fullvar.sh --dataset electricity    # single dataset
#   ./run_unet_fullvar.sh --resume                 # skip completed work
#   ./run_unet_fullvar.sh --pretrain-only          # just synthetic pretraining

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

cleanup() {
    trap '' EXIT ERR SIGTERM SIGINT SIGHUP
    local code=${1:-$?}
    [ "$code" -ne 0 ] && echo "[CLEANUP] exit $code — killing child processes..."
    kill -- -$$ 2>/dev/null || true
    wait 2>/dev/null || true
    [ "$code" -ne 0 ] && echo "[CLEANUP] GPU resources released."
    exit "$code"
}
trap 'cleanup $?' EXIT
trap 'cleanup 1' ERR SIGTERM SIGINT SIGHUP

# ============================================================================
# Defaults — full-variate U-Net config
# ============================================================================

MODEL_TYPE="unet"
AMP_FLAG="--amp"
IMAGE_HEIGHT=96
SUBSET_THRESHOLD=999999     # treat ALL datasets as native dim (no splitting)
SYNTHETIC_SAMPLES=75000     # reduced pool for high-V
ITRANSFORMER_TRIALS=20      # full HP search
SEED=42

SMOKE_TEST=""
PRETRAIN_ONLY=""
SINGLE_DATASET=""
RESUME=""
EXTRA_PY_ARGS=""

# ============================================================================
# Arg parsing
# ============================================================================

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
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Full-variate U-Net training (bf16, H=96, no subset splitting)."
            echo ""
            echo "Options:"
            echo "  --smoke-test         Quick validation run"
            echo "  --dataset NAME       Process only this dataset (default: traffic)"
            echo "  --pretrain-only      Stop after synthetic pretraining"
            echo "  --resume             Skip completed work"
            echo "  --seed N             Random seed (default: 42)"
            echo "  --wandb              Enable wandb logging"
            echo "  --checkpoint-dir D   Override checkpoint directory"
            echo "  --results-dir D      Override results directory"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Default to traffic if no dataset specified
if [ -z "$SINGLE_DATASET" ]; then
    SINGLE_DATASET="traffic"
fi

# ============================================================================
# Environment
# ============================================================================

if [ -d "venv" ]; then
    source venv/bin/activate
    echo "[INFO] venv activated"
fi

PYTHON="python -m models.diffusion_tsf.train_7var_pipeline"
BASE_ARGS="--seed $SEED $SMOKE_TEST $EXTRA_PY_ARGS"
BASE_ARGS="$BASE_ARGS --model-type $MODEL_TYPE $AMP_FLAG --image-height $IMAGE_HEIGHT"
BASE_ARGS="$BASE_ARGS --synthetic-samples $SYNTHETIC_SAMPLES"
BASE_ARGS="$BASE_ARGS --itransformer-trials $ITRANSFORMER_TRIALS"
BASE_ARGS="$BASE_ARGS --subset-threshold $SUBSET_THRESHOLD"

read LOOKBACK_LENGTH FORECAST_LENGTH LOOKBACK_OVERLAP < <(python3 -c "
from models.diffusion_tsf.train_7var_pipeline import LOOKBACK_LENGTH, FORECAST_LENGTH, LOOKBACK_OVERLAP
print(LOOKBACK_LENGTH, FORECAST_LENGTH, LOOKBACK_OVERLAP)
")

echo "============================================================"
echo "  U-Net Full-Variate Training"
echo "============================================================"
echo "  Model:        $MODEL_TYPE (bf16)"
echo "  Image height: $IMAGE_HEIGHT"
echo "  Synth pool:   $SYNTHETIC_SAMPLES"
echo "  iTransformer trials: $ITRANSFORMER_TRIALS"
echo "  Subset threshold: $SUBSET_THRESHOLD (no splitting)"
echo "  Dataset:      $SINGLE_DATASET"
echo "  Smoke test:   ${SMOKE_TEST:-no}"
echo ""

# ============================================================================
# Prep: recombine traffic CSV if needed
# ============================================================================

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

# ============================================================================
# Discover dimensionality (native — no splitting)
# ============================================================================

declare -A DATASET_DIM

discover_dims() {
    python3 -c "
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
    # No splitting: dim = native column count
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

# ============================================================================
# PHASE 1: Pretrain for native dimensionality
# ============================================================================

echo "============================================================"
echo "  PHASE 1: Synthetic Pretraining (dim=$target_dim)"
echo "============================================================"

$PYTHON --mode pretrain --n-variates "$target_dim" $BASE_ARGS

if [ -n "$PRETRAIN_ONLY" ]; then
    echo ""
    echo "[INFO] --pretrain-only: stopping after Phase 1"
    exit 0
fi

# ============================================================================
# PHASE 2: Fine-tune on the dataset
# ============================================================================

echo ""
echo "============================================================"
echo "  PHASE 2: Fine-tuning $SINGLE_DATASET (dim=$target_dim)"
echo "============================================================"

$PYTHON --mode finetune --dataset "$SINGLE_DATASET" --n-variates "$target_dim" $BASE_ARGS

# ============================================================================
# Summary
# ============================================================================

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
