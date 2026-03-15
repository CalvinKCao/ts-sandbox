#!/bin/bash
# run_ci_dit.sh — CI-DiT pipeline: channel-independent diffusion transformer
#
# Same structure as pipeline.sh but uses the CI-DiT backbone with:
#   - Half image height (64 instead of 128)
#   - bfloat16 mixed precision (AMP)
#   - Gradient checkpointing
#   - No subset splitting — all variates handled natively via channel-independent processing
#
# Usage:
#   ./run_ci_dit.sh                          # full pipeline
#   ./run_ci_dit.sh --smoke-test             # quick validation
#   ./run_ci_dit.sh --dataset traffic        # single dataset
#   ./run_ci_dit.sh --pretrain-only          # just pretraining

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
# Defaults
# ============================================================================

SMOKE_TEST=""
PRETRAIN_ONLY=""
SINGLE_DATASET=""
SEED=42
EXTRA_PY_ARGS=""

# CI-DiT specifics: no subset splitting, since each variate is processed independently.
# Set SUBSET_THRESHOLD absurdly high so every dataset stays at its native dim.
SUBSET_THRESHOLD=999999

while [[ $# -gt 0 ]]; do
    case $1 in
        --smoke-test)     SMOKE_TEST="--smoke-test"; shift ;;
        --dataset)        SINGLE_DATASET="$2"; shift 2 ;;
        --pretrain-only)  PRETRAIN_ONLY=1; shift ;;
        --seed)           SEED="$2"; shift 2 ;;
        --wandb)          EXTRA_PY_ARGS="$EXTRA_PY_ARGS --wandb"; shift ;;
        --checkpoint-dir) EXTRA_PY_ARGS="$EXTRA_PY_ARGS --checkpoint-dir $2"; shift 2 ;;
        --results-dir)    EXTRA_PY_ARGS="$EXTRA_PY_ARGS --results-dir $2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "CI-DiT pipeline: channel-independent DiT with AMP + gradient checkpointing"
            echo ""
            echo "Options:"
            echo "  --smoke-test         Quick validation run"
            echo "  --dataset NAME       Process only this dataset"
            echo "  --pretrain-only      Stop after synthetic pretraining"
            echo "  --seed N             Random seed (default: 42)"
            echo "  --wandb              Enable wandb logging"
            echo "  --checkpoint-dir D   Override checkpoint directory"
            echo "  --results-dir D      Override results directory"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ============================================================================
# Environment
# ============================================================================

if [ -d "venv" ]; then
    source venv/bin/activate
    echo "[INFO] venv activated"
fi

# CI-DiT flags: model type, image height 64, bfloat16 AMP, gradient checkpointing
PYTHON="python -m models.diffusion_tsf.train_7var_pipeline"
CI_DIT_FLAGS="--model-type ci_dit --image-height 64 --amp --gradient-checkpointing"
BASE_ARGS="--seed $SEED $SMOKE_TEST $EXTRA_PY_ARGS $CI_DIT_FLAGS"

read LOOKBACK_LENGTH FORECAST_LENGTH LOOKBACK_OVERLAP < <(python3 -c "
from models.diffusion_tsf.train_7var_pipeline import LOOKBACK_LENGTH, FORECAST_LENGTH, LOOKBACK_OVERLAP
print(LOOKBACK_LENGTH, FORECAST_LENGTH, LOOKBACK_OVERLAP)
")

echo "============================================================"
echo "  CI-DiT Pipeline"
echo "============================================================"
echo "  Backbone:     CI-DiT (channel-independent)"
echo "  Image height: 64"
echo "  AMP:          bfloat16"
echo "  Grad ckpt:    enabled"
echo "  Smoke test:   ${SMOKE_TEST:-no}"
echo "  Dataset:      ${SINGLE_DATASET:-all}"
echo ""

# ============================================================================
# Prep: recombine traffic CSV
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
# Discover dimensionality groups
# ============================================================================

# With SUBSET_THRESHOLD=999999, every dataset keeps its native column count.
# e.g. traffic=861, electricity=321, weather=21, ETTh1=7, etc.
declare -A DATASET_DIM
declare -A DATASET_NCOLS
declare -A DIM_DATASETS

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

threshold = $SUBSET_THRESHOLD
min_rows = $LOOKBACK_LENGTH + $FORECAST_LENGTH + $LOOKBACK_OVERLAP

for name, path in sorted(registry.items()):
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    n_rows = len(df)
    if n_rows < min_rows:
        import sys
        print(f'[SKIP] {name}: only {n_rows} rows (need {min_rows})', file=sys.stderr)
        continue
    n_cols = sum(1 for c in df.columns if c.lower() != 'date')
    # With CI-DiT: no subset splitting, every dataset stays at native dim
    dim = n_cols
    print(f'{name} {n_cols} {dim}')
"
}

echo "[INFO] Discovering dataset dimensionalities..."
PRETRAIN_DIMS=""

while IFS=' ' read -r ds ncols dim; do
    DATASET_DIM[$ds]=$dim
    DATASET_NCOLS[$ds]=$ncols
    DIM_DATASETS[$dim]="${DIM_DATASETS[$dim]} $ds"
    if [[ ! " $PRETRAIN_DIMS " =~ " $dim " ]]; then
        PRETRAIN_DIMS="$PRETRAIN_DIMS $dim"
    fi
done < <(discover_dims)

if [ -n "$SINGLE_DATASET" ]; then
    target_dim="${DATASET_DIM[$SINGLE_DATASET]}"
    if [ -z "$target_dim" ]; then
        echo "[ERROR] Unknown dataset: $SINGLE_DATASET"
        exit 1
    fi
    PRETRAIN_DIMS="$target_dim"
fi

echo "[INFO] Pretrain dims: $PRETRAIN_DIMS"
for dim in $PRETRAIN_DIMS; do
    echo "  dim=$dim: ${DIM_DATASETS[$dim]}"
done
echo ""

# ============================================================================
# PHASE 1: Pretrain for each unique dimensionality
# ============================================================================

echo "============================================================"
echo "  PHASE 1: Synthetic Pretraining (CI-DiT)"
echo "============================================================"

for dim in $PRETRAIN_DIMS; do
    echo ""
    echo "--- Pretraining CI-DiT dim=$dim ---"
    $PYTHON --mode pretrain --n-variates "$dim" $BASE_ARGS
done

if [ -n "$PRETRAIN_ONLY" ]; then
    echo ""
    echo "[INFO] --pretrain-only: stopping after Phase 1"
    exit 0
fi

# ============================================================================
# PHASE 2: Fine-tune each dataset (no subset splitting, all native dim)
# ============================================================================

echo ""
echo "============================================================"
echo "  PHASE 2: Fine-tuning (CI-DiT, all variates native)"
echo "============================================================"

if [ -n "$SINGLE_DATASET" ]; then
    DATASETS_TO_FINETUNE="$SINGLE_DATASET"
else
    DATASETS_TO_FINETUNE=""
    for ds in "${!DATASET_DIM[@]}"; do
        DATASETS_TO_FINETUNE="$DATASETS_TO_FINETUNE $ds"
    done
fi

for ds in $DATASETS_TO_FINETUNE; do
    dim="${DATASET_DIM[$ds]}"
    echo ""
    echo "--- Fine-tuning $ds (dim=$dim, all variates) ---"
    $PYTHON --mode finetune --dataset "$ds" --n-variates "$dim" $BASE_ARGS
done

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "============================================================"
echo "  CI-DiT PIPELINE COMPLETE"
echo "============================================================"

SUMMARY_CSV=$(find . -name "summary.csv" -path "*/results*" 2>/dev/null | head -1)
if [ -n "$SUMMARY_CSV" ]; then
    echo ""
    echo "Results:"
    head -20 "$SUMMARY_CSV"
fi

echo ""
echo "Done."
