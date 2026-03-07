#!/bin/bash
# pipeline.sh — single comprehensive script for training + evaluation
#
# Default: pretrain on synthetic data for each unique dimensionality, fine-tune
# all datasets, evaluate against iTransformer baselines, generate comparison viz.
#
# Usage:
#   ./pipeline.sh                          # full pipeline, 1 GPU
#   ./pipeline.sh --smoke-test             # quick validation (~2 min)
#   ./pipeline.sh --gpus 4                 # parallel subset fine-tuning
#   ./pipeline.sh --dataset electricity    # single dataset
#   ./pipeline.sh --pretrain-only          # just synthetic pretraining
#   ./pipeline.sh --resume                 # skip already-completed work

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Kill all child processes on exit/failure/signal so GPUs are released immediately.
# This catches orphaned background fine-tune jobs from the multi-GPU loop.
_PIPELINE_EXIT_CODE=0
cleanup() {
    _PIPELINE_EXIT_CODE=${_PIPELINE_EXIT_CODE:-$?}
    if [ "$_PIPELINE_EXIT_CODE" -ne 0 ]; then
        echo ""
        echo "[CLEANUP] Pipeline exited with code $_PIPELINE_EXIT_CODE — killing child processes..."
    fi
    kill -- -$$ 2>/dev/null || true
    wait 2>/dev/null || true
    if [ "$_PIPELINE_EXIT_CODE" -ne 0 ]; then
        echo "[CLEANUP] GPU resources released."
    fi
}
trap 'cleanup' EXIT
trap '_PIPELINE_EXIT_CODE=1; exit 1' ERR SIGTERM SIGINT SIGHUP

# ============================================================================
# Defaults
# ============================================================================

NUM_GPUS=1
SMOKE_TEST=""
PRETRAIN_ONLY=""
SINGLE_DATASET=""
RESUME=""
SEED=42
EXTRA_PY_ARGS=""
SUBSET_DIM=32
SUBSET_THRESHOLD=32

# ============================================================================
# Arg parsing
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --smoke-test)  SMOKE_TEST="--smoke-test"; shift ;;
        --gpus)        NUM_GPUS="$2"; shift 2 ;;
        --dataset)     SINGLE_DATASET="$2"; shift 2 ;;
        --pretrain-only) PRETRAIN_ONLY=1; shift ;;
        --resume)      RESUME=1; shift ;;
        --seed)        SEED="$2"; shift 2 ;;
        --subset-dim)  SUBSET_DIM="$2"; shift 2 ;;
        --wandb)       EXTRA_PY_ARGS="$EXTRA_PY_ARGS --wandb"; shift ;;
        --checkpoint-dir) EXTRA_PY_ARGS="$EXTRA_PY_ARGS --checkpoint-dir $2"; shift 2 ;;
        --results-dir)    EXTRA_PY_ARGS="$EXTRA_PY_ARGS --results-dir $2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --smoke-test         Quick validation run"
            echo "  --gpus N             Number of GPUs for parallel fine-tuning (default: 1)"
            echo "  --dataset NAME       Process only this dataset"
            echo "  --pretrain-only      Stop after synthetic pretraining"
            echo "  --resume             Skip completed work"
            echo "  --seed N             Random seed (default: 42)"
            echo "  --subset-dim N       Variate width for high-dim subsets (default: 32)"
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

# ============================================================================
# Environment
# ============================================================================

if [ -d "venv" ]; then
    source venv/bin/activate
    echo "[INFO] venv activated"
fi

PYTHON="python -m models.diffusion_tsf.train_7var_pipeline"
BASE_ARGS="--seed $SEED $SMOKE_TEST $EXTRA_PY_ARGS"

echo "============================================================"
echo "  Diffusion TSF Pipeline"
echo "============================================================"
echo "  GPUs:        $NUM_GPUS"
echo "  Subset dim:  $SUBSET_DIM"
echo "  Smoke test:  ${SMOKE_TEST:-no}"
echo "  Dataset:     ${SINGLE_DATASET:-all}"
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
# Discover dimensionality groups
# ============================================================================

# Counts columns per dataset, groups them into pretrain dims
declare -A DATASET_DIM
declare -A DATASET_NCOLS
declare -A DIM_DATASETS  # dim -> space-separated dataset names

discover_dims() {
    python3 -c "
import pandas as pd, os, json

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
subset_dim = $SUBSET_DIM

for name, path in sorted(registry.items()):
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path, nrows=1)
    n_cols = sum(1 for c in df.columns if c.lower() != 'date')
    dim = subset_dim if n_cols > threshold else n_cols
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

# Filter to single dataset if requested
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
echo "  PHASE 1: Synthetic Pretraining"
echo "============================================================"

for dim in $PRETRAIN_DIMS; do
    echo ""
    echo "--- Pretraining dim=$dim ---"
    $PYTHON --mode pretrain --n-variates "$dim" $BASE_ARGS
done

if [ -n "$PRETRAIN_ONLY" ]; then
    echo ""
    echo "[INFO] --pretrain-only: stopping after Phase 1"
    exit 0
fi

# ============================================================================
# PHASE 2: Fine-tune each dataset
# ============================================================================

echo ""
echo "============================================================"
echo "  PHASE 2: Fine-tuning"
echo "============================================================"

# Determine which datasets to process
if [ -n "$SINGLE_DATASET" ]; then
    DATASETS_TO_FINETUNE="$SINGLE_DATASET"
else
    DATASETS_TO_FINETUNE=""
    for ds in "${!DATASET_DIM[@]}"; do
        DATASETS_TO_FINETUNE="$DATASETS_TO_FINETUNE $ds"
    done
fi

# Sort datasets: native-dim first, then high-variate (to batch GPU work)
NATIVE_DATASETS=""
HIGHVAR_DATASETS=""
for ds in $DATASETS_TO_FINETUNE; do
    ncols="${DATASET_NCOLS[$ds]}"
    if [ "$ncols" -gt "$SUBSET_THRESHOLD" ]; then
        HIGHVAR_DATASETS="$HIGHVAR_DATASETS $ds"
    else
        NATIVE_DATASETS="$NATIVE_DATASETS $ds"
    fi
done

# --- Native-dim datasets (sequential, 1 GPU each) ---
for ds in $NATIVE_DATASETS; do
    dim="${DATASET_DIM[$ds]}"
    echo ""
    echo "--- Fine-tuning $ds (native dim=$dim) ---"
    $PYTHON --mode finetune --dataset "$ds" --n-variates "$dim" $BASE_ARGS
done

# --- High-variate datasets (parallel across GPUs) ---
for ds in $HIGHVAR_DATASETS; do
    dim="${DATASET_DIM[$ds]}"
    echo ""
    echo "--- Fine-tuning $ds (${DATASET_NCOLS[$ds]} cols -> ${dim}-dim subsets) ---"

    # Get subset list from Python (smoke test: only 1 subset to verify the path)
    SUBSET_JSON=$($PYTHON --mode list-subsets --dataset "$ds" --n-variates "$dim" --seed "$SEED")
    if [ -n "$SMOKE_TEST" ]; then
        SUBSET_JSON=$(echo "$SUBSET_JSON" | head -1)
    fi

    gpu=0
    running=0

    while IFS= read -r line; do
        subset_id=$(echo "$line" | python3 -c "import sys,json; print(json.load(sys.stdin)['subset_id'])")
        var_indices=$(echo "$line" | python3 -c "import sys,json; print(','.join(str(x) for x in json.load(sys.stdin)['variate_indices']))")

        echo "  [GPU $gpu] Launching $subset_id"
        CUDA_VISIBLE_DEVICES=$gpu $PYTHON \
            --mode finetune-subset \
            --dataset "$ds" \
            --subset-id "$subset_id" \
            --variate-indices "$var_indices" \
            --n-variates "$dim" \
            $BASE_ARGS &

        running=$((running + 1))
        gpu=$(( (gpu + 1) % NUM_GPUS ))

        # When all GPUs are busy, wait for the batch to finish
        if [ "$running" -ge "$NUM_GPUS" ]; then
            wait
            running=0
        fi
    done <<< "$SUBSET_JSON"

    # Wait for any stragglers
    wait
    echo "  [DONE] All subsets for $ds"
done

# ============================================================================
# PHASE 3: Full-dim iTransformer baselines (high-variate datasets)
# ============================================================================

if [ -n "$HIGHVAR_DATASETS" ]; then
    echo ""
    echo "============================================================"
    echo "  PHASE 3: Full-dim iTransformer Baselines"
    echo "============================================================"

    for ds in $HIGHVAR_DATASETS; do
        echo ""
        echo "--- Training full-dim iTransformer baseline for $ds ---"
        $PYTHON --mode baseline --dataset "$ds" $BASE_ARGS
    done
fi

# ============================================================================
# PHASE 4: Comparison Visualizations
# ============================================================================

echo ""
echo "============================================================"
echo "  PHASE 4: Comparison Visualizations"
echo "============================================================"

# Rebuild summary CSV
$PYTHON --mode evaluate $BASE_ARGS || true

# Generate comparison plots if the viz script exists
VIZ_SCRIPT="models/diffusion_tsf/visualize_comparison.py"
if [ -f "$VIZ_SCRIPT" ]; then
    echo "[INFO] Generating comparison plots..."
    python -m models.diffusion_tsf.visualize_comparison \
        --num-samples 3 --vars 3 2>/dev/null || \
        echo "[WARN] Visualization failed (may need completed checkpoints)"
fi

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

echo ""
echo "Run 'python summarize_results.py' for a detailed report."
echo ""
