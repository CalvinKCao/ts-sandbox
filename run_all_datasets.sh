#!/bin/bash
# run_all_datasets.sh - Universal training pipeline for all datasets
#
# Usage:
#   ./run_all_datasets.sh                   # Run full pipeline
#   ./run_all_datasets.sh --smoke-test      # Quick validation run
#   ./run_all_datasets.sh --pretrain-only   # Only pretrain on synthetic
#   ./run_all_datasets.sh --dataset ETTh1   # Fine-tune single dataset

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv if exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "[INFO] Activated venv"
fi

# ============================================================================
# Configuration
# ============================================================================

SYNTHETIC_SAMPLES=1000000
PRETRAIN_EPOCHS=10
FINETUNE_EPOCHS=20
BATCH_SIZE=16
LR=0.0001

CHECKPOINT_DIR="models/diffusion_tsf/checkpoints/universal_v2"
DATASETS_DIR="datasets"

# Parse arguments
SMOKE_TEST=""
PRETRAIN_ONLY=""
SINGLE_DATASET=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --smoke-test)
            SMOKE_TEST="--smoke-test"
            shift
            ;;
        --pretrain-only)
            PRETRAIN_ONLY="1"
            shift
            ;;
        --dataset)
            SINGLE_DATASET="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============================================================================
# Step 0: Prepare Data
# ============================================================================

echo "============================================================"
echo "STEP 0: Data Preparation"
echo "============================================================"

# Combine traffic files if needed
TRAFFIC_DIR="$DATASETS_DIR/traffic"
TRAFFIC_CSV="$TRAFFIC_DIR/traffic.csv"
TRAFFIC_PART1="$TRAFFIC_DIR/traffic_part1.csv"
TRAFFIC_PART2="$TRAFFIC_DIR/traffic_part2.csv"

if [ ! -f "$TRAFFIC_CSV" ]; then
    if [ -f "$TRAFFIC_PART1" ] && [ -f "$TRAFFIC_PART2" ]; then
        echo "[INFO] Combining traffic_part1.csv and traffic_part2.csv..."
        
        # Copy part1 entirely
        cat "$TRAFFIC_PART1" > "$TRAFFIC_CSV"
        
        # Append part2 without header
        tail -n +2 "$TRAFFIC_PART2" >> "$TRAFFIC_CSV"
        
        echo "[INFO] Created $TRAFFIC_CSV"
        wc -l "$TRAFFIC_CSV"
    else
        echo "[WARN] Traffic parts not found, skipping combination"
    fi
else
    echo "[INFO] traffic.csv already exists"
fi

# ============================================================================
# Smoke Test Mode
# ============================================================================

if [ -n "$SMOKE_TEST" ]; then
    echo "============================================================"
    echo "SMOKE TEST MODE"
    echo "============================================================"
    
    python -m models.diffusion_tsf.train_universal_v2 --smoke-test
    
    echo ""
    echo "[SUCCESS] Smoke test completed!"
    exit 0
fi

# ============================================================================
# Step 1: Pretrain on Synthetic Data (1M samples, 7 variates)
# ============================================================================

PRETRAINED_CKPT="$CHECKPOINT_DIR/pretrained_diffusion.pt"

if [ ! -f "$PRETRAINED_CKPT" ] || [ -n "$PRETRAIN_ONLY" ]; then
    echo "============================================================"
    echo "STEP 1: Pretraining on Synthetic Data"
    echo "============================================================"
    
    python -m models.diffusion_tsf.train_universal_v2 \
        --mode pretrain \
        --synthetic-samples $SYNTHETIC_SAMPLES \
        --pretrain-epochs $PRETRAIN_EPOCHS \
        --batch-size $BATCH_SIZE \
        --lr $LR \
        --checkpoint-dir "$CHECKPOINT_DIR"
    
    if [ -n "$PRETRAIN_ONLY" ]; then
        echo "[INFO] Pretrain-only mode, exiting"
        exit 0
    fi
else
    echo "[INFO] Pretrained checkpoint exists: $PRETRAINED_CKPT"
fi

# ============================================================================
# Step 2: Fine-tune on Real Datasets
# ============================================================================

echo "============================================================"
echo "STEP 2: Fine-tuning on Real Datasets"
echo "============================================================"

# 7-variate datasets (direct fine-tuning)
DATASETS_7VAR="ETTh1 ETTh2 ETTm1 ETTm2 illness"

# >7-variate datasets (requires CCM)
DATASETS_CCM="electricity weather exchange_rate traffic"

finetune_dataset() {
    local dataset=$1
    echo ""
    echo "------------------------------------------------------------"
    echo "Fine-tuning on: $dataset"
    echo "------------------------------------------------------------"
    
    python -m models.diffusion_tsf.train_universal_v2 \
        --mode finetune \
        --dataset "$dataset" \
        --finetune-epochs $FINETUNE_EPOCHS \
        --batch-size $BATCH_SIZE \
        --lr $LR \
        --checkpoint-dir "$CHECKPOINT_DIR" \
        --pretrained-path "$PRETRAINED_CKPT"
}

if [ -n "$SINGLE_DATASET" ]; then
    # Fine-tune single dataset
    finetune_dataset "$SINGLE_DATASET"
else
    # Fine-tune all datasets
    
    echo ""
    echo "[INFO] Fine-tuning 7-variate datasets (direct)..."
    for dataset in $DATASETS_7VAR; do
        finetune_dataset "$dataset"
    done
    
    echo ""
    echo "[INFO] Fine-tuning >7-variate datasets (with CCM)..."
    for dataset in $DATASETS_CCM; do
        finetune_dataset "$dataset"
    done
fi

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "============================================================"
echo "TRAINING COMPLETE"
echo "============================================================"
echo ""
echo "Checkpoints saved to: $CHECKPOINT_DIR"
echo ""
ls -la "$CHECKPOINT_DIR"
echo ""

# List cluster visualizations if they exist
CLUSTER_VIZS=$(find "$CHECKPOINT_DIR" -name "cluster_visualization.png" 2>/dev/null || true)
if [ -n "$CLUSTER_VIZS" ]; then
    echo "Cluster visualizations:"
    echo "$CLUSTER_VIZS"
fi





