#!/bin/bash
# run_all_datasets.sh - training loop for everything
#
# usage:
#   ./run_all_datasets.sh                   # do the whole thing
#   ./run_all_datasets.sh --smoke-test      # just a quick check
#   ./run_all_datasets.sh --pretrain-only   # just synthetic stuff
#   ./run_all_datasets.sh --dataset ETTh1   # fine tune one thing

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# get venv if its there
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "[INFO] venv on"
fi

# config stuff
SYNTHETIC_SAMPLES=1000000
PRETRAIN_EPOCHS=10
FINETUNE_EPOCHS=20
BATCH_SIZE=16
LR=0.0001

CHECKPOINT_DIR="models/diffusion_tsf/checkpoints/universal_v2"
DATASETS_DIR="datasets"

# args
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

# prepare data
echo "============================================================"
echo "prep data"
echo "============================================================"

# traffic files need to be squashed together
TRAFFIC_DIR="$DATASETS_DIR/traffic"
TRAFFIC_CSV="$TRAFFIC_DIR/traffic.csv"
TRAFFIC_PART1="$TRAFFIC_DIR/traffic_part1.csv"
TRAFFIC_PART2="$TRAFFIC_DIR/traffic_part2.csv"

if [ ! -f "$TRAFFIC_CSV" ]; then
    if [ -f "$TRAFFIC_PART1" ] && [ -f "$TRAFFIC_PART2" ]; then
        echo "[INFO] smashing traffic_part1 and part2 together..."
        
        # part1
        cat "$TRAFFIC_PART1" > "$TRAFFIC_CSV"
        
        # part2 without header
        tail -n +2 "$TRAFFIC_PART2" >> "$TRAFFIC_CSV"
        
        echo "[INFO] done $TRAFFIC_CSV"
        wc -l "$TRAFFIC_CSV"
    else
        echo "[WARN] missing traffic parts"
    fi
else
    echo "[INFO] traffic.csv exists"
fi

# smoke test
if [ -n "$SMOKE_TEST" ]; then
    echo "============================================================"
    echo "SMOKE TEST"
    echo "============================================================"
    
    python -m models.diffusion_tsf.train_universal_v2 --smoke-test
    
    echo ""
    echo "[SUCCESS] smoke test done"
    exit 0
fi

# pretrain (synthetic)
PRETRAINED_CKPT="$CHECKPOINT_DIR/pretrained_diffusion.pt"

if [ ! -f "$PRETRAINED_CKPT" ] || [ -n "$PRETRAIN_ONLY" ]; then
    echo "============================================================"
    echo "pretraining"
    echo "============================================================"
    
    python -m models.diffusion_tsf.train_universal_v2 \
        --mode pretrain \
        --synthetic-samples $SYNTHETIC_SAMPLES \
        --pretrain-epochs $PRETRAIN_EPOCHS \
        --batch-size $BATCH_SIZE \
        --lr $LR \
        --checkpoint-dir "$CHECKPOINT_DIR"
    
    if [ -n "$PRETRAIN_ONLY" ]; then
        echo "[INFO] pretrain only, stopping"
        exit 0
    fi
else
    echo "[INFO] already got pretrained ckpt: $PRETRAINED_CKPT"
fi

# finetune on real data
echo "============================================================"
echo "finetuning real stuff"
echo "============================================================"

# 7-var
DATASETS_7VAR="ETTh1 ETTh2 ETTm1 ETTm2 illness"

# >7-var (needs CCM)
DATASETS_CCM="electricity weather exchange_rate traffic"

finetune_dataset() {
    local dataset=$1
    echo ""
    echo "------------------------------------------------------------"
    echo "dataset: $dataset"
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
    finetune_dataset "$SINGLE_DATASET"
else
    # do all of them
    
    echo ""
    echo "[INFO] 7-var datasets..."
    for dataset in $DATASETS_7VAR; do
        finetune_dataset "$dataset"
    done
    
    echo ""
    echo "[INFO] CCM datasets..."
    for dataset in $DATASETS_CCM; do
        finetune_dataset "$dataset"
    done
fi

# training complete
echo ""
echo "============================================================"
echo "DONE"
echo "============================================================"
echo ""
echo "ckpts in: $CHECKPOINT_DIR"
echo ""
ls -la "$CHECKPOINT_DIR"
echo ""

# clusters?
CLUSTER_VIZS=$(find "$CHECKPOINT_DIR" -name "cluster_visualization.png" 2>/dev/null || true)
if [ -n "$CLUSTER_VIZS" ]; then
    echo "cluster viz:"
    echo "$CLUSTER_VIZS"
fi





