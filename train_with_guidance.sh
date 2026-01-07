#!/usr/bin/env bash
# Pretrain iTransformer on a dataset (univariate) and then train DiffusionTSF with that checkpoint as guidance.
# Usage (from repo root):
#   chmod +x train_with_guidance.sh && ./train_with_guidance.sh
#   chmod +x train_with_guidance.sh && ./train_with_guidance.sh --repr-mode pdf  # Use PDF (stripe) representation
#   chmod +x train_with_guidance.sh && ./train_with_guidance.sh --force-retrain  # Force retrain iTransformer
#   chmod +x train_with_guidance.sh && ./train_with_guidance.sh --stride 12     # Use stride of 12 (half day for hourly data)
#   chmod +x train_with_guidance.sh && ./train_with_guidance.sh --dry-run       # Test without actually training
#   chmod +x train_with_guidance.sh && ./train_with_guidance.sh --etth1-univariate  # Train on ETTh1 first variable (HUFL)
set -euo pipefail

# Parse arguments
FORCE_RETRAIN=false
REPR_MODE="cdf"  # Default to CDF (occupancy) mode
STRIDE=1  # Default stride
DRY_RUN=false  # Dry run mode for testing
USE_ETTH1_UNIVARIATE=false  # ETTh1 univariate mode (first variable: HUFL)
while [[ $# -gt 0 ]]; do
  case $1 in
    --force-retrain)
      FORCE_RETRAIN=true
      shift
      ;;
    --etth1-univariate|--etth1)
      USE_ETTH1_UNIVARIATE=true
      FORCE_RETRAIN=true  # Always retrain iTransformer for ETTh1 univariate
      shift
      ;;
    --repr-mode)
      if [[ -z "${2:-}" ]]; then
        echo "Error: --repr-mode requires an argument (pdf or cdf)"
        exit 1
      fi
      REPR_MODE="$2"
      if [[ "$REPR_MODE" != "pdf" && "$REPR_MODE" != "cdf" ]]; then
        echo "Error: --repr-mode must be 'pdf' or 'cdf', got: $REPR_MODE"
        exit 1
      fi
      shift 2
      ;;
    --stride)
      if [[ -z "${2:-}" ]]; then
        echo "Error: --stride requires an integer argument"
        exit 1
      fi
      STRIDE="$2"
      if ! [[ "$STRIDE" =~ ^[0-9]+$ ]] || [[ "$STRIDE" -le 0 ]]; then
        echo "Error: --stride must be a positive integer, got: $STRIDE"
        exit 1
      fi
      shift 2
      ;;
    --dry-run|--test)
      DRY_RUN=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--force-retrain] [--repr-mode pdf|cdf] [--stride N] [--dry-run|--test] [--etth1-univariate]"
      exit 1
      ;;
  esac
done

# Dataset-specific configuration
if [[ "${USE_ETTH1_UNIVARIATE}" == "true" ]]; then
  DATASET_NAME="ETTh1"
  DATASET_DIR="ETT-small"
  DATASET_FILE="ETTh1.csv"
  TARGET_COLUMN="HUFL"  # First variable in ETTh1
  NUM_VARIABLES=1
  SEASONAL_PERIOD=24  # Hourly data, daily seasonality
  ITRANS_DATA_TYPE="ETTh1"  # iTransformer data loader type
  echo "📊 Dataset: ETTh1 (univariate, first variable: ${TARGET_COLUMN})"
  echo "📊 Note: iTransformer will be retrained (--etth1-univariate implies --force-retrain)"
else
  DATASET_NAME="electricity"
  DATASET_DIR="electricity"
  DATASET_FILE="electricity.csv"
  TARGET_COLUMN="OT"
  NUM_VARIABLES=1
  SEASONAL_PERIOD=24  # Hourly data
  ITRANS_DATA_TYPE="custom"
  echo "📊 Dataset: Electricity (univariate, target: ${TARGET_COLUMN})"
fi

echo "📊 Representation mode: ${REPR_MODE}"
echo "📊 Dataset stride: ${STRIDE}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

# Activate venv if present
if [[ -f "${REPO_ROOT}/venv/bin/activate" ]]; then
  source "${REPO_ROOT}/venv/bin/activate"
fi

# ------------------------------------------------------------------
# Stage 1: Train iTransformer (univariate)
# ------------------------------------------------------------------
# iTransformer saves checkpoints to: ./checkpoints/{setting}/checkpoint.pth
# We'll use a dataset-specific checkpoints dir and find the checkpoint after training
if [[ "${USE_ETTH1_UNIVARIATE}" == "true" ]]; then
  ITRANS_CKPT_BASE="${REPO_ROOT}/checkpoints/itransformer_etth1_univariate"
else
  ITRANS_CKPT_BASE="${REPO_ROOT}/checkpoints/itransformer_guidance"
fi

# Check if any checkpoint exists in the base dir (before potentially deleting it)
EXISTING_CKPT=$(find "${ITRANS_CKPT_BASE}" -name "checkpoint.pth" 2>/dev/null | head -n1 || true)

# DRY RUN: Show what would happen and exit early
if [[ "${DRY_RUN}" == "true" ]]; then
  echo ""
  echo "🧪 DRY RUN MODE - No actual training will be performed"
  echo ""
  echo "📁 Stage 1 - iTransformer:"
  if [[ -n "${EXISTING_CKPT}" && "${FORCE_RETRAIN}" != "true" ]]; then
    echo "   ✅ Would reuse existing checkpoint: ${EXISTING_CKPT}"
  else
    if [[ "${FORCE_RETRAIN}" == "true" ]]; then
      echo "   🗑️  Would delete existing checkpoint (--force-retrain)"
    fi
    echo "   🔥 Would train iTransformer on ${DATASET_NAME} (univariate, target=${TARGET_COLUMN})"
    echo "   → Checkpoint dir: ${ITRANS_CKPT_BASE}"
  fi
  echo ""
  echo "📁 Stage 2 - DiffusionTSF:"
  echo "🔧 Training command that would be executed:"
  echo "python3 models/diffusion_tsf/train_electricity.py \\"
  echo "  --dataset \"${DATASET_NAME}\" \\"
  echo "  --target \"${TARGET_COLUMN}\" \\"
  echo "  --blur-sigma 1.0 \\"
  echo "  --emd-lambda 0 \\"
  echo "  --repr-mode \"${REPR_MODE}\" \\"
  echo "  --model-type unet \\"
  echo "  --kernel-size 3 9 \\"
  echo "  --stride \"${STRIDE}\" \\"
  echo "  --use-defaults \\"
  echo "  --use-time-ramp \\"
  echo "  --use-value-channel \\"
  echo "  --no-hybrid-condition \\"
  echo "  --use-guidance \\"
  echo "  --guidance-type itransformer \\"
  echo "  --guidance-checkpoint \"[SET AFTER iTransformer TRAINING]\""
  echo ""
  echo "✨ Dry run complete! Remove --dry-run to actually train."
  exit 0
fi

mkdir -p "${ITRANS_CKPT_BASE}"

# Force retrain: delete existing checkpoint
if [[ "${FORCE_RETRAIN}" == "true" && -n "${EXISTING_CKPT}" ]]; then
  echo "🗑️  --force-retrain: Deleting existing iTransformer checkpoint..."
  rm -rf "${ITRANS_CKPT_BASE}"
  mkdir -p "${ITRANS_CKPT_BASE}"
  EXISTING_CKPT=""
fi

if [[ -n "${EXISTING_CKPT}" ]]; then
  echo "✅ iTransformer checkpoint already exists at ${EXISTING_CKPT}, skipping training..."
  echo "   (Use --force-retrain to retrain from scratch)"
  ITRANS_CKPT="${EXISTING_CKPT}"
else
  echo "🔥 Training iTransformer on ${DATASET_NAME} (univariate, target=${TARGET_COLUMN})..."
  echo "   Using parameters from iTransformer paper (ICLR 2024), Section A.2"
  echo "   Paper defaults: batch=32, epochs=10, L∈{2,3,4}, D∈{256,512}, lr∈{1e-3,5e-4,1e-4}"
  
  cd models/iTransformer
  # iTransformer paper parameters (Section A.2):
  # - batch_size=32, train_epochs=10
  # - e_layers (L) ∈ {2, 3, 4}, d_model (D) ∈ {256, 512}
  # - learning_rate ∈ {1e-3, 5e-4, 1e-4}
  # - num_workers=10, patience=3
  # Paper uses lookback=96, but we use 512 to match diffusion model requirements
  python3 run.py \
    --is_training 1 \
    --root_path "${REPO_ROOT}/datasets/${DATASET_DIR}" \
    --data_path "${DATASET_FILE}" \
    --data "${ITRANS_DATA_TYPE}" \
    --model_id "${DATASET_NAME}_univariate_guidance" \
    --model iTransformer \
    --features S \
    --target "${TARGET_COLUMN}" \
    --freq h \
    --checkpoints "${ITRANS_CKPT_BASE}/" \
    --seq_len 512 \
    --label_len 48 \
    --pred_len 96 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --d_model 512 \
    --n_heads 8 \
    --e_layers 4 \
    --d_layers 1 \
    --d_ff 2048 \
    --factor 1 \
    --dropout 0.1 \
    --embed timeF \
    --activation gelu \
    --num_workers 10 \
    --itr 1 \
    --train_epochs 10 \
    --batch_size 32 \
    --patience 7 \
    --learning_rate 0.0001 \
    --des guidance \
    --loss MSE \
    --lradj type1 \
    --gpu 0 \
    --exp_name MTSF \
    --class_strategy projection \
    --use_norm 1
  
  cd "${REPO_ROOT}"
  
  # Find the checkpoint that was just created
  ITRANS_CKPT=$(find "${ITRANS_CKPT_BASE}" -name "checkpoint.pth" | head -n1)
  
  if [[ -z "${ITRANS_CKPT}" ]]; then
    echo "❌ ERROR: iTransformer training completed but no checkpoint.pth found in ${ITRANS_CKPT_BASE}"
    exit 1
  fi
  
  echo "✅ iTransformer checkpoint saved to: ${ITRANS_CKPT}"
fi

# ------------------------------------------------------------------
# Stage 2: Train DiffusionTSF with Visual Guide from iTransformer
# ------------------------------------------------------------------
echo ""
echo "🔥 Training DiffusionTSF on ${DATASET_NAME} (univariate) with iTransformer guidance..."
echo "   Checkpoint: ${ITRANS_CKPT}"
echo "   Representation mode: ${REPR_MODE}"
if [[ "$REPR_MODE" == "pdf" ]]; then
  echo "     → Using stripe/one-hot encoding (probability density)"
else
  echo "     → Using occupancy map encoding (cumulative distribution)"
fi
echo ""
echo "📊 Data splits (CHRONOLOGICAL, matches iTransformer):"
echo "   Train: first 70% of data"
echo "   Val:   next 10% of data"
echo "   Test:  last 20% of data (held out)"
echo ""
echo "📊 Stride & No-Leak Guarantee:"
echo "   Stride: ${STRIDE} timesteps between windows"
echo "   Window size: 608 timesteps (512 lookback + 96 forecast)"
echo "   Gap between splits: ceil(608/${STRIDE}) = $(( (608 + STRIDE - 1) / STRIDE )) indices"
echo "   → No inter-split overlap/leakage"
echo ""

# IMPORTANT: When --use-guidance with --guidance-type itransformer is set,
# the training script automatically uses CHRONOLOGICAL splits (70/10/20)
# to match iTransformer's training split and prevent data leakage.

# Value channel now uses last forecast_length values from past (no leakage)
# This gives the model context about recent value levels

python3 models/diffusion_tsf/train_electricity.py \
  --dataset "${DATASET_NAME}" \
  --target "${TARGET_COLUMN}" \
  --blur-sigma 1.0 \
  --emd-lambda 0 \
  --repr-mode "${REPR_MODE}" \
  --model-type unet \
  --kernel-size 3 9 \
  --stride "${STRIDE}" \
  --use-defaults \
  --use-time-ramp \
  --use-value-channel \
  --no-hybrid-condition \
  --use-guidance \
  --guidance-type itransformer \
  --guidance-checkpoint "${ITRANS_CKPT}"

echo ""
echo "✅ Training complete!"
