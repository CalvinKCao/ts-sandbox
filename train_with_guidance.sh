#!/usr/bin/env bash
# Pretrain iTransformer on Electricity (univariate) and then train DiffusionTSF with that checkpoint as guidance.
# Usage (from repo root):
#   chmod +x train_with_guidance.sh && ./train_with_guidance.sh
#   chmod +x train_with_guidance.sh && ./train_with_guidance.sh --repr-mode pdf  # Use PDF (stripe) representation
#   chmod +x train_with_guidance.sh && ./train_with_guidance.sh --force-retrain  # Force retrain iTransformer
#   chmod +x train_with_guidance.sh && ./train_with_guidance.sh --stride 12     # Use stride of 12 (half day for hourly data)
#   chmod +x train_with_guidance.sh && ./train_with_guidance.sh --dry-run       # Test without actually training
set -euo pipefail

# Parse arguments
FORCE_RETRAIN=false
REPR_MODE="cdf"  # Default to CDF (occupancy) mode
STRIDE=24  # Default stride (1 day for hourly data)
DRY_RUN=false  # Dry run mode for testing
while [[ $# -gt 0 ]]; do
  case $1 in
    --force-retrain)
      FORCE_RETRAIN=true
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
      echo "Usage: $0 [--force-retrain] [--repr-mode pdf|cdf] [--stride N] [--dry-run|--test]"
      exit 1
      ;;
  esac
done

echo "📊 Representation mode: ${REPR_MODE}"
echo "📊 Dataset stride: ${STRIDE}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

# Activate venv if present
if [[ -f "${REPO_ROOT}/venv/bin/activate" ]]; then
  source "${REPO_ROOT}/venv/bin/activate"
fi

# ------------------------------------------------------------------
# Stage 1: Train iTransformer on Electricity (univariate, target=OT)
# ------------------------------------------------------------------
# iTransformer saves checkpoints to: ./checkpoints/{setting}/checkpoint.pth
# We'll use a simple checkpoints dir and find the checkpoint after training
ITRANS_CKPT_BASE="${REPO_ROOT}/checkpoints/itransformer_guidance"
mkdir -p "${ITRANS_CKPT_BASE}"

# Check if any checkpoint exists in the base dir
EXISTING_CKPT=$(find "${ITRANS_CKPT_BASE}" -name "checkpoint.pth" 2>/dev/null | head -n1 || true)

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
  echo "🔥 Training iTransformer on Electricity (univariate)..."
  echo "   Using parameters from iTransformer paper (ICLR 2024), Section A.2"
  echo "   Paper defaults: batch=32, epochs=10, L∈{2,3,4}, D∈{256,512}, lr∈{1e-3,5e-4,1e-4}"
  
  cd models/iTransformer
  # iTransformer paper parameters (Section A.2):
  # - batch_size=32, train_epochs=10
  # - e_layers (L) ∈ {2, 3, 4}, d_model (D) ∈ {256, 512}
  # - learning_rate ∈ {1e-3, 5e-4, 1e-4}
  # - num_workers=10, patience=3
  # ECL dataset: 321 variates total, but we use univariate (features=S, enc_in=1)
  # Paper uses lookback=96, but we use 512 to match diffusion model requirements
  python3 run.py \
    --is_training 1 \
    --root_path "${REPO_ROOT}/datasets/electricity" \
    --data_path electricity.csv \
    --data custom \
    --model_id electricity_guidance \
    --model iTransformer \
    --features S \
    --target OT \
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
    --patience 3 \
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
echo "🔥 Training DiffusionTSF with iTransformer guidance..."
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

# DRY RUN MODE: Just validate and show what would happen
if [[ "${DRY_RUN}" == "true" ]]; then
  echo "🧪 DRY RUN MODE - No actual training will be performed"
  echo ""

  echo "📁 Checkpoint status:"
  if [[ -n "${EXISTING_CKPT}" ]]; then
    echo "   ✅ iTransformer checkpoint found: ${EXISTING_CKPT}"
    echo "   → Would reuse existing checkpoint (no retraining needed)"
  else
    echo "   ❌ No iTransformer checkpoint found"
    if [[ "${FORCE_RETRAIN}" == "true" ]]; then
      echo "   → Would train new iTransformer (--force-retrain specified)"
    else
      echo "   → Would train new iTransformer"
    fi
  fi
  echo ""

  echo "🔧 Training command that would be executed:"
  echo "python3 models/diffusion_tsf/train_electricity.py \\"
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
  if [[ -n "${EXISTING_CKPT}" ]]; then
    echo "  --guidance-checkpoint \"${EXISTING_CKPT}\""
  else
    echo "  --guidance-checkpoint \"[WOULD BE SET AFTER iTRANSFORMER TRAINING]\""
  fi
  echo ""
  echo "✨ Dry run complete! Remove --dry-run to actually train."
  exit 0
fi

# IMPORTANT: When --use-guidance with --guidance-type itransformer is set,
# the training script automatically uses CHRONOLOGICAL splits (70/10/20)
# to match iTransformer's training split and prevent data leakage.

# Value channel now uses last forecast_length values from past (no leakage)
# This gives the model context about recent value levels

python3 models/diffusion_tsf/train_electricity.py \
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
