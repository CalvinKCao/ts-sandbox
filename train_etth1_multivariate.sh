#!/usr/bin/env bash
# Train DiffusionTSF on ETTh1 dataset (multivariate, all 7 variables) with iTransformer guidance
# Variables: HUFL, HULL, MUFL, MULL, LUFL, LULL, OT
# Usage: chmod +x train_etth1_multivariate.sh && ./train_etth1_multivariate.sh
#        chmod +x train_etth1_multivariate.sh && ./train_etth1_multivariate.sh --force-retrain
set -euo pipefail

# Parse arguments
FORCE_RETRAIN=false
while [[ $# -gt 0 ]]; do
  case $1 in
    --force-retrain)
      FORCE_RETRAIN=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--force-retrain]"
      exit 1
      ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

# Activate venv if present
if [[ -f "${REPO_ROOT}/venv/bin/activate" ]]; then
  source "${REPO_ROOT}/venv/bin/activate"
fi

# ------------------------------------------------------------------
# Stage 1: Train iTransformer on ETTh1 (multivariate, all 7 variables)
# ------------------------------------------------------------------
ITRANS_CKPT_BASE="${REPO_ROOT}/checkpoints/itransformer_etth1_multivariate"
mkdir -p "${ITRANS_CKPT_BASE}"

# Check if any checkpoint exists
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
  echo "🔥 Training iTransformer on ETTh1 (multivariate, 7 variables)..."
  echo "   Variables: HUFL, HULL, MUFL, MULL, LUFL, LULL, OT"
  echo "   Using parameters from iTransformer paper (ICLR 2024)"
  echo ""
  
  cd models/iTransformer
  
  # ETTh1 multivariate: 7 variables (features=M means multivariate)
  # Paper parameters for ETT datasets
  python3 run.py \
    --is_training 1 \
    --root_path "${REPO_ROOT}/datasets/ETT-small" \
    --data_path ETTh1.csv \
    --data ETTh1 \
    --model_id etth1_multivariate_guidance \
    --model iTransformer \
    --features M \
    --target OT \
    --freq h \
    --checkpoints "${ITRANS_CKPT_BASE}/" \
    --seq_len 512 \
    --label_len 48 \
    --pred_len 96 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
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
  echo "Looking for checkpoint in ${ITRANS_CKPT_BASE}..."
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
echo "🔥 Training DiffusionTSF on ETTh1 (multivariate) with iTransformer guidance..."
echo "   Checkpoint: ${ITRANS_CKPT}"
echo ""
echo "📊 Data splits (CHRONOLOGICAL, matches iTransformer):"
echo "   Train: first 70% of data"
echo "   Val:   next 10% of data"
echo "   Test:  last 20% of data (held out)"
echo ""

python3 models/diffusion_tsf/train_electricity.py \
  --dataset ETTh1 \
  --multivariate \
  --blur-sigma 1.0 \
  --emd-lambda 0 \
  --repr-mode cdf \
  --model-type unet \
  --kernel-size 3 9 \
  --use-defaults \
  --use-time-ramp \
  --use-value-channel \
  --no-hybrid-condition \
  --use-guidance \
  --guidance-type itransformer \
  --guidance-checkpoint "${ITRANS_CKPT}"

echo ""
echo "✅ Training complete!"
