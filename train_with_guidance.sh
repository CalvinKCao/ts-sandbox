#!/usr/bin/env bash
# Pretrain iTransformer on Electricity (univariate) and then train DiffusionTSF with that checkpoint as guidance.
# Usage (from repo root):
#   chmod +x train_with_guidance.sh && ./train_with_guidance.sh
set -euo pipefail

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

if [[ -n "${EXISTING_CKPT}" ]]; then
  echo "✅ iTransformer checkpoint already exists at ${EXISTING_CKPT}, skipping training..."
  ITRANS_CKPT="${EXISTING_CKPT}"
else
  echo "🔥 Training iTransformer on Electricity (univariate)..."
  
  cd models/iTransformer
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
    --label_len 0 \
    --pred_len 96 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --d_model 512 \
    --n_heads 8 \
    --e_layers 3 \
    --d_ff 2048 \
    --factor 1 \
    --dropout 0.1 \
    --embed timeF \
    --activation gelu \
    --output_attention \
    --num_workers 4 \
    --itr 1 \
    --train_epochs 20 \
    --batch_size 16 \
    --patience 5 \
    --learning_rate 0.0001 \
    --des guidance \
    --loss MSE \
    --lradj type1 \
    --use_amp \
    --gpu 0 \
    --exp_name MTSF \
    --inverse \
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
echo ""

python3 models/diffusion_tsf/train_electricity.py \
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
