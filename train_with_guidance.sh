#!/usr/bin/env bash
# Pretrain iTransformer on Electricity (univariate) and then train DiffusionTSF with that checkpoint as guidance.
# Usage (from repo root):
#   wsl -e bash -c "bash train_with_guidance.sh"
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
# iTransformer saves as: checkpoints/[setting]/checkpoint.pth
# where setting = model_id_model_data_features_ft{seq_len}_sl{label_len}_ll{pred_len}_..._0
ITRANS_CKPT_DIR="${REPO_ROOT}/checkpoints/itransformer/itransformer_electricity_512_96_iTransformer_custom_S_ft512_sl0_ll96_pl512_dm8_el3_dl1_df2048_fc1_ebtimeF_ebTrue_dt_electricity_training_projection_0"
ITRANS_CKPT="${ITRANS_CKPT_DIR}/checkpoint.pth"

if [[ -f "${ITRANS_CKPT}" ]]; then
  echo "✅ iTransformer checkpoint already exists at ${ITRANS_CKPT}, skipping training..."
else
  mkdir -p "${ITRANS_CKPT_DIR}"
  echo "🔥 Training iTransformer..."

python3 models/iTransformer/run.py \
  --is_training 1 \
  --root_path "${REPO_ROOT}/datasets/electricity" \
  --data_path electricity.csv \
  --data custom \
  --model_id itransformer_electricity_512_96 \
  --model iTransformer \
  --features S \
  --target OT \
  --freq h \
  --checkpoints "${ITRANS_CKPT_DIR}/" \
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
  --des electricity_training \
  --loss MSE \
  --lradj type1 \
  --use_amp \
  --use_gpu True \
  --gpu 0 \
  --exp_name MTSF \
  --channel_independence False \
  --inverse \
  --class_strategy projection \
  --use_norm True
fi

# ------------------------------------------------------------------
# Stage 2: Train DiffusionTSF with Visual Guide from iTransformer
# ------------------------------------------------------------------
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

