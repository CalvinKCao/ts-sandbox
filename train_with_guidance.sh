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
ITRANS_CKPT="${REPO_ROOT}/checkpoints/itransformer/itransformer_electricity.pt"
mkdir -p "$(dirname "${ITRANS_CKPT}")"

python3 models/iTransformer/run.py \
  --is_training 1 \
  --root_path "${REPO_ROOT}/datasets/electricity" \
  --data_path electricity.csv \
  --data custom \
  --model_id itransformer_electricity_512_96 \
  --model iTransformer \
  --features S \
  --target OT \
  --seq_len 512 \
  --label_len 0 \
  --pred_len 96 \
  --e_layers 3 \
  --d_model 512 \
  --n_heads 8 \
  --d_ff 2048 \
  --dropout 0.1 \
  --factor 1 \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --train_epochs 20 \
  --patience 5 \
  --save_dir "$(dirname "${ITRANS_CKPT}")" \
  --ckpt "${ITRANS_CKPT}"

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

