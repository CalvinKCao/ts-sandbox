#!/usr/bin/env bash
# =============================================================================
# QUICK EXPERIMENT: Full Synthetic Pretrain Pipeline (Both Models)
# =============================================================================
# 
# PIPELINE:
#   1. Train iTransformer on 10k SYNTHETIC data
#   2. Train DiffusionTSF on 10k SYNTHETIC with synthetic iTransformer guidance
#   3. Fine-tune iTransformer on ETTh2 real data
#   4. Fine-tune DiffusionTSF on ETTh2 with fine-tuned iTransformer guidance
#
# This tests if pretraining BOTH models on synthetic helps!
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

# Configuration
EXPERIMENT_NAME="full_synthetic_pretrain"
CKPT_DIR="${REPO_ROOT}/models/diffusion_tsf/checkpoints/${EXPERIMENT_NAME}"
PARAMS_FILE="${REPO_ROOT}/models/diffusion_tsf/checkpoints/universal_synthetic_pretrain/best_params.json"
LOG_FILE="${REPO_ROOT}/logs/${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S).log"
SYNTHETIC_SIZE=10000

mkdir -p "${CKPT_DIR}"
mkdir -p "$(dirname "${LOG_FILE}")"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

log "============================================================================="
log "  FULL SYNTHETIC PRETRAIN EXPERIMENT"
log "============================================================================="
log ""
log "📁 Checkpoint dir: ${CKPT_DIR}"
log "📋 Params file: ${PARAMS_FILE}"
log "📝 Log file: ${LOG_FILE}"
log "🔢 Synthetic size: ${SYNTHETIC_SIZE}"
log ""

# Check if params file exists
if [[ ! -f "${PARAMS_FILE}" ]]; then
    log "❌ Error: Params file not found: ${PARAMS_FILE}"
    exit 1
fi

# =============================================================================
# STEP 1: Train iTransformer on 10k SYNTHETIC data
# =============================================================================
log "============================================================================="
log "  STEP 1: Train iTransformer on ${SYNTHETIC_SIZE} SYNTHETIC samples"
log "============================================================================="

SYNTH_GUIDANCE_DIR="${CKPT_DIR}/synthetic_guidance"
SYNTH_GUIDANCE_CKPT="${SYNTH_GUIDANCE_DIR}/checkpoint.pth"
mkdir -p "${SYNTH_GUIDANCE_DIR}"

if [[ -f "${SYNTH_GUIDANCE_CKPT}" ]]; then
    log "✅ Synthetic iTransformer already exists: ${SYNTH_GUIDANCE_CKPT}"
else
    log "🔥 Training iTransformer on synthetic data..."
    
    # First, generate synthetic data CSV for iTransformer
    SYNTH_DATA_DIR="${CKPT_DIR}/synthetic_data"
    SYNTH_CSV="${SYNTH_DATA_DIR}/synthetic_${SYNTHETIC_SIZE}.csv"
    mkdir -p "${SYNTH_DATA_DIR}"
    
    if [[ ! -f "${SYNTH_CSV}" ]]; then
        log "   Generating synthetic CSV for iTransformer..."
        python3 -c "
import sys
sys.path.insert(0, 'models/diffusion_tsf')
from realts import RealTS
import pandas as pd
import numpy as np

# Generate synthetic data
synth = RealTS(n_samples=${SYNTHETIC_SIZE}, lookback=512, forecast=96)
print(f'Generated {len(synth)} synthetic samples')

# Create a CSV with synthetic time series (concatenate lookback+forecast)
all_series = []
for i in range(min(${SYNTHETIC_SIZE}, len(synth))):
    past, future = synth[i]
    full = np.concatenate([past.numpy(), future.numpy()])
    all_series.append(full)

# Stack into DataFrame
data = np.stack(all_series)  # (n_samples, 608)
# Transpose to (608, n_samples) for time-major format
df = pd.DataFrame(data.T, columns=[f'var_{i}' for i in range(data.shape[0])])
df['date'] = pd.date_range('2020-01-01', periods=len(df), freq='H')
df = df[['date'] + [c for c in df.columns if c != 'date']]
df['OT'] = df['var_0']  # Use first variable as target
df.to_csv('${SYNTH_CSV}', index=False)
print(f'Saved synthetic CSV: ${SYNTH_CSV}')
print(f'Shape: {df.shape}')
" 2>&1 | tee -a "${LOG_FILE}"
    fi
    
    cd models/iTransformer
    
    python3 run.py \
        --is_training 1 \
        --root_path "${SYNTH_DATA_DIR}" \
        --data_path "synthetic_${SYNTHETIC_SIZE}.csv" \
        --data custom \
        --model iTransformer \
        --features S \
        --target "OT" \
        --freq h \
        --checkpoints "${SYNTH_GUIDANCE_DIR}" \
        --seq_len 512 \
        --label_len 48 \
        --pred_len 96 \
        --e_layers 4 \
        --d_model 512 \
        --d_ff 512 \
        --factor 1 \
        --enc_in 1 \
        --dec_in 1 \
        --c_out 1 \
        --des "synth_guidance" \
        --loss MSE \
        --lradj type1 \
        --gpu 0 \
        --train_epochs 20 \
        --batch_size 32 \
        --learning_rate 0.0001 \
        --patience 7 2>&1 | tee -a "${LOG_FILE}"
    
    cd "${REPO_ROOT}"
    
    # Find and move checkpoint
    ITRANS_FOUND=$(find "${SYNTH_GUIDANCE_DIR}" -name "checkpoint.pth" -type f 2>/dev/null | head -1 || true)
    if [[ -z "${ITRANS_FOUND}" ]]; then
        log "❌ Synthetic iTransformer training failed - no checkpoint found"
        exit 1
    fi
    if [[ "${ITRANS_FOUND}" != "${SYNTH_GUIDANCE_CKPT}" ]]; then
        mv "${ITRANS_FOUND}" "${SYNTH_GUIDANCE_CKPT}"
    fi
    log "✅ Synthetic iTransformer saved: ${SYNTH_GUIDANCE_CKPT}"
fi

# =============================================================================
# STEP 2: Train DiffusionTSF on 10k SYNTHETIC WITH synthetic guidance
# =============================================================================
log ""
log "============================================================================="
log "  STEP 2: Train DiffusionTSF on ${SYNTHETIC_SIZE} SYNTHETIC with guidance"
log "============================================================================="

SYNTH_DIFFUSION_CKPT="${CKPT_DIR}/diffusion_synthetic_pretrain.pt"

if [[ -f "${SYNTH_DIFFUSION_CKPT}" ]]; then
    log "✅ Synthetic DiffusionTSF already exists: ${SYNTH_DIFFUSION_CKPT}"
else
    log "🔥 Training DiffusionTSF on synthetic with synthetic iTransformer guidance..."
    
    python3 models/diffusion_tsf/train_electricity.py \
        --dataset ETTh2 \
        --params-file "${PARAMS_FILE}" \
        --synthetic-only \
        --synthetic-size ${SYNTHETIC_SIZE} \
        --repr-mode cdf \
        --model-type unet \
        --stride 1 \
        --blur-sigma 1.0 \
        --emd-lambda 0 \
        --kernel-size 3 9 \
        --use-time-ramp \
        --use-value-channel \
        --no-hybrid-condition \
        --use-guidance \
        --guidance-type itransformer \
        --guidance-checkpoint "${SYNTH_GUIDANCE_CKPT}" \
        --run-name "${EXPERIMENT_NAME}_synth" 2>&1 | tee -a "${LOG_FILE}"
    
    # Copy best model
    BEST_MODEL="${CKPT_DIR}/../${EXPERIMENT_NAME}_synth/best_model.pt"
    if [[ -f "${BEST_MODEL}" ]]; then
        cp "${BEST_MODEL}" "${SYNTH_DIFFUSION_CKPT}"
        log "✅ Synthetic DiffusionTSF saved: ${SYNTH_DIFFUSION_CKPT}"
    else
        # Try to find any checkpoint
        TRIAL_CKPT=$(find "${CKPT_DIR}/../${EXPERIMENT_NAME}_synth" -name "*.pt" -type f 2>/dev/null | head -1 || true)
        if [[ -n "${TRIAL_CKPT}" ]]; then
            cp "${TRIAL_CKPT}" "${SYNTH_DIFFUSION_CKPT}"
            log "✅ Synthetic DiffusionTSF saved: ${SYNTH_DIFFUSION_CKPT}"
        else
            log "❌ No checkpoint found after synthetic training"
            exit 1
        fi
    fi
fi

# =============================================================================
# STEP 3: Fine-tune iTransformer on ETTh2 real data
# =============================================================================
log ""
log "============================================================================="
log "  STEP 3: Fine-tune iTransformer on ETTh2 real data"
log "============================================================================="

REAL_GUIDANCE_DIR="${CKPT_DIR}/real_guidance"
REAL_GUIDANCE_CKPT="${REAL_GUIDANCE_DIR}/checkpoint.pth"
mkdir -p "${REAL_GUIDANCE_DIR}"

if [[ -f "${REAL_GUIDANCE_CKPT}" ]]; then
    log "✅ Real iTransformer already exists: ${REAL_GUIDANCE_CKPT}"
else
    log "🔥 Fine-tuning iTransformer on ETTh2 real data..."
    
    # Copy synthetic checkpoint as starting point
    mkdir -p "${REAL_GUIDANCE_DIR}"
    cp "${SYNTH_GUIDANCE_CKPT}" "${REAL_GUIDANCE_DIR}/pretrain_checkpoint.pth"
    
    cd models/iTransformer
    
    python3 run.py \
        --is_training 1 \
        --root_path "${REPO_ROOT}/datasets/ETT-small" \
        --data_path "ETTh2.csv" \
        --data ETTh2 \
        --model iTransformer \
        --features S \
        --target "MUFL" \
        --freq h \
        --checkpoints "${REAL_GUIDANCE_DIR}" \
        --seq_len 512 \
        --label_len 48 \
        --pred_len 96 \
        --e_layers 4 \
        --d_model 512 \
        --d_ff 512 \
        --factor 1 \
        --enc_in 1 \
        --dec_in 1 \
        --c_out 1 \
        --des "real_guidance" \
        --loss MSE \
        --lradj type1 \
        --gpu 0 \
        --train_epochs 20 \
        --batch_size 32 \
        --learning_rate 0.0001 \
        --patience 7 2>&1 | tee -a "${LOG_FILE}"
    
    cd "${REPO_ROOT}"
    
    # Find and move checkpoint
    ITRANS_FOUND=$(find "${REAL_GUIDANCE_DIR}" -name "checkpoint.pth" -type f 2>/dev/null | head -1 || true)
    if [[ -z "${ITRANS_FOUND}" ]]; then
        log "❌ Real iTransformer training failed - no checkpoint found"
        exit 1
    fi
    if [[ "${ITRANS_FOUND}" != "${REAL_GUIDANCE_CKPT}" ]]; then
        mv "${ITRANS_FOUND}" "${REAL_GUIDANCE_CKPT}"
    fi
    log "✅ Real iTransformer saved: ${REAL_GUIDANCE_CKPT}"
fi

# =============================================================================
# STEP 4: Fine-tune DiffusionTSF on ETTh2 with fine-tuned guidance
# =============================================================================
log ""
log "============================================================================="
log "  STEP 4: Fine-tune DiffusionTSF on ETTh2 with fine-tuned guidance"
log "============================================================================="

FINAL_CKPT="${CKPT_DIR}/final_etth2.pt"

if [[ -f "${FINAL_CKPT}" ]]; then
    log "✅ Final checkpoint already exists: ${FINAL_CKPT}"
else
    log "🔥 Fine-tuning DiffusionTSF on ETTh2 (from synthetic pretrain)..."
    
    python3 models/diffusion_tsf/train_electricity.py \
        --dataset ETTh2 \
        --params-file "${PARAMS_FILE}" \
        --pretrained-checkpoint "${SYNTH_DIFFUSION_CKPT}" \
        --finetune-mode \
        --repr-mode cdf \
        --model-type unet \
        --stride 1 \
        --blur-sigma 1.0 \
        --emd-lambda 0 \
        --kernel-size 3 9 \
        --use-time-ramp \
        --use-value-channel \
        --no-hybrid-condition \
        --use-guidance \
        --guidance-type itransformer \
        --guidance-checkpoint "${REAL_GUIDANCE_CKPT}" \
        --run-name "${EXPERIMENT_NAME}_final" 2>&1 | tee -a "${LOG_FILE}"
    
    # Copy best model
    BEST_MODEL="${CKPT_DIR}/../${EXPERIMENT_NAME}_final/best_model.pt"
    if [[ -f "${BEST_MODEL}" ]]; then
        cp "${BEST_MODEL}" "${FINAL_CKPT}"
        log "✅ Final model saved: ${FINAL_CKPT}"
    else
        TRIAL_CKPT=$(find "${CKPT_DIR}/../${EXPERIMENT_NAME}_final" -name "*.pt" -type f 2>/dev/null | head -1 || true)
        if [[ -n "${TRIAL_CKPT}" ]]; then
            cp "${TRIAL_CKPT}" "${FINAL_CKPT}"
            log "✅ Final model saved: ${FINAL_CKPT}"
        else
            log "⚠️ No checkpoint found after final training"
        fi
    fi
fi

# =============================================================================
# Summary
# =============================================================================
log ""
log "============================================================================="
log "  EXPERIMENT COMPLETE"
log "============================================================================="
log ""
log "📁 Output: ${CKPT_DIR}"
log "   ├── synthetic_data/               (generated synthetic CSV)"
log "   ├── synthetic_guidance/           (iTransformer on synthetic)"
log "   │   └── checkpoint.pth"
log "   ├── diffusion_synthetic_pretrain.pt  (DiffusionTSF on synthetic)"
log "   ├── real_guidance/                (iTransformer fine-tuned on ETTh2)"
log "   │   └── checkpoint.pth"
log "   └── final_etth2.pt                (DiffusionTSF fine-tuned)"
log ""
log "📊 Pipeline:"
log "   1. iTransformer: synthetic 10k → fine-tune ETTh2"
log "   2. DiffusionTSF: synthetic 10k (w/ synth guidance) → fine-tune ETTh2 (w/ real guidance)"
log ""
log "📝 Full log: ${LOG_FILE}"
