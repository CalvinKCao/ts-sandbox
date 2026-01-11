#!/usr/bin/env bash
# =============================================================================
# QUICK EXPERIMENT: 10k Synthetic Pretrain WITH Guidance
# =============================================================================
# Compares:
#   1. 10k synthetic pretrain WITH iTransformer guidance
#   2. 10k ETTh2 training WITH iTransformer guidance (from pretrained)
#
# Uses existing best params from universal_synthetic_pretrain/best_params.json
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

# Configuration
EXPERIMENT_NAME="guidance_10k_experiment"
CKPT_DIR="${REPO_ROOT}/models/diffusion_tsf/checkpoints/${EXPERIMENT_NAME}"
PARAMS_FILE="${REPO_ROOT}/models/diffusion_tsf/checkpoints/universal_synthetic_pretrain/best_params.json"
LOG_FILE="${REPO_ROOT}/logs/${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${CKPT_DIR}"
mkdir -p "$(dirname "${LOG_FILE}")"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

log "============================================================================="
log "  QUICK EXPERIMENT: 10k Synthetic + Guidance"
log "============================================================================="
log ""
log "📁 Checkpoint dir: ${CKPT_DIR}"
log "📋 Params file: ${PARAMS_FILE}"
log "📝 Log file: ${LOG_FILE}"
log ""

# Check if params file exists
if [[ ! -f "${PARAMS_FILE}" ]]; then
    log "❌ Error: Params file not found: ${PARAMS_FILE}"
    exit 1
fi

# =============================================================================
# STEP 1: Train iTransformer on ETTh2 (if not already done)
# =============================================================================
log "============================================================================="
log "  STEP 1: Train iTransformer Guidance on ETTh2"
log "============================================================================="

GUIDANCE_DIR="${CKPT_DIR}/guidance"
GUIDANCE_CKPT="${GUIDANCE_DIR}/checkpoint.pth"
mkdir -p "${GUIDANCE_DIR}"

# Check if we can reuse existing iTransformer from fine-tuned ETTh2
EXISTING_GUIDANCE="${REPO_ROOT}/models/diffusion_tsf/checkpoints/ETTh2_MUFL/guidance/checkpoint.pth"

if [[ -f "${EXISTING_GUIDANCE}" ]]; then
    log "✅ Reusing existing iTransformer from ETTh2_MUFL"
    cp "${EXISTING_GUIDANCE}" "${GUIDANCE_CKPT}"
elif [[ -f "${GUIDANCE_CKPT}" ]]; then
    log "✅ iTransformer already exists: ${GUIDANCE_CKPT}"
else
    log "🔥 Training new iTransformer on ETTh2..."
    
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
        --checkpoints "${GUIDANCE_DIR}" \
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
        --des "guidance" \
        --loss MSE \
        --lradj type1 \
        --gpu 0 \
        --train_epochs 20 \
        --batch_size 32 \
        --learning_rate 0.0001 \
        --patience 7 2>&1 | tee -a "${LOG_FILE}"
    
    cd "${REPO_ROOT}"
    
    # Find and move checkpoint
    ITRANS_FOUND=$(find "${GUIDANCE_DIR}" -name "checkpoint.pth" -type f 2>/dev/null | head -1 || true)
    if [[ -z "${ITRANS_FOUND}" ]]; then
        log "❌ iTransformer training failed - no checkpoint found"
        exit 1
    fi
    if [[ "${ITRANS_FOUND}" != "${GUIDANCE_CKPT}" ]]; then
        mv "${ITRANS_FOUND}" "${GUIDANCE_CKPT}"
    fi
    log "✅ iTransformer saved: ${GUIDANCE_CKPT}"
fi

# =============================================================================
# STEP 2: Pretrain on 10k Synthetic WITH Guidance
# =============================================================================
log ""
log "============================================================================="
log "  STEP 2: Pretrain on 10k Synthetic Samples WITH Guidance"
log "============================================================================="

PRETRAIN_CKPT="${CKPT_DIR}/pretrain_10k_with_guidance.pt"

if [[ -f "${PRETRAIN_CKPT}" ]]; then
    log "✅ Pretrain checkpoint already exists: ${PRETRAIN_CKPT}"
else
    log "🔥 Training on 10k synthetic samples with iTransformer guidance..."
    
    python3 models/diffusion_tsf/train_electricity.py \
        --dataset ETTh2 \
        --params-file "${PARAMS_FILE}" \
        --synthetic-only \
        --synthetic-size 10000 \
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
        --guidance-checkpoint "${GUIDANCE_CKPT}" \
        --run-name "${EXPERIMENT_NAME}" 2>&1 | tee -a "${LOG_FILE}"
    
    # Copy best model
    BEST_MODEL="${CKPT_DIR}/best_model.pt"
    if [[ -f "${BEST_MODEL}" ]]; then
        cp "${BEST_MODEL}" "${PRETRAIN_CKPT}"
        log "✅ Pretrain saved: ${PRETRAIN_CKPT}"
    else
        log "❌ No pretrain checkpoint found after training"
        exit 1
    fi
fi

# =============================================================================
# STEP 3: Train on 10k ETTh2 Samples WITH Guidance (from pretrained)
# =============================================================================
log ""
log "============================================================================="
log "  STEP 3: Train on 10k ETTh2 Samples WITH Guidance (from pretrained)"
log "============================================================================="

FINAL_CKPT="${CKPT_DIR}/final_etth2_with_guidance.pt"

if [[ -f "${FINAL_CKPT}" ]]; then
    log "✅ Final checkpoint already exists: ${FINAL_CKPT}"
else
    log "🔥 Training on 10k ETTh2 samples with guidance (starting from pretrained)..."
    
    python3 models/diffusion_tsf/train_electricity.py \
        --dataset ETTh2 \
        --params-file "${PARAMS_FILE}" \
        --pretrained-checkpoint "${PRETRAIN_CKPT}" \
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
        --guidance-checkpoint "${GUIDANCE_CKPT}" \
        --run-name "${EXPERIMENT_NAME}_etth2" 2>&1 | tee -a "${LOG_FILE}"
    
    # Copy best model
    ETT_BEST="${REPO_ROOT}/models/diffusion_tsf/checkpoints/${EXPERIMENT_NAME}_etth2/best_model.pt"
    if [[ -f "${ETT_BEST}" ]]; then
        cp "${ETT_BEST}" "${FINAL_CKPT}"
        log "✅ Final model saved: ${FINAL_CKPT}"
    else
        log "⚠️ No best_model.pt found, checking for trial checkpoints..."
        TRIAL_BEST=$(find "${REPO_ROOT}/models/diffusion_tsf/checkpoints/${EXPERIMENT_NAME}_etth2" -name "*.pt" -type f 2>/dev/null | head -1 || true)
        if [[ -n "${TRIAL_BEST}" ]]; then
            cp "${TRIAL_BEST}" "${FINAL_CKPT}"
            log "✅ Final model saved: ${FINAL_CKPT}"
        else
            log "❌ No checkpoint found after ETTh2 training"
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
log "   ├── guidance/checkpoint.pth     (iTransformer)"
log "   ├── pretrain_10k_with_guidance.pt  (10k synthetic pretrain)"
log "   └── final_etth2_with_guidance.pt   (10k ETTh2 fine-tuned)"
log ""
log "📊 To visualize:"
log "   python3 run_visualizations.py --dataset ETTh2"
log ""
log "📝 Full log: ${LOG_FILE}"

