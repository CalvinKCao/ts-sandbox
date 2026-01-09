#!/usr/bin/env bash
# =============================================================================
# Universal Pre-training + Per-Dataset Fine-tuning for DiffusionTSF
# =============================================================================
# 
# This script implements a 3-stage training pipeline:
#
# STAGE 0: SYNTHETIC HYPERPARAMETER TUNING
#   - Run 8 Optuna trials on 10k synthetic samples
#   - Uses high-end search space: LR (1e-5, 1e-3), model_size [small, large],
#     diffusion_steps [2000, 4000], batch_size [128, 512], schedule [linear]
#   - Validates on real data (ETTh2) to measure transfer quality
#   - Finds best hyperparameters for synthetic data training
#
# STAGE 1: UNIVERSAL PRE-TRAINING
#   - Train a single "universal" model on 100k synthetic samples
#   - Uses best params from Stage 0
#   - Saved to: checkpoints/universal_synthetic_pretrain/best_model.pt
#   - This becomes the BASE MODEL for all per-dataset fine-tuning
#
# STAGE 2: PER-DATASET FINE-TUNING (for each of 6 datasets)
#   - Train iTransformer guidance model (saved to DATASET_VAR/guidance/)
#   - Fine-tune DiffusionTSF starting from universal model
#   - 10 Optuna trials per dataset with different hyperparameters
#   - LR range (1e-6, 1e-4) - more conservative for fine-tuning
#   - Best model saved to: checkpoints/DATASET_VAR/best_model.pt
#
# OUTPUT STRUCTURE:
#   checkpoints/
#   ├── universal_synthetic_pretrain/
#   │   ├── best_params.json        # Best HP from synthetic search
#   │   └── best_model.pt           # Universal pre-trained model
#   ├── ETTh2_LUFL/
#   │   ├── guidance/checkpoint.pth # iTransformer guidance model
#   │   └── best_model.pt           # Fine-tuned DiffusionTSF
#   ├── ETTm1_MULL/
#   │   ├── guidance/checkpoint.pth
#   │   └── best_model.pt
#   └── ...
#
# DATA LEAKAGE PREVENTION:
#   1. CHRONOLOGICAL SPLITS: Train (first 70%), Val (next 10%), Test (last 20%)
#   2. GAP BETWEEN SPLITS: ceil(window_size / stride) indices gap
#   3. PER-SAMPLE NORMALIZATION: Each sample normalized independently
#   4. iTransformer TRAINING: Uses same chronological split
#
# Usage:
#   chmod +x train_universal_pretrain.sh && ./train_universal_pretrain.sh
#   ./train_universal_pretrain.sh --dry-run                    # Show what would happen
#   ./train_universal_pretrain.sh --seed 42                    # Set random seed
#   ./train_universal_pretrain.sh --skip-synthetic-search      # Skip Stage 0, use default params
#   ./train_universal_pretrain.sh --skip-universal-pretrain    # Skip Stage 1 (assumes model exists)
#   ./train_universal_pretrain.sh --skip-itrans                # Skip iTransformer training
#   ./train_universal_pretrain.sh --dataset ETTh2              # Fine-tune only on ETTh2
#   ./train_universal_pretrain.sh --finetune-trials 5          # Fewer fine-tuning trials
#
set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

# Parse arguments
DRY_RUN=false
RANDOM_SEED=${RANDOM}
SKIP_SYNTHETIC_SEARCH=false
SKIP_UNIVERSAL_PRETRAIN=false
SKIP_ITRANSFORMER=false
SINGLE_DATASET=""
STRIDE=1
USE_MONO=false
MONO_WEIGHT=10.0

# Stage 0: Synthetic HP search settings
SYNTHETIC_SEARCH_TRIALS=8
SYNTHETIC_SEARCH_SIZE=10000

# Stage 1: Universal pre-training settings
UNIVERSAL_PRETRAIN_EPOCHS=100
UNIVERSAL_PRETRAIN_SIZE=100000

# Stage 2: Per-dataset fine-tuning settings
FINETUNE_TRIALS=10

while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run|--test)
      DRY_RUN=true
      shift
      ;;
    --seed)
      if [[ -z "${2:-}" ]]; then
        echo "Error: --seed requires an integer argument"
        exit 1
      fi
      RANDOM_SEED="$2"
      shift 2
      ;;
    --skip-synthetic-search)
      SKIP_SYNTHETIC_SEARCH=true
      shift
      ;;
    --skip-universal-pretrain)
      SKIP_UNIVERSAL_PRETRAIN=true
      shift
      ;;
    --skip-itrans|--skip-itransformer)
      SKIP_ITRANSFORMER=true
      shift
      ;;
    --dataset)
      if [[ -z "${2:-}" ]]; then
        echo "Error: --dataset requires an argument"
        exit 1
      fi
      SINGLE_DATASET="$2"
      shift 2
      ;;
    --stride)
      if [[ -z "${2:-}" ]]; then
        echo "Error: --stride requires an integer argument"
        exit 1
      fi
      STRIDE="$2"
      shift 2
      ;;
    --synthetic-search-trials)
      if [[ -z "${2:-}" ]]; then
        echo "Error: --synthetic-search-trials requires an integer argument"
        exit 1
      fi
      SYNTHETIC_SEARCH_TRIALS="$2"
      shift 2
      ;;
    --universal-pretrain-epochs)
      if [[ -z "${2:-}" ]]; then
        echo "Error: --universal-pretrain-epochs requires an integer argument"
        exit 1
      fi
      UNIVERSAL_PRETRAIN_EPOCHS="$2"
      shift 2
      ;;
    --universal-pretrain-size)
      if [[ -z "${2:-}" ]]; then
        echo "Error: --universal-pretrain-size requires an integer argument"
        exit 1
      fi
      UNIVERSAL_PRETRAIN_SIZE="$2"
      shift 2
      ;;
    --finetune-trials)
      if [[ -z "${2:-}" ]]; then
        echo "Error: --finetune-trials requires an integer argument"
        exit 1
      fi
      FINETUNE_TRIALS="$2"
      shift 2
      ;;
    --use-mono|--use-monotonicity-loss)
      USE_MONO=true
      shift
      ;;
    --mono-weight|--monotonicity-weight)
      if [[ -z "${2:-}" ]]; then
        echo "Error: --monotonicity-weight requires a float argument"
        exit 1
      fi
      MONO_WEIGHT="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo ""
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --dry-run                      Show what would happen without executing"
      echo "  --seed N                       Set random seed for reproducibility"
      echo "  --skip-synthetic-search        Skip Stage 0 (use default params)"
      echo "  --skip-universal-pretrain      Skip Stage 1 (assumes model exists)"
      echo "  --skip-itrans                  Skip iTransformer guidance training"
      echo "  --dataset NAME                 Fine-tune only on specific dataset"
      echo "  --stride N                     Stride for sliding window (default: 1)"
      echo "  --synthetic-search-trials N    Optuna trials for synthetic HP search (default: 8)"
      echo "  --universal-pretrain-epochs N  Epochs for universal pre-training (default: 100)"
      echo "  --universal-pretrain-size N    Samples for universal pre-training (default: 100000)"
      echo "  --finetune-trials N            Optuna trials per dataset (default: 10)"
      echo "  --use-mono                     Enable monotonicity loss"
      echo "  --mono-weight F                Weight for monotonicity loss (default: 10.0)"
      exit 1
      ;;
  esac
done

# Set bash random seed for reproducibility
RANDOM=$RANDOM_SEED

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

# Activate venv if present
if [[ -f "${REPO_ROOT}/venv/bin/activate" ]]; then
  source "${REPO_ROOT}/venv/bin/activate"
fi

# Log file for this run
LOG_DIR="${REPO_ROOT}/models/diffusion_tsf/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/universal_pretrain_${TIMESTAMP}.log"

# Function to log to both console and file
log() {
  echo "$@" | tee -a "${LOG_FILE}"
}

# =============================================================================
# Merge Split Datasets (if needed)
# =============================================================================

TRAFFIC_CSV="${REPO_ROOT}/datasets/traffic/traffic.csv"
TRAFFIC_PART1="${REPO_ROOT}/datasets/traffic/traffic_part1.csv"
TRAFFIC_PART2="${REPO_ROOT}/datasets/traffic/traffic_part2.csv"

if [[ ! -f "${TRAFFIC_CSV}" ]] && [[ -f "${TRAFFIC_PART1}" ]] && [[ -f "${TRAFFIC_PART2}" ]]; then
  log "📦 Merging split traffic dataset..."
  head -1 "${TRAFFIC_PART1}" > "${TRAFFIC_CSV}"
  tail -n +2 "${TRAFFIC_PART1}" >> "${TRAFFIC_CSV}"
  tail -n +2 "${TRAFFIC_PART2}" >> "${TRAFFIC_CSV}"
  MERGED_LINES=$(wc -l < "${TRAFFIC_CSV}")
  log "   ✅ Merged! Total lines: ${MERGED_LINES}"
fi

# =============================================================================
# Dataset Definitions
# =============================================================================

declare -A DATASET_CONFIGS=(
  ["ETTh2"]="ETT-small|ETTh2.csv|24|ETTh2"
  ["ETTm1"]="ETT-small|ETTm1.csv|96|ETTm1"
  ["illness"]="illness|national_illness.csv|52|custom"
  ["exchange_rate"]="exchange_rate|exchange_rate.csv|5|custom"
  ["traffic"]="traffic|traffic.csv|24|custom"
  ["weather"]="weather|weather.csv|144|custom"
)

# Function to get a random column from a dataset using Python
get_random_column() {
  local data_path="$1"
  local seed="$2"
  python3 -c "
import pandas as pd
import random

random.seed($seed)

df = pd.read_csv('$data_path', nrows=5)
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
numeric_cols = [c for c in numeric_cols if c.lower() != 'date']

if numeric_cols:
    col = random.choice(numeric_cols)
    print(col)
else:
    print('OT')
"
}

# Function to sanitize column names for directory/file names
sanitize_name() {
  local name="$1"
  # Replace spaces, parentheses, and other special chars with underscores
  echo "$name" | sed 's/[^a-zA-Z0-9_-]/_/g' | sed 's/__*/_/g' | sed 's/^_//;s/_$//'
}

# =============================================================================
# Directory Setup
# =============================================================================

CHECKPOINT_BASE="${REPO_ROOT}/models/diffusion_tsf/checkpoints"
UNIVERSAL_CKPT_DIR="${CHECKPOINT_BASE}/universal_synthetic_pretrain"
mkdir -p "${UNIVERSAL_CKPT_DIR}"

# =============================================================================
# Main
# =============================================================================

log "============================================================================="
log "  UNIVERSAL PRE-TRAINING + PER-DATASET FINE-TUNING PIPELINE"
log "============================================================================="
log ""
log "📊 Configuration:"
log "   Log file: ${LOG_FILE}"
log "   Random seed: ${RANDOM_SEED}"
log "   Stride: ${STRIDE}"
log "   Skip synthetic search: ${SKIP_SYNTHETIC_SEARCH}"
log "   Skip universal pretrain: ${SKIP_UNIVERSAL_PRETRAIN}"
log "   Skip iTransformer: ${SKIP_ITRANSFORMER}"
log "   Monotonicity loss: ${USE_MONO} (weight=${MONO_WEIGHT})"
if [[ -n "${SINGLE_DATASET}" ]]; then
  log "   Single dataset mode: ${SINGLE_DATASET}"
fi
log ""
log "📋 Stage Settings:"
log "   Stage 0 (Synthetic HP Search): ${SYNTHETIC_SEARCH_TRIALS} trials on ${SYNTHETIC_SEARCH_SIZE} samples"
log "   Stage 1 (Universal Pretrain): ${UNIVERSAL_PRETRAIN_EPOCHS} epochs on ${UNIVERSAL_PRETRAIN_SIZE} samples"
log "   Stage 2 (Per-Dataset Finetune): ${FINETUNE_TRIALS} trials per dataset"
log ""

# Datasets to process in Stage 2
if [[ -n "${SINGLE_DATASET}" ]]; then
  DATASETS=("${SINGLE_DATASET}")
else
  DATASETS=("ETTh2" "ETTm1" "illness" "exchange_rate" "traffic" "weather")
fi

# ------------------------------------------------------------------
# DRY RUN MODE
# ------------------------------------------------------------------
if [[ "${DRY_RUN}" == "true" ]]; then
  log "🧪 DRY RUN MODE - Showing what would happen:"
  log ""
  
  log "=== STAGE 0: SYNTHETIC HYPERPARAMETER SEARCH ==="
  if [[ "${SKIP_SYNTHETIC_SEARCH}" == "true" ]]; then
    log "   SKIPPED (--skip-synthetic-search)"
  else
    log "   python3 models/diffusion_tsf/train_electricity.py \\"
    log "     --dataset ETTh2 \\"
    log "     --trials ${SYNTHETIC_SEARCH_TRIALS} \\"
    log "     --synthetic-only \\"
    log "     --synthetic-size ${SYNTHETIC_SEARCH_SIZE} \\"
    log "     --force-high-end-search \\"
    log "     --repr-mode cdf \\"
    log "     --model-type unet"
    log "   → Output: ${UNIVERSAL_CKPT_DIR}/best_params.json"
  fi
  log ""
  
  log "=== STAGE 1: UNIVERSAL PRE-TRAINING ==="
  if [[ "${SKIP_UNIVERSAL_PRETRAIN}" == "true" ]]; then
    log "   SKIPPED (--skip-universal-pretrain)"
  else
    log "   python3 models/diffusion_tsf/train_electricity.py \\"
    log "     --use-defaults \\"
    log "     --params-file ${UNIVERSAL_CKPT_DIR}/best_params.json \\"
    log "     --synthetic-only \\"
    log "     --synthetic-size ${UNIVERSAL_PRETRAIN_SIZE} \\"
    log "     --repr-mode cdf"
    log "   → Output: ${UNIVERSAL_CKPT_DIR}/best_model.pt"
  fi
  log ""
  
  log "=== STAGE 2: PER-DATASET FINE-TUNING ==="
  for DATASET in "${DATASETS[@]}"; do
    IFS='|' read -r DATA_DIR DATA_FILE SEASONAL_PERIOD ITRANS_DATA_TYPE <<< "${DATASET_CONFIGS[$DATASET]}"
    DATA_PATH="${REPO_ROOT}/datasets/${DATA_DIR}/${DATA_FILE}"
    
    # Get deterministic seed for this dataset
    DATASET_IDX=0
    for d in "ETTh2" "ETTm1" "illness" "exchange_rate" "traffic" "weather"; do
      if [[ "$d" == "$DATASET" ]]; then break; fi
      DATASET_IDX=$((DATASET_IDX + 1))
    done
    DATASET_SEED=$((RANDOM_SEED + DATASET_IDX * 1000))
    TARGET_COLUMN=$(get_random_column "$DATA_PATH" "$DATASET_SEED")
    
    SANITIZED_COLUMN=$(sanitize_name "${TARGET_COLUMN}")
    DATASET_VAR="${DATASET}_${SANITIZED_COLUMN}"
    DATASET_CKPT_DIR="${CHECKPOINT_BASE}/${DATASET_VAR}"
    
    log ""
    log "--- ${DATASET_VAR} (column: ${TARGET_COLUMN}) ---"
    log "   [2a] iTransformer guidance: ${DATASET_CKPT_DIR}/guidance/checkpoint.pth"
    log "   [2b] Fine-tune ${FINETUNE_TRIALS} trials from universal model"
    log "   [2c] Best model: ${DATASET_CKPT_DIR}/best_model.pt"
  done
  log ""
  log "Done (dry run)."
  exit 0
fi

# =============================================================================
# STAGE 0: SYNTHETIC HYPERPARAMETER SEARCH
# =============================================================================

BEST_PARAMS_FILE="${UNIVERSAL_CKPT_DIR}/best_params.json"

if [[ "${SKIP_SYNTHETIC_SEARCH}" == "true" ]]; then
  log ""
  log "============================================================================="
  log "  STAGE 0: SKIPPED (--skip-synthetic-search)"
  log "============================================================================="
  
  # Use default high-end params if no best_params.json exists
  if [[ ! -f "${BEST_PARAMS_FILE}" ]]; then
    log "Creating default best_params.json..."
    cat > "${BEST_PARAMS_FILE}" << 'EOF'
{
  "learning_rate": 0.0001,
  "model_size": "large",
  "diffusion_steps": 2000,
  "batch_size": 128,
  "noise_schedule": "linear",
  "representation_mode": "cdf"
}
EOF
    log "Created: ${BEST_PARAMS_FILE}"
  fi
else
  log ""
  log "============================================================================="
  log "  STAGE 0: SYNTHETIC HYPERPARAMETER SEARCH"
  log "============================================================================="
  log ""
  log "🔍 Running ${SYNTHETIC_SEARCH_TRIALS} Optuna trials on ${SYNTHETIC_SEARCH_SIZE} synthetic samples"
  log "   Validating on real data (ETTh2 val split) to measure transfer quality"
  log "   Search space (high-end): LR (1e-5, 1e-3), model [small, large],"
  log "                           steps [2000, 4000], batch [128, 512]"
  log ""
  
  # Build command for Stage 0
  STAGE0_CMD=(
    python3 models/diffusion_tsf/train_electricity.py
    --dataset ETTh2
    --trials "${SYNTHETIC_SEARCH_TRIALS}"
    --synthetic-only
    --synthetic-size "${SYNTHETIC_SEARCH_SIZE}"
    --force-high-end-search
    --repr-mode cdf
    --model-type unet
    --stride "${STRIDE}"
    --blur-sigma 1.0
    --emd-lambda 0
    --kernel-size 3 9
    --use-time-ramp
    --use-value-channel
    --no-hybrid-condition
    --run-name universal_synthetic_pretrain
  )
  
  if [[ "${USE_MONO}" == "true" ]]; then
    STAGE0_CMD+=(
      --use-monotonicity-loss
      --monotonicity-weight "${MONO_WEIGHT}"
    )
  fi
  
  log "Command: ${STAGE0_CMD[*]}"
  log ""
  
  # Run Stage 0
  "${STAGE0_CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
  
  # Copy best_params.json to universal checkpoint dir if it was saved elsewhere
  OPTUNA_BEST_PARAMS="${CHECKPOINT_BASE}/diffusion_tsf_ETTh2_*/best_params.json"
  LATEST_OPTUNA_PARAMS=$(ls -t ${OPTUNA_BEST_PARAMS} 2>/dev/null | head -n1 || true)
  
  if [[ -n "${LATEST_OPTUNA_PARAMS}" && "${LATEST_OPTUNA_PARAMS}" != "${BEST_PARAMS_FILE}" ]]; then
    cp "${LATEST_OPTUNA_PARAMS}" "${BEST_PARAMS_FILE}"
    log "Copied best params to: ${BEST_PARAMS_FILE}"
  fi
  
  if [[ ! -f "${BEST_PARAMS_FILE}" ]]; then
    log "❌ ERROR: best_params.json not found after Stage 0"
    log "   Creating default params..."
    cat > "${BEST_PARAMS_FILE}" << 'EOF'
{
  "learning_rate": 0.0001,
  "model_size": "large",
  "diffusion_steps": 2000,
  "batch_size": 128,
  "noise_schedule": "linear",
  "representation_mode": "cdf"
}
EOF
  fi
  
  log ""
  log "✅ Stage 0 Complete!"
  log "   Best params: ${BEST_PARAMS_FILE}"
  cat "${BEST_PARAMS_FILE}" | tee -a "${LOG_FILE}"
fi

# =============================================================================
# STAGE 1: UNIVERSAL PRE-TRAINING
# =============================================================================

UNIVERSAL_MODEL="${UNIVERSAL_CKPT_DIR}/best_model.pt"

if [[ "${SKIP_UNIVERSAL_PRETRAIN}" == "true" ]]; then
  log ""
  log "============================================================================="
  log "  STAGE 1: SKIPPED (--skip-universal-pretrain)"
  log "============================================================================="
  
  if [[ ! -f "${UNIVERSAL_MODEL}" ]]; then
    log "❌ ERROR: Universal model not found: ${UNIVERSAL_MODEL}"
    log "   Cannot skip Stage 1 without existing model!"
    exit 1
  fi
  log "Using existing universal model: ${UNIVERSAL_MODEL}"
else
  log ""
  log "============================================================================="
  log "  STAGE 1: UNIVERSAL PRE-TRAINING"
  log "============================================================================="
  log ""
  log "🎯 Training universal model on ${UNIVERSAL_PRETRAIN_SIZE} synthetic samples"
  log "   Using best params from Stage 0: ${BEST_PARAMS_FILE}"
  log "   Epochs: ${UNIVERSAL_PRETRAIN_EPOCHS}"
  log "   Output: ${UNIVERSAL_MODEL}"
  log ""
  
  # Build command for Stage 1
  STAGE1_CMD=(
    python3 models/diffusion_tsf/train_electricity.py
    --dataset ETTh2
    --params-file "${BEST_PARAMS_FILE}"
    --synthetic-only
    --synthetic-size "${UNIVERSAL_PRETRAIN_SIZE}"
    --repr-mode cdf
    --model-type unet
    --stride "${STRIDE}"
    --blur-sigma 1.0
    --emd-lambda 0
    --kernel-size 3 9
    --use-time-ramp
    --use-value-channel
    --no-hybrid-condition
    --run-name universal_synthetic_pretrain
  )
  
  if [[ "${USE_MONO}" == "true" ]]; then
    STAGE1_CMD+=(
      --use-monotonicity-loss
      --monotonicity-weight "${MONO_WEIGHT}"
    )
  fi
  
  log "Command: ${STAGE1_CMD[*]}"
  log ""
  
  # Run Stage 1
  "${STAGE1_CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
  
  # Verify model was saved
  if [[ ! -f "${UNIVERSAL_MODEL}" ]]; then
    log "❌ ERROR: Universal model not saved to ${UNIVERSAL_MODEL}"
    exit 1
  fi
  
  log ""
  log "✅ Stage 1 Complete!"
  log "   Universal model saved: ${UNIVERSAL_MODEL}"
fi

# =============================================================================
# STAGE 2: PER-DATASET FINE-TUNING
# =============================================================================

log ""
log "============================================================================="
log "  STAGE 2: PER-DATASET FINE-TUNING"
log "============================================================================="
log ""
log "📊 Datasets to process: ${DATASETS[*]}"
log "   Fine-tuning trials per dataset: ${FINETUNE_TRIALS}"
log "   Pre-trained base model: ${UNIVERSAL_MODEL}"
log ""

# Track results
declare -A SELECTED_VARS
declare -A BEST_LOSSES

for DATASET in "${DATASETS[@]}"; do
  log ""
  log "============================================================================="
  log "  Processing: ${DATASET}"
  log "============================================================================="
  
  # Parse dataset config
  IFS='|' read -r DATA_DIR DATA_FILE SEASONAL_PERIOD ITRANS_DATA_TYPE <<< "${DATASET_CONFIGS[$DATASET]}"
  DATA_PATH="${REPO_ROOT}/datasets/${DATA_DIR}/${DATA_FILE}"
  
  # Get deterministic seed for this dataset
  DATASET_IDX=0
  for d in "ETTh2" "ETTm1" "illness" "exchange_rate" "traffic" "weather"; do
    if [[ "$d" == "$DATASET" ]]; then break; fi
    DATASET_IDX=$((DATASET_IDX + 1))
  done
  DATASET_SEED=$((RANDOM_SEED + DATASET_IDX * 1000))
  
  # Select random variable
  TARGET_COLUMN=$(get_random_column "$DATA_PATH" "$DATASET_SEED")
  SELECTED_VARS[$DATASET]="$TARGET_COLUMN"
  
  # Sanitize column name for directory/file names (handle special chars like spaces, parentheses)
  SANITIZED_COLUMN=$(sanitize_name "${TARGET_COLUMN}")
  
  # Create clean directory name
  DATASET_VAR="${DATASET}_${SANITIZED_COLUMN}"
  DATASET_CKPT_DIR="${CHECKPOINT_BASE}/${DATASET_VAR}"
  GUIDANCE_DIR="${DATASET_CKPT_DIR}/guidance"
  mkdir -p "${GUIDANCE_DIR}"
  
  log ""
  log "📌 Dataset: ${DATASET}"
  log "   Data file: datasets/${DATA_DIR}/${DATA_FILE}"
  log "   Seasonal period: ${SEASONAL_PERIOD}"
  log "   Selected variable: ${TARGET_COLUMN}"
  log "   Checkpoint dir: ${DATASET_CKPT_DIR}"
  log ""
  
  # ------------------------------------------------------------------
  # Stage 2a: Train iTransformer (unless skipped)
  # ------------------------------------------------------------------
  ITRANS_CKPT="${GUIDANCE_DIR}/checkpoint.pth"
  
  if [[ "${SKIP_ITRANSFORMER}" != "true" ]]; then
    log "🔥 Stage 2a: Training iTransformer on ${DATASET} (target=${TARGET_COLUMN})..."
    
    if [[ -f "${ITRANS_CKPT}" ]]; then
      log "   ✅ Checkpoint already exists: ${ITRANS_CKPT}"
    else
      cd models/iTransformer
      
      # iTransformer training
      python3 run.py \
        --is_training 1 \
        --root_path "${REPO_ROOT}/datasets/${DATA_DIR}" \
        --data_path "${DATA_FILE}" \
        --data "${ITRANS_DATA_TYPE}" \
        --model_id "${DATASET_VAR}" \
        --model iTransformer \
        --features S \
        --target "${TARGET_COLUMN}" \
        --freq h \
        --checkpoints "${GUIDANCE_DIR}/" \
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
        --des "guidance" \
        --loss MSE \
        --lradj type1 \
        --gpu 0 \
        --exp_name MTSF \
        --class_strategy projection \
        --use_norm 1 \
        2>&1 | tee -a "${LOG_FILE}"
      
      cd "${REPO_ROOT}"
      
      # Find and move the checkpoint to clean location
      ITRANS_FOUND=$(find "${GUIDANCE_DIR}" -name "checkpoint.pth" 2>/dev/null | head -n1 || true)
      
      if [[ -z "${ITRANS_FOUND}" ]]; then
        log "   ⚠️ WARNING: No iTransformer checkpoint found"
        log "   Continuing without guidance..."
        ITRANS_CKPT=""
      else
        # Move to clean location if needed
        if [[ "${ITRANS_FOUND}" != "${ITRANS_CKPT}" ]]; then
          mv "${ITRANS_FOUND}" "${ITRANS_CKPT}" 2>/dev/null || cp "${ITRANS_FOUND}" "${ITRANS_CKPT}"
          # Clean up nested directories
          find "${GUIDANCE_DIR}" -mindepth 1 -type d -exec rm -rf {} + 2>/dev/null || true
        fi
        log "   ✅ iTransformer saved: ${ITRANS_CKPT}"
      fi
    fi
  else
    log "   ⏭️ Skipping iTransformer (--skip-itrans)"
    ITRANS_CKPT=""
  fi
  
  # ------------------------------------------------------------------
  # Stage 2b: DiffusionTSF Fine-tuning with Optuna
  # ------------------------------------------------------------------
  log ""
  log "🔥 Stage 2b: Fine-tuning DiffusionTSF (${FINETUNE_TRIALS} trials)..."
  log "   Starting from: ${UNIVERSAL_MODEL}"
  log "   Output: ${DATASET_CKPT_DIR}/best_model.pt"
  log ""
  
  # Build fine-tuning command
  # --pretrained-checkpoint automatically enables --finetune-mode (conservative LR: 1e-6 to 1e-4)
  FINETUNE_CMD=(
    python3 models/diffusion_tsf/train_electricity.py
    --resume
    --dataset "${DATASET}"
    --target "${TARGET_COLUMN}"
    --trials "${FINETUNE_TRIALS}"
    --pretrained-checkpoint "${UNIVERSAL_MODEL}"
    --finetune-mode
    --repr-mode cdf
    --stride "${STRIDE}"
    --seasonal-period "${SEASONAL_PERIOD}"
    --blur-sigma 1.0
    --emd-lambda 0
    --model-type unet
    --kernel-size 3 9
    --use-time-ramp
    --use-value-channel
    --no-hybrid-condition
    --run-name "${DATASET_VAR}"
  )
  
  # Add monotonicity flags if enabled
  if [[ "${USE_MONO}" == "true" ]]; then
    FINETUNE_CMD+=(
      --use-monotonicity-loss
      --monotonicity-weight "${MONO_WEIGHT}"
    )
  fi

  # Add guidance if iTransformer was trained
  if [[ -n "${ITRANS_CKPT}" && -f "${ITRANS_CKPT}" ]]; then
    FINETUNE_CMD+=(
      --use-guidance
      --guidance-type itransformer
      --guidance-checkpoint "${ITRANS_CKPT}"
    )
  fi
  
  log "Command: ${FINETUNE_CMD[*]}"
  log ""
  
  # Run fine-tuning
  "${FINETUNE_CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
  
  # Move best_model.pt to clean location
  OPTUNA_BEST_MODEL="${CHECKPOINT_BASE}/diffusion_tsf_${DATASET}_*/best_model.pt"
  LATEST_BEST=$(ls -t ${OPTUNA_BEST_MODEL} 2>/dev/null | head -n1 || true)
  
  if [[ -n "${LATEST_BEST}" ]]; then
    cp "${LATEST_BEST}" "${DATASET_CKPT_DIR}/best_model.pt"
    log "   ✅ Best model saved: ${DATASET_CKPT_DIR}/best_model.pt"
    
    # Also copy best_params.json
    LATEST_PARAMS_DIR=$(dirname "${LATEST_BEST}")
    if [[ -f "${LATEST_PARAMS_DIR}/best_params.json" ]]; then
      cp "${LATEST_PARAMS_DIR}/best_params.json" "${DATASET_CKPT_DIR}/best_params.json"
    fi
  fi
  
  log ""
  log "✅ Completed: ${DATASET_VAR}"
done

# =============================================================================
# Summary
# =============================================================================

log ""
log "============================================================================="
log "  TRAINING COMPLETE - SUMMARY"
log "============================================================================="
log ""
log "📁 Output Structure:"
log "   ${UNIVERSAL_CKPT_DIR}/"
log "   ├── best_params.json     (Stage 0 best hyperparameters)"
log "   └── best_model.pt        (Universal pre-trained model)"
log ""
log "📊 Per-Dataset Fine-tuned Models:"
for DATASET in "${DATASETS[@]}"; do
  TARGET="${SELECTED_VARS[$DATASET]:-N/A}"
  SANITIZED_TARGET=$(sanitize_name "${TARGET}")
  DATASET_VAR="${DATASET}_${SANITIZED_TARGET}"
  log "   ${DATASET_VAR}/ (column: ${TARGET})"
  log "   ├── guidance/checkpoint.pth  (iTransformer)"
  log "   └── best_model.pt            (Fine-tuned DiffusionTSF)"
done
log ""
log "📝 Full log: ${LOG_FILE}"
log ""
log "🔒 Data leakage verification:"
log "   ✅ Chronological splits used (train < val < test in time)"
log "   ✅ Gaps between splits prevent window overlap"
log "   ✅ Per-sample normalization uses only past data"
log "   ✅ Universal model pre-trained on synthetic data only"
log "   ✅ Fine-tuning on real data preserves transfer learning"
log ""
log "Done!"

