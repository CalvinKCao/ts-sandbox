#!/bin/bash
# =============================================================================
# Slurm self-resubmitting script — FactorizedDiT only (pretrain + finetune per dataset).
# Per dataset: dit (base), dit-h128, dit-pen0. U-Net ablations: branch multi-channel.
#
# USAGE (from login node):
#   ./run_experiments.sh
#   ./run_experiments.sh --smoke-test
#   ./run_experiments.sh --smoke-test electricity traffic
#   ./run_experiments.sh weather exchange_rate
#
# Run Python from this checkout instead of $SCRATCH/ts-sandbox:
#   TS_SANDBOX_PROJECT_ROOT_SUBMIT_DIR=1 ./run_experiments.sh
#
# Defaults: six smaller benchmarks on every scenario; electricity + traffic only on
# dit (base), not dit-h128 / dit-pen0 (too heavy for repeated HP tuning on ablations).
# Wall: 3h small datasets, 24h large (electricity, traffic); 15m smoke. Same as
# multi-channel run_experiments. Optional dataset args apply to all scenarios (see USAGE).
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ===========================================================================
# If NOT inside a Slurm job → submit ourselves with the right resources
# ===========================================================================
if [ -z "${SLURM_JOB_ID:-}" ]; then
    mkdir -p "$SCRIPT_DIR/results/bootstrap"
    SB_OUT='results/bootstrap/%x-%j.out'
    SB_ERR='results/bootstrap/%x-%j.err'

    SMALL_DATASETS=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "weather" "exchange_rate")
    LARGE_DATASETS=("electricity" "traffic")

    walltime_for_dataset() {
        local ds="$1" small="$2" large="$3" x
        for x in "${LARGE_DATASETS[@]}"; do
            if [ "$ds" = "$x" ]; then
                printf '%s' "$large"
                return 0
            fi
        done
        printf '%s' "$small"
    }

    IS_SMOKE=0
    if [ "${1:-}" = "--smoke-test" ]; then
        IS_SMOKE=1
        shift
    fi
    if [ "$#" -gt 0 ]; then
        DATASETS=("$@")
        for ds in "${DATASETS[@]}"; do
            DS_TAG="${ds//_/-}"
            WALLTIME="$(walltime_for_dataset "$ds" "03:00:00" "24:00:00")"
            if [ "$IS_SMOKE" -eq 1 ]; then WALLTIME="00:15:00"; fi

            for scenario in "dit" "dit-h128" "dit-pen0"; do
                JOB_NAME="${scenario}-${DS_TAG}"
                [ "$IS_SMOKE" -eq 1 ] && JOB_NAME="${JOB_NAME}-smoke"

                echo "Submitting $JOB_NAME ..."
                sbatch \
                    --job-name="$JOB_NAME" \
                    --account=aip-boyuwang \
                    --time="$WALLTIME" \
                    --nodes=1 \
                    --gres=gpu:l40s:1 \
                    --cpus-per-task=8 \
                    --mem=50G \
                    --chdir="$SCRIPT_DIR" \
                    --output="$SB_OUT" \
                    --error="$SB_ERR" \
                    --mail-type=END,FAIL \
                    --mail-user=ccao87@uwo.ca \
                    --export="ALL,SCENARIO=$scenario,DATASET=$ds,SMOKE=$IS_SMOKE" \
                    "$SCRIPT_DIR/run_experiments.sh"
            done
        done
    else
        for scenario in "dit" "dit-h128" "dit-pen0"; do
            if [ "$scenario" = "dit" ]; then
                _ds_list=("${SMALL_DATASETS[@]}" "${LARGE_DATASETS[@]}")
            else
                _ds_list=("${SMALL_DATASETS[@]}")
            fi
            for ds in "${_ds_list[@]}"; do
                DS_TAG="${ds//_/-}"
                WALLTIME="$(walltime_for_dataset "$ds" "03:00:00" "24:00:00")"
                if [ "$IS_SMOKE" -eq 1 ]; then WALLTIME="00:15:00"; fi

                JOB_NAME="${scenario}-${DS_TAG}"
                [ "$IS_SMOKE" -eq 1 ] && JOB_NAME="${JOB_NAME}-smoke"

                echo "Submitting $JOB_NAME ..."
                sbatch \
                    --job-name="$JOB_NAME" \
                    --account=aip-boyuwang \
                    --time="$WALLTIME" \
                    --nodes=1 \
                    --gres=gpu:l40s:1 \
                    --cpus-per-task=8 \
                    --mem=50G \
                    --chdir="$SCRIPT_DIR" \
                    --output="$SB_OUT" \
                    --error="$SB_ERR" \
                    --mail-type=END,FAIL \
                    --mail-user=ccao87@uwo.ca \
                    --export="ALL,SCENARIO=$scenario,DATASET=$ds,SMOKE=$IS_SMOKE" \
                    "$SCRIPT_DIR/run_experiments.sh"
            done
        done
    fi
    echo "All jobs submitted!"
    exit 0
fi

# ===========================================================================
# Inside Slurm Job — Do the actual training
# ===========================================================================
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
ALLIANCE_RUN_STEM="$(date +%m-%d)-${SLURM_JOB_ID}-${SLURM_JOB_NAME}"

RUN_RESULTS_ROOT="$SLURM_SUBMIT_DIR/results/$ALLIANCE_RUN_STEM"
RUN_LOG_DIR="$RUN_RESULTS_ROOT/logs"
RUN_CKPT_DIR="$RUN_RESULTS_ROOT/ckpts"
RUN_DATA_DIR="$RUN_RESULTS_ROOT/datasets"
mkdir -p "$RUN_LOG_DIR" "$RUN_CKPT_DIR" "$RUN_DATA_DIR"

LOG_FILENAME="$(basename "$ALLIANCE_RUN_STEM").log"
ALLIANCE_JOB_LOG="$RUN_LOG_DIR/$LOG_FILENAME"
export WANDB_NAME="$(basename "$ALLIANCE_RUN_STEM")"
export WANDB_DIR="$RUN_LOG_DIR/wandb"
mkdir -p "$WANDB_DIR"

exec >>"$ALLIANCE_JOB_LOG" 2>&1

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Scenario: $SCENARIO"
echo "Dataset: $DATASET"
echo "Started: $(date '+%m-%d %H:%M:%S')"
echo "=========================================="

module purge || true
module load StdEnv/2023 python/3.11 cuda/12.2 cudnn/8.9

if [ "${TS_SANDBOX_PROJECT_ROOT_SUBMIT_DIR:-}" = "1" ]; then
    export PROJECT_ROOT="$SLURM_SUBMIT_DIR"
elif [ -d "$SCRATCH/ts-sandbox" ]; then
    export PROJECT_ROOT="$SCRATCH/ts-sandbox"
elif [ -d "$HOME/ts-sandbox" ]; then
    export PROJECT_ROOT="$HOME/ts-sandbox"
else
    export PROJECT_ROOT="$SLURM_SUBMIT_DIR"
fi

if [ -z "${PROJECT:-}" ] && [ -d "$HOME/projects" ]; then
    shopt -s nullglob
    _m=("$HOME"/projects/def-* "$HOME"/projects/aip-*)
    shopt -u nullglob
    if [ "${#_m[@]}" -gt 0 ]; then
        export PROJECT=$(readlink -f "${_m[0]}")
    fi
fi

if [ -n "${PROJECT:-}" ]; then
    VENV_PATH="$PROJECT/$USER/diffusion-tsf/venv"
    if [ ! -d "$VENV_PATH" ]; then
        VENV_PATH="$PROJECT/$USER/diffusion-tsf-fullvar/venv"
    fi
    if [ -d "$VENV_PATH" ]; then
        export PATH="$VENV_PATH/bin:$PATH"
        echo "Reusing existing venv: $VENV_PATH"
    else
        source .venv/bin/activate
    fi
else
    source .venv/bin/activate
fi

SYNTH_CACHE_ROOT="$PROJECT_ROOT/synth_data"
mkdir -p "$SYNTH_CACHE_ROOT"

if [ ! -e "$RUN_DATA_DIR/repo" ]; then
    ln -s "$PROJECT_ROOT/datasets" "$RUN_DATA_DIR/repo"
fi

SMOKE_FLAG=""
if [ "$SMOKE" -eq 1 ]; then
    SMOKE_FLAG="--smoke-test"
fi

# guidance 0.2 first; dit-pen0 appends --guidance-penalty-weight 0 (last wins in argparse).
COMMON_ARGS=(
    "--dataset" "$DATASET"
    "--guidance-penalty-weight" "0.2"
    "--checkpoint-dir" "$RUN_CKPT_DIR"
    "--results-dir" "$RUN_DATA_DIR"
    "--synth-cache-dir" "$SYNTH_CACHE_ROOT"
    "--fresh"
    "--wandb"
)
if [ -n "$SMOKE_FLAG" ]; then
    COMMON_ARGS+=("$SMOKE_FLAG")
fi

SCENARIO_ARGS=()
if [ "$SCENARIO" == "dit" ]; then
    SCENARIO_ARGS=(--model-type dit --subset-id "dit-pen-0.2")
elif [ "$SCENARIO" == "dit-h128" ]; then
    SCENARIO_ARGS=(--model-type dit --image-height 128 --subset-id "dit-h128-pen-0.2")
elif [ "$SCENARIO" == "dit-pen0" ]; then
    SCENARIO_ARGS=(--model-type dit --guidance-penalty-weight 0 --subset-id "dit-pen-0")
else
    echo "Unknown scenario: $SCENARIO"
    exit 1
fi

echo "Running Python Pipeline..."

TARGET_DIM=7
if [ "$DATASET" = "weather" ]; then TARGET_DIM=21; fi
if [ "$DATASET" = "exchange_rate" ]; then TARGET_DIM=8; fi
if [ "$DATASET" = "ETTm1" ] || [ "$DATASET" = "ETTh1" ] || [ "$DATASET" = "ETTh2" ] || [ "$DATASET" = "ETTm2" ]; then TARGET_DIM=7; fi
if [ "$DATASET" = "electricity" ]; then TARGET_DIM=321; fi
if [ "$DATASET" = "traffic" ]; then TARGET_DIM=862; fi

echo "Running Phase 1 (Pretrain)..."
python3 models/diffusion_tsf/train_multivariate_pipeline.py \
    --mode pretrain \
    --n-variates "$TARGET_DIM" \
    "${COMMON_ARGS[@]}" \
    "${SCENARIO_ARGS[@]}"

echo "Running Phase 2 (Finetune)..."
python3 models/diffusion_tsf/train_multivariate_pipeline.py \
    --mode finetune \
    --n-variates "$TARGET_DIM" \
    "${COMMON_ARGS[@]}" \
    "${SCENARIO_ARGS[@]}"

echo "Pipeline complete."
