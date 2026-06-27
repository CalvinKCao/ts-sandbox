#!/bin/bash
# =============================================================================
# Top anchor_mse / CRPS delta panels: binary flat-subset staged vs fair MMPD.
#
# Default: grad-accum 1.5× lr-hi binary vs 06-16-mmpd-maskae-fair-13d on the
# seven flat-subset datasets (weather, traffic, exchange_rate, solar_Alabama,
# electricity, ETTh2, ETTh1).
#
# USAGE (login node, from $SCRATCH/ts-sandbox):
#   ./slurm_fair_mmpd_vs_binary_delta.sh
#   ./slurm_fair_mmpd_vs_binary_delta.sh --smoke-test
#   ./slurm_fair_mmpd_vs_binary_delta.sh --dataset ETTh1
#   ./slurm_fair_mmpd_vs_binary_delta.sh --binary-config binary_anchor_stationary_flat_subsets_grad_accum_150_lr_lo
#   ./slurm_fair_mmpd_vs_binary_delta.sh --skip-decomposition
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SMOKE=0
DATASET=""
SKIP_DECOMP=0

BINARY_CONFIG="binary_anchor_stationary_flat_subsets_grad_accum_150_lr_hi"
MMPD_RUN="results/datasets/06-16-mmpd-maskae-fair-13d"
OUTPUT_SUFFIX="reports/fair_mmpd_vs_grad_accum_150_lr_hi"
DATASETS_CSV="weather,traffic,exchange_rate,solar_Alabama,electricity,ETTh2,ETTh1"
TOP_K=20
PROB_DRAWS=3
INFER_BINARY=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-test|--smoke) SMOKE=1; shift ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --binary-config) BINARY_CONFIG="$2"; shift 2 ;;
        --mmpd-run) MMPD_RUN="$2"; shift 2 ;;
        --output-dir) OUTPUT_SUFFIX="$2"; shift 2 ;;
        --datasets) DATASETS_CSV="$2"; shift 2 ;;
        --top-k) TOP_K="$2"; shift 2 ;;
        --prob-draws) PROB_DRAWS="$2"; shift 2 ;;
        --skip-decomposition) SKIP_DECOMP=1; shift ;;
        --no-infer-binary) INFER_BINARY=0; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Login node: submit
# ---------------------------------------------------------------------------
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    if [[ -d "${SCRATCH:-}/ts-sandbox" ]]; then
        REPO="${SCRATCH}/ts-sandbox"
    elif [[ -d "$HOME/ts-sandbox" ]]; then
        REPO="$HOME/ts-sandbox"
    else
        REPO="$SCRIPT_DIR"
    fi
    if [[ "$REPO" == /home/* ]]; then
        echo "ERROR: submit from \$SCRATCH/ts-sandbox on Killarney, not /home." >&2
        exit 1
    fi

    SMOKE_SUFFIX=""
    if [[ "$SMOKE" -eq 1 ]]; then
        SMOKE_SUFFIX="-smoke"
        WALL="0:45:00"
        MEM="24G"
        CPUS=4
        GPU="gpu:l40s:1"
        PARTITION=""
    else
        WALL="2:00:00"
        MEM="32G"
        CPUS=6
        GPU="gpu:l40s:1"
        PARTITION=""
    fi

    RUN_STEM="$(date +%m-%d)-fair-mmpd-delta${SMOKE_SUFFIX}"
    LOG_DIR="$REPO/results/logs/${RUN_STEM}"
    mkdir -p "$LOG_DIR"

    SUBMIT_DATASETS="$DATASETS_CSV"
    if [[ -n "$DATASET" ]]; then
        SUBMIT_DATASETS="$DATASET"
    fi
    if [[ "$SMOKE" -eq 1 && -z "$DATASET" ]]; then
        SUBMIT_DATASETS="ETTh1"
        TOP_K=2
        PROB_DRAWS=2
    fi

    JOB_ARGS=(
        --binary-config "$BINARY_CONFIG"
        --mmpd-run "$MMPD_RUN"
        --output-dir "$OUTPUT_SUFFIX"
        --datasets "$SUBMIT_DATASETS"
        --top-k "$TOP_K"
        --prob-draws "$PROB_DRAWS"
    )
    [[ "$INFER_BINARY" -eq 1 ]] && JOB_ARGS+=(--infer-binary)
    [[ "$SKIP_DECOMP" -eq 1 ]] && JOB_ARGS+=(--skip-decomposition)

    echo "Submitting fair MMPD vs binary delta viz (${SMOKE_SUFFIX:-full})..."
    SBATCH_ARGS=(
        --job-name="fair-mmpd-delta${SMOKE_SUFFIX}"
        --account=aip-boyuwang
        --nodes=1
        --gres="$GPU"
        --cpus-per-task="$CPUS"
        --mem="$MEM"
        --time="$WALL"
        --output="$LOG_DIR/fair-mmpd-delta${SMOKE_SUFFIX}-%j.log"
        --error="$LOG_DIR/fair-mmpd-delta${SMOKE_SUFFIX}-%j.log"
        --mail-type=END,FAIL
        --mail-user=ccao87@uwo.ca
    )
    if [[ -n "$PARTITION" ]]; then
        SBATCH_ARGS+=(--partition="$PARTITION")
    fi

    sbatch "${SBATCH_ARGS[@]}" \
        --export=ALL,SMOKE="$SMOKE",BINARY_CONFIG="$BINARY_CONFIG",MMPD_RUN="$MMPD_RUN",OUTPUT_SUFFIX="$OUTPUT_SUFFIX",DATASETS_CSV="$SUBMIT_DATASETS",TOP_K="$TOP_K",PROB_DRAWS="$PROB_DRAWS",INFER_BINARY="$INFER_BINARY",SKIP_DECOMP="$SKIP_DECOMP" \
        "$SCRIPT_DIR/slurm_fair_mmpd_vs_binary_delta.sh" \
        "${JOB_ARGS[@]}"
    exit 0
fi

# ---------------------------------------------------------------------------
# Inside the job
# ---------------------------------------------------------------------------
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID   Node: $SLURMD_NODENAME"
echo "GPU:    $(nvidia-smi -L 2>/dev/null | head -1 || echo unknown)"
echo "Started: $(date)"
echo "=========================================="

module purge || true
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9

if [[ -d "${SCRATCH:-}/ts-sandbox" ]]; then
    PROJECT_ROOT="${SCRATCH}/ts-sandbox"
elif [[ -d "$HOME/ts-sandbox" ]]; then
    PROJECT_ROOT="$HOME/ts-sandbox"
else
    PROJECT_ROOT="$SCRIPT_DIR"
fi

cd "$PROJECT_ROOT"

if [[ -d "$PROJECT_ROOT/.venv" ]]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
elif [[ -d "$SLURM_TMPDIR/env" ]]; then
    source "$SLURM_TMPDIR/env/bin/activate"
else
    virtualenv --no-download "$SLURM_TMPDIR/env"
    source "$SLURM_TMPDIR/env/bin/activate"
    pip install --no-index --upgrade pip -q 2>/dev/null || pip install --upgrade pip -q
    pip install torch numpy pandas scipy scikit-learn tqdm matplotlib einops optuna wandb -q
    [[ -f requirements.txt ]] && pip install -r requirements.txt -q || true
fi

export WANDB_MODE=offline
export PYTHONUNBUFFERED=1

PY_ARGS=()
while [[ $# -gt 0 ]]; do
    PY_ARGS+=("$1")
    shift
done

python utils/visualize_fair_mmpd_vs_binary_delta.py "${PY_ARGS[@]}"

echo "Finished: $(date)"
