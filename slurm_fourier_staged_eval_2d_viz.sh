#!/bin/bash
# =============================================================================
# GT vs pred 2D coarse/fine + final 1D panels for recent Fourier staged ckpts.
#
# USAGE (Killarney login node, from $SCRATCH/ts-sandbox):
#   ./slurm_fourier_staged_eval_2d_viz.sh --smoke-test
#   ./slurm_fourier_staged_eval_2d_viz.sh
#   ./slurm_fourier_staged_eval_2d_viz.sh --stem 07-02-4041709-weather-binary_anchor_stationary_flat_subsets_grad_accum_150_fourier_flatline_blur
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SMOKE=0
STEM=""
CONFIG="configs/binary_anchor_stationary_flat_subsets_grad_accum_150_fourier_flatline_blur.yaml"
N_RANDOM=2
N_WORST=3

# Completed flatline_blur eval runs from the past week (dataset parsed from stem).
DEFAULT_STEMS=(
    07-02-4041709-weather-binary_anchor_stationary_flat_subsets_grad_accum_150_fourier_flatline_blur
    07-02-4041706-ETTh1-binary_anchor_stationary_flat_subsets_grad_accum_150_fourier_flatline_blur
    07-02-4041707-ETTm1-binary_anchor_stationary_flat_subsets_grad_accum_150_fourier_flatline_blur
    07-02-4041708-exchange_rate-binary_anchor_stationary_flat_subsets_grad_accum_150_fourier_flatline_blur
)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-test|--smoke) SMOKE=1; shift ;;
        --stem) STEM="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --n-random) N_RANDOM="$2"; shift 2 ;;
        --n-worst) N_WORST="$2"; shift 2 ;;
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

    if [[ "$SMOKE" -eq 1 ]]; then
        WALL="0:30:00"
        MEM="16G"
        CPUS=4
        GPU="gpu:l40s:1"
        JOB_STEMS=(07-02-4041709-weather-binary_anchor_stationary_flat_subsets_grad_accum_150_fourier_flatline_blur)
        N_RANDOM=1
        N_WORST=1
    else
        WALL="1:00:00"
        MEM="24G"
        CPUS=6
        GPU="gpu:l40s:1"
        if [[ -n "$STEM" ]]; then
            JOB_STEMS=("$STEM")
        else
            JOB_STEMS=("${DEFAULT_STEMS[@]}")
        fi
    fi

    RUN_STEM="$(date +%m-%d)-fourier-2d-viz"
    LOG_DIR="$REPO/results/logs/${RUN_STEM}"
    mkdir -p "$LOG_DIR"

    echo "Submitting fourier 2D pred viz (${#JOB_STEMS[@]} stems)..."
    sbatch \
        --job-name="fourier-2d-viz" \
        --account=aip-boyuwang \
        --nodes=1 \
        --gres="$GPU" \
        --cpus-per-task="$CPUS" \
        --mem="$MEM" \
        --time="$WALL" \
        --output="$LOG_DIR/fourier-2d-viz-%j.log" \
        --error="$LOG_DIR/fourier-2d-viz-%j.log" \
        --mail-type=END,FAIL \
        --mail-user=ccao87@uwo.ca \
        --export=ALL,SMOKE="$SMOKE",CONFIG="$CONFIG",N_RANDOM="$N_RANDOM",N_WORST="$N_WORST",STEMS_CSV="$(IFS=,; echo "${JOB_STEMS[*]}")" \
        "$SCRIPT_DIR/slurm_fourier_staged_eval_2d_viz.sh"
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
    [[ -f setup/requirements-killarney.txt ]] && pip install -r setup/requirements-killarney.txt -q || true
fi

export WANDB_MODE=offline
export PYTHONUNBUFFERED=1

IFS=',' read -ra STEMS <<< "${STEMS_CSV:-}"

for stem in "${STEMS[@]}"; do
    [[ -z "$stem" ]] && continue
    if [[ "$stem" =~ ^[0-9]{2}-[0-9]{2}-[0-9]+-([^-]+)- ]]; then
        dataset="${BASH_REMATCH[1]}"
    else
        echo "SKIP: cannot parse dataset from stem: $stem" >&2
        continue
    fi

    ckpt_dir="$PROJECT_ROOT/results/ckpts/$stem"
    results_dir="$PROJECT_ROOT/results/datasets/$stem"
    if [[ ! -d "$ckpt_dir" ]]; then
        echo "SKIP: missing ckpt dir $ckpt_dir" >&2
        continue
    fi

    echo "---- $dataset | $stem ----"
    python utils/visualize_staged_eval_2d_preds.py \
        --checkpoint-dir "$ckpt_dir" \
        --dataset "$dataset" \
        --results-dir "$results_dir" \
        --config "$CONFIG" \
        --n-random "$N_RANDOM" \
        --n-worst "$N_WORST" \
        --output-dir "$results_dir/viz/eval_2d_preds"
done

echo "Finished: $(date)"
