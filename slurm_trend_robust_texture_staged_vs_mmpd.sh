#!/bin/bash
# =============================================================================
# Trend-robust texture eval: staged binary vs MMPD (5 datasets, Killarney).
#
# Runs utils/eval_trend_robust_texture_staged_vs_mmpd.py on existing ckpts:
#   ETTh1, dalia, traffic (old 2-stage)
#   exchange_rate, PeMS (q99.5 MS staged)
# vs MMPD checkpoints from results/datasets/06-01-mmpd-binary-aligned.
#
# Eval protocol: 100% test windows, test_stride=2, 1 probabilistic sample.
#
# USAGE (login node, from $SCRATCH/ts-sandbox):
#   ./slurm_trend_robust_texture_staged_vs_mmpd.sh --smoke-test
#   ./slurm_trend_robust_texture_staged_vs_mmpd.sh
#   ./slurm_trend_robust_texture_staged_vs_mmpd.sh --dataset traffic
#   ./slurm_trend_robust_texture_staged_vs_mmpd.sh --force
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SMOKE=0
FORCE=0
DATASET=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-test|--smoke) SMOKE=1; shift ;;
        --force) FORCE=1; shift ;;
        --dataset) DATASET="$2"; shift 2 ;;
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
        MEM="32G"
        CPUS=4
    else
        WALL="12:00:00"
        MEM="60G"
        CPUS=8
    fi

    RUN_STEM="$(date +%m-%d)-robust-texture-staged-vs-mmpd${SMOKE_SUFFIX}"
    LOG_DIR="$REPO/results/logs/${RUN_STEM}"
    mkdir -p "$LOG_DIR"

    JOB_NAME="robust-tex${SMOKE_SUFFIX}"
    if [[ -n "$DATASET" ]]; then
        JOB_NAME="robust-tex-${DATASET}${SMOKE_SUFFIX}"
    fi

    SUBMIT_ARGS=()
    [[ "$SMOKE" -eq 1 ]] && SUBMIT_ARGS+=(--smoke-test)
    [[ "$FORCE" -eq 1 ]] && SUBMIT_ARGS+=(--force)
    [[ -n "$DATASET" ]] && SUBMIT_ARGS+=(--dataset "$DATASET")

    echo "Submitting robust texture eval (L40S, wall=$WALL)..."
    sbatch \
        --job-name="$JOB_NAME" \
        --account=aip-boyuwang \
        --nodes=1 \
        --gres=gpu:l40s:1 \
        --cpus-per-task="$CPUS" \
        --mem="$MEM" \
        --time="$WALL" \
        --output="$LOG_DIR/${JOB_NAME}-%j.log" \
        --error="$LOG_DIR/${JOB_NAME}-%j.log" \
        --mail-type=FAIL \
        --mail-user=ccao87@uwo.ca \
        "$SCRIPT_DIR/slurm_trend_robust_texture_staged_vs_mmpd.sh" \
        "${SUBMIT_ARGS[@]}"
    exit 0
fi

# ---------------------------------------------------------------------------
# Compute node
# ---------------------------------------------------------------------------
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID   Node: ${SLURMD_NODENAME:-unknown}"
echo "GPU:    $(nvidia-smi -L 2>/dev/null | head -1 || echo unknown)"
echo "Started: $(date)"
echo "=========================================="

if [[ -d "${SCRATCH:-}/ts-sandbox" ]]; then
    REPO="${SCRATCH}/ts-sandbox"
elif [[ -d "$HOME/ts-sandbox" ]]; then
    REPO="$HOME/ts-sandbox"
else
    REPO="$SCRIPT_DIR"
fi
cd "$REPO"

STORE="$REPO/results"
LOG_DIR="$STORE/logs"
mkdir -p "$LOG_DIR" "$STORE/datasets"

pip_retry() {
    local max_attempts=5 delay=20 attempt
    for attempt in $(seq 1 "$max_attempts"); do
        if "$@"; then return 0; fi
        if [[ "$attempt" -lt "$max_attempts" ]]; then
            echo "[setup] pip failed (attempt ${attempt}/${max_attempts}), retry in ${delay}s..."
            sleep "$delay"
            delay=$((delay + 2))
        fi
    done
    echo "[setup] pip failed after ${max_attempts} attempts: $*" >&2
    return 1
}

install_pipeline_deps() {
    pip_retry pip install --no-index --upgrade pip -q 2>/dev/null || pip_retry pip install -U pip -q
    if ! python -c "import torch" 2>/dev/null; then
        if ! pip_retry pip install --no-index 'torch==2.11.0+computecanada' numpy pandas scipy scikit-learn tqdm -q 2>/dev/null; then
            pip_retry pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q
            pip_retry pip install numpy pandas scipy scikit-learn tqdm -q
        fi
    fi
    pip_retry pip install --no-index wandb -q 2>/dev/null || pip_retry pip install wandb -q
    pip_retry pip install optuna einops pyyaml scikit-learn -q
    if ! pip_retry pip install --no-index matplotlib -q 2>/dev/null; then
        pip_retry pip install matplotlib -q 2>/dev/null || export DIFFUSION_TSF_SKIP_VIZ=1
    fi
    python -c "import optuna, wandb, einops, yaml"
}

_load_modules() {
    module purge 2>/dev/null || true
    module load StdEnv/2023 2>/dev/null || true
    module load python/3.11 2>/dev/null || true
    module load cuda/12.2 2>/dev/null || true
    module load cudnn/8.9 2>/dev/null || true
}

VENV=""
for cand in \
    "$STORE/venv" \
    "${SCRATCH:-}/${USER}/ts-sandbox/results/venv" \
    "${SCRATCH:-}/ts-sandbox/results/venv"; do
    if [[ -x "${cand}/bin/python" ]]; then
        VENV="$cand"
        break
    fi
done

_load_modules
if [[ -n "$VENV" ]]; then
    echo "[setup] Using persistent venv: $VENV"
    # shellcheck source=/dev/null
    source "$VENV/bin/activate"
    export PATH="$VENV/bin:$PATH"
    export PYTHON="$VENV/bin/python"
    install_pipeline_deps
else
    echo "[setup] Building venv on \${SLURM_TMPDIR:-/tmp}..."
    python -m venv "${SLURM_TMPDIR:-/tmp}/env"
    VENV="${SLURM_TMPDIR:-/tmp}/env"
    # shellcheck source=/dev/null
    source "$VENV/bin/activate"
    export PATH="$VENV/bin:$PATH"
    export PYTHON="$VENV/bin/python"
    install_pipeline_deps
fi

"$PYTHON" -c "import torch; assert torch.cuda.is_available(), 'CUDA required'; print('torch', torch.__version__, 'gpu', torch.cuda.get_device_name(0))"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TS_SANDBOX_REPO="$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

MMPD_ROOT="$REPO/results/datasets/06-01-mmpd-binary-aligned"
OUTPUT_DIR="$REPO/results/datasets/06-03-trend-robust-texture-staged-vs-mmpd"
MMPD_REPO="$REPO/temp/MMPD"
MMPD_DATA="$REPO/temp/mmpd_datasets"

EVAL_ARGS=(
    --output-dir "$OUTPUT_DIR"
    --mmpd-output-root "$MMPD_ROOT"
    --mmpd-repo "$MMPD_REPO"
    --mmpd-data-dir "$MMPD_DATA"
    --test-fraction 1.0
    --test-stride 2
    --sample-num 1
    --num-sampling-steps 20
    --probabilistic-sampler dpmpp
    --gmm-components 1
    --no-update-mmpd
)

if [[ "$SMOKE" -eq 1 ]]; then
    EVAL_ARGS+=(
        --datasets ETTh1
        --test-max-items 8
        --binary-batch-size 2
        --mmpd-eval-batch-size 4
    )
else
    if [[ -n "$DATASET" ]]; then
        EVAL_ARGS+=(--datasets "$DATASET")
    fi
    EVAL_ARGS+=(
        --binary-batch-size 8
        --mmpd-eval-batch-size 16
    )
fi

if [[ "$FORCE" -eq 1 ]]; then
    EVAL_ARGS+=(--force-binary-eval --force-mmpd-eval)
fi

echo "[eval] output=$OUTPUT_DIR mmpd=$MMPD_ROOT"
"$PYTHON" -u "$REPO/utils/eval_trend_robust_texture_staged_vs_mmpd.py" "${EVAL_ARGS[@]}"

echo "=========================================="
echo "Job complete: $(date)"
echo "Metrics: $OUTPUT_DIR/metrics.json"
echo "CSV:     $OUTPUT_DIR/robust_texture_metrics.csv"
echo "Log dir: $LOG_DIR"
echo "=========================================="
