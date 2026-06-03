#!/bin/bash
# =============================================================================
# Learned discriminator texture eval: staged binary vs MMPD (Killarney L40S).
#
# Default login-node behavior submits one independent 4h job per dataset so the
# five dataset evals run in parallel. Each job trains L=8,16,32 discriminators
# for GT vs binary_staged and GT vs MMPD using stochastic sample0.
#
# USAGE (login node, from $SCRATCH/ts-sandbox):
#   ./slurm_discriminator_texture_staged_vs_mmpd.sh --smoke-test
#   ./slurm_discriminator_texture_staged_vs_mmpd.sh
#   ./slurm_discriminator_texture_staged_vs_mmpd.sh --dataset traffic
#   ./slurm_discriminator_texture_staged_vs_mmpd.sh --slice-length 16
#   ./slurm_discriminator_texture_staged_vs_mmpd.sh --force-raw-eval
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SMOKE=0
FORCE_RAW=0
FORCE_TRAIN=0
DATASET=""
SLICE_LENGTH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-test|--smoke) SMOKE=1; shift ;;
        --force-raw-eval) FORCE_RAW=1; shift ;;
        --force-train) FORCE_TRAIN=1; shift ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --slice-length) SLICE_LENGTH="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

DATASETS=(ETTh1 dalia traffic exchange_rate PeMS)

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
        WALL="0:30:00"
        MEM="24G"
        CPUS=4
    else
        WALL="4:00:00"
        MEM="50G"
        CPUS=8
    fi

    RUN_STEM="$(date +%m-%d)-disc-texture-staged-vs-mmpd${SMOKE_SUFFIX}"
    LOG_DIR="$REPO/results/logs/${RUN_STEM}"
    mkdir -p "$LOG_DIR"
    if [[ ! -d "$REPO/temp/MMPD/.git" ]]; then
        echo "Preparing temp/MMPD checkout before parallel submissions..."
        mkdir -p "$REPO/temp"
        git clone https://github.com/Thinklab-SJTU/MMPD.git "$REPO/temp/MMPD"
    fi

    SUBMIT_DATASETS=("${DATASETS[@]}")
    if [[ -n "$DATASET" ]]; then
        SUBMIT_DATASETS=("$DATASET")
    fi
    if [[ "$SMOKE" -eq 1 && -z "$DATASET" ]]; then
        SUBMIT_DATASETS=(ETTh1)
    fi

    for ds in "${SUBMIT_DATASETS[@]}"; do
        JOB_NAME="disc-tex-${ds}${SMOKE_SUFFIX}"
        SUBMIT_ARGS=(--dataset "$ds")
        [[ "$SMOKE" -eq 1 ]] && SUBMIT_ARGS+=(--smoke-test)
        [[ "$FORCE_RAW" -eq 1 ]] && SUBMIT_ARGS+=(--force-raw-eval)
        [[ "$FORCE_TRAIN" -eq 1 ]] && SUBMIT_ARGS+=(--force-train)
        [[ -n "$SLICE_LENGTH" ]] && SUBMIT_ARGS+=(--slice-length "$SLICE_LENGTH")

        echo "Submitting discriminator texture eval for $ds (L40S, wall=$WALL)..."
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
            "$SCRIPT_DIR/slurm_discriminator_texture_staged_vs_mmpd.sh" \
            "${SUBMIT_ARGS[@]}"
    done
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
mkdir -p "$STORE/logs" "$STORE/datasets"

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
    pip_retry pip install optuna einops pyyaml scikit-learn -q
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

OUTPUT_DIR="$REPO/results/datasets/06-03-discriminator-texture-staged-vs-mmpd"
RAW_EVAL_DIR="$REPO/results/datasets/06-03-trend-robust-texture-staged-vs-mmpd"
MMPD_ROOT="$REPO/results/datasets/06-01-mmpd-binary-aligned"
MMPD_REPO="$REPO/temp/MMPD"
MMPD_DATA="$REPO/temp/mmpd_datasets"

EVAL_ARGS=(
    --output-dir "$OUTPUT_DIR"
    --raw-eval-dir "$RAW_EVAL_DIR"
    --mmpd-output-root "$MMPD_ROOT"
    --mmpd-repo "$MMPD_REPO"
    --mmpd-data-dir "$MMPD_DATA"
    --test-fraction 1.0
    --test-stride 2
    --num-sampling-steps 20
    --probabilistic-sampler dpmpp
    --gmm-components 1
    --datasets "$DATASET"
    --no-update-mmpd
)

if [[ -n "$SLICE_LENGTH" ]]; then
    EVAL_ARGS+=(--slice-lengths "$SLICE_LENGTH")
fi

if [[ "$SMOKE" -eq 1 ]]; then
    EVAL_ARGS+=(
        --smoke-test
        --raw-binary-batch-size 2
        --raw-mmpd-batch-size 4
    )
else
    EVAL_ARGS+=(
        --epochs 20
        --batch-size 512
        --raw-binary-batch-size 8
        --raw-mmpd-batch-size 16
    )
fi

if [[ "$FORCE_RAW" -eq 1 ]]; then
    EVAL_ARGS+=(--force-raw-eval)
fi

if [[ "$FORCE_TRAIN" -eq 1 ]]; then
    EVAL_ARGS+=(--force-train)
fi

echo "[eval] output=$OUTPUT_DIR raw=$RAW_EVAL_DIR dataset=$DATASET"
"$PYTHON" -u "$REPO/utils/eval_discriminator_texture_staged_vs_mmpd.py" "${EVAL_ARGS[@]}"

echo "[report] regenerating if combined metrics are present"
"$PYTHON" -u "$REPO/utils/report_discriminator_texture_staged_vs_mmpd.py" \
    --metrics "$OUTPUT_DIR/metrics.json" \
    --manifest "$OUTPUT_DIR/run_manifest.json" \
    --output "$REPO/reports/06-03_discriminator_texture_staged_vs_mmpd.md" || true

echo "=========================================="
echo "Job complete: $(date)"
echo "Metrics: $OUTPUT_DIR/metrics.json"
echo "CSV:     $OUTPUT_DIR/metrics.csv"
echo "Report:  $REPO/reports/06-03_discriminator_texture_staged_vs_mmpd.md"
echo "=========================================="
