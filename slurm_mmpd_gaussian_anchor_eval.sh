#!/bin/bash
# =============================================================================
# MMPD vs Gaussian-anchor comparison (train MMPD, eval shared 50% test subset).
#
# USAGE (from repo root on Killarney login node, preferably $SCRATCH/ts-sandbox):
#   ./slurm_mmpd_gaussian_anchor_eval.sh --smoke-test
#   ./slurm_mmpd_gaussian_anchor_eval.sh
#   ./slurm_mmpd_gaussian_anchor_eval.sh --skip-mmpd-train   # reuse trained MMPD ckpts
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE=0
SKIP_MMPD_TRAIN=0
SEED=2026
WALL="${WALL:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-test|--smoke) SMOKE=1; shift ;;
        --skip-mmpd-train) SKIP_MMPD_TRAIN=1; shift ;;
        --seed) SEED="$2"; shift 2 ;;
        --walltime|--time) WALL="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    if [[ "$SMOKE" -eq 1 ]]; then
        WALL="0:45:00"
        MEM="24G"
        CPUS=4
        JOB_NAME="mmpd-anchor-eval-smoke"
    else
        [[ -z "$WALL" ]] && WALL="2-00:00:00"
        MEM="60G"
        CPUS=8
        JOB_NAME="mmpd-anchor-eval"
    fi

    echo "Submitting ${JOB_NAME} wall=${WALL} on Killarney L40S..."
    sbatch \
        --job-name="$JOB_NAME" \
        --account=aip-boyuwang \
        --time="$WALL" \
        --nodes=1 \
        --gres=gpu:l40s:1 \
        --cpus-per-task="$CPUS" \
        --mem="$MEM" \
        --output=/dev/null \
        --error=/dev/null \
        --mail-type=END,FAIL \
        --mail-user=ccao87@uwo.ca \
        "$SCRIPT_DIR/slurm_mmpd_gaussian_anchor_eval.sh" \
        $([[ "$SMOKE" -eq 1 ]] && echo --smoke-test) \
        $([[ "$SKIP_MMPD_TRAIN" -eq 1 ]] && echo --skip-mmpd-train) \
        --seed "$SEED" \
        $([[ -n "$WALL" && "$SMOKE" -eq 0 ]] && echo --walltime "$WALL")
    exit 0
fi

cd "$SLURM_SUBMIT_DIR"
SCRIPT_DIR="$SLURM_SUBMIT_DIR"
PROJECT_ROOT="$SLURM_SUBMIT_DIR"
if [[ ! -f "$PROJECT_ROOT/utils/eval_mmpd_gaussian_anchor.py" ]]; then
    echo "ERROR: submit from the ts-sandbox repo root." >&2
    exit 1
fi
if [[ "$PROJECT_ROOT" == /home/* ]]; then
    echo "ERROR: Killarney GPU jobs should run from a scratch/project checkout, not /home." >&2
    exit 1
fi

RUN_STEM="$(date +%m-%d)-${SLURM_JOB_ID: -4}-mmpd-anchor-eval$([[ "$SMOKE" -eq 1 ]] && echo -smoke)"
LOG_FILE="./results/logs/${RUN_STEM}.log"
OUTPUT_DIR="./results/datasets/${RUN_STEM}"
mkdir -p "$(dirname "$LOG_FILE")" "$OUTPUT_DIR"
exec >>"$LOG_FILE" 2>&1

echo "=========================================="
echo "Job: $SLURM_JOB_NAME  ID: $SLURM_JOB_ID  Node: ${SLURMD_NODENAME:-unknown}"
echo "Repo: $PROJECT_ROOT"
echo "Output: $OUTPUT_DIR"
echo "GPU: $(nvidia-smi -L 2>/dev/null | head -1 || echo none)"
echo "Started: $(date)"
echo "=========================================="

module purge || true
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9

echo "[setup] Building venv on \$SLURM_TMPDIR..."
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index \
    'torch==2.11.0+computecanada' numpy pandas scipy scikit-learn tqdm einops optuna \
    -q
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA required; check torch pin/modules"
print("torch", torch.__version__, "gpu", torch.cuda.get_device_name(0))
PY

export PYTHONUNBUFFERED=1

# Latest finished Gaussian-anchor runs (auto-discovery also scans results/ckpts).
ANCHOR_ROOTS=(
    "./results/ckpts/05-26-3037-gauss-anchor-etth1"
    "./results/ckpts/05-26-3038-gauss-anchor-etth2"
    "./results/ckpts/05-26-3257-gauss-anchor-exchange_rate"
)
ANCHOR_ARGS=()
for root in "${ANCHOR_ROOTS[@]}"; do
    if [[ -d "$root" ]]; then
        ANCHOR_ARGS+=(--anchor-root "$root")
    else
        echo "[warn] missing anchor run dir: $root"
    fi
done

EVAL_ARGS=(
    --output-dir "$OUTPUT_DIR"
    --ckpt-base "$PROJECT_ROOT/results/ckpts"
    --mmpd-repo "$PROJECT_ROOT/temp/MMPD"
    --mmpd-data-dir "$PROJECT_ROOT/temp/mmpd_datasets"
    --seed "$SEED"
    --no-update-mmpd
    "${ANCHOR_ARGS[@]}"
)

if [[ "$SMOKE" -eq 1 ]]; then
    EVAL_ARGS+=(
        --datasets ETTh1
        --mmpd-train-epochs 1
        --mmpd-patience 1
        --test-fraction 0.02
        --test-max-items 32
        --sample-num 5
        --num-sampling-steps 5
        --gmm-components 5
        --gmm-iterations 3
        --mmpd-batch-size 16
        --mmpd-eval-batch-size 4
        --anchor-batch-size 4
    )
else
    EVAL_ARGS+=(
        --datasets ETTh1 ETTh2 exchange_rate
        --mmpd-train-epochs 20
        --mmpd-patience 5
        --test-fraction 0.5
        --sample-num 100
        --num-sampling-steps 20
        --gmm-components 10
        --gmm-iterations 10
        --mmpd-batch-size 32
        --mmpd-eval-batch-size 16
        --anchor-batch-size 16
    )
fi

if [[ "$SKIP_MMPD_TRAIN" -eq 1 ]]; then
    EVAL_ARGS+=(--skip-mmpd-train)
fi

echo "[eval] MMPD vs Gaussian-anchor comparison..."
python -u "$PROJECT_ROOT/utils/eval_mmpd_gaussian_anchor.py" "${EVAL_ARGS[@]}"

echo "=========================================="
echo "Done: $(date)"
echo "Log: $LOG_FILE"
echo "Metrics: $OUTPUT_DIR/metrics.json"
echo "=========================================="
