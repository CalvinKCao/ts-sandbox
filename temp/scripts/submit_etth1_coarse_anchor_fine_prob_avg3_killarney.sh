#!/bin/bash
# ETTh1 canvas128: coarse=anchor, fine=quad_t 3-sample mean (disposable probe).
#
# USAGE (Killarney login, from $SCRATCH/ts-sandbox-ordinal-fine):
#   ./temp/scripts/submit_etth1_coarse_anchor_fine_prob_avg3_killarney.sh
#   ./temp/scripts/submit_etth1_coarse_anchor_fine_prob_avg3_killarney.sh --smoke-test
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SMOKE=0
EXTRA=()
for arg in "$@"; do
    if [ "$arg" = "--smoke-test" ]; then
        SMOKE=1
    else
        EXTRA+=("$arg")
    fi
done

JOB_NAME="etth1-ca-fp-avg3"
if [ "$SMOKE" -eq 1 ]; then
    JOB_NAME="${JOB_NAME}-smoke"
    WALL="0:30:00"
    MEM="24G"
else
    WALL="2:00:00"
    MEM="40G"
fi

if [ -z "${SLURM_JOB_ID:-}" ]; then
    mkdir -p "$REPO_ROOT/results/slurm"
    echo "Submitting ${JOB_NAME} (L40S, ${WALL}, exclude kn002,kn010) from $REPO_ROOT ..."
    sbatch \
        --job-name="$JOB_NAME" \
        --account=aip-boyuwang \
        --time="$WALL" \
        --nodes=1 \
        --gres=gpu:l40s:1 \
        --cpus-per-task=8 \
        --mem="$MEM" \
        --exclude=kn002,kn010 \
        --export=ALL \
        --output="$REPO_ROOT/results/slurm/%x-%j.out" \
        --error="$REPO_ROOT/results/slurm/%x-%j.err" \
        --mail-type=END,FAIL \
        --mail-user=ccao87@uwo.ca \
        "$SCRIPT_DIR/submit_etth1_coarse_anchor_fine_prob_avg3_killarney.sh" "$@"
    exit 0
fi

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID   Node: ${SLURMD_NODENAME:-unknown}"
echo "GPU:    $(nvidia-smi -L 2>/dev/null | head -1 || echo unknown)"
echo "Started: $(date)"
echo "=========================================="

# Non-interactive sbatch may lack Lmod.
if ! type module >/dev/null 2>&1; then
    if [ -f /cvmfs/soft.computecanada.ca/config/profile/bash.sh ]; then
        export SKIP_CC_CVMFS="${SKIP_CC_CVMFS:-0}"
        set +u
        # shellcheck disable=SC1091
        source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
        set -u
    fi
fi
type module >/dev/null 2>&1 || { echo "ERROR: Lmod unavailable" >&2; exit 127; }
module purge || true
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
    PROJECT_ROOT="$(cd "$SLURM_SUBMIT_DIR" && pwd)"
else
    PROJECT_ROOT="$REPO_ROOT"
fi
case "$PROJECT_ROOT" in
    /scratch/*) ;;
    *)
        if [ -d /scratch/ccao87/ts-sandbox-ordinal-fine ]; then
            PROJECT_ROOT=/scratch/ccao87/ts-sandbox-ordinal-fine
        elif [ -d "${SCRATCH:-}/ts-sandbox-ordinal-fine" ]; then
            PROJECT_ROOT="$SCRATCH/ts-sandbox-ordinal-fine"
        else
            echo "ERROR: cannot resolve PROJECT_ROOT under scratch" >&2
            exit 1
        fi
        ;;
esac
cd "$PROJECT_ROOT"
export SCRATCH="${SCRATCH:-/scratch/ccao87}"
echo "PROJECT_ROOT=$PROJECT_ROOT"

REQ="$PROJECT_ROOT/setup/requirements-killarney.txt"
[[ -f "$REQ" ]] || { echo "ERROR: missing $REQ" >&2; exit 1; }
[[ -n "${SLURM_TMPDIR:-}" ]] || { echo "ERROR: SLURM_TMPDIR unset" >&2; exit 1; }

echo "[setup] Building node-local venv on \$SLURM_TMPDIR from $REQ"
virtualenv --no-download "$SLURM_TMPDIR/env"
# shellcheck disable=SC1091
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index -r "$REQ" -q
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
print(f"torch={torch.__version__} cuda={torch.version.cuda} device={torch.cuda.get_device_name(0)}")
PY

CKPT_DEFAULT="results/ckpts/08-03-4571065-ETTh1-binary_window_norm_patch_refine_canvas128_p64x6"
CKPT="${CKPT:-$CKPT_DEFAULT}"
if [ ! -d "$CKPT" ]; then
    CKPT="$(ls -d "$PROJECT_ROOT"/results/ckpts/*4571065* 2>/dev/null | head -1 || true)"
fi
[ -n "$CKPT" ] && [ -d "$CKPT" ] || {
    echo "ERROR: missing ETTh1 canvas128 ckpt (job 4571065)" >&2
    exit 1
}
CFG="configs/binary_window_norm_patch_refine_canvas128_p64x6.yaml"
[ -f "$CFG" ] || { echo "ERROR: missing $CFG" >&2; exit 1; }

STAMP="$(date +%m-%d-%H%M)"
if [ "$SMOKE" -eq 1 ]; then
    OUT="results/datasets/${STAMP}-etth1-coarse-anchor-fine-prob-avg3-smoke"
    SMOKE_FLAG=(--smoke-test)
else
    OUT="results/datasets/${STAMP}-etth1-coarse-anchor-fine-prob-avg3-full"
    SMOKE_FLAG=()
fi
mkdir -p "$OUT" results/slurm results/datasets

EVAL_PY="temp/scripts/eval_etth1_coarse_anchor_fine_prob_avg3.py"
[ -f "$EVAL_PY" ] || { echo "ERROR: missing $EVAL_PY" >&2; exit 1; }

python -u "$EVAL_PY" \
    --config "$CFG" \
    --ckpt "$CKPT" \
    --dataset ETTh1 \
    --n-samples 3 \
    --prob-sampler quad_t \
    --prob-steps 20 \
    --eval-test-stride 64 \
    --regular-test-stride 16 \
    --output-dir "$OUT" \
    "${SMOKE_FLAG[@]}" \
    "${EXTRA[@]}"

echo "Finished: $(date)"
echo "output_dir=$OUT"
if [ -f "$OUT/metrics.json" ]; then
    echo "----- metrics.json -----"
    cat "$OUT/metrics.json"
fi
