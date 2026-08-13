#!/bin/bash
# Killarney util job: canvas128 patch-guidance decoder-only test MSE/MAE.
# NOT a train wrapper — runs temp/scripts/eval_guidance_decoder_test_metrics.py on L40S.
#
# From $SCRATCH/ts-sandbox-ordinal-fine:
#   ./temp/scripts/submit_guidance_decoder_test_metrics_killarney.sh
#   DATASETS=ETTh1,ETTh2 ./temp/scripts/submit_guidance_decoder_test_metrics_killarney.sh
#   SMOKE=1 ./temp/scripts/submit_guidance_decoder_test_metrics_killarney.sh
#
set -euo pipefail
export PATH="/opt/slurm/bin:/cm/shared/apps/slurm/current/bin:${PATH:-/usr/bin:/bin}"
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATASETS="${DATASETS:-all}"
SMOKE="${SMOKE:-0}"
TIME_LIM="${TIME_LIM:-0:45:00}"
BATCH_SIZE="${BATCH_SIZE:-64}"
export DATASETS SMOKE BATCH_SIZE

JOB_TAG="$(echo "$DATASETS" | tr ':,/' '-' | tr '[:upper:]' '[:lower:]' | cut -c1-24)"
JOB_NAME="guid-dec-mse-${JOB_TAG}"

if [ -z "${SLURM_JOB_ID:-}" ]; then
    mkdir -p "$REPO_ROOT/results/slurm"
    echo "Submitting ${JOB_NAME} (L40S, ${TIME_LIM}) datasets=$DATASETS smoke=$SMOKE from $REPO_ROOT ..."
    sbatch \
        --job-name="$JOB_NAME" \
        --account=aip-boyuwang \
        --time="$TIME_LIM" \
        --nodes=1 \
        --gres=gpu:l40s:1 \
        --cpus-per-task=4 \
        --mem=32G \
        --exclude=kn010 \
        --export=ALL \
        --output="$REPO_ROOT/results/slurm/%x-%j.out" \
        --error="$REPO_ROOT/results/slurm/%x-%j.err" \
        --mail-type=END,FAIL \
        --mail-user=ccao87@uwo.ca \
        "$SCRIPT_DIR/submit_guidance_decoder_test_metrics_killarney.sh"
    exit 0
fi

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID   Node: ${SLURMD_NODENAME:-unknown}"
echo "GPU:    $(nvidia-smi -L 2>/dev/null | head -1 || echo unknown)"
echo "Started: $(date)"
echo "DATASETS=$DATASETS SMOKE=$SMOKE BATCH_SIZE=$BATCH_SIZE"
echo "=========================================="

if ! type module >/dev/null 2>&1; then
    if [ -f /cvmfs/soft.computecanada.ca/config/profile/bash.sh ]; then
        export SKIP_CC_CVMFS="${SKIP_CC_CVMFS:-0}"
        export FORCE_CC_CVMFS="${FORCE_CC_CVMFS:-0}"
        set +u
        # shellcheck disable=SC1091
        source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
        set -u
    elif [ -f /etc/profile.d/z00_lmod.sh ]; then
        # shellcheck disable=SC1091
        source /etc/profile.d/z00_lmod.sh
    fi
fi
type module >/dev/null 2>&1 || {
    echo "ERROR: Lmod 'module' unavailable after profile source" >&2
    exit 127
}
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
    "${SCRATCH}"/*) ;;
    *)
        if [ -d "${SCRATCH:-}/ts-sandbox-ordinal-fine" ]; then
            PROJECT_ROOT="$SCRATCH/ts-sandbox-ordinal-fine"
        elif [ -d "${SCRATCH:-}/ts-sandbox" ]; then
            PROJECT_ROOT="$SCRATCH/ts-sandbox"
        else
            echo "ERROR: cannot resolve PROJECT_ROOT under \$SCRATCH" >&2
            exit 1
        fi
        ;;
esac
cd "$PROJECT_ROOT"
echo "PROJECT_ROOT=$PROJECT_ROOT branch=$(git branch --show-current 2>/dev/null || echo '?')"

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
assert torch.cuda.is_available(), "CUDA unavailable in job env"
print(f"torch={torch.__version__} cuda={torch.version.cuda} device={torch.cuda.get_device_name(0)}")
PY

OUT_ROOT="$PROJECT_ROOT/temp/lean_disc_c128_results/guidance_decoder_test_metrics"
mkdir -p "$OUT_ROOT" "$PROJECT_ROOT/results/slurm"

CMD=(
    python -u temp/scripts/eval_guidance_decoder_test_metrics.py
    --out-dir "$OUT_ROOT"
    --device cuda
    --batch-size "$BATCH_SIZE"
)
if [ "$SMOKE" = "1" ]; then
    CMD+=(--smoke-test)
fi
if [ "$DATASETS" = "all" ]; then
    CMD+=(--all)
else
    CMD+=(--datasets "$DATASETS")
fi

echo "+ ${CMD[*]}"
"${CMD[@]}"
echo "Finished: $(date)"
echo "OUT_ROOT=$OUT_ROOT"
ls -la "$OUT_ROOT" || true
cat "$OUT_ROOT/summary_latest.md" 2>/dev/null || true
