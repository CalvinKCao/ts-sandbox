#!/usr/bin/env bash
# Diagnostic viz: H720 nostitch patch-refine redbox (3 random test windows × 3 ckpts).
# NOT a train wrapper — runs temp/scripts/viz_h720_nostitch_redbox.py on L40S.
#
# From /scratch/ccao87/ts-sandbox-h720-itrans:
#   ./temp/scripts/run_h720_nostitch_redbox_killarney.sh
set -euo pipefail
export PATH="/opt/slurm/bin:/cm/shared/apps/slurm/current/bin:${PATH:-/usr/bin:/bin}"
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

TIME_LIM="${TIME_LIM:-1:30:00}"
export TIME_LIM

if [ -z "${SLURM_JOB_ID:-}" ]; then
    mkdir -p "$REPO_ROOT/results/slurm"
    echo "Submitting viz-h720-nostitch-rb (L40S, ${TIME_LIM}) from $REPO_ROOT ..."
    sbatch \
        --job-name=viz-h720-nostitch-rb \
        --account=aip-boyuwang \
        --time="$TIME_LIM" \
        --nodes=1 \
        --gres=gpu:l40s:1 \
        --cpus-per-task=8 \
        --mem=50G \
        --exclude=kn010 \
        --export=ALL \
        --output="$REPO_ROOT/results/slurm/%x-%j.out" \
        --error="$REPO_ROOT/results/slurm/%x-%j.out" \
        --mail-type=END,FAIL \
        --mail-user=ccao87@uwo.ca \
        "$SCRIPT_DIR/run_h720_nostitch_redbox_killarney.sh"
    exit 0
fi

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID   Node: ${SLURMD_NODENAME:-unknown}"
echo "GPU:    $(nvidia-smi -L 2>/dev/null | head -1 || echo unknown)"
echo "Started: $(date)"
echo "=========================================="

if ! type module >/dev/null 2>&1; then
    if [ -f /cvmfs/soft.computecanada.ca/config/profile/bash.sh ]; then
        export SKIP_CC_CVMFS="${SKIP_CC_CVMFS:-0}"
        export FORCE_CC_CVMFS="${FORCE_CC_CVMFS:-0}"
        set +u
        # shellcheck disable=SC1091
        source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
        set +u
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
# Fail if we landed on the electricity worktree.
case "$PROJECT_ROOT" in
    *ts-sandbox-main-fullhp*)
        echo "ERROR: refusing to run from $PROJECT_ROOT" >&2
        exit 1
        ;;
esac
cd "$PROJECT_ROOT"
echo "PROJECT_ROOT=$PROJECT_ROOT HEAD=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

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

OUT_ROOT="$PROJECT_ROOT/results/datasets/h720-nostitch-redbox"
mkdir -p "$OUT_ROOT" "$PROJECT_ROOT/results/slurm"

python -u temp/scripts/viz_h720_nostitch_redbox.py \
    --output-root "$OUT_ROOT" \
    --device cuda \
    --n-samples 3 \
    --seed 42 \
    --sampler anchor \
    --num-sampling-steps 1 \
    --variables-to-plot 1 \
    --pack-test-stride 1 \
    --lookback 336 \
    --horizon 720

echo "Finished: $(date)"
echo "OUT_ROOT=$OUT_ROOT"
find "$OUT_ROOT" -name '*refine_boxes.jpg' | sort
cat "$OUT_ROOT/window_indices.json"
