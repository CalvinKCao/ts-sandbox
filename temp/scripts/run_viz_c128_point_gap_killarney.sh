#!/bin/bash
# Killarney util job: canvas128 binary↔MMPD top-10 anchor-gap + guidance redbox.
# NOT a train wrapper — runs temp/scripts/viz_c128_point_gap_and_redbox.py on L40S.
#
# From $SCRATCH/ts-sandbox-ordinal-fine:
#   DATASETS=ETTh2 SMOKE=1 ./temp/scripts/run_viz_c128_point_gap_killarney.sh
#   DATASETS=all ./temp/scripts/run_viz_c128_point_gap_killarney.sh
#   DATASETS=ETTh1,ETTh2,electricity ./temp/scripts/run_viz_c128_point_gap_killarney.sh
#
set -euo pipefail
export PATH="/opt/slurm/bin:/cm/shared/apps/slurm/current/bin:${PATH:-/usr/bin:/bin}"
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATASETS="${DATASETS:-all}"
SMOKE="${SMOKE:-0}"
TOP_K="${TOP_K:-10}"
VARS="${VARS:-99}"
RB_VARS="${RB_VARS:-0}"
TIME_LIM="${TIME_LIM:-6:00:00}"
export DATASETS SMOKE TOP_K VARS RB_VARS

JOB_TAG="$(echo "$DATASETS" | tr ':,/' '-' | tr '[:upper:]' '[:lower:]' | cut -c1-24)"
JOB_NAME="viz-c128-gap-${JOB_TAG}"

if [ -z "${SLURM_JOB_ID:-}" ]; then
    mkdir -p "$REPO_ROOT/results/slurm"
    echo "Submitting ${JOB_NAME} (L40S, ${TIME_LIM}) datasets=$DATASETS smoke=$SMOKE from $REPO_ROOT ..."
    sbatch \
        --job-name="$JOB_NAME" \
        --account=aip-boyuwang \
        --time="$TIME_LIM" \
        --nodes=1 \
        --gres=gpu:l40s:1 \
        --cpus-per-task=8 \
        --mem=48G \
        --exclude=kn010 \
        --export=ALL \
        --output="$REPO_ROOT/results/slurm/%x-%j.out" \
        --error="$REPO_ROOT/results/slurm/%x-%j.err" \
        --mail-type=END,FAIL \
        --mail-user=ccao87@uwo.ca \
        "$SCRIPT_DIR/run_viz_c128_point_gap_killarney.sh"
    exit 0
fi

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID   Node: ${SLURMD_NODENAME:-unknown}"
echo "GPU:    $(nvidia-smi -L 2>/dev/null | head -1 || echo unknown)"
echo "Started: $(date)"
echo "DATASETS=$DATASETS SMOKE=$SMOKE TOP_K=$TOP_K"
echo "=========================================="

# Non-interactive / bare --export=ALL submits may lack Lmod; init before module.
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
assert torch.cuda.is_available(), "CUDA unavailable in job env"
print(f"torch={torch.__version__} cuda={torch.version.cuda} device={torch.cuda.get_device_name(0)}")
PY

OUT_ROOT="$PROJECT_ROOT/temp/lean_disc_c128_results/viz_point_gap"
mkdir -p "$OUT_ROOT" "$PROJECT_ROOT/results/slurm"

# Legacy subset-dir + guidance name links for first-five ckpts
python - <<'PY'
from pathlib import Path
pairs = [
    ("08-03-4571065-ETTh1-binary_window_norm_patch_refine_canvas128_p64x6", "ETTh1_7v_s1", "ETTh1"),
    ("08-04-4601319-ETTh2-binary_window_norm_patch_refine_canvas128_p64x6_etth2", "ETTh2_7v_s1", "ETTh2"),
    ("08-04-4597056-exchange_rate-binary_window_norm_patch_refine_canvas128_p64x6_exchange_rate", "exchange_rate_8v_s1", "exchange_rate"),
]
base = Path("results/ckpts")
for stem, sid, legacy in pairs:
    ckpt = base / stem
    if not ckpt.is_dir():
        print(f"[skip] missing {ckpt}")
        continue
    want_dir = ckpt / sid
    leg_dir = ckpt / legacy
    if not want_dir.exists() and leg_dir.is_dir():
        want_dir.symlink_to(legacy)
        print(f"[link] {want_dir} -> {legacy}")
    for suffix in ("_patch_guidance.pt", "_patch_guidance_hp_best.pt"):
        want = ckpt / f"{sid}{suffix}"
        leg = ckpt / f"{legacy}{suffix}"
        if not want.exists() and leg.is_file():
            want.symlink_to(leg.name)
            print(f"[link] {want.name} -> {leg.name}")
PY

CMD=(python -u temp/scripts/viz_c128_point_gap_and_redbox.py --out-root "$OUT_ROOT" --device cuda)
if [ "$SMOKE" = "1" ]; then
    CMD+=(--test-max-items 24 --top-k 5 --variables-to-plot 2 --redbox-variables-to-plot 2)
else
    CMD+=(--top-k "$TOP_K" --variables-to-plot "$VARS" --redbox-variables-to-plot "$RB_VARS")
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
