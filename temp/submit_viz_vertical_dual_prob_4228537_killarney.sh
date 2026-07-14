#!/bin/bash
# =============================================================================
# Offline dpmpp sample viz for electricity ep20 g7 control ckpt (job 4228537).
#
# USAGE (Killarney login, repo = $SCRATCH/ts-sandbox):
#   cd "$SCRATCH/ts-sandbox" && git pull
#   ./temp/submit_viz_vertical_dual_prob_4228537_killarney.sh
#   ./temp/submit_viz_vertical_dual_prob_4228537_killarney.sh --n-windows 4 --n-samples 8 --steps 20
# =============================================================================

set -euo pipefail

# Prefer scratch checkout — sbatch copies the script into Spool, so
# BASH_SOURCE is unreliable on the compute node.
if [[ -d "${SCRATCH:-}/ts-sandbox" ]]; then
    REPO="$SCRATCH/ts-sandbox"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/temp/viz_vertical_dual_prob_samples_4228537.py" ]]; then
    REPO="$SLURM_SUBMIT_DIR"
else
    REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
SCRIPT_DIR="$REPO/temp"

RUN="${VIZ_RUN:-07-13-4228537-electricity-binary_noise_sched_ablation_vertical_dual_g7p0_ep20_fulleval}"
VIZ_PY="temp/viz_vertical_dual_prob_samples_4228537.py"
PY_ARGS=("$@")
if [[ ${#PY_ARGS[@]} -eq 0 ]]; then
    PY_ARGS=(--n-windows 4 --n-samples 8 --steps 20)
fi

# ---------------------------------------------------------------------------
# Login node → sbatch L40S
# ---------------------------------------------------------------------------
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    cd "$REPO"
    mkdir -p "$REPO/results/logs"
    echo "Submitting viz job (L40S, 1h) from $REPO run=$RUN ..."
    sbatch \
        --chdir="$REPO" \
        --job-name="viz-vd-prob-4228537" \
        --account=aip-boyuwang \
        --time=1:00:00 \
        --nodes=1 \
        --gres=gpu:l40s:1 \
        --cpus-per-task=4 \
        --mem=32G \
        --output="$REPO/results/logs/viz-vd-prob-4228537-%j.out" \
        --error="$REPO/results/logs/viz-vd-prob-4228537-%j.err" \
        --mail-type=END,FAIL \
        --mail-user=ccao87@uwo.ca \
        "$SCRIPT_DIR/submit_viz_vertical_dual_prob_4228537_killarney.sh" "${PY_ARGS[@]}"
    exit 0
fi

# ---------------------------------------------------------------------------
# Compute node
# ---------------------------------------------------------------------------
ts() { date +'%d-%H:%M:%S'; }
echo "$(ts) Job=$SLURM_JOB_ID node=${SLURMD_NODENAME:-?} REPO=$REPO"
echo "$(ts) GPU=$(nvidia-smi -L 2>/dev/null | head -1 || echo unknown)"

REQ="$REPO/setup/requirements-killarney.txt"
[[ -f "$REQ" ]] || { echo "ERROR: missing $REQ"; exit 1; }
[[ -n "${SLURM_TMPDIR:-}" ]] || { echo "ERROR: SLURM_TMPDIR unset"; exit 1; }
[[ -f "$REPO/$VIZ_PY" ]] || { echo "ERROR: missing $REPO/$VIZ_PY — git pull on feat/vertical-dual-concat"; exit 1; }

BEST="$REPO/results/ckpts/$RUN/electricity_4v_s1/vertical_dual/best.pt"
GUIDE="$REPO/results/ckpts/$RUN/electricity_4v_s1_patch_guidance.pt"
[[ -f "$BEST" ]] || { echo "ERROR: missing $BEST"; exit 1; }
[[ -f "$GUIDE" ]] || { echo "ERROR: missing $GUIDE (or *_hp_best.pt)"; exit 1; }

module purge 2>/dev/null || true
module load StdEnv/2023 python/3.11 cuda/12.2 cudnn/8.9 2>/dev/null || true

echo "$(ts) [setup] node-local venv from $REQ"
virtualenv --no-download "$SLURM_TMPDIR/env"
# shellcheck source=/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index -r "$REQ" -q
python -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, torch.cuda.get_device_name(0))"

export PYTHONUNBUFFERED=1
cd "$REPO"
echo "$(ts) running: python $VIZ_PY --run $RUN ${PY_ARGS[*]}"
python "$VIZ_PY" --run "$RUN" "${PY_ARGS[@]}"

OUT="$REPO/results/datasets/$RUN/viz/eval_prob_samples_offline"
echo "$(ts) done → $OUT"
ls -la "$OUT" 2>/dev/null || true
