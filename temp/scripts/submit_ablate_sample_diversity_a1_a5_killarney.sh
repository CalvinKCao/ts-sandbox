#!/bin/bash
# =============================================================================
# Comprehensive A1–A5 sample-diversity ablation + CRPS/MSE on ETTh1 & ETTh2
# vertical_dual checkpoints (inference-only).
#
# USAGE (Killarney login, repo = $SCRATCH/ts-sandbox):
#   cd "$SCRATCH/ts-sandbox" && git pull
#   ./temp/scripts/submit_ablate_sample_diversity_a1_a5_killarney.sh
#   ./temp/scripts/submit_ablate_sample_diversity_a1_a5_killarney.sh --smoke-test
# =============================================================================

set -euo pipefail

if [[ -d "${SCRATCH:-}/ts-sandbox" ]]; then
    REPO="$SCRATCH/ts-sandbox"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/temp/scripts/ablate_sample_diversity_a1_a5.py" ]]; then
    REPO="$SLURM_SUBMIT_DIR"
else
    REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)"
fi
SCRIPT_DIR="$REPO/temp/scripts"
PY_SCRIPT="temp/scripts/ablate_sample_diversity_a1_a5.py"

IS_SMOKE=0
PY_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--smoke-test" || "$arg" == "--smoke" ]]; then
        IS_SMOKE=1
    else
        PY_ARGS+=("$arg")
    fi
done

# ---------------------------------------------------------------------------
# Login node → sbatch L40S
# ---------------------------------------------------------------------------
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    cd "$REPO"
    mkdir -p "$REPO/results/logs"
    if [[ "$IS_SMOKE" -eq 1 ]]; then
        echo "Submitting SMOKE diversity ablation (L40S, 1h) from $REPO ..."
        sbatch \
            --chdir="$REPO" \
            --job-name="ablate-div-a1a5-smoke" \
            --account=aip-boyuwang \
            --time=1:00:00 \
            --nodes=1 \
            --gres=gpu:l40s:1 \
            --cpus-per-task=4 \
            --mem=32G \
            --output="$REPO/results/logs/ablate-div-a1a5-smoke-%j.out" \
            --error="$REPO/results/logs/ablate-div-a1a5-smoke-%j.err" \
            --mail-type=END,FAIL \
            --mail-user=ccao87@uwo.ca \
            "$SCRIPT_DIR/submit_ablate_sample_diversity_a1_a5_killarney.sh" \
            --smoke-test "${PY_ARGS[@]}"
    else
        # ~48 windows × 20 samples × 7 variants × 2 datasets; L40S ~6–10h ballpark
        echo "Submitting FULL diversity+CRPS ablation (L40S, 12h) from $REPO ..."
        sbatch \
            --chdir="$REPO" \
            --job-name="ablate-div-a1a5" \
            --account=aip-boyuwang \
            --time=12:00:00 \
            --nodes=1 \
            --gres=gpu:l40s:1 \
            --cpus-per-task=8 \
            --mem=50G \
            --output="$REPO/results/logs/ablate-div-a1a5-%j.out" \
            --error="$REPO/results/logs/ablate-div-a1a5-%j.err" \
            --mail-type=END,FAIL \
            --mail-user=ccao87@uwo.ca \
            "$SCRIPT_DIR/submit_ablate_sample_diversity_a1_a5_killarney.sh" \
            "${PY_ARGS[@]}"
    fi
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
[[ -f "$REPO/$PY_SCRIPT" ]] || {
    echo "ERROR: missing $REPO/$PY_SCRIPT — git pull on feat/vertical-dual-concat"
    exit 1
}

CKPT1="$REPO/results/ckpts/07-14-4241374-ETTh1-binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_uncompressed_bs_mid_vertical_dual_per_ds_best_g_aug_fixed20/ETTh1/vertical_dual/best.pt"
CKPT2="$REPO/results/ckpts/07-15-4263255-ETTh2-binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_uncompressed_bs_mid_vertical_dual_joint_g_lr_batch_s30r20/ETTh2/vertical_dual/best.pt"
[[ -f "$CKPT1" ]] || { echo "ERROR: missing $CKPT1"; exit 1; }
[[ -f "$CKPT2" ]] || { echo "ERROR: missing $CKPT2"; exit 1; }

module purge 2>/dev/null || true
module load StdEnv/2023 python/3.11 cuda/12.2 cudnn/8.9 2>/dev/null || true
command -v virtualenv >/dev/null || { echo "ERROR: virtualenv missing after module load"; exit 1; }

echo "$(ts) [setup] node-local venv from $REQ"
virtualenv --no-download "$SLURM_TMPDIR/env"
# shellcheck source=/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index -r "$REQ" -q
python -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, torch.cuda.get_device_name(0))"

export PYTHONUNBUFFERED=1
cd "$REPO"

OUT_DIR="results/datasets/diversity_ablation_a1_a5-${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR" "$REPO/results/logs"

if [[ "$IS_SMOKE" -eq 1 ]]; then
    MODE_ARGS=(--quick --out-dir "$OUT_DIR")
else
    MODE_ARGS=(--full --out-dir "$OUT_DIR")
fi

echo "$(ts) running: python $PY_SCRIPT ${MODE_ARGS[*]} ${PY_ARGS[*]:-}"
python "$PY_SCRIPT" "${MODE_ARGS[@]}" ${PY_ARGS[@]+"${PY_ARGS[@]}"}

echo "$(ts) done → $REPO/$OUT_DIR"
ls -la "$OUT_DIR" || true
[[ -f "$OUT_DIR/summary.json" ]] && python - <<PY
import json
from pathlib import Path
p = Path("$OUT_DIR/summary.json")
data = json.loads(p.read_text())
print("--- CRPS / MSE deltas vs baseline ---")
for res in data:
    print(res["tag"])
    for name, d in (res.get("delta_vs_baseline") or {}).items():
        print(f"  {name}: d_crps={d['d_crps']:+.4f} d_mse={d['d_sample_mean_mse']:+.4f} d_uniq={d['d_unique_bins']:+.3f}")
PY
