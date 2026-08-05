#!/bin/bash
# Short Killarney A/B for disc ~0.50 AUROC audit.
# Reuses disc_forecast_cache; no rematerialize. ETTh1 L8 only.
#
# From $SCRATCH/ts-sandbox-ordinal-fine:
#   ./temp/scripts/submit_disc_auroc_audit_ab.sh
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [ -z "${SLURM_JOB_ID:-}" ]; then
    mkdir -p "$REPO_ROOT/results/slurm"
    sbatch \
        --job-name=disc-auroc-audit-ab \
        --account=aip-boyuwang \
        --time=1:30:00 \
        --nodes=1 \
        --gres=gpu:l40s:1 \
        --cpus-per-task=8 \
        --mem=50G \
        --exclude=kn010 \
        --export=ALL \
        --output="$REPO_ROOT/results/slurm/%x-%j.out" \
        --error="$REPO_ROOT/results/slurm/%x-%j.err" \
        --mail-type=END,FAIL \
        --mail-user=ccao87@uwo.ca \
        "$SCRIPT_DIR/submit_disc_auroc_audit_ab.sh" "$@"
    exit 0
fi

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID   Node: ${SLURMD_NODENAME:-unknown}"
echo "GPU:    $(nvidia-smi -L 2>/dev/null | head -1 || echo unknown)"
echo "Started: $(date)"
echo "=========================================="

module purge || true
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
    PROJECT_ROOT="$(cd "$SLURM_SUBMIT_DIR" && pwd)"
else
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$PROJECT_ROOT"
echo "PROJECT_ROOT=$PROJECT_ROOT"

REQ="${REQUIREMENTS_TXT:-$PROJECT_ROOT/setup/requirements-killarney.txt}"
virtualenv --no-download "$SLURM_TMPDIR/env"
# shellcheck disable=SC1091
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index -r "$REQ" -q
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
print(f"torch={torch.__version__} device={torch.cuda.get_device_name(0)}")
PY

STAMP="$(date +%m-%d-%H%M)"
CKPT="${CKPT:-results/ckpts/08-03-4571065-ETTh1-binary_window_norm_patch_refine_canvas128_p64x6}"
CFG="${DISC_CONFIG:-configs/binary_window_norm_patch_refine_canvas128_p64x6.yaml}"
RUN_SPEC="window_norm_c128:${CKPT}:${CFG}"
MMPD_ROOT="${MMPD_ROOT:-results/datasets/07-29-0151-h96-ordinal-binary-nonordinal-mmpd-mmpd}"
BASE_OUT="results/datasets/${STAMP}-disc-auroc-audit-ab-ETTh1"
mkdir -p "$BASE_OUT" results/slurm

COMMON=(
    --dataset ETTh1
    --runs "$RUN_SPEC"
    --lookback 336
    --horizon 96
    --pack-test-stride 4
    --pack-splits val,test
    --train-fraction 0.8
    --val-fraction 0
    --fake-agg sample0
    --slice-lengths 8
    --candidate-only
    --num-sampling-steps 20
    --mmpd-output-root "$MMPD_ROOT"
    --reuse-forecast-cache
    --no-redbox-viz
    --epochs 8
    --patience 3
    --batch-size 128
)

# A: protocol default (unique_abs + bin-center) — expect ~0.50
OUT_A="${BASE_OUT}/A_unique_abs_binc"
mkdir -p "$OUT_A"
echo "===== A: unique_abs + bin-center ====="
python temp/scripts/eval_ablation_disc_l8_l16.py \
    "${COMMON[@]}" \
    --output-dir "$OUT_A" \
    --disc-bin-center-shift \
    --unique-absolute-slices \
    "$@"

# B: dense offsets (no unique_abs), same bin-center — tests inflation/leakage
OUT_B="${BASE_OUT}/B_dense_binc"
mkdir -p "$OUT_B"
echo "===== B: dense + bin-center ====="
python temp/scripts/eval_ablation_disc_l8_l16.py \
    "${COMMON[@]}" \
    --output-dir "$OUT_B" \
    --disc-bin-center-shift \
    --no-unique-absolute-slices \
    --max-train-examples 120000 \
    --max-eval-examples 40000 \
    "$@"

# C: unique_abs, NO bin-center (falls back to zscore) — tests level/bias cue
OUT_C="${BASE_OUT}/C_unique_abs_noz_binc"
mkdir -p "$OUT_C"
echo "===== C: unique_abs + no bin-center (zscore) ====="
python temp/scripts/eval_ablation_disc_l8_l16.py \
    "${COMMON[@]}" \
    --output-dir "$OUT_C" \
    --no-disc-bin-center-shift \
    --unique-absolute-slices \
    "$@"

python - <<PY
import json
from pathlib import Path
base = Path("$BASE_OUT")
rows = []
for tag in ("A_unique_abs_binc", "B_dense_binc", "C_unique_abs_noz_binc"):
    p = base / tag / "auroc_table.json"
    if not p.exists():
        rows.append({"tag": tag, "missing": True})
        continue
    for row in json.loads(p.read_text()):
        rows.append({"tag": tag, **row})
out = base / "ab_summary.json"
out.write_text(json.dumps(rows, indent=2))
print("=== A/B SUMMARY ===")
for r in rows:
    print(r)
print(f"wrote {out}")
PY

echo "Finished: $(date)"
echo "BASE_OUT=$BASE_OUT"
