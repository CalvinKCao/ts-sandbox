#!/usr/bin/env bash
# Narval A100 launcher for the 512x512 / 32x32-patch oracle smoke variant.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEPS=200; WALL="1:30:00"; WINDOW_INDEX=0; VARIATE=0
while [[ $# -gt 0 ]]; do case "$1" in --steps) STEPS="$2"; shift 2;; --time) WALL="$2"; shift 2;; --window-index) WINDOW_INDEX="$2"; shift 2;; --variate) VARIATE="$2"; shift 2;; *) echo "Unknown argument: $1" >&2; exit 2;; esac; done
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  [[ "$(hostname)" == *"narval"* ]] || { echo "ERROR: submit this from Narval." >&2; exit 2; }
  REPO="${SCRATCH:-}/ts-sandbox"; [[ -d "$REPO" ]] || { echo "ERROR: missing $REPO" >&2; exit 2; }; mkdir -p "$REPO/results/logs"
  exec sbatch --job-name=ordinal-refine-512 --account=def-boyuwang --nodes=1 --gpus=a100:1 --cpus-per-task=8 --mem=80G --time="$WALL" --output="$REPO/results/logs/ordinal-refine-512-%j.log" --error="$REPO/results/logs/ordinal-refine-512-%j.log" --mail-type=FAIL --export=ALL,ORDINAL_REPO="$REPO",ORDINAL_STEPS="$STEPS",ORDINAL_WINDOW_INDEX="$WINDOW_INDEX",ORDINAL_VARIATE="$VARIATE" "$SCRIPT_DIR/submit_ordinal_patch_refinement_512_narval_a100.sh"
fi
REPO="${ORDINAL_REPO:-$SCRIPT_DIR}"; cd "$REPO"; REQ="$REPO/setup/requirements-killarney.txt"
[[ -f "$REQ" && -n "${SLURM_TMPDIR:-}" ]] || { echo "ERROR: requirements file or SLURM_TMPDIR missing" >&2; exit 1; }
module purge 2>/dev/null || true; module load StdEnv/2023 python/3.11 cuda/12.2 cudnn/8.9 2>/dev/null || true
virtualenv --no-download "$SLURM_TMPDIR/ordinal-refine-512-env"; source "$SLURM_TMPDIR/ordinal-refine-512-env/bin/activate"; pip install --no-index --upgrade pip -q; pip install --no-index -r "$REQ" -q
export PYTHONUNBUFFERED=1 PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"; python -c "import torch; assert torch.cuda.is_available(); print('[gpu]',torch.cuda.get_device_name(0))"
python -u -m experiments.ordinal_patch_refinement_killtest.run_narval_smoke_512 --steps "${ORDINAL_STEPS:-200}" --window-index "${ORDINAL_WINDOW_INDEX:-0}" --variate "${ORDINAL_VARIATE:-0}" --output "$REPO/results/ordinal_patch_refinement_killtest/narval-512-${SLURM_JOB_ID}"
