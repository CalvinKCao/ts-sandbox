#!/bin/bash
# =============================================================================
# Train iTransformer + PatchTST on canvas128 leaderboard subsets (Killarney).
#
# One sbatch per dataset (array or login-node loop). Node-local venv on
# $SLURM_TMPDIR. Patches + PeMS 60/20/20 + published per-dataset script HPs handled in
# temp/scripts/{apply,run}_baselines_canvas128_*.py.
#
# USAGE (from repo root on login node, e.g. $SCRATCH/ts-sandbox):
#   ./temp/scripts/submit_baselines_canvas128_killarney.sh --smoke-test
#   ./temp/scripts/submit_baselines_canvas128_killarney.sh
#   ./temp/scripts/submit_baselines_canvas128_killarney.sh --model itransformer --datasets ETTh1,ETTh2
# =============================================================================

set -euo pipefail

MODEL="both"
DATASETS="ETTh1,ETTh2,electricity,traffic,exchange_rate,PeMS,solar_Alabama,ETTm1,ETTm2"
SMOKE=0
TIME_FULL="2:00:00"
TIME_SMOKE="0:45:00"
WALL_OVERRIDE=""
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --datasets) DATASETS="$2"; shift 2 ;;
    --smoke-test|--smoke) SMOKE=1; shift ;;
    --time) WALL_OVERRIDE="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Login-node: submit one job per dataset
# ---------------------------------------------------------------------------
if [ -z "${SLURM_JOB_ID:-}" ]; then
  SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
  REPO_ROOT="$(pwd)"
  if [ ! -f "$REPO_ROOT/temp/scripts/run_baselines_canvas128_subset.py" ]; then
    echo "ERROR: submit from repo root (missing temp/scripts/run_baselines_canvas128_subset.py)" >&2
    exit 1
  fi

  if [ -n "$WALL_OVERRIDE" ]; then
    WALL="$WALL_OVERRIDE"
  elif [ "$SMOKE" -eq 1 ]; then
    WALL="$TIME_SMOKE"
  else
    WALL="$TIME_FULL"
  fi

  IFS=',' read -r -a DS_ARR <<< "$DATASETS"
  mkdir -p "$REPO_ROOT/results/baselines_canvas128_subset/logs"
  for ds in "${DS_ARR[@]}"; do
    ds="$(echo "$ds" | xargs)"
    [ -n "$ds" ] || continue
    JOB_NAME="base-c128-${ds}"
    [ "$SMOKE" -eq 1 ] && JOB_NAME="base-c128-smoke-${ds}"
    EXTRA=(--model "$MODEL" --datasets "$ds")
    [ "$SMOKE" -eq 1 ] && EXTRA+=(--smoke-test)
    [ "$FORCE" -eq 1 ] && EXTRA+=(--force)
    echo "[submit] $ds wall=$WALL"
    sbatch \
      --job-name="$JOB_NAME" \
      --account=aip-boyuwang \
      --time="$WALL" \
      --nodes=1 \
      --gres=gpu:l40s:1 \
      --cpus-per-task=8 \
      --mem=50G \
      --output="$REPO_ROOT/results/baselines_canvas128_subset/logs/%x-%j.out" \
      --mail-type=END,FAIL \
      --mail-user=ccao87@uwo.ca \
      "$SCRIPT_PATH" \
        "${EXTRA[@]}"
  done
  exit 0
fi

# ---------------------------------------------------------------------------
# Compute-node body
# ---------------------------------------------------------------------------
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID   Node: ${SLURMD_NODENAME:-unknown}"
echo "GPU:    $(nvidia-smi -L 2>/dev/null | head -1 || echo unknown)"
echo "Started: $(date)"
echo "=========================================="

cd "${SLURM_SUBMIT_DIR:?}"
echo "PROJECT_ROOT=$PWD"

module purge || true
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9
command -v virtualenv >/dev/null || { echo "ERROR: virtualenv missing" >&2; exit 1; }

REQ="$PWD/setup/requirements-killarney.txt"
[ -f "$REQ" ] || { echo "ERROR: missing $REQ" >&2; exit 1; }

echo "[setup] node-local venv on \$SLURM_TMPDIR"
virtualenv --no-download "$SLURM_TMPDIR/env"
# shellcheck disable=SC1091
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index -r "$REQ" -q
# Optional; iTransformer import is stubbed if missing.
pip install --no-index reformer-pytorch 2>/dev/null \
  || pip install reformer-pytorch 2>/dev/null \
  || echo "[warn] reformer-pytorch not installed; using import stub"

python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print('torch', torch.__version__, 'gpu', torch.cuda.get_device_name(0))"

export PYTHONUNBUFFERED=1

mkdir -p temp results/baselines_canvas128_subset
CLONE_LOCK="$PWD/results/baselines_canvas128_subset/clone.lock"
(
  flock 9
  if [ ! -d temp/iTransformer/.git ]; then
    rm -rf temp/iTransformer
    git clone --depth 1 https://github.com/thuml/iTransformer.git temp/iTransformer
  fi
  if [ ! -d temp/PatchTST/.git ]; then
    rm -rf temp/PatchTST
    git clone --depth 1 https://github.com/yuqinie98/PatchTST.git temp/PatchTST
  fi
  find temp/iTransformer temp/PatchTST -name '*.py' -print0 | xargs -0 sed -i 's/np\.Inf/np.inf/g' || true
  python -u temp/scripts/apply_baseline_canvas128_patches.py
  python -u temp/scripts/export_canvas128_subset_csvs.py
) 9>"$CLONE_LOCK"

EXTRA_FLAGS=()
[ "$SMOKE" -eq 1 ] && EXTRA_FLAGS+=(--smoke-test)
[ "${FORCE:-0}" -eq 1 ] && EXTRA_FLAGS+=(--force)

python -u temp/scripts/run_baselines_canvas128_subset.py \
  --model "$MODEL" \
  --dataset "$DATASETS" \
  --seq-len 336 \
  --pred-len 96 \
  --force \
  "${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}"

echo "Finished: $(date)"
