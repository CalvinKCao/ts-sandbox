#!/usr/bin/env bash
# Diagnostic visualization backfill; submit from a Killarney checkout.
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  REPO="$(pwd)"
  [[ -f "$REPO/temp/scripts/backfill_fullvar_gap_redbox.py" ]] || { echo "run from repo root" >&2; exit 1; }
  mkdir -p "$REPO/results/fullvar-gap-redbox-backfill/logs"
  exec sbatch --parsable --job-name=fullvar-gap-redbox --account=aip-boyuwang --time=2:00:00 \
    --nodes=1 --gres=gpu:l40s:1 --cpus-per-task=8 --mem=50G \
    --output="$REPO/results/fullvar-gap-redbox-backfill/logs/%j.log" \
    --error="$REPO/results/fullvar-gap-redbox-backfill/logs/%j.log" \
    "$REPO/temp/scripts/submit_fullvar_gap_redbox_killarney.sh"
fi

REPO="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR required}"
cd "$REPO"
if ! type module >/dev/null 2>&1; then
  # Non-interactive Slurm shells do not always source the Alliance Lmod profile.
  export SKIP_CC_CVMFS="${SKIP_CC_CVMFS:-0}"
  export FORCE_CC_CVMFS="${FORCE_CC_CVMFS:-0}"
  set +u
  source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
  set -u
fi
type module >/dev/null 2>&1 || { echo "module command unavailable" >&2; exit 127; }
module purge || true
module load StdEnv/2023 python/3.11 cuda/12.2 cudnn/8.9
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index -r setup/requirements-killarney.txt -q
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"
python -u temp/scripts/backfill_fullvar_gap_redbox.py
