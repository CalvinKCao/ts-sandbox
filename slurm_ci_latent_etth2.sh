#!/bin/bash
#SBATCH --job-name=ci-etth2
#SBATCH --account=aip-boyuwang
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=50G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ccao87@uwo.ca
#SBATCH --signal=B:USR1@120

# =============================================================================
# CI latent diffusion on ETTh2 — full 4-stage pipeline
#
# Logs / ckpts / cache: ./results/{logs,ckpts,datasets}/ under the repo (submit dir).
#
# Submit:
#   sbatch slurm_ci_latent_etth2.sh
#
# Wall time: pass sbatch options BEFORE the script name:
#   sbatch --time=8:00:00 slurm_ci_latent_etth2.sh -- --stage 4
#
# Smoke test:
#   sbatch --job-name=ci-etth2-smoke slurm_ci_latent_etth2.sh -- --smoke-test
# =============================================================================

set -e
export PYTHONUNBUFFERED=1

EXTRA_ARGS=""
for a in "$@"; do
    [ "$a" = "--" ] && continue
    EXTRA_ARGS="$EXTRA_ARGS $a"
done

_INC=""
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/slurm_ci_latent_common.inc.sh" ]; then
    _INC="${SLURM_SUBMIT_DIR}/slurm_ci_latent_common.inc.sh"
else
    _SD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [ -f "${_SD}/slurm_ci_latent_common.inc.sh" ]; then
        _INC="${_SD}/slurm_ci_latent_common.inc.sh"
    fi
fi
if [ -z "$_INC" ]; then
    echo "ERROR: slurm_ci_latent_common.inc.sh not found. Run sbatch from repo root."
    exit 1
fi
# shellcheck source=slurm_ci_latent_common.inc.sh
source "$_INC"

if [ ! -f "$PROJECT_ROOT/datasets/ETT-small/ETTh2.csv" ]; then
    echo "ERROR: Missing datasets/ETT-small/ETTh2.csv"
    exit 1
fi

echo ""
echo "Running: train_ci_latent_etth2 --dataset ETTh2 --stage all $EXTRA_ARGS"
echo ""

run_py --dataset ETTh2 --stage all $EXTRA_ARGS

echo ""
echo "=========================================="
echo "Finished: $(date)"
echo "Shared pretrain: $SHARED"
echo "ETTh2 finetune:  $RUNROOT/ETTh2/"
echo "Results: $DIFFUSION_TS/results/"
echo "=========================================="
