#!/bin/bash
#SBATCH --job-name=ci-lat-precheck
#SBATCH --account=aip-boyuwang
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Same MODULE_ROOT / venv / datasets path as finetune jobs, but only runs import + CUDA smoke.
# Submit from repo root (so SLURM_SUBMIT_DIR contains slurm_ci_latent_common.inc.sh):
#
#   cd /scratch/$USER/ts-sandbox
#   sbatch slurm_ci_latent_precheck.sh
#   sbatch slurm_ci_latent_precheck.sh -- --dataset ETTh2   # one CSV only
#
# Fails fast if include missing, venv broken, torch/cuda broken, or registry CSVs missing.

set -e
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "CI latent PRECHECK  Job=$SLURM_JOB_ID  Node=$SLURMD_NODENAME"
echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "Started: $(date)"
echo "=========================================="

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
    echo "ERROR: slurm_ci_latent_common.inc.sh not found. cd to repo root before sbatch."
    exit 1
fi

EXTRA_ARGS=""
for a in "$@"; do
    [ "$a" = "--" ] && continue
    EXTRA_ARGS="$EXTRA_ARGS $a"
done

# shellcheck source=slurm_ci_latent_common.inc.sh
source "$_INC"

# Disable aggressive cleanup killing the job on normal exit from sourced trap — precheck is short.
trap - EXIT ERR SIGTERM SIGINT SIGUSR1

echo ""
echo "Running Python precheck (CUDA required on this GPU job)..."
"$PY" -u -m models.diffusion_tsf.ci_latent_precheck --require-cuda $EXTRA_ARGS

echo ""
echo "Precheck finished OK: $(date)"
echo "=========================================="
