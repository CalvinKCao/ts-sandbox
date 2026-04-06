#!/bin/bash
#SBATCH --job-name=ci-latent-multi
#SBATCH --account=aip-boyuwang
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=50G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ccao87@uwo.ca
#SBATCH --signal=B:USR1@120

# =============================================================================
# All-in-one job: shared stages 0-2 (if missing), then stages 3-4 per dataset.
#
# Prefer shorter separate jobs (parallel finetunes after bootstrap):
#   ./submit_ci_latent_multidataset_jobs.sh
#   ./submit_ci_latent_multidataset_jobs.sh -- --smoke-test
#
# This script: same work in one long allocation (default 7 days).
# Datasets: ETTh1, ETTh2, ETTm1, ETTm2, exchange_rate (7 of 8 cols, seed 42).
# Shared: models/diffusion_tsf/checkpoints_ci_etth2/
# Per-run: models/diffusion_tsf/checkpoints_ci_runs/<run_tag>/
# =============================================================================

set -e
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi -L 2>/dev/null | head -1 || echo 'unknown')"
echo "Started: $(date)"
echo "=========================================="

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

echo ""
echo "Shared dir: $SHARED"
echo "Run root:   $RUNROOT"
echo ""

if [ ! -f "$SHARED/itransformer_pretrained_synth.pt" ] || [ ! -f "$SHARED/diffusion_pretrained_H${IMAGE_H}.pt" ]; then
    echo "=== Bootstrap stages 0, 1, 2 (dataset tag ETTh2 only affects unused finetune paths) ==="
    run_py --dataset ETTh2 --stage 0 --stage4-trials 0 --no-wandb $EXTRA_ARGS
    run_py --dataset ETTh2 --stage 1 --stage4-trials 0 --no-wandb $EXTRA_ARGS
    run_py --dataset ETTh2 --stage 2 --stage4-trials 0 --no-wandb $EXTRA_ARGS
else
    echo "=== Skipping stages 0-2 (shared checkpoints present) ==="
fi

for DS in ETTh1 ETTh2 ETTm1 ETTm2; do
    echo ""
    echo "########## $DS: stage 3 (iTrans finetune) ##########"
    run_py --dataset "$DS" --stage 3 --stage4-trials 0 $EXTRA_ARGS
    echo ""
    echo "########## $DS: stage 4 (diffusion finetune + eval, ${STAGE4_TRIALS} Optuna trials) ##########"
    run_py --dataset "$DS" --stage 4 --stage4-trials "$STAGE4_TRIALS" $EXTRA_ARGS
done

echo ""
echo "########## exchange_rate (7-var subset, seed $EXCHANGE_SEED) ##########"
run_py --dataset exchange_rate --exchange-seed "$EXCHANGE_SEED" --stage 3 --stage4-trials 0 $EXTRA_ARGS
run_py --dataset exchange_rate --exchange-seed "$EXCHANGE_SEED" --stage 4 --stage4-trials "$STAGE4_TRIALS" $EXTRA_ARGS

echo ""
echo "=========================================="
echo "Finished: $(date)"
echo "Shared:   $SHARED"
echo "Per-run:  $RUNROOT"
echo "Results:  $DIFFUSION_TS/results/ci_latent_*_H${IMAGE_H}.json"
echo "=========================================="
