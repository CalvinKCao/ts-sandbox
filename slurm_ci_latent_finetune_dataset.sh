#!/bin/bash
#SBATCH --job-name=ci-lat-ft
#SBATCH --account=aip-boyuwang
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=50G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ccao87@uwo.ca
#SBATCH --signal=B:USR1@120

# One dataset: stage 3 (iTrans finetune) then stage 4 (diffusion + Optuna).
# Requires CI_DATASET in the environment (ETTh1, ETTh2, ETTm1, ETTm2, or exchange_rate).
# For exchange_rate set CI_EXCHANGE_SEED (default 42).
#
# Submitted by submit_ci_latent_multidataset_jobs.sh with afterok bootstrap.

set -e
export PYTHONUNBUFFERED=1

DS="${CI_DATASET:?Set CI_DATASET (e.g. sbatch --export=ALL,CI_DATASET=ETTh1 ...)}"
EXS="${CI_EXCHANGE_SEED:-42}"

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID  finetune dataset=$DS"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi -L 2>/dev/null | head -1 || echo 'unknown')"
echo "Started: $(date)"
echo "=========================================="

EXTRA_ARGS=""
for a in "$@"; do
    [ "$a" = "--" ] && continue
    EXTRA_ARGS="$EXTRA_ARGS $a"
done

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=slurm_ci_latent_common.inc.sh
source "$_SCRIPT_DIR/slurm_ci_latent_common.inc.sh"

echo ""
echo "Run root: $RUNROOT"
echo ""

if [ "$DS" = "exchange_rate" ]; then
    echo "########## exchange_rate seed=$EXS: stage 3 ##########"
    run_py --dataset exchange_rate --exchange-seed "$EXS" --stage 3 --stage4-trials 0 $EXTRA_ARGS
    echo "########## exchange_rate seed=$EXS: stage 4 (${STAGE4_TRIALS} trials) ##########"
    run_py --dataset exchange_rate --exchange-seed "$EXS" --stage 4 --stage4-trials "$STAGE4_TRIALS" $EXTRA_ARGS
else
    echo "########## $DS: stage 3 ##########"
    run_py --dataset "$DS" --stage 3 --stage4-trials 0 $EXTRA_ARGS
    echo "########## $DS: stage 4 (${STAGE4_TRIALS} trials) ##########"
    run_py --dataset "$DS" --stage 4 --stage4-trials "$STAGE4_TRIALS" $EXTRA_ARGS
fi

echo ""
echo "Finetune finished: $(date)  dataset=$DS"
echo "=========================================="
