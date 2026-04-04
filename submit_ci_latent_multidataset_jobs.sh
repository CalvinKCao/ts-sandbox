#!/usr/bin/env bash
# Submit from a login node (not inside a batch job): one bootstrap job + five parallel
# finetune jobs that wait for bootstrap. Shorter wall limits per job than the all-in-one script.
#
# Usage:
#   ./submit_ci_latent_multidataset_jobs.sh
#   ./submit_ci_latent_multidataset_jobs.sh -- --smoke-test
# Extra sbatch options for every job:
#   SBATCH_EXTRA="--time=24:00:00" ./submit_ci_latent_multidataset_jobs.sh
#
# Override account (must match your RAP):
#   SBATCH_ACCOUNT=def-yourgroup ./submit_ci_latent_multidataset_jobs.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

FORWARD=()
if [[ "${1:-}" == "--" ]]; then
    shift
    FORWARD=("$@")
fi

SB_EXTRA=()
if [[ -n "${SBATCH_EXTRA:-}" ]]; then
    # shellcheck disable=SC2206
    SB_EXTRA=( $SBATCH_EXTRA )
fi
if [[ -n "${SBATCH_ACCOUNT:-}" ]]; then
    SB_EXTRA+=(--account="$SBATCH_ACCOUNT")
fi

submit() {
    sbatch --parsable "${SB_EXTRA[@]}" "$@"
}

echo "Submitting bootstrap (stages 0-2)..."
BOOT_ID="$(submit "$ROOT/slurm_ci_latent_bootstrap.sh" -- "${FORWARD[@]}")"
echo "  bootstrap job id: $BOOT_ID"

echo "Submitting finetune jobs (afterok:$BOOT_ID)..."
for DS in ETTh1 ETTh2 ETTm1 ETTm2; do
    JID="$(submit \
        --dependency=afterok:"$BOOT_ID" \
        --job-name="ci-ft-${DS}" \
        --export=ALL,CI_DATASET="${DS}" \
        "$ROOT/slurm_ci_latent_finetune_dataset.sh" -- "${FORWARD[@]}")"
    echo "  $DS -> $JID"
done

JID="$(submit \
    --dependency=afterok:"$BOOT_ID" \
    --job-name=ci-ft-exrate \
    --export=ALL,CI_DATASET=exchange_rate,CI_EXCHANGE_SEED=42 \
    "$ROOT/slurm_ci_latent_finetune_dataset.sh" -- "${FORWARD[@]}")"
echo "  exchange_rate -> $JID"

echo ""
echo "Done. squeue -u \"\$USER\" | grep ci-"
echo "If bootstrap exits early (ckpts already exist), finetune jobs still run once bootstrap is OK."
