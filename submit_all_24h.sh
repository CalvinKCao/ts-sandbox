#!/usr/bin/env bash
# Throwaway: submit pretrain + per-dataset finetune jobs for the old pipeline.
# All datasets w/ fewer variates than weather, plus solar_Alabama and PeMS.
# Walltime: 24h L40S on Killarney.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

ACCOUNT=aip-boyuwang
MAIL=ccao87@uwo.ca
TIME_MIN=$((24 * 60))
LOG_DIR="$ROOT/results/slurm_logs"
mkdir -p "$LOG_DIR"

DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 illness exchange_rate solar_Alabama PeMS)

# --------- shared body run inside a Slurm job -----------------------------
if [ "${1:-}" = "__inner_pretrain" ]; then
    set -x
    cd "$ROOT"
    [ -f .venv/bin/activate ] && source .venv/bin/activate || source venv/bin/activate
    exec ./train_universal_pretrain.sh --seed 42
elif [ "${1:-}" = "__inner_finetune" ]; then
    DS="$2"
    set -x
    cd "$ROOT"
    [ -f .venv/bin/activate ] && source .venv/bin/activate || source venv/bin/activate
    exec ./train_universal_pretrain.sh \
        --seed 42 \
        --skip-synthetic-search \
        --skip-universal-pretrain \
        --dataset "$DS"
fi

# --------- login-side: submit jobs ----------------------------------------
SBATCH_COMMON=(
    --account="$ACCOUNT"
    --nodes=1 --gres=gpu:l40s:1 --cpus-per-task=8 --mem=50G
    --time="$TIME_MIN"
    --mail-type=END,FAIL --mail-user="$MAIL"
    --output="$LOG_DIR/%x-%j.out" --error="$LOG_DIR/%x-%j.err"
    --chdir="$ROOT"
)

echo ">>> submitting stage0+1 pretrain (universal)"
PRETRAIN_JID=$(sbatch --parsable \
    --job-name=oldpipe-pretrain \
    "${SBATCH_COMMON[@]}" \
    "$0" __inner_pretrain)
echo "    pretrain job id: $PRETRAIN_JID"

for DS in "${DATASETS[@]}"; do
    DS_TAG="${DS//_/-}"
    echo ">>> submitting finetune-$DS_TAG (afterok:$PRETRAIN_JID)"
    sbatch --parsable \
        --job-name="oldpipe-ft-$DS_TAG" \
        --dependency="afterok:$PRETRAIN_JID" \
        "${SBATCH_COMMON[@]}" \
        "$0" __inner_finetune "$DS"
done

echo "done. monitor with: squeue -u \$USER --me"
