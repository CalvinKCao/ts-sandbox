#!/usr/bin/env bash
# Throwaway: submit pretrain + per-dataset finetune jobs for the old pipeline.
# All datasets w/ fewer variates than weather, plus solar_Alabama and PeMS.
# Walltime: 24h L40S on Killarney.

ACCOUNT=aip-boyuwang
MAIL=ccao87@uwo.ca
TIME_LIMIT_PRETRAIN="3-00:00:00"
TIME_LIMIT_FINETUNE="1-00:00:00"

DATASETS=(ETTh1 ETTh2 ETTm1 exchange_rate)

# --------- shared body run inside a Slurm job -----------------------------
# SLURM_SUBMIT_DIR is set by Slurm to the directory sbatch was run from.
# BASH_SOURCE[0] points at the spool copy under /cm/local/.../spool/, not
# the repo — do not use it to anchor paths inside the job body.
if [ "${1:-}" = "__inner_pretrain" ] || [ "${1:-}" = "__inner_finetune" ]; then
    set -euo pipefail
    ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set}"
    cd "$ROOT"
    LOG_DIR="$ROOT/results/slurm_logs"
    mkdir -p "$LOG_DIR"

    module purge || true
    module load StdEnv/2023 python/3.11 cuda/12.2 cudnn/8.9 || true

    # Rebuild venv on fast local NVMe — avoids Lustre import latency
    virtualenv --no-download "$SLURM_TMPDIR/env"
    source "$SLURM_TMPDIR/env/bin/activate"
    pip install --no-index --upgrade pip
    pip install --no-index torch numpy pandas scikit-learn optuna
    pip install --no-index wandb
    pip install reformer-pytorch matplotlib

    # Install repo packages if setup.py / pyproject exists
    [ -f setup.py ] || [ -f pyproject.toml ] && pip install --no-index -e . --no-build-isolation || true

    if [ "${1:-}" = "__inner_pretrain" ]; then
        set -x
        # Check if universal model already exists to potentially skip
        UNIVERSAL_MODEL="$ROOT/models/diffusion_tsf/checkpoints/universal_synthetic_pretrain/best_model.pt"
        SKIP_FLAGS=""
        if [ -f "$UNIVERSAL_MODEL" ]; then
            echo ">>> Universal model found at $UNIVERSAL_MODEL. Will resume/skip stages as needed."
            # The script itself handles resume if --resume is passed inside it (which it is)
        fi
        exec ./train_universal_pretrain.sh --seed 42 --patience 10
    else
        DS="$2"
        set -x
        exec ./train_universal_pretrain.sh \
            --seed 42 \
            --skip-synthetic-search \
            --skip-universal-pretrain \
            --dataset "$DS" \
            --patience 25
    fi
fi

# --------- login-side: submit jobs ----------------------------------------
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT/results/slurm_logs"
mkdir -p "$LOG_DIR"

SBATCH_COMMON=(
    "--account=$ACCOUNT"
    --nodes=1 --gres=gpu:h100:1 --cpus-per-task=8 --mem=50G
    --mail-type=END,FAIL "--mail-user=$MAIL"
    "--output=$LOG_DIR/%x-%j.out" "--error=$LOG_DIR/%x-%j.err"
    "--chdir=$ROOT"
)

echo ">>> submitting stage0+1 pretrain (universal)"
PRETRAIN_JID=$(sbatch --parsable \
    --job-name=oldpipe-pretrain \
    "--time=$TIME_LIMIT_PRETRAIN" \
    "${SBATCH_COMMON[@]}" \
    "$0" __inner_pretrain)
echo "    pretrain job id: $PRETRAIN_JID"

for DS in "${DATASETS[@]}"; do
    DS_TAG="${DS//_/-}"
    echo ">>> submitting finetune-$DS_TAG (afterok:$PRETRAIN_JID)"
    sbatch --parsable \
        --job-name="oldpipe-ft-$DS_TAG" \
        --dependency="afterok:$PRETRAIN_JID" \
        "--time=$TIME_LIMIT_FINETUNE" \
        "${SBATCH_COMMON[@]}" \
        "$0" __inner_finetune "$DS"
done

echo "done. monitor with: squeue -u \$USER --me"
