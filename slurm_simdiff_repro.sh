#!/bin/bash
# Slurm: SimDiff reproduction (ETTh1 + exchange_rate), L=H=96, comparable eval metrics.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "${SLURM_JOB_ID:-}" ]; then
    mkdir -p "$SCRIPT_DIR/results_simdiff/bootstrap"
    for ds in ETTh1 exchange; do
        JOB="simdiff_${ds}_96"
        [ "$ds" = "exchange" ] && JOB="simdiff_exchange_96"
        sbatch \
            --job-name="$JOB" \
            --account=aip-boyuwang \
            --time=24:00:00 \
            --nodes=1 \
            --gres=gpu:l40s:1 \
            --cpus-per-task=8 \
            --mem=50G \
            --chdir="$SCRIPT_DIR" \
            --output="results_simdiff/bootstrap/%x-%j.out" \
            --error="results_simdiff/bootstrap/%x-%j.err" \
            --export=ALL,DATASET=$ds \
            "$SCRIPT_DIR/slurm_simdiff_repro.sh"
    done
    echo "Submitted SimDiff repro jobs."
    exit 0
fi

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
exec >"results_simdiff/logs/${SLURM_JOB_ID}_${SLURM_JOB_NAME}.log" 2>&1

module purge || true
module load StdEnv/2023 python/3.11 cuda/12.2 cudnn/8.9

if [ -n "${SLURM_TMPDIR:-}" ]; then
    python3 -m venv "$SLURM_TMPDIR/env"
    source "$SLURM_TMPDIR/env/bin/activate"
    pip install --no-index --upgrade pip
    pip install --no-index torch numpy scipy pandas scikit-learn tqdm matplotlib einops beartype reformer-pytorch
else
    source .venv/bin/activate
fi

pip install -q einops beartype scikit-learn reformer-pytorch 2>/dev/null || true

if [ ! -d "$SLURM_SUBMIT_DIR/SimDiff" ]; then
    echo "ERROR: SimDiff/ not found. Clone Dear-Sloth/SimDiff into repo root on the cluster."
    exit 1
fi

case "${DATASET:-ETTh1}" in
    ETTh1)
        if [ -f scripts/simdiff/etth1_96_96.sh ]; then
            bash scripts/simdiff/etth1_96_96.sh
        else
            echo "Train SimDiff ETTh1 via SimDiff/script/etth1.sh (see baselines/simdiff/README.md)"
            exit 1
        fi
        python baselines/simdiff/eval_comparable.py --dataset ETTh1 --skip-alignment-check
        ;;
    exchange|exchange_rate)
        if [ -f scripts/simdiff/exchange_96_96.sh ]; then
            bash scripts/simdiff/exchange_96_96.sh
        else
            echo "Train SimDiff exchange via SimDiff/script/exchange.sh (see baselines/simdiff/README.md)"
            exit 1
        fi
        python baselines/simdiff/eval_comparable.py --dataset exchange_rate --skip-alignment-check
        ;;
    *)
        echo "Unknown DATASET=$DATASET"; exit 1 ;;
esac
