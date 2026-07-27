#!/usr/bin/env bash
# Quick Killarney probe: load the Narval vertical-dual checkpoints and the
# existing ordinal-normalized MMPD Decoder checkpoints, then train a tiny
# univariate discriminator on each dataset after shared ordinal snapping.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASETS="ETTh1,traffic,exchange_rate"
SEED=42
WALL="0:45:00"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --datasets) DATASETS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --time) WALL="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
    [[ -d "$REPO/.git" ]] || {
        echo "ERROR: expected this launcher under <repo>/temp" >&2
        exit 2
    }
    [[ "$REPO" != /home/* ]] || {
        echo "ERROR: submit from the checkout under /scratch, not /home: $REPO" >&2
        exit 2
    }
    MMPD_ROOT="$REPO/results/datasets/07-08-mmpd-decoder-ordinal-norm-lb336-hz720"
    [[ -d "$MMPD_ROOT/mmpd_out/checkpoints" ]] || {
        echo "ERROR: missing MMPD campaign checkpoints: $MMPD_ROOT" >&2
        exit 2
    }
    [[ -d "$REPO/temp/MMPD" ]] || {
        echo "ERROR: missing $REPO/temp/MMPD; clone the pinned MMPD checkout before running." >&2
        exit 2
    }

    RUN_STEM="$(date +%m-%d)-univariate-ladder-smoke"
    IFS=',' read -r -a DATASET_LIST <<< "$DATASETS"
    for DATASET in "${DATASET_LIST[@]}"; do
        case "$DATASET" in
            ETTh1|traffic|exchange_rate) ;;
            *) echo "ERROR: unsupported dataset: $DATASET" >&2; exit 2 ;;
        esac
        JOB="uni-ladder-${DATASET}"
        sbatch \
            --job-name="$JOB" \
            --account=aip-boyuwang \
            --nodes=1 --gres=gpu:l40s:1 --cpus-per-task=8 --mem=50G --time="$WALL" \
            --output="$REPO/results/logs/${JOB}-%j.log" \
            --error="$REPO/results/logs/${JOB}-%j.log" \
            --mail-type=FAIL --mail-user=ccao87@uwo.ca \
            --export=ALL,DISC_REPO="$REPO",DISC_DATASET="$DATASET",DISC_SEED="$SEED",DISC_MMPD_ROOT="$MMPD_ROOT",DISC_RUN_STEM="$RUN_STEM" \
            "$REPO/temp/run_univariate_disc_ladder_smoke_killarney.sh"
    done
    exit 0
fi

REPO="${DISC_REPO:?}"
DATASET="${DISC_DATASET:?}"
MMPD_ROOT="${DISC_MMPD_ROOT:?}"
RUN_STEM="${DISC_RUN_STEM:?}"
cd "$REPO"

module purge || true
module load StdEnv/2023 python/3.11 cuda/12.2 cudnn/8.9
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index -r "$REPO/setup/requirements-killarney.txt" -q
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"

DISC_DIR="$REPO/results/datasets/$RUN_STEM"
RAW_DIR="$REPO/results/datasets/${RUN_STEM}-raw"
COMMON_ARGS=(
    --datasets "$DATASET"
    --fake-sources binary_staged mmpd
    --anchor-config bce_dist_guidance_cond_3x336_overlap_value_width_fixed_hp
    --binary-config configs/bce_dist_guidance_cond_3x336_overlap_value_width_fixed_hp.yaml
    --mmpd-output-root "$MMPD_ROOT"
    --mmpd-backbone Decoder
    --mmpd-repo "$REPO/temp/MMPD"
    --mmpd-data-dir "$REPO/temp/mmpd_datasets"
    --lookback 336 --horizon 720 --test-stride 4
    --pack-splits test
    --test-max-items 8 --max-windows 8
    --raw-eval-dir "$RAW_DIR"
    --bin-match-filter all
    --ordinal-ladder-quantize
    --no-update-mmpd
    --gpu 0
)

python -u "$REPO/temp/inspect_univariate_disc_ladder.py" \
    "${COMMON_ARGS[@]}" \
    --output-dir "$DISC_DIR" \
    --force-raw-eval

python -u "$REPO/utils/eval_discriminator_binary_vs_mmpd_univariate.py" \
    "${COMMON_ARGS[@]}" \
    --output-dir "$DISC_DIR" \
    --slice-lengths 16 \
    --candidate-only --nonoverlapping-patches --no-offset-embedding \
    --epochs 2 --patience 2 --batch-size 32 \
    --max-train-examples 128 --max-eval-examples 64 \
    --max-batches-per-epoch 2 \
    --force-train --save-checkpoints
