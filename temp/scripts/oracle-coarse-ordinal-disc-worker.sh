#!/usr/bin/env bash
set -euo pipefail

REPO="${ORDINAL_REPO:?}"
DATASET="${ORDINAL_DATASET:?}"
KIND="${ORDINAL_KIND:?}"
SEED="${ORDINAL_SEED:?}"
BINARY_INPUT="${ORDINAL_BINARY_INPUT:-}"
cd "$REPO"
module purge || true
module load StdEnv/2023 python/3.11 cuda/12.2 cudnn/8.9
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index -r "$REPO/setup/requirements-killarney.txt" -q
python -c "import torch; assert torch.cuda.is_available(), 'CUDA required'; print(torch.__version__, torch.cuda.get_device_name(0))"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
STEM="$(date +%m-%d)-${SLURM_JOB_ID: -3}-ordinal-${KIND}-${DATASET}"
OUTPUT="$REPO/results/datasets/$STEM"

if [[ "$KIND" == binary ]]; then
    if [[ -n "$BINARY_INPUT" ]]; then
        echo "[binary] reusing held-out binary pack: $BINARY_INPUT"
        python -u -m experiments.ordinal_patch_refinement_killtest.full_experiment \
            --dataset "$DATASET" --resolution 256 --seed "$SEED" \
            --disc-only-input "$BINARY_INPUT" --output "$OUTPUT"
    else
        echo "[binary] no reusable held-out pack found; retraining oracle-coarse refiner"
        python -u -m experiments.ordinal_patch_refinement_killtest.full_experiment \
            --dataset "$DATASET" --resolution 256 --seed "$SEED" --output "$OUTPUT"
    fi
else
    python -u -m experiments.ordinal_patch_refinement_killtest.run_mmpd_ordinal_upscale_tpe_ema \
        --config configs/mmpd_ordinal_upscale_lb96_hz16.yaml \
        --dataset "$DATASET" --seed "$SEED" --output "$OUTPUT"
fi
