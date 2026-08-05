#!/usr/bin/env bash
# One-off Narval launcher for the fair oracle-coarse 16 -> 256 discriminator test.
#
# Binary source: the original 32x8 binary XOR refiner.  It receives the
# ground-truth 16-bin future, vertically upsamples it to 256 bins, and refines
# that canvas.  It is not a 336 -> 96 forecasting checkpoint.
#
# MMPD source: the matching 1D task, conditioned on the same oracle 16-bin
# future plus the final 8 full-resolution past points, predicting 256-bin
# ranks over the same 16-point future.
set -euo pipefail

DATASETS="ETTh1,exchange_rate,electricity,traffic"
SEED=42
BINARY_TIME="8:00:00"
BINARY_EVAL_TIME="1:00:00"
MMPD_TIME="1:30:00"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --datasets) DATASETS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --binary-time) BINARY_TIME="$2"; shift 2 ;;
        --binary-eval-time) BINARY_EVAL_TIME="$2"; shift 2 ;;
        --mmpd-time) MMPD_TIME="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

[[ "$(hostname)" == *narval* ]] || {
    echo "ERROR: run this from a Narval login node." >&2
    exit 2
}
REPO="${SCRATCH:-}/ts-sandbox"
[[ -d "$REPO" ]] || { echo "ERROR: expected checkout at $REPO" >&2; exit 2; }
[[ -f "$REPO/configs/mmpd_ordinal_upscale_lb96_hz16.yaml" ]] || {
    echo "ERROR: missing ordinal-upscale config in $REPO" >&2
    exit 2
}
mkdir -p "$REPO/results/logs" "$REPO/results/datasets" "$REPO/temp"

MMPD_REPO="$REPO/temp/MMPD"
if [[ ! -d "$MMPD_REPO/.git" ]]; then
    git clone https://github.com/Thinklab-SJTU/MMPD.git "$MMPD_REPO"
fi
if [[ -f "$MMPD_REPO/utils/tools.py" ]] && grep -q 'np\.Inf' "$MMPD_REPO/utils/tools.py"; then
    sed -i 's/np\.Inf/np.inf/g' "$MMPD_REPO/utils/tools.py"
fi

WORKER="$REPO/temp/scripts/oracle-coarse-ordinal-disc-worker.sh"
cat > "$WORKER" <<'WORKER'
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
WORKER
chmod +x "$WORKER"

IFS=',' read -r -a DATASET_LIST <<< "$DATASETS"
for DATASET in "${DATASET_LIST[@]}"; do
    case "$DATASET" in
        ETTh1|exchange_rate|electricity|traffic) ;;
        *) echo "ERROR: unsupported dataset: $DATASET" >&2; exit 2 ;;
    esac
    case "$DATASET" in
        ETTh1) OLD_BINARY_RUN="full-ETTh1-256-66077307" ;;
        exchange_rate) OLD_BINARY_RUN="full-exchange_rate-256-66077308" ;;
        electricity) OLD_BINARY_RUN="full-electricity-256-66077309" ;;
        traffic) OLD_BINARY_RUN="full-traffic-256-66077310" ;;
    esac
    OLD_BINARY_INPUT="$REPO/results/ordinal_patch_refinement_killtest/$OLD_BINARY_RUN/heldout_windows.npz"
    if [[ -f "$OLD_BINARY_INPUT" ]]; then
        echo "binary $DATASET: reuse $OLD_BINARY_INPUT"
        BINARY_INPUT="$OLD_BINARY_INPUT"
        BINARY_WALL="$BINARY_EVAL_TIME"
    else
        echo "binary $DATASET: no persisted held-out pack at $OLD_BINARY_INPUT; will retrain"
        BINARY_INPUT=""
        BINARY_WALL="$BINARY_TIME"
    fi
    for KIND in binary mmpd; do
        if [[ "$KIND" == binary ]]; then
            WALL="$BINARY_WALL"
        else
            WALL="$MMPD_TIME"
        fi
        JOB="ord-${KIND}-${DATASET}"
        JOB_ID=$(sbatch --parsable \
            --job-name="$JOB" \
            --account=def-boyuwang \
            --nodes=1 --gpus=a100:1 --cpus-per-task=8 --mem=80G --time="$WALL" \
            --output="$REPO/results/logs/${JOB}-%j.log" \
            --error="$REPO/results/logs/${JOB}-%j.log" \
            --mail-type=FAIL \
            --export=ALL,ORDINAL_REPO="$REPO",ORDINAL_DATASET="$DATASET",ORDINAL_KIND="$KIND",ORDINAL_SEED="$SEED",ORDINAL_BINARY_INPUT="$BINARY_INPUT" \
            "$WORKER")
        echo "$KIND $DATASET: $JOB_ID"
    done
done
