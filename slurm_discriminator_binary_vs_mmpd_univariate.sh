#!/bin/bash
# Univariate real-vs-fake discriminator shards (binary vs GT, MMPD vs GT).
#
# Same fair protocol as the multivariate texture disc, but each example is a
# single-variate L-patch. Shards are dataset × fake_source (like the original).
#
# Login-node usage:
#   ./slurm_discriminator_binary_vs_mmpd_univariate.sh --datasets ETTh1,traffic,... ...
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Login node: parse CLI, submit shards. Do NOT clobber --export on compute.
# ---------------------------------------------------------------------------
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    DATASETS=(ETTh1 traffic electricity exchange_rate)
    FAKE_SOURCES=(binary_staged mmpd)
    DATASET=""
    FAKE_SOURCE=""
    SMOKE=0
    FORCE_RAW=0
    FORCE_TRAIN=0
    WALL_OVERRIDE=""
    SLICE_LENGTHS="8,16,32"
    PACK_SPLITS=""
    PACK_FRACTION=""
    ANCHOR_CONFIG=""
    ANCHOR_CONFIG_BY_DATASET=""
    BINARY_CONFIG=""
    BINARY_CONFIG_BY_DATASET=""
    MMPD_OUTPUT_SUFFIX="results/datasets/07-10-mmpd-decoder-paper-lb336-hz720-subset"
    MMPD_ROOT_OVERRIDE=""
    DISC_OUTPUT_SUFFIX="results/datasets/disc-lb336-hz720-ordinal-four-patch-only-fair-univariate-bin16"
    RAW_OUTPUT_SUFFIX="results/datasets/disc-lb336-hz720-ordinal-four-raw-trainval25"
    LOOKBACK=336
    HORIZON=720
    TEST_STRIDE=1
    MMPD_BACKBONE=Decoder
    ALLOW_REUSED_MMPD_ROOT=0
    MERGE_PARTIALS=0
    # Snap GT+binary+MMPD through binary's dual-scale 16x16 ordinal lattice.
    BIN_MATCH_FILTER="all"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --datasets)
                IFS=',' read -r -a DATASETS <<< "$2"
                shift 2
                ;;
            --fake-sources)
                IFS=',' read -r -a FAKE_SOURCES <<< "$2"
                shift 2
                ;;
            --dataset) DATASET="$2"; shift 2 ;;
            --fake-source) FAKE_SOURCE="$2"; shift 2 ;;
            --smoke-test) SMOKE=1; shift ;;
            --force-raw-eval) FORCE_RAW=1; shift ;;
            --force-train) FORCE_TRAIN=1; shift ;;
            --time) WALL_OVERRIDE="$2"; shift 2 ;;
            --slice-lengths) SLICE_LENGTHS="${2// /,}"; shift 2 ;;
            --pack-splits) PACK_SPLITS="$2"; shift 2 ;;
            --pack-fraction) PACK_FRACTION="$2"; shift 2 ;;
            --anchor-config) ANCHOR_CONFIG="$2"; shift 2 ;;
            --anchor-config-by-dataset) ANCHOR_CONFIG_BY_DATASET="$2"; shift 2 ;;
            --binary-config) BINARY_CONFIG="$2"; shift 2 ;;
            --binary-config-by-dataset) BINARY_CONFIG_BY_DATASET="$2"; shift 2 ;;
            --mmpd-run) MMPD_OUTPUT_SUFFIX="results/datasets/$2"; shift 2 ;;
            --mmpd-root) MMPD_ROOT_OVERRIDE="$2"; shift 2 ;;
            --disc-run) DISC_OUTPUT_SUFFIX="results/datasets/$2"; shift 2 ;;
            --raw-run) RAW_OUTPUT_SUFFIX="results/datasets/$2"; shift 2 ;;
            --lookback) LOOKBACK="$2"; shift 2 ;;
            --horizon) HORIZON="$2"; shift 2 ;;
            --test-stride) TEST_STRIDE="$2"; shift 2 ;;
            --mmpd-backbone) MMPD_BACKBONE="$2"; shift 2 ;;
            --allow-reused-mmpd-root) ALLOW_REUSED_MMPD_ROOT=1; shift ;;
            --bin-match-filter) BIN_MATCH_FILTER="$2"; shift 2 ;;
            --merge-partials-only) MERGE_PARTIALS=1; shift ;;
            *)
                echo "Unknown arg: $1" >&2
                exit 1
                ;;
        esac
    done

    if [[ -d "${SCRATCH:-}/ts-sandbox" ]]; then
        REPO="${SCRATCH}/ts-sandbox"
    elif [[ -d "$HOME/ts-sandbox" ]]; then
        REPO="$HOME/ts-sandbox"
    else
        REPO="$SCRIPT_DIR"
    fi
    if [[ "$REPO" == /home/* ]]; then
        echo "ERROR: submit from \$SCRATCH/ts-sandbox on Killarney, not /home." >&2
        exit 1
    fi
    cd "$REPO"
    mkdir -p results/logs

    slurm_encode() { printf '%s' "${1//,/;}" ; }

    if [[ "$MERGE_PARTIALS" -eq 1 ]]; then
        sbatch \
            --job-name=disc-uni-merge \
            --account=aip-boyuwang \
            --nodes=1 --cpus-per-task=2 --mem=4G --time=0:15:00 \
            --output=results/logs/disc-uni-merge-%j.log \
            --error=results/logs/disc-uni-merge-%j.log \
            --mail-type=FAIL --mail-user=ccao87@uwo.ca \
            --export=ALL,MERGE_PARTIALS=1,DISC_OUTPUT_SUFFIX="$DISC_OUTPUT_SUFFIX",RAW_OUTPUT_SUFFIX="$RAW_OUTPUT_SUFFIX" \
            "$SCRIPT_DIR/slurm_discriminator_binary_vs_mmpd_univariate.sh"
        exit 0
    fi

    # Univariate expands examples by n_variates (~4–8×); give headroom.
    WALL="2:00:00"
    [[ "$SMOKE" -eq 1 ]] && WALL="0:30:00"
    [[ -n "$WALL_OVERRIDE" ]] && WALL="$WALL_OVERRIDE"

    SUBMIT_DATASETS=("${DATASETS[@]}")
    [[ -n "$DATASET" ]] && SUBMIT_DATASETS=("$DATASET")
    [[ "$SMOKE" -eq 1 && -z "$DATASET" ]] && SUBMIT_DATASETS=(ETTh1)

    SUBMIT_SOURCES=("${FAKE_SOURCES[@]}")
    [[ -n "$FAKE_SOURCE" ]] && SUBMIT_SOURCES=("$FAKE_SOURCE")
    [[ "$SMOKE" -eq 1 && -z "$FAKE_SOURCE" ]] && SUBMIT_SOURCES=(binary_staged)

    JOB_IDS=()
    for ds in "${SUBMIT_DATASETS[@]}"; do
        for src in "${SUBMIT_SOURCES[@]}"; do
            echo "Submitting univariate real-vs-fake disc for $ds / $src (L40S, wall=$WALL)..."
            job_id="$(sbatch --parsable \
                --job-name="disc-uni-${ds}-${src}" \
                --account=aip-boyuwang \
                --nodes=1 \
                --gres=gpu:l40s:1 \
                --cpus-per-task=8 \
                --mem=50G \
                --time="$WALL" \
                --output="results/logs/disc-uni-${ds}-${src}-%j.log" \
                --error="results/logs/disc-uni-${ds}-${src}-%j.log" \
                --mail-type=END,FAIL \
                --mail-user=ccao87@uwo.ca \
                --export=ALL,DATASET="$ds",FAKE_SOURCE="$src",SMOKE="$SMOKE",FORCE_RAW="$FORCE_RAW",FORCE_TRAIN="$FORCE_TRAIN",SLICE_LENGTHS="$(slurm_encode "$SLICE_LENGTHS")",PACK_SPLITS="$(slurm_encode "$PACK_SPLITS")",PACK_FRACTION="$PACK_FRACTION",ANCHOR_CONFIG="$ANCHOR_CONFIG",ANCHOR_CONFIG_BY_DATASET="$(slurm_encode "$ANCHOR_CONFIG_BY_DATASET")",BINARY_CONFIG="$BINARY_CONFIG",BINARY_CONFIG_BY_DATASET="$(slurm_encode "$BINARY_CONFIG_BY_DATASET")",MMPD_OUTPUT_SUFFIX="$MMPD_OUTPUT_SUFFIX",MMPD_ROOT_OVERRIDE="$MMPD_ROOT_OVERRIDE",DISC_OUTPUT_SUFFIX="$DISC_OUTPUT_SUFFIX",RAW_OUTPUT_SUFFIX="$RAW_OUTPUT_SUFFIX",LOOKBACK="$LOOKBACK",HORIZON="$HORIZON",TEST_STRIDE="$TEST_STRIDE",MMPD_BACKBONE="$MMPD_BACKBONE",BIN_MATCH_FILTER="$BIN_MATCH_FILTER",ALLOW_REUSED_MMPD_ROOT="$ALLOW_REUSED_MMPD_ROOT",MERGE_PARTIALS=0 \
                "$SCRIPT_DIR/slurm_discriminator_binary_vs_mmpd_univariate.sh")"
            JOB_IDS+=("$job_id")
            echo "  -> job $job_id"
        done
    done

    dep=$(IFS=:; echo "${JOB_IDS[*]}")
    echo "Submitting merge after ${dep}..."
    sbatch \
        --job-name=disc-uni-merge \
        --account=aip-boyuwang \
        --nodes=1 --cpus-per-task=2 --mem=4G --time=0:15:00 \
        --dependency=afterok:${dep} \
        --output=results/logs/disc-uni-merge-%j.log \
        --error=results/logs/disc-uni-merge-%j.log \
        --mail-type=FAIL --mail-user=ccao87@uwo.ca \
        --export=ALL,MERGE_PARTIALS=1,DISC_OUTPUT_SUFFIX="$DISC_OUTPUT_SUFFIX",RAW_OUTPUT_SUFFIX="$RAW_OUTPUT_SUFFIX" \
        "$SCRIPT_DIR/slurm_discriminator_binary_vs_mmpd_univariate.sh"
    exit 0
fi

# ---------------------------------------------------------------------------
# Inside the job
# ---------------------------------------------------------------------------
if [[ -d "${SCRATCH:-}/ts-sandbox" ]]; then
    REPO="${SCRATCH}/ts-sandbox"
elif [[ -d "$HOME/ts-sandbox" ]]; then
    REPO="$HOME/ts-sandbox"
else
    REPO="$SCRIPT_DIR"
fi
cd "$REPO"

slurm_decode() { printf '%s' "${1//;/,}" ; }
ANCHOR_CONFIG_BY_DATASET="$(slurm_decode "${ANCHOR_CONFIG_BY_DATASET:-}")"
BINARY_CONFIG_BY_DATASET="$(slurm_decode "${BINARY_CONFIG_BY_DATASET:-}")"
PACK_SPLITS="$(slurm_decode "${PACK_SPLITS:-}")"
SLICE_LENGTHS="$(slurm_decode "${SLICE_LENGTHS:-8;16;32}")"

DISC_OUTPUT_SUFFIX="${DISC_OUTPUT_SUFFIX:-results/datasets/disc-lb336-hz720-ordinal-four-patch-only-fair-univariate-bin16}"
RAW_OUTPUT_SUFFIX="${RAW_OUTPUT_SUFFIX:-results/datasets/disc-lb336-hz720-ordinal-four-raw-trainval25}"
MMPD_OUTPUT_SUFFIX="${MMPD_OUTPUT_SUFFIX:-results/datasets/07-10-mmpd-decoder-paper-lb336-hz720-subset}"
BIN_MATCH_FILTER="${BIN_MATCH_FILTER:-all}"
OUTPUT_DIR="$REPO/$DISC_OUTPUT_SUFFIX"
RAW_EVAL_DIR="$REPO/$RAW_OUTPUT_SUFFIX"
MMPD_ROOT="${MMPD_ROOT_OVERRIDE:-$REPO/$MMPD_OUTPUT_SUFFIX}"
SLICE_LENGTHS="${SLICE_LENGTHS// /;}"
SLICE_LENGTHS="${SLICE_LENGTHS//,/;}"
IFS=';' read -r -a SLICE_ARR <<< "$SLICE_LENGTHS"

REQ="$REPO/setup/requirements-killarney.txt"
[[ -f "$REQ" ]] || { echo "ERROR: missing $REQ"; exit 1; }
[[ -n "${SLURM_TMPDIR:-}" ]] || { echo "ERROR: SLURM_TMPDIR unset"; exit 1; }

module purge 2>/dev/null || true
module load StdEnv/2023 python/3.11 2>/dev/null || true
if [[ "${MERGE_PARTIALS:-0}" -ne 1 ]]; then
    module load cuda/12.2 cudnn/8.9 2>/dev/null || true
fi
virtualenv --no-download "$SLURM_TMPDIR/env"
# shellcheck source=/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
export PYTHON="$SLURM_TMPDIR/env/bin/python"
pip install --no-index --upgrade pip -q
pip install --no-index -r "$REQ" -q

export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

if [[ "${MERGE_PARTIALS:-0}" -eq 1 ]]; then
    "$PYTHON" -u "$REPO/utils/eval_discriminator_binary_vs_mmpd_univariate.py" \
        --merge-partials-only \
        --output-dir "$OUTPUT_DIR" \
        --raw-eval-dir "$RAW_EVAL_DIR"
    exit 0
fi

if [[ -z "${DATASET:-}" || -z "${FAKE_SOURCE:-}" ]]; then
    echo "ERROR: DATASET/FAKE_SOURCE env empty inside GPU job." >&2
    exit 1
fi
if [[ -z "${ANCHOR_CONFIG:-}" && -z "${ANCHOR_CONFIG_BY_DATASET:-}" ]]; then
    echo "ERROR: missing ANCHOR_CONFIG / ANCHOR_CONFIG_BY_DATASET in job env." >&2
    exit 1
fi

if [[ "${ALLOW_REUSED_MMPD_ROOT:-0}" -eq 1 ]]; then
    "$PYTHON" -u "$REPO/temp/scripts/check_mmpd_reused_decoder_root.py" "$MMPD_ROOT" --datasets "$DATASET"
else
    "$PYTHON" -u "$REPO/temp/scripts/check_mmpd_instance_campaign.py" "$MMPD_ROOT"
fi
"$PYTHON" -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"

EVAL_ARGS=(
    --datasets "$DATASET"
    --fake-sources "$FAKE_SOURCE"
    --slice-lengths "${SLICE_ARR[@]}"
    --output-dir "$OUTPUT_DIR"
    --raw-eval-dir "$RAW_EVAL_DIR"
    --mmpd-output-root "$MMPD_ROOT"
    --mmpd-backbone "${MMPD_BACKBONE:-Decoder}"
    --no-update-mmpd
    --lookback "${LOOKBACK:-336}"
    --horizon "${HORIZON:-720}"
    --test-stride "${TEST_STRIDE:-1}"
    --candidate-only
    --nonoverlapping-patches
    --no-offset-embedding
    --no-ordinal-ladder-quantize
    --bin-match-filter "$BIN_MATCH_FILTER"
    --bin-image-height 16
    --bin-coarse-height 16
    --bin-fine-height 16
    --save-checkpoints
    --no-merge-metrics
)

[[ -n "${ANCHOR_CONFIG:-}" ]] && EVAL_ARGS+=(--anchor-config "$ANCHOR_CONFIG")
[[ -n "${BINARY_CONFIG:-}" ]] && EVAL_ARGS+=(--binary-config "$BINARY_CONFIG")
[[ -n "${ANCHOR_CONFIG_BY_DATASET:-}" ]] && EVAL_ARGS+=(--anchor-config-by-dataset "$ANCHOR_CONFIG_BY_DATASET")
[[ -n "${BINARY_CONFIG_BY_DATASET:-}" ]] && EVAL_ARGS+=(--binary-config-by-dataset "$BINARY_CONFIG_BY_DATASET")
[[ -n "${PACK_SPLITS:-}" ]] && EVAL_ARGS+=(--pack-splits "$PACK_SPLITS")
[[ -n "${PACK_FRACTION:-}" ]] && EVAL_ARGS+=(--pack-fraction "$PACK_FRACTION")
[[ "${FORCE_RAW:-0}" -eq 1 ]] && EVAL_ARGS+=(--force-raw-eval)
[[ "${FORCE_TRAIN:-0}" -eq 1 ]] && EVAL_ARGS+=(--force-train)
[[ "${SMOKE:-0}" -eq 1 ]] && EVAL_ARGS+=(--smoke-test)
EVAL_ARGS+=(--no-mmpd-ordinal-norm --mmpd-instance-norm --mmpd-to-binary-dataset-norm)
if [[ -n "${DISC_NATIVE_REPR_STRIDE:-}" ]]; then
    EVAL_ARGS+=(--native-repr-stride "$DISC_NATIVE_REPR_STRIDE")
fi

echo "[env] DATASET=$DATASET FAKE_SOURCE=$FAKE_SOURCE SMOKE=${SMOKE:-0} FORCE_TRAIN=${FORCE_TRAIN:-0}"
echo "[env] BIN_MATCH_FILTER=$BIN_MATCH_FILTER (binary 16x16 ordinal lattice)"
echo "[env] ANCHOR_CONFIG_BY_DATASET=${ANCHOR_CONFIG_BY_DATASET:-}"
echo "[env] PACK_SPLITS=${PACK_SPLITS:-} PACK_FRACTION=${PACK_FRACTION:-}"
echo "[env] SLICE_LENGTHS=${SLICE_ARR[*]}"
echo "[run] ${EVAL_ARGS[*]}"
"$PYTHON" -u "$REPO/utils/eval_discriminator_binary_vs_mmpd_univariate.py" "${EVAL_ARGS[@]}"
echo "Done: $(date)"
