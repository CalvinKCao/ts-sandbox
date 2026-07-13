#!/bin/bash
# =============================================================================
# Learned discriminator texture eval: staged binary probabilistic preds vs MMPD.
#
# One independent L40S job per (dataset, fake_source); CPU merge rebuilds
# metrics.json + report when shards finish.
#
# USAGE (login node, from $SCRATCH/ts-sandbox):
#   ./slurm_discriminator_texture_staged_vs_mmpd.sh --smoke-test
#   ./temp/submit_discriminator_texture_lb336_hz720_ordinal_four_killarney.sh
#   ./slurm_discriminator_texture_staged_vs_mmpd.sh --dataset ETTh1 --fake-source mmpd
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SMOKE=0
FORCE_RAW=0
FORCE_TRAIN=0
MERGE_PARTIALS=0
DATASET=""
FAKE_SOURCE=""
SLICE_LENGTH=""
BIN_MATCH_FILTER=""

ANCHOR_CONFIG="binary_anchor_stationary_flat_subsets_ema099"
ANCHOR_CONFIG_BY_DATASET=""
BINARY_CONFIG=""
BINARY_CONFIG_BY_DATASET=""
CKPT_BASE_SUFFIX="results/ckpts"
MMPD_OUTPUT_SUFFIX="results/datasets/06-13-binary-mmpd-subset-compare"
DISC_OUTPUT_SUFFIX="results/datasets/06-14-disc-texture-flat-subsets-ema099-vs-mmpd"
RAW_OUTPUT_SUFFIX="results/datasets/06-14-raw-texture-flat-subsets-ema099-vs-mmpd"
REPORT_SUFFIX="reports/06-14_discriminator_texture_flat_subsets_ema099_vs_mmpd.md"
IMPORT_MMPD_PACKS_FROM=""
DATASETS_CSV=""
TEST_STRIDE=""
TEST_FRACTION=""
LOOKBACK=""
HORIZON=""
MMPD_BACKBONE="Decoder"
WALL_OVERRIDE=""

resolve_output_dir() {
    local repo_root="$1"
    local base="${repo_root}/${DISC_OUTPUT_SUFFIX}"
    if [[ -n "$BIN_MATCH_FILTER" ]]; then
        echo "${base}-binmatch-${BIN_MATCH_FILTER}"
    else
        echo "$base"
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-test|--smoke) SMOKE=1; shift ;;
        --force-raw-eval) FORCE_RAW=1; shift ;;
        --force-train) FORCE_TRAIN=1; shift ;;
        --merge-partials-only) MERGE_PARTIALS=1; shift ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --fake-source) FAKE_SOURCE="$2"; shift 2 ;;
        --slice-length) SLICE_LENGTH="$2"; shift 2 ;;
        --bin-match-filter) BIN_MATCH_FILTER="$2"; shift 2 ;;
        --anchor-config) ANCHOR_CONFIG="$2"; shift 2 ;;
        --anchor-config-by-dataset) ANCHOR_CONFIG_BY_DATASET="$2"; shift 2 ;;
        --binary-config) BINARY_CONFIG="$2"; shift 2 ;;
        --binary-config-by-dataset) BINARY_CONFIG_BY_DATASET="$2"; shift 2 ;;
        --mmpd-run) MMPD_OUTPUT_SUFFIX="results/datasets/$2"; shift 2 ;;
        --disc-run) DISC_OUTPUT_SUFFIX="results/datasets/$2"; shift 2 ;;
        --raw-run) RAW_OUTPUT_SUFFIX="results/datasets/$2"; shift 2 ;;
        --report) REPORT_SUFFIX="$2"; shift 2 ;;
        --import-mmpd-packs-from) IMPORT_MMPD_PACKS_FROM="$2"; shift 2 ;;
        --datasets) DATASETS_CSV="$2"; shift 2 ;;
        --test-stride) TEST_STRIDE="$2"; shift 2 ;;
        --test-fraction) TEST_FRACTION="$2"; shift 2 ;;
        --lookback) LOOKBACK="$2"; shift 2 ;;
        --horizon) HORIZON="$2"; shift 2 ;;
        --mmpd-backbone) MMPD_BACKBONE="$2"; shift 2 ;;
        --time) WALL_OVERRIDE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ -n "$BIN_MATCH_FILTER" && "$BIN_MATCH_FILTER" != "mmpd" && "$BIN_MATCH_FILTER" != "both" && "$BIN_MATCH_FILTER" != "all" ]]; then
    echo "ERROR: --bin-match-filter must be one of: mmpd, both, all" >&2
    exit 1
fi

DATASETS=(ETTh1 ETTh2 exchange_rate weather electricity traffic solar_Alabama)
if [[ -n "$DATASETS_CSV" ]]; then
    IFS=',' read -ra DATASETS <<< "$DATASETS_CSV"
fi
FAKE_SOURCES=(binary_staged mmpd)

RUN_TAG="$(basename "$DISC_OUTPUT_SUFFIX")"

append_run_config_args() {
    local -n _out=$1
    _out+=(--anchor-config "$ANCHOR_CONFIG")
    _out+=(--mmpd-run "$(basename "$MMPD_OUTPUT_SUFFIX")")
    _out+=(--disc-run "$(basename "$DISC_OUTPUT_SUFFIX")")
    _out+=(--raw-run "$(basename "$RAW_OUTPUT_SUFFIX")")
    _out+=(--report "$REPORT_SUFFIX")
    _out+=(--mmpd-backbone "$MMPD_BACKBONE")
    [[ -n "${ANCHOR_CONFIG_BY_DATASET:-}" ]] && _out+=(--anchor-config-by-dataset "$ANCHOR_CONFIG_BY_DATASET")
    [[ -n "${BINARY_CONFIG:-}" ]] && _out+=(--binary-config "$BINARY_CONFIG")
    [[ -n "${BINARY_CONFIG_BY_DATASET:-}" ]] && _out+=(--binary-config-by-dataset "$BINARY_CONFIG_BY_DATASET")
    [[ -n "${IMPORT_MMPD_PACKS_FROM:-}" ]] && _out+=(--import-mmpd-packs-from "$IMPORT_MMPD_PACKS_FROM")
    [[ -n "${TEST_STRIDE:-}" ]] && _out+=(--test-stride "$TEST_STRIDE")
    [[ -n "${TEST_FRACTION:-}" ]] && _out+=(--test-fraction "$TEST_FRACTION")
    [[ -n "${LOOKBACK:-}" ]] && _out+=(--lookback "$LOOKBACK")
    [[ -n "${HORIZON:-}" ]] && _out+=(--horizon "$HORIZON")
    [[ -n "${WALL_OVERRIDE:-}" ]] && _out+=(--time "$WALL_OVERRIDE")
}

# ---------------------------------------------------------------------------
# Login node: submit
# ---------------------------------------------------------------------------
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
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

    BIN_SUFFIX=""
    [[ -n "$BIN_MATCH_FILTER" ]] && BIN_SUFFIX="-binmatch-${BIN_MATCH_FILTER}"

    if [[ "$MERGE_PARTIALS" -eq 1 ]]; then
        SMOKE_SUFFIX=""
        [[ "$SMOKE" -eq 1 ]] && SMOKE_SUFFIX="-smoke"
        RUN_STEM="$(date +%m-%d)-${RUN_TAG}${SMOKE_SUFFIX}${BIN_SUFFIX}"
        LOG_DIR="$REPO/results/logs/${RUN_STEM}"
        mkdir -p "$LOG_DIR"
        MERGE_ARGS=(--merge-partials-only)
        append_run_config_args MERGE_ARGS
        echo "Submitting merge-only job..."
        sbatch \
            --job-name="disc-tex-merge${SMOKE_SUFFIX}${BIN_SUFFIX}" \
            --account=aip-boyuwang \
            --nodes=1 \
            --cpus-per-task=2 \
            --mem=4G \
            --time=0:15:00 \
            --output="$LOG_DIR/disc-tex-merge${SMOKE_SUFFIX}${BIN_SUFFIX}-%j.log" \
            --error="$LOG_DIR/disc-tex-merge${SMOKE_SUFFIX}${BIN_SUFFIX}-%j.log" \
            --mail-type=FAIL \
            --mail-user=ccao87@uwo.ca \
            --export=ALL,MERGE_PARTIALS=1,SMOKE="$SMOKE",BIN_MATCH_FILTER="$BIN_MATCH_FILTER" \
            "$SCRIPT_DIR/slurm_discriminator_texture_staged_vs_mmpd.sh" \
            "${MERGE_ARGS[@]}"
        exit 0
    fi

    SMOKE_SUFFIX=""
    if [[ "$SMOKE" -eq 1 ]]; then
        SMOKE_SUFFIX="-smoke"
        WALL="0:45:00"
        MEM="24G"
        CPUS=4
    else
        WALL="8:00:00"
        MEM="50G"
        CPUS=8
    fi
    [[ -n "$WALL_OVERRIDE" ]] && WALL="$WALL_OVERRIDE"

    RUN_STEM="$(date +%m-%d)-${RUN_TAG}${SMOKE_SUFFIX}${BIN_SUFFIX}"
    LOG_DIR="$REPO/results/logs/${RUN_STEM}"
    mkdir -p "$LOG_DIR"

    SUBMIT_DATASETS=("${DATASETS[@]}")
    if [[ -n "$DATASET" ]]; then
        SUBMIT_DATASETS=("$DATASET")
    fi
    if [[ "$SMOKE" -eq 1 && -z "$DATASET" ]]; then
        SUBMIT_DATASETS=(ETTh1)
    fi

    SUBMIT_SOURCES=("${FAKE_SOURCES[@]}")
    if [[ -n "$FAKE_SOURCE" ]]; then
        SUBMIT_SOURCES=("$FAKE_SOURCE")
    fi
    if [[ "$SMOKE" -eq 1 && -z "$FAKE_SOURCE" ]]; then
        SUBMIT_SOURCES=(binary_staged)
    fi

    NEED_MMPD=0
    for src in "${SUBMIT_SOURCES[@]}"; do
        [[ "$src" == "mmpd" ]] && NEED_MMPD=1
    done
    if [[ "$NEED_MMPD" -eq 1 && -z "${IMPORT_MMPD_PACKS_FROM:-}" && ! -d "$REPO/temp/MMPD/.git" ]]; then
        echo "Preparing temp/MMPD checkout before parallel submissions..."
        mkdir -p "$REPO/temp"
        git clone https://github.com/Thinklab-SJTU/MMPD.git "$REPO/temp/MMPD"
    fi

    JOB_IDS=()
    for ds in "${SUBMIT_DATASETS[@]}"; do
        for src in "${SUBMIT_SOURCES[@]}"; do
            JOB_WALL="$WALL"
            JOB_NAME="disc-tex-${ds}-${src}${SMOKE_SUFFIX}"
            SUBMIT_ARGS=(--dataset "$ds" --fake-source "$src")
            append_run_config_args SUBMIT_ARGS
            [[ "$SMOKE" -eq 1 ]] && SUBMIT_ARGS+=(--smoke-test)
            [[ "$FORCE_RAW" -eq 1 ]] && SUBMIT_ARGS+=(--force-raw-eval)
            [[ "$FORCE_TRAIN" -eq 1 ]] && SUBMIT_ARGS+=(--force-train)
            [[ -n "$SLICE_LENGTH" ]] && SUBMIT_ARGS+=(--slice-length "$SLICE_LENGTH")
            [[ -n "$BIN_MATCH_FILTER" ]] && SUBMIT_ARGS+=(--bin-match-filter "$BIN_MATCH_FILTER")

            echo "Submitting discriminator texture eval for $ds / $src (L40S, wall=$JOB_WALL)..."
            job_id="$(sbatch --parsable \
                --job-name="$JOB_NAME" \
                --account=aip-boyuwang \
                --nodes=1 \
                --gres=gpu:l40s:1 \
                --cpus-per-task="$CPUS" \
                --mem="$MEM" \
                --time="$JOB_WALL" \
                --output="$LOG_DIR/${JOB_NAME}-%j.log" \
                --error="$LOG_DIR/${JOB_NAME}-%j.log" \
                --mail-type=FAIL \
                --mail-user=ccao87@uwo.ca \
                --export=ALL,DATASET="$ds",FAKE_SOURCE="$src",SMOKE="$SMOKE",SLICE_LENGTH="$SLICE_LENGTH",FORCE_RAW="$FORCE_RAW",FORCE_TRAIN="$FORCE_TRAIN",BIN_MATCH_FILTER="$BIN_MATCH_FILTER",ANCHOR_CONFIG="$ANCHOR_CONFIG",ANCHOR_CONFIG_BY_DATASET="$ANCHOR_CONFIG_BY_DATASET",BINARY_CONFIG="$BINARY_CONFIG",BINARY_CONFIG_BY_DATASET="$BINARY_CONFIG_BY_DATASET",IMPORT_MMPD_PACKS_FROM="$IMPORT_MMPD_PACKS_FROM",MMPD_BACKBONE="$MMPD_BACKBONE",TEST_STRIDE="$TEST_STRIDE",TEST_FRACTION="$TEST_FRACTION",LOOKBACK="$LOOKBACK",HORIZON="$HORIZON",MMPD_OUTPUT_SUFFIX="$MMPD_OUTPUT_SUFFIX",DISC_OUTPUT_SUFFIX="$DISC_OUTPUT_SUFFIX",RAW_OUTPUT_SUFFIX="$RAW_OUTPUT_SUFFIX",REPORT_SUFFIX="$REPORT_SUFFIX" \
                "$SCRIPT_DIR/slurm_discriminator_texture_staged_vs_mmpd.sh" \
                "${SUBMIT_ARGS[@]}")"
            JOB_IDS+=("$job_id")
        done
    done

    if [[ "${#JOB_IDS[@]}" -gt 0 ]]; then
        dep="afterok:$(IFS=:; echo "${JOB_IDS[*]}")"
        echo "Submitting merge job (depends on ${#JOB_IDS[@]} shard job(s))..."
        MERGE_ARGS=(--merge-partials-only)
        append_run_config_args MERGE_ARGS
        sbatch \
            --dependency="$dep" \
            --job-name="disc-tex-merge${SMOKE_SUFFIX}${BIN_SUFFIX}" \
            --account=aip-boyuwang \
            --nodes=1 \
            --cpus-per-task=2 \
            --mem=4G \
            --time=0:15:00 \
            --output="$LOG_DIR/disc-tex-merge${SMOKE_SUFFIX}${BIN_SUFFIX}-%j.log" \
            --error="$LOG_DIR/disc-tex-merge${SMOKE_SUFFIX}${BIN_SUFFIX}-%j.log" \
            --mail-type=FAIL \
            --mail-user=ccao87@uwo.ca \
            --export=ALL,MERGE_PARTIALS=1,SMOKE="$SMOKE",BIN_MATCH_FILTER="$BIN_MATCH_FILTER",DISC_OUTPUT_SUFFIX="$DISC_OUTPUT_SUFFIX",RAW_OUTPUT_SUFFIX="$RAW_OUTPUT_SUFFIX",REPORT_SUFFIX="$REPORT_SUFFIX" \
            "$SCRIPT_DIR/slurm_discriminator_texture_staged_vs_mmpd.sh" \
            "${MERGE_ARGS[@]}"
    fi
    exit 0
fi

# ---------------------------------------------------------------------------
# Compute node
# ---------------------------------------------------------------------------
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID   Node: ${SLURMD_NODENAME:-unknown}"
if [[ "${MERGE_PARTIALS:-0}" -eq 1 ]]; then
    echo "Mode:   merge partials only"
else
    echo "GPU:    $(nvidia-smi -L 2>/dev/null | head -1 || echo unknown)"
    echo "Shard:  dataset=${DATASET:-?} fake_source=${FAKE_SOURCE:-?} bin_match=${BIN_MATCH_FILTER:-off}"
fi
echo "Started: $(date)"
echo "=========================================="

if [[ -d "${SCRATCH:-}/ts-sandbox" ]]; then
    REPO="${SCRATCH}/ts-sandbox"
elif [[ -d "$HOME/ts-sandbox" ]]; then
    REPO="$HOME/ts-sandbox"
else
    REPO="$SCRIPT_DIR"
fi
cd "$REPO"

STORE="$REPO/results"
mkdir -p "$STORE/logs" "$STORE/datasets"

MMPD_OUTPUT_SUFFIX="${MMPD_OUTPUT_SUFFIX:-results/datasets/06-13-binary-mmpd-subset-compare}"
DISC_OUTPUT_SUFFIX="${DISC_OUTPUT_SUFFIX:-results/datasets/06-14-disc-texture-flat-subsets-ema099-vs-mmpd}"
RAW_OUTPUT_SUFFIX="${RAW_OUTPUT_SUFFIX:-results/datasets/06-14-raw-texture-flat-subsets-ema099-vs-mmpd}"
REPORT_SUFFIX="${REPORT_SUFFIX:-reports/06-14_discriminator_texture_flat_subsets_ema099_vs_mmpd.md}"

REQ="$REPO/setup/requirements-killarney.txt"
[[ -f "$REQ" ]] || { echo "ERROR: missing $REQ — run ./setup/killarney_freeze_requirements.sh on login node" >&2; exit 1; }
[[ -n "${SLURM_TMPDIR:-}" ]] || { echo "ERROR: SLURM_TMPDIR is not set." >&2; exit 1; }

module purge 2>/dev/null || true
module load StdEnv/2023 python/3.11 2>/dev/null || true
if [[ "${MERGE_PARTIALS:-0}" -ne 1 ]]; then
    module load cuda/12.2 cudnn/8.9 2>/dev/null || true
fi
command -v virtualenv >/dev/null || { echo "ERROR: virtualenv not available after module load." >&2; exit 1; }

echo "[setup] Building node-local venv on \$SLURM_TMPDIR from $REQ"
virtualenv --no-download "$SLURM_TMPDIR/env"
# shellcheck source=/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
export PYTHON="$SLURM_TMPDIR/env/bin/python"
pip install --no-index --upgrade pip -q
pip install --no-index -r "$REQ" -q
if [[ "${MERGE_PARTIALS:-0}" -eq 1 ]]; then
    "$PYTHON" -c "import torch, yaml, matplotlib; print('torch', torch.__version__)"
else
    "$PYTHON" -c "import torch, yaml, matplotlib; assert torch.cuda.is_available(), 'CUDA required'; print('torch', torch.__version__, 'gpu', torch.cuda.get_device_name(0))"
fi

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TS_SANDBOX_REPO="$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

OUTPUT_DIR="$(resolve_output_dir "$REPO")"
RAW_EVAL_DIR="$REPO/$RAW_OUTPUT_SUFFIX"
MMPD_ROOT="$REPO/$MMPD_OUTPUT_SUFFIX"
CKPT_BASE="$REPO/$CKPT_BASE_SUFFIX"
REPORT_PATH="$REPO/$REPORT_SUFFIX"
ANCHOR_CONFIG="${ANCHOR_CONFIG:-binary_anchor_stationary_flat_subsets_ema099}"
MMPD_BACKBONE="${MMPD_BACKBONE:-Decoder}"

if [[ "${MERGE_PARTIALS:-0}" -eq 1 ]]; then
    echo "[merge] output=$OUTPUT_DIR"
    "$PYTHON" -u "$REPO/utils/eval_discriminator_texture_staged_vs_mmpd.py" \
        --merge-partials-only \
        --output-dir "$OUTPUT_DIR" \
        --raw-eval-dir "$RAW_EVAL_DIR"

    echo "[report] regenerating discriminator report"
    "$PYTHON" -u "$REPO/utils/report_discriminator_texture_staged_vs_mmpd.py" \
        --metrics "$OUTPUT_DIR/metrics.json" \
        --manifest "$OUTPUT_DIR/run_manifest.json" \
        --output "$REPORT_PATH" || true

    echo "=========================================="
    echo "Merge complete: $(date)"
    echo "Metrics: $OUTPUT_DIR/metrics.json"
    echo "Plots:   $OUTPUT_DIR/disc_confusions/"
    echo "Ckpts:   $OUTPUT_DIR/checkpoints/"
    echo "Report:  $REPORT_PATH"
    echo "=========================================="
    exit 0
fi

MMPD_REPO="$REPO/temp/MMPD"
MMPD_DATA="$REPO/temp/mmpd_datasets"

if [[ -z "${DATASET:-}" || -z "${FAKE_SOURCE:-}" ]]; then
    echo "ERROR: shard jobs require DATASET and FAKE_SOURCE (via --export from login submit)." >&2
    exit 1
fi

EVAL_ARGS=(
    --output-dir "$OUTPUT_DIR"
    --raw-eval-dir "$RAW_EVAL_DIR"
    --ckpt-base "$CKPT_BASE"
    --anchor-config "$ANCHOR_CONFIG"
    --mmpd-output-root "$MMPD_ROOT"
    --mmpd-backbone "$MMPD_BACKBONE"
    --mmpd-repo "$MMPD_REPO"
    --mmpd-data-dir "$MMPD_DATA"
    --test-fraction "${TEST_FRACTION:-1.0}"
    --test-stride "${TEST_STRIDE:-2}"
    --lookback "${LOOKBACK:-96}"
    --horizon "${HORIZON:-96}"
    --num-sampling-steps 20
    --probabilistic-sampler dpmpp
    --gmm-components 1
    --datasets "$DATASET"
    --fake-sources "$FAKE_SOURCE"
    --no-merge-metrics
    --no-update-mmpd
)

[[ -n "${ANCHOR_CONFIG_BY_DATASET:-}" ]] && EVAL_ARGS+=(--anchor-config-by-dataset "$ANCHOR_CONFIG_BY_DATASET")
[[ -n "${BINARY_CONFIG:-}" ]] && EVAL_ARGS+=(--binary-config "$BINARY_CONFIG")
[[ -n "${BINARY_CONFIG_BY_DATASET:-}" ]] && EVAL_ARGS+=(--binary-config-by-dataset "$BINARY_CONFIG_BY_DATASET")
[[ -n "${IMPORT_MMPD_PACKS_FROM:-}" ]] && EVAL_ARGS+=(--import-mmpd-packs-from "$IMPORT_MMPD_PACKS_FROM")

if [[ -n "${SLICE_LENGTH:-}" ]]; then
    EVAL_ARGS+=(--slice-lengths "$SLICE_LENGTH")
fi

if [[ "${SMOKE:-0}" -eq 1 ]]; then
    EVAL_ARGS+=(
        --smoke-test
        --raw-binary-batch-size 1
        --raw-mmpd-batch-size 2
    )
else
    EVAL_ARGS+=(
        --epochs 20
        --batch-size 512
        --raw-binary-batch-size 2
        --raw-mmpd-batch-size 8
        --save-checkpoints
        --visualize-confusions
        --viz-per-bucket 2
    )
fi

if [[ "${FORCE_RAW:-0}" -eq 1 ]]; then
    EVAL_ARGS+=(--force-raw-eval)
fi

if [[ "${FORCE_TRAIN:-0}" -eq 1 ]]; then
    EVAL_ARGS+=(--force-train)
fi

if [[ -n "${BIN_MATCH_FILTER:-}" ]]; then
    EVAL_ARGS+=(--bin-match-filter "$BIN_MATCH_FILTER")
fi

echo "[eval] output=$OUTPUT_DIR raw=$RAW_EVAL_DIR dataset=$DATASET fake=$FAKE_SOURCE lookback=${LOOKBACK:-96} horizon=${HORIZON:-96} bin_match=${BIN_MATCH_FILTER:-off}"
"$PYTHON" -u "$REPO/utils/eval_discriminator_texture_staged_vs_mmpd.py" "${EVAL_ARGS[@]}"

echo "=========================================="
echo "Shard complete: $(date)"
echo "Partial: $OUTPUT_DIR/partials/${DATASET}__${FAKE_SOURCE}.json"
echo "Ckpts:   $OUTPUT_DIR/checkpoints/${DATASET}_${FAKE_SOURCE}_L*.pt"
echo "Plots:   $OUTPUT_DIR/disc_confusions/${DATASET}_${FAKE_SOURCE}_L*/"
echo "=========================================="
