#!/bin/bash
# =============================================================================
# Learned discriminator texture eval: staged binary vs MMPD (Killarney L40S).
#
# Default login-node behavior submits one independent job per (dataset, fake
# source) so binary and MMPD discriminators train in parallel (10 jobs for 5
# datasets). A lightweight CPU merge job rebuilds metrics.json when all shards
# finish.
#
# USAGE (login node, from $SCRATCH/ts-sandbox):
#   ./slurm_discriminator_texture_staged_vs_mmpd.sh --smoke-test
#   ./slurm_discriminator_texture_staged_vs_mmpd.sh
#   ./slurm_discriminator_texture_staged_vs_mmpd.sh --dataset traffic
#   ./slurm_discriminator_texture_staged_vs_mmpd.sh --dataset ETTh1 --fake-source mmpd
#   ./slurm_discriminator_texture_staged_vs_mmpd.sh --slice-length 16
#   ./slurm_discriminator_texture_staged_vs_mmpd.sh --force-raw-eval
#   ./slurm_discriminator_texture_staged_vs_mmpd.sh --merge-partials-only
#   ./slurm_discriminator_texture_staged_vs_mmpd.sh --bin-match-filter mmpd
#   ./slurm_discriminator_texture_staged_vs_mmpd.sh --bin-match-filter all --force-train
# Resubmit with purged split + checkpoints + confusion PNGs:
#   ./slurm_discriminator_texture_staged_vs_mmpd.sh --bin-match-filter all --force-train
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

resolve_output_dir() {
    local repo_root="$1"
    local base="${repo_root}/results/datasets/06-03-discriminator-texture-staged-vs-mmpd"
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
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ -n "$BIN_MATCH_FILTER" && "$BIN_MATCH_FILTER" != "mmpd" && "$BIN_MATCH_FILTER" != "both" && "$BIN_MATCH_FILTER" != "all" ]]; then
    echo "ERROR: --bin-match-filter must be one of: mmpd, both, all" >&2
    exit 1
fi

DATASETS=(ETTh1 dalia traffic exchange_rate PeMS)
FAKE_SOURCES=(binary_staged mmpd)

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
        RUN_STEM="$(date +%m-%d)-disc-texture-staged-vs-mmpd${SMOKE_SUFFIX}${BIN_SUFFIX}"
        LOG_DIR="$REPO/results/logs/${RUN_STEM}"
        mkdir -p "$LOG_DIR"
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
            "$SCRIPT_DIR/slurm_discriminator_texture_staged_vs_mmpd.sh"
        exit 0
    fi

    SMOKE_SUFFIX=""
    if [[ "$SMOKE" -eq 1 ]]; then
        SMOKE_SUFFIX="-smoke"
        WALL="0:30:00"
        MEM="24G"
        CPUS=4
    else
        WALL="3:00:00"
        MEM="50G"
        CPUS=8
    fi

    RUN_STEM="$(date +%m-%d)-disc-texture-staged-vs-mmpd${SMOKE_SUFFIX}${BIN_SUFFIX}"
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
    if [[ "$NEED_MMPD" -eq 1 && ! -d "$REPO/temp/MMPD/.git" ]]; then
        echo "Preparing temp/MMPD checkout before parallel submissions..."
        mkdir -p "$REPO/temp"
        git clone https://github.com/Thinklab-SJTU/MMPD.git "$REPO/temp/MMPD"
    fi

    JOB_IDS=()
    for ds in "${SUBMIT_DATASETS[@]}"; do
        for src in "${SUBMIT_SOURCES[@]}"; do
            JOB_NAME="disc-tex-${ds}-${src}${SMOKE_SUFFIX}"
            SUBMIT_ARGS=(--dataset "$ds" --fake-source "$src")
            [[ "$SMOKE" -eq 1 ]] && SUBMIT_ARGS+=(--smoke-test)
            [[ "$FORCE_RAW" -eq 1 ]] && SUBMIT_ARGS+=(--force-raw-eval)
            [[ "$FORCE_TRAIN" -eq 1 ]] && SUBMIT_ARGS+=(--force-train)
            [[ -n "$SLICE_LENGTH" ]] && SUBMIT_ARGS+=(--slice-length "$SLICE_LENGTH")
            [[ -n "$BIN_MATCH_FILTER" ]] && SUBMIT_ARGS+=(--bin-match-filter "$BIN_MATCH_FILTER")

            echo "Submitting discriminator texture eval for $ds / $src (L40S, wall=$WALL)..."
            job_id="$(sbatch --parsable \
                --job-name="$JOB_NAME" \
                --account=aip-boyuwang \
                --nodes=1 \
                --gres=gpu:l40s:1 \
                --cpus-per-task="$CPUS" \
                --mem="$MEM" \
                --time="$WALL" \
                --output="$LOG_DIR/${JOB_NAME}-%j.log" \
                --error="$LOG_DIR/${JOB_NAME}-%j.log" \
                --mail-type=FAIL \
                --mail-user=ccao87@uwo.ca \
                --export=ALL,DATASET="$ds",FAKE_SOURCE="$src",SMOKE="$SMOKE",SLICE_LENGTH="$SLICE_LENGTH",FORCE_RAW="$FORCE_RAW",FORCE_TRAIN="$FORCE_TRAIN",BIN_MATCH_FILTER="$BIN_MATCH_FILTER" \
                "$SCRIPT_DIR/slurm_discriminator_texture_staged_vs_mmpd.sh" \
                "${SUBMIT_ARGS[@]}")"
            JOB_IDS+=("$job_id")
        done
    done

    if [[ "${#JOB_IDS[@]}" -gt 0 ]]; then
        dep="afterok:$(IFS=:; echo "${JOB_IDS[*]}")"
        echo "Submitting merge job (depends on ${#JOB_IDS[@]} shard job(s))..."
        sbatch \
            --dependency="$dep" \
            --job-name="disc-tex-merge${SMOKE_SUFFIX}" \
            --account=aip-boyuwang \
            --nodes=1 \
            --cpus-per-task=2 \
            --mem=4G \
            --time=0:15:00 \
            --output="$LOG_DIR/disc-tex-merge${SMOKE_SUFFIX}-%j.log" \
            --error="$LOG_DIR/disc-tex-merge${SMOKE_SUFFIX}-%j.log" \
            --mail-type=FAIL \
            --mail-user=ccao87@uwo.ca \
            --export=ALL,MERGE_PARTIALS=1,SMOKE="$SMOKE",BIN_MATCH_FILTER="$BIN_MATCH_FILTER" \
            "$SCRIPT_DIR/slurm_discriminator_texture_staged_vs_mmpd.sh"
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

pip_retry() {
    local max_attempts=5 delay=20 attempt
    for attempt in $(seq 1 "$max_attempts"); do
        if "$@"; then return 0; fi
        if [[ "$attempt" -lt "$max_attempts" ]]; then
            echo "[setup] pip failed (attempt ${attempt}/${max_attempts}), retry in ${delay}s..."
            sleep "$delay"
            delay=$((delay + 2))
        fi
    done
    echo "[setup] pip failed after ${max_attempts} attempts: $*" >&2
    return 1
}

install_pipeline_deps() {
    pip_retry pip install --no-index --upgrade pip -q 2>/dev/null || pip_retry pip install -U pip -q
    if ! python -c "import torch" 2>/dev/null; then
        if ! pip_retry pip install --no-index 'torch==2.11.0+computecanada' numpy pandas scipy scikit-learn tqdm -q 2>/dev/null; then
            pip_retry pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q
            pip_retry pip install numpy pandas scipy scikit-learn tqdm -q
        fi
    fi
    pip_retry pip install optuna einops pyyaml scikit-learn matplotlib -q
}

_load_modules() {
    module purge 2>/dev/null || true
    module load StdEnv/2023 2>/dev/null || true
    module load python/3.11 2>/dev/null || true
    if [[ "${MERGE_PARTIALS:-0}" -ne 1 ]]; then
        module load cuda/12.2 2>/dev/null || true
        module load cudnn/8.9 2>/dev/null || true
    fi
}

VENV=""
for cand in \
    "$STORE/venv" \
    "${SCRATCH:-}/${USER}/ts-sandbox/results/venv" \
    "${SCRATCH:-}/ts-sandbox/results/venv"; do
    if [[ -x "${cand}/bin/python" ]]; then
        VENV="$cand"
        break
    fi
done

_load_modules
if [[ -n "$VENV" ]]; then
    echo "[setup] Using persistent venv: $VENV"
    # shellcheck source=/dev/null
    source "$VENV/bin/activate"
    export PATH="$VENV/bin:$PATH"
    export PYTHON="$VENV/bin/python"
    install_pipeline_deps
else
    echo "[setup] Building venv on \${SLURM_TMPDIR:-/tmp}..."
    python -m venv "${SLURM_TMPDIR:-/tmp}/env"
    VENV="${SLURM_TMPDIR:-/tmp}/env"
    # shellcheck source=/dev/null
    source "$VENV/bin/activate"
    export PATH="$VENV/bin:$PATH"
    export PYTHON="$VENV/bin/python"
    install_pipeline_deps
fi

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TS_SANDBOX_REPO="$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

OUTPUT_DIR="$(resolve_output_dir "$REPO")"
RAW_EVAL_DIR="$REPO/results/datasets/06-03-trend-robust-texture-staged-vs-mmpd"
MMPD_ROOT="$REPO/results/datasets/06-01-mmpd-binary-aligned"
MMPD_REPO="$REPO/temp/MMPD"
MMPD_DATA="$REPO/temp/mmpd_datasets"

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
        --output "$REPO/reports/06-03_discriminator_texture_staged_vs_mmpd.md" || true

    "$PYTHON" -u "$REPO/utils/report_trend_robust_texture_staged_vs_mmpd.py" || true

    echo "=========================================="
    echo "Merge complete: $(date)"
    echo "Metrics: $OUTPUT_DIR/metrics.json"
    echo "Plots:   $OUTPUT_DIR/disc_confusions/"
    echo "Ckpts:   $OUTPUT_DIR/checkpoints/"
    echo "Report:  $REPO/reports/06-03_discriminator_texture_staged_vs_mmpd.md"
    echo "=========================================="
    exit 0
fi

if [[ -z "${DATASET:-}" || -z "${FAKE_SOURCE:-}" ]]; then
    echo "ERROR: shard jobs require DATASET and FAKE_SOURCE (via --export from login submit)." >&2
    exit 1
fi

"$PYTHON" -c "import torch; assert torch.cuda.is_available(), 'CUDA required'; print('torch', torch.__version__, 'gpu', torch.cuda.get_device_name(0))"

EVAL_ARGS=(
    --output-dir "$OUTPUT_DIR"
    --raw-eval-dir "$RAW_EVAL_DIR"
    --mmpd-output-root "$MMPD_ROOT"
    --mmpd-repo "$MMPD_REPO"
    --mmpd-data-dir "$MMPD_DATA"
    --test-fraction 1.0
    --test-stride 2
    --num-sampling-steps 20
    --probabilistic-sampler dpmpp
    --gmm-components 1
    --datasets "$DATASET"
    --fake-sources "$FAKE_SOURCE"
    --no-merge-metrics
    --no-update-mmpd
)

if [[ -n "${SLICE_LENGTH:-}" ]]; then
    EVAL_ARGS+=(--slice-lengths "$SLICE_LENGTH")
fi

if [[ "${SMOKE:-0}" -eq 1 ]]; then
    EVAL_ARGS+=(
        --smoke-test
        --raw-binary-batch-size 2
        --raw-mmpd-batch-size 4
    )
else
    EVAL_ARGS+=(
        --epochs 20
        --batch-size 512
        --raw-binary-batch-size 8
        --raw-mmpd-batch-size 16
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

echo "[eval] output=$OUTPUT_DIR raw=$RAW_EVAL_DIR dataset=$DATASET fake=$FAKE_SOURCE bin_match=${BIN_MATCH_FILTER:-off}"
"$PYTHON" -u "$REPO/utils/eval_discriminator_texture_staged_vs_mmpd.py" "${EVAL_ARGS[@]}"

echo "=========================================="
echo "Shard complete: $(date)"
echo "Partial: $OUTPUT_DIR/partials/${DATASET}__${FAKE_SOURCE}.json"
echo "Ckpts:   $OUTPUT_DIR/checkpoints/${DATASET}_${FAKE_SOURCE}_L*.pt"
echo "Plots:   $OUTPUT_DIR/disc_confusions/${DATASET}_${FAKE_SOURCE}_L*/"
echo "=========================================="
