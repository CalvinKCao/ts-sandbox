#!/bin/bash
# =============================================================================
# Pure compute-node worker script for diffusion pipeline.
#
# USAGE (do not call directly; use submit_binary.sh):
#   sbatch slurm_worker.sh --config configs/binary_anchor.yaml --dataset ETTh1
#
# Venv: node-local fast path — rebuilds on $SLURM_TMPDIR from
# setup/requirements-killarney.txt (generate via setup/killarney_freeze_requirements.sh).
# =============================================================================

set -euo pipefail

PY_ARGS=("$@")
ORDINAL_DISC_MODE="${GRID_EVAL_ORDINAL_PATCH_REFINE_MMPD:-0}"
ORDINAL_DISC_MERGE="${GRID_ORDINAL_DISC_MERGE:-0}"
ORDINAL_ASSERT_ONLY="${GRID_ORDINAL_ASSERT_ONLY:-0}"

ts() { date +'%d-%H:%M:%S'; }

echo "$(ts) =========================================="
echo "$(ts) Job: $SLURM_JOB_NAME  ID: $SLURM_JOB_ID  Node: ${SLURMD_NODENAME:-unknown}"
echo "$(ts) =========================================="

STORE="${GRID_STORE:-$SCRATCH/ts-sandbox/results}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
REQ="$REPO/setup/requirements-killarney.txt"

# Fail fast before venv build if the repo checkout is missing the requested config.
CONFIG_REL=""
for ((i = 1; i < $#; i++)); do
    if [[ "${!i}" == "--config" ]]; then
        j=$((i + 1))
        CONFIG_REL="${!j}"
        break
    fi
done
if [[ -n "$CONFIG_REL" ]]; then
    CONFIG_PATH="$REPO/$CONFIG_REL"
    if [[ ! -f "$CONFIG_PATH" ]]; then
        BRANCH="$(git -C "$REPO" branch --show-current 2>/dev/null || echo unknown)"
        echo "ERROR: config not found: $CONFIG_PATH" >&2
        echo "ERROR: repo branch=$BRANCH — git checkout feat/patch-decoder-cross-variate-ctx && git pull" >&2
        exit 1
    fi
fi

[[ -f "$REQ" ]] || { echo "ERROR: missing $REQ — run ./setup/killarney_freeze_requirements.sh on login node" >&2; exit 1; }
[[ -n "${SLURM_TMPDIR:-}" ]] || { echo "ERROR: SLURM_TMPDIR is not set." >&2; exit 1; }

module purge 2>/dev/null || true
module load StdEnv/2023 python/3.11 2>/dev/null || true
if [[ "$ORDINAL_DISC_MERGE" -ne 1 ]]; then
    module load cuda/12.2 cudnn/8.9 2>/dev/null || true
fi
command -v virtualenv >/dev/null || { echo "ERROR: virtualenv not available after module load." >&2; exit 1; }

echo "$(ts) [setup] Building node-local venv on \$SLURM_TMPDIR from $REQ"
virtualenv --no-download "$SLURM_TMPDIR/env"
# shellcheck source=/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index -r "$REQ" -q
if [[ "$ORDINAL_DISC_MERGE" -eq 1 ]]; then
    python -c "import torch, optuna, wandb, einops, yaml; print('torch', torch.__version__, 'cpu merge worker')"
else
    python -c "import torch, optuna, wandb, einops, yaml; assert torch.cuda.is_available(), 'CUDA is not available (check driver compatibility)!'; print('torch', torch.__version__, 'gpu', torch.cuda.get_device_name(0))"
fi

export PYTHONUNBUFFERED=1

cd "$REPO"
if [[ ! -f "models/diffusion_tsf/train_multivariate_pipeline.py" ]]; then
    echo "ERROR: slurm_worker.sh must be submitted from repo root." >&2
    exit 1
fi

# Explicit deferred mode used only by the h96 ordinal patch-refine campaign.
# Checkpoints are intentionally validated here: this job is submitted before
# its afterok parents have produced them, so a login-node existence check would
# reject a correct DAG.
if [[ "$ORDINAL_DISC_MODE" -eq 1 ]]; then
    [[ -n "${GRID_DISC_OUTPUT:-}" ]] || { echo "ERROR: GRID_DISC_OUTPUT is unset" >&2; exit 1; }
    [[ -n "${GRID_RAW_DISC_OUTPUT:-}" ]] || { echo "ERROR: GRID_RAW_DISC_OUTPUT is unset" >&2; exit 1; }
    [[ -n "${GRID_ORDINAL_DISC_EVALUATOR:-}" ]] || { echo "ERROR: GRID_ORDINAL_DISC_EVALUATOR is unset" >&2; exit 1; }
    [[ -n "${GRID_ORDINAL_BINARY_CONFIG:-}" ]] || { echo "ERROR: GRID_ORDINAL_BINARY_CONFIG is unset" >&2; exit 1; }
    [[ -f "$GRID_ORDINAL_DISC_EVALUATOR" ]] || { echo "ERROR: ordinal evaluator missing: $GRID_ORDINAL_DISC_EVALUATOR" >&2; exit 1; }
    [[ -f "$GRID_ORDINAL_BINARY_CONFIG" ]] || { echo "ERROR: ordinal binary config missing: $GRID_ORDINAL_BINARY_CONFIG" >&2; exit 1; }
    if [[ "$ORDINAL_DISC_MERGE" -eq 1 ]]; then
        echo "$(ts) [eval] merging ordinal patch-refine discriminator partials"
        python -u "$GRID_ORDINAL_DISC_EVALUATOR" \
            --merge-partials-only \
            --output-dir "$GRID_DISC_OUTPUT" \
            --raw-eval-dir "$GRID_RAW_DISC_OUTPUT"
        echo "$(ts) Done"
        exit 0
    fi

    [[ -n "${GRID_DATASET:-}" ]] || { echo "ERROR: GRID_DATASET is unset" >&2; exit 1; }
    [[ -n "${GRID_EXISTING_CKPT:-}" ]] || { echo "ERROR: GRID_EXISTING_CKPT is unset" >&2; exit 1; }
    [[ -n "${GRID_MMPD_ROOT:-}" ]] || { echo "ERROR: GRID_MMPD_ROOT is unset" >&2; exit 1; }
    [[ -d "$GRID_EXISTING_CKPT" ]] || { echo "ERROR: binary checkpoint root missing: $GRID_EXISTING_CKPT" >&2; exit 1; }
    [[ -d "$GRID_MMPD_ROOT" ]] || { echo "ERROR: MMPD output root missing: $GRID_MMPD_ROOT" >&2; exit 1; }
    mapfile -t coarse_ckpts < <(find "$GRID_EXISTING_CKPT" -maxdepth 3 -type f -path '*/coarse/best.pt' | sort)
    mapfile -t refine_ckpts < <(find "$GRID_EXISTING_CKPT" -maxdepth 3 -type f -path '*/patch_refine/best.pt' | sort)
    [[ "${#coarse_ckpts[@]}" -eq 1 ]] || { echo "ERROR: expected exactly one coarse best.pt under $GRID_EXISTING_CKPT" >&2; exit 1; }
    [[ "${#refine_ckpts[@]}" -eq 1 ]] || { echo "ERROR: expected exactly one patch_refine best.pt under $GRID_EXISTING_CKPT" >&2; exit 1; }
    echo "$(ts) [eval] ordinal patch-refine checkpoint: $GRID_EXISTING_CKPT"
    echo "$(ts) [eval] MMPD root: $GRID_MMPD_ROOT"
    ORDINAL_ASSERT_ARGS=()
    [[ "$ORDINAL_ASSERT_ONLY" -eq 0 ]] || ORDINAL_ASSERT_ARGS=(--assert-only --assert-max-windows "${GRID_ASSERT_MAX_WINDOWS:-8}")
    # Match MMPD pack index grid (eval_test_stride=4); keep full every-4 pack pool
    # (test_fraction=1.0, disc_index_stride=1). Unique-seg AR previously crawled when
    # the worker rebuilt the wrong stride-1/480 pool.
    # GRID_SLICE_LENGTHS uses ; or , separators (default 8;16;32)
    _sl_raw="${GRID_SLICE_LENGTHS:-8;16;32}"
    _sl_raw="${_sl_raw//,/;}"
    IFS=';' read -r -a SLICE_ARR <<< "${_sl_raw}"
    [[ "${#SLICE_ARR[@]}" -ge 1 ]] || { echo "ERROR: empty GRID_SLICE_LENGTHS=$_sl_raw" >&2; exit 1; }
    echo "$(ts) [eval] slice_lengths=${SLICE_ARR[*]}"
    python -u "$GRID_ORDINAL_DISC_EVALUATOR" \
        --datasets "$GRID_DATASET" \
        --checkpoint-dir "$GRID_EXISTING_CKPT" \
        --mmpd-output-root "$GRID_MMPD_ROOT" \
        --binary-config "$GRID_ORDINAL_BINARY_CONFIG" \
        --output-dir "$GRID_DISC_OUTPUT" \
        --raw-eval-dir "$GRID_RAW_DISC_OUTPUT" \
        --pack-test-stride 4 \
        --test-stride 4 \
        --test-fraction "${GRID_ORDINAL_TEST_FRACTION:-1.0}" \
        --disc-index-stride "${GRID_ORDINAL_DISC_INDEX_STRIDE:-1}" \
        --raw-binary-batch-size "${GRID_ORDINAL_BINARY_BATCH:-8}" \
        --slice-lengths "${SLICE_ARR[@]}" \
        --force-raw-eval \
        --force-train \
        "${ORDINAL_ASSERT_ARGS[@]}"
    echo "$(ts) Done"
    exit 0
fi

if [[ "${GRID_EVAL_PATCH_REFINE:-0}" -eq 1 ]]; then
    [[ -n "${GRID_DATASET:-}" ]] || { echo "ERROR: GRID_DATASET is unset" >&2; exit 1; }
    [[ -n "${GRID_EXISTING_CKPT:-}" ]] || { echo "ERROR: GRID_EXISTING_CKPT is unset" >&2; exit 1; }
    [[ -n "${GRID_DISC_OUTPUT:-}" ]] || { echo "ERROR: GRID_DISC_OUTPUT is unset" >&2; exit 1; }
    [[ -d "$GRID_EXISTING_CKPT" ]] || { echo "ERROR: checkpoint root missing: $GRID_EXISTING_CKPT" >&2; exit 1; }

    echo "$(ts) [eval] fixed h96 patch-refine checkpoint: $GRID_EXISTING_CKPT"
    echo "$(ts) [eval] discriminator output: $GRID_DISC_OUTPUT"
    python -u temp/eval_univariate_patch_refine_vs_gt.py \
        --datasets "$GRID_DATASET" \
        --checkpoint-dir "$GRID_EXISTING_CKPT" \
        --output-dir "$GRID_DISC_OUTPUT" \
        --test-stride 4 \
        --slice-lengths 8 16 32 \
        --force-raw-eval \
        --force-train
    echo "$(ts) Done"
    exit 0
fi

DATE_STR="${GRID_DATE_STR:-$(date +%m-%d)}"
DS="${GRID_DATASET:-unknown}"
CFG_NAME="${GRID_CFG_NAME:-run}"
RUN_STEM="${GRID_RUN_STEM:-${DATE_STR}-${SLURM_JOB_ID}-${DS}-${CFG_NAME}}"

# Align Slurm display name with the full run stem once the job id is known.
if [[ -n "${SLURM_JOB_ID:-}" && "${SLURM_JOB_NAME:-}" != "$RUN_STEM" ]]; then
    scontrol update "JobId=${SLURM_JOB_ID}" "JobName=${RUN_STEM}" 2>/dev/null || true
fi


CKPT_DIR="$STORE/ckpts/${RUN_STEM}"
DATA_DIR="$STORE/datasets/${RUN_STEM}"
mkdir -p "$CKPT_DIR" "$DATA_DIR"

# Benchmark CSVs live in the repo clone ($SCRATCH/ts-sandbox/datasets), not under $STORE/datasets.
BENCHMARK_DATASETS="${DATASETS_DIR:-$REPO/datasets}"
[[ -d "$BENCHMARK_DATASETS" ]] || {
    echo "ERROR: benchmark data directory not found at $BENCHMARK_DATASETS" >&2
    exit 1
}

echo "$(ts) Repo: $PWD"
echo "$(ts) Checkpoints: $CKPT_DIR"
echo "$(ts) Results: $DATA_DIR"
echo "$(ts) Benchmark CSVs: $BENCHMARK_DATASETS"
echo "$(ts) GPUs: $(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ') ($(nvidia-smi -L 2>/dev/null | head -1 || echo none))"

PY_ARGS+=(
    --checkpoint-dir "$CKPT_DIR"
    --results-dir "$DATA_DIR"
    --datasets-dir "$BENCHMARK_DATASETS"
)

echo "$(ts) [train] Starting pipeline: ${PY_ARGS[*]}"
python -u -m models.diffusion_tsf.train_multivariate_pipeline "${PY_ARGS[@]}"

echo "$(ts) =========================================="
echo "$(ts) Done"
echo "$(ts) =========================================="
