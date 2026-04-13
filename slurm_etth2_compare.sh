#!/bin/bash
# =============================================================================
# ETTh2: Gaussian vs Binary diffusion — Killarney, job-chained
#
# Dependency graph:
#   A  iTrans HP + pretrain + Gaussian diff HP + pretrain
#   B  (copy iTrans from A) → Binary diff HP + pretrain        [afterok:A]
#   C  Gaussian finetune ETTh2                                  [afterok:A]
#   D  Binary  finetune ETTh2                                   [afterok:B]
#
# USAGE (from ts-sandbox repo root on the login node):
#   ./slurm_etth2_compare.sh           # full H100 run
#   ./slurm_etth2_compare.sh --smoke   # L40S smoke test — verifies full chain
#
# HOW TO SMOKE TEST:
#   --smoke submits all 4 jobs with 30-min time limits and passes --smoke-test
#   to the Python pipeline (1 epoch, 4 samples, 1 HP trial).  Each job should
#   finish in ~5 min on L40S, so the whole A→B→D chain completes in ~15-20 min.
#   Watch with:  squeue -u $USER
#   Check logs:  tail -f $STORE/logs/A-gauss-pretrain-<JOB_ID>.out
# =============================================================================

set -euo pipefail

# ---- Parse flags ------------------------------------------------------------
SMOKE=0
for arg in "$@"; do [ "$arg" = "--smoke" ] && SMOKE=1; done

# ---- Repo root --------------------------------------------------------------
if   [ -d "${SCRATCH:-}/ts-sandbox" ]; then REPO="$SCRATCH/ts-sandbox"
elif [ -d "$HOME/ts-sandbox" ];         then REPO="$HOME/ts-sandbox"
else echo "ERROR: ts-sandbox not found in SCRATCH or HOME" && exit 1
fi

# ---- Per-user storage under PROJECT -----------------------------------------
# Alliance convention: $PROJECT/$USER/... for your working data
if [ -z "${PROJECT:-}" ]; then
    PROJECT=$(ls -d "$HOME/projects/aip-"* "$HOME/projects/def-"* 2>/dev/null \
              | head -1 || true)
fi
[ -z "${PROJECT:-}" ] && { echo "ERROR: \$PROJECT not set"; exit 1; }

export STORE="$PROJECT/$USER/diffusion-tsf-etth2"
export GAUSS_CKPT="$STORE/checkpoints_gauss"
export GAUSS_RESULTS="$STORE/results_gauss"
export BINARY_CKPT="$STORE/checkpoints_binary"
export BINARY_RESULTS="$STORE/results_binary"
LOG_DIR="$STORE/logs"

mkdir -p "$GAUSS_CKPT" "$GAUSS_RESULTS" \
         "$BINARY_CKPT" "$BINARY_RESULTS" \
         "$LOG_DIR"

# Keep datasets accessible from STORE without copying GBs
[ ! -e "$STORE/datasets" ] && ln -s "$REPO/datasets" "$STORE/datasets"

# ---- Venv path (exported so jobs can see it) --------------------------------
export VENV="$PROJECT/$USER/diffusion-tsf/venv"
[ ! -d "$VENV" ] && export VENV="$STORE/venv"

# ---- Repo path (exported) ---------------------------------------------------
export REPO

# ---- Resources + flags ------------------------------------------------------
if [ "$SMOKE" -eq 1 ]; then
    GPU_ARGS=(--gres=gpu:l40s:1)
    WALL_PRETRAIN="0:30:00"
    WALL_FINETUNE="0:30:00"
    MEM="16G"; CPUS=4
    export SMOKE_FLAG="--smoke-test"
    SUFFIX="-smoke"
else
    GPU_ARGS=(--partition=gpubase_h100_b4 --gpus-per-node=h100:1)
    WALL_PRETRAIN="14:00:00"
    WALL_FINETUNE="8:00:00"
    MEM="60G"; CPUS=6
    export SMOKE_FLAG=""
    SUFFIX=""
fi

# Common Python flags (exported so job bodies can use $SMOKE_FLAG etc.)
export PY="python -u -m models.diffusion_tsf.train_multivariate_pipeline"
export PY_COMMON="--n-variates 7 --amp --synthetic-samples 100000 --itransformer-trials 20 $SMOKE_FLAG"

# ---- Shared job body: module load + venv activate + cd ----------------------
# Written as a quoted heredoc into a temp file so each job can source it.
# (Avoids duplicating 15 lines × 4 jobs while keeping expansion safe.)
PREAMBLE_FILE="$(mktemp /tmp/job_preamble_XXXXXX.sh)"
cat > "$PREAMBLE_FILE" << 'PREAMBLE'
set -euo pipefail
echo "======================================================="
echo "  Job: $SLURM_JOB_NAME   ID: $SLURM_JOB_ID"
echo "  Node: $SLURMD_NODENAME"
echo "  GPU:  $(nvidia-smi -L 2>/dev/null | head -1 || echo none)"
echo "  Started: $(date)"
echo "======================================================="

module purge
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9

if [ ! -d "$VENV" ]; then
    echo "[setup] Creating venv at $VENV ..."
    python -m venv "$VENV"
    source "$VENV/bin/activate"
    pip install --upgrade pip -q
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q
    pip install numpy pandas scipy scikit-learn optuna wandb tqdm matplotlib einops -q
    [ -f "$REPO/requirements.txt" ] && pip install -r "$REPO/requirements.txt" -q
else
    source "$VENV/bin/activate"
fi

export WANDB_MODE=offline
export PYTHONUNBUFFERED=1

echo "[info] python: $(which python)"
cd "$REPO"
echo "[info] cwd: $(pwd)"
echo ""
PREAMBLE
# Will be sourced by each job via:  source "$PREAMBLE_FILE"
export PREAMBLE_FILE

# ============================================================================
# JOB A — Gaussian pretrain
# ============================================================================

echo "Submitting A: Gaussian pretrain..."

JOB_A=$(sbatch --parsable \
    --job-name="etth2-gauss-pretrain${SUFFIX}" \
    --account=aip-boyuwang \
    --nodes=1 --cpus-per-task="$CPUS" --mem="$MEM" \
    "${GPU_ARGS[@]}" \
    --time="$WALL_PRETRAIN" \
    --output="$LOG_DIR/A-gauss-pretrain-%j.out" \
    --error="$LOG_DIR/A-gauss-pretrain-%j.err" \
    --mail-type=FAIL --mail-user=ccao87@uwo.ca \
    << 'ENDSCRIPT'
#!/bin/bash
source "$PREAMBLE_FILE"

$PY --mode pretrain \
    --checkpoint-dir "$GAUSS_CKPT" \
    --results-dir    "$GAUSS_RESULTS" \
    $PY_COMMON

echo "[A] Gaussian pretrain done: $(date)"
ENDSCRIPT
)
echo "  -> A: $JOB_A"


# ============================================================================
# JOB B — Binary pretrain (copy iTrans from A, then binary diff HP + pretrain)
# ============================================================================

echo "Submitting B: Binary pretrain [afterok:$JOB_A]..."

JOB_B=$(sbatch --parsable \
    --job-name="etth2-binary-pretrain${SUFFIX}" \
    --account=aip-boyuwang \
    --nodes=1 --cpus-per-task="$CPUS" --mem="$MEM" \
    "${GPU_ARGS[@]}" \
    --time="$WALL_PRETRAIN" \
    --dependency="afterok:$JOB_A" \
    --output="$LOG_DIR/B-binary-pretrain-%j.out" \
    --error="$LOG_DIR/B-binary-pretrain-%j.err" \
    --mail-type=FAIL --mail-user=ccao87@uwo.ca \
    << 'ENDSCRIPT'
#!/bin/bash
source "$PREAMBLE_FILE"

# Reuse iTrans artifacts from the Gaussian run — pipeline skips iTrans pretrain
# if the checkpoint already exists, so we only pay for binary diffusion HP + pretrain.
echo "[B] Copying iTrans artifacts from Gaussian checkpoint dir..."
cp -v "$GAUSS_CKPT/pretrained_itransformer.pt" "$BINARY_CKPT/pretrained_itransformer.pt"
[ -f "$GAUSS_CKPT/itrans_hp.json" ] && \
    cp -v "$GAUSS_CKPT/itrans_hp.json" "$BINARY_CKPT/itrans_hp.json"

echo "[B] Running binary diffusion HP + pretrain..."
$PY --mode pretrain \
    --binary-diffusion \
    --checkpoint-dir "$BINARY_CKPT" \
    --results-dir    "$BINARY_RESULTS" \
    $PY_COMMON

echo "[B] Binary pretrain done: $(date)"
ENDSCRIPT
)
echo "  -> B: $JOB_B"


# ============================================================================
# JOB C — Gaussian finetune ETTh2
# ============================================================================

echo "Submitting C: Gaussian finetune ETTh2 [afterok:$JOB_A]..."

JOB_C=$(sbatch --parsable \
    --job-name="etth2-gauss-finetune${SUFFIX}" \
    --account=aip-boyuwang \
    --nodes=1 --cpus-per-task="$CPUS" --mem="$MEM" \
    "${GPU_ARGS[@]}" \
    --time="$WALL_FINETUNE" \
    --dependency="afterok:$JOB_A" \
    --output="$LOG_DIR/C-gauss-finetune-%j.out" \
    --error="$LOG_DIR/C-gauss-finetune-%j.err" \
    --mail-type=FAIL --mail-user=ccao87@uwo.ca \
    << 'ENDSCRIPT'
#!/bin/bash
source "$PREAMBLE_FILE"

$PY --mode finetune \
    --dataset ETTh2 \
    --checkpoint-dir "$GAUSS_CKPT" \
    --results-dir    "$GAUSS_RESULTS" \
    $PY_COMMON

echo "[C] Gaussian finetune done: $(date)"
ENDSCRIPT
)
echo "  -> C: $JOB_C"


# ============================================================================
# JOB D — Binary finetune ETTh2
# ============================================================================

echo "Submitting D: Binary finetune ETTh2 [afterok:$JOB_B]..."

JOB_D=$(sbatch --parsable \
    --job-name="etth2-binary-finetune${SUFFIX}" \
    --account=aip-boyuwang \
    --nodes=1 --cpus-per-task="$CPUS" --mem="$MEM" \
    "${GPU_ARGS[@]}" \
    --time="$WALL_FINETUNE" \
    --dependency="afterok:$JOB_B" \
    --output="$LOG_DIR/D-binary-finetune-%j.out" \
    --error="$LOG_DIR/D-binary-finetune-%j.err" \
    --mail-type=FAIL --mail-user=ccao87@uwo.ca \
    << 'ENDSCRIPT'
#!/bin/bash
source "$PREAMBLE_FILE"

$PY --mode finetune \
    --dataset ETTh2 \
    --binary-diffusion \
    --checkpoint-dir "$BINARY_CKPT" \
    --results-dir    "$BINARY_RESULTS" \
    $PY_COMMON

echo "[D] Binary finetune done: $(date)"
ENDSCRIPT
)
echo "  -> D: $JOB_D"


# ============================================================================
# Summary
# ============================================================================

echo ""
echo "=================================================================="
echo "  Jobs submitted:"
echo ""
echo "  A $JOB_A  Gaussian pretrain"
echo "  B $JOB_B  Binary pretrain      [afterok:$JOB_A]"
echo "  C $JOB_C  Gaussian finetune    [afterok:$JOB_A]"
echo "  D $JOB_D  Binary finetune      [afterok:$JOB_B]"
echo ""
echo "  Logs (tail -f to watch live):"
echo "    $LOG_DIR/"
echo ""
echo "  Monitor:     squeue -u \$USER"
echo "  Cancel all:  scancel $JOB_A $JOB_B $JOB_C $JOB_D"
echo "=================================================================="
