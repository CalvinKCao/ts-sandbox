#!/bin/bash
# =============================================================================
# ETTh2 Gaussian vs Binary diffusion comparison — Killarney, job-chained
#
# Dependency graph:
#   A  iTrans HP + pretrain + Gaussian diffusion HP + pretrain
#   B  (copy iTrans from A) → Binary diffusion HP + pretrain    [afterok:A]
#   C  Gaussian finetune ETTh2                                   [afterok:A]
#   D  Binary  finetune ETTh2                                    [afterok:B]
#
# USAGE (from ts-sandbox repo root on login node):
#   ./slurm_etth2_compare.sh              # full L40S run
#   ./slurm_etth2_compare.sh --smoke      # smoke test (~20 min per job)
# =============================================================================

set -euo pipefail

# ---- Parse flags ------------------------------------------------------------
SMOKE=0
for arg in "$@"; do [ "$arg" = "--smoke" ] && SMOKE=1; done

# ---- Repo root --------------------------------------------------------------
if   [ -d "$SCRATCH/ts-sandbox" ]; then REPO="$SCRATCH/ts-sandbox"
elif [ -d "$HOME/ts-sandbox"    ]; then REPO="$HOME/ts-sandbox"
else echo "ERROR: ts-sandbox not found in SCRATCH or HOME" && exit 1
fi

# ---- Storage (per-user under PROJECT) ---------------------------------------
if [ -z "${PROJECT:-}" ]; then
    PROJECT=$(ls -d "$HOME/projects/aip-"* "$HOME/projects/def-"* 2>/dev/null \
              | head -1 || true)
fi
if [ -z "${PROJECT:-}" ]; then
    echo "ERROR: \$PROJECT not set and no ~/projects/aip-* or def-* found"
    exit 1
fi

STORE="$PROJECT/$USER/diffusion-tsf-etth2"
GAUSS_CKPT="$STORE/checkpoints_gauss"
GAUSS_RESULTS="$STORE/results_gauss"
BINARY_CKPT="$STORE/checkpoints_binary"
BINARY_RESULTS="$STORE/results_binary"
LOG_DIR="$STORE/logs"

mkdir -p "$GAUSS_CKPT" "$GAUSS_RESULTS" "$BINARY_CKPT" "$BINARY_RESULTS" "$LOG_DIR"

if [ ! -e "$STORE/datasets" ]; then
    ln -s "$REPO/datasets" "$STORE/datasets"
fi

# ---- Resources --------------------------------------------------------------
if [ "$SMOKE" -eq 1 ]; then
    WALL_PRETRAIN="0:30:00"
    WALL_FINETUNE="0:30:00"
    MEM="16G"
    CPUS=4
    SMOKE_FLAG="--smoke-test"
    SUFFIX="-smoke"
else
    WALL_PRETRAIN="2-00:00:00"
    WALL_FINETUNE="1-00:00:00"
    MEM="32G"
    CPUS=6
    SMOKE_FLAG=""
    SUFFIX=""
fi

# ---- Shared venv path -------------------------------------------------------
VENV="$PROJECT/$USER/diffusion-tsf/venv"
[ ! -d "$VENV" ] && VENV="$STORE/venv"

# ---- Python command helpers -------------------------------------------------
PY_BASE="python -u -m models.diffusion_tsf.train_multivariate_pipeline"
PY_COMMON="--n-variates 7 --amp --synthetic-samples 100000 --itransformer-trials 20 ${SMOKE_FLAG}"

# ---- Preamble (runs at job start on the compute node) -----------------------
# Note: submission-time vars like $REPO/$VENV are baked in now.
#       Runtime Slurm vars like $SLURM_JOB_ID are escaped and expand later.
read -r -d '' JOB_PREAMBLE << 'PREAMBLE_EOF' || true
set -euo pipefail
echo "======================================================="
echo "  Job: $SLURM_JOB_NAME  ID: $SLURM_JOB_ID"
echo "  Node: $SLURMD_NODENAME"
echo "  GPU: $(nvidia-smi -L 2>/dev/null | head -1 || echo none)"
echo "  Started: $(date)"
echo "======================================================="

module purge
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9
PREAMBLE_EOF

# ---- Helper: write a job script to a tmpfile and submit it -----------------
# Usage: submit_job <tmpfile> [extra sbatch args...]
# Returns the job ID via stdout.
submit_job() {
    local tmpfile="$1"; shift
    sbatch --parsable "$@" "$tmpfile"
}


# ==========================================================================
# JOB A — Gaussian pretrain
# ==========================================================================
echo "Submitting Job A: Gaussian pretrain..."

TMP_A=$(mktemp /tmp/slurm_jobA_XXXXXX.sh)
cat > "$TMP_A" << SCRIPT_EOF
#!/bin/bash
#SBATCH --job-name=etth2-gauss-pretrain${SUFFIX}
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=${WALL_PRETRAIN}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --output=${LOG_DIR}/A-gauss-pretrain-%j.out
#SBATCH --error=${LOG_DIR}/A-gauss-pretrain-%j.err
#SBATCH --account=aip-boyuwang
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ccao87@uwo.ca

${JOB_PREAMBLE}

# Activate venv (create if missing)
if [ ! -d "${VENV}" ]; then
    echo "[setup] Creating venv at ${VENV}..."
    python -m venv "${VENV}"
    source "${VENV}/bin/activate"
    pip install --upgrade pip -q
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q
    pip install numpy pandas scipy scikit-learn optuna wandb tqdm matplotlib einops -q
    [ -f "${REPO}/requirements.txt" ] && pip install -r "${REPO}/requirements.txt" -q
else
    source "${VENV}/bin/activate"
fi

export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
cd "${REPO}"

${PY_BASE} \
    --mode pretrain \
    --checkpoint-dir ${GAUSS_CKPT} \
    --results-dir    ${GAUSS_RESULTS} \
    ${PY_COMMON}

echo "[A] Gaussian pretrain complete: \$(date)"
SCRIPT_EOF

JOB_A=$(submit_job "$TMP_A")
rm -f "$TMP_A"
echo "  -> Job A ID: ${JOB_A}"


# ==========================================================================
# JOB B — Binary pretrain (depends on A)
# ==========================================================================
echo "Submitting Job B: Binary pretrain (depends on A=${JOB_A})..."

TMP_B=$(mktemp /tmp/slurm_jobB_XXXXXX.sh)
cat > "$TMP_B" << SCRIPT_EOF
#!/bin/bash
#SBATCH --job-name=etth2-binary-pretrain${SUFFIX}
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=${WALL_PRETRAIN}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --output=${LOG_DIR}/B-binary-pretrain-%j.out
#SBATCH --error=${LOG_DIR}/B-binary-pretrain-%j.err
#SBATCH --account=aip-boyuwang
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ccao87@uwo.ca
#SBATCH --dependency=afterok:${JOB_A}

${JOB_PREAMBLE}
source "${VENV}/bin/activate"
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
cd "${REPO}"

# Reuse iTrans checkpoint from Gaussian run — skip redundant synthetic pretrain
echo "[B] Copying iTrans artifacts from Gaussian checkpoint dir..."
cp -v "${GAUSS_CKPT}/pretrained_itransformer.pt" "${BINARY_CKPT}/pretrained_itransformer.pt"
[ -f "${GAUSS_CKPT}/itrans_hp.json" ] && \
    cp -v "${GAUSS_CKPT}/itrans_hp.json" "${BINARY_CKPT}/itrans_hp.json"

${PY_BASE} \
    --mode pretrain \
    --binary-diffusion \
    --checkpoint-dir ${BINARY_CKPT} \
    --results-dir    ${BINARY_RESULTS} \
    ${PY_COMMON}

echo "[B] Binary pretrain complete: \$(date)"
SCRIPT_EOF

JOB_B=$(submit_job "$TMP_B")
rm -f "$TMP_B"
echo "  -> Job B ID: ${JOB_B}"


# ==========================================================================
# JOB C — Gaussian finetune ETTh2 (depends on A)
# ==========================================================================
echo "Submitting Job C: Gaussian finetune ETTh2 (depends on A=${JOB_A})..."

TMP_C=$(mktemp /tmp/slurm_jobC_XXXXXX.sh)
cat > "$TMP_C" << SCRIPT_EOF
#!/bin/bash
#SBATCH --job-name=etth2-gauss-finetune${SUFFIX}
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=${WALL_FINETUNE}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --output=${LOG_DIR}/C-gauss-finetune-%j.out
#SBATCH --error=${LOG_DIR}/C-gauss-finetune-%j.err
#SBATCH --account=aip-boyuwang
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ccao87@uwo.ca
#SBATCH --dependency=afterok:${JOB_A}

${JOB_PREAMBLE}
source "${VENV}/bin/activate"
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
cd "${REPO}"

${PY_BASE} \
    --mode finetune \
    --dataset ETTh2 \
    --checkpoint-dir ${GAUSS_CKPT} \
    --results-dir    ${GAUSS_RESULTS} \
    ${PY_COMMON}

echo "[C] Gaussian ETTh2 finetune complete: \$(date)"
SCRIPT_EOF

JOB_C=$(submit_job "$TMP_C")
rm -f "$TMP_C"
echo "  -> Job C ID: ${JOB_C}"


# ==========================================================================
# JOB D — Binary finetune ETTh2 (depends on B)
# ==========================================================================
echo "Submitting Job D: Binary finetune ETTh2 (depends on B=${JOB_B})..."

TMP_D=$(mktemp /tmp/slurm_jobD_XXXXXX.sh)
cat > "$TMP_D" << SCRIPT_EOF
#!/bin/bash
#SBATCH --job-name=etth2-binary-finetune${SUFFIX}
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=${WALL_FINETUNE}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --output=${LOG_DIR}/D-binary-finetune-%j.out
#SBATCH --error=${LOG_DIR}/D-binary-finetune-%j.err
#SBATCH --account=aip-boyuwang
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ccao87@uwo.ca
#SBATCH --dependency=afterok:${JOB_B}

${JOB_PREAMBLE}
source "${VENV}/bin/activate"
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
cd "${REPO}"

${PY_BASE} \
    --mode finetune \
    --dataset ETTh2 \
    --binary-diffusion \
    --checkpoint-dir ${BINARY_CKPT} \
    --results-dir    ${BINARY_RESULTS} \
    ${PY_COMMON}

echo "[D] Binary ETTh2 finetune complete: \$(date)"
SCRIPT_EOF

JOB_D=$(submit_job "$TMP_D")
rm -f "$TMP_D"
echo "  -> Job D ID: ${JOB_D}"


# ==========================================================================
# Summary
# ==========================================================================
echo ""
echo "=================================================================="
echo "  All jobs submitted — dependency chain:"
echo ""
echo "  A: ${JOB_A}  Gaussian pretrain"
echo "  B: ${JOB_B}  Binary  pretrain   [afterok:${JOB_A}]"
echo "  C: ${JOB_C}  Gaussian finetune  [afterok:${JOB_A}]"
echo "  D: ${JOB_D}  Binary  finetune   [afterok:${JOB_B}]"
echo ""
echo "  Logs:             ${LOG_DIR}/"
echo "  Gaussian ckpts:   ${GAUSS_CKPT}/"
echo "  Binary   ckpts:   ${BINARY_CKPT}/"
echo "  Gaussian results: ${GAUSS_RESULTS}/"
echo "  Binary   results: ${BINARY_RESULTS}/"
echo ""
echo "  Monitor:   squeue -u \$USER"
echo "  Cancel all: scancel ${JOB_A} ${JOB_B} ${JOB_C} ${JOB_D}"
echo "=================================================================="
