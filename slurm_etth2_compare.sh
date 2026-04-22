#!/bin/bash
# =============================================================================
# ETTh2: Gaussian vs Binary diffusion — Killarney, job-chained
#
# RUN ON THE LOGIN NODE (not inside sbatch):
#   ./slurm_etth2_compare.sh [--smoke]
#   bash slurm_etth2_compare.sh [--smoke]
# Do NOT run:  sbatch slurm_etth2_compare.sh
#   (that would submit *this* file as one job; this script is a wrapper that
#   calls sbatch four times. If your site requires it, the #SBATCH lines below
#   make accidental sbatch parse — still prefer bash on the login node.)
#
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=etth2-chain-submit
#
# Dependency graph:
#   A  iTrans HP + pretrain + Gaussian diff HP + pretrain
#   B  (copy iTrans from A) → Binary diff HP + pretrain        [afterok:A]
#   C  Gaussian finetune ETTh2                                  [afterok:A]
#   D  Binary  finetune ETTh2                                   [afterok:B]
#
# USAGE (from ts-sandbox repo root on the login node):
#   ./slurm_etth2_compare.sh           # full run (L40S in script)
#   ./slurm_etth2_compare.sh --smoke   # smoke test — verifies full chain
#
# WANDB: jobs pass --wandb. API key: repo-root wandb_api_key.txt (see wandb_api_key.example.txt),
#   or wandb login / export WANDB_API_KEY (Slurm forwards with --export=ALL).
# Runs use online mode by default; metrics land under $STORE/wandb/.
#
# Resume after pretrain timeout: re-submit the same script. If
#   $GAUSS_CKPT/pretrained_diffusion_last.pt exists, diffusion pretrain continues
#   automatically (same checkpoint dir). Finished runs delete that file.
#
# HOW TO SMOKE TEST (pick one):
#   1) Pip + imports only (fastest — catches bad PyPI names before burning GPU hours):
#        salloc ...   # short GPU alloc, see scripts/killarney_smoke_pip.sh header
#        bash scripts/killarney_smoke_pip.sh
#   2) Full Slurm chain miniature — this script with --smoke:
#        ./slurm_etth2_compare.sh --smoke
#      Submits A–D with short walls; passes --smoke-test (1 epoch, tiny samples, 1 HP trial).
#      Watch:  squeue -u $USER
#      Logs:   tail -f $STORE/logs/A-gauss-pretrain-<JOB_ID>.out
#   3) Local (WSL / laptop):  pytest models/diffusion_tsf/tests/ -q
#      (no Alliance wheel cache — does not replace 1 or 2.)
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

# ---- Storage ----------------------------------------------------------------
# Default: $SCRATCH (always set on Killarney, large, fast — right place for
# active training).  Set STORE before running to override, e.g. to keep
# results in PROJECT for longer-term storage:
#   STORE=$PROJECT/$USER/diffusion-tsf-etth2 ./slurm_etth2_compare.sh
if [ -z "${STORE:-}" ]; then
    if [ -z "${SCRATCH:-}" ]; then
        echo "ERROR: \$SCRATCH is not set. Set STORE manually and re-run."
        exit 1
    fi
    export STORE="$SCRATCH/diffusion-tsf-etth2"
else
    export STORE
fi
export GAUSS_CKPT="$STORE/checkpoints_gauss"
export GAUSS_RESULTS="$STORE/results_gauss"
export BINARY_CKPT="$STORE/checkpoints_binary"
export BINARY_RESULTS="$STORE/results_binary"
LOG_DIR="$STORE/logs"

mkdir -p "$GAUSS_CKPT" "$GAUSS_RESULTS" \
         "$BINARY_CKPT" "$BINARY_RESULTS" \
         "$LOG_DIR"

# Print resolved paths up front so you always know where to look
echo "=================================================================="
echo "  Storage root:  $STORE"
echo "  Logs:          $LOG_DIR"
echo "  (set STORE=\$PROJECT/\$USER/diffusion-tsf-etth2 to access later)"
echo "=================================================================="

# Keep datasets accessible from STORE without copying GBs
# Use -L to check for the symlink itself (not its target) to avoid broken-symlink traps
[ ! -L "$STORE/datasets" ] && [ ! -e "$STORE/datasets" ] && \
    ln -s "$REPO/datasets" "$STORE/datasets"

# ---- Venv path (exported so jobs can see it) --------------------------------
# $PROJECT is not set on Killarney login nodes — walk ~/projects/ safely without
# relying on a glob that exits non-zero when nothing matches.
export VENV="$STORE/venv"   # default; overridden below if a pre-built venv exists
for _d in ~/projects/aip-* ~/projects/def-*; do
    [ -d "$_d/$USER/diffusion-tsf/venv" ] && export VENV="$_d/$USER/diffusion-tsf/venv" && break
done

# ---- Repo path (exported) ---------------------------------------------------
export REPO

# ---- Resources + flags ------------------------------------------------------
if [ "$SMOKE" -eq 1 ]; then
    # L40S for smoke: much shorter queue, plenty for a 1-epoch sanity check.
    # Request >=20 min even for smoke — pip install from wheel cache takes 3-5 min.
    GPU_ARGS=(--gres=gpu:l40s:1)
    WALL_PRETRAIN="0:25:00"
    WALL_FINETUNE="0:25:00"
    MEM="16G"; CPUS=4
    export SMOKE_FLAG="--smoke-test"
    SUFFIX="-smoke"
else
    # Full run on L40S (512-LB / 96-FC + AMP + bs=128).
    # Wall-time budget (L40S, ~4× slower than H100):
    #   Job A/B (HP searches + full synthetic pretrain): ~44 h -> request 2 days
    #   Job C/D (ETTh2 finetune + eval):                 ~10 h -> request 14 h
    GPU_ARGS=(--gres=gpu:l40s:1)
    WALL_PRETRAIN="2-00:00:00"
    WALL_FINETUNE="0-14:00:00"
    MEM="60G"; CPUS=6
    export SMOKE_FLAG=""
    SUFFIX=""
fi

# Common Python flags (exported so job bodies can use $SMOKE_FLAG etc.)
export PY="python -u -m models.diffusion_tsf.train_multivariate_pipeline"
export PY_COMMON="--n-variates 7 --amp --synthetic-samples 60000 --itransformer-trials 20 --wandb --wandb-project diffusion-tsf $SMOKE_FLAG"

# ---- Shared job body: module load + venv activate + cd ----------------------
# Written as a quoted heredoc into a temp file so each job can source it.
# (Avoids duplicating 15 lines × 4 jobs while keeping expansion safe.)
# Written to $STORE (shared filesystem) so compute nodes can source it —
# /tmp on the login node is NOT visible from compute nodes.
PREAMBLE_FILE="$STORE/job_preamble.sh"
cat > "$PREAMBLE_FILE" << 'PREAMBLE'
set -euo pipefail
echo "======================================================="
echo "  Job: $SLURM_JOB_NAME   ID: $SLURM_JOB_ID"
echo "  Node: $SLURMD_NODENAME"
echo "  GPU:  $(nvidia-smi -L 2>/dev/null | head -1 || echo none)"
echo "  Started: $(date)"
echo "======================================================="

# || true required: sticky modules (CCconfig, gentoo, compiler stack) refuse to
# unload and module purge exits non-zero, killing the job under set -e.
module purge || true
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9

# Rebuild venv on node-local NVMe each job — avoids catastrophically slow imports
# from Lustre (/scratch, /project). `import torch` alone can take 5-15 min on
# a cold Lustre node; $SLURM_TMPDIR reads take seconds.
echo "[setup] Building venv on \$SLURM_TMPDIR ..."
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip -q

# Alliance CA wheel cache first; PyPI fallback for torch stack.
# (Avoid:  cmd1 || cmd2 && cmd3  — if cmd1 succeeds, cmd3 still runs; if cmd2 fails, cmd3 is skipped.)
if pip install --no-index torch torchvision numpy pandas scipy scikit-learn tqdm -q 2>/dev/null; then
    :
else
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q
    pip install numpy pandas scipy scikit-learn tqdm -q
fi

# PyPI (hyphenated PyPI name — NOT reformer_pytorch). Pin matches models/iTransformer/requirements.txt
pip install "wandb>=0.25.0" optuna matplotlib einops -q
pip install "reformer-pytorch==1.4.4" -q

[ -f "$REPO/requirements.txt" ] && pip install -r "$REPO/requirements.txt" -q || true

echo "[setup] Venv ready: $(which python)"

# Persist run metadata on scratch; syncs to the cloud when WANDB_API_KEY is set
# and mode is online (default). For air-gapped runs: export WANDB_MODE=offline
# then `wandb sync $WANDB_DIR/offline-run-*` from a machine with a key.
export WANDB_DIR="${STORE}/wandb"
mkdir -p "$WANDB_DIR"
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
echo "  Quick check: squeue -j $JOB_A,$JOB_B,$JOB_C,$JOB_D -o '%.18i %.28j %.10T %.20R'"
echo "  Reasons:     sacct -j $JOB_A,$JOB_B,$JOB_C,$JOB_D -X --format=JobID,JobName,State,ExitCode,Reason"
echo "  Cancel all:  scancel $JOB_A $JOB_B $JOB_C $JOB_D"
echo "=================================================================="
