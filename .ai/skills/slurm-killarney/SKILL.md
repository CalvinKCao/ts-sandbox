---
name: slurm-killarney
description: Boilerplate Slurm submission scripts for Killarney (Alliance Canada). Use when writing a new Slurm job script — provides the self-resubmitting single-job template and the multi-job dependency-chain template. Account, user, email, and module stack are pre-filled for ccao87/aip-boyuwang.
---

# Slurm Script Templates — Killarney

Two patterns used in this repo. Pick one and fill in the blanks marked `# TODO`.

---

## Pattern 1 — Self-resubmitting single job

The script detects whether it's running on the login node or inside a Slurm job.
From the login node it calls `sbatch` on itself; inside the job it does the real work.

```bash
#!/bin/bash
# =============================================================================
# TODO: one-line description of what this job does
#
# USAGE (from repo root on login node):
#   ./TODO_script_name.sh --smoke-test   # L40S, 15 min
#   ./TODO_script_name.sh                # H100 full run
#   ./TODO_script_name.sh --dataset ETTh1
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Login node: submit ourselves with the right resources
# ---------------------------------------------------------------------------
if [ -z "$SLURM_JOB_ID" ]; then
    IS_SMOKE=0
    for arg in "$@"; do [ "$arg" = "--smoke-test" ] && IS_SMOKE=1; done

    if [ "$IS_SMOKE" -eq 1 ]; then
        echo "Submitting SMOKE TEST (L40S, 8 GB, 15 min)..."
        sbatch \
            --job-name=TODO-job-smoke \
            --account=aip-boyuwang \
            --time=0:15:00 \
            --nodes=1 \
            --gres=gpu:l40s:1 \
            --cpus-per-task=2 \
            --mem=8G \
            --output=TODO-job-smoke-%j.out \
            --error=TODO-job-smoke-%j.err \
            --mail-type=END,FAIL \
            --mail-user=ccao87@uwo.ca \
            "$SCRIPT_DIR/TODO_script_name.sh" "$@"
    else
        echo "Submitting FULL RUN (H100, 60 GB, 1.5 days)..."
        sbatch \
            --job-name=TODO-job \
            --account=aip-boyuwang \
            --partition=gpubase_h100_b4 \
            --time=1-12:00:00 \
            --nodes=1 \
            --gpus-per-node=h100:1 \
            --cpus-per-task=6 \
            --mem=60G \
            --output=TODO-job-%j.out \
            --error=TODO-job-%j.err \
            --mail-type=BEGIN,END,FAIL \
            --mail-user=ccao87@uwo.ca \
            "$SCRIPT_DIR/TODO_script_name.sh" "$@"
    fi
    exit 0
fi

# ---------------------------------------------------------------------------
# Inside the job — do the actual work
# ---------------------------------------------------------------------------
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID   Node: $SLURMD_NODENAME"
echo "GPU:    $(nvidia-smi -L 2>/dev/null | head -1 || echo unknown)"
echo "Started: $(date)"
echo "=========================================="

module purge || true   # || true: sticky modules exit non-zero, kills job under set -e
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9

# Repo root: prefer $SCRATCH (no GPU work from /home on Killarney)
if   [ -d "$SCRATCH/TODO-repo-name" ]; then PROJECT_ROOT="$SCRATCH/TODO-repo-name"
elif [ -d "$HOME/TODO-repo-name" ];    then PROJECT_ROOT="$HOME/TODO-repo-name"
else echo "ERROR: repo not found" && exit 1
fi

# Per-user PROJECT storage (venv, checkpoints, results) — not the bare group root
if [ -z "${PROJECT:-}" ]; then
    shopt -s nullglob
    _m=("$HOME"/projects/aip-* "$HOME"/projects/def-*)
    shopt -u nullglob
    [ "${#_m[@]}" -gt 0 ] && PROJECT=$(readlink -f "${_m[0]}")
fi
[ -z "${PROJECT:-}" ] && echo "ERROR: PROJECT not found" && exit 1

STORAGE_ROOT="$PROJECT/$USER/TODO-app-name"
mkdir -p "$STORAGE_ROOT/checkpoints" "$STORAGE_ROOT/results"

# Venv on node-local NVMe — avoids 5-15 min cold-import from Lustre
echo "[setup] Building venv on \$SLURM_TMPDIR ..."
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip -q

# Alliance CA pre-built wheel cache (fast, no network needed)
pip install --no-index torch torchvision numpy pandas scipy scikit-learn tqdm -q 2>/dev/null || \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q && \
    pip install numpy pandas scipy scikit-learn tqdm -q
# Packages absent from the wheel cache
pip install optuna wandb matplotlib einops -q
[ -f "$PROJECT_ROOT/requirements.txt" ] && pip install -r "$PROJECT_ROOT/requirements.txt" -q || true

export WANDB_MODE=offline
export PYTHONUNBUFFERED=1

cd "$PROJECT_ROOT"

# Parse args passed through from the login-node submission
SMOKE_TEST=""; DATASET="TODO-default-dataset"
while [[ $# -gt 0 ]]; do
    case $1 in
        --smoke-test) SMOKE_TEST="--smoke-test"; shift ;;
        --dataset)    DATASET="$2"; shift 2 ;;
        *)            echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# TODO: actual Python command
# ---------------------------------------------------------------------------
python -u -m TODO.module \
    --dataset "$DATASET" \
    --checkpoint-dir "$STORAGE_ROOT/checkpoints" \
    --results-dir    "$STORAGE_ROOT/results" \
    $SMOKE_TEST

echo "========================================"
echo "Job complete: $(date)"
echo "Results: $STORAGE_ROOT/results"
echo "========================================"
```

---

## Pattern 2 — Dependency-chained jobs (submitted from the login node)

Use when you need sequential or fan-out jobs (e.g. pretrain → finetune, or A/B comparison).
The login-node script submits all jobs at once via `sbatch --parsable` + `--dependency`.

```bash
#!/bin/bash
# =============================================================================
# TODO: description — e.g. "Gaussian vs Binary comparison on ETTh2, 4 chained jobs"
#
# Dependency graph:
#   A  TODO: pretrain phase
#   B  TODO: second pretrain or parallel branch   [afterok:A]
#   C  TODO: finetune / eval                      [afterok:A]
#   D  TODO: finetune / eval (other branch)       [afterok:B]
#
# USAGE:
#   ./TODO_chain_script.sh           # full L40S run
#   ./TODO_chain_script.sh --smoke   # L40S smoke test, verifies full chain
# =============================================================================

set -euo pipefail

SMOKE=0
for arg in "$@"; do [ "$arg" = "--smoke" ] && SMOKE=1; done

# Repo root
if   [ -d "${SCRATCH:-}/TODO-repo-name" ]; then REPO="$SCRATCH/TODO-repo-name"
elif [ -d "$HOME/TODO-repo-name" ];         then REPO="$HOME/TODO-repo-name"
else echo "ERROR: repo not found" && exit 1
fi

# Storage root — override with STORE=... ./script.sh to redirect to PROJECT
if [ -z "${STORE:-}" ]; then
    [ -z "${SCRATCH:-}" ] && echo "ERROR: \$SCRATCH not set; export STORE manually" && exit 1
    export STORE="$SCRATCH/TODO-app-name"
fi
LOG_DIR="$STORE/logs"
mkdir -p "$STORE" "$LOG_DIR"
# TODO: add per-branch subdirs, e.g.:
#   export BRANCH_A_CKPT="$STORE/checkpoints_branchA"
#   mkdir -p "$BRANCH_A_CKPT"

echo "Storage: $STORE"
echo "Logs:    $LOG_DIR"

# Dataset symlink (avoids copying GBs)
[ ! -L "$STORE/datasets" ] && [ ! -e "$STORE/datasets" ] && \
    ln -s "$REPO/datasets" "$STORE/datasets"

# Venv path: prefer a pre-built one from PROJECT if it exists
export VENV="$STORE/venv"
for _d in ~/projects/aip-* ~/projects/def-*; do
    [ -d "$_d/$USER/TODO-app-name/venv" ] && export VENV="$_d/$USER/TODO-app-name/venv" && break
done
export REPO

# Resources
if [ "$SMOKE" -eq 1 ]; then
    GPU_ARGS=(--gres=gpu:l40s:1)
    WALL_LONG="0:25:00"; WALL_SHORT="0:25:00"
    MEM="16G"; CPUS=4
    export SMOKE_FLAG="--smoke-test"; SUFFIX="-smoke"
else
    GPU_ARGS=(--gres=gpu:l40s:1)        # TODO: switch to H100 if you need >48 GB VRAM
    WALL_LONG="2-00:00:00"; WALL_SHORT="0-14:00:00"
    MEM="60G"; CPUS=6
    export SMOKE_FLAG=""; SUFFIX=""
fi

# Shared preamble written to STORE (must be on a shared FS, not /tmp)
export PREAMBLE_FILE="$STORE/job_preamble.sh"
cat > "$PREAMBLE_FILE" << 'PREAMBLE'
set -euo pipefail
echo "Job: $SLURM_JOB_NAME  ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi -L 2>/dev/null | head -1 || echo none)"
echo "Started: $(date)"

module purge || true
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9

echo "[setup] Building venv on \$SLURM_TMPDIR ..."
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index torch torchvision numpy pandas scipy scikit-learn tqdm -q 2>/dev/null || \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q && \
    pip install numpy pandas scipy scikit-learn tqdm -q
pip install optuna wandb matplotlib einops -q
[ -f "$REPO/requirements.txt" ] && pip install -r "$REPO/requirements.txt" -q || true

export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
cd "$REPO"
PREAMBLE
export PREAMBLE_FILE

export PY="python -u -m TODO.module"
export PY_COMMON="--n-variates 7 --amp --synthetic-samples 100000 $SMOKE_FLAG"

# ---- Job A ------------------------------------------------------------------
echo "Submitting A: TODO description..."
JOB_A=$(sbatch --parsable \
    --job-name="TODO-A${SUFFIX}" \
    --account=aip-boyuwang \
    --nodes=1 --cpus-per-task="$CPUS" --mem="$MEM" \
    "${GPU_ARGS[@]}" \
    --time="$WALL_LONG" \
    --output="$LOG_DIR/A-TODO-%j.out" \
    --error="$LOG_DIR/A-TODO-%j.err" \
    --mail-type=FAIL --mail-user=ccao87@uwo.ca \
    << 'ENDSCRIPT'
#!/bin/bash
source "$PREAMBLE_FILE"
$PY --mode TODO \
    --checkpoint-dir "$TODO_CKPT" \
    $PY_COMMON
echo "[A] done: $(date)"
ENDSCRIPT
)
echo "  -> A: $JOB_A"

# ---- Job B (depends on A) ---------------------------------------------------
echo "Submitting B: TODO description [afterok:$JOB_A]..."
JOB_B=$(sbatch --parsable \
    --job-name="TODO-B${SUFFIX}" \
    --account=aip-boyuwang \
    --nodes=1 --cpus-per-task="$CPUS" --mem="$MEM" \
    "${GPU_ARGS[@]}" \
    --time="$WALL_LONG" \
    --dependency="afterok:$JOB_A" \
    --output="$LOG_DIR/B-TODO-%j.out" \
    --error="$LOG_DIR/B-TODO-%j.err" \
    --mail-type=FAIL --mail-user=ccao87@uwo.ca \
    << 'ENDSCRIPT'
#!/bin/bash
source "$PREAMBLE_FILE"
# TODO: optionally copy artifacts from A, e.g.:
#   cp "$TODO_A_CKPT/some_checkpoint.pt" "$TODO_B_CKPT/"
$PY --mode TODO \
    --checkpoint-dir "$TODO_B_CKPT" \
    $PY_COMMON
echo "[B] done: $(date)"
ENDSCRIPT
)
echo "  -> B: $JOB_B"

# ---- Summary ----------------------------------------------------------------
echo ""
echo "=================================================================="
echo "  A $JOB_A  TODO-A"
echo "  B $JOB_B  TODO-B   [afterok:$JOB_A]"
echo ""
echo "  Logs:       $LOG_DIR/"
echo "  Monitor:    squeue -u \$USER"
echo "  Cancel all: scancel $JOB_A $JOB_B"
echo "=================================================================="
```

---

## Quick reference

| Thing | Value |
|---|---|
| Account | `aip-boyuwang` |
| User / email | `ccao87` / `ccao87@uwo.ca` |
| Default GPU | L40S (`--gres=gpu:l40s:1`) — shorter queue |
| Heavy GPU | H100 (`--partition=gpubase_h100_b4 --gpus-per-node=h100:1`) |
| Smoke wall time | `0:15:00` (L40S) — request ≥20 min to cover pip install |
| Full wall (L40S) | `2-00:00:00` pretrain, `0-14:00:00` finetune |
| Full wall (H100) | `1-12:00:00` |
| Modules | `StdEnv/2023 python/3.11 cuda/12.2 cudnn/8.9` |
| Venv | Always rebuild on `$SLURM_TMPDIR` — never activate from Lustre |
| `module purge` | Always `module purge || true` under `set -e` |
| PREAMBLE_FILE | Write to `$STORE` (shared FS), not `/tmp` (not visible on compute) |
