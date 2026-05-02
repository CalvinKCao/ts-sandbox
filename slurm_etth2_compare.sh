#!/bin/bash
# =============================================================================
# ETTh2: Gaussian diffusion epoch sweep — Killarney, job-chained
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
#   A  iTrans HP + pretrain + Gaussian diffusion HP + pretrain
#      while exporting best-so-far diffusion checkpoints at epochs 10/20/40
#   B10/B20/B40  copy matching exported checkpoint → finetune/eval ETTh2 [afterok:A]
#
# USAGE (from ts-sandbox repo root on the login node):
#   ./slurm_etth2_compare.sh           # full run (L40S)
#   ./slurm_etth2_compare.sh --smoke   # smoke test — verifies CPU/Python path locally; Slurm chain exports epoch 1
#
# WANDB: jobs pass --wandb and expect WANDB_API_KEY in the environment
# (Slurm forwards with --export=ALL).
# Runs use online mode by default; wandb cache under ./results/logs/<stem>_wandb/.
#
# Resume after pretrain timeout: re-submit the same script. If
#   results/ckpts/<job-A-stem>/pretrained_diffusion_last.pt exists (see
#   results/logs/etth2-last-gauss-ckpt.txt), pretrain continues in that dir.
#
# HOW TO SMOKE TEST (pick one):
#   1) Pip + imports on a short GPU alloc: fresh venv on the node, pip install
#      project deps, import torch (and friends) — catches bad wheels before long jobs.
#   2) Full Slurm chain miniature — this script with --smoke:
#        ./slurm_etth2_compare.sh --smoke
#      Submits A + one epoch-1 downstream job with short walls; passes --smoke-test.
#      Watch:  squeue -u $USER
#      Logs:   tail -f results/logs/<MM-DD>-<jobid4>-gauss-pretrain.log
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

# ---- Artifacts (Alliance layout) --------------------------------------------
# All logs, checkpoints, and run outputs live under ./results/ relative to the
# directory from which you invoke this wrapper (same as SLURM_SUBMIT_DIR for
# submitted jobs). See .ai/skills/alliancecan (results/{logs,ckpts,datasets}).

SBATCH_JOBS="$REPO/results/logs/etth2-chain-job-scripts"
mkdir -p "$SBATCH_JOBS" "$REPO/results/logs" "$REPO/results/ckpts" "$REPO/results/datasets"

echo "=================================================================="
echo "  Repo / submit dir:  $REPO"
echo "  Artifacts:          $REPO/results/{logs,ckpts,datasets}/"
echo "=================================================================="

# ---- Venv path (informational; batch preamble builds venv on SLURM_TMPDIR) ---
export VENV="${REPO}/.venv"
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
    MILESTONES=(1)
else
    # Full pretrain stays on L40S to avoid the H100 queue. A 2-day L40S job
    # timed out during diffusion epoch 40/200 after HP tuning, so request more
    # wall and rely on per-epoch resume snapshots if another chunk is needed.
    GPU_ARGS=(--gres=gpu:l40s:1)
    WALL_PRETRAIN="3-00:00:00"
    WALL_FINETUNE="0-14:00:00"
    MEM="60G"; CPUS=6
    export SMOKE_FLAG=""
    SUFFIX=""
    MILESTONES=(10 20 40)
fi

MILESTONE_CSV=$(IFS=,; echo "${MILESTONES[*]}")

# Common Python flags (exported so job bodies can use $SMOKE_FLAG etc.)
export PY="python -u -m models.diffusion_tsf.train_multivariate_pipeline"
export PY_COMMON="--n-variates 7 --amp --synthetic-samples 60000 --itransformer-trials 20 --diffusion-export-epochs ${MILESTONE_CSV} --wandb --wandb-project diffusion-tsf $SMOKE_FLAG"

# ---- Shared job body: module load + venv activate + cd ----------------------
# Written as a quoted heredoc into a temp file so each job can source it.
# (Avoids duplicating 15 lines × 4 jobs while keeping expansion safe.)
# Written under results/logs so compute nodes see the same path as the repo checkout.
PREAMBLE_FILE="$REPO/results/logs/etth2-chain-preamble.sh"
# Unquoted heredoc: bake ${REPO} at submit time. Use \$ for compute-node expansion.
cat > "$PREAMBLE_FILE" <<PREAMBLE
set -euo pipefail
: "\${ALLIANCE_RUN_SLUG:?set ALLIANCE_RUN_SLUG before sourcing preamble}"
cd "\${SLURM_SUBMIT_DIR}"
mkdir -p results/logs results/ckpts results/datasets
ALLIANCE_RUN_STEM="\$(date +%m-%d)-\${SLURM_JOB_ID: -4}-\${ALLIANCE_RUN_SLUG}"
export ALLIANCE_RUN_STEM
export ALLIANCE_JOB_LOG="\${SLURM_SUBMIT_DIR}/results/logs/\${ALLIANCE_RUN_STEM}.log"
touch "\${ALLIANCE_JOB_LOG}"
exec >>"\${ALLIANCE_JOB_LOG}" 2>&1

echo "======================================================="
echo "  Job: \$SLURM_JOB_NAME   ID: \$SLURM_JOB_ID"
echo "  Node: \$SLURMD_NODENAME"
echo "  GPU:  \$(nvidia-smi -L 2>/dev/null | head -1 || echo none)"
echo "  Started: \$(date)"
echo "  Log: \${ALLIANCE_JOB_LOG}"
echo "======================================================="

# || true required: sticky modules (CCconfig, gentoo, compiler stack) refuse to
# unload and module purge exits non-zero, killing the job under set -e.
module purge || true
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9

# Rebuild venv on node-local NVMe each job — avoids catastrophically slow imports
# from Lustre (/scratch, /project). (import torch) alone can take 5-15 min on
# a cold Lustre node; \$SLURM_TMPDIR reads take seconds.
echo "[setup] Building venv on \$SLURM_TMPDIR ..."
virtualenv --no-download "\$SLURM_TMPDIR/env"
source "\$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip -q

# Alliance CA wheel cache first; PyPI fallback for torch stack.
# (Avoid:  cmd1 || cmd2 && cmd3  — if cmd1 succeeds, cmd3 still runs; if cmd2 fails, cmd3 is skipped.)
if pip install --no-index torch torchvision numpy pandas scipy scikit-learn tqdm -q 2>/dev/null; then
    :
else
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q
    pip install numpy pandas scipy scikit-learn tqdm -q
fi

# wandb MUST come from the Alliance wheelhouse (--no-index). Recent wandb on PyPI
# ships as sdist that builds wandb-core in Go at metadata-generation time, and compute
# nodes have no go binary. Observed failure (job 3249152):
#   "Did not find the 'go' binary" -> metadata-generation-failed -> set -e kills the job
#   with an almost-empty log that only shows "[setup] Building venv on SLURM_TMPDIR".
# Ref: wiki_docs/Weights_&_Biases_(wandb).md — Alliance docs say: pip install --no-index wandb.
pip install --no-index wandb -q

# Try Alliance wheelhouse for the rest; PyPI fallback individually so one missing wheel
# doesn't abort the whole install (optuna/einops sometimes lag in the wheelhouse).
for pkg in optuna matplotlib einops; do
    pip install --no-index "\$pkg" -q 2>/dev/null || pip install "\$pkg" -q
done

# reformer-pytorch — pure Python, not in the wheel cache. Pin matches models/iTransformer/requirements.txt
# (PyPI name is hyphenated, NOT reformer_pytorch).
pip install "reformer-pytorch==1.4.4" -q

[ -f "${REPO}/requirements.txt" ] && pip install -r "${REPO}/requirements.txt" -q || true

echo "[setup] Venv ready: \$(which python)"

# Persist run metadata on scratch; syncs to the cloud when WANDB_API_KEY is set
# and mode is online (default). For air-gapped runs: export WANDB_MODE=offline
# then run: wandb sync \$WANDB_DIR/offline-run-* from a machine with a key.
export WANDB_DIR="\${SLURM_SUBMIT_DIR}/results/logs/\${ALLIANCE_RUN_STEM}_wandb"
mkdir -p "\$WANDB_DIR"
if [ -z "\${WANDB_API_KEY:-}" ]; then
    echo "[wandb] ERROR: WANDB_API_KEY is not set."
    echo "[wandb] Export WANDB_API_KEY from https://wandb.ai/authorize and re-submit."
    exit 2
fi
echo "[wandb] Using WANDB_API_KEY from environment."
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[info] python: \$(which python)"
cd "${REPO}"
echo "[info] cwd: \$(pwd)"
echo ""

wandb_upload_job_logs() {
    local checkpoint_dir="\$1"
    shift
    local run_id_file="\${checkpoint_dir}/wandb_run_id.txt"
    if [ ! -f "\$run_id_file" ]; then
        echo "[wandb] WARN: no run id file at \$run_id_file; skipping log upload."
        return 0
    fi
    local run_id
    run_id="\$(tr -d '[:space:]' < "\$run_id_file")"
    [ -z "\$run_id" ] && echo "[wandb] WARN: empty run id in \$run_id_file; skipping." && return 0

    local files=()
    local f
    for f in "\$@"; do
        [ -f "\$f" ] && files+=("\$f")
    done
    [ "\${#files[@]}" -eq 0 ] && echo "[wandb] WARN: no log files found to upload." && return 0

    python - "\$run_id" "\${files[@]}" <<'PY' || true
import os
import sys
import wandb

run_id = sys.argv[1]
files = sys.argv[2:]
project = os.environ.get("WANDB_PROJECT", "diffusion-tsf")
job_id = os.environ.get("SLURM_JOB_ID", "unknown")
job_name = os.environ.get("SLURM_JOB_NAME", "unknown")

run = wandb.init(project=project, id=run_id, resume="must", reinit=True)
artifact = wandb.Artifact(f"slurm-job-logs-{job_id}", type="logs")
artifact.metadata.update({"slurm_job_id": job_id, "slurm_job_name": job_name})
for path in files:
    if os.path.isfile(path):
        artifact.add_file(path)
run.log_artifact(artifact)
run.finish()
print(f"[wandb] Uploaded {len(files)} log file(s) for job {job_id}.")
PY
}
PREAMBLE
# Will be sourced by each job via:  source "$PREAMBLE_FILE"
export PREAMBLE_FILE

# ============================================================================
# Write per-job scripts (paths baked at submit time), then sbatch FILE
# ============================================================================

cat > "$SBATCH_JOBS/job_A.sh" <<JOB_A
#!/bin/bash
export ALLIANCE_RUN_SLUG=gauss-pretrain
source "${PREAMBLE_FILE}"

GAUSS_CKPT="\${SLURM_SUBMIT_DIR}/results/ckpts/\${ALLIANCE_RUN_STEM}"
GAUSS_RESULTS="\${SLURM_SUBMIT_DIR}/results/datasets/\${ALLIANCE_RUN_STEM}"
mkdir -p "\$GAUSS_CKPT" "\$GAUSS_RESULTS"

${PY} --mode pretrain \\
    --checkpoint-dir "\$GAUSS_CKPT" \\
    --results-dir    "\$GAUSS_RESULTS" \\
    ${PY_COMMON}

echo "\$GAUSS_CKPT" > "\${SLURM_SUBMIT_DIR}/results/logs/etth2-last-gauss-ckpt.txt"
echo "[A] Gaussian pretrain done: \$(date)"
wandb_upload_job_logs "\$GAUSS_CKPT" "\${ALLIANCE_JOB_LOG}"
JOB_A
chmod +x "$SBATCH_JOBS/job_A.sh"

cat > "$SBATCH_JOBS/job_B.sh" <<JOB_B
#!/bin/bash
export ALLIANCE_RUN_SLUG=binary-pretrain
source "${PREAMBLE_FILE}"

GAUSS_CKPT="\$(cat "\${SLURM_SUBMIT_DIR}/results/logs/etth2-last-gauss-ckpt.txt")"
BINARY_CKPT="\${SLURM_SUBMIT_DIR}/results/ckpts/\${ALLIANCE_RUN_STEM}"
BINARY_RESULTS="\${SLURM_SUBMIT_DIR}/results/datasets/\${ALLIANCE_RUN_STEM}"
mkdir -p "\$BINARY_CKPT" "\$BINARY_RESULTS"

echo "[B] Copying iTrans artifacts from Gaussian checkpoint dir..."
if [ -f "\$BINARY_CKPT/.smoke_test" ]; then
    echo "[B] Removing stale binary smoke-test artifacts before copying iTrans..."
    rm -f "\$BINARY_CKPT/pretrained_itransformer.pt" \\
          "\$BINARY_CKPT/pretrained_diffusion.pt" \\
          "\$BINARY_CKPT/pretrained_diffusion_last.pt" \\
          "\$BINARY_CKPT/itrans_hp.json" \\
          "\$BINARY_CKPT/diff_hp.json" \\
          "\$BINARY_CKPT/.smoke_test"
fi
cp -v "\$GAUSS_CKPT/pretrained_itransformer.pt" "\$BINARY_CKPT/pretrained_itransformer.pt"
[ -f "\$GAUSS_CKPT/itrans_hp.json" ] && \\
    cp -v "\$GAUSS_CKPT/itrans_hp.json" "\$BINARY_CKPT/itrans_hp.json"

echo "[B] Running binary diffusion HP + pretrain..."
${PY} --mode pretrain \\
    --binary-diffusion \\
    --checkpoint-dir "\$BINARY_CKPT" \\
    --results-dir    "\$BINARY_RESULTS" \\
    ${PY_COMMON}

echo "\$BINARY_CKPT" > "\${SLURM_SUBMIT_DIR}/results/logs/etth2-last-binary-ckpt.txt"
echo "[B] Binary pretrain done: \$(date)"
JOB_B
chmod +x "$SBATCH_JOBS/job_B.sh"

cat > "$SBATCH_JOBS/job_C.sh" <<JOB_C
#!/bin/bash
export ALLIANCE_RUN_SLUG=gauss-finetune-etth2
source "${PREAMBLE_FILE}"

GAUSS_CKPT="\$(cat "\${SLURM_SUBMIT_DIR}/results/logs/etth2-last-gauss-ckpt.txt")"
GAUSS_RESULTS="\${SLURM_SUBMIT_DIR}/results/datasets/\${ALLIANCE_RUN_STEM}"
mkdir -p "\$GAUSS_RESULTS"

${PY} --mode finetune \\
    --dataset ETTh2 \\
    --checkpoint-dir "\$GAUSS_CKPT" \\
    --results-dir    "\$GAUSS_RESULTS" \\
    ${PY_COMMON}

echo "[C] Gaussian finetune done: \$(date)"
JOB_C
chmod +x "$SBATCH_JOBS/job_C.sh"

cat > "$SBATCH_JOBS/job_D.sh" <<JOB_D
#!/bin/bash
export ALLIANCE_RUN_SLUG=binary-finetune-etth2
source "${PREAMBLE_FILE}"

BINARY_CKPT="\$(cat "\${SLURM_SUBMIT_DIR}/results/logs/etth2-last-binary-ckpt.txt")"
BINARY_RESULTS="\${SLURM_SUBMIT_DIR}/results/datasets/\${ALLIANCE_RUN_STEM}"
mkdir -p "\$BINARY_RESULTS"

${PY} --mode finetune \\
    --dataset ETTh2 \\
    --binary-diffusion \\
    --checkpoint-dir "\$BINARY_CKPT" \\
    --results-dir    "\$BINARY_RESULTS" \\
    ${PY_COMMON}

echo "[D] Binary finetune done: \$(date)"
JOB_D
chmod +x "$SBATCH_JOBS/job_D.sh"

echo "  Batch scripts: $SBATCH_JOBS/job_A.sh plus generated milestone jobs"

# ============================================================================
# JOB A — Gaussian pretrain
# ============================================================================

echo "Submitting A: Gaussian pretrain..."

JOB_A=$(sbatch --parsable --export=ALL \
    --job-name="etth2-gauss-pretrain${SUFFIX}" \
    --account=aip-boyuwang \
    --nodes=1 --cpus-per-task="$CPUS" --mem="$MEM" \
    "${GPU_ARGS[@]}" \
    --time="$WALL_PRETRAIN" \
    --chdir="${REPO}" \
    --output=/dev/null \
    --error=/dev/null \
    --mail-type=FAIL --mail-user=ccao87@uwo.ca \
    "$SBATCH_JOBS/job_A.sh")
echo "  -> A: $JOB_A"

echo ""
echo "Submitting downstream ETTh2 finetune/eval jobs for exported Gaussian checkpoints..."
MILESTONE_JOB_IDS=()

for EPOCH in "${MILESTONES[@]}"; do
    cat > "$SBATCH_JOBS/job_gauss_epoch${EPOCH}.sh" <<JOB_EPOCH
#!/bin/bash
export ALLIANCE_RUN_SLUG=gauss-epoch${EPOCH}-finetune
source "${PREAMBLE_FILE}"

GAUSS_SRC="\$(cat "\${SLURM_SUBMIT_DIR}/results/logs/etth2-last-gauss-ckpt.txt")"
EPOCH_CKPT="\${SLURM_SUBMIT_DIR}/results/ckpts/\${ALLIANCE_RUN_STEM}"
EPOCH_RESULTS="\${SLURM_SUBMIT_DIR}/results/datasets/\${ALLIANCE_RUN_STEM}"
mkdir -p "\$EPOCH_CKPT" "\$EPOCH_RESULTS"

echo "[epoch ${EPOCH}] Preparing checkpoint dir: \$EPOCH_CKPT"
cp -v "\$GAUSS_SRC/pretrained_itransformer.pt" "\$EPOCH_CKPT/pretrained_itransformer.pt"
[ -f "\$GAUSS_SRC/itrans_hp.json" ] && cp -v "\$GAUSS_SRC/itrans_hp.json" "\$EPOCH_CKPT/itrans_hp.json"
[ -f "\$GAUSS_SRC/diff_hp.json" ] && cp -v "\$GAUSS_SRC/diff_hp.json" "\$EPOCH_CKPT/diff_hp.json"
cp -v "\$GAUSS_SRC/pretrained_diffusion_best_epoch${EPOCH}.pt" "\$EPOCH_CKPT/pretrained_diffusion.pt"

${PY} --mode finetune \
    --dataset ETTh2 \
    --checkpoint-dir "\$EPOCH_CKPT" \
    --results-dir    "\$EPOCH_RESULTS" \
    ${PY_COMMON}

echo "[epoch ${EPOCH}] ETTh2 finetune/eval done: \$(date)"
wandb_upload_job_logs "\$EPOCH_CKPT" "\${ALLIANCE_JOB_LOG}"
JOB_EPOCH
    chmod +x "$SBATCH_JOBS/job_gauss_epoch${EPOCH}.sh"

    JOB_E=$(sbatch --parsable --export=ALL \
        --job-name="etth2-gauss-ep${EPOCH}${SUFFIX}" \
        --account=aip-boyuwang \
        --nodes=1 --cpus-per-task="$CPUS" --mem="$MEM" \
        "${GPU_ARGS[@]}" \
        --time="$WALL_FINETUNE" \
        --dependency="afterok:$JOB_A" \
        --chdir="${REPO}" \
        --output=/dev/null \
        --error=/dev/null \
        --mail-type=FAIL --mail-user=ccao87@uwo.ca \
        "$SBATCH_JOBS/job_gauss_epoch${EPOCH}.sh")
    MILESTONE_JOB_IDS+=("$JOB_E")
    echo "  -> epoch ${EPOCH}: $JOB_E"
done

echo ""
echo "=================================================================="
echo "  Jobs submitted:"
echo ""
echo "  A $JOB_A  Gaussian pretrain + exports (${MILESTONE_CSV})"
for i in "${!MILESTONES[@]}"; do
    echo "  ${MILESTONE_JOB_IDS[$i]}  ETTh2 finetune/eval from best-so-far epoch ${MILESTONES[$i]} [afterok:$JOB_A]"
done
echo ""
echo "  Exported checkpoints will be read from (under job A ckpt dir, see etth2-last-gauss-ckpt.txt):"
for EPOCH in "${MILESTONES[@]}"; do
    echo "    pretrained_diffusion_best_epoch${EPOCH}.pt"
done
echo ""
echo "  Logs:        $REPO/results/logs/*.log"
echo "  Checkpoints: $REPO/results/ckpts/"
echo "  Monitor:     squeue -u \$USER"
echo "  Cancel all:  scancel $JOB_A ${MILESTONE_JOB_IDS[*]}"
echo "=================================================================="
exit 0


# ============================================================================
# JOB B — Binary pretrain (copy iTrans from A, then binary diff HP + pretrain)
# ============================================================================

echo "Submitting B: Binary pretrain [afterok:$JOB_A]..."

JOB_B=$(sbatch --parsable --export=ALL \
    --job-name="etth2-binary-pretrain${SUFFIX}" \
    --account=aip-boyuwang \
    --nodes=1 --cpus-per-task="$CPUS" --mem="$MEM" \
    "${GPU_ARGS[@]}" \
    --time="$WALL_PRETRAIN" \
    --dependency="afterok:$JOB_A" \
    --chdir="${REPO}" \
    --output=/dev/null \
    --error=/dev/null \
    --mail-type=FAIL --mail-user=ccao87@uwo.ca \
    "$SBATCH_JOBS/job_B.sh")
echo "  -> B: $JOB_B"


# ============================================================================
# JOB C — Gaussian finetune ETTh2
# ============================================================================

echo "Submitting C: Gaussian finetune ETTh2 [afterok:$JOB_A]..."

JOB_C=$(sbatch --parsable --export=ALL \
    --job-name="etth2-gauss-finetune${SUFFIX}" \
    --account=aip-boyuwang \
    --nodes=1 --cpus-per-task="$CPUS" --mem="$MEM" \
    "${GPU_ARGS[@]}" \
    --time="$WALL_FINETUNE" \
    --dependency="afterok:$JOB_A" \
    --chdir="${REPO}" \
    --output=/dev/null \
    --error=/dev/null \
    --mail-type=FAIL --mail-user=ccao87@uwo.ca \
    "$SBATCH_JOBS/job_C.sh")
echo "  -> C: $JOB_C"


# ============================================================================
# JOB D — Binary finetune ETTh2
# ============================================================================

echo "Submitting D: Binary finetune ETTh2 [afterok:$JOB_B]..."

JOB_D=$(sbatch --parsable --export=ALL \
    --job-name="etth2-binary-finetune${SUFFIX}" \
    --account=aip-boyuwang \
    --nodes=1 --cpus-per-task="$CPUS" --mem="$MEM" \
    "${GPU_ARGS[@]}" \
    --time="$WALL_FINETUNE" \
    --dependency="afterok:$JOB_B" \
    --chdir="${REPO}" \
    --output=/dev/null \
    --error=/dev/null \
    --mail-type=FAIL --mail-user=ccao87@uwo.ca \
    "$SBATCH_JOBS/job_D.sh")
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
echo "    $REPO/results/logs/"
echo ""
echo "  Monitor:     squeue -u \$USER"
echo "  Quick check: squeue -j $JOB_A,$JOB_B,$JOB_C,$JOB_D -o '%.18i %.28j %.10T %.20R'"
echo "  Reasons:     sacct -j $JOB_A,$JOB_B,$JOB_C,$JOB_D -X --format=JobID,JobName,State,ExitCode,Reason"
echo "  Cancel all:  scancel $JOB_A $JOB_B $JOB_C $JOB_D"
echo "=================================================================="
