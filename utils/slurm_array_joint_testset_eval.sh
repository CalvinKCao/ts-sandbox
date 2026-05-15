#!/bin/bash
# Parallel test-set evaluation for joint finetuned runs (gB / gC).
#
# Submit from anywhere; the job cwd must be the repo (same layout as training).
#
#   export TS="$SCRATCH/ts-sandbox"   # or /scratch/ccao87/ts-sandbox
#   export PROJECT=/abs/path/to/projects/def-...   # optional absolute path (not CCDB slug)
#   export VENV=...   # optional; else $PROJECT/$USER/diffusion-tsf/venv (or -fullvar)
#   mkdir -p "$TS/results/logs"
#   sbatch --chdir="$TS" --export=TS,R="${TS}/results" utils/slurm_array_joint_testset_eval.sh
#   Avoid --export=ALL here: some shells set PROJECT to the Slurm account slug (e.g.
#   aip-boyuwang), which is not the ~/projects/... path this script needs for venvs.
#
# Tune concurrency: --array=0-11%4  (example: max 4 GPUs at once)

#SBATCH --job-name=joint-testset-eval
#SBATCH --account=aip-boyuwang
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-11
# Logs land in repo cwd (set by sbatch --chdir); parent dir must exist before submit.
#SBATCH --output=results/logs/joint-testset-eval_%A_%a.out
#SBATCH --error=results/logs/joint-testset-eval_%A_%a.err

# Slurm often runs batch scripts with /bin/sh (dash), which rejects bash arrays.
if [ -z "${BASH_VERSION:-}" ]; then
  exec /bin/bash "$0" "$@"
fi

set -euo pipefail
set -x

: "${TS:?Pass TS (repo root), e.g. sbatch --export=ALL,TS=\$SCRATCH/ts-sandbox}"
: "${R:=${TS}/results}"

stems=(
  05-14-1295-joint-ft-ETTh1-gC
  05-14-1296-joint-ft-ETTh2-gC
  05-14-1297-joint-ft-ETTm1-gC
  05-14-1298-joint-ft-ETTm2-gC
  05-14-1299-joint-ft-exchange_rate-gC
  05-14-1300-joint-ft-illness-gC
  05-14-1303-joint-ft-ETTh1-gB
  05-14-1304-joint-ft-ETTh2-gB
  05-14-1305-joint-ft-ETTm1-gB
  05-14-1306-joint-ft-ETTm2-gB
  05-14-1307-joint-ft-exchange_rate-gB
  05-14-1308-joint-ft-illness-gB
)

stem="${stems[$SLURM_ARRAY_TASK_ID]:?bad SLURM_ARRAY_TASK_ID}"

cd "$TS"

# Filesystem root for venvs (run.sh calls this PROJECT). --export=ALL often injects
# PROJECT=aip-boyuwang (CCDB group / Slurm account), which is not a path — ignore it.
project_fs=""
if [ -n "${PROJECT:-}" ] && [ "${PROJECT#/}" != "${PROJECT}" ] && [ -d "$PROJECT" ]; then
  project_fs=$(readlink -f "$PROJECT")
fi
if [ -z "$project_fs" ] && [ -d "$HOME/projects" ]; then
  shopt -s nullglob
  _m=("$HOME"/projects/def-* "$HOME"/projects/aip-*)
  shopt -u nullglob
  if [ "${#_m[@]}" -gt 0 ]; then
    project_fs=$(readlink -f "${_m[0]}")
  fi
fi

# Explicit VENV wins only if it is a real venv (survives wrong paths from ALL).
if [ -n "${VENV:-}" ] && [ ! -f "${VENV}/bin/activate" ]; then
  VENV=""
fi
if [ -z "${VENV:-}" ]; then
  if [ -z "$project_fs" ]; then
    echo "ERROR: could not resolve project space under ~/projects (def-*|aip-*)."
    echo "Raw PROJECT=${PROJECT:-unset} (ignored unless absolute directory)."
    exit 1
  fi
  for v in "$project_fs/$USER/diffusion-tsf/venv" "$project_fs/$USER/diffusion-tsf-fullvar/venv"; do
    if [ -f "$v/bin/activate" ]; then
      VENV="$v"
      break
    fi
  done
fi

if [ ! -f "${VENV:-}/bin/activate" ]; then
  echo "ERROR: no venv found. Set VENV to the venv root (directory containing bin/activate)."
  echo "project_fs=${project_fs:-none} raw_PROJECT=${PROJECT:-unset}"
  exit 2
fi

module purge || true
module load StdEnv/2023 python/3.11 cuda/12.2 cudnn/8.9

# shellcheck disable=SC1090
source "$VENV/bin/activate"

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

mkdir -p "$R/$stem/eval_test"

python -m models.diffusion_tsf.eval_joint_testset \
  --results-dir "$R/$stem/eval_test" \
  --checkpoint-dir "$R/$stem/ckpts" \
  --full-test
