#!/bin/bash
# Parallel test-set evaluation for joint finetuned runs (gB / gC).
#
# Submit from anywhere; the job cwd must be the repo (same layout as training).
#
#   export TS="$SCRATCH/ts-sandbox"   # or /scratch/ccao87/ts-sandbox
#   export PROJECT=/path/from/projects   # optional; venv discovery
#   export VENV="$PROJECT/$USER/diffusion-tsf/venv"   # optional override
#   mkdir -p "$TS/results/logs"
#   sbatch --chdir="$TS" --export=ALL,TS,R="${TS}/results" utils/slurm_array_joint_testset_eval.sh
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

if [ -z "${VENV:-}" ]; then
  _candidates=()
  [ -n "${PROJECT:-}" ] && _candidates+=("$PROJECT/$USER/diffusion-tsf/venv")
  [ -n "${PROJECT:-}" ] && _candidates+=("$PROJECT/$USER/diffusion-tsf-fullvar/venv")
  _candidates+=("$HOME/projects/$USER/diffusion-tsf/venv")
  for v in "${_candidates[@]}"; do
    if [ -f "$v/bin/activate" ]; then
      VENV="$v"
      break
    fi
  done
fi

if [ ! -f "${VENV:-}/bin/activate" ]; then
  echo "ERROR: no venv found. Set VENV to your activate path (see run.sh)."
  echo "Tried PROJECT=${PROJECT:-unset}"
  exit 2
fi

module purge
module load StdEnv/2023 python/3.11 cuda/12.2 cudnn/8.9

# shellcheck disable=SC1090
source "$VENV/bin/activate"

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

mkdir -p "$R/$stem/eval_test"

python -m models.diffusion_tsf.eval_joint_testset \
  --results-dir "$R/$stem/eval_test" \
  --checkpoint-dir "$R/$stem/ckpts" \
  --full-test
