#!/usr/bin/env bash
set -euo pipefail
S="$(cd "$(dirname "${BASH_SOURCE[0]}")"&&pwd)"; R=256; D=ETTh1; T=8:00:00
while [[ $# -gt 0 ]]; do case "$1" in --resolution) R=$2;shift 2;;--dataset)D=$2;shift 2;;--time)T=$2;shift 2;;*)exit 2;;esac;done
if [[ -z ${SLURM_JOB_ID:-} ]];then exec sbatch --account=def-boyuwang --gpus=a100:1 --cpus-per-task=8 --mem=80G --time=$T --job-name=ord-full-$D-$R --export=ALL,ORD_REPO=${SCRATCH}/ts-sandbox,ORD_R=$R,ORD_D=$D "$S/submit_ordinal_patch_refinement_full_narval.sh";fi
cd "$ORD_REPO"; module load StdEnv/2023 python/3.11 cuda/12.2 cudnn/8.9; source "$SLURM_TMPDIR/env/bin/activate" 2>/dev/null || { virtualenv --no-download "$SLURM_TMPDIR/env";source "$SLURM_TMPDIR/env/bin/activate";pip install --no-index -r setup/requirements-killarney.txt;}; PYTHONPATH=$PWD python -m experiments.ordinal_patch_refinement_killtest.full_experiment --dataset "$ORD_D" --resolution "$ORD_R" --output "results/ordinal_patch_refinement_killtest/full-${ORD_D}-${ORD_R}-${SLURM_JOB_ID}"
