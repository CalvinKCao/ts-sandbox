#!/bin/bash
# Instant-exit queue probe. Resource request is set at sbatch time.
set -euo pipefail
echo "host=$(hostname) job=${SLURM_JOB_ID:-} node=${SLURMD_NODENAME:-}"
echo "start=$(date -Is)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-}"
nvidia-smi -L 2>/dev/null || echo "nvidia-smi unavailable"
echo "done=$(date -Is)"
exit 0
