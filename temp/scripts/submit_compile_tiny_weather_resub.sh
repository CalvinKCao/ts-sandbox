#!/bin/bash
# Resubmit the directional torch.compile Weather update benchmark on Killarney.
#SBATCH --job-name=compile-tiny-resub
#SBATCH --account=aip-boyuwang
#SBATCH --partition=gpubase_l40s_b1
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=results/logs/compile-tiny-resub-%j.out
#SBATCH --error=results/logs/compile-tiny-resub-%j.out

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:?submit this script from the repository root}"
cd "$REPO_ROOT"
BENCHMARK_CONFIG="${BENCHMARK_CONFIG:-configs/binary_window_norm_patch_refine_canvas128_p64x6_weather_allv_msdefault_fixed.yaml}"
BENCHMARK_CHECKPOINT_ROOT="${BENCHMARK_CHECKPOINT_ROOT:-results/ckpts/08-14-4794022-weather-binary_window_norm_patch_refine_canvas128_p64x6_weather_allv_msdefault_fixed}"
BENCHMARK_CHECKPOINT_RUN="${BENCHMARK_CHECKPOINT_RUN:-weather_allv_s8}"
module purge || true
module load StdEnv/2023 python/3.11 cuda/12.2 cudnn/8.9

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip -q
pip install --no-index -r setup/requirements-killarney.txt -q
python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'

PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" python -u temp/scripts/benchmark_torch_compile_weather.py \
  --config "$BENCHMARK_CONFIG" \
  --checkpoint-root "$BENCHMARK_CHECKPOINT_ROOT" \
  --checkpoint-run "$BENCHMARK_CHECKPOINT_RUN" \
  --warmup-updates 0 \
  --timed-updates 1 \
  "$@"
