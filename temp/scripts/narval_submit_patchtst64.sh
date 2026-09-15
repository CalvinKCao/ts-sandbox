#!/bin/bash
# Login-node PatchTST/64 fullT resub. Do not scancel 2954159 / 2977742 / 5433386.
source /etc/profile >/dev/null 2>&1 || true
set -euo pipefail
cd /scratch/ccao87/ts-sandbox-fullT || exit 1

echo "HOST=$(hostname) PWD=$PWD DATE=$(date)"
echo "=== runner isolation ==="
grep -n "_out_dir_for\|hz{int(pred_len)}_lb" temp/scripts/run_baselines_canvas128_subset.py | head -20

echo "=== apply patches once ==="
python3 temp/scripts/apply_baseline_canvas128_patches.py

echo "=== protected ==="
squeue -j 2954159,2977742 -o "%.18i %.40j %.2t %.10M %.10l %R" || true

SUB=configs/compare_baselines_allv_fullT.yaml
EVAL=binary_10pct,test_100pct
BASE=./temp/scripts/submit_baselines_canvas128_killarney.sh
L64=512
chmod +x "$BASE"

echo "==== PatchTST/64 submit ===="
for h in 96 192 336 720; do
  "$BASE" --model patchtst --datasets ETTh1,ETTh2,ETTm1,ETTm2,weather,exchange_rate \
    --seq-len "$L64" --pred-len "$h" --subset-yaml "$SUB" --eval-subsets "$EVAL" --force --gpu a100 --time 1:20:00
  "$BASE" --model patchtst --datasets solar_Alabama \
    --seq-len "$L64" --pred-len "$h" --subset-yaml "$SUB" --eval-subsets "$EVAL" --force --gpu a100 --time 5:00:00
  "$BASE" --model patchtst --datasets electricity \
    --seq-len "$L64" --pred-len "$h" --subset-yaml "$SUB" --eval-subsets "$EVAL" --force --gpu a100 --time 10:00:00
  "$BASE" --model patchtst --datasets traffic \
    --seq-len "$L64" --pred-len "$h" --subset-yaml "$SUB" --eval-subsets "$EVAL" --force --gpu a100 --time 6:00:00
  DYN_T=8:00:00
  if [ "$h" = 720 ]; then DYN_T=14:00:00; fi
  "$BASE" --model patchtst --datasets dynamic \
    --seq-len "$L64" --pred-len "$h" --subset-yaml "$SUB" --eval-subsets "$EVAL" --force --gpu a100 --time "$DYN_T"
done

# ILI: official P=24 S=2 has no /64 script. L=148 yields 64 patches.
for h in 24 36 48 60; do
  "$BASE" --model patchtst --datasets illness \
    --seq-len 148 --pred-len "$h" --subset-yaml "$SUB" --eval-subsets "$EVAL" --force --gpu a100 --time 1:20:00
done

# PeMS: no official PatchTST script. L=512 P=16 S=8 -> 64 patches.
for h in 12 24 48 96; do
  "$BASE" --model patchtst --datasets PeMS \
    --seq-len "$L64" --pred-len "$h" --subset-yaml "$SUB" --eval-subsets "$EVAL" --force --gpu a100 --time 3:30:00
done

echo "==== squeue base-c128 ===="
squeue -u ccao87 -o "%.18i %.40j %.2t %.10M %.10l %R" | grep -E "JOBID|base-c128" || true
echo "==== protected still up ===="
squeue -j 2954159,2977742 -o "%.18i %.40j %.2t %.10M %.10l %R" || true
echo DONE
