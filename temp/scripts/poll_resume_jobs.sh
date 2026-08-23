#!/bin/bash
set +e
cd /scratch/ccao87/ts-sandbox
echo === DATE ===
date
echo === SINFO L40S ===
sinfo -h -o "%P %l %m %G" | grep -i l40s
echo === PARTITION MaxMemPerNode ===
for p in gpubase_l40s_b1 gpubase_l40s_b2 gpubase_l40s_b3 gpubase_l40s_b4 gpubase_l40s_b5; do
  echo "-- $p --"
  scontrol show partition "$p" | tr ' ' '\n' | grep -iE 'PartitionName|MaxTime|DefMemPerNode|MaxMemPerNode|MaxMemPerCPU|Nodes|PriorityTier'
done
echo === SQUEUE ===
squeue -u ccao87 -o "%i %T %M %L %P %m %N %j"
echo === CONFIGS ===
ls -1 configs/*electricity*allv* configs/*pems03* configs/*pems07* configs/*pems08* configs/*traffic*s1_groups* configs/*dynamic*s1_groups* 2>/dev/null
echo === CKPT DIRS ===
ls -ld results/ckpts/*electricity_allv* results/ckpts/*pems03* results/ckpts/*pems07* results/ckpts/*pems08* results/ckpts/*traffic_v000_430_s1* results/ckpts/*traffic_v431_861_s1* results/ckpts/*dynamic_allv_s1* 2>/dev/null
echo === GIT ===
git branch --show-current
git log -1 --oneline
echo === TRAFFIC 458 LOG ===
L458=$(ls -1t results/logs/*traffic_v000_430_s1* 2>/dev/null | head -1)
echo "LOG458=$L458"
if [ -n "$L458" ]; then
  tail -n 40 "$L458"
  echo "-- eval batches --"
  grep -E "staged eval batch|staged eval start|already evaluated|cached:|patch_refine|coarse" "$L458" | tail -25
fi
echo === TRAFFIC 459 LOG ===
L459=$(ls -1t results/logs/*traffic_v431_861_s1* 2>/dev/null | head -1)
echo "LOG459=$L459"
if [ -n "$L459" ]; then
  tail -n 40 "$L459"
  echo "-- eval batches --"
  grep -E "staged eval batch|staged eval start|already evaluated|cached:|patch_refine|coarse" "$L459" | tail -25
fi
echo === ELEC 4862420 LOG TAIL ===
L420=$(ls -1t results/logs/*4862420* results/logs/*electricity_allv* 2>/dev/null | head -3)
echo "L420 candidates:"
echo "$L420"
for f in $L420; do
  echo "-- $f --"
  grep -E "OUT OF MEMORY|Killed|patch_refine|coarse|cached:|pinned|GiB|copy_|pack" "$f" | tail -30
  echo "-- last 20 --"
  tail -n 20 "$f"
done
echo === PEMS LOGS ===
for tag in pems03 pems07 pems08; do
  f=$(ls -1t results/logs/*${tag}* 2>/dev/null | head -1)
  echo "-- $f --"
  grep -E "trial|TIMEOUT|cached:|patch_guidance|coarse|epoch|best val" "$f" | tail -20
  echo "-- last 15 --"
  tail -n 15 "$f"
done
echo === DYNAMIC LOG ===
f=$(ls -1t results/logs/*dynamic_allv_s1* 2>/dev/null | head -1)
echo "DYN=$f"
grep -E "staged eval batch|anchor_mse|already evaluated|cached:|patch_refine" "$f" | tail -20
echo "-- last 15 --"
tail -n 15 "$f"
echo === CKPT CONTENTS ===
for d in results/ckpts/*electricity_allv* results/ckpts/*pems03* results/ckpts/*pems07* results/ckpts/*pems08* results/ckpts/*dynamic_allv_s1*; do
  [ -d "$d" ] || continue
  echo "==== $d ===="
  find "$d" -maxdepth 4 -type f \( -name 'best.pt' -o -name 'metadata.json' -o -name '*guidance*.pt' -o -name 'trial_*.pt' -o -name '*hp_best.pt' -o -name 'staged_results*.json' \) | sort
done
echo === EVAL PARTIALS DYNAMIC ===
find results/datasets/*dynamic_allv_s1* -name '*staged*' -o -name '*partial*' 2>/dev/null | head -40
echo === TRAFFIC EVAL PARTIALS ===
find results/datasets/*traffic_v000_430_s1* results/datasets/*traffic_v431_861_s1* -name '*staged*' 2>/dev/null | head -40
