#!/bin/bash
set -euo pipefail
cd /scratch/ccao87/ts-sandbox
echo "=== pre-submit $(date) ==="
git branch --show-current
git log -1 --oneline

echo "=== traffic ckpt trees ==="
for d in results/ckpts/08-18-4870458-traffic-binary_window_norm_patch_refine_canvas128_p64x6_traffic_v000_430_s1_groups_msdefault_fixed \
         results/ckpts/08-18-4870459-traffic-binary_window_norm_patch_refine_canvas128_p64x6_traffic_v431_861_s1_groups_msdefault_fixed; do
  echo "---- $d ----"
  find "$d" -maxdepth 4 -type f \( -name 'best.pt' -o -name 'metadata.json' -o -name '*guidance*.pt' \) | sort
done

echo "=== scancel traffic 4870458 4870459 ==="
scancel 4870458 4870459
sleep 2
squeue -u ccao87 -o "%i %T %j" || true

echo "=== submit resumes ==="
# Cartesian-safe: one config x one dataset per call.

echo "--- electricity 321 allv 500G 2d ---"
if ! ./submit_binary.sh --resume \
  --configs binary_window_norm_patch_refine_canvas128_p64x6_electricity_allv_msdefault_fixed \
  --datasets electricity \
  --mem 500G --time 2-00:00:00; then
  echo "500G rejected; retrying max-under-node 500000M"
  ./submit_binary.sh --resume \
    --configs binary_window_norm_patch_refine_canvas128_p64x6_electricity_allv_msdefault_fixed \
    --datasets electricity \
    --mem 500000M --time 2-00:00:00
fi

echo "--- PEMS03 350G 1d ---"
./submit_binary.sh --resume \
  --configs binary_window_norm_patch_refine_canvas128_p64x6_pems03_allv_msdefault_fixed \
  --datasets PEMS03 \
  --mem 350G --time 1-00:00:00

echo "--- PEMS07 450G 1d ---"
./submit_binary.sh --resume \
  --configs binary_window_norm_patch_refine_canvas128_p64x6_pems07_allv_msdefault_fixed \
  --datasets PEMS07 \
  --mem 450G --time 1-00:00:00

echo "--- PEMS08 350G 1d ---"
./submit_binary.sh --resume \
  --configs binary_window_norm_patch_refine_canvas128_p64x6_pems08_allv_msdefault_fixed \
  --datasets PEMS08 \
  --mem 350G --time 1-00:00:00

echo "--- dynamic eval 350G 12h ---"
./submit_binary.sh --resume \
  --configs binary_window_norm_patch_refine_canvas128_p64x6_dynamic_allv_s1_groups_msdefault_fixed \
  --datasets dynamic \
  --mem 350G --time 12:00:00

echo "--- traffic 0-430 eval 450G 1d ---"
./submit_binary.sh --resume \
  --configs binary_window_norm_patch_refine_canvas128_p64x6_traffic_v000_430_s1_groups_msdefault_fixed \
  --datasets traffic \
  --mem 450G --time 1-00:00:00

echo "--- traffic 431-861 eval 450G 1d ---"
./submit_binary.sh --resume \
  --configs binary_window_norm_patch_refine_canvas128_p64x6_traffic_v431_861_s1_groups_msdefault_fixed \
  --datasets traffic \
  --mem 450G --time 1-00:00:00

echo "=== post-submit squeue $(date) ==="
squeue -u ccao87 -o "%i %T %M %L %P %m %j"
