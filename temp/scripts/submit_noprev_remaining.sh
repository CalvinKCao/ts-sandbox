#!/bin/bash
set -euo pipefail
cd /scratch/ccao87/ts-sandbox-no-prev-refine
./submit_binary.sh --configs binary_window_norm_patch_refine_canvas128_p64x6_electricity_no_prev_cond --datasets electricity --time 20:00:00 --mem 80G
./submit_binary.sh --configs binary_window_norm_patch_refine_canvas128_p64x6_exchange_no_prev_cond --datasets exchange_rate --time 20:00:00 --mem 80G
./submit_binary.sh --configs binary_window_norm_patch_refine_canvas128_p64x6_pems_no_prev_cond --datasets PeMS --time 20:00:00 --mem 80G
./submit_binary.sh --configs binary_window_norm_patch_refine_canvas128_p64x6_solar_no_prev_cond --datasets solar_Alabama --time 20:00:00 --mem 80G
./submit_binary.sh --configs binary_window_norm_patch_refine_canvas128_p64x6_ettm1_no_prev_cond --datasets ETTm1 --time 20:00:00 --mem 80G
./submit_binary.sh --configs binary_window_norm_patch_refine_canvas128_p64x6_ettm2_no_prev_cond --datasets ETTm2 --time 20:00:00 --mem 80G
./submit_binary.sh --configs binary_window_norm_patch_refine_canvas128_p64x6_weather_no_prev_cond --datasets weather --time 20:00:00 --mem 80G
