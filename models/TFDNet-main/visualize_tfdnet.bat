@echo off
set RESULTS_DIR=test_results\ETTh1_512_96_TFDNet_ETTh1_ftM_sl512_ll48_pl96_dm512_nh8_el2_dl1_df64_fc1_ebtimeF_dtTrue_Exp_0

python visualize_results.py ^
  --folder_path %RESULTS_DIR% ^
  --num_samples 10 ^
  --num_channels 7
