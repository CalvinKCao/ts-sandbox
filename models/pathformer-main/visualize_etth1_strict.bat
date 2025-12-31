@echo off
REM Visualize Pathformer predictions on ETTh1 for the strict DILATE model

echo ========================================
echo Visualizing Pathformer on ETTh1 (Strict DILATE)
echo ========================================
echo.

REM Visualize prediction length 96
echo Visualizing pred_len=96...
python visualize_etth2.py ^
  --root_path C:\Users\kevin\dev\ts-sandbox\datasets\ETT-small ^
  --data_path ETTh1.csv ^
  --model_id ETTh1_96_96_fft_dilate_strict ^
  --model PathFormer ^
  --data ETTh1 ^
  --features M ^
  --seq_len 96 ^
  --pred_len 96 ^
  --num_nodes 7 ^
  --layer_nums 3 ^
  --k 3 ^
  --d_model 4 ^
  --d_ff 64 ^
  --patch_size_list 16 12 8 32 12 8 6 4 8 6 4 2 ^
  --checkpoints C:\Users\kevin\dev\ts-sandbox\checkpoints ^
  --setting ETTh1_96_96_fft_dilate_strict_PathFormer_ftETTh1_slM_pl96_96
