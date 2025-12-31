@echo off
REM Visualize Pathformer predictions on ETTh1

echo ========================================
echo Visualizing Pathformer on ETTh1
echo ========================================
echo.

REM Visualize prediction length 96
echo Visualizing pred_len=96...
python visualize_etth2.py ^
  --root_path C:\Users\kevin\dev\ts-sandbox\datasets\ETT_small ^
  --data_path ETTh1.csv ^
  --model_id ETTh1_96_96 ^
  --model PathFormer ^
  --data ETTh1 ^
  --features M ^
  --seq_len 96 ^
  --pred_len 96 ^
  --num_nodes 7 ^
  --layer_nums 3 ^
  --patch_size_list 16 12 8 32 12 8 6 4 8 6 4 2 ^
  --k 3 ^
  --d_model 4 ^
  --d_ff 64 ^
  --residual_connection 1 ^
  --loss_type mae ^
  --num_samples 5 ^
  --save_metrics

echo.
REM Visualize prediction length 192
echo Visualizing pred_len=192...
python visualize_etth2.py ^
  --root_path C:\Users\kevin\dev\ts-sandbox\datasets\ETT_small ^
  --data_path ETTh1.csv ^
  --model_id ETTh1_96_192 ^
  --model PathFormer ^
  --data ETTh1 ^
  --features M ^
  --seq_len 96 ^
  --pred_len 192 ^
  --num_nodes 7 ^
  --layer_nums 3 ^
  --patch_size_list 16 12 8 32 12 8 6 4 8 6 4 2 ^
  --k 3 ^
  --d_model 4 ^
  --d_ff 64 ^
  --residual_connection 1 ^
  --loss_type mae ^
  --num_samples 5 ^
  --save_metrics

echo.
REM Visualize prediction length 336
echo Visualizing pred_len=336...
python visualize_etth2.py ^
  --root_path C:\Users\kevin\dev\ts-sandbox\datasets\ETT_small ^
  --data_path ETTh1.csv ^
  --model_id ETTh1_96_336 ^
  --model PathFormer ^
  --data ETTh1 ^
  --features M ^
  --seq_len 96 ^
  --pred_len 336 ^
  --num_nodes 7 ^
  --layer_nums 3 ^
  --patch_size_list 16 12 8 32 12 8 6 16 8 6 4 16 ^
  --k 3 ^
  --d_model 4 ^
  --d_ff 64 ^
  --residual_connection 0 ^
  --loss_type mae ^
  --num_samples 5 ^
  --save_metrics

echo.
REM Visualize prediction length 720
echo Visualizing pred_len=720...
python visualize_etth2.py ^
  --root_path C:\Users\kevin\dev\ts-sandbox\datasets\ETT_small ^
  --data_path ETTh1.csv ^
  --model_id ETTh1_96_720 ^
  --model PathFormer ^
  --data ETTh1 ^
  --features M ^
  --seq_len 96 ^
  --pred_len 720 ^
  --num_nodes 7 ^
  --layer_nums 3 ^
  --patch_size_list 16 12 8 32 12 8 6 4 8 6 4 2 ^
  --k 2 ^
  --d_model 4 ^
  --d_ff 64 ^
  --residual_connection 0 ^
  --loss_type mae ^
  --num_samples 5 ^
  --save_metrics

echo.
echo ========================================
echo Visualization complete!
echo Check visualizations\ folder for outputs
echo ========================================
