@echo off
REM Training script demonstrating FFT-DILATE loss on ETTh1
REM This applies DILATE loss only to high-frequency components

echo ========================================
echo FFT-DILATE Loss Training on ETTh1
echo ========================================
echo.
echo This script demonstrates the new FFT-DILATE loss which:
echo - Applies DILATE loss only to high-frequency components (shape-aware)
echo - Uses MAE loss for low-frequency components
echo - Frequency threshold is configurable (default 80%% = top 20%% frequencies)
echo.

set seq_len=96
set model_name=PathFormer
set root_path_name=C:\Users\kevin\dev\ts-sandbox\datasets\ETT_small
set data_path_name=ETTh1.csv
set model_id_name=ETTh1
set data_name=ETTh1

REM Create logs directory if it doesn't exist
if not exist "logs" mkdir logs
if not exist "logs\LongForecasting" mkdir logs\LongForecasting

REM ========================================
REM Example 1: FFT-DILATE with 80%% threshold (top 20%% frequencies get DILATE)
REM ========================================
echo.
echo Training with FFT-DILATE (80%% threshold - top 20%% frequencies)...
python -u run.py ^
  --is_training 1 ^
  --root_path %root_path_name% ^
  --data_path %data_path_name% ^
  --model_id %model_id_name%_%seq_len%_96_fft_dilate ^
  --model %model_name% ^
  --data %data_name% ^
  --features M ^
  --seq_len %seq_len% ^
  --pred_len 96 ^
  --patch_size_list 16 12 8 32 12 8 6 4 8 6 4 2 ^
  --num_nodes 7 ^
  --layer_nums 3 ^
  --batch_norm 0 ^
  --residual_connection 1 ^
  --k 3 ^
  --d_model 4 ^
  --d_ff 64 ^
  --train_epochs 30 ^
  --patience 10 ^
  --lradj TST ^
  --itr 1 ^
  --batch_size 128 ^
  --learning_rate 0.001 ^
  --loss_type fft_dilate ^
  --dilate_alpha 0.5 ^
  --dilate_gamma 0.01 ^
  --freq_threshold 80.0 ^
  --metric mae >logs\LongForecasting\%model_name%_%model_id_name%_%seq_len%_96_fft_dilate_80.log

REM ========================================
REM Example 2: FFT-DILATE with 70%% threshold (top 30%% frequencies get DILATE)
REM ========================================
echo.
echo Training with FFT-DILATE (70%% threshold - top 30%% frequencies)...
python -u run.py ^
  --is_training 1 ^
  --root_path %root_path_name% ^
  --data_path %data_path_name% ^
  --model_id %model_id_name%_%seq_len%_96_fft_dilate_70 ^
  --model %model_name% ^
  --data %data_name% ^
  --features M ^
  --seq_len %seq_len% ^
  --pred_len 96 ^
  --patch_size_list 16 12 8 32 12 8 6 4 8 6 4 2 ^
  --num_nodes 7 ^
  --layer_nums 3 ^
  --batch_norm 0 ^
  --residual_connection 1 ^
  --k 3 ^
  --d_model 4 ^
  --d_ff 64 ^
  --train_epochs 30 ^
  --patience 10 ^
  --lradj TST ^
  --itr 1 ^
  --batch_size 128 ^
  --learning_rate 0.001 ^
  --loss_type fft_dilate ^
  --dilate_alpha 0.5 ^
  --dilate_gamma 0.01 ^
  --freq_threshold 70.0 ^
  --metric mae >logs\LongForecasting\%model_name%_%model_id_name%_%seq_len%_96_fft_dilate_70.log

REM ========================================
REM Example 3: FFT-DILATE with 90%% threshold (top 10%% frequencies get DILATE)
REM ========================================
echo.
echo Training with FFT-DILATE (90%% threshold - top 10%% frequencies)...
python -u run.py ^
  --is_training 1 ^
  --root_path %root_path_name% ^
  --data_path %data_path_name% ^
  --model_id %model_id_name%_%seq_len%_96_fft_dilate_90 ^
  --model %model_name% ^
  --data %data_name% ^
  --features M ^
  --seq_len %seq_len% ^
  --pred_len 96 ^
  --patch_size_list 16 12 8 32 12 8 6 4 8 6 4 2 ^
  --num_nodes 7 ^
  --layer_nums 3 ^
  --batch_norm 0 ^
  --residual_connection 1 ^
  --k 3 ^
  --d_model 4 ^
  --d_ff 64 ^
  --train_epochs 30 ^
  --patience 10 ^
  --lradj TST ^
  --itr 1 ^
  --batch_size 128 ^
  --learning_rate 0.001 ^
  --loss_type fft_dilate ^
  --dilate_alpha 0.6 ^
  --dilate_gamma 0.01 ^
  --freq_threshold 90.0 ^
  --metric mae >logs\LongForecasting\%model_name%_%model_id_name%_%seq_len%_96_fft_dilate_90.log

echo.
echo ========================================
echo FFT-DILATE training complete!
echo ========================================
echo.
echo Key hyperparameters:
echo   --loss_type fft_dilate       : Use frequency-selective DILATE loss
echo   --freq_threshold 80.0         : Percentile threshold (80 = top 20%% are high-freq)
echo   --dilate_alpha 0.5            : Shape vs temporal weight for high frequencies
echo   --dilate_gamma 0.01           : Soft-DTW smoothing parameter
echo.
echo How it works:
echo   1. FFT transforms predictions and targets to frequency domain
echo   2. High frequencies (above threshold) get DILATE loss (shape-aware)
echo   3. Low frequencies (below threshold) get MAE loss (point-wise)
echo   4. Losses are weighted by signal energy
echo.
echo Check logs in: logs\LongForecasting\
