@echo off
REM Training script for ETTh1 based on exact paper specifications
REM Paper: Pathformer: Multi-scale Transformers with Adaptive Pathways for Time Series Forecasting
REM Published at ICLR 2024

REM Create logs directory if it doesn't exist
if not exist "logs" mkdir logs
if not exist "logs\LongForecasting" mkdir logs\LongForecasting

REM Paper specifications:
REM - Input length H = 96
REM - Adam optimizer with learning rate 10^-3 (0.001)
REM - L1 Loss (MAE)
REM - Early stopping within 10 epochs
REM - 3 Adaptive Multi-Scale Blocks (AMS Blocks)
REM - 4 different patch sizes per block from pool: {2, 3, 6, 12, 16, 24, 32}
REM - Prediction lengths F = {96, 192, 336, 720}

set seq_len=96
set model_name=PathFormer
set root_path_name=C:\Users\kevin\dev\ts-sandbox\datasets\ETT_small
set data_path_name=ETTh1.csv
set model_id_name=ETTh1
set data_name=ETTh1

echo ========================================
echo Training Pathformer on ETTh1
echo Using EXACT paper hyperparameters
echo ========================================
echo.
echo Paper specifications:
echo - Input length: 96
echo - Learning rate: 0.001 (paper states 10^-3)
echo - Loss function: L1 Loss (MAE)
echo - Early stopping: 10 epochs patience
echo - Optimizer: Adam
echo - 3 AMS Blocks with 4 patch sizes each
echo - Patch size pool: {2, 3, 6, 12, 16, 24, 32}
echo.

REM Prediction length 96
echo Training for prediction length 96...
python -u run.py ^
  --is_training 1 ^
  --root_path %root_path_name% ^
  --data_path %data_path_name% ^
  --model_id %model_id_name%_%seq_len%_96 ^
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
  --metric mae >logs\LongForecasting\%model_name%_%model_id_name%_%seq_len%_96_paper_exact.log
