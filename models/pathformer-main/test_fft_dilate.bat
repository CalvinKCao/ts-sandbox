@echo off
REM Quick test of FFT-DILATE loss

echo Testing FFT-DILATE loss on ETTh1...
echo.

python run.py ^
  --is_training 1 ^
  --root_path C:\Users\kevin\dev\ts-sandbox\datasets\ETT-small ^
  --data_path ETTh1.csv ^
  --model_id ETTh1_fft_test ^
  --model PathFormer ^
  --data ETTh1 ^
  --features M ^
  --seq_len 96 ^
  --pred_len 96 ^
  --patch_size_list 16 12 8 32 12 8 6 4 8 6 4 2 ^
  --num_nodes 7 ^
  --layer_nums 3 ^
  --k 3 ^
  --d_model 4 ^
  --d_ff 64 ^
  --train_epochs 3 ^
  --batch_size 128 ^
  --learning_rate 0.001 ^
  --loss_type fft_dilate ^
  --freq_threshold 80.0 ^
  --dilate_alpha 0.5 ^
  --dilate_gamma 0.01

echo.
echo Test complete! Check that FFT-DILATE loss is being used in the output above.
pause
