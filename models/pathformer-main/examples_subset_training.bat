@echo off
REM Example commands for running the subset training script
REM Copy and paste these commands into your terminal

echo ========================================
echo PATHFORMER SUBSET TRAINING - EXAMPLES
echo ========================================
echo.
echo Choose an example to run:
echo.
echo [1] Quick test (10%% train, 5%% tune, 5 epochs)
echo     python train_subset_with_tuning.py --train_subset 0.1 --tune_subset 0.05 --train_epochs 5
echo.
echo [2] Default (25%% train, 10%% tune, 10 epochs)
echo     python train_subset_with_tuning.py
echo.
echo [3] Medium (50%% train, 15%% tune, 15 epochs)
echo     python train_subset_with_tuning.py --train_subset 0.5 --tune_subset 0.15 --train_epochs 15
echo.
echo [4] Skip tuning - use specific hyperparams
echo     python train_subset_with_tuning.py --train_subset 0.25 --skip_tuning --learning_rate 0.001 --d_model 32
echo.
echo [5] Full training with tuning (100%% data)
echo     python train_subset_with_tuning.py --train_subset 1.0 --tune_subset 0.2 --train_epochs 20
echo.
echo ========================================
echo Press any key to run Example 1 (Quick test)...
pause >nul

python train_subset_with_tuning.py --train_subset 0.1 --tune_subset 0.05 --train_epochs 5 --num_workers 0

pause
