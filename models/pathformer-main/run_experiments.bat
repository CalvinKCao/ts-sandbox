@echo off
REM Run multiple experiments with Pathformer on ETTh2
REM This script trains the model with different loss functions and then visualizes results

echo ========================================
echo Pathformer ETTh2 Experiment Runner
echo ========================================
echo.

REM Check if we should skip training
set SKIP_TRAINING=0
if "%1"=="--skip-training" set SKIP_TRAINING=1
if "%1"=="--visualize-only" set SKIP_TRAINING=1

if %SKIP_TRAINING%==0 (
    echo Starting training experiments...
    echo.
    
    echo ----------------------------------------
    echo Experiment 1: Training with MAE Loss
    echo ----------------------------------------
    python train_etth2.py --loss_type mae --train_epochs 30 --batch_size 512
    echo.
    
    echo ----------------------------------------
    echo Experiment 2: Training with MSE Loss
    echo ----------------------------------------
    python train_etth2.py --loss_type mse --train_epochs 30 --batch_size 512
    echo.
    
    echo ----------------------------------------
    echo Experiment 3: Training with DILATE Loss (Balanced)
    echo ----------------------------------------
    python train_etth2.py --loss_type dilate --dilate_alpha 0.5 --dilate_gamma 0.01 --train_epochs 30 --batch_size 512
    echo.
    
    echo ----------------------------------------
    echo Experiment 4: Training with DILATE Loss (Shape-focused)
    echo ----------------------------------------
    python train_etth2.py --loss_type dilate --dilate_alpha 0.8 --dilate_gamma 0.01 --train_epochs 30 --batch_size 512
    echo.
    
    echo ----------------------------------------
    echo Experiment 5: Training with DILATE Loss (Temporal-focused)
    echo ----------------------------------------
    python train_etth2.py --loss_type dilate --dilate_alpha 0.2 --dilate_gamma 0.01 --train_epochs 30 --batch_size 512
    echo.
    
    echo ========================================
    echo Training Complete!
    echo ========================================
    echo.
) else (
    echo Skipping training phase...
    echo.
)

echo ========================================
echo Starting Visualization Phase
echo ========================================
echo.

echo Visualizing MAE results...
python visualize_etth2.py --setting ETTh2_PathFormer_ftM_sl96_pl96_0_mae --save_metrics
echo.

echo Visualizing MSE results...
python visualize_etth2.py --setting ETTh2_PathFormer_ftM_sl96_pl96_0_mse --save_metrics
echo.

echo Visualizing DILATE (alpha=0.5) results...
python visualize_etth2.py --setting ETTh2_PathFormer_ftM_sl96_pl96_0_dilate_a0.5_g0.01 --save_metrics
echo.

echo Visualizing DILATE (alpha=0.8) results...
python visualize_etth2.py --setting ETTh2_PathFormer_ftM_sl96_pl96_0_dilate_a0.8_g0.01 --save_metrics
echo.

echo Visualizing DILATE (alpha=0.2) results...
python visualize_etth2.py --setting ETTh2_PathFormer_ftM_sl96_pl96_0_dilate_a0.2_g0.01 --save_metrics
echo.

echo ========================================
echo All Experiments Complete!
echo ========================================
echo.
echo Results saved in:
echo - Checkpoints: ./checkpoints/
echo - Visualizations: ./visualizations/
echo - Test Results: ./test_results/
echo.
echo To run only visualization (skip training):
echo   run_experiments.bat --visualize-only
echo.

pause
