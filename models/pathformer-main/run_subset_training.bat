@echo off
REM Quick script to run subset training with hyperparameter tuning
REM 
REM Usage examples:
REM   run_subset_training.bat              (uses defaults: 25% train, 10% tune)
REM   run_subset_training.bat 0.5 0.15     (50% train, 15% tune)

setlocal

set TRAIN_SUBSET=0.25
set TUNE_SUBSET=0.10

if not "%1"=="" set TRAIN_SUBSET=%1
if not "%2"=="" set TUNE_SUBSET=%2

echo ========================================
echo Running Pathformer Subset Training
echo ========================================
echo Train subset: %TRAIN_SUBSET%
echo Tune subset: %TUNE_SUBSET%
echo ========================================
echo.

python train_subset_with_tuning.py --train_subset %TRAIN_SUBSET% --tune_subset %TUNE_SUBSET%

echo.
echo ========================================
echo Training complete!
echo ========================================

pause
