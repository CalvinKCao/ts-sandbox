@echo off
REM Quick start script - Train and visualize with default settings

echo ========================================
echo Quick Start: Pathformer on ETTh2
echo ========================================
echo.

echo Training with MAE loss (default)...
python train_etth2.py --train_epochs 20
echo.

echo Training complete! Now visualizing results...
python visualize_etth2.py --save_metrics
echo.

echo ========================================
echo Done!
echo ========================================
echo.
echo Check the following directories:
echo - Checkpoints: ./checkpoints/
echo - Visualizations: ./visualizations/
echo.
echo To try DILATE loss, run:
echo   python train_etth2.py --loss_type dilate
echo   python visualize_etth2.py
echo.

pause
