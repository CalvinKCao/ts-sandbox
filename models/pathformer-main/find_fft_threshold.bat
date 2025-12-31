@echo off
REM FFT Threshold Finder - Find optimal frequency threshold for FFT-DILATE loss

echo ========================================
echo FFT Threshold Finder
echo ========================================
echo.
echo This script analyzes your dataset to find the optimal
echo frequency threshold for FFT-DILATE loss.
echo.
echo It will create visualizations showing:
echo   - High and low frequency decomposition
echo   - Energy distribution across thresholds
echo   - Recommendations based on your data
echo.

REM Default: Analyze ETTh1 dataset
python fft_threshold_finder.py ^
  --dataset C:\Users\kevin\dev\ts-sandbox\datasets\ETT-small\ETTh1.csv ^
  --seq_len 96 ^
  --num_samples 3 ^
  --thresholds 50 60 70 75 80 85 90 95 ^
  --output_dir visualizations\fft_threshold_analysis_etth1

echo.
echo ========================================
echo Analysis complete!
echo Check: visualizations\fft_threshold_analysis_etth1
echo ========================================
echo.

pause
