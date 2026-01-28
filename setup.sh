#!/bin/bash
# ============================================================================
# DIFFUSION TSF - SMART SETUP SCRIPT
# Handles: venv, auto-CUDA detection, and all dependencies
# ============================================================================

set -e

echo "🚀 Starting Smart Setup..."

# 1. Install System Dependencies
echo "📦 Installing system packages..."
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev python3-venv git sqlite3

# 2. Create and Activate Virtual Environment
echo "🐍 Creating virtual environment (venv)..."
python3 -m venv venv
source venv/bin/activate

# 3. Upgrade pip in venv
pip install --upgrade pip

# 4. Auto-detect CUDA Version
echo "🔍 Detecting CUDA capability..."
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep -Po 'CUDA Version: \K[0-9]+\.[0-9]+' | head -n 1)
    echo "Found CUDA version: $CUDA_VERSION"
    
    # Logic to select best PyTorch index URL
    if (( $(echo "$CUDA_VERSION >= 12.4" | bc -l) )); then
        WHL_URL="https://download.pytorch.org/whl/cu124"
    elif (( $(echo "$CUDA_VERSION >= 12.1" | bc -l) )); then
        WHL_URL="https://download.pytorch.org/whl/cu121"
    elif (( $(echo "$CUDA_VERSION >= 11.8" | bc -l) )); then
        WHL_URL="https://download.pytorch.org/whl/cu118"
    else
        echo "⚠️  Older CUDA detected ($CUDA_VERSION). Defaulting to cu118."
        WHL_URL="https://download.pytorch.org/whl/cu118"
    fi
else
    echo "❌ ERROR: nvidia-smi not found. Ensure GPU drivers are installed."
    exit 1
fi

# 5. Install PyTorch with Detected CUDA
echo "🔥 Installing PyTorch from $WHL_URL..."
pip install torch torchvision --index-url "$WHL_URL"

# 6. Install Project Requirements
echo "📚 Installing libraries..."
pip install numpy pandas optuna tqdm requests matplotlib scikit-learn reformer-pytorch

# 7. Verification
echo "✅ Setup Complete. Running verification..."
python -c "
import torch
import sys
print(f'\n--- Environment Report ---')
print(f'Python:           {sys.version.split()[0]}')
print(f'PyTorch:          {torch.__version__}')
print(f'CUDA available:   {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:              {torch.cuda.get_device_name(0)}')
    print(f'VRAM:             {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
    print(f'Active Index:     $WHL_URL')
"

echo "============================================================"
echo "🎉 ALL DONE!"
echo "============================================================"
echo "To use this environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To start training:"
echo "  cd models/diffusion_tsf"
echo "  python test_run.py"
echo "============================================================"