#!/usr/bin/env bash
# Script to run visualizations on all datasets
#
# Usage:
#   ./run_viz.sh              # Generate 5 samples per dataset
#   ./run_viz.sh --samples 10 # Generate 10 samples per dataset

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

# Activate venv if present
if [[ -f "${REPO_ROOT}/venv/bin/activate" ]]; then
  source "${REPO_ROOT}/venv/bin/activate"
fi

echo "============================================================================="
echo "  VISUALIZATION: DiffusionTSF"
echo "============================================================================="
echo "Generating sample forecasts with diffusion vs iTransformer comparison"
echo ""

# Pass all arguments to run_visualizations.py
python3 run_visualizations.py "$@"

echo ""
echo "Visualizations complete."
echo "Output saved to: ${REPO_ROOT}/models/diffusion_tsf/final_visualizations/"

