#!/usr/bin/env bash
# Script to run full test set evaluation on all datasets
#
# Usage:
#   ./run_eval.sh              # Fast mode: stride=8 (default)
#   ./run_eval.sh --stride 1   # Full mode: all test samples
#   ./run_eval.sh --dry-run    # Quick test

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

# Activate venv if present
if [[ -f "${REPO_ROOT}/venv/bin/activate" ]]; then
  source "${REPO_ROOT}/venv/bin/activate"
fi

echo "============================================================================="
echo "  FULL TEST SET EVALUATION: DiffusionTSF"
echo "============================================================================="
echo "Metrics: MSE, MAE, Gradient MAE (local patterns), Gradient Correlation"
echo ""
echo "Options: $@"
echo ""

# Pass all arguments to evaluate_all.py
python3 evaluate_all.py "$@"

echo ""
echo "Evaluation complete."

