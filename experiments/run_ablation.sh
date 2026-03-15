#!/bin/bash
# Run all ablation experiments sequentially.
# Usage: bash experiments/run_ablation.sh

set -e

echo "=== Ablation Study ==="

for model in cnn_only lstm_only transformer; do
    echo ""
    echo "--- Training: ${model} ---"
    python experiments/train.py --model ${model}
    echo "--- Done: ${model} ---"
done

echo ""
echo "=== All ablation experiments complete ==="
echo "Results:"
ls -1 results/test_results_*.yaml
