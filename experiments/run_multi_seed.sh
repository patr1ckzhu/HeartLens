#!/bin/bash
# Run the main CNN-LSTM experiment with 3 seeds for mean ± std reporting.
# Usage: bash experiments/run_multi_seed.sh

set -e

echo "=== Multi-seed CNN-LSTM (5-superclass, 12-lead) ==="
for seed in 42 123 456; do
    echo ""
    echo "--- Seed ${seed} ---"
    python experiments/train.py --model cnn_lstm --seed ${seed}
    # Rename results to include seed
    mv results/test_results_cnn_lstm_superclass_12_lead.yaml \
       results/test_results_cnn_lstm_superclass_12_lead_seed${seed}.yaml
    mv results/best_model_cnn_lstm_superclass_12_lead.pt \
       results/best_model_cnn_lstm_superclass_12_lead_seed${seed}.pt
done

echo ""
echo "=== All seeds complete ==="

# Summarise results
python -c "
import yaml, numpy as np
seeds = [42, 123, 456]
aucs = []
for s in seeds:
    with open(f'results/test_results_cnn_lstm_superclass_12_lead_seed{s}.yaml') as f:
        r = yaml.safe_load(f)
        auc = np.mean([v for k, v in r.items() if k.startswith('auc_')])
        aucs.append(auc)
        print(f'  Seed {s}: macro AUC = {auc:.4f}')
print(f'  Mean ± Std: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}')
"
