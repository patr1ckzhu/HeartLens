"""
Traditional ML baseline for ECG classification.

Extracts handcrafted statistical and morphological features from ECG
signals, then trains a Random Forest classifier. This serves as a
non-deep-learning baseline to quantify the benefit of learned
representations (CNN/LSTM) over engineered features.

Usage:
    python experiments/baseline.py
"""
import os
import sys
import argparse

import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score, f1_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import build_datasets, SUPERCLASS_NAMES


def extract_features(signal: np.ndarray) -> np.ndarray:
    """Extract statistical and morphological features from a single ECG.

    Computes per-lead summary statistics that capture amplitude distribution,
    signal energy, and waveform shape without requiring explicit beat
    segmentation. These are typical features used in pre-deep-learning
    ECG classification pipelines.

    Args:
        signal: ECG signal of shape (time, leads).

    Returns:
        Feature vector of shape (num_features,).
    """
    features = []
    for lead in range(signal.shape[1]):
        s = signal[:, lead]
        features.extend([
            np.mean(s),
            np.std(s),
            np.min(s),
            np.max(s),
            np.max(s) - np.min(s),       # peak-to-peak amplitude
            np.median(s),
            np.percentile(s, 25),
            np.percentile(s, 75),
            np.mean(np.abs(s)),            # mean absolute amplitude
            np.sqrt(np.mean(s ** 2)),      # RMS
            np.sum(s ** 2),                # signal energy
            np.mean(np.diff(s) ** 2),      # mean squared first difference
            np.sum(np.abs(np.diff(np.sign(s)))),  # zero-crossing rate
        ])

    # Inter-lead features (correlations between key lead pairs)
    if signal.shape[1] >= 12:
        for i, j in [(0, 1), (0, 6), (6, 7)]:  # I-II, I-V1, V1-V2
            features.append(np.corrcoef(signal[:, i], signal[:, j])[0, 1])

    return np.array(features, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print("Loading PTB-XL dataset ...")
    train_ds, val_ds, test_ds = build_datasets(
        data_dir=cfg["data"]["data_dir"],
        sampling_rate=cfg["data"]["sampling_rate"],
        single_lead=False,
        cache_dir=cfg["data"]["cache_dir"],
    )

    # Extract features
    print("Extracting features ...")
    X_train = np.array([extract_features(s) for s in tqdm(train_ds.signals)])
    X_val = np.array([extract_features(s) for s in tqdm(val_ds.signals)])
    X_test = np.array([extract_features(s) for s in tqdm(test_ds.signals)])

    y_train = train_ds.labels
    y_val = val_ds.labels
    y_test = test_ds.labels

    # Train Random Forest (one-vs-rest for multi-label)
    print("\nTraining Random Forest baseline ...")
    rf = OneVsRestClassifier(
        RandomForestClassifier(n_estimators=300, max_depth=20, n_jobs=-1, random_state=42),
    )
    rf.fit(X_train, y_train)

    # Evaluate
    rf_probs = rf.predict_proba(X_test)
    rf_preds = rf.predict(X_test)

    print("\n--- Random Forest Results ---")
    per_class_auc = []
    for i, name in enumerate(SUPERCLASS_NAMES):
        if y_test[:, i].sum() > 0:
            auc = roc_auc_score(y_test[:, i], rf_probs[:, i])
            print(f"  {name:5s}: AUC = {auc:.4f}")
            per_class_auc.append(auc)
    macro_auc = np.mean(per_class_auc)
    macro_f1 = f1_score(y_test, rf_preds, average="macro", zero_division=0)
    print(f"  Macro AUC: {macro_auc:.4f}")
    print(f"  Macro F1:  {macro_f1:.4f}")

    # Save results
    save_dir = cfg["output"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    results = {f"auc_{name}": float(per_class_auc[i]) for i, name in enumerate(SUPERCLASS_NAMES)}
    results["macro_auc"] = float(macro_auc)
    results["macro_f1"] = float(macro_f1)

    results_path = os.path.join(save_dir, "test_results_random_forest_baseline.yaml")
    with open(results_path, "w") as f:
        yaml.dump(results, f, default_flow_style=False)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
