"""
Comprehensive evaluation with bootstrap confidence intervals and
multiple metrics. Operates on saved model checkpoints without
retraining.

Usage:
    python evaluation/robust_eval.py --checkpoint results/best_model_cnn_lstm_12_lead.pt
    python evaluation/robust_eval.py --checkpoint results/best_model_cnn_lstm_12_lead.pt --task subclass
"""
import os
import sys
import argparse

import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    average_precision_score,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import (
    build_datasets, SUPERCLASS_NAMES, SUBCLASS_NAMES,
    NUM_SUPERCLASSES, NUM_SUBCLASSES,
)
from models.cnn_lstm import CNNLSTM


def bootstrap_metric(y_true, y_score, metric_fn, n_bootstrap=1000, seed=42):
    """Compute a metric with 95% bootstrap confidence interval.

    Args:
        y_true: Ground truth labels.
        y_score: Predicted scores/probabilities.
        metric_fn: Callable(y_true, y_score) -> float.
        n_bootstrap: Number of bootstrap iterations.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (mean, ci_lower, ci_upper).
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        try:
            s = metric_fn(y_true[idx], y_score[idx])
            scores.append(s)
        except ValueError:
            continue

    scores = np.array(scores)
    mean = np.mean(scores)
    ci_lo = np.percentile(scores, 2.5)
    ci_hi = np.percentile(scores, 97.5)
    return mean, ci_lo, ci_hi


def compute_full_metrics(y_true, y_prob, class_names, threshold=0.5):
    """Compute AUC, PR-AUC, F1, Precision, Recall per class and macro."""
    y_pred = (y_prob >= threshold).astype(np.float32)
    results = {}

    per_class_auc = []
    per_class_prauc = []

    for i, name in enumerate(class_names):
        if y_true[:, i].sum() > 0 and y_true[:, i].sum() < len(y_true):
            auc_mean, auc_lo, auc_hi = bootstrap_metric(
                y_true[:, i], y_prob[:, i], roc_auc_score
            )
            prauc = average_precision_score(y_true[:, i], y_prob[:, i])

            results[name] = {
                "auc": f"{auc_mean:.4f} ({auc_lo:.4f}-{auc_hi:.4f})",
                "pr_auc": round(float(prauc), 4),
                "f1": round(float(f1_score(y_true[:, i], y_pred[:, i], zero_division=0)), 4),
                "precision": round(float(precision_score(y_true[:, i], y_pred[:, i], zero_division=0)), 4),
                "recall": round(float(recall_score(y_true[:, i], y_pred[:, i], zero_division=0)), 4),
                "support": int(y_true[:, i].sum()),
            }
            per_class_auc.append(auc_mean)
            per_class_prauc.append(prauc)

    # Macro averages with bootstrap CI
    def macro_auc_fn(yt, yp):
        aucs = []
        for i in range(yt.shape[1]):
            if yt[:, i].sum() > 0 and yt[:, i].sum() < len(yt):
                aucs.append(roc_auc_score(yt[:, i], yp[:, i]))
        return np.mean(aucs) if aucs else 0.0

    macro_mean, macro_lo, macro_hi = bootstrap_metric(
        y_true, y_prob, macro_auc_fn
    )

    # Sample-weighted AUC (weighted by class prevalence)
    weighted_auc = roc_auc_score(y_true, y_prob, average="weighted")

    results["_macro"] = {
        "auc": f"{macro_mean:.4f} ({macro_lo:.4f}-{macro_hi:.4f})",
        "auc_weighted": round(float(weighted_auc), 4),
        "pr_auc": round(float(np.mean(per_class_prauc)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "precision_macro": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
    }

    return results


@torch.no_grad()
def get_predictions(model, loader, device):
    """Run inference and collect all labels and predicted probabilities."""
    model.eval()
    all_labels, all_probs = [], []
    for signals, labels in loader:
        signals = signals.to(device)
        logits = model(signals)
        all_labels.append(labels.numpy())
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(all_labels), np.concatenate(all_probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--task", default="superclass", choices=["superclass", "subclass"])
    parser.add_argument("--single-lead", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.task == "subclass":
        num_classes = NUM_SUBCLASSES
        class_names = SUBCLASS_NAMES
    else:
        num_classes = NUM_SUPERCLASSES
        class_names = SUPERCLASS_NAMES

    in_channels = 1 if args.single_lead else cfg["model"]["in_channels"]

    # Load model
    model = CNNLSTM(in_channels=in_channels, num_classes=num_classes).to(device)
    sd = torch.load(args.checkpoint, map_location=device, weights_only=True)
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    model.load_state_dict(sd)

    # Load test data
    _, _, test_ds = build_datasets(
        data_dir=cfg["data"]["data_dir"],
        sampling_rate=cfg["data"]["sampling_rate"],
        single_lead=args.single_lead,
        task=args.task,
        cache_dir=cfg["data"]["cache_dir"],
    )
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)

    # Get predictions
    y_true, y_prob = get_predictions(model, test_loader, device)

    # Compute comprehensive metrics
    print(f"\n{'='*60}")
    print(f"Comprehensive Evaluation: {args.checkpoint}")
    print(f"Task: {args.task} ({num_classes} classes), Single-lead: {args.single_lead}")
    print(f"Test samples: {len(y_true)}")
    print(f"{'='*60}\n")

    results = compute_full_metrics(y_true, y_prob, class_names)

    # Print per-class results
    print(f"{'Class':<12} {'AUC (95% CI)':<28} {'PR-AUC':<8} {'F1':<8} {'Prec':<8} {'Rec':<8} {'N':<6}")
    print("-" * 78)
    for name in class_names:
        if name in results:
            r = results[name]
            print(f"{name:<12} {r['auc']:<28} {r['pr_auc']:<8} {r['f1']:<8} {r['precision']:<8} {r['recall']:<8} {r['support']:<6}")

    print("-" * 78)
    m = results["_macro"]
    print(f"\nMacro AUC:       {m['auc']}")
    print(f"Weighted AUC:    {m['auc_weighted']}")
    print(f"Macro PR-AUC:    {m['pr_auc']}")
    print(f"Macro F1:        {m['f1_macro']}")
    print(f"Weighted F1:     {m['f1_weighted']}")
    print(f"Macro Precision: {m['precision_macro']}")
    print(f"Macro Recall:    {m['recall_macro']}")

    # Save
    save_dir = cfg["output"]["save_dir"]
    lead_tag = "single_lead" if args.single_lead else "12_lead"
    out_path = os.path.join(save_dir, f"comprehensive_{args.task}_{lead_tag}.yaml")
    with open(out_path, "w") as f:
        yaml.dump(results, f, default_flow_style=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
