"""
Generate ROC curves and confusion matrix for the report.

Usage:
    python evaluation/plot_roc_cm.py --checkpoint results/best_model_cnn_lstm_superclass_12_lead_seed42.pt
"""
import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, multilabel_confusion_matrix
import torch
from torch.utils.data import DataLoader
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import build_datasets, SUPERCLASS_NAMES, NUM_SUPERCLASSES
from models.cnn_lstm import CNNLSTM

matplotlib.rcParams.update({"font.size": 11, "font.family": "serif"})

COLORS = ["#2196f3", "#f44336", "#ff9800", "#4caf50", "#9c27b0"]


@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_labels, all_probs = [], []
    for signals, labels in loader:
        signals = signals.to(device)
        logits = model(signals)
        all_labels.append(labels.numpy())
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(all_labels), np.concatenate(all_probs)


def plot_roc_curves(y_true, y_prob, save_path):
    """Plot per-class ROC curves on a single figure."""
    fig, ax = plt.subplots(figsize=(6, 5.5))

    for i, (name, color) in enumerate(zip(SUPERCLASS_NAMES, COLORS)):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=1.8,
                label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Per-Class ROC Curves (CNN-LSTM, 12-Lead)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_confusion_matrix(y_true, y_prob, threshold, save_path):
    """Plot multi-label confusion matrices as a single row of heatmaps."""
    y_pred = (y_prob >= threshold).astype(int)
    mcm = multilabel_confusion_matrix(y_true, y_pred)

    fig, axes = plt.subplots(1, 5, figsize=(14, 2.8))

    for i, (name, cm, ax) in enumerate(zip(SUPERCLASS_NAMES, mcm, axes)):
        # cm is 2x2: [[TN, FP], [FN, TP]]
        tn, fp, fn, tp = cm.ravel()
        total = tn + fp + fn + tp

        # Normalise by row (true label)
        cm_norm = np.array([[tn / (tn + fp), fp / (tn + fp)],
                            [fn / (fn + tp), tp / (fn + tp)]])

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="equal")

        # Annotate with counts and percentages
        for row in range(2):
            for col in range(2):
                count = cm[row, col]
                pct = cm_norm[row, col]
                color = "white" if pct > 0.6 else "black"
                ax.text(col, row, f"{count}\n({pct:.0%})",
                        ha="center", va="center", fontsize=8, color=color)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Neg", "Pos"], fontsize=8)
        ax.set_yticklabels(["Neg", "Pos"], fontsize=8)
        ax.set_title(name, fontsize=10, fontweight="bold")

        if i == 0:
            ax.set_ylabel("True", fontsize=9)
        ax.set_xlabel("Predicted", fontsize=9)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",
                        default="results/best_model_cnn_lstm_superclass_12_lead_seed42.pt")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default="report/figures")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNNLSTM(in_channels=12, num_classes=NUM_SUPERCLASSES).to(device)
    sd = torch.load(args.checkpoint, map_location=device, weights_only=True)
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    model.load_state_dict(sd)

    _, _, test_ds = build_datasets(
        data_dir=cfg["data"]["data_dir"],
        sampling_rate=cfg["data"]["sampling_rate"],
        cache_dir=cfg["data"]["cache_dir"],
    )
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)

    y_true, y_prob = get_predictions(model, test_loader, device)

    os.makedirs(args.output_dir, exist_ok=True)
    plot_roc_curves(y_true, y_prob,
                    os.path.join(args.output_dir, "roc_curves.pdf"))
    plot_confusion_matrix(y_true, y_prob, threshold=0.5,
                          os.path.join(args.output_dir, "confusion_matrix.pdf"))


if __name__ == "__main__":
    main()
