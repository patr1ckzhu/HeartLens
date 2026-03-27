"""
Generate additional figures for the report.

Produces ablation comparison, single-lead vs 12-lead comparison,
and class distribution plots.

Usage:
    python evaluation/plot_report_figures.py
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({"font.size": 11, "font.family": "serif"})

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = "report/figures"
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def plot_ablation():
    """Bar chart comparing macro AUC across all models."""
    models = ["Random\nForest", "LSTM-\nonly", "CNN-\nonly", "CNN-LSTM\n(ours)", "CNN-\nTransformer"]
    aucs = [0.861, 0.908, 0.912, 0.914, 0.913]
    colors = ["#9e9e9e", "#7cb4c9", "#7cb4c9", "#1a6fa0", "#7cb4c9"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(models, aucs, color=colors, edgecolor="white", width=0.6)

    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{auc:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Macro AUC")
    ax.set_ylim(0.75, 0.95)
    ax.set_title("Model Comparison on PTB-XL (5 Superclasses)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "ablation_comparison.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_single_vs_12lead():
    """Grouped bar chart: 12-lead vs single-lead per-class AUC."""
    auc_12 = [0.945, 0.928, 0.932, 0.923, 0.834]
    auc_1 = [0.891, 0.780, 0.877, 0.817, 0.795]

    x = np.arange(len(CLASSES))
    width = 0.32

    fig, ax = plt.subplots(figsize=(7, 4))
    bars1 = ax.bar(x - width / 2, auc_12, width, label="12-lead", color="#1a6fa0")
    bars2 = ax.bar(x + width / 2, auc_1, width, label="Single-lead (Lead I)", color="#e07b54")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES)
    ax.set_ylim(0.7, 1.0)
    ax.set_title("Twelve-Lead vs. Single-Lead Classification")
    ax.legend(loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "single_vs_12lead.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_class_distribution():
    """Bar chart showing PTB-XL class imbalance."""
    # Approximate counts from PTB-XL (records can have multiple labels)
    counts = {
        "NORM": 9514,
        "MI": 5469,
        "STTC": 5238,
        "CD": 4898,
        "HYP": 2649,
    }

    fig, ax = plt.subplots(figsize=(6, 3.5))
    colors = ["#4caf50", "#f44336", "#ff9800", "#2196f3", "#9c27b0"]
    bars = ax.bar(counts.keys(), counts.values(), color=colors, edgecolor="white", width=0.55)

    for bar, count in zip(bars, counts.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
                f"{count:,}", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Number of Records")
    ax.set_title("PTB-XL Diagnostic Class Distribution")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "class_distribution.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_ablation()
    plot_single_vs_12lead()
    plot_class_distribution()
    print("Done.")
