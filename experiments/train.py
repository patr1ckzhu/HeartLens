"""
Training script for ECG classification models.

Usage:
    python experiments/train.py --config configs/default.yaml
    python experiments/train.py --config configs/default.yaml --single-lead
"""
import os
import sys
import argparse
import time

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import build_datasets, SUPERCLASS_NAMES, NUM_SUPERCLASSES
from models.cnn_lstm import CNNLSTM


def compute_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute multi-label classification metrics.

    Args:
        labels: Ground truth, shape (N, num_classes).
        probs: Predicted probabilities, shape (N, num_classes).
        threshold: Decision threshold for F1 calculation.

    Returns:
        Dict with macro AUC, per-class AUC, and macro F1.
    """
    metrics = {}

    # Per-class AUC (skip classes with no positive samples in batch)
    per_class_auc = []
    for i, name in enumerate(SUPERCLASS_NAMES):
        if labels[:, i].sum() > 0 and labels[:, i].sum() < len(labels):
            auc = roc_auc_score(labels[:, i], probs[:, i])
            metrics[f"auc_{name}"] = auc
            per_class_auc.append(auc)

    metrics["macro_auc"] = np.mean(per_class_auc) if per_class_auc else 0.0

    preds = (probs >= threshold).astype(np.float32)
    metrics["macro_f1"] = f1_score(labels, preds, average="macro", zero_division=0)

    return metrics


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0

    for signals, labels in loader:
        signals = signals.to(device)
        labels = labels.to(device)

        logits = model(signals)
        loss = criterion(logits, labels)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        total_loss += loss.item() * signals.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict]:
    """Evaluate model on a dataset. Returns loss and metrics dict."""
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs = []

    for signals, labels in loader:
        signals = signals.to(device)
        labels = labels.to(device)

        logits = model(signals)
        loss = criterion(logits, labels)

        total_loss += loss.item() * signals.size(0)
        all_labels.append(labels.cpu().numpy())
        all_probs.append(torch.sigmoid(logits).cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(all_labels, all_probs)

    return avg_loss, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--single-lead", action="store_true",
                        help="Train with Lead I only (Apple Watch simulation)")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Override single-lead from command line
    single_lead = args.single_lead or cfg["data"]["single_lead"]

    # Device selection
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Build datasets
    print("Loading PTB-XL dataset ...")
    train_ds, val_ds, test_ds = build_datasets(
        data_dir=cfg["data"]["data_dir"],
        sampling_rate=cfg["data"]["sampling_rate"],
        single_lead=single_lead,
        cache_dir=cfg["data"]["cache_dir"],
    )
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True,
        num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False,
        num_workers=4, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg["training"]["batch_size"], shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # Build model
    in_channels = 1 if single_lead else cfg["model"]["in_channels"]
    model = CNNLSTM(
        in_channels=in_channels,
        num_classes=cfg["model"]["num_classes"],
        cnn_channels=cfg["model"]["cnn_channels"],
        cnn_kernels=cfg["model"]["cnn_kernels"],
        lstm_hidden=cfg["model"]["lstm_hidden"],
        lstm_layers=cfg["model"]["lstm_layers"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Multi-label classification with class-frequency weighting to
    # mitigate the NORM-dominated class imbalance
    train_label_freq = train_ds.labels.mean(axis=0)
    pos_weight = torch.tensor(
        (1 - train_label_freq) / (train_label_freq + 1e-8),
        dtype=torch.float32,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=cfg["training"]["epochs"],
    )

    # Training loop
    save_dir = cfg["output"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    tag = "single_lead" if single_lead else "12_lead"

    best_auc = 0.0
    patience_counter = 0
    patience = cfg["training"]["early_stopping_patience"]

    print(f"\nStarting training ({tag}) ...")
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        t0 = time.time()

        train_loss = train_epoch(model, train_loader, criterion, optimiser, device)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr = optimiser.param_groups[0]["lr"]
        auc = val_metrics["macro_auc"]

        print(
            f"Epoch {epoch:3d}/{cfg['training']['epochs']} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_auc={auc:.4f} | lr={lr:.2e} | {elapsed:.1f}s"
        )

        if auc > best_auc:
            best_auc = auc
            patience_counter = 0
            ckpt_path = os.path.join(save_dir, f"best_model_{tag}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  → Saved best model (AUC={best_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    # Test evaluation
    print("\n--- Test Set Evaluation ---")
    model.load_state_dict(torch.load(os.path.join(save_dir, f"best_model_{tag}.pt"), weights_only=True))
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)

    print(f"Test loss: {test_loss:.4f}")
    print(f"Macro AUC: {test_metrics['macro_auc']:.4f}")
    print(f"Macro F1:  {test_metrics['macro_f1']:.4f}")
    for name in SUPERCLASS_NAMES:
        key = f"auc_{name}"
        if key in test_metrics:
            print(f"  {name:5s}: AUC = {test_metrics[key]:.4f}")

    # Save test results
    results_path = os.path.join(save_dir, f"test_results_{tag}.yaml")
    with open(results_path, "w") as f:
        yaml.dump(test_metrics, f, default_flow_style=False)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
