"""
PTB-XL dataset loader and PyTorch Dataset.

Handles label parsing (SCP codes to superclass/subclass mapping),
official stratified splits, and optional caching of preprocessed signals.
"""
import os
import ast
from typing import Literal

import numpy as np
import pandas as pd
import wfdb
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from data.preprocess import preprocess_signal


# PTB-XL diagnostic superclass mapping
SUPERCLASS_MAP = {
    "NORM": 0,
    "MI": 1,
    "STTC": 2,
    "CD": 3,
    "HYP": 4,
}

SUPERCLASS_NAMES = list(SUPERCLASS_MAP.keys())
NUM_SUPERCLASSES = len(SUPERCLASS_MAP)


def load_ptbxl_metadata(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load PTB-XL metadata and SCP statement descriptions.

    Args:
        data_dir: Path to the extracted PTB-XL directory.

    Returns:
        Tuple of (record metadata DataFrame, SCP statements DataFrame).
    """
    meta = pd.read_csv(
        os.path.join(data_dir, "ptbxl_database.csv"),
        index_col="ecg_id",
    )
    meta.scp_codes = meta.scp_codes.apply(ast.literal_eval)

    scp = pd.read_csv(os.path.join(data_dir, "scp_statements.csv"), index_col=0)
    # Only keep diagnostic statements for superclass mapping
    scp = scp[scp.diagnostic == 1]

    return meta, scp


def encode_superclass_labels(
    scp_codes: dict,
    scp_df: pd.DataFrame,
) -> np.ndarray:
    """Convert SCP code dict to a multi-hot superclass vector.

    Each SCP code maps to a diagnostic_class (superclass). We aggregate
    all codes present in the record and set the corresponding bits.

    Args:
        scp_codes: Dict of {scp_code: likelihood}, e.g. {'NORM': 100.0}.
        scp_df: SCP statements DataFrame with diagnostic_class column.

    Returns:
        Binary vector of shape (NUM_SUPERCLASSES,).
    """
    label = np.zeros(NUM_SUPERCLASSES, dtype=np.float32)
    for code, likelihood in scp_codes.items():
        if code in scp_df.index and likelihood > 0:
            superclass = scp_df.loc[code].diagnostic_class
            if superclass in SUPERCLASS_MAP:
                label[SUPERCLASS_MAP[superclass]] = 1.0
    return label


def load_signals(
    meta: pd.DataFrame,
    data_dir: str,
    sampling_rate: int = 500,
) -> np.ndarray:
    """Load all ECG waveforms into a single numpy array.

    Args:
        meta: PTB-XL metadata DataFrame.
        data_dir: Path to the extracted PTB-XL directory.
        sampling_rate: Which version to load (100 or 500 Hz).

    Returns:
        Array of shape (num_records, time_steps, 12).
    """
    col = "filename_hr" if sampling_rate == 500 else "filename_lr"
    signals = []
    for fname in tqdm(meta[col], desc="Loading signals"):
        sig, _ = wfdb.rdsamp(os.path.join(data_dir, fname))
        signals.append(sig)
    return np.array(signals, dtype=np.float32)


class PTBXLDataset(Dataset):
    """PyTorch dataset for PTB-XL ECG records.

    Loads preprocessed signals and multi-hot superclass labels.
    Supports 12-lead and single-lead (Lead I) modes.

    Args:
        signals: Preprocessed ECG signals, shape (N, time, 12).
        labels: Multi-hot label array, shape (N, num_classes).
        single_lead: If True, only use Lead I (index 0) to simulate
            consumer-grade devices like Apple Watch.
    """

    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        single_lead: bool = False,
        augment: bool = False,
    ):
        self.signals = signals
        self.labels = labels
        self.single_lead = single_lead
        self.augment = augment

    def __len__(self) -> int:
        return len(self.signals)

    def _apply_augmentation(self, sig: np.ndarray) -> np.ndarray:
        """Apply random augmentations to simulate real-world signal variation.

        Augmentations are mild enough to preserve diagnostic morphology
        while improving generalisation to unseen recording conditions.
        """
        # Gaussian noise (simulates electrode noise)
        if np.random.random() < 0.5:
            sig = sig + np.random.normal(0, 0.05, sig.shape).astype(np.float32)

        # Amplitude scaling (simulates gain variation across devices)
        if np.random.random() < 0.5:
            scale = np.random.uniform(0.8, 1.2)
            sig = sig * scale

        # Baseline wander (low-frequency sinusoidal drift)
        if np.random.random() < 0.3:
            t = np.arange(sig.shape[0], dtype=np.float32)
            freq = np.random.uniform(0.1, 0.5)
            amplitude = np.random.uniform(0.0, 0.1)
            wander = amplitude * np.sin(2 * np.pi * freq * t / sig.shape[0])
            sig = sig + wander[:, np.newaxis]

        # Random temporal crop and pad back (slight phase shift)
        if np.random.random() < 0.3:
            shift = np.random.randint(-50, 50)
            sig = np.roll(sig, shift, axis=0)

        return sig

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sig = self.signals[idx].copy()  # (time, leads)

        if self.augment:
            sig = self._apply_augmentation(sig)

        if self.single_lead:
            sig = sig[:, 0:1]

        sig = torch.from_numpy(sig).permute(1, 0)  # (leads, time)
        label = torch.from_numpy(self.labels[idx])
        return sig, label


def build_datasets(
    data_dir: str,
    sampling_rate: int = 500,
    single_lead: bool = False,
    cache_dir: str | None = None,
) -> tuple[PTBXLDataset, PTBXLDataset, PTBXLDataset]:
    """Build train, validation, and test datasets from PTB-XL.

    Uses the official stratified folds: 1-8 for training, 9 for
    validation, 10 for testing (following Wagner et al. 2020).

    Args:
        data_dir: Path to extracted PTB-XL directory.
        sampling_rate: 100 or 500 Hz.
        single_lead: Whether to use only Lead I.
        cache_dir: If set, cache preprocessed data to disk for faster
            subsequent loading.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"ptbxl_{sampling_rate}hz.npz")

    # Try loading from cache
    if cache_path and os.path.exists(cache_path):
        print(f"Loading preprocessed data from cache: {cache_path}")
        cached = np.load(cache_path)
        signals = cached["signals"]
        labels = cached["labels"]
        folds = cached["folds"]
    else:
        meta, scp_df = load_ptbxl_metadata(data_dir)
        signals = load_signals(meta, data_dir, sampling_rate)
        labels = np.array([
            encode_superclass_labels(codes, scp_df)
            for codes in meta.scp_codes
        ])
        folds = meta.strat_fold.values

        # Preprocess all signals
        print("Preprocessing signals ...")
        fs = float(sampling_rate)
        for i in tqdm(range(len(signals)), desc="Filtering"):
            signals[i] = preprocess_signal(signals[i], fs=fs)

        if cache_path:
            np.savez_compressed(cache_path, signals=signals, labels=labels, folds=folds)
            print(f"Cached preprocessed data to {cache_path}")

    # Official PTB-XL splits
    train_mask = folds <= 8
    val_mask = folds == 9
    test_mask = folds == 10

    return (
        PTBXLDataset(signals[train_mask], labels[train_mask], single_lead, augment=True),
        PTBXLDataset(signals[val_mask], labels[val_mask], single_lead, augment=False),
        PTBXLDataset(signals[test_mask], labels[test_mask], single_lead, augment=False),
    )
