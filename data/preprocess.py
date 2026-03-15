"""
ECG signal preprocessing routines.

Implements bandpass filtering and normalisation for 12-lead and single-lead
ECG signals. Filter parameters follow clinical conventions for resting ECG
analysis (0.5–40 Hz passband).
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt


def bandpass_filter(
    signal: np.ndarray,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    fs: float = 500.0,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter.

    This removes baseline wander (< 0.5 Hz) and high-frequency noise
    (> 40 Hz) while preserving diagnostically relevant morphology.

    Args:
        signal: Input signal of shape (time,) or (time, leads).
        lowcut: Lower cutoff frequency in Hz.
        highcut: Upper cutoff frequency in Hz.
        fs: Sampling frequency in Hz.
        order: Butterworth filter order.

    Returns:
        Filtered signal with the same shape as input.
    """
    nyq = 0.5 * fs
    sos = butter(order, [lowcut / nyq, highcut / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, signal, axis=0)


def normalise(signal: np.ndarray) -> np.ndarray:
    """Per-record z-score normalisation.

    Each lead is independently normalised to zero mean and unit variance.
    This accounts for amplitude differences across recording equipment
    and patient physiology.

    Args:
        signal: Shape (time, leads) or (time,).

    Returns:
        Normalised signal.
    """
    mean = np.mean(signal, axis=0, keepdims=True)
    std = np.std(signal, axis=0, keepdims=True)
    # Guard against silent leads (std ≈ 0)
    std = np.where(std < 1e-8, 1.0, std)
    return (signal - mean) / std


def preprocess_signal(
    signal: np.ndarray,
    fs: float = 500.0,
    apply_filter: bool = True,
) -> np.ndarray:
    """Full preprocessing pipeline: bandpass filter then normalise.

    Args:
        signal: Raw ECG, shape (time, leads) or (time,).
        fs: Sampling frequency in Hz.
        apply_filter: Whether to apply bandpass filtering.

    Returns:
        Preprocessed signal as float32.
    """
    if apply_filter:
        signal = bandpass_filter(signal, fs=fs)
    signal = normalise(signal)
    return signal.astype(np.float32)
