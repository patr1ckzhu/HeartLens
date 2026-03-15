"""
HeartLens interactive demo.

Provides a Gradio interface for uploading ECG files (Apple Watch CSV
or standard 12-lead), running classification, and displaying results
with Grad-CAM visualisation and LLM-generated explanations.

Usage:
    python demo/app.py
"""
import os
import sys
import io
import tempfile

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocess import preprocess_signal
from data.dataset import SUPERCLASS_NAMES, SUPERCLASS_MAP
from models.cnn_lstm import CNNLSTM
from evaluation.gradcam import GradCAM1D, plot_ecg_with_gradcam
from llm.explain import generate_explanation


def load_apple_watch_ecg(csv_path: str) -> tuple[np.ndarray, float]:
    """Parse Apple Watch ECG export CSV.

    Apple Watch exports contain a metadata header followed by single-lead
    voltage samples (one per line) in microvolts at 512 Hz.

    Returns:
        Tuple of (signal array of shape (samples,), sampling rate).
    """
    with open(csv_path) as f:
        lines = f.readlines()

    # Find where the numeric data starts (after metadata rows)
    data_start = 0
    sample_rate = 512.0
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("Sample Rate"):
            parts = line.split(",")
            sample_rate = float(parts[1].split()[0])
        # First line that's a plain number marks the data start
        try:
            float(line)
            data_start = i
            break
        except ValueError:
            continue

    values = []
    for line in lines[data_start:]:
        line = line.strip()
        if line:
            try:
                values.append(float(line))
            except ValueError:
                continue

    signal = np.array(values, dtype=np.float32)
    # Convert from µV to mV for consistency
    signal = signal / 1000.0
    return signal, sample_rate


def prepare_signal_for_model(
    signal: np.ndarray,
    source_fs: float,
    target_fs: float = 500.0,
    target_length: int = 5000,
    single_lead: bool = True,
) -> np.ndarray:
    """Resample and pad/crop a signal to match model input requirements.

    Args:
        signal: 1D signal array.
        source_fs: Original sampling frequency.
        target_fs: Model's expected sampling frequency.
        target_length: Expected number of samples.
        single_lead: If True, return shape (time, 1).

    Returns:
        Preprocessed signal ready for model input.
    """
    from scipy.signal import resample

    # Resample if frequencies differ
    if abs(source_fs - target_fs) > 1.0:
        num_samples = int(len(signal) * target_fs / source_fs)
        signal = resample(signal, num_samples)

    # Take centre crop or zero-pad to target length
    if len(signal) > target_length:
        start = (len(signal) - target_length) // 2
        signal = signal[start : start + target_length]
    elif len(signal) < target_length:
        pad_total = target_length - len(signal)
        pad_left = pad_total // 2
        signal = np.pad(signal, (pad_left, pad_total - pad_left))

    signal = signal.reshape(-1, 1) if single_lead else signal
    signal = preprocess_signal(signal, fs=target_fs)
    return signal


def create_demo(
    checkpoint_path: str = "results/best_model_cnn_lstm_12_lead.pt",
    single_lead_checkpoint: str = "results/best_model_single_lead.pt",
):
    """Build and launch the Gradio demo interface."""
    import gradio as gr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load single-lead model (for Apple Watch ECGs)
    sl_model = CNNLSTM(in_channels=1, num_classes=5).to(device)
    if os.path.exists(single_lead_checkpoint):
        sd = torch.load(single_lead_checkpoint, map_location=device, weights_only=True)
        sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
        sl_model.load_state_dict(sd)
    sl_model.eval()

    target_layer = sl_model.cnn[-2]
    gradcam = GradCAM1D(sl_model, target_layer)

    def analyse_ecg(file):
        if file is None:
            return None, "Please upload an ECG file."

        file_path = file.name if hasattr(file, "name") else file

        try:
            signal, fs = load_apple_watch_ecg(file_path)
        except Exception as e:
            return None, f"Error reading file: {e}"

        # Prepare for model
        processed = prepare_signal_for_model(signal, fs, single_lead=True)
        tensor = torch.from_numpy(processed).permute(1, 0).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            logits = sl_model(tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        pred_probs = {name: float(probs[i]) for i, name in enumerate(SUPERCLASS_NAMES)}

        # Grad-CAM for top predicted class
        top_class = max(pred_probs, key=pred_probs.get)
        cam = gradcam.generate(tensor, SUPERCLASS_MAP[top_class])

        # Plot
        fig = plot_ecg_with_gradcam(
            signal=processed,
            cam=cam,
            pred_probs=pred_probs,
            lead_idx=0,
            fs=500.0,
            title=f"HeartLens Analysis (Lead I) — Top: {top_class}",
        )

        # Replot for Gradio (need to return as figure)
        fig2 = plot_ecg_with_gradcam(
            signal=processed, cam=cam, pred_probs=pred_probs,
            lead_idx=0, fs=500.0,
            title=f"HeartLens Analysis (Lead I) — Top: {top_class}",
        )

        # Save to temp file for display
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        fig2.savefig(tmp.name, dpi=150, bbox_inches="tight")
        plt.close(fig2)

        # LLM explanation
        explanation = generate_explanation(pred_probs, cam, fs=500.0)

        return tmp.name, explanation

    demo = gr.Interface(
        fn=analyse_ecg,
        inputs=gr.File(label="Upload ECG (Apple Watch CSV)"),
        outputs=[
            gr.Image(label="ECG with Grad-CAM Attention"),
            gr.Textbox(label="AI Explanation", lines=12),
        ],
        title="HeartLens — AI ECG Screening Assistant",
        description=(
            "Upload an Apple Watch ECG export (CSV) for automated cardiac "
            "screening. The system classifies abnormalities using a CNN-LSTM "
            "model and highlights diagnostically relevant regions."
        ),
    )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(share=False)
