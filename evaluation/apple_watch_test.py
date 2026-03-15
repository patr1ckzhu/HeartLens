"""
Apple Watch ECG analysis script.

Loads Apple Watch ECG exports, runs them through the single-lead model,
generates Grad-CAM visualisations, and produces LLM explanations. Output
figures and text are saved for inclusion in the report.

Usage:
    python evaluation/apple_watch_test.py --ecg-dir /path/to/apple_health_export/electrocardiograms
"""
import os
import sys
import glob
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo.app import load_apple_watch_ecg, prepare_signal_for_model
from data.dataset import SUPERCLASS_NAMES, SUPERCLASS_MAP
from models.cnn_lstm import CNNLSTM
from evaluation.gradcam import GradCAM1D, plot_ecg_with_gradcam
from llm.explain import generate_explanation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ecg-dir",
        default=os.path.expanduser("~/apple_health_export/electrocardiograms"),
    )
    parser.add_argument("--checkpoint", default="results/best_model_single_lead.pt")
    parser.add_argument("--output-dir", default="results/apple_watch")
    parser.add_argument("--max-files", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cpu")  # Apple Watch analysis runs on CPU

    # Load single-lead model
    model = CNNLSTM(in_channels=1, num_classes=5).to(device)
    if os.path.exists(args.checkpoint):
        sd = torch.load(args.checkpoint, map_location=device, weights_only=True)
        sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
        model.load_state_dict(sd)
    else:
        print(f"Warning: checkpoint {args.checkpoint} not found, using random weights")
    model.eval()

    target_layer = model.cnn[-2]
    gradcam = GradCAM1D(model, target_layer)

    # Find ECG files
    csv_files = sorted(glob.glob(os.path.join(args.ecg_dir, "ecg_*.csv")))
    if not csv_files:
        print(f"No ECG files found in {args.ecg_dir}")
        return

    csv_files = csv_files[: args.max_files]
    print(f"Processing {len(csv_files)} Apple Watch ECG files ...\n")

    for csv_path in csv_files:
        fname = os.path.basename(csv_path).replace(".csv", "")
        print(f"--- {fname} ---")

        # Load and preprocess
        signal, fs = load_apple_watch_ecg(csv_path)
        processed = prepare_signal_for_model(signal, fs, single_lead=True)
        tensor = torch.from_numpy(processed).permute(1, 0).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        pred_probs = {name: float(probs[i]) for i, name in enumerate(SUPERCLASS_NAMES)}

        # Print results
        for name, prob in sorted(pred_probs.items(), key=lambda x: -x[1]):
            marker = "***" if prob > 0.5 else "   "
            print(f"  {marker} {name}: {prob:.3f}")

        # Grad-CAM for top class
        top_class = max(pred_probs, key=pred_probs.get)
        cam = gradcam.generate(tensor, SUPERCLASS_MAP[top_class])

        # Save figure
        save_path = os.path.join(args.output_dir, f"{fname}_gradcam.png")
        plot_ecg_with_gradcam(
            signal=processed,
            cam=cam,
            pred_probs=pred_probs,
            lead_idx=0,
            fs=500.0,
            title=f"Apple Watch ECG: {fname}",
            save_path=save_path,
        )

        # LLM explanation
        explanation = generate_explanation(pred_probs, cam, fs=500.0)
        txt_path = os.path.join(args.output_dir, f"{fname}_explanation.txt")
        with open(txt_path, "w") as f:
            f.write(explanation)

        print()

    print(f"Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
