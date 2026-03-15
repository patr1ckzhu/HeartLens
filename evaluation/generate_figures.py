"""
Generate Grad-CAM visualisation figures for the report.

Loads the trained model and produces ECG plots with attention overlays
for representative samples from each diagnostic class.

Usage:
    python evaluation/generate_figures.py --checkpoint results/best_model_cnn_lstm_12_lead.pt
"""
import os
import sys
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import build_datasets, SUPERCLASS_NAMES, SUPERCLASS_MAP
from models.cnn_lstm import CNNLSTM
from evaluation.gradcam import GradCAM1D, plot_ecg_with_gradcam
from llm.explain import generate_explanation, identify_gradcam_regions


def find_representative_samples(dataset, num_per_class: int = 2) -> dict[str, list[int]]:
    """Find indices of high-confidence samples for each class.

    Selects samples where a single class dominates the label vector,
    which makes the Grad-CAM visualisation most interpretable.
    """
    samples = {name: [] for name in SUPERCLASS_NAMES}
    labels = dataset.labels

    for cls_idx, cls_name in enumerate(SUPERCLASS_NAMES):
        # Samples where this class is positive
        positive = np.where(labels[:, cls_idx] == 1)[0]
        # Prefer samples with fewer co-occurring labels (cleaner examples)
        label_counts = labels[positive].sum(axis=1)
        sorted_idx = positive[np.argsort(label_counts)]
        samples[cls_name] = sorted_idx[:num_per_class].tolist()

    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="results/best_model_cnn_lstm_12_lead.pt")
    parser.add_argument("--data-dir", default="data/raw/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")
    parser.add_argument("--cache-dir", default="data/cached")
    parser.add_argument("--output-dir", default="results/figures")
    parser.add_argument("--num-per-class", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model (without torch.compile for hook compatibility)
    model = CNNLSTM(in_channels=12, num_classes=5).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    # Strip '_orig_mod.' prefix added by torch.compile
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    # Hook into the last residual conv block
    target_layer = model.cnn[-2]  # Last ResConvBlock before final pool
    gradcam = GradCAM1D(model, target_layer)

    # Load test data
    _, _, test_ds = build_datasets(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
    )

    samples = find_representative_samples(test_ds, args.num_per_class)

    for cls_name, indices in samples.items():
        for rank, idx in enumerate(indices):
            sig_tensor, label = test_ds[idx]
            sig_tensor = sig_tensor.unsqueeze(0).to(device)

            # Model prediction
            with torch.no_grad():
                logits = model(sig_tensor)
                probs = torch.sigmoid(logits).cpu().numpy()[0]

            pred_probs = {name: float(probs[i]) for i, name in enumerate(SUPERCLASS_NAMES)}

            # Grad-CAM for the target class
            cls_idx = SUPERCLASS_MAP[cls_name]
            cam = gradcam.generate(sig_tensor, cls_idx)

            # Original signal for plotting (un-transposed)
            signal = test_ds.signals[idx]  # (time, leads)

            save_path = os.path.join(args.output_dir, f"gradcam_{cls_name}_{rank}.png")
            plot_ecg_with_gradcam(
                signal=signal,
                cam=cam,
                pred_probs=pred_probs,
                lead_idx=0,
                title=f"Grad-CAM: {cls_name} (Lead I)",
                save_path=save_path,
            )

            # Generate LLM explanation for the first sample of each class
            if rank == 0:
                explanation = generate_explanation(pred_probs, cam)
                txt_path = os.path.join(args.output_dir, f"explanation_{cls_name}.txt")
                with open(txt_path, "w") as f:
                    f.write(explanation)
                print(f"  Explanation saved to {txt_path}")

    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
