"""
Grad-CAM for 1D ECG signals.

Generates class-discriminative heatmaps that highlight which temporal
regions of the ECG the model relies on for each diagnostic prediction.
This provides clinical interpretability by linking model decisions to
recognisable waveform morphology (e.g. ST-segment, QRS complex).
"""
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


class GradCAM1D:
    """Gradient-weighted Class Activation Mapping for 1D convolutional models.

    Hooks into a target convolutional layer and uses the gradient signal
    flowing back from a target class to weight the layer's feature maps.
    The resulting heatmap indicates which time steps influenced the
    prediction most.

    Args:
        model: Trained classification model (must not be torch.compiled).
        target_layer: The nn.Module to hook into (typically the last conv
            block in the CNN encoder).
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    @torch.enable_grad()
    def generate(
        self,
        x: torch.Tensor,
        target_class: int,
    ) -> np.ndarray:
        """Compute Grad-CAM heatmap for a single input.

        Args:
            x: Input ECG tensor, shape (1, leads, time).
            target_class: Index of the class to explain.

        Returns:
            Heatmap array of shape (time,), normalised to [0, 1].
        """
        x = x.clone().requires_grad_(True)
        output = self.model(x)

        self.model.zero_grad()
        output[0, target_class].backward()

        # Channel-wise importance weights via global average pooling
        weights = self.gradients.mean(dim=-1, keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = F.relu(cam)

        # Interpolate back to original signal length
        cam = F.interpolate(
            cam.unsqueeze(1),
            size=x.shape[-1],
            mode="linear",
            align_corners=False,
        ).squeeze()

        cam = cam.cpu().numpy()
        cam = cam / (cam.max() + 1e-8)
        return cam


def plot_ecg_with_gradcam(
    signal: np.ndarray,
    cam: np.ndarray,
    pred_probs: dict[str, float],
    lead_idx: int = 0,
    fs: float = 500.0,
    title: str = "",
    save_path: str | None = None,
):
    """Plot ECG waveform with Grad-CAM heatmap overlay.

    The waveform is colour-coded by model attention: red segments
    correspond to regions the model weighted heavily, blue segments
    to regions it largely ignored.

    Args:
        signal: ECG signal, shape (time, leads) or (time,).
        cam: Grad-CAM heatmap, shape (time,).
        pred_probs: Dict mapping class names to predicted probabilities.
        lead_idx: Which lead to plot (for multi-lead signals).
        fs: Sampling frequency for time axis labelling.
        title: Figure title.
        save_path: If provided, save figure to this path.
    """
    if signal.ndim == 2:
        signal = signal[:, lead_idx]

    time_axis = np.arange(len(signal)) / fs

    fig, axes = plt.subplots(2, 1, figsize=(14, 5), height_ratios=[3, 1],
                              sharex=True, gridspec_kw={"hspace": 0.08})

    # Top panel: ECG waveform coloured by Grad-CAM intensity
    points = np.array([time_axis, signal]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap="RdYlBu_r", norm=plt.Normalize(0, 1))
    lc.set_array(cam[:-1])
    lc.set_linewidth(1.2)
    axes[0].add_collection(lc)
    axes[0].set_xlim(time_axis[0], time_axis[-1])
    axes[0].set_ylim(signal.min() - 0.3, signal.max() + 0.3)
    axes[0].set_ylabel("Amplitude (normalised)")

    if title:
        axes[0].set_title(title, fontsize=12)

    # Prediction label in top-right
    pred_text = "  ".join(f"{k}: {v:.2f}" for k, v in pred_probs.items())
    axes[0].text(
        0.99, 0.95, pred_text, transform=axes[0].transAxes,
        ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    # Bottom panel: Grad-CAM heatmap as filled area
    axes[1].fill_between(time_axis, cam, alpha=0.7, color="crimson")
    axes[1].set_ylabel("Attention")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylim(0, 1.1)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    plt.close(fig)
    return fig
