"""
CNN-only baseline for ECG classification.

Same convolutional encoder as CNNLSTM but replaces the LSTM with global
average pooling. This serves as an ablation to isolate the contribution
of temporal modelling.
"""
import torch
import torch.nn as nn

from models.cnn_lstm import ConvBlock


class CNNOnly(nn.Module):
    """Pure CNN model with global average pooling over time.

    Args:
        in_channels: Number of ECG leads.
        num_classes: Number of output classes.
        cnn_channels: Output channels for each conv block.
        cnn_kernels: Kernel sizes for each conv block.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_channels: int = 12,
        num_classes: int = 5,
        cnn_channels: list[int] = [64, 128, 256, 256],
        cnn_kernels: list[int] = [15, 11, 7, 5],
        dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__()
        layers = []
        ch_in = in_channels
        for ch_out, ks in zip(cnn_channels, cnn_kernels):
            layers.append(ConvBlock(ch_in, ch_out, ks, pool_size=2, dropout=dropout * 0.5))
            ch_in = ch_out
        self.cnn = nn.Sequential(*layers)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(cnn_channels[-1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x)
        return self.head(features)
