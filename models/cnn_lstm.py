"""
CNN-LSTM model for multi-label ECG classification.

Architecture overview:
  1. A stack of 1D convolutional blocks extracts local morphological
     features (e.g. QRS shape, ST-segment deviations) from the raw signal.
  2. The CNN output, which retains a reduced temporal dimension, is fed
     into a bidirectional LSTM to capture inter-beat and rhythm-level
     dependencies across the full recording.
  3. A linear head maps the pooled LSTM representation to multi-label
     logits (one per diagnostic superclass).
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Single 1D convolution block: Conv → BatchNorm → ReLU → MaxPool."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pool_size: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pool_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNNLSTM(nn.Module):
    """CNN-LSTM for multi-label ECG classification.

    Args:
        in_channels: Number of ECG leads (12 for standard, 1 for single-lead).
        num_classes: Number of output classes.
        cnn_channels: Output channels for each convolutional block.
        cnn_kernels: Kernel sizes for each convolutional block.
        lstm_hidden: Hidden size of the LSTM.
        lstm_layers: Number of LSTM layers.
        dropout: Dropout rate applied in conv blocks and before the head.
    """

    def __init__(
        self,
        in_channels: int = 12,
        num_classes: int = 5,
        cnn_channels: list[int] = [64, 128, 256, 256],
        cnn_kernels: list[int] = [15, 11, 7, 5],
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        assert len(cnn_channels) == len(cnn_kernels)

        # Build CNN encoder
        layers = []
        ch_in = in_channels
        for ch_out, ks in zip(cnn_channels, cnn_kernels):
            layers.append(ConvBlock(ch_in, ch_out, ks, pool_size=2, dropout=dropout * 0.5))
            ch_in = ch_out
        # Final pooling to further reduce temporal length
        layers.append(nn.MaxPool1d(4))
        self.cnn = nn.Sequential(*layers)

        # Bidirectional LSTM over the CNN feature sequence
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        # Classification head
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, leads, time).

        Returns:
            Logits of shape (batch, num_classes). Apply sigmoid for
            probabilities in multi-label setting.
        """
        # CNN feature extraction: (batch, leads, time) → (batch, channels, reduced_time)
        features = self.cnn(x)

        # Reshape for LSTM: (batch, reduced_time, channels)
        features = features.permute(0, 2, 1)

        # LSTM temporal modelling
        lstm_out, _ = self.lstm(features)

        # Global average pooling over time
        pooled = lstm_out.mean(dim=1)

        return self.head(pooled)
