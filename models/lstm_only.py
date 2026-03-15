"""
LSTM-only baseline for ECG classification.

Operates directly on the ECG signal without convolutional feature extraction.
The input is downsampled by a factor of 10 (500 Hz → 50 Hz) to keep the
sequence length manageable for the LSTM.
"""
import torch
import torch.nn as nn


class LSTMOnly(nn.Module):
    """LSTM model without CNN feature extraction.

    A linear projection maps each downsampled time step from lead-space
    to a higher-dimensional feature space before feeding into the LSTM.

    Args:
        in_channels: Number of ECG leads.
        num_classes: Number of output classes.
        lstm_hidden: Hidden size of the LSTM.
        lstm_layers: Number of LSTM layers.
        dropout: Dropout rate.
        downsample_factor: Temporal downsampling factor to reduce sequence
            length from 5000 to a tractable size for LSTM.
    """

    def __init__(
        self,
        in_channels: int = 12,
        num_classes: int = 5,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3,
        downsample_factor: int = 10,
        **kwargs,
    ):
        super().__init__()
        self.downsample_factor = downsample_factor

        # Project from lead-space to a richer feature space
        self.input_proj = nn.Linear(in_channels, lstm_hidden)

        self.lstm = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, leads, time) → downsample and transpose
        x = x[:, :, :: self.downsample_factor]  # (batch, leads, time/ds)
        x = x.permute(0, 2, 1)  # (batch, time/ds, leads)

        x = self.input_proj(x)  # (batch, time/ds, hidden)
        lstm_out, _ = self.lstm(x)
        pooled = lstm_out.mean(dim=1)
        return self.head(pooled)
