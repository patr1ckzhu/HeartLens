"""
CNN-Transformer baseline for ECG classification.

Uses the same CNN encoder as CNNLSTM but replaces the bidirectional LSTM
with a Transformer encoder. This ablation compares self-attention against
recurrent temporal modelling.
"""
import math

import torch
import torch.nn as nn

from models.cnn_lstm import ConvBlock


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence position awareness."""

    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class CNNTransformer(nn.Module):
    """CNN encoder followed by Transformer encoder for ECG classification.

    Args:
        in_channels: Number of ECG leads.
        num_classes: Number of output classes.
        cnn_channels: Output channels for each conv block.
        cnn_kernels: Kernel sizes for each conv block.
        num_heads: Number of attention heads.
        num_transformer_layers: Number of Transformer encoder layers.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_channels: int = 12,
        num_classes: int = 5,
        cnn_channels: list[int] = [64, 128, 256, 256],
        cnn_kernels: list[int] = [15, 11, 7, 5],
        num_heads: int = 8,
        num_transformer_layers: int = 2,
        dropout: float = 0.4,
        **kwargs,
    ):
        super().__init__()
        layers = []
        ch_in = in_channels
        for ch_out, ks in zip(cnn_channels, cnn_kernels):
            layers.append(ConvBlock(ch_in, ch_out, ks, pool_size=2, dropout=dropout * 0.5))
            ch_in = ch_out
        layers.append(nn.MaxPool1d(2))
        self.cnn = nn.Sequential(*layers)

        d_model = cnn_channels[-1]
        self.pos_enc = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_layers,
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x)  # (batch, channels, time)
        features = features.permute(0, 2, 1)  # (batch, time, channels)
        features = self.pos_enc(features)
        features = self.transformer(features)
        pooled = features.mean(dim=1)
        return self.head(pooled)
