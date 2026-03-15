"""
CNN-LSTM model for multi-label ECG classification.

Architecture overview:
  1. A stack of residual 1D convolutional blocks with squeeze-and-excitation
     (SE) attention extracts local morphological features from the raw signal.
     Residual connections stabilise gradient flow and allow each block to
     learn refinements on top of the identity mapping. SE modules recalibrate
     channel responses so the network can emphasise diagnostically relevant
     features (e.g. ST-segment shape) while suppressing noise channels.
  2. The CNN output retains sufficient temporal resolution for a bidirectional
     LSTM to capture inter-beat and rhythm-level dependencies.
  3. A linear head maps the pooled LSTM representation to multi-label logits.
"""
import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel-wise recalibration.

    Learns per-channel scaling factors so the model can amplify informative
    feature maps and suppress less useful ones.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(x).unsqueeze(-1)  # (batch, channels, 1)
        return x * scale


class ResConvBlock(nn.Module):
    """Residual 1D convolution block with SE attention.

    Conv → BN → ReLU → Conv → BN → SE → residual add → ReLU → MaxPool.
    A 1x1 convolution is used on the skip path when channel dimensions change.
    """

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

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(pool_size)
        self.dropout = nn.Dropout(dropout)

        # Learnable skip projection when channel count changes
        if in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)

        # Match temporal dimension for residual add (pool may differ by ±1)
        min_len = min(out.size(-1), identity.size(-1))
        out = out[..., :min_len] + identity[..., :min_len]
        out = self.relu(out)
        out = self.pool(out)
        out = self.dropout(out)
        return out


# Keep the old name available so ablation models that import ConvBlock still work
ConvBlock = ResConvBlock


class CNNLSTM(nn.Module):
    """CNN-LSTM for multi-label ECG classification.

    Compared to the initial version, this adds:
      - Residual skip connections in every conv block
      - SE channel attention after each block
      - Reduced final pooling to preserve temporal resolution for LSTM

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

        # Build residual CNN encoder
        blocks = []
        ch_in = in_channels
        for ch_out, ks in zip(cnn_channels, cnn_kernels):
            blocks.append(ResConvBlock(ch_in, ch_out, ks, pool_size=2, dropout=dropout * 0.5))
            ch_in = ch_out
        # Reduced final pooling (was 4, now 2) to keep more temporal steps
        # for the LSTM. With pool=2: 5000 → 2500 → 1250 → 625 → 312 → 156
        blocks.append(nn.MaxPool1d(2))
        self.cnn = nn.Sequential(*blocks)

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
        features = self.cnn(x)

        # (batch, channels, time) → (batch, time, channels)
        features = features.permute(0, 2, 1)

        lstm_out, _ = self.lstm(features)

        # Global average pooling over time
        pooled = lstm_out.mean(dim=1)

        return self.head(pooled)
