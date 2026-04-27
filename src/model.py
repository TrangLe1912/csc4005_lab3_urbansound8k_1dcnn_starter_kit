from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class Conv1DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5, pool_size: int = 2):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Feature1DCNN(nn.Module):
    """1D-CNN cho chuỗi đặc trưng MFCC/log-mel.

    Input shape: [batch, feature_channels, time_frames]
    Ví dụ MFCC: [batch, 40, T]
    """

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        hidden_channels: Sequence[int] = (64, 128, 128),
        dropout: float = 0.35,
    ):
        super().__init__()
        channels = [input_channels, *hidden_channels]
        blocks = [Conv1DBlock(channels[i], channels[i + 1]) for i in range(len(hidden_channels))]
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels[-1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


class RawWaveform1DCNN(nn.Module):
    """1D-CNN trực tiếp trên waveform.

    Input shape: [batch, 1, samples]
    Đây là cấu hình mở rộng, khó hơn MFCC/log-mel.
    """

    def __init__(
        self,
        num_classes: int,
        hidden_channels: Sequence[int] = (32, 64, 128),
        dropout: float = 0.4,
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, hidden_channels[0], kernel_size=80, stride=4, padding=38),
            nn.BatchNorm1d(hidden_channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(hidden_channels[0], hidden_channels[1], kernel_size=25, stride=2, padding=12),
            nn.BatchNorm1d(hidden_channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(hidden_channels[1], hidden_channels[2], kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(hidden_channels[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels[-1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def build_model(
    model_name: str,
    input_channels: int,
    num_classes: int,
    hidden_channels: Sequence[int] = (64, 128, 128),
    dropout: float = 0.35,
) -> nn.Module:
    if model_name in {'mfcc_1dcnn', 'logmel_1dcnn', 'feature_1dcnn'}:
        return Feature1DCNN(
            input_channels=input_channels,
            num_classes=num_classes,
            hidden_channels=hidden_channels,
            dropout=dropout,
        )
    if model_name == 'raw_1dcnn':
        return RawWaveform1DCNN(
            num_classes=num_classes,
            hidden_channels=hidden_channels,
            dropout=dropout,
        )
    raise ValueError(f'Unsupported model_name: {model_name}')
