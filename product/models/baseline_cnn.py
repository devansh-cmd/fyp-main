"""
Baseline CNN
=============
Lightweight 5-layer custom CNN trained from scratch (no pretrained weights).
Used as a sanity-check lower bound in Term-1 to confirm the experimental
pipeline was functioning before introducing transfer learning.

In practice, ResNet-50 with ImageNet pretraining dominates this baseline on
every evaluated dataset; BaselineCNN is retained for completeness and as an
educational reference for custom architecture construction.

Devansh Dev — FYP 2026
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    """
    Custom 5-layer CNN for audio spectrogram classification.

    Args:
        num_classes (int): Number of output classes.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 112x112
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 56x56
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.head(x)
