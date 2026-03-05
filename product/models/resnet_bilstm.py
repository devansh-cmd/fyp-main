"""
ResNet-BiLSTM (Phase 6)
========================
Adds temporal modelling to the ResNet50+CA baseline to close the gap
to SOTA on clinical speech spectrograms (Italian PD, target F1 ≥ 0.95).

Architecture (advisor recommendation):
  ResNet50 backbone
  → CoordinateAttention at layer4          (best Phase 5 attention)
  → AdaptiveAvgPool2d(1, T)                (collapse frequency only → keep time axis)
  → reshape to (B, T, C)                   (treat time columns as sequence steps)
  → 2-layer Bidirectional LSTM
  → mean pooling over T
  → Dropout → Linear → num_classes

Why this helps over plain ResNet50+CA:
  - Spectrograms encode pathological tremor as *persistent* TF patterns across time.
  - A CNN sees each T column locally. BiLSTM sees context across ALL time steps.
  - Bidirectional: captures both forward (onset→peak) and backward (decay) pattern.
  - Frequency-only pooling preserves the temporal sequence: each step = one time slice.

Registered in model_builder.py as backbone='resnet50', attention='ca_lstm'
  → model_type argument: 'resnet50_ca_lstm'
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

import sys
import os
sys.path.append(os.path.dirname(__file__))
from coordinate_attention import CoordinateAttention


class ResNetBiLSTM(nn.Module):
    """
    ResNet50 + CoordinateAttention + BiLSTM temporal head.

    Args:
        num_classes  (int):   Number of output classes.
        dropout      (float): Dropout in LSTM inter-layer and classifier head.
        lstm_hidden  (int):   LSTM hidden units per direction.
        lstm_layers  (int):   Number of stacked LSTM layers.
        freeze_backbone (bool): Freeze ResNet50 backbone initially.
    """

    RESNET_DIM = 2048  # ResNet50 layer4 output channels

    def __init__(self, num_classes=2, dropout=0.5,
                 lstm_hidden=256, lstm_layers=2, freeze_backbone=True):
        super().__init__()

        # --- ResNet50 backbone (no avgpool, no fc) ---
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.layer0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1   # (B, 256,  56, 56) for 224px input
        self.layer2 = resnet.layer2   # (B, 512,  28, 28)
        self.layer3 = resnet.layer3   # (B, 1024, 14, 14)
        self.layer4 = resnet.layer4   # (B, 2048,  7,  7)

        # --- CoordinateAttention after layer4 ---
        # (best single-point attention from Phase 5 results)
        self.ca = CoordinateAttention(self.RESNET_DIM, self.RESNET_DIM)

        # --- Frequency-only pooling → keep time axis ---
        # Input:  (B, 2048, H, W)   — H=freq, W=time after ResNet
        # Output: (B, 2048, 1, W)   — collapse freq only
        # Squeeze: (B, 2048, W)     → permute → (B, W, 2048) for LSTM
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))  # pool freq rows only

        # --- BiLSTM ---
        self.lstm = nn.LSTM(
            input_size=self.RESNET_DIM,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
        )

        # --- Classifier head ---
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(lstm_hidden * 2, num_classes),  # *2 for bidirectional
        )

        # --- Initial freeze ---
        if freeze_backbone:
            for layer in [self.layer0, self.layer1, self.layer2,
                          self.layer3, self.layer4]:
                for p in layer.parameters():
                    p.requires_grad = False
            # CA and LSTM are always trainable from the start

    def forward(self, x):
        # ResNet50 feature extraction
        x = self.layer0(x)   # (B, 64,  56, 56)
        x = self.layer1(x)   # (B, 256, 56, 56)
        x = self.layer2(x)   # (B, 512, 28, 28)
        x = self.layer3(x)   # (B, 1024,14, 14)
        x = self.layer4(x)   # (B, 2048, 7,  7)

        # CoordinateAttention
        x = self.ca(x)        # (B, 2048, 7,  7)

        # Assert correct axis orientation: (B, C, Freq, Time) before pooling Freq
        # The user requested an explicit check to ensure we don't silently pool the wrong axis if transposed
        assert x.shape[-2] == 7 and x.shape[-1] == 7, \
            f"Unexpected spatial dims before freq_pool: {x.shape} — expected (B, C, 7, 7) for 224px input"
        
        # Collapse frequency axis only: (B, 2048, 7, 7) → (B, 2048, 1, 7)
        x = self.freq_pool(x)     # (B, 2048, 1, W)
        
        assert x.shape[-2] == 1, f"Freq axis not collapsed: {x.shape}"
        x = x.squeeze(2)          # (B, 2048, W)
        x = x.permute(0, 2, 1)   # (B, W, 2048)  — (batch, time, features)

        # BiLSTM: (B, W, 2048) → (B, W, lstm_hidden*2)
        lstm_out, _ = self.lstm(x)

        # Mean pooling over time steps: (B, lstm_hidden*2)
        out = lstm_out.mean(dim=1)

        return self.classifier(out)

    def unfreeze_backbones(self):
        """Unfreeze deep ResNet layers (layer3, layer4) for fine-tuning."""
        for p in self.layer3.parameters():
            p.requires_grad = True
        for p in self.layer4.parameters():
            p.requires_grad = True
        print(">>> ResNetBiLSTM: unfroze layer3 + layer4.")


if __name__ == "__main__":
    print("Testing ResNetBiLSTM...")
    x = torch.randn(2, 3, 224, 224)

    # Test frozen mode (default)
    m = ResNetBiLSTM(num_classes=2, freeze_backbone=True)
    out = m(x)
    assert out.shape == (2, 2), f"Shape mismatch: {out.shape}"
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"  Output shape:     {out.shape}")
    print(f"  Total params:     {total:,}")
    print(f"  Trainable params: {trainable:,} (frozen backbone)")

    # Test unfrozen mode
    m2 = ResNetBiLSTM(num_classes=2, freeze_backbone=False)
    out2 = m2(x)
    assert out2.shape == (2, 2), f"Shape mismatch: {out2.shape}"
    trainable2 = sum(p.numel() for p in m2.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable2:,} (fully unfrozen)")

    # Test unfreeze_backbones method
    m.unfreeze_backbones()
    trainable3 = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable3:,} (after unfreeze layer3+4)")

    print("[PASS] ResNetBiLSTM test passed!")
