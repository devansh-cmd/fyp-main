"""
DualCNN-SA-LSTM (Phase 6)
==========================
Novel architecture combining dual CNN branches + self-attention + LSTM
to push toward SOTA on clinical speech datasets.

Architecture:
  EfficientNetV2-S  ─┐
                      ├─► Concat (1280+2048) → Linear → SA → LSTM → Classifier
  ResNet-50         ─┘

Motivation:
  - EfficientNet captures fine-grained frequency details (compound scaling)
  - ResNet captures structural/shape features (residual identity paths)
  - Joint representation gives richer feature space than either alone
  - SA block finds cross-dataset relationships in the joint feature space
  - LSTM models temporal dependencies across spectrogram patches
    (treating the HxW spatial grid as a sequence of time steps)

This is equivalent to the SOTA CNN-LSTM papers (Aversano et al. 2024, F1=0.97)
plus the novel SA fusion layer between CNN and LSTM.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    ResNet50_Weights,
    EfficientNet_V2_S_Weights,
)

sys.path.append(os.path.dirname(__file__))
from freq_prior_attention import FrequencyPriorSelfAttention


class DualCNNSALSTM(nn.Module):
    """
    EfficientNetV2-S + ResNet-50 dual-branch → FP-SA → LSTM → Classifier

    Args:
        num_classes (int): Number of output classes.
        dropout (float):   Dropout rate for LSTM and classifier head.
        lstm_hidden (int): LSTM hidden size.
        lstm_layers (int): Number of LSTM layers.
        freeze_backbones (bool): If True, freeze backbone params initially.
    """

    EFFICIENTNET_DIM = 1280   # EfficientNetV2-S last conv channels
    RESNET_DIM       = 2048   # ResNet-50 layer4 channels

    def __init__(
        self,
        num_classes: int = 2,
        dropout: float = 0.5,
        lstm_hidden: int = 512,
        lstm_layers: int = 2,
        freeze_backbones: bool = True,
        use_sa: bool = True,
    ) -> None:
        super().__init__()
        self.use_sa = use_sa

        # --- EfficientNetV2-S branch ---
        effnet = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
        # Remove classifier, keep feature extractor
        self.effnet_features = effnet.features
        self.effnet_pool = nn.AdaptiveAvgPool2d((7, 7))   # Force 7x7 output

        # --- ResNet-50 branch ---
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # Remove the avgpool and fc to keep spatial feature map
        self.resnet_layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.resnet_layer1 = resnet.layer1
        self.resnet_layer2 = resnet.layer2
        self.resnet_layer3 = resnet.layer3
        self.resnet_layer4 = resnet.layer4
        # Output: (B, 2048, 7, 7)

        # --- Joint projection ---
        joint_dim = 512
        self.joint_proj = nn.Sequential(
            nn.Conv2d(self.EFFICIENTNET_DIM + self.RESNET_DIM, joint_dim, 1, bias=False),
            nn.BatchNorm2d(joint_dim),
            nn.ReLU(inplace=True),
        )

        # --- Frequency-Prior Self-Attention (best from Phase 5C) ---
        self.sa = FrequencyPriorSelfAttention(joint_dim, num_heads=8, reduction=4)

        # --- LSTM temporal modelling ---
        # Treat the 7x7 spatial grid as a sequence of 49 time steps (row-major)
        self.lstm = nn.LSTM(
            input_size=joint_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True,
        )

        # --- Classifier head ---
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, num_classes),  # *2 for bidirectional
        )

        # --- Optional initial freeze ---
        if freeze_backbones:
            for p in self.effnet_features.parameters():
                p.requires_grad = False
            for layer in [self.resnet_layer0, self.resnet_layer1,
                          self.resnet_layer2, self.resnet_layer3,
                          self.resnet_layer4]:
                for p in layer.parameters():
                    p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 0. Input check: (B, 3, 224, 224) expected
        B, C, H, W = x.shape
        assert H == 224 and W == 224, f"Expected 224x224 input, got {x.shape}"
        
        # 1. EfficientNet branch: (B, 3, 224, 224) → (B, 1280, 7, 7)
        eff_feat = self.effnet_features(x)
        eff_feat = self.effnet_pool(eff_feat)
        assert eff_feat.shape[-3:] == (self.EFFICIENTNET_DIM, 7, 7), f"EffNet shape wrong: {eff_feat.shape}"

        # 2. ResNet branch: (B, 3, 224, 224) → (B, 2048, 7, 7)
        res = self.resnet_layer0(x)
        res = self.resnet_layer1(res)
        res = self.resnet_layer2(res)
        res = self.resnet_layer3(res)
        res_feat = self.resnet_layer4(res)
        assert res_feat.shape[-3:] == (self.RESNET_DIM, 7, 7), f"ResNet shape wrong: {res_feat.shape}"

        # 3. Concatenate and project: (B, 1280+2048, 7, 7) → (B, 512, 7, 7)
        joint = torch.cat([eff_feat, res_feat], dim=1)
        assert joint.shape[1] == self.EFFICIENTNET_DIM + self.RESNET_DIM, f"Concat shape wrong: {joint.shape}"
        joint = self.joint_proj(joint)
        assert joint.shape[-3:] == (512, 7, 7), f"Fusion projection shape wrong: {joint.shape}"

        # 4. Self-attention on joint features
        if self.use_sa:
            joint_sa = self.sa(joint)   # (B, 512, 7, 7)
            assert joint_sa.shape == joint.shape, f"SA output shape mismatch: {joint_sa.shape} != {joint.shape}"
        else:
            joint_sa = joint

        # 5. Reshape to sequence for LSTM: (B, 49, 512)
        # We flatten the spatial (7x7) dimension into a 49-step sequence
        seq = joint_sa.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        assert seq.shape[1:] == (49, 512), f"LSTM sequence shape wrong: {seq.shape}"

        # 6. LSTM temporal modeling: (B, 49, 512) → (B, 49, lstm_hidden*2)
        out, _ = self.lstm(seq)

        # 7. Mean pooling over sequence time-steps
        out = out.mean(dim=1)   # (B, lstm_hidden*2)

        return self.classifier(out)

    def unfreeze_backbones(self) -> None:
        """Unfreeze deep layers for fine-tuning (call at unfreeze_at epoch)."""
        # Only unfreeze the deep layers (layer3+, last 2 EfficientNet blocks)
        for p in self.resnet_layer4.parameters():
            p.requires_grad = True
        for p in self.resnet_layer3.parameters():
            p.requires_grad = True
        # Last 2 EfficientNet feature blocks
        for block in list(self.effnet_features.children())[-2:]:
            for p in block.parameters():
                p.requires_grad = True


if __name__ == "__main__":
        print("Testing DualCNNSALSTM Shape Progression...")
        x = torch.randn(2, 3, 224, 224)
        m = DualCNNSALSTM(num_classes=2, freeze_backbones=False)
        out = m(x)
        assert out.shape == (2, 2), f"Final output shape mismatch: {out.shape}"
        total = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        
        print("\nAll assert checks passed perfectly!")
        print(f"Final Output Shape: {out.shape}")
        print(f"Total params:       {total:,}")
        print(f"Trainable params:   {trainable:,}")
        print("\n[PASS] DualCNNSALSTM local smoke test passed!")
