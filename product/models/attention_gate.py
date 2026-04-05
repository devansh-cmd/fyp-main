"""
Attention Gate
===============
Paper: "Attention U-Net: Learning Where to Look for the Pancreas",
       Oktay et al., MIDL 2018.  Adapted from the medical image segmentation
       context to sequential CNN feature maps.
Ref:   https://arxiv.org/abs/1804.03999

Uses a gating signal (typically a global-average-pooled context vector) to
suppress irrelevant spatial activations in skip-connection features.  Phase-4
ablation showed AG added ~8 M parameters with no consistent F1 improvement
over CoordinateAttention on clinical speech data; rejected from final model.

Devansh Dev — FYP 2026
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AttentionGate(nn.Module):
    """
    Attention Gate (simplified).
    Often used in skip connections but here adapted as a general bottleneck attention.
    
    Args:
        F_g (int): Gating signal channels (optional)
        F_l (int): Input feature channels
        F_int (int): Internal channels for the bottleneck
    """
    def __init__(self, F_g: int, F_l: int, F_int: int) -> None:
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # g is usually from a coarser layer, x from a finer one.
        # If used in-place, g can be the result of a global avg pool + convolution.
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

class SingleInputAttentionGate(nn.Module):
    """
    Simpler version for use in sequential backbones.
    Uses global-average-pooled features as the gating signal.
    """
    def __init__(self, in_channels: int, F_int: int | None = None) -> None:
        super(SingleInputAttentionGate, self).__init__()
        if F_int is None:
            F_int = max(8, in_channels // 2)
            
        self.gate_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.ReLU(inplace=True)
        )
        
        self.W_g = nn.Sequential(
            nn.Conv2d(in_channels, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.gate_conv(x)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

if __name__ == "__main__":
    print("Testing Attention Gate module...")
    x = torch.randn(1, 64, 56, 56)
    gate = SingleInputAttentionGate(64)
    out = gate(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    assert x.shape == out.shape
    print("[PASS] Attention Gate test passed!")
