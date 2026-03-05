"""
CA-Gated Self-Attention (Gated-SA)
====================================
Novel contribution: Use Coordinate Attention's direction-aware spatial pooling
to generate a frequency/time saliency mask that *modulates* the self-attention
Keys before softmax. This creates a composed mechanism where:

  1. CA computes a spatial importance signal (H-direction + W-direction)
  2. The combined CA mask gates which spatial tokens the SA attends to most

Unlike stacking CA then SA as independent sequential modules, Gated-SA
*integrates* CA into the SA computation itself — the CA gate is not a
feature transformation but an attention prior that shapes the distribution
before softmax.

Rationale: CA already identifies which time-frequency regions contain the most
discriminative information. Injecting this into SA's key space ensures that
self-attention concentrates on those clinically relevant regions rather than
being diffuse across the full 7×7 feature map.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CAGatedSelfAttention(nn.Module):
    """
    Coordinate-Attention Gated Self-Attention.

    Args:
        in_channels (int): Number of input feature channels.
        num_heads (int):   Number of attention heads.
        reduction (int):   Reduction ratio for CA gate and bottleneck.
    """

    def __init__(self, in_channels, num_heads=8, reduction=8):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads

        # --- CA-based Gate Generator ---
        # Lightweight CA-style pooling to generate a spatial saliency map
        mid_ca = max(in_channels // reduction, 32)
        self.gate_pool_h = nn.AdaptiveAvgPool2d((None, 1))  # pool W → 1
        self.gate_pool_w = nn.AdaptiveAvgPool2d((1, None))  # pool H → 1
        self.gate_conv = nn.Conv2d(in_channels, mid_ca, 1, bias=False)
        self.gate_norm = nn.BatchNorm2d(mid_ca)
        self.gate_act = nn.Hardswish()
        self.gate_h = nn.Conv2d(mid_ca, in_channels, 1, bias=False)
        self.gate_w = nn.Conv2d(mid_ca, in_channels, 1, bias=False)

        # --- Self-Attention in bottleneck space ---
        inner = max(in_channels // reduction, num_heads * 8)
        inner = (inner // num_heads) * num_heads
        self.inner = inner

        self.proj_in = nn.Conv2d(in_channels, inner, 1, bias=False)
        self.to_q = nn.Linear(inner, inner, bias=False)
        self.to_k = nn.Linear(inner, inner, bias=False)
        self.to_v = nn.Linear(inner, inner, bias=False)
        self.norm = nn.LayerNorm(inner)
        self.scale = (inner // num_heads) ** -0.5
        self.proj_out = nn.Conv2d(inner, in_channels, 1, bias=False)
        self.out_norm = nn.GroupNorm(32, in_channels)

    def _compute_ca_gate(self, x):
        """Compute CA-style spatial saliency gate ∈ (0,1), shape (B, C, H, W)."""
        B, C, H, W = x.shape
        h_pool = self.gate_pool_h(x)   # (B, C, H, 1)
        w_pool = self.gate_pool_w(x)   # (B, C, 1, W)
        # Concatenate along spatial dim for joint encoding
        cat = torch.cat([h_pool, w_pool.permute(0, 1, 3, 2)], dim=2)  # (B, C, H+W, 1)
        cat = self.gate_conv(cat)
        cat = self.gate_norm(cat)
        cat = self.gate_act(cat)
        # Split back
        g_h, g_w = cat[:, :, :H, :], cat[:, :, H:, :]
        gate_h = torch.sigmoid(self.gate_h(g_h))          # (B, C, H, 1)
        gate_w = torch.sigmoid(self.gate_w(g_w.permute(0, 1, 3, 2)))  # (B, C, 1, W)
        return gate_h * gate_w   # (B, C, H, W) — combined spatial gate

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        # 1. Compute CA-derived spatial gate
        gate = self._compute_ca_gate(x)   # (B, C, H, W)

        # 2. Project to bottleneck
        inner = self.proj_in(x)            # (B, inner, H, W)
        seq = inner.flatten(2).permute(0, 2, 1)   # (B, N, inner)
        seq = self.norm(seq)

        # 3. Project gate to inner dim (avg over channels) and flatten
        # gate_inner: spatial weight per token (B, N)
        gate_spatial = gate.mean(dim=1).flatten(1)  # (B, N)

        # 4. Q, K, V — gate modifies Key logits (additive log-space bias)
        h, d = self.num_heads, self.inner // self.num_heads
        q = self.to_q(seq).reshape(B, N, h, d).permute(0, 2, 1, 3)
        k = self.to_k(seq).reshape(B, N, h, d).permute(0, 2, 1, 3)
        v = self.to_v(seq).reshape(B, N, h, d).permute(0, 2, 1, 3)

        # Raw attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale   # (B, h, N, N)

        # Inject CA gate as a key-side additive bias: tokens with high saliency
        # attract more attention. gate_spatial → log bias on key dimension
        gate_bias = gate_spatial.log().clamp(min=-5).unsqueeze(1).unsqueeze(2)  # (B,1,1,N)
        scores = scores + gate_bias   # broadcast across heads and query positions

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(B, N, self.inner)

        # 5. Reshape and residual
        out = out.permute(0, 2, 1).reshape(B, self.inner, H, W)
        out = self.proj_out(out)
        out = self.out_norm(out)
        return x + out


if __name__ == "__main__":
    print("Testing CAGatedSelfAttention...")
    x = torch.randn(2, 2048, 7, 7)
    m = CAGatedSelfAttention(2048, num_heads=8, reduction=8)
    out = m(x)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape}"
    params = sum(p.numel() for p in m.parameters())
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Params: {params:,}")
    print("[PASS] CAGatedSelfAttention test passed!")
