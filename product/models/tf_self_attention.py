"""
Factorised Time-Frequency Self-Attention (TF-SA)
=================================================
Novel contribution: instead of treating the 2D spectrogram feature map as a flat
sequence of spatial tokens, we decompose attention along the *physical* axes:

  1. Temporal SA: for each frequency row, attend across time columns
     → captures temporal dependencies at each frequency band
  2. Frequency SA: for each time column, attend across frequency rows
     → captures harmonic/spectral relationships at each time step

This factorisation is:
  - Architecturally motivated by the distinct physical meaning of each axis
  - Computationally cheaper: O(H·W² + W·H²) vs O((HW)²) for full SA
  - Novel for clinical speech pathology (PD, dementia) — no prior work applies this

For ResNet-50 layer4 output (B, 2048, 7, 7):
  H=W=7, so N=49 for full SA but only 7 for each factorised pass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FactorisedTFSelfAttention(nn.Module):
    """
    Factorised Time-Frequency Self-Attention for 2D feature maps.

    Applies:
      1) Row-wise  (temporal) self-attention:  each row attends over W columns
      2) Col-wise  (frequency) self-attention: each col attends over H rows

    Args:
        in_channels (int): Number of input feature channels.
        num_heads (int):   Number of attention heads per axis pass.
        reduction (int):   Channel reduction ratio for inner projection.
    """

    def __init__(self, in_channels, num_heads=4, reduction=8):
        super().__init__()
        self.in_channels = in_channels

        inner = max(in_channels // reduction, num_heads * 8)
        inner = (inner // num_heads) * num_heads  # round to multiple of heads
        self.inner = inner

        # Shared projection into bottleneck space
        self.proj_in = nn.Conv2d(in_channels, inner, 1, bias=False)

        # --- Temporal (row-wise) attention ---
        self.t_q = nn.Linear(inner, inner, bias=False)
        self.t_k = nn.Linear(inner, inner, bias=False)
        self.t_v = nn.Linear(inner, inner, bias=False)
        self.t_norm = nn.LayerNorm(inner)

        # --- Frequency (col-wise) attention ---
        self.f_q = nn.Linear(inner, inner, bias=False)
        self.f_k = nn.Linear(inner, inner, bias=False)
        self.f_v = nn.Linear(inner, inner, bias=False)
        self.f_norm = nn.LayerNorm(inner)

        self.num_heads = num_heads
        self.scale = (inner // num_heads) ** -0.5

        # Project back to original channels
        self.proj_out = nn.Conv2d(inner, in_channels, 1, bias=False)
        self.out_norm = nn.GroupNorm(32, in_channels)

    def _mhsa(self, x, q_fn, k_fn, v_fn, norm_fn):
        """Scaled dot-product MHSA on a (B, N, C) sequence."""
        B, N, C = x.shape
        h, d = self.num_heads, C // self.num_heads
        x = norm_fn(x)
        q = q_fn(x).reshape(B, N, h, d).permute(0, 2, 1, 3)
        k = k_fn(x).reshape(B, N, h, d).permute(0, 2, 1, 3)
        v = v_fn(x).reshape(B, N, h, d).permute(0, 2, 1, 3)
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        out = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(B, N, C)
        return out

    def forward(self, x):
        B, C, H, W = x.shape

        # Project to inner space: (B, inner, H, W)
        inner = self.proj_in(x)

        # --- Temporal pass: attend across W for each H row ---
        # Reshape to (B*H, W, inner)
        t_in = inner.permute(0, 2, 3, 1).reshape(B * H, W, self.inner)
        t_out = t_in + self._mhsa(t_in, self.t_q, self.t_k, self.t_v, self.t_norm)
        t_out = t_out.reshape(B, H, W, self.inner).permute(0, 3, 1, 2)  # (B, inner, H, W)

        # --- Frequency pass: attend across H for each W column ---
        # Reshape to (B*W, H, inner)
        f_in = t_out.permute(0, 3, 2, 1).reshape(B * W, H, self.inner)
        f_out = f_in + self._mhsa(f_in, self.f_q, self.f_k, self.f_v, self.f_norm)
        f_out = f_out.reshape(B, W, H, self.inner).permute(0, 3, 2, 1)  # (B, inner, H, W)

        # Project back and residual
        out = self.proj_out(f_out)
        out = self.out_norm(out)
        return x + out


if __name__ == "__main__":
    print("Testing FactorisedTFSelfAttention...")
    x = torch.randn(2, 2048, 7, 7)
    m = FactorisedTFSelfAttention(2048, num_heads=4, reduction=8)
    out = m(x)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape}"
    params = sum(p.numel() for p in m.parameters())
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Params: {params:,}")
    print("[PASS] FactorisedTFSelfAttention test passed!")
