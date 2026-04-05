"""
Frequency-Prior Self-Attention (FP-SA)
=======================================
Novel contribution: inject learnable *frequency-band positional embeddings*
into the attention key computation, biasing the model toward frequency ranges
that are clinically relevant for the target pathology.

Motivation:
  - Parkinson's Disease voice tremor: 4–8 Hz modulation, formant F0 shifts (70–300 Hz)
  - Dementia (Pitt Corpus): prosodic anomalies in 0–500 Hz range
  - Heart sounds (PhysioNet): S1/S2 events in 20–500 Hz band

Standard self-attention treats all spatial positions identically. FP-SA adds a
learned frequency bias F ∈ ℝ^(H × inner_dim) to the Keys computed along each row
of the feature map (H = frequency axis). This enables the model to learn
*which frequency bands are most diagnostic* from data.

For ResNet-50 layer4 (B, 2048, 7, 7): H=7 frequency "bands" in latent space.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyPriorSelfAttention(nn.Module):
    """
    Self-Attention with learnable frequency-band key biases.

    Adds a trainable embedding per frequency-axis row to the attention Keys,
    allowing the model to weight frequency bands by their diagnostic relevance.

    Args:
        in_channels (int): Number of input feature channels.
        num_heads (int):   Number of attention heads.
        reduction (int):   Channel bottleneck ratio.
    """

    def __init__(self, in_channels: int, num_heads: int = 8, reduction: int = 8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads

        inner = max(in_channels // reduction, num_heads * 8)
        inner = (inner // num_heads) * num_heads
        self.inner = inner

        # Channel projection
        self.proj_in = nn.Conv2d(in_channels, inner, 1, bias=False)

        # Q, K, V projections
        self.to_q = nn.Linear(inner, inner, bias=False)
        self.to_k = nn.Linear(inner, inner, bias=False)
        self.to_v = nn.Linear(inner, inner, bias=False)

        # --- Frequency-Prior Key Bias ---
        # Will be initialised at forward time based on actual H
        # We register as a parameter but resize on first forward pass
        self._freq_bias = None
        self._freq_bias_H = None

        self.scale = (inner // num_heads) ** -0.5
        self.norm = nn.LayerNorm(inner)

        # Output
        self.proj_out = nn.Conv2d(inner, in_channels, 1, bias=False)
        self.out_norm = nn.GroupNorm(32, in_channels)

    def _get_freq_bias(self, H: int, device: torch.device) -> nn.Parameter:
        """Lazily initialise frequency-prior bias tensor."""
        if self._freq_bias is None or self._freq_bias_H != H:
            # Shape: (H, inner) — one embedding per frequency "band" (row)
            self._freq_bias = nn.Parameter(
                torch.zeros(H, self.inner, device=device)
            )
            nn.init.trunc_normal_(self._freq_bias, std=0.02)
            self._freq_bias_H = H
            # Register so it's properly tracked by optimiser
            self.register_parameter('freq_bias', self._freq_bias)
        return self._freq_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W

        # Project to inner
        inner = self.proj_in(x)          # (B, inner, H, W)
        seq = inner.flatten(2).permute(0, 2, 1)  # (B, N, inner)
        seq = self.norm(seq)

        # Build frequency bias: expand (H, inner) → (N, inner)
        freq_bias = self._get_freq_bias(H, x.device)           # (H, inner)
        freq_bias_expanded = freq_bias.unsqueeze(1).expand(H, W, self.inner)
        freq_bias_expanded = freq_bias_expanded.reshape(N, self.inner)  # (N, inner)

        # Q, K, V — inject frequency prior into Keys
        h, d = self.num_heads, self.inner // self.num_heads
        q = self.to_q(seq).reshape(B, N, h, d).permute(0, 2, 1, 3)
        # Keys get the frequency-band bias added before reshaping
        k_raw = self.to_k(seq) + freq_bias_expanded.unsqueeze(0)  # (B, N, inner)
        k = k_raw.reshape(B, N, h, d).permute(0, 2, 1, 3)
        v = self.to_v(seq).reshape(B, N, h, d).permute(0, 2, 1, 3)

        # Scaled dot-product attention
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        out = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(B, N, self.inner)

        # Reshape back and residual
        out = out.permute(0, 2, 1).reshape(B, self.inner, H, W)
        out = self.proj_out(out)
        out = self.out_norm(out)
        return x + out


if __name__ == "__main__":
    print("Testing FrequencyPriorSelfAttention...")
    x = torch.randn(2, 2048, 7, 7)
    m = FrequencyPriorSelfAttention(2048, num_heads=8, reduction=8)
    out = m(x)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape}"
    params = sum(p.numel() for p in m.parameters())
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Params: {params:,}")
    print("[PASS] FrequencyPriorSelfAttention test passed!")
