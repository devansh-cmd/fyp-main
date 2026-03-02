import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialSelfAttention(nn.Module):
    """
    Lightweight Multi-Head Self-Attention for 2D feature maps.

    Treats each spatial location as a token, computes scaled dot-product
    attention across all spatial positions, then reshapes back to the
    original feature map shape.

    Includes a projection bottleneck to keep parameter count manageable:
    - Full channels (2048) → bottleneck (256) → attention → back to 2048
    - Residual connection + LayerNorm for training stability

    Args:
        in_channels (int): Number of input channels (e.g. 2048 for ResNet layer4)
        num_heads (int): Number of attention heads (default: 8)
        reduction (int): Bottleneck reduction ratio (default: 8, so 2048 → 256)
    """

    def __init__(self, in_channels, num_heads=8, reduction=8):
        super(SpatialSelfAttention, self).__init__()

        self.in_channels = in_channels
        self.num_heads = num_heads

        # Bottleneck dim — must be divisible by num_heads
        inner_dim = max(in_channels // reduction, num_heads * 8)
        inner_dim = (inner_dim // num_heads) * num_heads  # round to multiple of heads

        self.inner_dim = inner_dim

        # Project input channels → bottleneck
        self.to_inner = nn.Conv2d(in_channels, inner_dim, kernel_size=1, bias=False)
        self.norm_in = nn.LayerNorm(inner_dim)

        # Q, K, V projections (operate in bottleneck space)
        self.to_q = nn.Linear(inner_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(inner_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(inner_dim, inner_dim, bias=False)

        self.scale = (inner_dim // num_heads) ** -0.5

        # Project back to original channels
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, in_channels, bias=False),
        )

        # Output normalisation
        self.norm_out = nn.LayerNorm(in_channels)

    def forward(self, x):
        B, C, H, W = x.shape

        # Project to bottleneck: (B, C, H, W) → (B, inner_dim, H, W)
        inner = self.to_inner(x)

        # Reshape to sequence: (B, inner_dim, H, W) → (B, H*W, inner_dim)
        inner = inner.flatten(2).permute(0, 2, 1)  # (B, N, inner_dim)
        inner = self.norm_in(inner)

        # Multi-head QKV
        h = self.num_heads
        d = self.inner_dim // h

        q = self.to_q(inner).reshape(B, -1, h, d).permute(0, 2, 1, 3)  # (B, h, N, d)
        k = self.to_k(inner).reshape(B, -1, h, d).permute(0, 2, 1, 3)
        v = self.to_v(inner).reshape(B, -1, h, d).permute(0, 2, 1, 3)

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, h, N, N)
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)  # (B, h, N, d)
        out = out.permute(0, 2, 1, 3).reshape(B, H * W, self.inner_dim)  # (B, N, inner_dim)

        # Project back to original channel space
        out = self.to_out(out)  # (B, N, C)
        out = self.norm_out(out)

        # Reshape back to feature map and add residual
        out = out.permute(0, 2, 1).reshape(B, C, H, W)
        return x + out


if __name__ == "__main__":
    print("Testing SpatialSelfAttention module...")
    x = torch.randn(2, 2048, 7, 7)  # ResNet50 layer4 output
    sa = SpatialSelfAttention(2048, num_heads=8, reduction=8)
    out = sa(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    assert x.shape == out.shape, "Shape mismatch!"

    # Count params
    params = sum(p.numel() for p in sa.parameters())
    print(f"Parameters: {params:,}")
    print("[PASS] SpatialSelfAttention test passed!")
