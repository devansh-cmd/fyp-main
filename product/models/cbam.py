"""
Convolutional Block Attention Module (CBAM)

Official implementation:
https://github.com/Jongchan/attention-module

Paper: "CBAM: Convolutional Block Attention Module" (ECCV 2018)
Authors: Sanghyun Woo, Jongchan Park, Joon-Young Lee, In So Kweon

Devansh Dev - FYP Implementation
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    Channel Attention Module.
    
    Applies attention across channels using both average and max pooling.
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Average pooling path
        avg_out = self.fc(self.avg_pool(x))
        # Max pooling path
        max_out = self.fc(self.max_pool(x))
        # Combine and apply sigmoid
        out = self.sigmoid(avg_out + max_out)
        return x * out


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module.
    
    Applies attention across spatial dimensions using channel-wise pooling.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1  # 7x7 works better in practice
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Channel-wise average pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # Channel-wise max pooling
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # Concatenate along channel dimension
        out = torch.cat([avg_out, max_out], dim=1)
        # Apply convolution and sigmoid
        out = self.sigmoid(self.conv(out))
        return x * out


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    
    Sequentially applies channel attention followed by spatial attention.
    
    Args:
        in_channels (int): Number of input channels
        reduction_ratio (int): Channel reduction ratio for MLP (default: 16)
        kernel_size (int): Kernel size for spatial attention conv (default: 7)
    
    Example:
        >>> cbam = CBAM(in_channels=512, reduction_ratio=16)
        >>> x = torch.randn(1, 512, 28, 28)
        >>> out = cbam(x)
        >>> print(out.shape)  # torch.Size([1, 512, 28, 28])
    """
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        # Apply channel attention
        x = self.channel_attention(x)
        # Apply spatial attention
        x = self.spatial_attention(x)
        return x


# Test function
if __name__ == "__main__":
    # Test CBAM module
    print("Testing CBAM module...")
    
    # Create dummy input (batch_size=2, channels=256, height=56, width=56)
    x = torch.randn(2, 256, 56, 56)
    print(f"Input shape: {x.shape}")
    
    # Create CBAM module
    cbam = CBAM(in_channels=256, reduction_ratio=16, kernel_size=7)
    
    # Forward pass
    out = cbam(x)
    print(f"Output shape: {out.shape}")
    
    # Verify shape is preserved
    assert out.shape == x.shape, "Output shape should match input shape"
    print("[PASS] CBAM module test passed!")
    
    # Test with different channel sizes (ResNet block outputs)
    test_configs = [
        (64, 256),    # After layer1
        (128, 512),   # After layer2
        (256, 1024),  # After layer3
        (512, 2048),  # After layer4
    ]
    
    print("\nTesting CBAM with ResNet block configurations:")
    for spatial_size, channels in test_configs:
        x = torch.randn(1, channels, spatial_size, spatial_size)
        cbam = CBAM(in_channels=channels)
        out = cbam(x)
        print(f"  Channels={channels}, Spatial={spatial_size}x{spatial_size} -> [OK]")
    
    print("\n[PASS] All tests passed!")
