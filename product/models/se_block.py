"""
Squeeze-and-Excitation Block (SE Block)

Official implementation:
https://github.com/moskomule/senet.pytorch

Paper: "Squeeze-and-Excitation Networks" (CVPR 2018)
Authors: Jie Hu, Li Shen, Gang Sun

Devansh Dev - FYP Implementation
"""

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
    
    Adaptively recalibrates channel-wise feature responses by explicitly
    modeling interdependencies between channels.
    
    Args:
        in_channels (int): Number of input channels
        reduction_ratio (int): Channel reduction ratio for bottleneck (default: 16)
    
    Example:
        >>> se = SEBlock(in_channels=512, reduction_ratio=16)
        >>> x = torch.randn(1, 512, 28, 28)
        >>> out = se(x)
        >>> print(out.shape)  # torch.Size([1, 512, 28, 28])
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        # TODO: try reduction_ratio=8 if this doesn't work well
        
        # Squeeze: Global average pooling
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        
        # Excitation: Two FC layers with ReLU and Sigmoid
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        
        # Squeeze: Global spatial information into channel descriptor
        # [B, C, H, W] -> [B, C, 1, 1]
        squeeze = self.squeeze(x).view(batch_size, channels)
        
        # Excitation: Adaptive channel-wise recalibration
        # [B, C] -> [B, C]
        excitation = self.excitation(squeeze).view(batch_size, channels, 1, 1)
        
        # Scale: Multiply input by channel attention weights
        # [B, C, H, W] * [B, C, 1, 1] -> [B, C, H, W]
        return x * excitation.expand_as(x)


# Test function
if __name__ == "__main__":
    # Test SE Block
    print("Testing SE Block...")
    
    # Create dummy input (batch_size=2, channels=256, height=56, width=56)
    x = torch.randn(2, 256, 56, 56)
    print(f"Input shape: {x.shape}")
    
    # Create SE block
    se = SEBlock(in_channels=256, reduction_ratio=16)
    
    # Forward pass
    out = se(x)
    print(f"Output shape: {out.shape}")
    
    # Verify shape is preserved
    assert out.shape == x.shape, "Output shape should match input shape"
    print("[PASS] SE Block test passed!")
    
    # Test with different channel sizes (ResNet block outputs)
    test_configs = [
        (64, 256),    # After layer1
        (128, 512),   # After layer2
        (256, 1024),  # After layer3
        (512, 2048),  # After layer4
    ]
    
    print("\nTesting SE Block with ResNet block configurations:")
    for spatial_size, channels in test_configs:
        x = torch.randn(1, channels, spatial_size, spatial_size)
        se = SEBlock(in_channels=channels, reduction_ratio=16)
        out = se(x)
        print(f"  Channels={channels}, Spatial={spatial_size}x{spatial_size} -> [OK]")
    
    # Test parameter count
    se_test = SEBlock(in_channels=2048, reduction_ratio=16)
    total_params = sum(p.numel() for p in se_test.parameters())
    print(f"\nSE Block parameters (2048 channels, r=16): {total_params:,}")
    
    print("\n[PASS] All tests passed!")
