import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights, MobileNet_V2_Weights

class HybridNet(nn.Module):
    """
    Deep Hybrid Ensemble
    Combines ResNet50 (Semantic Specialist) and MobileNetV2 (Spatial Specialist)
    using a learnable Gated Fusion mechanism.
    """
    def __init__(self, num_classes: int, pretrained: bool = True, dropout: float = 0.5):
        super(HybridNet, self).__init__()
        
        # 1. Initialize Backbones
        resnet50_w = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        mobilenet_w = MobileNet_V2_Weights.DEFAULT if pretrained else None
        
        res = models.resnet50(weights=resnet50_w)
        mob = models.mobilenet_v2(weights=mobilenet_w)
        
        # 2. ResNet Branch (Semantic)
        # We keep the blocks separate to allow train_unified.py to find 'layer4'
        self.resnet_conv = nn.Sequential(
            res.conv1, res.bn1, res.relu, res.maxpool,
            res.layer1, res.layer2, res.layer3
        )
        self.layer4 = res.layer4
        self.resnet_gap = res.avgpool
        
        # 3. MobileNet Branch (Spatial)
        # We keep 'features' named so train_unified.py can find it
        self.features = mob.features
        self.mobilenet_gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # 4. Gated Fusion Mechanism (The "Smart Knob")
        total_features = 2048 + 1280 # 3328
        self.alpha = nn.Parameter(torch.full((total_features,), 0.5))
        
        # 5. Scale Alignment
        self.gate_bn = nn.BatchNorm1d(total_features)
        
        # 6. Classifier (MLP)
        self.classifier = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Branch A: ResNet50
        res_feat = self.resnet_conv(x)
        res_feat = self.layer4(res_feat)
        res_feat = self.resnet_gap(res_feat)
        res_feat = torch.flatten(res_feat, 1) # (B, 2048)
        
        # Branch B: MobileNetV2
        mob_feat = self.features(x)
        mob_feat = self.mobilenet_gap(mob_feat)
        mob_feat = torch.flatten(mob_feat, 1) # (B, 1280)
        
        # Concatenate Specialists
        combined = torch.cat((res_feat, mob_feat), dim=1) # (B, 3328)
        
        # Apply Gated Mask: Features_Fused = Features_Combined * sigmoid(alpha)
        mask = torch.sigmoid(self.alpha)
        gated = combined * mask
        
        # Scale Alignment to prevent ResNet from "drowning out" MobileNet
        aligned = self.gate_bn(gated)
        
        # MLP Classifier
        logits = self.classifier(aligned)
        
        return logits

def get_hybrid_model(num_classes: int, pretrained: bool = True, dropout: float = 0.5):
    return HybridNet(num_classes=num_classes, pretrained=pretrained, dropout=dropout)

if __name__ == "__main__":
    # Quick architecture verify
    model = get_hybrid_model(num_classes=2)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"Hybrid Output shape: {output.shape}")
    print(f"Alpha parameter shape: {model.alpha.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
