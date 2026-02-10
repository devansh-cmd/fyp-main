import torch.nn as nn
from torchvision import models
from torchvision.models import (
    ResNet50_Weights, 
    MobileNet_V2_Weights,
    ResNet18_Weights
)

# Import individual attention blocks
# Using robust imports for Windows multiprocessing/spawn compatibility
try:
    from se_block import SEBlock
    from cbam import CBAM
    from coordinate_attention import CoordinateAttention
    from triplet_attention import TripletAttention
    from attention_gate import SingleInputAttentionGate
    from mobilenet_v2 import get_mobilenet_v2
    from hybrid_net import get_hybrid_model
except ImportError:
    try:
        from .se_block import SEBlock
        from .cbam import CBAM
        from .coordinate_attention import CoordinateAttention
        from .triplet_attention import TripletAttention
        from .attention_gate import SingleInputAttentionGate
        from .mobilenet_v2 import get_mobilenet_v2
        from .hybrid_net import get_hybrid_model
    except ImportError:
        # Fallback for diverse execution contexts
        import sys
        import os
        sys.path.append(os.path.dirname(__file__))
        from se_block import SEBlock
        from cbam import CBAM
        from coordinate_attention import CoordinateAttention
        from triplet_attention import TripletAttention
        from attention_gate import SingleInputAttentionGate
        from mobilenet_v2 import get_mobilenet_v2
        from hybrid_net import get_hybrid_model

def attach_attention_to_resnet(model, attention_type):
    """
    Hooks attention blocks into the ResNet architecture.
    Standard Strategy: Inject attention at the end of EVERY residual block.
    """
    if not attention_type:
        return model
    
    attention_type = attention_type.lower()
    
    def get_block(in_channels):
        if attention_type == 'se':
            return SEBlock(in_channels)
        elif attention_type == 'cbam':
            return CBAM(in_channels)
        elif attention_type == 'ca':
            return CoordinateAttention(in_channels, in_channels)
        elif attention_type == 'triplet':
            return TripletAttention()
        elif attention_type == 'gate':
            return SingleInputAttentionGate(in_channels)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

    # Injection logic: Wrap every block in a stage
    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        layer = getattr(model, layer_name)
        new_blocks = []
        for block in layer:
            if hasattr(block, 'conv3'):
                c = block.conv3.out_channels
            else:
                c = block.conv2.out_channels
            
            new_blocks.append(nn.Sequential(block, get_block(c)))
        
        setattr(model, layer_name, nn.Sequential(*new_blocks))
    
    return model

def attach_attention_to_mobilenet(model, attention_type):
    """
    Hooks attention blocks into MobileNetV2 architecture.
    """
    if not attention_type:
        return model
    
    attention_type = attention_type.lower()
    
    def get_block(in_channels):
        if attention_type == 'se':
            return SEBlock(in_channels)
        if attention_type == 'cbam':
            return CBAM(in_channels)
        if attention_type == 'ca':
            return CoordinateAttention(in_channels, in_channels)
        if attention_type == 'triplet':
            return TripletAttention()
        if attention_type == 'gate':
            return SingleInputAttentionGate(in_channels)
        return None

    last_channels = model.last_channel
    model.features = nn.Sequential(model.features, get_block(last_channels))
    
    return model

def build_augmented_model(backbone_name, attention_type, num_classes, dropout=0.5):
    """
    Primary factory function to create any backbone + attention combination.
    """
    backbone_name = backbone_name.lower()
    
    if backbone_name == 'resnet50':
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model = attach_attention_to_resnet(model, attention_type)
        # Standardize ResNet classifier with dropout for regularization
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(model.fc.in_features, num_classes)
        )
        
    elif backbone_name == 'resnet18':
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model = attach_attention_to_resnet(model, attention_type)
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(model.fc.in_features, num_classes)
        )
        
    elif backbone_name == 'mobilenetv2':
        if attention_type:
            model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
            model = attach_attention_to_mobilenet(model, attention_type)
            model.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(model.last_channel, num_classes)
            )
        else:
            model = get_mobilenet_v2(num_classes=num_classes, dropout=dropout)
        
    elif backbone_name == 'hybrid':
        model = get_hybrid_model(num_classes=num_classes, dropout=dropout)
        
    else:
        raise ValueError(f"Backbone {backbone_name} not supported by the factory yet.")
        
    return model
