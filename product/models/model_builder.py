import torch.nn as nn
from torchvision import models
from torchvision.models import (
    ResNet50_Weights, 
    MobileNet_V2_Weights,
    ResNet18_Weights,
    EfficientNet_B0_Weights
)

# Import individual attention blocks
try:
    from .se_block import SEBlock
    from .cbam import CBAM
    from .coordinate_attention import CoordinateAttention
    from .triplet_attention import TripletAttention
    from .attention_gate import SingleInputAttentionGate
except ImportError:
    # Handle direct script execution or different import contexts
    from se_block import SEBlock
    from cbam import CBAM
    from coordinate_attention import CoordinateAttention
    from triplet_attention import TripletAttention
    from attention_gate import SingleInputAttentionGate

def attach_attention_to_resnet(model, attention_type):
    """
    Hooks attention blocks into the ResNet architecture at the end of each stage.
    """
    if not attention_type:
        return model
    
    attention_type = attention_type.lower()
    
    # Map attention types to their constructor and output channel sizes for ResNet stages
    # Channels for ResNet50: layer1: 256, layer2: 512, layer3: 1024, layer4: 2048
    # Channels for ResNet18: layer1: 64, layer2: 128, layer3: 256, layer4: 512
    
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

    # Inspect model to determine channel sizes
    # We look at the last child of each layer
    c1 = model.layer1[-1].conv3.out_channels if hasattr(model.layer1[-1], 'conv3') else model.layer1[-1].conv2.out_channels
    c2 = model.layer2[-1].conv3.out_channels if hasattr(model.layer2[-1], 'conv3') else model.layer2[-1].conv2.out_channels
    c3 = model.layer3[-1].conv3.out_channels if hasattr(model.layer3[-1], 'conv3') else model.layer3[-1].conv2.out_channels
    c4 = model.layer4[-1].conv3.out_channels if hasattr(model.layer4[-1], 'conv3') else model.layer4[-1].conv2.out_channels

    model.layer1 = nn.Sequential(model.layer1, get_block(c1))
    model.layer2 = nn.Sequential(model.layer2, get_block(c2))
    model.layer3 = nn.Sequential(model.layer3, get_block(c3))
    model.layer4 = nn.Sequential(model.layer4, get_block(c4))
    
    return model

def attach_attention_to_mobilenet(model, attention_type):
    """
    Hooks attention blocks into MobileNetV2 architecture.
    In MobileNetV2, we typically inject after the point-wise (1x1) convolution of the bottleneck.
    For simplicity, we'll attach to the end of selected feature stages.
    """
    if not attention_type:
        return model
    
    attention_type = attention_type.lower()
    
    def get_block(in_channels):
        if attention_type == 'se': return SEBlock(in_channels)
        if attention_type == 'cbam': return CBAM(in_channels)
        if attention_type == 'ca': return CoordinateAttention(in_channels, in_channels)
        if attention_type == 'triplet': return TripletAttention()
        if attention_type == 'gate': return SingleInputAttentionGate(in_channels)
        return None

    # MobileNetV2 features is a Sequential. We can wrap it or inject inside.
    # We'll attach to the final feature maps before GAP
    last_channels = model.last_channel
    # Wrap the entire feature extractor with the attention block at the end
    model.features = nn.Sequential(model.features, get_block(last_channels))
    
    return model

def build_augmented_model(backbone_name, attention_type, num_classes):
    """
    Primary factory function to create any backbone + attention combination.
    """
    backbone_name = backbone_name.lower()
    
    if backbone_name == 'resnet50':
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model = attach_attention_to_resnet(model, attention_type)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif backbone_name == 'resnet18':
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model = attach_attention_to_resnet(model, attention_type)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif backbone_name == 'mobilenetv2':
        model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        model = attach_attention_to_mobilenet(model, attention_type)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
        
    else:
        raise ValueError(f"Backbone {backbone_name} not supported by the factory yet.")
        
    return model
