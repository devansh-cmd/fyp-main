import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V2_Weights

def get_mobilenet_v2(num_classes: int, pretrained: bool = True, dropout: float = 0.2):
    """
    Returns a standard torchvision MobileNetV2 with a custom classifier head.
    This ensures compatibility with the training pipeline (e.g., parameter freezing).
    """
    weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    
    # Replace the classifier head
    # MobileNetV2 has a classifier that is nn.Sequential(Dropout(dropout), Linear(last_channel, num_classes))
    last_channel = model.last_channel
    model.classifier[0] = nn.Dropout(p=dropout)
    model.classifier[1] = nn.Linear(last_channel, num_classes)
    
    return model

if __name__ == "__main__":
    import torch
    # Quick sanity check
    model = get_mobilenet_v2(num_classes=2)
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"MobileNetV2 Output shape: {output.shape}")
    assert output.shape == (1, 2)
    print("MobileNetV2 sanity check passed.")
