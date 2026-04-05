from __future__ import annotations

from typing import Optional

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
    from self_attention import SpatialSelfAttention
    from tf_self_attention import FactorisedTFSelfAttention
    from freq_prior_attention import FrequencyPriorSelfAttention
    from gated_self_attention import CAGatedSelfAttention
    from dual_cnn_sa_lstm import DualCNNSALSTM
    from resnet_bilstm import ResNetBiLSTM
    from mobilenet_v2 import get_mobilenet_v2
    from hybrid_net import get_hybrid_model
except ImportError:
    try:
        from .se_block import SEBlock
        from .cbam import CBAM
        from .coordinate_attention import CoordinateAttention
        from .triplet_attention import TripletAttention
        from .attention_gate import SingleInputAttentionGate
        from .self_attention import SpatialSelfAttention
        from .tf_self_attention import FactorisedTFSelfAttention
        from .freq_prior_attention import FrequencyPriorSelfAttention
        from .gated_self_attention import CAGatedSelfAttention
        from .dual_cnn_sa_lstm import DualCNNSALSTM
        from .resnet_bilstm import ResNetBiLSTM
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
        from self_attention import SpatialSelfAttention
        from tf_self_attention import FactorisedTFSelfAttention
        from freq_prior_attention import FrequencyPriorSelfAttention
        from gated_self_attention import CAGatedSelfAttention
        from dual_cnn_sa_lstm import DualCNNSALSTM
        from resnet_bilstm import ResNetBiLSTM
        from mobilenet_v2 import get_mobilenet_v2
        from hybrid_net import get_hybrid_model


def _get_attention_block(attention_type: str, in_channels: int) -> nn.Module:
    """Create a single attention block by type string."""
    attention_type = attention_type.lower()
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
    elif attention_type == 'sa':
        return SpatialSelfAttention(in_channels)
    # --- Novel SA variants ---
    elif attention_type == 'tf_sa':
        return FactorisedTFSelfAttention(in_channels)
    elif attention_type == 'fp_sa':
        return FrequencyPriorSelfAttention(in_channels)
    elif attention_type == 'gated_sa':
        return CAGatedSelfAttention(in_channels)
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")


def _get_layer_channels(layer: nn.Sequential) -> int:
    """Get output channels from the last block of a ResNet layer."""
    last_block = layer[-1]
    if hasattr(last_block, 'conv3'):
        return last_block.conv3.out_channels
    else:
        return last_block.conv2.out_channels


def _inject_at_layer_end(layer: nn.Sequential, attention_type: str) -> nn.Sequential:
    """Append a single attention module after the last block of a ResNet layer."""
    c = _get_layer_channels(layer)
    return nn.Sequential(*layer, _get_attention_block(attention_type, c))


def _inject_mixed_resnet(model: nn.Module, early_type: str, late_type: str) -> nn.Module:
    """
    Mixed attention injection for ResNet:
    - early_type: injected at end of layer1, layer2, layer3
    - late_type:  injected at end of layer4

    Dr. Li strategy: "CA for first 3 blocks, AG/SA for the last block"
    """
    model.layer1 = _inject_at_layer_end(model.layer1, early_type)
    model.layer2 = _inject_at_layer_end(model.layer2, early_type)
    model.layer3 = _inject_at_layer_end(model.layer3, early_type)
    model.layer4 = _inject_at_layer_end(model.layer4, late_type)
    return model


def attach_attention_to_resnet(model: nn.Module, attention_type: Optional[str]) -> nn.Module:
    """
    Standard single-point injection: one attention module at end of layer4.
    Supports: ca, gate, sa, se, cbam, triplet
    """
    if not attention_type:
        return model
    model.layer4 = _inject_at_layer_end(model.layer4, attention_type)
    return model


def attach_attention_to_mobilenet(model: nn.Module, attention_type: Optional[str]) -> nn.Module:
    """
    Append a single attention module after MobileNetV2 features.
    """
    if not attention_type:
        return model
    last_channels = model.last_channel
    model.features = nn.Sequential(
        model.features, _get_attention_block(attention_type, last_channels)
    )
    return model


def build_augmented_model(
    backbone_name: str,
    attention_type: Optional[str],
    num_classes: int,
    dropout: float = 0.5,
) -> nn.Module:
    """
    Primary factory function to create any backbone + attention combination.

    Supported backbone+attention combos:
      resnet50_ca        — ResNet50 + Coordinate Attention at layer4
      resnet50_gate      — ResNet50 + Attention Gate at layer4
      resnet50_sa        — ResNet50 + Self-Attention at layer4
      resnet50_ca_ag     — ResNet50 + CA at layers 1-3, AG at layer4  (mixed)
      resnet50_ca_sa     — ResNet50 + CA at layers 1-3, SA at layer4  (mixed)
      resnet50_ca_lstm   — ResNet50 + CA + BiLSTM temporal head (Phase 6 SOTA push)
      hybrid_ca / hybrid_gate / hybrid_sa / hybrid_ca_ag / hybrid_ca_sa — same on HybridNet
      dual               — DualCNNSALSTM: EfficientNetV2-S + ResNet50 + FP-SA + BiLSTM
    """
    backbone_name = backbone_name.lower()

    # --- Parse compound attention type for mixed strategies ---
    # e.g. "ca_ag" → early='ca', late='gate'
    # e.g. "ca_sa" → early='ca', late='sa'
    mixed_map = {
        'ca_ag':  ('ca', 'gate'),
        'ca_sa':  ('ca', 'sa'),
    }
    is_mixed = attention_type and attention_type.lower() in mixed_map
    if is_mixed:
        early_type, late_type = mixed_map[attention_type.lower()]

    if backbone_name == 'resnet50':
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        if is_mixed:
            model = _inject_mixed_resnet(model, early_type, late_type)
        else:
            model = attach_attention_to_resnet(model, attention_type)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

    elif backbone_name == 'resnet18':
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        if is_mixed:
            model = _inject_mixed_resnet(model, early_type, late_type)
        else:
            model = attach_attention_to_resnet(model, attention_type)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
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
        if attention_type:
            # ResNet branch
            if is_mixed:
                # Inject early type in layers 1-3, late type in layer4
                model.layer1 = _inject_at_layer_end(model.layer1, early_type)
                model.layer2 = _inject_at_layer_end(model.layer2, early_type)
                model.layer3 = _inject_at_layer_end(model.layer3, early_type)
                model.layer4 = _inject_at_layer_end(model.layer4, late_type)
                # MobileNet branch — use late type (more discriminative)
                mob_channels = 1280
                model.features = nn.Sequential(
                    model.features, _get_attention_block(late_type, mob_channels)
                )
            else:
                # Single injection at layer4 + MobileNet features
                c = _get_layer_channels(model.layer4)
                model.layer4 = nn.Sequential(
                    *model.layer4, _get_attention_block(attention_type, c)
                )
                mob_channels = 1280
                model.features = nn.Sequential(
                    model.features, _get_attention_block(attention_type, mob_channels)
                )

    elif backbone_name == 'dual':
        # Dual-CNN + SA + LSTM: EfficientNetV2-S + ResNet-50 → FP-SA → BiLSTM
        # attention_type is ignored — SA is baked into the architecture
        model = DualCNNSALSTM(num_classes=num_classes, dropout=dropout)

    else:
        raise ValueError(f"Backbone '{backbone_name}' not supported by the factory.")

    return model


# ---------------------------------------------------------------------------
# Phase 6 helpers: handle backbone names that encode their own architecture
# ---------------------------------------------------------------------------

# Patch build_augmented_model to intercept compound names like 'resnet50_ca_lstm'
# BEFORE the generic backbone+attention split runs.
_orig_build = build_augmented_model


def build_augmented_model(  # noqa: F811
    backbone_name: str,
    attention_type: Optional[str],
    num_classes: int,
    dropout: float = 0.5,
) -> nn.Module:
    """
    Extended factory — wraps _orig_build and intercepts Phase 6 special cases.
    New entries:
      backbone='resnet50', attention='ca_lstm'  →  ResNetBiLSTM(CA + BiLSTM)
    """
    # Intercept ResNetBiLSTM before the generic path tries to parse 'ca_lstm'
    if backbone_name == 'resnet50' and attention_type == 'ca_lstm':
        return ResNetBiLSTM(num_classes=num_classes, dropout=dropout)
        
    # Intercept DualCNNSALSTM which parses as 'dual'/'cnn_sa_lstm' and prepare for 3-seed runs
    if backbone_name == 'dual' and attention_type == 'cnn_sa_lstm':
        return DualCNNSALSTM(num_classes=num_classes, dropout=dropout, use_sa=True) 

    # Intercept DualCNNLSTM (ablation without SA)
    if backbone_name == 'dual' and attention_type == 'cnn_lstm':
        return DualCNNSALSTM(num_classes=num_classes, dropout=dropout, use_sa=False)

    return _orig_build(backbone_name, attention_type, num_classes, dropout)
