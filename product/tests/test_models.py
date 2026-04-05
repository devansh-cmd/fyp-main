"""
Model shape-contract and architectural-integrity tests.

Fast tests (no @pytest.mark.slow) use random weights and verify output
shapes without downloading pretrained checkpoints.

Slow tests (@pytest.mark.slow) use ImageNet-pretrained weights and verify
the full forward pass of the novel architecture.  Run with:
    pytest -m slow
"""
import pytest
import torch
import torch.nn as nn

from model_builder import build_augmented_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BATCH = 2
DUMMY = torch.randn(BATCH, 3, 224, 224)


def _build(model_str: str, num_classes: int = 2, dropout: float = 0.0) -> nn.Module:
    """Parse model_str exactly as train_unified.py does and build the model."""
    first = model_str.find("_")
    if first == -1:
        backbone, attention = model_str, None
    else:
        backbone = model_str[:first]
        attention = model_str[first + 1:]
    return build_augmented_model(backbone, attention, num_classes, dropout=dropout)


# ---------------------------------------------------------------------------
# Output-shape contracts (pretrained weights — marked slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("model_str,num_classes", [
    ("resnet50", 2),
    ("resnet50", 7),    # EmoDB: 7 emotion classes
    ("resnet50", 50),   # ESC-50: 50 sound classes
    ("resnet50_se", 2),
    ("resnet50_ca", 2),
    ("resnet50_gate", 2),
    ("resnet50_sa", 2),
    ("resnet50_ca_ag", 2),
    ("resnet50_ca_sa", 2),
    ("resnet50_ca_lstm", 2),
    ("dual_cnn_sa_lstm", 2),
    ("dual_cnn_lstm", 2),
])
def test_model_output_shape(model_str: str, num_classes: int) -> None:
    """Every factory model must return (batch, num_classes) logits."""
    model = _build(model_str, num_classes=num_classes)
    model.eval()
    with torch.no_grad():
        out = model(DUMMY)
    assert out.shape == (BATCH, num_classes), (
        f"{model_str}: expected ({BATCH}, {num_classes}), got {out.shape}"
    )


@pytest.mark.slow
def test_model_output_is_float32() -> None:
    """Logits must be float32 — guards against accidental dtype promotion."""
    model = _build("resnet50", num_classes=2)
    model.eval()
    with torch.no_grad():
        out = model(DUMMY)
    assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# DualCNNSALSTM architectural contracts
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_dual_cnn_sa_lstm_shape_with_sa() -> None:
    """Novel architecture forward pass — FP-SA enabled (use_sa=True)."""
    from dual_cnn_sa_lstm import DualCNNSALSTM
    model = DualCNNSALSTM(num_classes=2, freeze_backbones=False, use_sa=True)
    model.eval()
    with torch.no_grad():
        out = model(DUMMY)
    assert out.shape == (BATCH, 2)


@pytest.mark.slow
def test_dual_cnn_sa_lstm_shape_without_sa() -> None:
    """Ablation variant: FP-SA disabled (use_sa=False) must still run."""
    from dual_cnn_sa_lstm import DualCNNSALSTM
    model = DualCNNSALSTM(num_classes=2, freeze_backbones=False, use_sa=False)
    model.eval()
    with torch.no_grad():
        out = model(DUMMY)
    assert out.shape == (BATCH, 2)


@pytest.mark.slow
def test_dual_cnn_sa_lstm_parameter_count() -> None:
    """
    Paper states ~37.1 M parameters.
    Allow ±8 M tolerance for minor version differences in torchvision.
    """
    from dual_cnn_sa_lstm import DualCNNSALSTM
    model = DualCNNSALSTM(num_classes=2, freeze_backbones=False)
    total = sum(p.numel() for p in model.parameters())
    assert 29_000_000 <= total <= 46_000_000, (
        f"Unexpected parameter count: {total:,}  (expected ~37.1 M)"
    )


@pytest.mark.slow
def test_freeze_backbones_reduces_trainable_params() -> None:
    """
    freeze_backbones=True must produce fewer trainable params than False.
    The classifier head must always remain trainable.
    """
    from dual_cnn_sa_lstm import DualCNNSALSTM
    frozen = DualCNNSALSTM(num_classes=2, freeze_backbones=True)
    unfrozen = DualCNNSALSTM(num_classes=2, freeze_backbones=False)

    frozen_trainable = sum(p.numel() for p in frozen.parameters() if p.requires_grad)
    unfrozen_trainable = sum(p.numel() for p in unfrozen.parameters() if p.requires_grad)

    assert frozen_trainable < unfrozen_trainable, (
        "freeze_backbones=True should reduce trainable parameter count"
    )
    # Classifier head must always be trainable
    head_params = sum(p.numel() for p in frozen.classifier.parameters() if p.requires_grad)
    assert head_params > 0, "Classifier head must always be trainable"


@pytest.mark.slow
def test_unfreeze_backbones_increases_trainable_params() -> None:
    """unfreeze_backbones() must expose additional parameters to the optimiser."""
    from dual_cnn_sa_lstm import DualCNNSALSTM
    model = DualCNNSALSTM(num_classes=2, freeze_backbones=True)
    before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.unfreeze_backbones()
    after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert after > before, "unfreeze_backbones() must increase trainable param count"


@pytest.mark.slow
def test_gradient_flows_through_classifier() -> None:
    """Backward pass must produce non-None gradients in the classifier head."""
    from dual_cnn_sa_lstm import DualCNNSALSTM
    model = DualCNNSALSTM(num_classes=2, freeze_backbones=False)
    model.train()
    out = model(DUMMY)
    loss = out.sum()
    loss.backward()
    for name, p in model.classifier.named_parameters():
        assert p.grad is not None, f"No gradient for classifier.{name}"


# ---------------------------------------------------------------------------
# Model factory validation
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_factory_raises_on_unknown_backbone() -> None:
    """Factory must raise ValueError for unrecognised backbone names."""
    with pytest.raises(ValueError, match="not supported"):
        build_augmented_model("unknown_backbone", None, 2)


@pytest.mark.slow
def test_factory_raises_on_unknown_attention() -> None:
    """Factory must raise ValueError for unrecognised attention types."""
    with pytest.raises(ValueError, match="Unknown attention type"):
        build_augmented_model("resnet50", "notreal", 2)
