"""
Shape-contract tests for all attention modules.

These tests are fast (CPU-only, no weight downloads) and verify that every
attention block preserves the spatial dimensions of its input tensor — a
fundamental contract required for residual skip connections to work.
"""
import pytest
import torch

from freq_prior_attention import FrequencyPriorSelfAttention
from coordinate_attention import CoordinateAttention
from se_block import SEBlock
from cbam import CBAM
from triplet_attention import TripletAttention
from self_attention import SpatialSelfAttention
from tf_self_attention import FactorisedTFSelfAttention
from gated_self_attention import CAGatedSelfAttention


# ---------------------------------------------------------------------------
# Frequency-Prior Self-Attention (novel contribution)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("channels,H,W", [
    (512, 7, 7),
    (256, 14, 14),
    (128, 28, 28),
])
def test_fp_sa_preserves_shape(channels: int, H: int, W: int) -> None:
    """FP-SA must return the same spatial shape as its input (residual add)."""
    model = FrequencyPriorSelfAttention(channels, num_heads=8, reduction=4)
    model.eval()
    x = torch.randn(2, channels, H, W)
    with torch.no_grad():
        out = model(x)
    assert out.shape == x.shape, f"FP-SA shape mismatch: {out.shape} != {x.shape}"


def test_fp_sa_output_dtype() -> None:
    """FP-SA output should be float32 (no unintended upcasting)."""
    model = FrequencyPriorSelfAttention(512, num_heads=8, reduction=4)
    model.eval()
    x = torch.randn(1, 512, 7, 7)
    with torch.no_grad():
        out = model(x)
    assert out.dtype == torch.float32


def test_fp_sa_frequency_bias_registered() -> None:
    """Frequency-band bias must be a learnable parameter after first forward pass."""
    model = FrequencyPriorSelfAttention(512, num_heads=8, reduction=4)
    x = torch.randn(1, 512, 7, 7)
    _ = model(x)  # trigger lazy init
    param_names = [n for n, _ in model.named_parameters()]
    assert "_freq_bias" in param_names, "_freq_bias not registered as a Parameter"


# ---------------------------------------------------------------------------
# Coordinate Attention
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("channels", [64, 256, 512, 2048])
def test_coordinate_attention_preserves_shape(channels: int) -> None:
    """CA must preserve (B, C, H, W) — required for residual identity path."""
    model = CoordinateAttention(channels, channels)
    model.eval()
    x = torch.randn(2, channels, 7, 7)
    with torch.no_grad():
        out = model(x)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# Squeeze-and-Excitation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("channels", [64, 256, 512, 2048])
def test_se_block_preserves_shape(channels: int) -> None:
    model = SEBlock(channels)
    model.eval()
    x = torch.randn(2, channels, 7, 7)
    with torch.no_grad():
        out = model(x)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# CBAM
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("channels", [64, 256, 512])
def test_cbam_preserves_shape(channels: int) -> None:
    model = CBAM(channels)
    model.eval()
    x = torch.randn(2, channels, 7, 7)
    with torch.no_grad():
        out = model(x)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# Triplet Attention
# ---------------------------------------------------------------------------

def test_triplet_attention_preserves_shape() -> None:
    model = TripletAttention()
    model.eval()
    x = torch.randn(2, 512, 7, 7)
    with torch.no_grad():
        out = model(x)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# Spatial Self-Attention and Factorised TF Self-Attention
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("channels", [256, 512])
def test_spatial_self_attention_preserves_shape(channels: int) -> None:
    model = SpatialSelfAttention(channels)
    model.eval()
    x = torch.randn(2, channels, 7, 7)
    with torch.no_grad():
        out = model(x)
    assert out.shape == x.shape


@pytest.mark.parametrize("channels", [256, 512])
def test_factorised_tf_sa_preserves_shape(channels: int) -> None:
    model = FactorisedTFSelfAttention(channels)
    model.eval()
    x = torch.randn(2, channels, 7, 7)
    with torch.no_grad():
        out = model(x)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# CA-Gated Self-Attention
# ---------------------------------------------------------------------------

def test_ca_gated_sa_preserves_shape() -> None:
    model = CAGatedSelfAttention(512)
    model.eval()
    x = torch.randn(2, 512, 7, 7)
    with torch.no_grad():
        out = model(x)
    assert out.shape == x.shape
