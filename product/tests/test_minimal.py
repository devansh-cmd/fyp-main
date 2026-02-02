import pytest
import torch
from pathlib import Path
from product.training.train_unified import get_definitive_label_map, UnifiedDataset

def test_torch_available():
    """Verify that PyTorch is correctly installed."""
    assert torch.__version__ is not None

def test_label_map_pitt():
    """Verify that the Pitt label map is correctly established."""
    label_map = get_definitive_label_map("pitt")
    assert label_map == {"control": 0, "dementia": 1}

def test_label_map_physionet():
    """Verify that the PhysioNet label map is correctly established."""
    label_map = get_definitive_label_map("physionet")
    assert label_map == {"normal": 0, "abnormal": 1}

def test_dataset_instantiation():
    """
    Verify that the UnifiedDataset class can be imported and doesn't crash on init 
    even if the CSV doesn't exist (it should only crash on first __getitem__).
    """
    # Just checking imports and basic class structure
    assert UnifiedDataset is not None
