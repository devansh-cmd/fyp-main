"""
Pytest configuration for the FYP test suite.
Adds product/models and product/training to sys.path so tests can import
model and training modules without installation.
"""
import sys
from pathlib import Path

# Project root: three levels up from product/tests/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "product" / "models"
TRAINING_DIR = PROJECT_ROOT / "product" / "training"

for p in (str(MODELS_DIR), str(TRAINING_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)
