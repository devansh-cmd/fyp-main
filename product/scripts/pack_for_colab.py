"""
Pack project files for Colab. Uses Python zipfile for cross-platform compatibility.
Run from PROJECT root: python product/scripts/pack_for_colab.py
"""
import zipfile
import os
from pathlib import Path

ROOT = Path(".")
ZIP_NAME = "colab_all_kfold.zip"

# What to include
DIRS_TO_COPY = [
    "product/training",
    "product/models",
    "product/artifacts/splits",
    "product/artifacts/runs/pitt",
    "product/audio_preprocessing/outputs/spectrograms_pitt",
    "product/audio_preprocessing/outputs/spectrograms",
    "product/audio_preprocessing/outputs/spectrograms_emodb",
]

EXCLUDE_PATTERNS = ["__pycache__", ".pyc", "tensorboard"]

def should_exclude(path_str):
    return any(p in path_str for p in EXCLUDE_PATTERNS)

print(f"Packing {ZIP_NAME}...")
count = 0
with zipfile.ZipFile(ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as zf:
    for dir_path in DIRS_TO_COPY:
        src = ROOT / dir_path
        if not src.exists():
            print(f"  SKIP (not found): {dir_path}")
            continue
        for filepath in src.rglob("*"):
            if filepath.is_file() and not should_exclude(str(filepath)):
                # Use forward slashes for cross-platform compatibility
                arcname = filepath.as_posix()
                zf.write(filepath, arcname)
                count += 1
        file_count = len(list(src.rglob("*")))
        print(f"  Added: {dir_path} ({file_count} items)")

size_mb = os.path.getsize(ZIP_NAME) / (1024 * 1024)
print(f"\nDone! {ZIP_NAME} ({size_mb:.1f} MB, {count} files)")
print("Upload this to the root of your Google Drive.")
