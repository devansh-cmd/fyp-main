import os
import zipfile
import glob

zip_files = [
    'phase6_esc50_dual_cnn_sa_lstm_results.zip',
    'phase6_esc50_resnet50_ca_lstm_results.zip',
    'phase6_pcgita_dual_cnn_sa_lstm_results.zip',
    'phase6_pcgita_resnet50_ca_lstm_results (1).zip'
]

target_dir = 'product/artifacts/runs'
os.makedirs(target_dir, exist_ok=True)

for zf_name in zip_files:
    if os.path.exists(zf_name):
        print(f"Extracting {zf_name}...")
        with zipfile.ZipFile(zf_name, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
            print(f"  -> Extracted {len(zip_ref.namelist())} files.")
    else:
        print(f"File not found: {zf_name}")
