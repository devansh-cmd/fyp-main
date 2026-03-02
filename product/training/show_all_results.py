import json
import numpy as np
from pathlib import Path

KNOWN_DATASETS = ['emodb', 'esc50', 'italian_pd', 'pcgita', 'physionet', 'pitt']
KNOWN_MODELS = ['resnet50', 'mobilenetv2', 'hybrid']

runs_dir = Path('product/artifacts/runs')
results = {}

for summary_file in sorted(runs_dir.glob('**/summary.json')):
    run_name = summary_file.parent.name  # e.g. italian_pd_resnet50_fold2

    # Try to extract dataset and model by matching known names
    dataset = next((d for d in KNOWN_DATASETS if run_name.startswith(d)), None)
    if not dataset:
        continue
    remainder = run_name[len(dataset)+1:]  # e.g. resnet50_fold2
    model = next((m for m in KNOWN_MODELS if remainder.startswith(m)), None)
    if not model:
        continue
    fold_part = remainder[len(model)+1:]  # e.g. fold2 or s42
    if not fold_part.startswith('fold'):
        continue  # skip legacy seed-based runs

    with open(summary_file) as f:
        s = json.load(f)
    key = (dataset, model)
    if key not in results:
        results[key] = []
    results[key].append(s['best_macro_f1'])

print(f"{'Dataset':<14} {'Model':<14} {'Status':<6}  {'Mean F1':<8}  Std")
print('-' * 55)
prev_ds = None
for (ds, model), scores in sorted(results.items()):
    if ds != prev_ds and prev_ds is not None:
        print()
    prev_ds = ds
    n = len(scores)
    mean = np.mean(scores)
    std = np.std(scores)
    status = 'DONE' if n == 5 else f'{n}/5 🔄'
    print(f"{ds:<14} {model:<14} {status:<8}  {mean:.4f}    {std:.4f}")
