import json
import os
import numpy as np

base_dir = r"c:\FYP\PROJECT\product\artifacts\runs"

datasets = ["esc50", "pcgita", "emodb", "italian_pd"]
models = ["resnet50_ca_lstm", "dual_cnn_sa_lstm"]

print(f"{'Dataset':<15} {'Model':<25} {'Mean F1':>8} {'Std':>8} {'AUC':>8} {'Std':>8}")
print("-" * 75)

results = []

for root, _, files in os.walk(base_dir):
    if "summary.json" in files:
        # Determine dataset and model from run_name (the basename of root)
        run_name = os.path.basename(root)
        
        ds_match = None
        for ds in datasets:
            if run_name.startswith(ds):
                ds_match = ds
                break
                
        mod_match = None
        for mod in models:
            if mod in run_name:
                mod_match = mod
                break
                
        if ds_match and mod_match:
            try:
                with open(os.path.join(root, "summary.json"), "r") as f:
                    data = json.load(f)
                    best_f1 = data.get("best_macro_f1", 0)
                    auc = data.get("final_auc", 0)
                    results.append((ds_match, mod_match, best_f1, auc))
            except Exception:
                pass

aggregated = {}
for ds, mod, f1, auc in results:
    if ds not in aggregated:
        aggregated[ds] = {}
    if mod not in aggregated[ds]:
        aggregated[ds][mod] = {"f1": [], "auc": []}
    aggregated[ds][mod]["f1"].append(f1)
    aggregated[ds][mod]["auc"].append(auc)

for ds in sorted(aggregated.keys()):
    for mod in sorted(aggregated[ds].keys()):
        f1s = aggregated[ds][mod]["f1"]
        aucs = aggregated[ds][mod]["auc"]
        mean_f1 = np.mean(f1s)
        std_f1 = np.std(f1s)
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        print(f"{ds:<15} {mod:<25} {mean_f1:>8.4f} {std_f1:>8.4f} {mean_auc:>8.4f} {std_auc:>8.4f} ({len(f1s)} runs)")
