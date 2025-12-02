"""
Aggregate experiment results across multiple seeds.
Reads all *_summary_e*.json files and creates comparison tables.
"""
import json
import glob
from pathlib import Path
import pandas as pd

def load_all_results():
    """Load all summary JSON files"""
    results = []
    
    # Find all summary files (ResNet, AlexNet, Baseline)
    summary_files = (
        glob.glob('product/artifacts/runs/**/*summary*.json', recursive=True)
    )
    
    # Exclude _BROKEN folder
    summary_files = [f for f in summary_files if '_BROKEN' not in f]
    
    for filepath in summary_files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            # Extract info from path
            path_parts = Path(filepath).parts
            run_name = path_parts[-2]  # e.g., "resnet50_cbam_seed42" or "emodb_cbam_seed123"
            
            # Determine dataset
            dataset = "EmoDB" if run_name.startswith("emodb_") else "ESC-50"
            
            # Determine model type
            if "cbam" in run_name.lower():
                model = "ResNet50+CBAM"
            elif "_se_" in run_name.lower() or run_name.lower().endswith("_se"):
                model = "ResNet50+SE"
            elif "alex" in run_name.lower():
                model = "AlexNet"
            elif "baseline" in run_name.lower() and "resnet" not in run_name.lower():
                model = "Baseline CNN"
            elif "resnet50" in run_name.lower() or "resnet_50" in run_name.lower() or "20251119" in run_name:
                model = "ResNet50"
            else:
                model = "Unknown"
            
            # Extract seed if present
            seed = None
            if "seed42" in run_name:
                seed = 42
            elif "seed123" in run_name:
                seed = 123
            elif "seed999" in run_name:
                seed = 999
            
            # Add to results
            results.append({
                'Dataset': dataset,
                'Model': model,
                'Seed': seed,
                'Run': run_name,
                'Val_Accuracy': data.get('final_val_acc', 0) * 100,  # Convert to percentage
                'Macro_F1': data.get('macro_f1', 0),
                'Top3_Accuracy': data.get('top3_accuracy', 0) * 100 if 'top3_accuracy' in data else None,
                'Filepath': filepath
            })
            
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
    
    return pd.DataFrame(results)

def aggregate_by_model(df):
    """Aggregate results by dataset and model (mean ± std across seeds)"""
    grouped = df.groupby(['Dataset', 'Model']).agg({
        'Val_Accuracy': ['mean', 'std', 'count'],
        'Macro_F1': ['mean', 'std'],
        'Top3_Accuracy': ['mean', 'std']
    }).round(3)
    
    return grouped

def create_latex_table(df):
    """Create LaTeX table format"""
    print("\n" + "="*80)
    print("LATEX TABLE FORMAT")
    print("="*80)
    
    for dataset in df['Dataset'].unique():
        subset = df[df['Dataset'] == dataset]
        print(f"\n\\textbf{{{dataset}}}\n")
        print("\\begin{tabular}{lcccc}")
        print("\\hline")
        print("Model & Val Accuracy (\\%) & Macro-F1 & Top-3 Acc (\\%) & Runs \\\\")
        print("\\hline")
        
        for model in subset['Model'].unique():
            model_data = subset[subset['Model'] == model]
            acc_mean = model_data['Val_Accuracy'].mean()
            acc_std = model_data['Val_Accuracy'].std()
            f1_mean = model_data['Macro_F1'].mean()
            f1_std = model_data['Macro_F1'].std()
            count = len(model_data)
            
            top3 = model_data['Top3_Accuracy'].mean()
            top3_str = f"{top3:.1f}" if pd.notna(top3) else "-"
            
            print(f"{model} & {acc_mean:.1f} $\\pm$ {acc_std:.1f} & "
                  f"{f1_mean:.3f} $\\pm$ {f1_std:.3f} & {top3_str} & {count} \\\\")
        
        print("\\hline")
        print("\\end{tabular}\n")

def main():
    print("Loading experiment results...")
    df = load_all_results()
    
    if df.empty:
        print("No results found!")
        return
    
    print(f"\nFound {len(df)} experiment runs\n")
    
    # Show all results
    print("="*80)
    print("ALL RESULTS")
    print("="*80)
    print(df.to_string(index=False))
    
    # Save to CSV
    df.to_csv('product/artifacts/all_results.csv', index=False)
    print("\n[SAVED] product/artifacts/all_results.csv")
    
    # Aggregated summary
    print("\n" + "="*80)
    print("AGGREGATED SUMMARY (Mean +/- Std across seeds)")
    print("="*80)
    agg = aggregate_by_model(df)
    print(agg)
    
    # Save aggregated
    agg.to_csv('product/artifacts/aggregated_results.csv')
    print("\n[SAVED] product/artifacts/aggregated_results.csv")
    
    # LaTeX format
    create_latex_table(df)
    
    print("\n" + "="*80)
    print("SUMMARY BY DATASET")
    print("="*80)
    for dataset in df['Dataset'].unique():
        subset = df[df['Dataset'] == dataset]
        print(f"\n{dataset}: {len(subset)} runs")
        print(subset[['Model', 'Seed', 'Val_Accuracy', 'Macro_F1']].to_string(index=False))

if __name__ == "__main__":
    main()
