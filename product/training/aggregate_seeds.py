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
            
            config = data.get('config', {})
            
            # Determine dataset
            dataset = config.get('dataset', 'Unknown')
            if dataset == 'Unknown':
                # Fallback to path parts
                path_parts = Path(filepath).parts
                if 'esc50' in filepath.lower(): dataset = 'ESC-50'
                elif 'emodb' in filepath.lower(): dataset = 'EmoDB'
                elif 'italian' in filepath.lower(): dataset = 'Italian PD'
                elif 'physionet' in filepath.lower(): dataset = 'PhysioNet'
            
            # Determine model type
            model = config.get('model_type', 'Unknown')
            if model == 'resnet50': model = 'ResNet50'
            
            # Extract seed
            seed = config.get('seed')
            
            # Extract metrics
            acc = data.get('best_val_acc') or data.get('final_val_acc') or data.get('val_acc', 0)
            f1 = data.get('macro_f1') or data.get('final_macro_f1') or data.get('final_f1', 0)
            auc = data.get('final_auc') or data.get('best_auc', 0)
            
            # Add to results
            results.append({
                'Dataset': dataset.upper(),
                'Model': model,
                'Seed': seed,
                'Val_Accuracy': acc * 100,  # Convert to percentage
                'Macro_F1': f1,
                'AUC': auc,
                'Run': config.get('run_name', Path(filepath).parent.name)
            })
            
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
    
    return pd.DataFrame(results)

def aggregate_by_model(df):
    """Aggregate results by dataset and model (mean ± std across seeds)"""
    grouped = df.groupby(['Dataset', 'Model']).agg({
        'Val_Accuracy': ['mean', 'std', 'count'],
        'Macro_F1': ['mean', 'std'],
        'AUC': ['mean', 'std']
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
        print("Model & Val Accuracy (\\%) & Macro-F1 & AUC & Runs \\\\")
        print("\\hline")
        
        for model in subset['Model'].unique():
            model_data = subset[subset['Model'] == model]
            acc_mean = model_data['Val_Accuracy'].mean()
            acc_std = model_data['Val_Accuracy'].std()
            f1_mean = model_data['Macro_F1'].mean()
            f1_std = model_data['Macro_F1'].std()
            auc_mean = model_data['AUC'].mean()
            auc_std = model_data['AUC'].std()
            count = len(model_data)
            
            print(f"{model} & {acc_mean:.1f} $\\pm$ {acc_std:.1f} & "
                  f"{f1_mean:.3f} $\\pm$ {f1_std:.3f} & "
                  f"{auc_mean:.3f} $\\pm$ {auc_std:.3f} & {count} \\\\")
        
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
