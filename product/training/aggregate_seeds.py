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
            run_dir = Path(filepath).parent
            
            # Determine dataset
            dataset = config.get('dataset', 'Unknown')
            if dataset == 'Unknown':
                if 'esc50' in filepath.lower():
                    dataset = 'ESC-50'
                elif 'emodb' in filepath.lower():
                    dataset = 'EmoDB'
                elif 'italian' in filepath.lower():
                    dataset = 'Italian PD'
                elif 'physionet' in filepath.lower():
                    dataset = 'PhysioNet'
                elif 'pitt' in filepath.lower():
                    dataset = 'Pitt'
            
            # Determine model type
            model = config.get('model_type', 'Unknown')
            if model == 'resnet50':
                model = 'ResNet50'
            
            # Extract seed
            seed = config.get('seed')
            
            # Extract metrics from summary
            acc = data.get('best_val_acc') or data.get('final_val_acc') or data.get('val_acc', 0)
            f1 = data.get('best_macro_f1') or data.get('final_macro_f1') or data.get('macro_f1', 0)
            auc = data.get('final_auc') or data.get('best_auc', 0)
            c_recall = data.get('final_control_recall') or 0.0
            
            prec, recall = 0.0, 0.0
            
            # Try to load classification report for more details
            report_path = run_dir / 'best_classification_report.json'
            if report_path.exists():
                with open(report_path, 'r') as rf:
                    report = json.load(rf)
                macro_avg = report.get('macro avg', {})
                prec = macro_avg.get('precision', 0)
                recall = macro_avg.get('recall', 0)
                # Sometimes f1 in the report is more accurate than the summary's 'final_macro_f1'
                if f1 == 0:
                    f1 = macro_avg.get('f1-score', 0)

            # Add to results
            results.append({
                'Dataset': dataset.upper(),
                'Model': model,
                'Seed': seed,
                'Val_Accuracy': acc * 100,  # Convert to percentage
                'Precision': prec,
                'Recall': recall,
                'Macro_F1': f1,
                'AUC': auc,
                'ControlRecall': c_recall,
                'Run': config.get('run_name', run_dir.name)
            })
            
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
    
    return pd.DataFrame(results)

def aggregate_by_model(df):
    """Aggregate results by dataset and model (mean ± std across seeds)"""
    grouped = df.groupby(['Dataset', 'Model']).agg({
        'Val_Accuracy': ['mean', 'std', 'count'],
        'Precision': ['mean', 'std'],
        'Recall': ['mean', 'std'],
        'Macro_F1': ['mean', 'std'],
        'AUC': ['mean', 'std'],
        'ControlRecall': ['mean', 'std']
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
        print("\\begin{tabular}{lcccccc}")
        print("\\hline")
        print("Model & Val Acc (\\%) & Macro-F1 & AUC & C-Recall & Runs \\\\")
        print("\\hline")
        
        for model in subset['Model'].unique():
            model_data = subset[subset['Model'] == model]
            acc_mean = model_data['Val_Accuracy'].mean()
            acc_std = model_data['Val_Accuracy'].std()
            prec_mean = model_data['Precision'].mean()
            prec_std = model_data['Precision'].std()
            recall_mean = model_data['Recall'].mean()
            recall_std = model_data['Recall'].std()
            f1_mean = model_data['Macro_F1'].mean()
            f1_std = model_data['Macro_F1'].std()
            auc_mean = model_data['AUC'].mean()
            auc_std = model_data['AUC'].std()
            count = len(model_data)
            
            print(f"{model} & {acc_mean:.1f} $\\pm$ {acc_std:.1f} & "
                  f"{f1_mean:.3f} $\\pm$ {f1_std:.3f} & "
                  f"{auc_mean:.3f} $\\pm$ {auc_std:.3f} & "
                  f"{model_data['ControlRecall'].mean():.3f} & {count} \\\\")
        
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
    df.to_csv('product/artifacts/all_results_gold.csv', index=False)
    print("\n[SAVED] product/artifacts/all_results_gold.csv")
    
    # Aggregated summary
    print("\n" + "="*80)
    print("AGGREGATED SUMMARY (Mean +/- Std across seeds)")
    print("="*80)
    agg = aggregate_by_model(df)
    print(agg)
    
    # Save aggregated
    agg.to_csv('product/artifacts/aggregated_results_gold.csv')
    print("\n[SAVED] product/artifacts/aggregated_results_gold.csv")
    
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
