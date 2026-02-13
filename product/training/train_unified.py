import json
import random
import sys
from pathlib import Path

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from PIL import Image

# Add project root to sys.path for model imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT / "product" / "models"))  # noqa: E402

from model_builder import build_augmented_model  # noqa: E402

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_definitive_label_map(dataset_name):
    """Static mapping to ensure classes never swap between seeds."""
    # Static mappings to ensure Positive (Disease/Event) class is always at index 1 for ROC-AUC
    classes = {
        "emodb": ["anger", "boredom", "disgust", "fear", "happiness", "neutral", "sadness"],
        "italian_pd": ["HC", "PD"],  # Index 0: Health (HC), Index 1: Parkinson (PD)
        "physionet": ["normal", "abnormal"],    # Index 0: Normal, Index 1: Abnormal
        "pitt": ["control", "dementia"],        # Index 0: Control, Index 1: Dementia
        "esc50": sorted([
            "airplane", "breathing", "brushing_teeth", "can_opening", "car_horn",
            "cat", "chainsaw", "chirping_birds", "church_bells", "clapping",
            "clock_alarm", "clock_tick", "coughing", "cow", "crackling_fire",
            "crickets", "crow", "crying_baby", "dog", "door_wood_creaks",
            "door_wood_knock", "drinking_sipping", "engine", "fireworks",
            "footsteps", "frog", "glass_breaking", "hand_saw", "helicopter",
            "hen", "insects", "keyboard_typing", "laughing", "mouse_click",
            "pig", "pouring_water", "rain", "rooster", "sea_waves", "sheep",
            "siren", "sneezing", "snoring", "thunderstorm", "toilet_flush",
            "train", "vacuum_cleaner", "washing_machine", "water_drops", "wind"
        ])
    }
    lst = classes.get(dataset_name.lower())
    if not lst:
        return None
    # Use the order prescribed in the lists above (not sorted alphabetically)
    return {lab: i for i, lab in enumerate(lst)}

def is_integer_label(val):
    """Check if a label is already an integer or numpy integer."""
    return isinstance(val, (int, np.integer)) or (isinstance(val, str) and val.isdigit())

class UnifiedDataset(Dataset):
    """
    Standardized Dataset for all domains.
    Expects CSV with 'path' (or 'filepath') and 'label' columns.
    """
    def __init__(self, csv_path: Path, dataset_name: str, img_size: int = 224, label_map: dict = None):
        self.df = pd.read_csv(csv_path)
        self.dataset_name = dataset_name
        self.img_size = img_size
        
        # Determine path column
        if 'path' in self.df.columns:
            self.path_col = 'path'
        elif 'filepath' in self.df.columns:
            self.path_col = 'filepath'
        else:
            raise KeyError(f"No path/filepath column found in {csv_path}")

        # Label mapping setup
        if label_map:
            self.label_map = label_map
        else:
            # Always use definitive mapping for scientific consistency
            self.label_map = get_definitive_label_map(self.dataset_name)
            if not self.label_map:
                 unique_labels = sorted(self.df['label'].unique())
                 self.label_map = {lab: i for i, lab in enumerate(unique_labels)}
            print(f"Using definitive label mapping for {self.dataset_name}: {self.label_map}")
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav_rel_path = row[self.path_col]
        
        # Spectrogram mapping (standard: _orig.png)
        # Note: This logic assumes the 'path' in medical CSVs points to .wav
        # but we need the .png from the outputs folder.
        # However, for consistency, we expect CSVs to point EXACTLY at the files we want 
        # OR we handle dataset-specific stem logic.#
        
        # We assume for unified training, we pass CSVs that point to generated SPECTROGRAMS
        # or we fix the paths here. Let's make it flexible.
        p = Path(wav_rel_path)
        if not p.is_absolute():
            # Try raw path first
            full_path = PROJECT_ROOT / p
            if not full_path.exists():
                # Fallback: some Pitt paths are missing 'product/' prefix
                full_path = PROJECT_ROOT / "product" / p
            p = full_path
            
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")

        img = Image.open(p).convert("RGB")
        img = img.resize((self.img_size, self.img_size), resample=Image.BILINEAR)

        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        x = (x - mean) / std

        # Handle mapping: use index directly if it's already an integer, 
        # otherwise look it up in the label_map.
        raw_label = row['label']
        if is_integer_label(raw_label):
            label = int(raw_label)
        else:
            label = self.label_map[raw_label]

        return x, torch.tensor(label, dtype=torch.long)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["esc50", "emodb", "italian_pd", "physionet", "pitt"])
    ap.add_argument("--model_type", default="resnet50", help="e.g. resnet50, resnet50_se, resnet50_ca")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4) # Standard for pathology
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--unfreeze_at", type=int, default=0, help="Epoch at which to unfreeze backbone deep layers")
    ap.add_argument("--weighted_loss", action="store_true", help="Enable class weighting to handle imbalance")
    ap.add_argument("--dropout", type=float, default=0.5, help="Dropout rate for the classifier head")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fold", type=int, default=None, help="K-Fold index (0-4). If set, uses fold-indexed CSVs.")
    ap.add_argument("--run_name", default="")
    ap.add_argument("--train_csv", default=None, help="Optional override for training CSV path")
    ap.add_argument("--val_csv", default=None, help="Optional override for validation CSV path")
    return ap.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Dataset Definitions
    # When --fold is set, use fold-indexed CSVs; otherwise fall back to legacy single-split CSVs.
    fold_suffix = f"_fold{args.fold}" if args.fold is not None else ""
    
    DS_CONFIG = {
        "esc50": {
            "train": f"product/artifacts/splits/train{fold_suffix}.csv",
            "val": f"product/artifacts/splits/val{fold_suffix}.csv",
            "num_classes": 50
        },
        "emodb": {
            "train": f"product/artifacts/splits/train_emodb{fold_suffix}.csv",
            "val": f"product/artifacts/splits/val_emodb{fold_suffix}.csv",
            "num_classes": 7
        },
        "italian_pd": {
            "train": f"product/artifacts/splits/train_italian{fold_suffix}.csv" if args.fold is not None else "product/artifacts/splits/train_italian_png.csv",
            "val": f"product/artifacts/splits/val_italian{fold_suffix}.csv" if args.fold is not None else "product/artifacts/splits/val_italian_png.csv",
            "num_classes": 2
        },
        "physionet": {
            "train": f"product/artifacts/splits/train_physionet{fold_suffix}.csv" if args.fold is not None else "product/artifacts/splits/train_physionet_png.csv",
            "val": f"product/artifacts/splits/val_physionet{fold_suffix}.csv" if args.fold is not None else "product/artifacts/splits/val_physionet_png.csv",
            "num_classes": 2
        },
        "pitt": {
            "train": f"product/artifacts/splits/train_pitt{fold_suffix}.csv" if args.fold is not None else "product/artifacts/splits/train_pitt_segments.csv",
            "val": f"product/artifacts/splits/val_pitt{fold_suffix}.csv" if args.fold is not None else "product/artifacts/splits/val_pitt_segments.csv",
            "num_classes": 2
        }
    }
    
    config = DS_CONFIG[args.dataset]
    train_csv = Path(args.train_csv) if args.train_csv else (PROJECT_ROOT / config["train"])
    val_csv = Path(args.val_csv) if args.val_csv else (PROJECT_ROOT / config["val"])
    num_classes = config["num_classes"]
    
    # Include fold in run_name for directory isolation
    if args.fold is not None:
        run_name = args.run_name if args.run_name else f"{args.dataset}_{args.model_type}_fold{args.fold}"
    else:
        run_name = args.run_name if args.run_name else f"{args.dataset}_{args.model_type}_s{args.seed}"
    LOG_DIR = PROJECT_ROOT / "product" / "artifacts" / "runs" / args.dataset / run_name
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"--- Unified Training: {args.dataset} | {args.model_type} | Seed {args.seed} ---")
    
    # Setup Dataset/Loader
    train_ds = UnifiedDataset(train_csv, dataset_name=args.dataset)
    val_ds = UnifiedDataset(val_csv, dataset_name=args.dataset, label_map=train_ds.label_map)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Build Model via Factory
    parts = args.model_type.split('_')
    backbone = parts[0]
    attention = parts[1] if len(parts) > 1 else None
    model = build_augmented_model(backbone, attention, num_classes, dropout=args.dropout)
    model = model.to(device)
    
    def unfreeze_backbones(model, model_type):
        """Dynamic unfreezing logic for late features."""
        if model_type == 'hybrid':
            # Unfreeze ResNet branch deep layers
            if hasattr(model, 'layer4'):
                for param in model.layer4.parameters():
                    param.requires_grad = True
            # Unfreeze MobileNet branch deep layers
            if hasattr(model, 'features'):
                for i in range(14, len(model.features)):
                    for param in model.features[i].parameters():
                        param.requires_grad = True
        elif hasattr(model, 'layer4'):
            for param in model.layer4.parameters():
                param.requires_grad = True
        elif hasattr(model, 'features'):
            for i in range(14, len(model.features)):
                for param in model.features[i].parameters():
                    param.requires_grad = True
        elif hasattr(model, 'model') and hasattr(model.model, 'features'):
             for i in range(14, len(model.model.features)):
                for param in model.model.features[i].parameters():
                    param.requires_grad = True
        print(">>> Dynamic Unfreeze Executed: Late Features are now trainable.")

    # Protocol: Transfer Learning Initial State
    for param in model.parameters():
        param.requires_grad = False
    
    # Always unfreeze head/gate
    if hasattr(model, 'alpha'):
        model.alpha.requires_grad = True
        for param in model.gate_bn.parameters():
            param.requires_grad = True
    
    if hasattr(model, 'fc'):
        for param in model.fc.parameters():
            param.requires_grad = True
    elif hasattr(model, 'classifier'):
        for param in model.classifier.parameters():
            param.requires_grad = True
            
    # Initial Unfreeze if no warm-up requested
    if args.unfreeze_at == 0:
        unfreeze_backbones(model, args.model_type)
        
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized with {trainable_params:,} trainable parameters.")
            
    # Calculate Class Weights for Imbalanced Datasets (e.g. Pitt)
    criterion_weight = None
    if args.weighted_loss:
        labels = train_ds.df['label'].values
        # Handle both string and int labels
        if not is_integer_label(labels[0]):
            labels = [train_ds.label_map[l] for l in labels]
        
        class_counts = np.bincount(labels)
        total_samples = len(labels)
        # Weight = Total / (Num_Classes * Class_Count)
        weights = total_samples / (len(class_counts) * class_counts)
        criterion_weight = torch.tensor(weights, dtype=torch.float32).to(device)
        print(f"Applying Class Weights: {weights} (Targeting Class 0 Failure)")

    criterion = nn.CrossEntropyLoss(weight=criterion_weight)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    
    writer = SummaryWriter(str(LOG_DIR))
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "macro_f1": []}
    best_macro_f1 = 0.0
    
    for epoch in range(1, args.epochs + 1):
        if epoch == args.unfreeze_at + 1 and args.unfreeze_at > 0:
            unfreeze_backbones(model, args.model_type)
            # Re-init optimizer to include newly unfrozen parameters
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
            
        model.train()
        train_loss, train_correct, seen = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            
            preds = logits.argmax(1)
            train_loss += loss.item() * yb.size(0)
            train_correct += (preds == yb).sum().item()
            seen += yb.size(0)
            
        t_loss, t_acc = train_loss/seen, train_correct/seen
        
        # Validation
        model.eval()
        v_loss, v_correct, v_seen = 0.0, 0, 0
        all_probs, all_targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                v_loss += criterion(logits, yb).item() * yb.size(0)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())
                all_targets.append(yb.cpu().numpy())
                v_correct += (logits.argmax(1) == yb).sum().item()
                v_seen += yb.size(0)
        
        val_loss, val_acc = v_loss/v_seen, v_correct/v_seen
        y_true = np.concatenate(all_targets)
        y_probs = np.concatenate(all_probs)
        y_pred = np.argmax(y_probs, axis=1)
        
        # --- Bias-Correction Metrics ---
        rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        macro_f1 = rep.get("macro avg", {}).get("f1-score", 0.0)
        
        # Monitor Minority Class (Control) Recall - Priority for Bias Correction
        # In most medical tasks, Class 0 is Control, Class 1 is Pathology
        control_recall = rep.get("0", {}).get("recall", 0.0)
        
        # 2. ROC-AUC
        try:
            if num_classes == 2:
                auc = roc_auc_score(y_true, y_probs[:, 1])
            else:
                auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
        except Exception as e:
            auc = 0.0
        
        print(f"Epoch {epoch:2d} | Train Acc: {t_acc:.3f} | Val Acc: {val_acc:.3f} | Macro F1: {macro_f1:.3f} | C-Recall: {control_recall:.3f} | AUC: {auc:.3f}")
        
        writer.add_scalar("Loss/Train", t_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Accuracy/Val", val_acc, epoch)
        writer.add_scalar("F1/Val_Macro", macro_f1, epoch)
        writer.add_scalar("Recall/Control", control_recall, epoch)
        writer.add_scalar("AUC/Val", auc, epoch)
        
        history["train_loss"].append(t_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(val_acc)
        history["macro_f1"].append(macro_f1)
        
        # Selection Criteria: Macro F1 (Unbiased toward majority class)
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            LOG_DIR.mkdir(parents=True, exist_ok=True) # Ensure dir exists before save
            torch.save(model.state_dict(), LOG_DIR / "best_model.pt")
            with open(LOG_DIR / "best_classification_report.json", "w") as f:
                json.dump(rep, f, indent=2)

    # Final Summary Save
    summary = {
        "best_macro_f1": float(best_macro_f1),
        "final_macro_f1": float(macro_f1),
        "final_auc": float(auc),
        "final_control_recall": float(control_recall),
        "config": vars(args)
    }
    LOG_DIR.mkdir(parents=True, exist_ok=True) # Ensure dir exists before summary save
    with open(LOG_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Save Final Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cm, cmap='Blues')
    ax.set_title(f"Confusion Matrix: {args.dataset} {args.model_type}")
    plt.savefig(LOG_DIR / "confusion_matrix.png")
    plt.close()

    print(f"Done. Best Macro F1: {best_macro_f1:.4f} | Final AUC: {auc:.4f} | C-Recall: {control_recall:.4f}")
    writer.close()

if __name__ == "__main__":
    main()
