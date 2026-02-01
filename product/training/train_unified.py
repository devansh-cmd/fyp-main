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
        "italian_pd": ["health", "parkinson"],  # Index 0: Health, Index 1: Parkinson
        "physionet": ["normal", "abnormal"],    # Index 0: Normal, Index 1: Abnormal
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
    if not lst: return None
    # Use the order prescribed in the lists above (not sorted alphabetically)
    return {lab: i for i, lab in enumerate(lst)}

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
            p = PROJECT_ROOT / p
            
        # Fallback to _orig.png if pointing to .wav
        if p.suffix == '.wav':
             # Need to find where the spectrogram actually is...
             # This is a bit messy. Let's assume the CSV provides the correct path for now.
             pass

        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")

        img = Image.open(p).convert("RGB")
        img = img.resize((self.img_size, self.img_size), resample=Image.BILINEAR)

        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        x = (x - mean) / std

        label = self.label_map[row['label']]
        return x, torch.tensor(label, dtype=torch.long)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["esc50", "emodb", "italian_pd", "physionet"])
    ap.add_argument("--model_type", default="resnet50", help="e.g. resnet50, resnet50_se, resnet50_ca")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4) # Standard for pathology
    ap.add_argument("--seed", type=int, default=42)
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
    DS_CONFIG = {
        "esc50": {
            "train": "product/artifacts/splits/train.csv", # Assuming standard split
            "val": "product/artifacts/splits/val.csv",
            "num_classes": 50
        },
        "emodb": {
            "train": "product/artifacts/splits/train_emodb.csv",
            "val": "product/artifacts/splits/val_emodb.csv",
            "num_classes": 7
        },
        "italian_pd": {
            "train": "product/artifacts/splits/train_italian_png.csv",
            "val": "product/artifacts/splits/val_italian_png.csv",
            "num_classes": 2
        },
        "physionet": {
            "train": "product/artifacts/splits/train_physionet_png.csv",
            "val": "product/artifacts/splits/val_physionet_png.csv",
            "num_classes": 2
        }
    }
    
    # Ensure CSVs exist or create them if we only have .wav ones
    # Note: Medical CSVs currently point to .wav files.
    # We should have CSVs that point to the .png files for training.
    
    config = DS_CONFIG[args.dataset]
    train_csv = Path(args.train_csv) if args.train_csv else (PROJECT_ROOT / config["train"])
    val_csv = Path(args.val_csv) if args.val_csv else (PROJECT_ROOT / config["val"])
    num_classes = config["num_classes"]
    
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
    model = build_augmented_model(backbone, attention, num_classes)
    
    # Protocol: Transfer Learning (Unfreeze Layer 4 + Head)
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze Layer 4
    if hasattr(model, 'layer4'):
        for param in model.layer4.parameters():
            param.requires_grad = True
    
    # Unfreeze Head
    if hasattr(model, 'fc'):
        for param in model.fc.parameters():
            param.requires_grad = True
    elif hasattr(model, 'classifier'):
        for param in model.classifier.parameters():
            param.requires_grad = True
            
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-2)
    
    writer = SummaryWriter(str(LOG_DIR))
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
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
        
        # --- Clinical & Statistical Metrics ---
        # 1. Macro F1 (Balance check across classes)
        rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        macro_f1 = rep.get("macro avg", {}).get("f1-score", 0.0)
        
        # 2. ROC-AUC (One-vs-Rest Macro for Multiclass, standard for Binary)
        try:
            if num_classes == 2:
                auc = roc_auc_score(y_true, y_probs[:, 1])
            else:
                auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
        except Exception as e:
            print(f"[WARN] ROC-AUC calculation failed: {e}")
            auc = 0.0
        
        print(f"Epoch {epoch:2d} | Train Acc: {t_acc:.3f} | Val Acc: {val_acc:.3f} | Macro F1: {macro_f1:.3f} | AUC: {auc:.3f}")
        
        writer.add_scalar("Loss/Train", t_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Accuracy/Val", val_acc, epoch)
        writer.add_scalar("F1/Val_Macro", macro_f1, epoch)
        writer.add_scalar("AUC/Val", auc, epoch)
        
        history["train_loss"].append(t_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), LOG_DIR / "best_model.pt")
            # Save the best report
            with open(LOG_DIR / "best_classification_report.json", "w") as f:
                json.dump(rep, f, indent=2)

    # Final Summary Save
    summary = {
        "best_val_acc": float(best_val_acc),
        "final_macro_f1": float(macro_f1),
        "final_auc": float(auc),
        "config": vars(args)
    }
    with open(LOG_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Save Final Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cm, cmap='Blues')
    ax.set_title(f"Confusion Matrix: {args.dataset} {args.model_type}")
    plt.savefig(LOG_DIR / "confusion_matrix.png")
    plt.close()

    print(f"Done. Best Acc: {best_val_acc:.4f} | Final AUC: {auc:.4f}")
    writer.close()

if __name__ == "__main__":
    main()
