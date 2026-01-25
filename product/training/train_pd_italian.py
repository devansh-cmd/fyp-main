import argparse
import json
import random
import time
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from PIL import Image
from torchvision import models
from torchvision.models import ResNet50_Weights

# Add project root to sys.path for model imports
# Assumes script is in product/training/
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT / "product" / "models"))

from se_block import SEBlock
from cbam import CBAM
from coordinate_attention import CoordinateAttention
from triplet_attention import TripletAttention
from attention_gate import SingleInputAttentionGate

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class PDDataset(Dataset):
    """
    Dataset for Italian PD Spectrograms.
    Constructs paths from the 'filename' column in the split CSVs.
    """
    def __init__(self, csv_path: Path, spec_dir: Path, img_size: int = 224):
        self.df = pd.read_csv(csv_path)
        self.spec_dir = spec_dir
        self.img_size = img_size
        
        # Mapping: HC -> 0, PD -> 1
        self.label_map = {"HC": 0, "PD": 1}
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Consistent mapping with generate_spectrograms_italian.py
        # Generation script used: Path(wav_path).stem + "_orig.png"
        # On Windows, Path("file..wav").stem is "file."
        # We'll replicate that logic exactly.
        wav_name = row['filename']
        stem = Path(wav_name).stem
        spec_name = f"{stem}_orig.png"
        spec_path = self.spec_dir / spec_name

        if not spec_path.exists():
            # Fallback for double dots if stem logic differs or for other mismatches
            if ".." in wav_name:
                alt_stem = wav_name.replace(".wav", "")
                alt_spec_name = f"{alt_stem}_orig.png"
                alt_path = self.spec_dir / alt_spec_name
                if alt_path.exists():
                    spec_path = alt_path
            
            if not spec_path.exists():
                raise FileNotFoundError(f"Spectrogram not found: {spec_path} (Checked {spec_name})")

        img = Image.open(spec_path).convert("RGB")
        img = img.resize((self.img_size, self.img_size), resample=Image.BILINEAR)

        # Normalize [0, 1] then ImageNet normalization
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        x = (x - mean) / std

        label = self.label_map[row['label']]
        return x, torch.tensor(label, dtype=torch.long)

# Import unified model builder
sys.path.append(str(PROJECT_ROOT / "product" / "models"))
from model_builder import build_augmented_model

def build_model(model_type: str, num_classes: int):
    """
    Builds the model using the unified model_builder.
    Expects model_type like 'resnet50_se' or 'mobilenetv2_ca'.
    """
    parts = model_type.split('_')
    backbone = parts[0]
    attention = parts[1] if len(parts) > 1 else None
    
    return build_augmented_model(backbone, attention, num_classes)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_type", default="resnet50", choices=["resnet50", "resnet50_se", "resnet50_cbam", "resnet50_ca", "resnet50_triplet", "resnet50_gate"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4) # Lower LR for transfer learning
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run_name", default="")
    return ap.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Path configuration
    SPEC_DIR = PROJECT_ROOT / "product" / "audio_preprocessing" / "outputs" / "spectrograms_italian"
    SPLIT_DIR = PROJECT_ROOT / "product" / "artifacts" / "splits"
    TRAIN_CSV = SPLIT_DIR / "train_italian.csv"
    VAL_CSV = SPLIT_DIR / "val_italian.csv"
    
    LOG_DIR = PROJECT_ROOT / "product" / "artifacts" / "runs" / "italian_pd"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    run_name = args.run_name if args.run_name else f"{args.model_type}_{time.strftime('%m%d_%H%M')}"
    tb_logdir = LOG_DIR / run_name
    writer = SummaryWriter(str(tb_logdir))
    
    print(f"--- Italian PD Training: {args.model_type} ---")
    print(f"Log dir: {tb_logdir}")
    
    # Dataset & Dataloaders
    train_ds = PDDataset(TRAIN_CSV, SPEC_DIR)
    val_ds = PDDataset(VAL_CSV, SPEC_DIR)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model_type, num_classes=2)
    
    # Transfer learning: freeze backbone, train head + added blocks
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze newly added blocks and the fc head
    if args.model_type != "resnet50":
        # Any module that is not standard resnet part (i.e. our SE/CBAM)
        for name, child in model.named_children():
            if name in ['layer1', 'layer2', 'layer3', 'layer4']:
                # The second element in Sequential is our attention block
                if isinstance(child, nn.Sequential) and len(child) > 1:
                    for p in child[1].parameters():
                        p.requires_grad = True
    
    for param in model.fc.parameters():
        param.requires_grad = True
        
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        # TRAIN
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
            
        t_loss = train_loss / seen
        t_acc = train_correct / seen
        writer.add_scalar("Loss/Train", t_loss, epoch)
        writer.add_scalar("Accuracy/Train", t_acc, epoch)
        
        # VAL
        model.eval()
        v_loss, v_correct, v_seen = 0.0, 0, 0
        all_probs, all_targets, all_preds = [], [], []
        
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                
                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(1)
                
                v_loss += loss.item() * yb.size(0)
                v_correct += (preds == yb).sum().item()
                v_seen += yb.size(0)
                
                all_probs.append(probs.cpu().numpy())
                all_targets.append(yb.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                
        val_loss = v_loss / v_seen
        val_acc = v_correct / v_seen
        
        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        y_true = np.concatenate(all_targets)
        y_probs = np.concatenate(all_probs)
        y_pred = np.concatenate(all_preds)
        
        auc = roc_auc_score(y_true, y_probs[:, 1])
        
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Accuracy/Val", val_acc, epoch)
        writer.add_scalar("AUC/Val", auc, epoch)
        
        print(f"Epoch {epoch:2d}/{args.epochs} | Train Loss: {t_loss:.4f} Acc: {t_acc:.3f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} AUC: {auc:.3f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), tb_logdir / "best_model.pt")

    # --- FINAL EVALUATION & SAVING ---
    import matplotlib.pyplot as plt
    
    # Calculate Final Metrics
    acc = accuracy_score(y_true, y_pred)
    rep = classification_report(y_true, y_pred, target_names=["HC", "PD"], output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    # top_k accuracy (k=1 is just acc, but for 2 classes we just report acc)
    summary = {
        "final_val_acc": float(acc),
        "auc": float(auc),
        "macro_f1": float(rep.get("macro avg", {}).get("f1-score", 0.0)),
        "macro_precision": float(rep.get("macro avg", {}).get("precision", 0.0)),
        "macro_recall": float(rep.get("macro avg", {}).get("recall", 0.0)),
        "weighted_f1": float(rep.get("weighted avg", {}).get("f1-score", 0.0)),
    }
    
    # Save JSON files
    with open(tb_logdir / "metrics_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    with open(tb_logdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    # Plot Confusion Matrix
    fig = plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {args.model_type}")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["HC", "PD"])
    plt.yticks(tick_marks, ["HC", "PD"])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
            
    plt.tight_layout()
    fig.savefig(tb_logdir / "confusion_matrix.png")
    plt.close(fig)

    print(f"\n--- Final Results ({args.model_type}) ---")
    print(f"Acc: {acc:.4f} | AUC: {auc:.4f} | F1: {summary['macro_f1']:.4f}")
    print(f"Results saved to: {tb_logdir}")
    writer.close()

if __name__ == "__main__":
    main()
