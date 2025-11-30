"""
Res-Net 50 Transfer Learning model for ESC-50 spectrogram classification.

This script almost mirrors the baseline implementation in every way:
Devansh Dev 17-11-2025
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from torchvision import models
from torchvision.models import ResNet50_Weights

# import SE Block
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "models"))
from se_block import SEBlock

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class SpectrogramCSVDataset(Dataset):  # dataset(same as baseline)

    def __init__(self, csv_path: str, label2id: dict, img_size: int = 224):
        self.df = pd.read_csv(csv_path)
        assert {"filepath", "label"}.issubset(
            self.df.columns
        ), f"{csv_path} missing required columns!"
        self.label2id = label2id
        self.img_size = img_size

        missing = [p for p in self.df["filepath"] if not Path(p).exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} missing files (first 5): {missing[:5]}"
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["filepath"]
        y = self.label2id[str(row["label"])]

        img = Image.open(path).convert("RGB")
        img = img.resize((self.img_size, self.img_size), resample=Image.BILINEAR)

        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)

        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        x = (x - mean) / std

        return x, torch.tensor(y, dtype=torch.long)


def build_label_mapping(*csv_paths):  # utility, mapped label to id
    labels = set()
    for p in csv_paths:
        df = pd.read_csv(p)
        labels |= set(df["label"].astype(str).unique())
    labels = sorted(labels)
    return {lab: i for i, lab in enumerate(labels)}, labels


def parse_args():  # CLI argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", default=".", help="Repo root for relative paths")
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--logdir", default="product/artifacts/runs")
    ap.add_argument("--run_name", default="")
    return ap.parse_args()
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# To plot and log to TensorBoard
def plot_and_add_figure(
    writer: SummaryWriter,
    tag: str,
    fig_path: Path,
    xs,
    ys_dict,
    title,
    xlabel,
    ylabel,
    step: int,
):
    fig = plt.figure()
    for k, v in ys_dict.items():
        plt.plot(xs, v, label=k)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    writer.add_figure(tag, fig, global_step=step)
    fig.savefig(fig_path)
    plt.close(fig)


def main():  # Actual main training function
    args = parse_args()
    root = Path(args.project_root)
    run_root = root / args.logdir
    run_root.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    run_name = args.run_name if args.run_name else time.strftime("%Y%m%d-%H%M%S")
    tb_logdir = run_root / run_name
    tb_logdir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(tb_logdir))
    # Mapping setup
    label2id, labels = build_label_mapping(args.train_csv, args.val_csv)
    num_classes = len(labels)
    with open(tb_logdir / "label_mapping.json", "w") as f:
        json.dump({"labels": labels, "label2id": label2id}, f, indent=2)
    # Dataset and Dataloaders
    train_ds = SpectrogramCSVDataset(args.train_csv, label2id, img_size=args.img_size)
    val_ds = SpectrogramCSVDataset(args.val_csv, label2id, img_size=args.img_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    try:
        xb, yb = next(iter(train_loader))
        print(
            "DEBUG batch:",
            xb.shape,
            yb.shape,
            "min/max",
            float(xb.min()),
            float(xb.max()),
        )
    except Exception as e:
        print("[WARN] Couldn't fetch debug batch:", e)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model setup - ResNet50 with SE blocks
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    
    # Insert SE blocks after each ResNet block
    model.layer1 = nn.Sequential(model.layer1, SEBlock(256))
    model.layer2 = nn.Sequential(model.layer2, SEBlock(512))
    model.layer3 = nn.Sequential(model.layer3, SEBlock(1024))
    model.layer4 = nn.Sequential(model.layer4, SEBlock(2048))
    
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)

    for param in model.parameters():  # Freeze all initially
        param.requires_grad = False
    
    # Unfreeze layer4 + SE + classifier
    for p in model.layer4.parameters():
        p.requires_grad = True
    for p in model.fc.parameters():
        p.requires_grad = True

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_ckpt = tb_logdir / "resnet50_se_best.pt"

    global_step = 0
    for epoch in range(1, args.epochs + 1):  # Linear probing ResNet-50 (fr trainable)
        model.train()
        running_loss, running_correct, seen = 0.0, 0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                preds = logits.argmax(1)
                running_loss += loss.item() * yb.size(0)
                running_correct += (preds == yb).sum().item()
                seen += yb.size(0)

            writer.add_scalar("train/loss_step", float(loss.item()), global_step)
            global_step += 1

        train_loss = running_loss / max(1, seen)
        train_acc = running_correct / max(1, seen)
        writer.add_scalar("train/loss_epoch", train_loss, epoch)
        writer.add_scalar("train/acc_epoch", train_acc, epoch)

        model.eval()
        v_loss, v_correct, v_seen = 0.0, 0, 0
        all_preds, all_targets, all_probs = [], [], []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                v_loss += loss.item() * yb.size(0)
                preds = logits.argmax(1)
                probs = torch.softmax(logits, dim=1)  # Get probabilities for ROC-AUC
                v_correct += (preds == yb).sum().item()
                v_seen += yb.size(0)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(yb.cpu().numpy())
                all_probs.append(probs.cpu().numpy())  # Store probabilities

        val_loss = v_loss / max(1, v_seen)
        val_acc = v_correct / max(1, v_seen)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/acc", val_acc, epoch)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "num_classes": num_classes,
                    "img_size": args.img_size,
                    "labels": labels,
                },
                best_ckpt,
            )

        print(
            f"Epoch {epoch}/{args.epochs} | train {train_loss:.4f}/{train_acc:.3f} | val {val_loss:.4f}/{val_acc:.3f}"
        )

        epochs_axis = list(range(1, epoch + 1))
        plot_and_add_figure(
            writer,
            tag="fig/acc_curve",
            fig_path=tb_logdir / f"acc_curve_e{epoch}.png",
            xs=epochs_axis,
            ys_dict={"Train Acc": history["train_acc"], "Val Acc": history["val_acc"]},
            title="Training vs Validation Accuracy",
            xlabel="Epoch",
            ylabel="Accuracy",
            step=epoch,
        )
        plot_and_add_figure(
            writer,
            tag="fig/loss_curve",
            fig_path=tb_logdir / f"loss_curve_e{epoch}.png",
            xs=epochs_axis,
            ys_dict={
                "Train Loss": history["train_loss"],
                "Val Loss": history["val_loss"],
            },
            title="Training vs Validation Loss",
            xlabel="Epoch",
            ylabel="Loss",
            step=epoch,
        )

    # Use predictions from final epoch (already collected in validation loop)
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    y_probs = np.concatenate(all_probs)

    acc = accuracy_score(y_true, y_pred)
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate ROC-AUC (macro, one-vs-rest for multi-class)
    try:
        from sklearn.metrics import roc_auc_score
        auc_macro = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
    except Exception as e:
        print(f"[WARN] Could not compute ROC-AUC: {e}")
        auc_macro = 0.0
    
    # Calculate Top-3 Accuracy
    top3_preds = np.argsort(y_probs, axis=1)[:, -3:]  # Top 3 class indices
    top3_correct = np.any(top3_preds == y_true[:, None], axis=1)
    top3_acc = top3_correct.mean()

    fig = plt.figure(figsize=(10, 8))
    plt.imshow(cm, aspect="auto")
    plt.title("Confusion Matrix (Validation)")
    plt.colorbar(label="Count")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    writer.add_figure("fig/confusion_matrix", fig, global_step=args.epochs)
    fig.savefig(tb_logdir / f"confusion_matrix_e{args.epochs}.png")
    plt.close(fig)

    with open(tb_logdir / f"resnet50_se_metrics_e{args.epochs}.json", "w") as f:
        json.dump(history, f, indent=2)

    # Extract full per-class metrics
    per_class_metrics = {}
    for k, v in rep.items():
        if isinstance(v, dict) and "recall" in v:
            per_class_metrics[k] = {
                "precision": float(v["precision"]),
                "recall": float(v["recall"]),
                "f1-score": float(v["f1-score"]),
                "support": int(v["support"])
            }
    
    summary = {
        "final_val_acc": float(acc),
        "top3_acc": float(top3_acc),
        "auc_macro": float(auc_macro),
        "macro_f1": float(rep.get("macro avg", {}).get("f1-score", 0.0)),
        "macro_precision": float(rep.get("macro avg", {}).get("precision", 0.0)),
        "macro_recall": float(rep.get("macro avg", {}).get("recall", 0.0)),
        "weighted_f1": float(rep.get("weighted avg", {}).get("f1-score", 0.0)),
        "per_class_metrics": per_class_metrics,
    }
    with open(tb_logdir / f"resnet50_se_summary_e{args.epochs}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[Summary] val_acc={acc:.4f}  top3_acc={top3_acc:.4f}  macroF1={summary['macro_f1']:.4f}  AUC={auc_macro:.4f}"
    )
    print(f"Saved best checkpoint: {best_ckpt}")
    print("Done.")
    writer.close()


if __name__ == "__main__":
    main()
