import argparse, json, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from product.datasets.esc50_png_dataset import ESC50PNGDataset
from product.models.baseline_cnn import BaselineCNN
import pandas as pd

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", required=True)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=1)   # ← run 1 epoch
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--augment_png", action="store_true")
    return ap.parse_args()

def accuracy(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()

def main():
    args = parse_args()
    root = Path(args.project_root)
    spec_dir = root / "product" / "audio_preprocessing" / "outputs" / "spectrograms"
    esc50_csv = root / "product" / "audio_preprocessing" / "data" / "ESC-50" / "meta" / "esc50.csv"
    split_dir = root / "product" / "artifacts" / "splits"
    run_dir = root / "product" / "artifacts" / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    # datasets
    train_csv = split_dir / "train.csv"
    val_csv = split_dir / "val.csv"

    train_ds = ESC50PNGDataset(spec_dir, esc50_csv, train_csv, img_size=args.img_size, normalize=True, augment=args.augment_png)
    val_ds   = ESC50PNGDataset(spec_dir, esc50_csv, val_csv,   img_size=args.img_size, normalize=True, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BaselineCNN(num_classes=len(train_ds.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ---- 1 epoch training loop ----
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for epoch in range(args.epochs):
        model.train()
        tot_loss, tot_acc, n = 0.0, 0.0, 0
        for x,y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            bs = y.size(0)
            tot_loss += loss.item() * bs
            tot_acc  += (out.argmax(1)==y).float().sum().item()
            n += bs
        train_loss, train_acc = tot_loss/n, tot_acc/n

        model.eval()
        v_loss, v_acc, m = 0.0, 0.0, 0
        with torch.no_grad():
            for x,y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                out = model(x)
                loss = criterion(out, y)
                bs = y.size(0)
                v_loss += loss.item() * bs
                v_acc  += (out.argmax(1)==y).float().sum().item()
                m += bs
        val_loss, val_acc = v_loss/m, v_acc/m

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch+1}/{args.epochs} | train {train_loss:.4f}/{train_acc:.3f} | val {val_loss:.4f}/{val_acc:.3f}")

    # save checkpoint + metrics
    ckpt_path = run_dir / f"baseline_cnn_e{args.epochs}.pt"
    torch.save({"model_state": model.state_dict(),
                "classes": train_ds.classes,
                "img_size": args.img_size}, ckpt_path)
    with open(run_dir / f"baseline_cnn_metrics_e{args.epochs}.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"Saved: {ckpt_path}")
    print("Done.")

if __name__ == "__main__":
    main()
