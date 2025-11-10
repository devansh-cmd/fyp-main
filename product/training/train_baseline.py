import argparse, json, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from product.datasets.esc50_png_dataset import ESC50PNGDataset
from product.models.baseline_cnn import BaselineCNN
import pandas as pd
import os, json, math
import torch, numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", required=True)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=20)   
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--augment_png", action="store_true")
    ap.add_argument("--logdir", type=str, default="runs\\baseline")
    # parse_args(): add
    ap.add_argument("--limit", type=int, default=0, help="limit train/val samples for quick runs")
    ap.add_argument("--tiny_overfit", type=int, default=0, help="train on N samples and use the same for val")
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--run_name", type=str, default="")
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
    run_name = time.strftime("%Y%m%d-%H%M%S") if not args.run_name else args.run_name
    tb_logdir = run_dir / run_name
    tb_logdir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(tb_logdir))
    metrics_log = run_dir / "metrics.jsonl"
    # datasets
    train_csv = split_dir / "train.csv"
    val_csv = split_dir / "val.csv"

    train_ds = ESC50PNGDataset(spec_dir, esc50_csv, train_csv, img_size=args.img_size, normalize=True, augment=args.augment_png)
    if args.tiny_overfit:
        val_ds= train_ds
    else:
        val_ds   = ESC50PNGDataset(spec_dir, esc50_csv, val_csv,   img_size=args.img_size, normalize=True, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    
    x_dbg, y_dbg = next(iter(train_loader))
    print("DEBUG batch:", x_dbg.shape, y_dbg.shape, "min/max", float(x_dbg.min()), float(x_dbg.max()))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    
    try:
        x_dbg, y_dbg = next(iter(train_loader))
        print("DEBUG batch:", x_dbg.shape, y_dbg.shape, "min/max", float(x_dbg.min()), float(x_dbg.max()))
    except Exception as e:
        print("[ERROR] failed to fetch a batch from train_loader:", e)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BaselineCNN(num_classes=len(train_ds.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    run_name = time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(str(Path(args.logdir) / run_name))
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = float("-inf")
    metrics_log = Path(args.logdir) / "metrics.jsonl" 
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
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val",   val_loss,   epoch)
        writer.add_scalar("acc/train",  train_acc,  epoch)
        writer.add_scalar("acc/val",    val_acc,    epoch)
        writer.flush()
        print(f"Epoch {epoch+1}/{args.epochs} | train {train_loss:.4f}/{train_acc:.3f} | val {val_loss:.4f}/{val_acc:.3f}")
        epoch_record = {
            "epoch": int(epoch + 1),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "train_acc": float(train_acc),
            "val_acc": float(val_acc),
        }
        with open(str(metrics_log), "a", encoding="utf-8") as f:
            f.write(json.dumps(epoch_record))
            f.write("\n")

        # ---- save best checkpoint by val_acc ----
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": train_ds.classes,
                    "img_size": int(args.img_size),
                },
                run_dir / "baseline_best.pt",
            )
    # save checkpoint + metrics
    ckpt_path = run_dir / f"baseline_cnn_e{args.epochs}.pt"
    torch.save({"model_state": model.state_dict(),
                "classes": train_ds.classes,
                "img_size": args.img_size}, ckpt_path)
    with open(run_dir / f"baseline_cnn_metrics_e{args.epochs}.json", "w") as f:
        json.dump(history, f, indent=2)
    # Evaluate on validation set (load best checkpoint if present)
    best_path = run_dir / "baseline_best.pt"
    if best_path.exists():
        state = torch.load(best_path, map_location=device)
        model.load_state_dict(state["model_state"])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device, non_blocking=True)
            out = model(x)
            preds = out.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    mj = run_dir / "metrics.jsonl"
    mj2 = run_dir / f"baseline_cnn_metrics_e{max(args.epochs,1)}.json"
    if len(history["train_acc"]) == 0:
        if mj.exists():
         try:
            import pandas as pd
            df = pd.read_json(mj, lines=True)
            history["train_acc"] = df["train_acc"].tolist()
            history["val_acc"]   = df["val_acc"].tolist()
            history["train_loss"]= df["train_loss"].tolist()
            history["val_loss"]  = df["val_loss"].tolist()
            args.epochs = len(df)
         except Exception as e:
            print("[WARN] failed to read metrics.jsonl:", e)
    elif mj2.exists():
        try: 
            with open(mj2, "r") as f:
                h = json.load(f)
            history["train_acc"] = h["train_acc"]
            history["val_acc"]   = h["val_acc"]
            history["train_loss"]= h["train_loss"]
            history["val_loss"]  = h["val_loss"]
            args.epochs = len(history["train_acc"])
        except Exception as e:
            print("[WARN] failed to read metrics.json1:",e)
     # Basic sanity for predictions
    if len(all_preds) == 0:
        print("[WARN] No predictions were produced on validation set; skipping confusion matrix / metrics plots.")
        acc = 0.0
        rep = {}
        cm = None
    else:
        acc = accuracy_score(all_labels, all_preds)
        rep = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
        cm  = confusion_matrix(all_labels, all_preds)

    # Confusion matrix 
    if cm is not None:
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, aspect='auto')
        plt.colorbar(label="Count")
        plt.title("Confusion Matrix (Validation)")
        plt.xlabel("Predicted"); plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(run_dir / f"confusion_matrix_e{args.epochs}.png")
        plt.close()
    else:
         print("[INFO] skipping confusion matrix (no predictions).")
    # Curves from in-memory history 
    n_epochs = len(history["train_acc"])
    print("DEBUG: history lengths:",
          "train_acc", len(history["train_acc"]),
          "val_acc", len(history["val_acc"]),
          "train_loss", len(history["train_loss"]),
          "val_loss", len(history["val_loss"]),
          "args.epochs", args.epochs)
    if n_epochs == 0:
        print("[WARN] No training history available — skipping accuracy/loss plots.")
    else:
        epochs_axis = range(1, n_epochs + 1)

        plt.figure()
        plt.plot(epochs_axis, history["train_acc"], label="Train Acc")
        plt.plot(epochs_axis, history["val_acc"],   label="Val Acc")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy")
        plt.title("Training vs Validation Accuracy")
        plt.legend()
        plt.savefig(str(run_dir / f"acc_curve_e{n_epochs}.png"))
        plt.close()

        plt.figure()
        plt.plot(epochs_axis, history["train_loss"], label="Train Loss")
        plt.plot(epochs_axis, history["val_loss"],   label="Val Loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.savefig(str(run_dir / f"loss_curve_e{n_epochs}.png"))
        plt.close()
    # Summary JSON (final report) 
    summary = {
        "final_val_acc": float(acc),
        "macro_f1": float(rep["macro avg"]["f1-score"]),
        "weighted_f1": float(rep["weighted avg"]["f1-score"]),
        "per_class_recall": {k: v["recall"] for k, v in rep.items() if k.isdigit()}
    }
    with open(run_dir / f"baseline_summary_e{args.epochs}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[Summary] val_acc={acc:.4f}  macroF1={summary['macro_f1']:.4f}  weightedF1={summary['weighted_f1']:.4f}")
    print(f"Saved: {ckpt_path}")
    print("Done.")
    writer.close()
if __name__ == "__main__":
    main()
