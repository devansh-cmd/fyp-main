"""
Grad-CAM visualisation for DualCNN-FPSA-BiLSTM / ResNet-50 models.

Generates a figure showing which frequency regions the model attends to
on PD vs HC spectrograms — directly supports the FP-SA narrative.

Usage:
    # Using best available ResNet-50 model (Italian PD fold 0):
    python product/scripts/gradcam_visualise.py \
        --model_path product/artifacts/runs/italian_pd/italian_pd_resnet50_fold0/best_model.pt \
        --model_type resnet50 \
        --val_csv product/artifacts/splits/val_italian_fold0.csv \
        --dataset italian_pd \
        --out_dir product/artifacts/gradcam \
        --n_samples 4

    # Using dual_cnn model (once weights are available):
    python product/scripts/gradcam_visualise.py \
        --model_path path/to/dual_cnn_best_model.pt \
        --model_type dual_cnn_sa_lstm \
        --val_csv product/artifacts/splits/val_italian_fold0.csv \
        --dataset italian_pd \
        --out_dir product/artifacts/gradcam \
        --n_samples 4
"""

import argparse
import sys
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from product.models.model_builder import build_augmented_model

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

LABEL_MAPS = {
    "italian_pd": {"HC": 0, "PD": 1},
    "pcgita":     {"HC": 0, "PD": 1},
    "emodb":      {"anger": 0, "boredom": 1, "disgust": 2, "fear": 3,
                   "happiness": 4, "neutral": 5, "sadness": 6},
    "esc50":      {str(i): i for i in range(50)},
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class GradCAM:
    """
    Grad-CAM for any model with a spatial feature map.
    Hooks the target layer, accumulates gradients and activations,
    and produces a heatmap.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None

        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        # For LSTM models output may be a tuple; take the tensor
        if isinstance(output, tuple):
            output = output[0]
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        g = grad_output[0]
        if g is not None:
            self.gradients = g.detach()

    def __call__(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        Returns a Grad-CAM heatmap (H, W) normalised to [0, 1].
        """
        self.model.zero_grad()
        output = self.model(x)
        # Handle models that return logits directly
        if isinstance(output, tuple):
            output = output[0]

        score = output[0, class_idx]
        score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks did not fire. Check target_layer.")

        # Global average pool the gradients: (C,)
        grads = self.gradients  # (1, C, H, W) or (1, C)
        acts  = self.activations  # (1, C, H, W) or (1, C)

        if grads.dim() == 4:
            weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
            cam = (weights * acts).sum(dim=1).squeeze(0)     # (H, W)
        else:
            # Flat features — reshape to square for visualisation
            weights = grads.mean(dim=0)
            cam = (weights * acts.squeeze(0)).sum(dim=0)
            side = int(cam.numel() ** 0.5)
            cam = cam[:side * side].view(side, side)

        cam = F.relu(cam)
        cam = cam - cam.min()
        denom = cam.max()
        if denom > 0:
            cam = cam / denom
        return cam.cpu().numpy()

    def remove(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def get_target_layer(model, model_type: str):
    """Return the last spatial conv layer for Grad-CAM."""
    mt = model_type.lower()
    if "dual" in mt:
        # DualCNNSALSTM: use ResNet layer4
        return model.resnet_layer4[-1].conv3 if hasattr(model, "resnet_layer4") else None
    elif "resnet" in mt:
        # Standard ResNet-50 with optional attention wrapper
        backbone = getattr(model, "backbone", model)
        return backbone.layer4[-1].conv3
    elif "mobilenet" in mt:
        backbone = getattr(model, "backbone", model)
        return backbone.features[-1][0]
    else:
        raise ValueError(f"Cannot determine target layer for model_type={model_type}")


def load_samples(val_csv: str, dataset: str, n_pd: int = 4, n_hc: int = 4):
    """Load n_pd PD and n_hc HC sample paths from the val CSV."""
    project_root = Path(__file__).resolve().parents[2]
    df = pd.read_csv(val_csv)

    col = "path" if "path" in df.columns else "filepath"
    label_col = "label"

    pd_rows = df[df[label_col] == "PD"].sample(min(n_pd, len(df[df[label_col] == "PD"])),
                                                 random_state=42)
    hc_rows = df[df[label_col] == "HC"].sample(min(n_hc, len(df[df[label_col] == "HC"])),
                                                 random_state=42)

    samples = []
    for _, row in pd.concat([pd_rows, hc_rows]).iterrows():
        p = Path(row[col])
        if not p.is_absolute():
            p = project_root / p
        samples.append({"path": str(p), "label": row[label_col]})

    return samples


def run_gradcam(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build model — map model_type string to (backbone, attention)
    mt = args.model_type
    if mt in ("dual_cnn_sa_lstm", "dual_cnn_lstm"):
        backbone  = "dual"
        attention = mt.replace("dual_cnn_", "cnn_")
    elif mt == "resnet50":
        backbone, attention = "resnet50", None
    elif mt == "mobilenetv2":
        backbone, attention = "mobilenetv2", None
    elif mt.startswith("resnet50_"):
        backbone  = "resnet50"
        attention = mt[len("resnet50_"):]
    else:
        parts     = mt.split("_", 1)
        backbone  = parts[0]
        attention = parts[1] if len(parts) > 1 else None

    num_classes = 2 if args.dataset in ("italian_pd", "pcgita") else (
        7 if args.dataset == "emodb" else 50)

    model = build_augmented_model(backbone, attention, num_classes=num_classes, dropout=0.5)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Target layer
    target_layer = get_target_layer(model, args.model_type)
    gcam = GradCAM(model, target_layer)

    # Load samples
    samples = load_samples(args.val_csv, args.dataset,
                           n_pd=args.n_samples, n_hc=args.n_samples)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_map = LABEL_MAPS.get(args.dataset, {})

    # build the output figure
    n_total = len(samples)
    fig = plt.figure(figsize=(4 * n_total, 8))
    fig.suptitle("Grad-CAM: frequency regions attended by the model\n"
                 "Top row: original spectrogram  |  Bottom row: Grad-CAM overlay",
                 fontsize=12, y=1.01)

    gs = gridspec.GridSpec(2, n_total, figure=fig, hspace=0.05, wspace=0.05)

    for col_idx, sample in enumerate(samples):
        img_path = sample["path"]
        true_label = sample["label"]

        if not os.path.exists(img_path):
            print(f"  MISSING: {img_path} — skipping")
            continue

        # Load & preprocess
        img_pil = Image.open(img_path).convert("RGB")
        img_pil_resized = img_pil.resize((224, 224))
        x = transform(img_pil_resized).unsqueeze(0).to(device)
        x.requires_grad_(True)

        # Predicted class
        with torch.no_grad():
            logits = model(x)
        pred_idx = logits.argmax(dim=1).item()

        # Target = true class for Grad-CAM (shows what model looks at for that class)
        target_idx = label_map.get(true_label, pred_idx)

        # Grad-CAM
        x_grad = transform(img_pil_resized).unsqueeze(0).to(device).requires_grad_(True)
        heatmap = gcam(x_grad, target_idx)  # (H, W) in [0,1]

        # Resize heatmap to image size
        heatmap_up = F.interpolate(
            torch.tensor(heatmap).unsqueeze(0).unsqueeze(0),
            size=(224, 224), mode="bilinear", align_corners=False
        ).squeeze().numpy()

        # Overlay
        img_np = np.array(img_pil_resized) / 255.0
        heatmap_rgb = plt.cm.jet(heatmap_up)[..., :3]
        overlay = 0.55 * img_np + 0.45 * heatmap_rgb
        overlay = np.clip(overlay, 0, 1)

        # Determine pred label string
        inv_map = {v: k for k, v in label_map.items()}
        pred_label = inv_map.get(pred_idx, str(pred_idx))
        correct = "(correct)" if pred_label == true_label else "(wrong)"

        # Row 0: original
        ax0 = fig.add_subplot(gs[0, col_idx])
        ax0.imshow(img_pil_resized)
        ax0.set_title(f"True: {true_label}\nPred: {pred_label} {correct}",
                      fontsize=9)
        ax0.axis("off")

        # Row 1: overlay
        ax1 = fig.add_subplot(gs[1, col_idx])
        ax1.imshow(overlay)
        ax1.axis("off")

    gcam.remove()

    out_path = out_dir / f"gradcam_{args.dataset}_{args.model_type}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser("Grad-CAM visualisation")
    parser.add_argument("--model_path",  required=True,
                        help="Path to best_model.pt")
    parser.add_argument("--model_type",  required=True,
                        help="e.g. resnet50, resnet50_ca, dual_cnn_sa_lstm")
    parser.add_argument("--val_csv",     required=True,
                        help="Validation CSV for the fold used")
    parser.add_argument("--dataset",     required=True,
                        help="italian_pd | pcgita | emodb | esc50")
    parser.add_argument("--out_dir",     default="product/artifacts/gradcam")
    parser.add_argument("--n_samples",   type=int, default=4,
                        help="Number of PD and HC samples each")
    args = parser.parse_args()
    run_gradcam(args)


if __name__ == "__main__":
    main()
