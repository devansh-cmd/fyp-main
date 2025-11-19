from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.transforms as T


class ESC50PNGDataset(Dataset):
    """
    ESC-50 dataset loader for PNG spectrogram images.
    Infers the class from esc50.csv (category column) using the base wav id.
    Expects filenames like: 1-100032-A-0_orig.png, 1-100032-A-0_noisy.png, etc.
    """

    def __init__(
        self,
        spec_dir,
        esc50_csv,
        split_csv,
        img_size=224,
        normalize=True,
        augment=False,
    ):
        self.spec_dir = Path(spec_dir)
        self.df_meta = pd.read_csv(esc50_csv)
        self.df_split = pd.read_csv(split_csv)  # columns: filepath, label
        self.classes = sorted(self.df_meta["category"].unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

        # transforms
        tx = [T.Resize((img_size, img_size)), T.ToTensor()]
        if normalize:
            tx.append(T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        self.transform = T.Compose(tx)

        # optional light PNG-level aug
        self.augment = augment
        self.aug = T.RandomApply(
            [
                T.RandomAffine(degrees=5, translate=(0.02, 0.02)),
            ],
            p=0.5,
        )

        # precompute sample list
        self.samples = []
        for _, r in self.df_split.iterrows():
            p = Path(r["filepath"])
            y = self.class_to_idx[r["label"]]
            self.samples.append((p, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, y = self.samples[idx]
        img = Image.open(p).convert("RGB")
        if self.augment:
            img = self.aug(img)
        img = self.transform(img)
        return img, torch.tensor(y, dtype=torch.long)
