"""Datasets + collation for CTC training.

`SyntheticDataset` generates augmented formula images on the fly (infinite in
practice; we expose a nominal length so DataLoader can iterate epochs).
`FolderDataset` loads real labeled images so you can fine-tune on collected
handwriting. `collate_fn` pads a batch to equal width and packs CTC targets.
"""
from __future__ import annotations

import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from . import synth
from .charset import encode
from .config import IMG_HEIGHT


def to_tensor(arr_uint8: np.ndarray) -> torch.Tensor:
    """(H, W) uint8 -> (1, H, W) float tensor normalized to roughly [-1, 1].

    The SAME normalization is used at inference time (see infer.py), so keep
    these in sync if you change it.
    """
    x = arr_uint8.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    return torch.from_numpy(x).unsqueeze(0)


class SyntheticDataset(Dataset):
    def __init__(self, length: int = 50000):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        arr, label = synth.make_sample()
        return to_tensor(arr), torch.tensor(encode(label), dtype=torch.long), label


class FolderDataset(Dataset):
    """Real labeled data. `labels_file` is lines of: <filename>\\t<label>.

    With ``augment=True`` each access applies the same synthetic augmentation
    (rotation, shear, stroke thickness, noise, blur) to the real image, so a
    small hand-collected set (~30 photos) yields lots of varied training views
    and the model doesn't just memorize those exact crops.
    """

    def __init__(self, image_dir: str, labels_file: str, augment: bool = True):
        self.image_dir = image_dir
        self.augment = augment
        self.items: list[tuple[str, str]] = []
        with open(labels_file, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line or "\t" not in line:
                    continue
                fname, label = line.split("\t", 1)
                self.items.append((fname, label))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fname, label = self.items[idx]
        img = Image.open(os.path.join(self.image_dir, fname)).convert("L")
        if self.augment:
            img = synth._augment(img)
        arr = synth.to_fixed_height(img, IMG_HEIGHT)
        return to_tensor(arr), torch.tensor(encode(label), dtype=torch.long), label


def collate_fn(batch):
    """Pad images to the batch's max width; pack targets for CTCLoss."""
    imgs, targets, labels = zip(*batch)
    max_w = max(t.shape[2] for t in imgs)
    padded = torch.full((len(imgs), 1, IMG_HEIGHT, max_w), 1.0)  # pad with white(=1)
    for i, im in enumerate(imgs):
        padded[i, :, :, : im.shape[2]] = im
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    targets_cat = torch.cat(targets) if targets else torch.zeros(0, dtype=torch.long)
    return padded, targets_cat, target_lengths, list(labels)
