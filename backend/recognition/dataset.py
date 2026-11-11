"""Datasets + collation helpers.

`FolderDataset` loads real labeled handwriting (collected with
clients/desktop/capture.py) through the SAME preparation the mobile recognizer
uses at inference (preprocess.prepare_crop), optionally re-augmented on every
access. train.py mixes these samples into its batches when you pass
--real-dir / --real-labels.

`SyntheticDataset` and `collate_fn` are the original on-the-fly v1 pipeline for
the legacy CRNN; the mobile recognizer trains from a pre-generated set instead
(see gen_data.py).
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from . import preprocess, synth
from .charset import encode
from .config import IMG_HEIGHT


def to_tensor(arr_uint8: np.ndarray) -> torch.Tensor:
    """(H, W) uint8 -> (1, H, W) float tensor normalized to roughly [-1, 1].

    The SAME normalization is used at inference time (preprocess.to_model_input),
    so keep these in sync if you change it.
    """
    x = arr_uint8.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    return torch.from_numpy(x).unsqueeze(0)


class SyntheticDataset(Dataset):
    """Legacy v1 samples (fonts on a clean page) for the original CRNN, passed
    through the v1 crop cleanup that the CRNN's inference path applies."""

    def __init__(self, length: int = 50000):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        arr, label = synth.make_sample(clean=True)
        return to_tensor(arr), torch.tensor(encode(label), dtype=torch.long), label


class FolderDataset(Dataset):
    """Real labeled data. `labels_file` is lines of: <filename>\\t<label>.

    With ``augment=True`` each access lightly "re-photographs" the real image
    (tilt, zoom, lighting, blur, noise), so a small hand-collected set (~30
    photos) yields lots of varied training views and the model doesn't just
    memorize those exact crops.
    """

    def __init__(self, image_dir: str, labels_file: str, augment: bool = True):
        self.image_dir = image_dir
        self.augment = augment
        self.items: list[tuple[str, str]] = []
        self._rng = random.Random(0)
        self._nprng = np.random.default_rng(0)
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
        img = np.asarray(Image.open(os.path.join(self.image_dir, fname)).convert("L"))
        if self.augment:
            from . import synthv2          # needs OpenCV; only when augmenting
            img = synthv2.augment_photo(img, self._rng, self._nprng)
        arr = preprocess.prepare_crop(img)  # same preparation as inference
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
