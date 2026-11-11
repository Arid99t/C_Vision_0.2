"""EMNIST glyph bank: real handwritten letters and digits to build formulas from.

Fonts give one fixed shape per character. Real handwriting varies from one
character to the next, so the v2 data generator (synthv2.py) also composes
formulas out of individual handwritten characters taken from EMNIST "ByClass"
(NIST Special Database 19: 62 classes, digits + upper + lower case).

The raw download is 560 MB and lives outside the repo (see config.WORK_DIR).
`build()` distils it once into two small arrays per split, which every data
worker then memory-maps:

    glyphs_<split>.npy   (N, 28, 28) uint8, ink is bright
    chars_<split>.npy    (N,) uint8 ASCII codes

The official train/test split is kept, so the "test" glyphs come from samples
the model never trains on.

Build:  python -m backend.recognition.glyphs
"""
from __future__ import annotations

import gzip
import os
import struct
import zipfile

import numpy as np

from .config import EMNIST_ZIP, GLYPH_DIR

EMNIST_URL = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
PER_CLASS = {"train": 2500, "test": 400}


def _member(z: zipfile.ZipFile, name: str):
    return gzip.GzipFile(fileobj=z.open(name))


def _paths(split: str):
    return (os.path.join(GLYPH_DIR, f"glyphs_{split}.npy"),
            os.path.join(GLYPH_DIR, f"chars_{split}.npy"))


def build(per_class: dict | None = None, seed: int = 0) -> None:
    """Distil the EMNIST zip into the compact per-split glyph arrays."""
    per_class = per_class or PER_CLASS
    if not os.path.exists(EMNIST_ZIP):
        raise FileNotFoundError(f"{EMNIST_ZIP} not found. Download {EMNIST_URL} there first.")
    rng = np.random.default_rng(seed)
    with zipfile.ZipFile(EMNIST_ZIP) as z:
        mapping = {}
        for line in z.read("gzip/emnist-byclass-mapping.txt").decode().splitlines():
            if line.strip():
                cls, code = line.split()
                mapping[int(cls)] = int(code)

        for split, k in per_class.items():
            with _member(z, f"gzip/emnist-byclass-{split}-labels-idx1-ubyte.gz") as f:
                _, n = struct.unpack(">II", f.read(8))
                labels = np.frombuffer(f.read(n), np.uint8)

            # take 2x what we need per class, then drop degenerate glyphs below
            want = []
            for cls in sorted(mapping):
                idx = np.nonzero(labels == cls)[0]
                if idx.size > 2 * k:
                    idx = rng.choice(idx, 2 * k, replace=False)
                want.append(idx)
            want = np.sort(np.concatenate(want))

            # stream the (large) image file, keeping only the wanted records
            imgs = np.empty((want.size, 28, 28), np.uint8)
            with _member(z, f"gzip/emnist-byclass-{split}-images-idx3-ubyte.gz") as f:
                _, total, rows, cols = struct.unpack(">IIII", f.read(16))
                pos = j = 0
                while pos < total and j < want.size:
                    cnt = min(20000, total - pos)
                    buf = np.frombuffer(f.read(cnt * rows * cols), np.uint8).reshape(cnt, rows, cols)
                    hi = int(np.searchsorted(want, pos + cnt))
                    # EMNIST stores images transposed; flip them upright here
                    imgs[j:hi] = buf[want[j:hi] - pos].transpose(0, 2, 1)
                    j, pos = hi, pos + cnt

            chars = np.array([mapping[c] for c in labels[want]], np.uint8)
            # quality filter: enough ink to be a character, not a solid blob
            area = (imgs > 80).sum(axis=(1, 2))
            ok = (area >= 28) & (area <= 430)
            imgs, chars = imgs[ok], chars[ok]
            keep = []
            for code in np.unique(chars):
                idx = np.nonzero(chars == code)[0]
                keep.append(idx[:k])
            keep = np.sort(np.concatenate(keep))
            imgs, chars = imgs[keep], chars[keep]

            os.makedirs(GLYPH_DIR, exist_ok=True)
            gpath, cpath = _paths(split)
            np.save(gpath, imgs)
            np.save(cpath, chars)
            counts = np.bincount(chars, minlength=128)
            least = min(int(counts[c]) for c in np.unique(chars))
            print(f"{split}: {imgs.shape[0]} glyphs, {np.unique(chars).size} classes, "
                  f"smallest class {least} -> {gpath}")


def available(split: str = "train") -> bool:
    return all(os.path.exists(p) for p in _paths(split))


class GlyphBank:
    """Memory-mapped handwritten glyphs, sampled by character."""

    def __init__(self, split: str = "train"):
        gpath, cpath = _paths(split)
        if not (os.path.exists(gpath) and os.path.exists(cpath)):
            raise FileNotFoundError(
                f"Glyph bank for '{split}' not built. Run: python -m backend.recognition.glyphs")
        self.imgs = np.load(gpath, mmap_mode="r")
        chars = np.load(cpath)
        self.index = {chr(int(c)): np.nonzero(chars == c)[0] for c in np.unique(chars)}

    def has(self, ch: str) -> bool:
        return ch in self.index

    def sample(self, ch: str, rng) -> np.ndarray:
        """One (28, 28) uint8 glyph for `ch` (bright ink on black)."""
        idx = self.index[ch]
        return np.asarray(self.imgs[int(idx[rng.randrange(idx.size)])])


if __name__ == "__main__":
    build()
