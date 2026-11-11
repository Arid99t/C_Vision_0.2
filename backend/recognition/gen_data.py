"""Pre-generate recognizer training / benchmark sets to disk.

Samples are rendered by a pool of light worker processes (numpy + Pillow +
OpenCV only, no torch) and written to one flat file, so the training process
can stay single and small — this matters on a machine with little free RAM.

A set <name> is two files in config.RECDATA_DIR:

    <name>.bin   uint8 pixels: every sample is (32, W), stored row-major, back to back
    <name>.npz   widths (uint16), labels (one '\\n'-joined string), modes

Usage:
    python -m backend.recognition.gen_data --name train --n 400000
    python -m backend.recognition.gen_data --name bench_fonts  --split test --mode font  --n 3000 --seed 7
    python -m backend.recognition.gen_data --name bench_glyphs --split test --mode glyph --n 3000 --seed 8
"""
from __future__ import annotations

import argparse
import os
import time
from multiprocessing import Pool

import numpy as np

from .config import IMG_HEIGHT, RECDATA_DIR

CHUNK = 500


def _work(job):
    split, seed, n, mode, level = job
    from .synthv2 import Synth          # imported in the worker, not the parent
    gen = Synth(split, seed=seed, level=level)
    pixels, widths, labels, modes = [], [], [], []
    failures = 0
    while len(widths) < n:
        try:
            arr, label, how = gen.sample(mode=mode)
        except Exception:                 # one odd sample must not kill a long run...
            failures += 1
            if failures > max(5, n // 50):   # ...but a systematic fault must surface
                raise
            continue
        pixels.append(arr.tobytes())
        widths.append(arr.shape[1])
        labels.append(label)
        modes.append(how[0])
    return b"".join(pixels), widths, labels, modes, failures


def generate(name: str, n: int, split: str = "train", seed: int = 0, procs: int = 4,
             mode: str | None = None, level: float = 1.0) -> str:
    os.makedirs(RECDATA_DIR, exist_ok=True)
    bin_path = os.path.join(RECDATA_DIR, f"{name}.bin")
    jobs = [(split, seed * 100003 + i, min(CHUNK, n - i * CHUNK), mode, level)
            for i in range((n + CHUNK - 1) // CHUNK)]
    widths, labels, modes = [], [], []
    start, done, skipped = time.time(), 0, 0
    with open(bin_path, "wb") as f, Pool(procs) as pool:
        # imap (ordered) keeps the set identical from run to run for a given seed
        for blob, w, lab, md, failed in pool.imap(_work, jobs):
            f.write(blob)
            widths += w
            labels += lab
            modes += md
            done += len(w)
            skipped += failed
            if done % 20000 < CHUNK:
                rate = done / (time.time() - start)
                print(f"  {done}/{n}  {rate:.0f}/s  eta {(n - done) / max(rate, 1):.0f}s", flush=True)
    np.savez(os.path.join(RECDATA_DIR, f"{name}.npz"),
             widths=np.asarray(widths, np.uint16), labels="\n".join(labels),
             modes="".join(modes))
    size = os.path.getsize(bin_path) / 1e6
    print(f"{name}: {done} samples, {size:.0f} MB, {time.time() - start:.0f}s, "
          f"{skipped} failed renders skipped -> {bin_path}")
    return bin_path


class Store:
    """Read-only view of a generated set (memory-mapped pixels)."""

    def __init__(self, name: str):
        base = os.path.join(RECDATA_DIR, name)
        if not os.path.exists(base + ".bin"):
            raise FileNotFoundError(f"No generated set '{name}' in {RECDATA_DIR}. "
                                    f"Run: python -m backend.recognition.gen_data --name {name}")
        meta = np.load(base + ".npz")
        self.widths = meta["widths"].astype(np.int64)
        self.labels = str(meta["labels"]).split("\n")
        self.modes = str(meta["modes"])
        self.offsets = np.concatenate([[0], np.cumsum(self.widths * IMG_HEIGHT)])
        self.pixels = np.memmap(base + ".bin", dtype=np.uint8, mode="r")
        assert len(self.labels) == self.widths.size, "index and labels disagree"

    def __len__(self) -> int:
        return int(self.widths.size)

    def image(self, i: int) -> np.ndarray:
        w = int(self.widths[i])
        o = int(self.offsets[i])
        return np.asarray(self.pixels[o:o + IMG_HEIGHT * w]).reshape(IMG_HEIGHT, w)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Pre-generate recognizer data")
    p.add_argument("--name", required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--procs", type=int, default=4)
    p.add_argument("--mode", default=None, choices=["font", "glyph"])
    p.add_argument("--level", type=float, default=1.0, help="harshness of the camera model")
    a = p.parse_args()
    generate(a.name, a.n, a.split, a.seed, a.procs, a.mode, a.level)
