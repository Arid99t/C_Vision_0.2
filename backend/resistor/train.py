"""Train the resistor band reader.

    python tools/fetch_resistor_photos.py                (once: real photos for training and testing)
    python -m backend.resistor.render --make 120000      (once: writes the training set to disk)
    python -m backend.resistor.train --batch 96 --epochs 8

The training set (drawings plus re-photographed real photos, see render.py) is
read from disk chunk by chunk, with extra variation added on the GPU. On the
development GTX 1650, 115,000 training images took about 17 minutes per epoch
(132.4 minutes for eight epochs); timings depend on the machine and workload.
Three scores are printed each epoch:

  drawn   fresh drawings never trained on        (all bands right / value right)
  real    real photos never trained on, from <work>/resistor/real/test,
          each at three sizes                     (value right)
  quiet   pictures with no resistor that correctly gave no reading

The checkpoint kept is the one with the best `drawn` value score. Real photos
are excluded from training and checkpoint selection, but their scores are
visible during development. They are regression checks, not a blind test of
generalisation; new photos from the target camera still need evaluation.
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn

from ..recognition.config import WEIGHTS_DIR
from .bands import COLOURS, decode
from .model import BandReader, greedy
from .render import DATA_DIR, H, REAL_DIR, W

WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "resistor.pt")


def to_input(x_uint8: torch.Tensor) -> torch.Tensor:
    """(N, H, W, 3) uint8 -> (N, 3, H, W) float in [-1, 1]."""
    return x_uint8.permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)


def vary(x: torch.Tensor) -> torch.Tensor:
    """Cheap extra variety on the GPU: exposure, contrast, colour cast, noise, upside-down."""
    n = x.size(0)
    g = torch.empty(n, 1, 1, 1, device=x.device)
    x = (x + 1) * 0.5
    x = x * g.uniform_(0.85, 1.15)                                     # exposure
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * torch.empty_like(g).uniform_(0.85, 1.15) + mean   # contrast
    x = x * torch.empty(n, 3, 1, 1, device=x.device).uniform_(0.92, 1.08)   # colour cast
    x = x + torch.randn_like(x) * torch.empty_like(g).uniform_(0, 0.02)
    flip = torch.rand(n, device=x.device) < 0.5
    x[flip] = x[flip].flip(2)                                          # upside-down: the bands read the same
    return x.clamp_(0, 1) * 2 - 1


def mirror(x: np.ndarray, y: np.ndarray, n: np.ndarray, rng) -> None:
    """Turn half the pictures left-to-right (in place): the bands then read in reverse."""
    for i in np.nonzero(rng.random(len(x)) < 0.5)[0]:
        x[i] = x[i, :, ::-1]
        y[i, :n[i]] = y[i, :n[i]][::-1]


def value_matches(pred: list[int], target: list[int]) -> bool:
    want = decode([COLOURS[i - 1] for i in target])
    got = decode([COLOURS[i - 1] for i in pred]) if 3 <= len(pred) <= 5 else None
    return bool(want and got and abs(want["ohms"] - got["ohms"]) < 1e-9 * max(1.0, want["ohms"])
                and want["tolerance"] == got["tolerance"])


def score_drawn(model, x, y, n, device):
    model.eval()
    exact = value = parts = quiet = empties = 0
    with torch.no_grad():
        for i in range(0, len(x), 512):
            pred = greedy(model(to_input(torch.from_numpy(x[i:i + 512]).to(device))))
            for p, target, k in zip(pred, y[i:i + 512].tolist(), n[i:i + 512].tolist()):
                target = target[:k]
                exact += p == target
                if not target:
                    empties += 1
                    quiet += not p
                else:
                    parts += 1
                    value += value_matches(p, target)
    return {"exact": exact / len(x), "value": value / max(1, parts), "quiet": quiet / max(1, empties)}


def load_real_test():
    """Held-out real photos, each as taken and at 75 % and 55 % size on a white table."""
    xs, ys = [], []
    for path in sorted(glob.glob(os.path.join(REAL_DIR, "test", "*.jpg"))):
        names = [b for b in os.path.basename(path).split("_")[0].split("-") if b in COLOURS]
        rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        for fill in (1.0, 0.75, 0.55):
            canvas = np.full_like(rgb, 250)
            small = cv2.resize(rgb, (int(w * fill), int(h * fill)), interpolation=cv2.INTER_AREA)
            y0, x0 = (h - small.shape[0]) // 2, (w - small.shape[1]) // 2
            canvas[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
            xs.append(cv2.resize(canvas, (W, H), interpolation=cv2.INTER_AREA))
            ys.append([1 + COLOURS.index(b) for b in names])
    return (np.stack(xs), ys) if xs else (None, [])


def score_real(model, real_x, real_y, device):
    if real_x is None:
        return float("nan")
    model.eval()
    with torch.no_grad():
        pred = greedy(model(to_input(torch.from_numpy(real_x).to(device))))
    return sum(value_matches(p, t) for p, t in zip(pred, real_y)) / len(real_y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=192)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--data", default=DATA_DIR)
    ap.add_argument("--out", default=WEIGHTS_PATH)
    args = ap.parse_args()

    chunks = sorted(glob.glob(os.path.join(args.data, "chunk_*.npz")))
    if len(chunks) < 2:
        raise SystemExit(f"No training set in {args.data}. Run: python -m backend.resistor.render --make 120000")
    val = np.load(chunks[-1])
    val_x, val_y, val_n = val["x"], val["y"], val["n"]
    train_chunks = chunks[:-1]
    real_x, real_y = load_real_test()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BandReader().to(device)
    per_chunk = len(val_x)
    steps = args.epochs * len(train_chunks) * (per_chunk // args.batch)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    rng = np.random.default_rng(0)
    print(f"device {device} | {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters | "
          f"{len(train_chunks) * per_chunk} training pictures | {len(real_y)} real test views", flush=True)

    best, started = -1.0, time.time()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running, seen = time.time(), 0.0, 0
        for ci in rng.permutation(len(train_chunks)):
            data = np.load(train_chunks[ci])
            x, y, n = data["x"].copy(), data["y"].copy(), data["n"]
            mirror(x, y, n, rng)
            order = rng.permutation(len(x))
            for i in range(0, len(order) - args.batch + 1, args.batch):
                idx = np.sort(order[i:i + args.batch])
                xb = vary(to_input(torch.from_numpy(x[idx]).to(device)))
                nb = torch.from_numpy(n[idx])
                yb = torch.from_numpy(y[idx])
                targets = torch.cat([row[:k] for row, k in zip(yb, nb.tolist())]) if int(nb.sum()) else yb.new_zeros(0)
                logp = model(xb)
                loss = ctc(logp, targets, torch.full((len(idx),), logp.size(0), dtype=torch.long), nb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                sched.step()
                running += loss.item() * len(idx)
                seen += len(idx)
        drawn = score_drawn(model, val_x, val_y, val_n, device)
        real = score_real(model, real_x, real_y, device)
        note = ""
        if drawn["value"] > best:
            best = drawn["value"]
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "metrics": {**drawn, "real_test_value": real, "real_test_views": len(real_y)}}, args.out)
            note = "  saved"
        eta = (time.time() - started) / epoch * (args.epochs - epoch) / 60
        print(f"epoch {epoch}/{args.epochs}  loss {running / max(1, seen):.4f}  drawn {drawn['exact']:.3f}/{drawn['value']:.3f}  "
              f"real {real:.3f}  quiet {drawn['quiet']:.3f}  {time.time() - t0:.0f}s  ETA {eta:.0f} min{note}", flush=True)
    print(f"Done in {(time.time() - started) / 60:.1f} min. Kept the checkpoint with drawn value accuracy {best:.3f} -> {args.out}")


if __name__ == "__main__":
    main()
