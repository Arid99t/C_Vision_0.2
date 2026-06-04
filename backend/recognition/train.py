"""Train the CRNN handwriting recognizer with CTC loss.

Quick start (synthetic data, CPU is fine for a first pass; GPU much faster):

    python -m backend.recognition.train --epochs 8 --steps 800 --batch 32

The checkpoint (model weights + charset + image height) is written to
backend/recognition/weights/crnn.pt, which the API/clients load automatically.

To fine-tune on real handwriting, collect images + a labels.tsv and pass
--real-dir / --real-labels; they get mixed with synthetic samples.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
from torch.utils.data import ConcatDataset, DataLoader

from .charset import ITOS, NUM_CLASSES, decode_greedy
from .config import IMG_HEIGHT, WEIGHTS_DIR, WEIGHTS_PATH
from .dataset import FolderDataset, SyntheticDataset, collate_fn
from .model import build_model


def _accuracy(model, batch, device) -> float:
    imgs, _, _, labels = batch
    with torch.no_grad():
        logp = model(imgs.to(device))            # (T, B, C)
        preds = logp.argmax(2).permute(1, 0)     # (B, T)
    correct = sum(decode_greedy(p.tolist()) == lab for p, lab in zip(preds, labels))
    return correct / len(labels)


# progress log lives at the project root so it's easy to find and `Get-Content -Wait`
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_PATH = os.path.join(REPO_ROOT, "training.log")


def _make_logger():
    """Return a log(msg) that prints (flushed) AND appends to training.log."""
    try:
        sys.stdout.reconfigure(line_buffering=True)  # stream even when redirected
    except (AttributeError, ValueError):
        pass
    fh = open(LOG_PATH, "w", encoding="utf-8")

    def log(msg=""):
        print(msg, flush=True)
        fh.write(msg + "\n")
        fh.flush()

    return log, fh


def train(args):
    log, _fh = _make_logger()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device} | classes: {NUM_CLASSES} | log file: {LOG_PATH}")

    datasets = [SyntheticDataset(length=args.steps * args.batch)]
    if args.real_dir and args.real_labels:
        real = FolderDataset(args.real_dir, args.real_labels)
        # oversample real data so it isn't drowned out by synthetic
        datasets += [real] * max(1, args.real_oversample)
        log(f"Mixing {len(real)} real samples (x{args.real_oversample}).")
    dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, collate_fn=collate_fn,
                        drop_last=True, persistent_workers=args.workers > 0)

    model = build_model(device)
    if args.resume and os.path.exists(WEIGHTS_PATH):
        ckpt = torch.load(WEIGHTS_PATH, map_location=device)
        model.load_state_dict(ckpt["model"])
        log("Resumed from existing checkpoint.")

    ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, args.epochs // 2), gamma=0.3)

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        ep_start = time.time()
        for step, (imgs, targets, target_lengths, labels) in enumerate(loader, 1):
            imgs = imgs.to(device)
            logp = model(imgs)                                   # (T, B, C)
            T, B, _ = logp.shape
            input_lengths = torch.full((B,), T, dtype=torch.long)
            loss = ctc(logp, targets.to(device), input_lengths, target_lengths)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            running += loss.item()
            if step % args.log_every == 0:
                pct = 100 * step / len(loader)
                log(f"  epoch {epoch}/{args.epochs}  step {step}/{len(loader)} "
                    f"({pct:3.0f}%)  loss {running / step:.4f}")
        sched.step()

        # quick eval on a fresh synthetic batch
        model.eval()
        eval_batch = collate_fn([SyntheticDataset(1)[0] for _ in range(args.batch)])
        acc = _accuracy(model, eval_batch, device)
        ep_time = time.time() - ep_start
        eta = ep_time * (args.epochs - epoch)
        log(f"epoch {epoch}/{args.epochs} done: avg_loss {running / len(loader):.4f} | "
            f"exact-match acc ~{acc:.1%} | {ep_time:.0f}s/epoch | ETA {eta/60:.1f} min")
        torch.save({"model": model.state_dict(), "itos": ITOS,
                    "img_height": IMG_HEIGHT}, WEIGHTS_PATH)
        log(f"  saved -> {WEIGHTS_PATH}")

    log(f"Done in {(time.time() - start)/60:.1f} min.")
    _fh.close()


def build_argparser():
    p = argparse.ArgumentParser(description="Train CRNN handwriting recognizer")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--steps", type=int, default=800, help="synthetic batches per epoch")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default=None, help="cuda|cpu (auto if unset)")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--real-dir", default=None)
    p.add_argument("--real-labels", default=None)
    p.add_argument("--real-oversample", type=int, default=20)
    return p


if __name__ == "__main__":
    train(build_argparser().parse_args())
