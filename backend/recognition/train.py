r"""Train a formula recognizer with CTC loss.

The mobile recognizer (model.MobileRec) trains on a PRE-GENERATED set of
camera-realistic samples (see synthv2.py / gen_data.py), read from one
memory-mapped file by a single process:

    python -m backend.recognition.glyphs                 # once: handwritten glyph bank
    python -m backend.recognition.fontbank               # once: handwriting fonts
    python -m backend.recognition.gen_data --name train --n 480000
    python -m backend.recognition.gen_data --name val --n 4000 --seed 5
    python -m backend.recognition.train                  # -> weights/rec_mobile.pt
    python -m backend.recognition.export                 # -> static/models/*.onnx

Fine-tune on YOUR handwriting (collect it with clients/desktop/capture.py):

    python -m backend.recognition.train --resume ^
        --real-dir data\real\images --real-labels data\real\labels.tsv

Each epoch is scored on held-out sets (fonts and handwritten glyphs the model
never trains on); the best checkpoint by that score is the one kept.
"""
from __future__ import annotations

import argparse
import copy
import math
import os
import random
import sys
import time

import numpy as np          # noqa: I001  (must precede torch: OpenMP DLL load order on Windows)
import torch
import torch.nn.functional as F

from .charset import ITOS, NUM_CLASSES, STOI, decode_greedy, encode
from .config import IMG_HEIGHT, MOBILE_WEIGHTS_PATH, REPO_ROOT, WEIGHTS_DIR
from .gen_data import Store
from .model import build_model, count_params

# progress log lives at the project root so it's easy to find and `Get-Content -Wait`
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


# --------------------------------------------------------------------------
# batching
# --------------------------------------------------------------------------
REF_WIDTH = 128   # a "--batch N" batch holds about N samples of this width


def width_batches(widths: np.ndarray, batch: int, rng: np.random.Generator) -> list:
    """Random batches whose members have similar widths (so little padding),
    sized by a PIXEL budget rather than a fixed count: about `batch` samples of
    REF_WIDTH, fewer when they are wide, more (up to 2x) when they are narrow.

    A fixed count made the widest batches three times as large as typical ones;
    on a 4 GB card that pushed memory over the limit and into system RAM,
    which is drastically slower. With a budget, memory use is flat."""
    order = rng.permutation(len(widths))
    chunk, budget = batch * 64, batch * REF_WIDTH
    out = []
    for s in range(0, len(order), chunk):
        part = order[s:s + chunk]
        part = part[np.argsort(widths[part], kind="stable")]
        padded = (widths[part] + 15) // 16 * 16        # ascending: the last one sets the width
        i = 0
        while i < len(part):
            j = i + 1
            while j < len(part) and j - i < 2 * batch and (j - i + 1) * int(padded[j]) <= budget:
                j += 1
            # a handful of leftovers makes a batch too small for stable batch-norm
            # statistics; skip it (they are picked up again in the next shuffle)
            if j - i >= max(2, batch // 16) or int(padded[i]) * 2 > budget:
                out.append(part[i:j])
            i = j
    return [out[i] for i in rng.permutation(len(out))]


def collate(images: list, labels: list):
    """Pad (32, W) uint8 images with white to a common width (multiple of 16)."""
    wmax = (max(a.shape[1] for a in images) + 15) // 16 * 16
    x = np.full((len(images), 1, IMG_HEIGHT, wmax), 255, np.uint8)
    for i, a in enumerate(images):
        x[i, 0, :, :a.shape[1]] = a
    targets = [encode(lab) for lab in labels]
    return (torch.from_numpy(x),
            torch.tensor([a.shape[1] for a in images], dtype=torch.long),
            torch.tensor([c for t in targets for c in t], dtype=torch.long),
            torch.tensor([len(t) for t in targets], dtype=torch.long))


def to_device(x_u8, widths, device, augment: bool):
    """uint8 batch -> normalized float on `device`, with light GPU augmentation.

    The stored samples already carry the heavy camera degradations; this only
    keeps the model from memorising exact pixels: a horizontal stretch, a
    contrast/brightness nudge and a little noise.
    """
    x = x_u8.to(device, non_blocking=True).float().div_(255.0).sub_(0.5).div_(0.5)
    widths = widths.to(device)
    if augment:
        f = random.uniform(0.88, 1.12)
        old_w = x.shape[3]
        new_w = max(16, int(round(old_w * f / 8.0)) * 8)
        if new_w != old_w:
            x = F.interpolate(x, size=(IMG_HEIGHT, new_w), mode="bilinear", align_corners=False)
            widths = (widths.float() * new_w / old_w).round().clamp_(min=8).long()
        b = x.shape[0]
        gain = 1.0 + (torch.rand(b, 1, 1, 1, device=device) - 0.5) * 0.3
        bias = (torch.rand(b, 1, 1, 1, device=device) - 0.5) * 0.16
        noise = torch.randn_like(x) * (torch.rand(b, 1, 1, 1, device=device) * 0.06)
        x = (x * gain + bias + noise).clamp_(-1.0, 1.0)
    return x, widths


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@torch.no_grad()
def evaluate(model, store: Store, device, batch: int = 256, limit: int | None = None):
    """Greedy-decode a stored set. Returns (exact-match accuracy, character error rate)."""
    model.eval()
    n = len(store) if limit is None else min(limit, len(store))
    order = np.argsort(store.widths[:n], kind="stable")
    exact = edits = chars = 0
    for s in range(0, n, batch):
        idx = order[s:s + batch]
        labels = [store.labels[int(i)] for i in idx]
        x_u8, widths, _, _ = collate([store.image(int(i)) for i in idx], labels)
        x, widths = to_device(x_u8, widths, device, augment=False)
        pred = model(x).argmax(2).permute(1, 0).cpu().numpy()        # (B, T)
        lens = model.out_len(widths).clamp(max=pred.shape[1]).cpu().numpy()
        for row, t, label in zip(pred, lens, labels):
            text = decode_greedy(row[:int(t)].tolist())
            exact += text == label
            edits += edit_distance(text, label)
            chars += len(label)
    return exact / max(1, n), edits / max(1, chars)


# --------------------------------------------------------------------------
# real handwriting (optional fine-tuning data)
# --------------------------------------------------------------------------
class RealSet:
    """Labeled crops of real handwriting; every draw is re-augmented."""

    def __init__(self, image_dir: str, labels_file: str):
        from PIL import Image
        self.items = []
        with open(labels_file, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if "\t" not in line:
                    continue
                fname, label = line.split("\t", 1)
                label = "".join(c for c in label if c in STOI)
                path = os.path.join(image_dir, fname)
                if label and os.path.exists(path):
                    self.items.append((np.asarray(Image.open(path).convert("L")), label))
        if not self.items:
            raise RuntimeError(f"No usable samples listed in {labels_file}")
        self.rng = random.Random(0)
        self.nprng = np.random.default_rng(0)

    def __len__(self):
        return len(self.items)

    def sample(self):
        from . import preprocess, synthv2
        img, label = self.items[self.rng.randrange(len(self.items))]
        shot = synthv2.augment_photo(img, self.rng, self.nprng)
        return preprocess.prepare_crop(shot), label


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------
def train(args):
    log, fh = _make_logger()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_path = args.out or (MOBILE_WEIGHTS_PATH if args.arch == "mobile"
                            else os.path.join(WEIGHTS_DIR, "crnn_v2data.pt"))
    finetune = args.resume or bool(args.init)
    epochs = args.epochs or (6 if finetune else 14)
    lr = args.lr or (3e-4 if finetune else 2e-3)

    store = Store(args.data)
    evals = {name: Store(name) for name in args.eval.split(",") if name}
    real = RealSet(args.real_dir, args.real_labels) if args.real_dir and args.real_labels else None
    nprng = np.random.default_rng(args.seed)
    # one full pass by default; short epochs when fine-tuning on a handful of real samples
    full_pass = len(width_batches(store.widths, args.batch, nprng))
    steps_per_epoch = args.steps or (min(600, full_pass) if real is not None else full_pass)

    model = build_model(device, args.arch, args.size)
    init = args.init or (out_path if args.resume else None)
    if init:
        if not os.path.exists(init):
            raise FileNotFoundError(f"--resume/--init: no checkpoint at {init}")
        model.load_state_dict(torch.load(init, map_location=device)["model"])
        log(f"Initialised from {init}")
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    for m in ema.modules():                     # deepcopy leaves RNN weights unflattened
        if isinstance(m, torch.nn.RNNBase):
            m.flatten_parameters()

    log(f"Device: {device} | arch: {args.arch}/{args.size} | params: {count_params(model):,} | "
        f"classes: {NUM_CLASSES}")
    log(f"Data: {args.data} ({len(store):,} samples) | {epochs} epochs x {steps_per_epoch} steps "
        f"x batch {args.batch} | lr {lr:g} | log: {LOG_PATH}")
    if real:
        log(f"Mixing in {len(real)} real samples ({args.real_frac:.0%} of every batch).")

    total_steps = epochs * steps_per_epoch
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.wd)
    warmup = max(0.06, min(0.3, 500.0 / total_steps))        # at least ~500 warm-up steps
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=total_steps, pct_start=warmup,
        div_factor=20.0, final_div_factor=200.0)
    # Mixed precision is opt-in: GTX 16xx cards (no tensor cores) are known to
    # return NaNs in half precision, and this model is small enough for float32.
    amp = bool(args.amp) and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    # No cudnn autotuning: with dozens of input widths it keeps a workspace per
    # shape, which costs more memory than it saves time on this small model.
    torch.backends.cudnn.benchmark = False

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    def save(path, score):
        torch.save({"model": ema.state_dict(), "itos": ITOS, "img_height": IMG_HEIGHT,
                    "arch": args.arch, "size": args.size, "score": score}, path)

    best, step, skipped, start = -1.0, 0, 0, time.time()
    n_real = int(round(args.real_frac * args.batch)) if real else 0
    for epoch in range(1, epochs + 1):
        model.train()
        plan = width_batches(store.widths, args.batch, nprng)[:steps_per_epoch]
        running, counted, ep_start = 0.0, 0, time.time()
        for i, idx in enumerate(plan, 1):
            images = [store.image(int(j)) for j in idx]
            labels = [store.labels[int(j)] for j in idx]
            for k in range(min(n_real, len(images))):        # swap in real handwriting
                images[k], labels[k] = real.sample()
            x_u8, widths, targets, target_lens = collate(images, labels)
            x, widths = to_device(x_u8, widths, device, augment=True)
            with torch.autocast("cuda", enabled=amp):
                logp = model(x)                               # (T, B, C)
            in_lens = model.out_len(widths).clamp(max=logp.shape[0])
            loss = ctc(logp.float(), targets.to(device), in_lens, target_lens.to(device))
            opt.zero_grad(set_to_none=True)
            step += 1
            # Never let one bad batch poison the weights: a non-finite loss or
            # gradient skips the update (the LR schedule still advances).
            if not torch.isfinite(loss):
                skipped += 1
                if skipped > 50:
                    raise RuntimeError("Loss is repeatedly non-finite; lower --lr (and avoid --amp).")
                sched.step()
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if torch.isfinite(grad_norm):
                scaler.step(opt)
            else:
                skipped += 1
            scaler.update()
            sched.step()

            decay = min(0.999, (1.0 + step) / (10.0 + step))   # EMA of the weights
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(decay).add_(pm.detach(), alpha=1.0 - decay)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)

            running += loss.item()
            counted += 1
            if i % args.log_every == 0:
                rate = i / (time.time() - ep_start)
                log(f"  epoch {epoch}/{epochs}  step {i}/{len(plan)} ({100 * i / len(plan):3.0f}%)  "
                    f"loss {running / max(1, counted):.4f}  {rate:.1f} it/s")

        scores = {name: evaluate(ema, st, device) for name, st in evals.items()}
        mean_acc = float(np.mean([a for a, _ in scores.values()])) if scores else 0.0
        ep_time = time.time() - ep_start
        detail = " | ".join(f"{n}: {a:.1%} (CER {c:.2%})" for n, (a, c) in scores.items())
        note = f" | {skipped} non-finite steps skipped" if skipped else ""
        if device == "cuda":
            note += f" | GPU peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB"
            torch.cuda.empty_cache()
        log(f"epoch {epoch}/{epochs} done: avg_loss {running / max(1, counted):.4f} | {detail} | "
            f"{ep_time:.0f}s/epoch | ETA {ep_time * (epochs - epoch) / 60:.1f} min{note}")
        save(out_path.replace(".pt", "_last.pt"), mean_acc)
        if mean_acc >= best:
            best = mean_acc
            save(out_path, mean_acc)
            log(f"  saved best ({best:.1%}) -> {out_path}")

    log(f"Done in {(time.time() - start) / 60:.1f} min. Best held-out exact match: {best:.1%}")
    fh.close()


def build_argparser():
    p = argparse.ArgumentParser(description="Train a formula recognizer (CTC)")
    p.add_argument("--arch", default="mobile", choices=["mobile", "crnn"])
    p.add_argument("--size", default="base", choices=["base", "small"], help="mobile width preset")
    p.add_argument("--data", default="train", help="generated set to train on (see gen_data.py)")
    p.add_argument("--eval", default="val,bench_fonts,bench_glyphs",
                   help="comma-separated generated sets scored after every epoch")
    p.add_argument("--epochs", type=int, default=None, help="default 14 (6 when fine-tuning)")
    p.add_argument("--steps", type=int, default=None,
                   help="batches per epoch; default one full pass (600 with --real-dir)")
    p.add_argument("--batch", type=int, default=128,
                   help="samples per batch at 128 px width (batches are sized by total width)")
    p.add_argument("--lr", type=float, default=None, help="peak LR; default 2e-3 (3e-4 fine-tuning)")
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--device", default=None, help="cuda|cpu (auto if unset)")
    p.add_argument("--amp", action="store_true",
                   help="mixed precision (faster on RTX cards; can produce NaNs on GTX 16xx)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=250)
    p.add_argument("--resume", action="store_true", help="continue from the output checkpoint")
    p.add_argument("--init", default=None, help="start from this checkpoint instead")
    p.add_argument("--out", default=None, help="checkpoint path (default weights/rec_mobile.pt)")
    p.add_argument("--real-dir", default=None)
    p.add_argument("--real-labels", default=None)
    p.add_argument("--real-frac", type=float, default=0.25,
                   help="share of each batch drawn from the real samples")
    return p


if __name__ == "__main__":
    train(build_argparser().parse_args())
