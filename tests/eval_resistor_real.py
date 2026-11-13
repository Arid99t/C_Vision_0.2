"""Measure the resistor reader on REAL photos.

    python tests/eval_resistor_real.py                      the photos kept in tests/fixtures/resistors/
    python tests/eval_resistor_real.py --dir my_photos       your own

Each picture shows ONE resistor lying lengthwise (left-right), filling roughly
a third to most of the width, cropped to about 10:3 like the box in the app.
Name the file after the bands as you see them, left to right, then anything:

    yellow-violet-red-gold.jpg        brown-black-orange-gold_desk2.png

The script reports whether resistance and tolerance (after working out the
direction) both match, and repeats each crop at a few sizes, blurs and brightness levels to
show how much margin there is. The v2 model mixes drawings with augmented real
training photos. These evaluation photos are excluded from training, but were
inspected during development; this is a regression benchmark, not a blind
test. The seven repository fixtures also occur in the larger downloaded test
set, and transformed views are not independent photos. Neither score measures
accuracy through a head-worn camera.
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.resistor.bands import COLOURS, decode, format_ohms  # noqa: E402

MODEL = os.path.join(ROOT, "backend", "static", "models", "resistor.onnx")
FIXTURES = os.path.join(ROOT, "tests", "fixtures", "resistors")
H, W = 48, 160


def load_session():
    import onnxruntime as ort
    return ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])


def read(session, rgb: np.ndarray) -> list[str]:
    """RGB picture (any size, about 10:3) -> band names left to right."""
    img = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
    x = (img.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)[None]
    logp = session.run(None, {"image": x})[0][:, 0]
    out, prev = [], 0
    for k in logp.argmax(1):
        if k != prev and k != 0:
            out.append(COLOURS[k - 1])
        prev = k
    return out


def variants(rgb: np.ndarray):
    """The photo as taken, then smaller, blurred, lower resolution, darker, warmer and reversed."""
    h, w = rgb.shape[:2]
    yield "as taken", rgb
    for fill in (0.75, 0.55):                                   # the part smaller inside the box
        canvas = np.full_like(rgb, int(np.median(rgb[:4])))
        canvas[:] = np.median(rgb[:3].reshape(-1, 3), axis=0)
        small = cv2.resize(rgb, (int(w * fill), int(h * fill)), interpolation=cv2.INTER_AREA)
        y0, x0 = (h - small.shape[0]) // 2, (w - small.shape[1]) // 2
        canvas[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
        yield f"{int(fill * 100)} % size", canvas
    yield "blurred", cv2.GaussianBlur(rgb, (0, 0), max(1.0, w / 160 * 1.2))
    yield "low resolution", cv2.resize(cv2.resize(rgb, (80, 24), interpolation=cv2.INTER_AREA), (w, h))
    yield "darker", np.clip(rgb.astype(np.float32) * 0.6, 0, 255).astype(np.uint8)
    yield "warm light", np.clip(rgb.astype(np.float32) * np.array([1.12, 1.0, 0.8]), 0, 255).astype(np.uint8)
    yield "flipped", rgb[:, ::-1].copy()


def evaluate(folder: str, verbose: bool = True) -> dict:
    session = load_session()
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    totals: dict[str, list[int]] = {}
    for name in files:
        truth = [b for b in os.path.splitext(name)[0].split("_")[0].split("-") if b in COLOURS]
        want = decode(truth)
        if not want:
            print(f"skip {name}: the file name is not a legal colour code")
            continue
        rgb = cv2.cvtColor(cv2.imread(os.path.join(folder, name)), cv2.COLOR_BGR2RGB)
        row = []
        for label, img in variants(rgb):
            bands = read(session, img)
            got = decode(bands) if 3 <= len(bands) <= 5 else None
            ok = bool(got and abs(got["ohms"] - want["ohms"]) < 1e-9 * max(1.0, want["ohms"])
                      and got["tolerance"] == want["tolerance"])
            totals.setdefault(label, []).append(ok)
            row.append(f"{label}: {'ok' if ok else (format_ohms(got['ohms']) if got else 'none')}")
        if verbose:
            print(f"{name}  ({format_ohms(want['ohms'])} ±{want['tolerance']:g}%)\n    " + " | ".join(row))
    summary = {label: (sum(v), len(v)) for label, v in totals.items()}
    if verbose and summary:
        print("\nRight resistance and tolerance / crops:")
        for label, (ok, n) in summary.items():
            print(f"  {label:<16} {ok}/{n}")
    return summary


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):                 # the Windows console cannot print the ohm sign
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=FIXTURES)
    args = ap.parse_args()
    if not os.path.isdir(args.dir) or not os.path.exists(MODEL):
        sys.exit(f"Need photos in {args.dir} and the exported model {MODEL}")
    evaluate(args.dir)
