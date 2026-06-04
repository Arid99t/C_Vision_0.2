r"""Collect labeled samples of YOUR handwriting to fine-tune the recognizer.

The synthetic model learns from fonts; this captures real ink so it learns your
actual letters (so 'Na' stops reading as 'N4', 'H' as 'Cl', etc.).

Workflow:
  1. Run this, write a formula on paper, hold it to the webcam.
  2. The detected crop is boxed live. Press SPACE to save it under a label.
  3. Collect ~20-40 (a mix: H2, O2, Na, Cl2, CO2, H2O, ...), then fine-tune:

       python -m backend.recognition.train --resume ^
           --real-dir data\real\images --real-labels data\real\labels.tsv

Fast batch mode (recommended) — capture many of one formula, then switch:
    python -m clients.desktop.capture --label H2
    # write H2 a few ways, press SPACE each time; press n to type a new label

Keys:  SPACE save · n set/change label · u undo last · q quit
"""
from __future__ import annotations

import argparse
import os
import re

try:
    import cv2
except ImportError:  # pragma: no cover
    print("Missing opencv. Install: pip install opencv-python")
    raise SystemExit(1)

from clients.desktop.ar_app import detect_formula_regions, draw_label, GREEN, YELLOW

DEFAULT_IMG_DIR = os.path.join("data", "real", "images")
DEFAULT_LABELS = os.path.join("data", "real", "labels.tsv")


def _safe(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "_", label)


def save_sample(frame, label, img_dir, labels_file, pad=8) -> str | None:
    """Detect the formula in `frame`, crop it, save the image, append the label.

    Returns the saved filename, or None if nothing was detected to crop.
    """
    boxes = detect_formula_regions(frame, max_regions=1)
    if not boxes:
        return None
    x, y, w, h = boxes[0]
    H, W = frame.shape[:2]
    crop = frame[max(0, y - pad):min(H, y + h + pad), max(0, x - pad):min(W, x + w + pad)]
    if crop.size == 0:
        return None

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(os.path.dirname(labels_file) or ".", exist_ok=True)
    idx = sum(1 for _ in open(labels_file, encoding="utf-8")) if os.path.exists(labels_file) else 0
    fname = f"{idx:04d}_{_safe(label)}.png"
    cv2.imwrite(os.path.join(img_dir, fname), crop)
    with open(labels_file, "a", encoding="utf-8") as f:
        f.write(f"{fname}\t{label}\n")
    return fname


def _count(labels_file) -> int:
    if not os.path.exists(labels_file):
        return 0
    return sum(1 for line in open(labels_file, encoding="utf-8") if line.strip())


def _undo(img_dir, labels_file):
    """Remove the most recently saved sample (file + last tsv line)."""
    if not os.path.exists(labels_file):
        return None
    lines = [ln for ln in open(labels_file, encoding="utf-8") if ln.strip()]
    if not lines:
        return None
    last = lines.pop()
    fname = last.split("\t", 1)[0]
    try:
        os.remove(os.path.join(img_dir, fname))
    except OSError:
        pass
    with open(labels_file, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return fname


def run(args):
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}")
        return
    label = args.label
    if not label:
        label = input("Enter the formula label to capture (e.g. H2): ").strip()

    print("Keys: SPACE save | n new label | u undo | q quit")
    flash = ""
    flash_until = 0
    import time
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        view = frame.copy()
        boxes = detect_formula_regions(frame, max_regions=1)
        for (x, y, w, h) in boxes:
            cv2.rectangle(view, (x, y), (x + w, y + h), GREEN, 2)
        n = _count(args.labels)
        hud = f"label: {label or '(none)'}   saved: {n}"
        draw_label(view, hud, (10, 26), GREEN if boxes else YELLOW, 0.6)
        draw_label(view, "SPACE save  n label  u undo  q quit",
                   (10, view.shape[0] - 12), (255, 255, 255), 0.5)
        if time.time() < flash_until:
            draw_label(view, flash, (10, 52), YELLOW, 0.6)
        cv2.imshow("Chem_Vision capture", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            if not label:
                print("Set a label first (press n).")
                continue
            saved = save_sample(frame, label, args.img_dir, args.labels)
            if saved:
                flash, flash_until = f"saved {saved}", time.time() + 1.0
                print(f"saved {saved}")
            else:
                flash, flash_until = "no formula detected - reposition", time.time() + 1.0
        elif key == ord("n"):
            cv2.destroyWindow("Chem_Vision capture")
            label = input("New label: ").strip()
        elif key == ord("u"):
            removed = _undo(args.img_dir, args.labels)
            flash, flash_until = (f"undid {removed}" if removed else "nothing to undo"), time.time() + 1.0
            print(flash)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nTotal samples: {_count(args.labels)} -> {args.labels}")
    print("Fine-tune with:\n  python -m backend.recognition.train --resume "
          f"--real-dir {args.img_dir} --real-labels {args.labels}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Capture handwriting samples for fine-tuning")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--label", default="", help="capture all SPACE presses under this label")
    p.add_argument("--img-dir", default=DEFAULT_IMG_DIR)
    p.add_argument("--labels", default=DEFAULT_LABELS)
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
