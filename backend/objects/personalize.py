r"""Teach the object spotter YOUR room.

The stock model (build.py) is told what to find in words and has never seen
your things, so it misses some and occasionally mislabels others. The reliable
way to get close to "no mistakes" for one room is to train a small model on
photos of that room. This script automates it:

  1. Take 100-300 photos of the room with the phone: every object you care
     about, from several angles and distances, in the lighting you will use.
     (A slow video walk-through works too.)

  2. Label them automatically with a LARGE teacher model on the PC:

         .venv\Scripts\python.exe -m backend.objects.personalize label --photos D:\room_photos
         .venv\Scripts\python.exe -m backend.objects.personalize label --video D:\room.mp4

     It writes a dataset to <work dir>\room and, for every photo, a preview
     with the boxes drawn in <work dir>\room\preview. LOOK THROUGH THE PREVIEWS:
     delete the photo + its .txt for any that are labelled wrongly (or fix the
     .txt). The student can only be as right as these labels.

  3. Train the small model on them and install it for the app:

         .venv\Scripts\python.exe -m backend.objects.personalize train --epochs 60

     The previous model is kept as objects_before_room.onnx / .json.

Needs the build tools (requirements-build.txt). Training is far faster with a
CUDA build of PyTorch in the venv; on CPU expect an hour or more.

Honest limits: the result is tuned to that room and those objects. It will be
much better there and no better (often worse) anywhere else, and it inherits
any labelling mistakes you leave in step 2.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time

os.environ.setdefault("YOLO_AUTOINSTALL", "false")   # see build.py

from .build import ULTRA_DIR, load_vocabulary  # noqa: E402
from .detect import MODELS_DIR, iou_one_to_many  # noqa: E402

ROOM_DIR = os.path.join(os.path.dirname(ULTRA_DIR), "room")
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _settings():
    from ultralytics import settings
    weights = os.path.join(ULTRA_DIR, "weights")
    os.makedirs(weights, exist_ok=True)
    settings.update({"weights_dir": weights, "datasets_dir": os.path.join(ULTRA_DIR, "datasets"),
                     "runs_dir": os.path.join(ULTRA_DIR, "runs")})
    return weights


def _device():
    import torch
    return 0 if torch.cuda.is_available() else "cpu"


def _gather(photos: str | None, video: str | None, every: int, long_edge: int = 1280) -> list:
    """Collect the frames to label as (name, BGR image), downscaled if huge."""
    import cv2
    frames = []
    if photos:
        for name in sorted(os.listdir(photos)):
            if name.lower().endswith(IMAGE_EXT):
                img = cv2.imread(os.path.join(photos, name))
                if img is not None:
                    frames.append((os.path.splitext(name)[0], img))
    if video:
        cap = cv2.VideoCapture(video)
        i = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if i % every == 0:
                frames.append((f"frame_{i:06d}", img))
            i += 1
        cap.release()
    out = []
    for name, img in frames:
        h, w = img.shape[:2]
        s = long_edge / max(h, w)
        if s < 1:
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        out.append((name, img))
    return out


def label(photos: str | None = None, video: str | None = None, teacher: str = "l",
          conf: float = 0.45, imgsz: int = 960, every: int = 15, val_share: float = 0.12) -> dict:
    """Step 2: pseudo-label the photos with a large open-vocabulary teacher."""
    import cv2
    import numpy as np
    from ultralytics import YOLOE

    weights = _settings()
    frames = _gather(photos, video, every)
    if not frames:
        raise SystemExit("no photos found (give --photos FOLDER and/or --video FILE)")
    labels = load_vocabulary()
    prompts = [p for lab in labels for p in lab["prompts"]]
    prompt_label = [i for i, lab in enumerate(labels) for _ in lab["prompts"]]

    here = os.getcwd()
    os.chdir(weights)
    try:
        path = os.path.join(weights, f"yoloe-26{teacher}-seg.pt")
        model = YOLOE(path if os.path.exists(path) else f"yoloe-26{teacher}-seg.pt")
        model.set_classes(prompts, model.get_text_pe(prompts))
    finally:
        os.chdir(here)

    if os.path.isdir(ROOM_DIR):
        shutil.rmtree(ROOM_DIR)
    for sub in ("images/train", "images/val", "labels/train", "labels/val", "preview"):
        os.makedirs(os.path.join(ROOM_DIR, sub), exist_ok=True)
    rng = random.Random(0)
    counts = {lab["label"]: 0 for lab in labels}
    t0, device = time.time(), _device()
    for k, (name, img) in enumerate(frames):
        r = model.predict(img, imgsz=imgsz, conf=0.2, device=device, verbose=False)[0]
        boxes = r.boxes.xyxy.cpu().numpy().reshape(-1, 4)
        cls = np.array([prompt_label[int(c)] for c in r.boxes.cls.cpu().numpy()], dtype=int)
        score = r.boxes.conf.cpu().numpy()
        order = np.argsort(-score)
        keep = []
        for i in order:                         # same suppression as the deployed decoder
            if score[i] < conf:
                continue
            if keep:
                ious = iou_one_to_many(boxes[i], boxes[keep])
                if (ious >= np.where(cls[keep] == cls[i], 0.5, 0.8)).any():
                    continue
            keep.append(int(i))
        split = "val" if rng.random() < val_share else "train"
        h, w = img.shape[:2]
        cv2.imwrite(os.path.join(ROOM_DIR, "images", split, name + ".jpg"), img)
        preview = img.copy()
        with open(os.path.join(ROOM_DIR, "labels", split, name + ".txt"), "w", encoding="utf-8") as f:
            for i in keep:
                x1, y1, x2, y2 = boxes[i]
                f.write(f"{cls[i]} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                        f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n")
                counts[labels[cls[i]]["label"]] += 1
                cv2.rectangle(preview, (int(x1), int(y1)), (int(x2), int(y2)), (60, 220, 60), 2)
                cv2.putText(preview, f"{labels[cls[i]]['label']} {score[i]:.2f}", (int(x1) + 3, max(16, int(y1) - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 220, 60), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(ROOM_DIR, "preview", name + ".jpg"), preview)
        if (k + 1) % 25 == 0:
            print(f"  labelled {k + 1}/{len(frames)} photos", flush=True)

    with open(os.path.join(ROOM_DIR, "dataset.yaml"), "w", encoding="utf-8") as f:
        f.write(f"path: {ROOM_DIR.replace(os.sep, '/')}\ntrain: images/train\nval: images/val\nnames:\n")
        for i, lab in enumerate(labels):
            f.write(f"  {i}: {lab['label']}\n")
    found = {k: v for k, v in counts.items() if v}
    print(f"\n{len(frames)} photos labelled by yoloe-26{teacher} in {time.time() - t0:.0f}s -> {ROOM_DIR}")
    print("objects found:", ", ".join(f"{k} {v}" for k, v in sorted(found.items(), key=lambda kv: -kv[1])) or "none")
    missing = [k for k, v in counts.items() if not v]
    print("never found:", ", ".join(missing))
    print(f"\nNow check {os.path.join(ROOM_DIR, 'preview')} and remove or fix wrong labels, then run:  "
          f"python -m backend.objects.personalize train")
    return counts


def train(epochs: int = 60, size: str = "n", imgsz: int = 640, batch: int = 8,
          name: str = "objects") -> dict:
    """Step 3: fine-tune a small closed-set detector on the room dataset and install it."""
    from ultralytics import YOLO

    data = os.path.join(ROOM_DIR, "dataset.yaml")
    if not os.path.exists(data):
        raise SystemExit("no room dataset yet: run the 'label' step first")
    weights = _settings()
    labels = load_vocabulary()
    here = os.getcwd()
    os.chdir(weights)
    try:
        model = YOLO(f"yolo11{size}.pt")
        model.train(data=data, epochs=epochs, imgsz=imgsz, batch=batch, workers=0, device=_device(),
                    project=os.path.join(ULTRA_DIR, "runs"), name="room", exist_ok=True,
                    patience=25, plots=False, verbose=False)
        best = os.path.join(ULTRA_DIR, "runs", "room", "weights", "best.pt")
        tuned = YOLO(best)
        metrics = tuned.val(data=data, imgsz=imgsz, device=_device(), workers=0, plots=False, verbose=False)
        exported = tuned.export(format="onnx", imgsz=imgsz, dynamic=True, simplify=True, opset=17,
                                device="cpu", verbose=False)
        params = sum(p.numel() for p in tuned.model.parameters())
    finally:
        os.chdir(here)

    out_model = os.path.join(MODELS_DIR, name + ".onnx")
    out_meta = os.path.join(MODELS_DIR, name + ".json")
    for path in (out_model, out_meta):               # keep the model being replaced
        if os.path.exists(path):
            base, ext = os.path.splitext(path)
            shutil.copyfile(path, base + "_before_room" + ext)
    shutil.copyfile(exported, out_model)
    meta = {
        "model": f"yolo11{size}-room", "licence": "AGPL-3.0 (Ultralytics YOLO11)",
        "params": params, "bytes": os.path.getsize(out_model), "file": os.path.basename(out_model),
        "long_edge": imgsz, "stride": 32, "nms_iou": 0.5, "cross_iou": 0.8,
        "prompts": [lab["label"] for lab in labels],
        "labels": [{"label": lab["label"], "prompts": [i], "show": lab.get("show", True),
                    "threshold": 0.5, "calibrated": False, "hue": int((i * 137.5) % 360)}
                   for i, lab in enumerate(labels)],
        "room_val_map50": round(float(metrics.box.map50), 3),
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(f"\nroom model: mAP50 {meta['room_val_map50']:.2f} on the held-out room photos, "
          f"{meta['bytes'] / 1e6:.1f} MB -> {out_model}")
    print("(that score is against the teacher's labels on photos of the same room, "
          "so it says the student learned them, not that they are right)")
    return meta


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Tune the object spotter to your own room")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("label", help="auto-label your photos with a large teacher model")
    a.add_argument("--photos", default=None, help="folder of photos")
    a.add_argument("--video", default=None, help="a video walk-through")
    a.add_argument("--every", type=int, default=15, help="use every Nth video frame")
    a.add_argument("--teacher", default="l", choices=["s", "m", "l", "x"], help="YOLOE-26 size")
    a.add_argument("--conf", type=float, default=0.45, help="keep labels at least this confident")
    a.add_argument("--imgsz", type=int, default=960)
    b = sub.add_parser("train", help="train the small model on the labelled photos")
    b.add_argument("--epochs", type=int, default=60)
    b.add_argument("--size", default="n", choices=["n", "s"])
    b.add_argument("--imgsz", type=int, default=640)
    b.add_argument("--batch", type=int, default=8)
    b.add_argument("--name", default="objects", help="output file stem (the app loads 'objects')")
    args = p.parse_args()
    if args.cmd == "label":
        label(args.photos, args.video, args.teacher, args.conf, args.imgsz, args.every)
    else:
        train(args.epochs, args.size, args.imgsz, args.batch, args.name)
