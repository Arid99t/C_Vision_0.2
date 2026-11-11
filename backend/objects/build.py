"""Build the on-device object model from vocabulary.json.

    .venv\\Scripts\\python.exe -m backend.objects.build            (default size)
    .venv\\Scripts\\python.exe -m backend.objects.build --size s   (bigger, more accurate, slower)

How it works: YOLOE is an OPEN-VOCABULARY detector — instead of a fixed list of
classes it is told what to find in words. Here the prompts of vocabulary.json
are encoded once with its text encoder and baked into the detection head, and
the result is exported as an ordinary fixed-class ONNX model: the phone needs
neither the text encoder nor PyTorch.

Writes  backend/static/models/objects.onnx   the detector
        backend/static/models/objects.json   prompts -> labels, thresholds, sizes

Pretrained weights and the text encoder (~270 MB) are downloaded on first use
into <work dir>/ultra (see recognition/config.WORK_DIR), not into the repo.

Licence note: YOLOE and the Ultralytics package are AGPL-3.0. That is fine for
personal, research and classroom use; distributing an app that includes this
model requires complying with the AGPL or an Ultralytics licence.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time

# Ultralytics will otherwise pip-install "missing" packages on the fly, which
# can silently replace PyTorch in the active environment.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

from backend.recognition.config import WORK_DIR  # noqa: E402

from .detect import META_PATH, MODEL_PATH, MODELS_DIR  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
VOCAB_PATH = os.path.join(_HERE, "vocabulary.json")
THRESHOLDS_PATH = os.path.join(_HERE, "thresholds.json")   # written by evaluate.py
ULTRA_DIR = os.path.join(WORK_DIR, "ultra")
DEFAULT_THRESHOLD = 0.4     # for labels evaluate.py has no ground truth to calibrate
MIN_SHOW_PRECISION = 0.75   # labels measured as less reliable than this are not drawn


def apply_calibration(meta_labels: list, calibration: dict) -> None:
    """Set each label's threshold from evaluate.py's measurements, and stop
    drawing labels that were measured as unreliable: a wrong name on screen is
    worse than no name. vocabulary.json can force one back on with
    "show": true."""
    for lab in meta_labels:
        cal = calibration.get(lab["label"])
        if not cal:
            lab.setdefault("threshold", DEFAULT_THRESHOLD)
            lab["calibrated"] = False
            continue
        lab.update(threshold=float(cal["threshold"]), calibrated=True,
                   precision=cal.get("precision"), recall=cal.get("recall"))
        lab.pop("hidden_reason", None)
        if cal.get("precision") is not None and cal["precision"] < MIN_SHOW_PRECISION:
            if lab.get("forced"):
                continue
            lab["show"] = False
            lab["hidden_reason"] = f"measured precision {cal['precision']:.0%}"
        elif lab.get("wanted", True):
            lab["show"] = True


def load_vocabulary() -> list:
    with open(VOCAB_PATH, encoding="utf-8") as f:
        labels = json.load(f)["labels"]
    seen = set()
    for lab in labels:
        for p in lab["prompts"]:
            if p in seen:
                raise ValueError(f"prompt {p!r} is listed twice in vocabulary.json")
            seen.add(p)
    return labels


def build(size: str = "n", long_edge: int = 640, name: str = "objects") -> dict:
    """Export YOLOE-26<size> with the vocabulary baked in as
    static/models/<name>.onnx + <name>.json ("objects" is what the app loads)."""
    out_model = os.path.join(MODELS_DIR, name + ".onnx")
    out_meta = os.path.join(MODELS_DIR, name + ".json")
    import numpy as np  # noqa: F401  (before torch: OpenMP DLL load order on Windows)
    from ultralytics import YOLOE, settings

    weights_dir = os.path.join(ULTRA_DIR, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    settings.update({"weights_dir": weights_dir,
                     "datasets_dir": os.path.join(ULTRA_DIR, "datasets"),
                     "runs_dir": os.path.join(ULTRA_DIR, "runs")})
    labels = load_vocabulary()
    prompts = [p for lab in labels for p in lab["prompts"]]

    here = os.getcwd()
    os.chdir(weights_dir)                      # downloads land in the current directory
    try:
        t0 = time.time()
        seg_path = os.path.join(weights_dir, f"yoloe-26{size}-seg.pt")
        seg = YOLOE(seg_path if os.path.exists(seg_path) else f"yoloe-26{size}-seg.pt")
        embeddings = seg.get_text_pe(prompts)          # encode the prompts once
        det = YOLOE(f"yoloe-26{size}.yaml")            # detection-only twin of the checkpoint
        det.load(seg_path if os.path.exists(seg_path) else f"yoloe-26{size}-seg.pt")
        det.set_classes(prompts, embeddings)           # bake the vocabulary into the head
        exported = det.export(format="onnx", imgsz=long_edge, dynamic=True, simplify=True,
                              opset=17, device="cpu", verbose=False)
        params = sum(p.numel() for p in det.model.parameters())
    finally:
        os.chdir(here)

    os.makedirs(MODELS_DIR, exist_ok=True)
    shutil.copyfile(exported, out_model)

    calibration = {}
    if os.path.exists(THRESHOLDS_PATH):
        with open(THRESHOLDS_PATH, encoding="utf-8") as f:
            calibration = json.load(f).get(f"yoloe-26{size}", {})
    meta_labels, k = [], 0
    for i, lab in enumerate(labels):
        idx = list(range(k, k + len(lab["prompts"])))
        k += len(lab["prompts"])
        meta_labels.append({
            "label": lab["label"], "prompts": idx,
            "show": lab.get("show", True),
            "wanted": lab.get("show", True),     # what vocabulary.json asked for
            "forced": lab.get("show") is True,   # "show": true written out = show regardless
            "threshold": DEFAULT_THRESHOLD, "calibrated": False,
            "hue": int((i * 137.5) % 360),
        })
    apply_calibration(meta_labels, calibration)
    meta = {
        "model": f"yoloe-26{size}", "licence": "AGPL-3.0 (Ultralytics YOLOE)",
        "params": params, "bytes": os.path.getsize(out_model),
        "file": os.path.basename(out_model), "long_edge": long_edge, "stride": 32,
        "nms_iou": 0.5, "cross_iou": 0.8, "prompts": prompts, "labels": meta_labels,
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(f"{meta['model']}: {len(labels)} labels from {len(prompts)} prompts, "
          f"{params / 1e6:.1f} M parameters, {meta['bytes'] / 1e6:.1f} MB, "
          f"built in {time.time() - t0:.0f}s -> {out_model}")
    return meta


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build the on-device object model")
    p.add_argument("--size", default="n", choices=["n", "s", "m", "l"],
                   help="YOLOE-26 size: n = fastest (default), s/m/l = more accurate, slower")
    p.add_argument("--long-edge", type=int, default=640, help="default input size")
    p.add_argument("--name", default="objects",
                   help="output file stem; the app loads 'objects' (use another to compare sizes)")
    a = p.parse_args()
    build(a.size, a.long_edge, a.name)
