"""Run the exported object model and decode its output.

This is the reference implementation of what static/js/objects.js does on the
phone — same letterboxing, same score merging, same suppression — so accuracy
measured here (evaluate.py) is the accuracy of the deployed pipeline.

Needs only numpy + onnxruntime (no PyTorch).

Model contract (backend/static/models/objects.onnx + objects.json):
  input  "images"  (1, 3, H, W) RGB in 0..1, H and W multiples of 32
  output           (1, 4 + P, N): per candidate box cx, cy, w, h in input
                   pixels, then one score (0..1) per PROMPT
Several prompts describe one on-screen LABEL; a label's score is the best of
its prompts.
"""
from __future__ import annotations

import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(os.path.dirname(_HERE), "static", "models")
MODEL_PATH = os.path.join(MODELS_DIR, "objects.onnx")
META_PATH = os.path.join(MODELS_DIR, "objects.json")
PAD = 114 / 255.0


def letterbox(rgb: np.ndarray, long_edge: int, stride: int = 32):
    """Scale so the longer side is `long_edge`, then pad right/bottom up to a
    multiple of `stride`. Returns (CHW float32 0..1, scale)."""
    h, w = rgb.shape[:2]
    scale = long_edge / max(h, w)
    nw, nh = max(1, int(w * scale + 0.5)), max(1, int(h * scale + 0.5))
    if (nw, nh) != (w, h):
        from PIL import Image
        rgb = np.asarray(Image.fromarray(rgb).resize((nw, nh), Image.BILINEAR))
    pw, ph = -(-nw // stride) * stride, -(-nh // stride) * stride
    canvas = np.full((ph, pw, 3), PAD, np.float32)
    canvas[:nh, :nw] = rgb.astype(np.float32) / 255.0
    return canvas.transpose(2, 0, 1)[None], scale


def iou_one_to_many(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area + areas - inter, 1e-9)


def decode(output: np.ndarray, meta: dict, scale: float, floor: float | None = None) -> list:
    """Raw model output -> [{'label', 'score', 'box': [x1, y1, x2, y2]}] in the
    ORIGINAL image's pixels, best first.

    `floor` overrides every label's own threshold (used when calibrating)."""
    out = output[0]                                   # (4 + P, N)
    boxes, scores = out[:4].T, out[4:].T              # (N, 4), (N, P)
    labels = meta["labels"]
    # label score = best of its prompts
    per_label = np.stack([scores[:, lab["prompts"]].max(axis=1) for lab in labels], axis=1)
    best = per_label.argmax(axis=1)
    conf = per_label[np.arange(len(best)), best]
    thr = np.array([lab["threshold"] for lab in labels])[best] if floor is None else floor
    keep = np.nonzero(conf >= thr)[0]
    if keep.size == 0:
        return []
    keep = keep[np.argsort(-conf[keep])][:400]
    cx, cy, w, h = boxes[keep].T
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1) / scale
    lab_idx, conf = best[keep], conf[keep]
    same, other = meta.get("nms_iou", 0.5), meta.get("cross_iou", 0.8)
    chosen = []
    for i in range(len(keep)):                        # greedy suppression, best first
        ok = True
        if chosen:
            ious = iou_one_to_many(xyxy[i], xyxy[chosen])
            limit = np.where(lab_idx[chosen] == lab_idx[i], same, other)
            ok = bool((ious < limit).all())
        if ok:
            chosen.append(i)
    return [{"label": labels[lab_idx[i]]["label"], "score": float(conf[i]),
             "box": [float(v) for v in xyxy[i]], "show": labels[lab_idx[i]].get("show", True)}
            for i in chosen]


class ObjectDetector:
    def __init__(self, model_path: str = MODEL_PATH, meta_path: str = META_PATH, threads: int = 4):
        import onnxruntime as ort
        with open(meta_path, encoding="utf-8") as f:
            self.meta = json.load(f)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def raw(self, rgb: np.ndarray, long_edge: int | None = None):
        x, scale = letterbox(rgb, long_edge or self.meta["long_edge"], self.meta.get("stride", 32))
        return self.session.run(None, {self.input_name: x})[0], scale

    def detect(self, rgb: np.ndarray, long_edge: int | None = None, floor: float | None = None) -> list:
        """RGB uint8 image -> detections (see decode)."""
        out, scale = self.raw(rgb, long_edge)
        return decode(out, self.meta, scale, floor)


def available() -> bool:
    return os.path.exists(MODEL_PATH) and os.path.exists(META_PATH)
