"""Inference wrapper around the trained recognizers.

Designed to degrade gracefully: importing this module never requires torch or
onnxruntime, so the API can call `recognizer_available()` to decide whether the
/recognize endpoint is live. `get_recognizer()` lazily builds a cached
Recognizer and raises a clear error if no model can be run.

Which model is used, first match wins:

  1. static/models/formula_rec.onnx   mobile model through onnxruntime
                                      (CPU, no PyTorch needed on the server)
  2. weights/rec_mobile.pt            mobile model through PyTorch
  3. weights/crnn.pt                  the original desktop CRNN through PyTorch

Each model is fed by the preprocessing it was trained with: the mobile model
by preprocess.prepare_crop (v2), the CRNN by preprocess.clean_crop (v1).
"""
from __future__ import annotations

import json
import math
import os
from typing import Optional, Union

import numpy as np
from PIL import Image

from . import decode, preprocess, synth  # all torch-free
from .charset import ITOS
from .config import IMG_HEIGHT, MOBILE_WEIGHTS_PATH, ONNX_PATH, REC_META_PATH, WEIGHTS_PATH

ImageLike = Union[str, np.ndarray, Image.Image]


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def onnxruntime_available() -> bool:
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


def _find_model(weights_path: Optional[str] = None):
    """Return (kind, path) with kind in {'onnx', 'torch'}, or (None, None)."""
    if weights_path:                                   # an explicit file was requested
        if not os.path.exists(weights_path):
            return None, None
        if weights_path.lower().endswith(".onnx"):
            return ("onnx", weights_path) if onnxruntime_available() else (None, None)
        return ("torch", weights_path) if torch_available() else (None, None)
    if os.path.exists(ONNX_PATH) and onnxruntime_available():
        return "onnx", ONNX_PATH
    if torch_available():
        for path in (MOBILE_WEIGHTS_PATH, WEIGHTS_PATH):
            if os.path.exists(path):
                return "torch", path
    return None, None


def recognizer_available(weights_path: Optional[str] = None) -> bool:
    """True if recognition can actually run (a model on disk + a runtime for it)."""
    return _find_model(weights_path)[0] is not None


def _to_gray(image: ImageLike) -> np.ndarray:
    """Anything image-like -> (h, w) uint8 grayscale array."""
    if isinstance(image, str):
        return np.asarray(Image.open(image).convert("L"))
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("L"))
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 3:                     # (H,W,3/4) -> drop to gray
            return np.asarray(Image.fromarray(arr[..., :3].astype(np.uint8)).convert("L"))
        return arr.astype(np.uint8)
    raise TypeError(f"Unsupported image type: {type(image)}")


def _is_known(text: str) -> bool:
    """True if every formula on the line is a compound the app knows about."""
    from backend.chemistry.formula import KNOWN_FORMULAS, normalize_ocr
    parts = [p.strip() for p in text.split("+")]
    return all(p and normalize_ocr(p) in KNOWN_FORMULAS for p in parts)


class Recognizer:
    def __init__(self, weights_path: Optional[str] = None, device: Optional[str] = None):
        kind, path = _find_model(weights_path)
        if kind is None:
            raise FileNotFoundError(
                "No runnable recognition model. Train one (python -m "
                "backend.recognition.train) and/or install torch or onnxruntime.")
        self.kind, self.path = kind, path
        if kind == "onnx":
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = min(4, os.cpu_count() or 1)
            self.session = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            meta = {}
            if os.path.exists(REC_META_PATH):
                with open(REC_META_PATH, encoding="utf-8") as f:
                    meta = json.load(f)
            self.itos = meta.get("charset") or ITOS
            self.arch = meta.get("arch", "mobile")
            self.img_height = meta.get("img_height", IMG_HEIGHT)
            self.device = "cpu"
        else:
            import torch
            from .model import load_checkpoint
            self.torch = torch
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.model, ckpt = load_checkpoint(path, self.device)
            self.itos = ckpt.get("itos") or ITOS
            self.arch = ckpt.get("arch", "crnn")
            self.img_height = ckpt.get("img_height", IMG_HEIGHT)

    # --- pipeline stages (public so tools/tests can inspect each one) -------
    def model_image(self, image: ImageLike) -> np.ndarray:
        """The (32, W) uint8 image the model reads for this crop."""
        gray = _to_gray(image)
        if self.arch == "mobile":
            return preprocess.prepare_crop(gray)
        pil = preprocess.clean_crop(Image.fromarray(gray, "L"))
        return synth.to_fixed_height(pil, self.img_height)

    def log_probs(self, model_img: np.ndarray) -> np.ndarray:
        """(32, W) uint8 -> (T, C) per-timestep log-probabilities."""
        x = preprocess.to_model_input(model_img)[None, None]          # (1,1,H,W)
        if self.kind == "onnx":
            return self.session.run(None, {self.input_name: x})[0][:, 0, :]
        with self.torch.no_grad():
            logp = self.model(self.torch.from_numpy(x).to(self.device))
        return logp[:, 0, :].float().cpu().numpy()

    def _greedy(self, lp: np.ndarray) -> str:
        out, prev = [], -1
        for idx in lp.argmax(1).tolist():
            if idx != prev and idx != 0:
                out.append(self.itos[idx])
            prev = idx
        return "".join(out)

    def predict(self, image: ImageLike) -> dict:
        """Read one cropped formula image.

        Returns {'raw', 'confidence', 'greedy', 'model'}:
          raw         best reading. Decoding is chemistry-constrained (a CTC beam
                      search over strings that can be valid formula lines, see
                      decode.py) and prefers a known compound when two readings
                      are close. Falls back to `greedy` when nothing valid
                      survives, e.g. the crop is not a formula at all.
          confidence  the model's probability for that whole string (summed over
                      all alignments), 0..1. Ambiguous or junk crops score low.
          greedy      the unconstrained per-column reading, for diagnostics.
        """
        lp = self.log_probs(self.model_image(image))
        greedy = self._greedy(lp)
        best = decode.choose_reading(decode.beam_candidates(lp, self.itos), _is_known)
        if best is not None:
            text, conf = best[0], math.exp(best[1])
        else:
            text, conf = greedy, float(np.exp(lp.max(1).sum()))
        return {"raw": text, "confidence": round(min(1.0, conf), 4),
                "greedy": greedy, "model": self.arch}


_RECOGNIZER: Optional[Recognizer] = None


def get_recognizer(weights_path: Optional[str] = None) -> Recognizer:
    """Return a process-wide cached Recognizer (builds it on first call)."""
    global _RECOGNIZER
    if _RECOGNIZER is None:
        _RECOGNIZER = Recognizer(weights_path)
    return _RECOGNIZER
