"""Inference wrapper around the trained CRNN.

Designed to degrade gracefully: importing this module never requires torch, so
the API can call `recognizer_available()` to decide whether the /recognize
endpoint is live. `get_recognizer()` lazily builds a cached Recognizer and
raises a clear error if torch or the weights are missing.
"""
from __future__ import annotations

import os
from typing import Optional, Union

import numpy as np
from PIL import Image

from . import synth  # torch-free: provides to_fixed_height
from .config import IMG_HEIGHT, WEIGHTS_PATH

ImageLike = Union[str, np.ndarray, Image.Image]


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def recognizer_available(weights_path: str = WEIGHTS_PATH) -> bool:
    """True if we can actually run recognition (torch installed + weights on disk)."""
    return torch_available() and os.path.exists(weights_path)


def _to_pil_gray(image: ImageLike) -> Image.Image:
    if isinstance(image, str):
        return Image.open(image).convert("L")
    if isinstance(image, Image.Image):
        return image.convert("L")
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 3:                     # (H,W,3/4) -> drop to gray
            arr = arr[..., :3]
            return Image.fromarray(arr.astype(np.uint8)).convert("L")
        return Image.fromarray(arr.astype(np.uint8), "L")
    raise TypeError(f"Unsupported image type: {type(image)}")


class Recognizer:
    def __init__(self, weights_path: str = WEIGHTS_PATH, device: Optional[str] = None):
        if not torch_available():
            raise RuntimeError("PyTorch is not installed; cannot run recognition.")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"No trained weights at {weights_path}. Train first: "
                f"python -m backend.recognition.train"
            )
        import torch
        from .model import build_model

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(weights_path, map_location=self.device)
        self.itos = ckpt.get("itos")
        self.img_height = ckpt.get("img_height", IMG_HEIGHT)
        self.model = build_model(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    def _decode(self, indices) -> str:
        out, prev = [], -1
        for idx in indices:
            idx = int(idx)
            if idx != prev and idx != 0:
                out.append(self.itos[idx])
            prev = idx
        return "".join(out)

    def _preprocess(self, image: ImageLike):
        pil = _to_pil_gray(image)
        arr = synth.to_fixed_height(pil, self.img_height)      # (H, W) uint8
        x = arr.astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5                                     # match training
        t = self.torch.from_numpy(x).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        return t.to(self.device)

    def predict(self, image: ImageLike) -> dict:
        """Return {'raw': str, 'confidence': float} for one cropped formula image."""
        x = self._preprocess(image)
        with self.torch.no_grad():
            logp = self.model(x)                # (T, 1, C)
        probs = logp.exp().squeeze(1)           # (T, C)
        idx_seq = probs.argmax(1)               # (T,)
        text = self._decode(idx_seq.tolist())
        conf = float(probs.max(1).values.mean().item())
        return {"raw": text, "confidence": round(conf, 4)}


_RECOGNIZER: Optional[Recognizer] = None


def get_recognizer(weights_path: str = WEIGHTS_PATH) -> Recognizer:
    """Return a process-wide cached Recognizer (builds it on first call)."""
    global _RECOGNIZER
    if _RECOGNIZER is None:
        _RECOGNIZER = Recognizer(weights_path)
    return _RECOGNIZER
