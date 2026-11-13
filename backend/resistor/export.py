"""Export the trained resistor band reader for the on-device app.

    python -m backend.resistor.export

Writes  backend/static/models/resistor.onnx   the model (float32, fixed 48x160 input)
        backend/static/models/resistor.json   colours, input size, how it scored

and checks that onnxruntime gives the same answers as PyTorch on fresh
pictures before declaring success.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from ..recognition.config import MODELS_DIR
from .bands import COLOURS
from .model import BandReader, greedy
from .render import H, W, load_bank, render_sample
from .train import WEIGHTS_PATH

ONNX_PATH = os.path.join(MODELS_DIR, "resistor.onnx")
META_PATH = os.path.join(MODELS_DIR, "resistor.json")


def export(weights: str = WEIGHTS_PATH, onnx_path: str = ONNX_PATH, meta_path: str = META_PATH) -> dict:
    ckpt = torch.load(weights, map_location="cpu")
    model = BandReader()
    model.load_state_dict(ckpt["model"])
    model.eval()
    dummy = torch.zeros(1, 3, H, W)
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    torch.onnx.export(model, dummy, onnx_path, input_names=["image"], output_names=["logp"], opset_version=17,
                      dynamo=False)

    # parity: the exported graph must agree with PyTorch
    import onnxruntime as ort
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(4242)
    bank = load_bank()
    worst, same, total = 0.0, 0, 64
    for _ in range(total):
        img, _labels = render_sample(rng, bank)
        x = torch.from_numpy(img).permute(2, 0, 1).float().div(127.5).sub(1.0)[None]
        with torch.no_grad():
            ref = model(x)
        out = session.run(None, {"image": x.numpy()})[0]
        worst = max(worst, float(np.abs(out - ref.numpy()).max()))
        same += greedy(torch.from_numpy(out)) == greedy(ref)
    if same != total:
        raise RuntimeError(f"ONNX and PyTorch disagree on {total - same} of {total} pictures")

    meta = {
        "arch": "BandReader", "params": sum(p.numel() for p in model.parameters()),
        "input": {"height": H, "width": W, "channels": "RGB", "normalize": "x / 127.5 - 1"},
        "output": "log-probabilities, shape (40, 1, 13): time step, batch, class",
        "classes": ["<blank>"] + COLOURS,
        "trained_on": "drawn pictures plus re-photographed real photos (backend/resistor/render.py)",
        "validation": ckpt.get("metrics"), "epoch": ckpt.get("epoch"),
        "onnx_max_abs_diff": worst, "file": os.path.basename(onnx_path),
        "exported": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=WEIGHTS_PATH)
    args = ap.parse_args()
    info = export(args.weights)
    print(f"exported {ONNX_PATH}  ({os.path.getsize(ONNX_PATH) / 1024:.0f} kB, {info['params'] / 1e6:.2f} M parameters)")
    print("validation:", info["validation"], "| max ONNX difference:", f"{info['onnx_max_abs_diff']:.2e}")
