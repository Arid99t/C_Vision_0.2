"""Export the mobile recognizer for deployment.

    python -m backend.recognition.export            # ONNX + metadata (+ int8 if it holds up)

Writes into backend/static/models/, which the web app is served from:

    formula_rec.onnx        float32 model; runs in onnxruntime (server) and
                            onnxruntime-web (phone, WebAssembly). Input
                            "image" (1, 1, 32, W) in [-1, 1], output "logp"
                            (T, 1, classes) log-probabilities, T = ceil(W / 2).
    formula_rec.int8.onnx   statically quantized copy (smaller, faster on CPU).
                            Only kept when it loses under 1 point of accuracy.
    formula_rec.json        everything a client needs besides the weights:
                            charset, sizes and the preprocessing constants, so
                            the JavaScript twin never drifts from the Python.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np          # noqa: I001  (before torch: OpenMP DLL load order on Windows)
import torch

from . import decode, preprocess
from .charset import ITOS, decode_greedy
from .config import (IMG_HEIGHT, MAX_WIDTH, MIN_WIDTH, MOBILE_WEIGHTS_PATH, MODELS_DIR,
                     ONNX_INT8_PATH, ONNX_PATH, REC_META_PATH)
from .model import count_params, load_checkpoint


def export_onnx(weights: str = MOBILE_WEIGHTS_PATH, out: str = ONNX_PATH, opset: int = 17) -> dict:
    """Write the float32 ONNX model and check it against PyTorch. Returns stats."""
    import onnxruntime as ort

    model, ckpt = load_checkpoint(weights, "cpu")
    if ckpt.get("arch") != "mobile":
        raise ValueError("export expects a mobile recognizer checkpoint")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.onnx.export(
        model, torch.zeros(1, 1, IMG_HEIGHT, 128), out,
        input_names=["image"], output_names=["logp"],
        dynamic_axes={"image": {3: "width"}, "logp": {0: "steps"}},
        opset_version=opset, do_constant_folding=True, dynamo=False)

    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    worst = 0.0
    gen = torch.Generator().manual_seed(0)
    for w in (MIN_WIDTH, 24, 47, 100, 173, 256, MAX_WIDTH):
        x = (torch.rand(1, 1, IMG_HEIGHT, w, generator=gen) * 2 - 1)
        with torch.no_grad():
            ref = model(x).numpy()
        got = sess.run(None, {"image": x.numpy()})[0]
        assert got.shape == ref.shape, f"shape mismatch at width {w}: {got.shape} vs {ref.shape}"
        worst = max(worst, float(np.abs(ref - got).max()))
    if worst > 1e-3:
        raise RuntimeError(f"ONNX output differs from PyTorch by {worst:g}")
    return {"params": count_params(model), "size": ckpt.get("size", "base"),
            "score": ckpt.get("score"), "max_abs_diff_vs_torch": worst,
            "bytes": os.path.getsize(out)}


def write_meta(stats: dict, path: str = REC_META_PATH, int8: bool = False) -> None:
    meta = {
        "arch": "mobile", "size": stats["size"], "params": stats["params"],
        "charset": ITOS, "img_height": IMG_HEIGHT, "min_width": MIN_WIDTH,
        "max_width": MAX_WIDTH, "time_stride": 2,
        "normalize": {"scale": 255.0, "mean": 0.5, "std": 0.5},
        "preprocess": {
            "work_height": preprocess.WORK_HEIGHT, "ink_threshold": preprocess.INK_THRESHOLD,
            "margin_frac": preprocess.MARGIN_FRAC, "rot_min_elong": preprocess.ROT_MIN_ELONG,
            "rot_min_deg": preprocess.ROT_MIN_DEG, "rot_max_deg": preprocess.ROT_MAX_DEG,
        },
        "decode": {"beam_width": 8, "topk_chars": 6, "lexicon_bonus": decode.LEXICON_BONUS},
        "files": {"float32": os.path.basename(ONNX_PATH),
                  "int8": os.path.basename(ONNX_INT8_PATH) if int8 else None},
        "heldout_exact_match": stats.get("score"),
        "exported": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)


def eval_onnx(path: str, store_name: str, limit: int = 1500, threads: int = 1):
    """Greedy exact-match accuracy and mean ms per crop of an ONNX model on a stored set."""
    import onnxruntime as ort
    from .gen_data import Store

    store = Store(store_name)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
    n = min(limit, len(store))
    hit, spent = 0, 0.0
    for i in range(n):
        x = preprocess.to_model_input(store.image(i))[None, None]
        t0 = time.perf_counter()
        lp = sess.run(None, {"image": x})[0][:, 0, :]
        spent += time.perf_counter() - t0
        hit += decode_greedy(lp.argmax(1).tolist()) == store.labels[i]
    return hit / n, 1000.0 * spent / n


def quantize(src: str = ONNX_PATH, dst: str = ONNX_INT8_PATH, calib: str = "val", n: int = 400):
    """Static int8 quantization of the convolutions, calibrated on real samples."""
    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                          quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process
    from .gen_data import Store

    store = Store(calib)

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.i = 0

        def get_next(self):
            if self.i >= min(n, len(store)):
                return None
            x = preprocess.to_model_input(store.image(self.i))[None, None]
            self.i += 1
            return {"image": x}

    pre = dst + ".pre.onnx"
    quant_pre_process(src, pre)
    quantize_static(pre, dst, Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    op_types_to_quantize=["Conv"])
    for junk in (pre, pre + ".data"):
        if os.path.exists(junk):
            os.remove(junk)


def main():
    p = argparse.ArgumentParser(description="Export the mobile recognizer to ONNX")
    p.add_argument("--weights", default=MOBILE_WEIGHTS_PATH)
    p.add_argument("--no-int8", action="store_true", help="skip the quantized copy")
    p.add_argument("--eval", default="bench_fonts,bench_glyphs",
                   help="generated sets used to compare float32 vs int8")
    a = p.parse_args()

    stats = export_onnx(a.weights)
    print(f"float32: {ONNX_PATH}  {stats['bytes'] / 1e6:.2f} MB, {stats['params']:,} params, "
          f"max |onnx - torch| = {stats['max_abs_diff_vs_torch']:.2e}")

    keep_int8 = False
    if not a.no_int8:
        sets = [s for s in a.eval.split(",") if s]
        try:
            quantize()
            f32 = [eval_onnx(ONNX_PATH, s) for s in sets]
            i8 = [eval_onnx(ONNX_INT8_PATH, s) for s in sets]
            for s, (fa, ft), (ia, it) in zip(sets, f32, i8):
                print(f"  {s}: float32 {fa:.1%} @ {ft:.1f} ms | int8 {ia:.1%} @ {it:.1f} ms "
                      f"(1 CPU thread)")
            drop = float(np.mean([fa - ia for (fa, _), (ia, _) in zip(f32, i8)]))
            keep_int8 = drop < 0.01
            print(f"int8: {os.path.getsize(ONNX_INT8_PATH) / 1e6:.2f} MB, mean accuracy change "
                  f"{-drop:+.2%} -> {'kept' if keep_int8 else 'discarded'}")
        except Exception as exc:  # noqa: BLE001  quantization is optional
            print(f"int8 quantization skipped: {exc}")
        if not keep_int8 and os.path.exists(ONNX_INT8_PATH):
            os.remove(ONNX_INT8_PATH)

    write_meta(stats, int8=keep_int8)
    print(f"metadata: {REC_META_PATH}")
    print(f"models directory: {MODELS_DIR}")


if __name__ == "__main__":
    main()
