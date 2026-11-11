"""Head-to-head benchmark: mobile recognizer vs the original CRNN.

Both models read the SAME raw camera-style crops, each through its own full
pipeline (its preprocessing, the shared chemistry-constrained decoder), so the
comparison is what a user would actually see.

Test sets (all generated with fixed seeds; none is used in training):
  fonts    handwriting FONTS held out from training, through the camera model
  glyphs   formulas composed from individual handwritten characters (EMNIST
           test split), through the camera model
  gentle   the same two kinds under mild conditions (level 0.35)
  v1       the original generator's samples (fonts on a clean page) — the
           distribution the CRNN was trained on

Reports exact-match accuracy, character error rate, and — what matters for a
live app — how many reads clear the confidence gate and how many of THOSE are
right. Also parameters, file size and single-thread CPU time per crop.

    python tests/bench_recognizer.py [N per set, default 600]

These are synthetic benchmarks. Real handwriting from a real camera is the
test that counts; collect some with clients/desktop/capture.py and use
tests/eval_real.py.
"""
import os
import random
import sys
import time
import warnings

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from backend.chemistry import normalize_ocr  # noqa: E402
from backend.recognition import infer, synth  # noqa: E402
from backend.recognition.config import MOBILE_WEIGHTS_PATH, ONNX_PATH, WEIGHTS_PATH  # noqa: E402
from backend.recognition.train import edit_distance  # noqa: E402

GATE = 0.5      # the clients' default --min-confidence


def build_sets(n: int) -> dict:
    from backend.recognition.synthv2 import Synth
    sets = {}
    gen = Synth("test", seed=101)
    sets["fonts"] = [gen.crop(mode="font")[:2] for _ in range(n)]
    gen = Synth("test", seed=102)
    sets["glyphs"] = [gen.crop(mode="glyph")[:2] for _ in range(n)]
    gen = Synth("test", seed=103, level=0.35)
    sets["gentle"] = [gen.crop()[:2] for _ in range(n)]
    random.seed(104)
    np.random.seed(104)
    sets["v1"] = [synth.make_sample() for _ in range(n)]
    return sets


def score(rec, samples) -> dict:
    exact = edits = chars = accepted = accepted_ok = 0
    for img, label in samples:
        out = rec.predict(img)
        pred, want = normalize_ocr(out["raw"]) or "", normalize_ocr(label)
        ok = pred == want
        exact += ok
        edits += edit_distance(pred, want)
        chars += len(want)
        if out["confidence"] >= GATE:
            accepted += 1
            accepted_ok += ok
    n = len(samples)
    return {"exact": exact / n, "cer": edits / max(1, chars), "coverage": accepted / n,
            "precision": accepted_ok / max(1, accepted)}


def cpu_time(rec, samples, reps: int = 60) -> float:
    """Median ms for the model alone (prepared image -> scores), one CPU thread."""
    imgs = [rec.model_image(s[0]) for s in samples[:reps]]
    if rec.kind == "torch":
        import torch
        torch.set_num_threads(1)
        rec.model.to("cpu")
        rec.device = "cpu"
    for im in imgs[:5]:
        rec.log_probs(im)
    times = []
    for im in imgs:
        t0 = time.perf_counter()
        rec.log_probs(im)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def main(n: int = 600):
    models = []
    if os.path.exists(WEIGHTS_PATH) and infer.torch_available():
        models.append(("original CRNN", infer.Recognizer(WEIGHTS_PATH), WEIGHTS_PATH))
    if os.path.exists(ONNX_PATH) and infer.onnxruntime_available():
        import onnxruntime as ort
        rec = infer.Recognizer(ONNX_PATH)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1                       # single thread, like the CRNN timing
        rec.session = ort.InferenceSession(ONNX_PATH, opts, providers=["CPUExecutionProvider"])
        models.append(("mobile (ONNX)", rec, ONNX_PATH))
    elif os.path.exists(MOBILE_WEIGHTS_PATH) and infer.torch_available():
        models.append(("mobile (PyTorch)", infer.Recognizer(MOBILE_WEIGHTS_PATH), MOBILE_WEIGHTS_PATH))
    if not models:
        print("No trained model found.")
        return 1

    print(f"Generating {n} samples per set ...")
    sets = build_sets(n)
    results = {}
    for name, rec, _ in models:
        results[name] = {k: score(rec, v) for k, v in sets.items()}

    print(f"\nExact-match accuracy (character error rate) on {n} samples per set")
    print(f"{'':<18}" + "".join(f"{k:>18}" for k in sets))
    for name in results:
        print(f"{name:<18}" + "".join(
            f"{results[name][k]['exact']:>9.1%} ({results[name][k]['cer']:>5.1%})" for k in sets))

    print(f"\nWith the confidence gate at {GATE}: share of reads accepted / share of those that are right")
    print(f"{'':<18}" + "".join(f"{k:>18}" for k in sets))
    for name in results:
        print(f"{name:<18}" + "".join(
            f"{results[name][k]['coverage']:>8.0%} / {results[name][k]['precision']:>6.1%}" for k in sets))

    print("\nCost")
    for name, rec, path in models:
        if rec.kind == "torch":
            params = sum(p.numel() for p in rec.model.parameters())
        else:
            import onnx
            from onnx import numpy_helper
            params = sum(numpy_helper.to_array(t).size for t in onnx.load(path).graph.initializer)
        ms = cpu_time(rec, sets["fonts"])
        print(f"  {name:<18} {params / 1e6:>5.2f} M parameters   {os.path.getsize(path) / 1e6:>6.2f} MB on disk"
              f"   {ms:>6.1f} ms per crop (1 CPU thread)")
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 600))
