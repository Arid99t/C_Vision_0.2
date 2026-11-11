r"""Score the recognizers on YOUR handwriting — the test that actually counts.

Collect samples first (each SPACE press saves one labelled crop):

    python -m clients.desktop.capture --label H2O

then:

    python tests\eval_real.py                       (uses data\real)
    python tests\eval_real.py path\to\images path\to\labels.tsv

Prints, for every trained model found, how many samples it reads exactly right
and lists the misreads. If the mobile model does poorly on your writing,
fine-tune it on these samples:

    python -m backend.recognition.train --resume --real-dir data\real\images --real-labels data\real\labels.tsv
    python -m backend.recognition.export
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.chemistry import normalize_ocr  # noqa: E402
from backend.recognition import infer  # noqa: E402
from backend.recognition.config import MOBILE_WEIGHTS_PATH, ONNX_PATH, WEIGHTS_PATH  # noqa: E402


def main(image_dir: str, labels_file: str) -> int:
    if not os.path.exists(labels_file):
        print(f"No labels file at {labels_file}. Capture samples first (see this file's header).")
        return 1
    items = []
    with open(labels_file, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if "\t" in line:
                name, label = line.split("\t", 1)
                if os.path.exists(os.path.join(image_dir, name)):
                    items.append((os.path.join(image_dir, name), label))
    if not items:
        print("No samples found.")
        return 1

    models = []
    if os.path.exists(ONNX_PATH) and infer.onnxruntime_available():
        models.append(("mobile (ONNX)", ONNX_PATH))
    elif os.path.exists(MOBILE_WEIGHTS_PATH) and infer.torch_available():
        models.append(("mobile (PyTorch)", MOBILE_WEIGHTS_PATH))
    if os.path.exists(WEIGHTS_PATH) and infer.torch_available():
        models.append(("original CRNN", WEIGHTS_PATH))
    if not models:
        print("No trained model found.")
        return 1

    print(f"{len(items)} real samples from {image_dir}\n")
    for name, path in models:
        rec = infer.Recognizer(path)
        right, wrong = 0, []
        for img, label in items:
            out = rec.predict(img)
            pred = normalize_ocr(out["raw"]) or ""
            if pred == normalize_ocr(label):
                right += 1
            else:
                wrong.append((os.path.basename(img), label, pred, out["confidence"]))
        print(f"{name}: {right}/{len(items)} = {right / len(items):.1%} read exactly right")
        for fname, label, pred, conf in wrong[:25]:
            print(f"    {fname:<28} wrote {label!r:<14} read {pred!r:<14} confidence {conf:.2f}")
        if len(wrong) > 25:
            print(f"    ... and {len(wrong) - 25} more")
        print()
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        sys.exit(main(sys.argv[1], sys.argv[2]))
    sys.exit(main(os.path.join("data", "real", "images"), os.path.join("data", "real", "labels.tsv")))
