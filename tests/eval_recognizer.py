"""Evaluate the trained recognizer on fresh synthetic samples it has never seen.
Run: python tests/eval_recognizer.py [N] [SEED]

Pass the same SEED to compare two recognizer versions on identical samples.
"""
import os
import random
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.recognition import synth                # noqa: E402
from backend.recognition.infer import recognizer_available, get_recognizer  # noqa: E402
from backend.chemistry import normalize_ocr          # noqa: E402


def main(n=200, seed=None):
    if seed is not None:
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
    if not recognizer_available():
        print("No trained model found. Train first: python -m backend.recognition.train")
        return 1
    rec = get_recognizer()

    # single-formula samples (skip the 'A + B' equation fragments for a clean metric)
    samples = []
    while len(samples) < n:
        text = synth.random_formula()
        if "+" not in text:
            samples.append(text)

    exact = 0
    shown = 0
    print(f"Evaluating on {n} unseen synthetic samples...\n")
    print(f"{'PRED':<14}{'TRUTH':<14}{'conf':>6}  ok")
    print("-" * 42)
    for text in samples:
        arr, label = synth.make_sample(text)
        out = rec.predict(arr)
        pred = normalize_ocr(out["raw"])
        ok = pred == label
        exact += ok
        if shown < 25:
            print(f"{pred:<14}{label:<14}{out['confidence']:>6.2f}  {'Y' if ok else '.'}")
            shown += 1

    print("-" * 42)
    print(f"\nExact-match accuracy: {exact}/{n} = {exact / n:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 200,
                  int(sys.argv[2]) if len(sys.argv) > 2 else None))
