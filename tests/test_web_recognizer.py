"""The exported model must read the same thing in the browser as on the server.

Runs the on-device recognizer (static/js/recognizer.js + ONNX Runtime Web) in
headless Chromium and the Python recognizer on identical crops, and compares
the readings and confidences. Also reports how long one read takes in the
browser's WebAssembly runtime.

Run: python tests/test_web_recognizer.py
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

import numpy as np  # noqa: E402

from tests.webharness import JS_B64, b64, browser_page, playwright_available  # noqa: E402
from backend.recognition import infer  # noqa: E402
from backend.recognition.config import ONNX_PATH  # noqa: E402

if not playwright_available():
    print("SKIP: playwright / uvicorn not installed")
    from tests.webharness import skip_unavailable
    skip_unavailable("Optional dependency or generated asset unavailable; see message above")
if not (os.path.exists(ONNX_PATH) and infer.onnxruntime_available()):
    print("SKIP: no exported model (python -m backend.recognition.export)")
    from tests.webharness import skip_unavailable
    skip_unavailable("Optional dependency or generated asset unavailable; see message above")

from tests.test_web_parity import _crops  # noqa: E402

READ_JS = """async (items) => {
    %s
    const { loadChem } = await import('/static/js/chem.js');
    const { Recognizer } = await import('/static/js/recognizer.js');
    const { threads } = await import('/static/js/ortenv.js');
    const chem = await loadChem();
    const rec = await Recognizer.load(chem);
    const out = [];
    for (const it of items) {
      const r = await rec.read(fromB64(it.b), it.h, it.w);
      out.push({ raw: r.raw, formula: r.formula, conf: r.confidence, ms: r.ms, valid: r.valid });
    }
    return { out, threads, isolated: self.crossOriginIsolated };
}""" % JS_B64


def test_browser_recognizer():
    crops, source = _crops(80)
    rec = infer.Recognizer(ONNX_PATH)
    with browser_page() as (page, _base):
        payload = [{"b": b64(c.tobytes()), "h": int(c.shape[0]), "w": int(c.shape[1])} for c in crops]
        res = page.evaluate(READ_JS, payload)
        problems = list(page.problems)
    assert not problems, f"browser errors: {problems[:3]}"
    same, worst = 0, 0.0
    for crop, got in zip(crops, res["out"]):
        want = rec.predict(crop)
        if got["raw"] == want["raw"]:
            same += 1
            worst = max(worst, abs(got["conf"] - want["confidence"]))
    assert same >= 0.97 * len(crops), f"only {same}/{len(crops)} readings agree with the server"
    assert worst < 0.02, f"confidence differs by up to {worst:.3f}"
    times = sorted(r["ms"] for r in res["out"][5:])
    print(f"ok   browser == server on {same}/{len(crops)} {source} crops "
          f"(confidence within {worst:.4f})")
    print(f"     WebAssembly: {res['threads']} thread(s), cross-origin isolated: {res['isolated']}; "
          f"one read (prepare + model + decode) median {times[len(times) // 2]:.1f} ms, "
          f"p90 {times[int(len(times) * 0.9)]:.1f} ms in headless Chromium")
    print("\n1/1 passed")


if __name__ == "__main__":
    try:
        test_browser_recognizer()
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: {exc!r}"[:1500])
        print("\n0/1 passed")
        sys.exit(1)
