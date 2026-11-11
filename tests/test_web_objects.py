"""The object spotter must behave the same in the browser as on the server.

  1. decode parity   the SAME raw model output is decoded by objects.js and by
                     backend/objects/detect.py -> identical detections.
  2. end to end      the browser letterboxes a photo, runs the model in
                     WebAssembly and decodes; the result must agree with the
                     server on the same photo.
  3. speed           how long one pass takes in the browser.

Uses a few COCO val2017 photos when that archive is present in the work
directory, otherwise a synthetic desk scene (which only exercises 1 and 3).

Run: python tests/test_web_objects.py
"""
import io
import os
import sys
import warnings
import zipfile

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from tests.webharness import JS_B64, b64, browser_page, playwright_available  # noqa: E402
from backend.objects import detect as od  # noqa: E402
from backend.recognition.config import WORK_DIR  # noqa: E402

if not playwright_available():
    print("SKIP: playwright / uvicorn not installed")
    from tests.webharness import skip_unavailable
    skip_unavailable("Optional dependency or generated asset unavailable; see message above")
if not od.available():
    print("SKIP: no object model (python -m backend.objects.build)")
    from tests.webharness import skip_unavailable
    skip_unavailable("Optional dependency or generated asset unavailable; see message above")

from PIL import Image  # noqa: E402

TEST_DIR = os.path.join(ROOT, "backend", "static", "test")
PHOTOS = ["000000000139.jpg", "000000000632.jpg", "000000001000.jpg", "000000002153.jpg"]


def _photos():
    """[(url path, RGB array)] — COCO photos if available, else a synthetic scene."""
    os.makedirs(TEST_DIR, exist_ok=True)
    out = []
    zpath = os.path.join(WORK_DIR, "coco", "val2017.zip")
    if os.path.exists(zpath):
        with zipfile.ZipFile(zpath) as z:
            names = set(z.namelist())
            for name in PHOTOS:
                if "val2017/" + name in names:
                    data = z.read("val2017/" + name)
                    with open(os.path.join(TEST_DIR, "obj_" + name), "wb") as f:
                        f.write(data)
                    out.append(("/static/test/obj_" + name, np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))))
    if not out:
        from tests.scenes import make_scene
        frame, _ = make_scene([("H2O", (0.5, 0.4))], size=(640, 480))
        rgb = np.ascontiguousarray(frame[..., ::-1])
        Image.fromarray(rgb).save(os.path.join(TEST_DIR, "obj_scene.png"))
        out.append(("/static/test/obj_scene.png", rgb))
    return out


def _iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def test_browser_objects():
    photos = _photos()
    det = od.ObjectDetector()
    meta = det.meta
    raws = [det.raw(rgb) for _, rgb in photos]
    with browser_page() as (page, _base):
        # 1. identical decoding of identical raw output
        payload = [{"b": b64(np.ascontiguousarray(out[0], np.float32).tobytes()),
                    "P": int(out.shape[1] - 4), "N": int(out.shape[2]), "scale": float(scale)}
                   for out, scale in raws]
        js = page.evaluate("""async ({items, meta}) => {
            %s
            const { decode } = await import('/static/js/objects.js');
            return items.map((it) => decode(new Float32Array(fromB64(it.b).buffer), it.P, it.N, meta, it.scale, 0.2)
              .map((d) => ({ label: d.label, score: d.score, box: d.box })));
        }""" % JS_B64, {"items": payload, "meta": meta})
        total = 0
        for (out, scale), got in zip(raws, js):
            want = od.decode(out, meta, scale, floor=0.2)
            assert len(got) == len(want), f"decode: js {len(got)} detections, python {len(want)}"
            for g, w in zip(got, want):
                assert g["label"] == w["label"], f"decode label: js {g['label']} python {w['label']}"
                assert abs(g["score"] - w["score"]) < 1e-5
                assert max(abs(a - b) for a, b in zip(g["box"], w["box"])) < 0.05
            total += len(want)
        print(f"ok   objects.js decodes exactly like the server ({total} detections on {len(photos)} photos)")

        # 2 + 3. the whole browser pipeline, timed
        res = page.evaluate("""async ({urls, meta}) => {
            const { ObjectSpotter } = await import('/static/js/objects.js');
            const { threads } = await import('/static/js/ortenv.js');
            const spot = await ObjectSpotter.load(meta);
            const out = [], times = {};
            for (const edge of [640, 480, 384]) {
              spot.longEdge = edge;
              const ms = [];
              for (const url of urls) {
                const img = new Image();
                img.src = url;
                await img.decode();
                const t0 = performance.now();
                const dets = await spot.detect(img, 0, 0, img.naturalWidth, img.naturalHeight);
                ms.push(performance.now() - t0);
                if (edge === 640) out.push(dets.map((d) => ({ label: d.label, score: d.score, box: d.box })));
              }
              ms.sort((a, b) => a - b);
              times[edge] = ms[Math.floor(ms.length / 2)];
            }
            return { out, times, threads };
        }""", {"urls": [u for u, _ in photos], "meta": meta})
        problems = list(page.problems)
    assert not problems, f"browser errors: {problems[:3]}"
    matched = strong = 0
    for (_, rgb), got in zip(photos, res["out"]):
        want = det.detect(rgb)
        for w in want:
            if w["score"] < 0.5:
                continue                       # borderline boxes may differ with resampling
            strong += 1
            matched += any(g["label"] == w["label"] and _iou(g["box"], w["box"]) > 0.7 for g in got)
    if strong:
        assert matched >= 0.85 * strong, f"browser found {matched} of the server's {strong} confident objects"
        print(f"ok   browser pipeline agrees with the server on {matched}/{strong} confident objects")
    else:
        print("ok   browser pipeline ran (no confident objects in the test image to compare)")
    t = res["times"]
    print(f"     {meta['model']} in WebAssembly ({res['threads']} threads, headless Chromium): "
          f"{t['640']:.0f} ms at 640 px, {t['480']:.0f} ms at 480 px, {t['384']:.0f} ms at 384 px")
    print("\n2/2 passed")


if __name__ == "__main__":
    try:
        test_browser_objects()
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: {exc!r}"[:1500])
        print("\nfailed")
        sys.exit(1)
