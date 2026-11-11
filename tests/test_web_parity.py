"""The phone runs JavaScript twins of the Python vision/chemistry code. These
tests run both on the same inputs, in a real (headless) browser, and compare:

  preprocess.js  vs  recognition/preprocess.prepare_crop   (pixel for pixel)
  decode.js      vs  recognition/decode                    (grammar + beam search)
  chem.js        vs  chemistry/formula + reactions         (parse, clean-up, lookup)
  detect.js      vs  clients/desktop/ar_app.detect_formula_regions

Needs Playwright + Chromium. Run: python tests/test_web_parity.py
"""
import json
import os
import random
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

from tests.webharness import JS_B64, b64, browser_page, playwright_available, unb64  # noqa: E402

if not playwright_available():
    print("SKIP: playwright / uvicorn not installed")
    from tests.webharness import skip_unavailable
    skip_unavailable("Optional dependency or generated asset unavailable; see message above")
try:
    import cv2  # noqa: E402
except ImportError:
    print("SKIP: opencv not installed")
    from tests.webharness import skip_unavailable
    skip_unavailable("Optional dependency or generated asset unavailable; see message above")

from backend.chemistry import describe_formula, normalize_ocr, parse_formula, react  # noqa: E402
from backend.chemistry.formula import ELEMENTS  # noqa: E402
from backend.recognition import decode, preprocess  # noqa: E402
from backend.recognition.charset import ITOS, STOI  # noqa: E402
from clients.desktop import ar_app  # noqa: E402


def _crops(n=120):
    """Camera-like crops: from the real generator when its data is installed,
    otherwise simple drawn text with tilt, shading and noise."""
    out = []
    try:
        from backend.recognition.synthv2 import Synth
        gen = Synth("test", seed=123)
        while len(out) < n:
            out.append(gen.crop()[0])
        return out, "generator"
    except Exception:  # noqa: BLE001  (fonts / glyph bank not built on this machine)
        rng = random.Random(5)
        words = ["H2O", "CO2", "NaCl", "C6H12O6", "Fe2O3 + CO", "Ca(OH)2", "CH4", "H2SO4"]
        while len(out) < n:
            text = rng.choice(words)
            img = np.full((rng.randint(60, 150), 60 + 34 * len(text)), rng.randint(170, 250), np.float32)
            cv2.putText(img, text, (20, int(img.shape[0] * 0.68)), cv2.FONT_HERSHEY_SIMPLEX,
                        rng.uniform(0.9, 1.8), rng.randint(0, 70), rng.randint(2, 5), cv2.LINE_AA)
            m = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), rng.uniform(-22, 22), 1)
            img = cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), borderValue=float(img[0, 0]))
            img *= np.linspace(rng.uniform(0.7, 1), rng.uniform(0.7, 1), img.shape[1])[None, :]
            img += np.random.default_rng(len(out)).normal(0, rng.uniform(0, 6), img.shape)
            out.append(np.clip(img, 0, 255).astype(np.uint8))
        return out, "drawn text"


def test_preprocess(page):
    crops, source = _crops()
    payload = [{"b": b64(c.tobytes()), "h": int(c.shape[0]), "w": int(c.shape[1])} for c in crops]
    results = page.evaluate("""async (items) => {
        %s
        const m = await import('/static/js/preprocess.js');
        return items.map((it) => {
          const r = m.prepareCrop(fromB64(it.b), it.h, it.w);
          return { b: toB64(r.data), w: r.width };
        });
    }""" % JS_B64, payload)
    worst, total, same_shape = 0, 0.0, 0
    for crop, res in zip(crops, results):
        want = preprocess.prepare_crop(crop)
        got = np.frombuffer(unb64(res["b"]), np.uint8).reshape(32, res["w"])
        if got.shape != want.shape:
            continue          # a borderline tilt/crop decision flipped by float rounding
        same_shape += 1
        diff = np.abs(got.astype(int) - want.astype(int))
        worst = max(worst, int(diff.max()))
        total += float(diff.mean())
    assert same_shape >= 0.97 * len(crops), f"only {same_shape}/{len(crops)} outputs have the same size"
    assert worst <= 2, f"JS and Python differ by up to {worst} gray levels"
    assert total / same_shape < 0.05, f"mean difference {total / same_shape:.3f} gray levels"
    print(f"ok   preprocess.js == prepare_crop on {same_shape}/{len(crops)} {source} crops "
          f"(max diff {worst} level, mean {total / same_shape:.4f})")


def _peaky_logits(text, rng, noise=1.2):
    """(T, C) log-probabilities that mostly spell `text`, with competing noise."""
    steps = []
    for ch in text:
        steps += [STOI[ch]] * rng.randint(1, 3) + [0] * rng.randint(1, 2)
    steps = [0] + steps
    logits = rng_normal(rng, (len(steps), len(ITOS))) * noise
    for t, c in enumerate(steps):
        logits[t, c] += rng.uniform(2.0, 6.0)
    logits -= np.log(np.exp(logits).sum(1, keepdims=True))
    return logits.astype(np.float32)


def rng_normal(rng, shape):
    return np.array([[rng.gauss(0, 1) for _ in range(shape[1])] for _ in range(shape[0])])


def test_decode(page):
    rng = random.Random(11)
    texts = ["H2O", "CO2", "NaCl", "Ca(OH)2", "C6H12O6", "CH4 + H2O", "Fe2O3", "H2SO4", "KMnO4",
             "Al2(SO4)3", "HCl", "CO", "Cl2", "NH4Cl", "C12H22O11", "Mg", "O2", "Zn + HCl"]
    cases = [_peaky_logits(rng.choice(texts), rng, noise=rng.uniform(0.5, 2.2)) for _ in range(60)]
    strings = ["H2O", "Ca(OH)2", "CH4 + H2O", "H2+O2", "M", "Ca(", "CH4 +", "Hl", "hCl", "2H", "H02",
               "()", "Ca)", "Xz", "N4a", "Q", "JCl", "H20", "Al2(SO4)3", "CH4  + H2O", "Mg(NO3)2", "C"]
    out = page.evaluate("""async ({cases, itos, elements, strings}) => {
        %s
        const d = await import('/static/js/decode.js');
        const g = d.makeGrammar(elements);
        const grammar = strings.map((s) => { const st = g.validPrefix(s);
          return st === null ? 'dead' : (g.isComplete(st) ? 'complete' : 'prefix'); });
        const decoded = cases.map((c) => {
          const lp = new Float32Array(fromB64(c.b).buffer);
          const cands = d.beamCandidates(lp, c.T, c.C, itos, g);
          return { cands, greedy: d.greedy(lp, c.T, c.C, itos) };
        });
        return { grammar, decoded };
    }""" % JS_B64, {"cases": [{"b": b64(c.tobytes()), "T": c.shape[0], "C": c.shape[1]} for c in cases],
                    "itos": ITOS, "elements": sorted(ELEMENTS), "strings": strings})
    for s, got in zip(strings, out["grammar"]):
        st = decode.valid_prefix(s)
        want = "dead" if st is None else ("complete" if decode.is_complete(st) else "prefix")
        assert got == want, f"grammar disagrees on {s!r}: js={got} py={want}"
    agree = 0
    for lp, res in zip(cases, out["decoded"]):
        want = decode.beam_candidates(lp.astype(np.float64), ITOS)
        got = res["cands"]
        assert len(got) == len(want), f"candidate count differs: js={len(got)} py={len(want)}"
        if want:
            assert got[0]["text"] == want[0][0], f"top reading differs: js={got[0]['text']!r} py={want[0][0]!r}"
            assert abs(got[0]["logp"] - want[0][1]) < 1e-3, "top score differs"
            agree += 1
    print(f"ok   decode.js == decode.py: grammar on {len(strings)} strings, beam search on "
          f"{len(cases)} matrices ({agree} with a valid reading)")


def test_chem(page):
    raw = ["H2O", "H20", "C02", "CO2", "Ca(OH)2", "Ca(0H)2", "NaOH", "Na0H", "C10H8", "Xz", "2H",
           "H₂SO₄", " Fe2 O3 ", "C2H6O", "CH3COOH", "(NH4)2SO4", "Mg(NO3)2", "O", "0", "CO(NH2)2",
           "H2O2", "C6H1206", "hcl", "Al2(SO4)3", "K2Cr2O7", "Co", "CO"]
    pairs = [["CH4", "H2O"], ["H2O", "CH4"], ["HCl", "NaOH"], ["Na", "Cl2"], ["H20", "CH4"],
             ["Fe", "O2"], ["C3H8", "O2"], ["C2H6O", "O2"], ["C9H20", "O2"], ["He", "Ne"],
             ["CaCO3"], ["AgNO3", "NaCl"], ["CuSO4", "Fe"], ["H2", "H2"]]
    out = page.evaluate("""async ({raw, pairs}) => {
        const { loadChem } = await import('/static/js/chem.js');
        const chem = await loadChem();
        return {
          norm: raw.map((r) => chem.normalizeOcr(r)),
          parsed: raw.map((r) => chem.parseFormula(chem.normalizeOcr(r))),
          names: raw.map((r) => chem.describe(r).name),
          reactions: pairs.map((p) => { const r = chem.reactionFor(p);
            return r ? { equation: r.equation, name: r.name, plan: !!r.plan } : null; }),
        };
    }""", {"raw": raw, "pairs": pairs})
    for r, norm, parsed, name in zip(raw, out["norm"], out["parsed"], out["names"]):
        assert norm == normalize_ocr(r), f"normalize {r!r}: js={norm!r} py={normalize_ocr(r)!r}"
        assert parsed == parse_formula(normalize_ocr(r)), f"parse {r!r}: js={parsed} py={parse_formula(normalize_ocr(r))}"
        assert name == describe_formula(r)["name"], f"name {r!r}: js={name!r} py={describe_formula(r)['name']!r}"
    planned = 0
    for p, got in zip(pairs, out["reactions"]):
        want = react(p)
        if not want["ok"]:
            assert got is None, f"{p}: js found a reaction, python did not"
            continue
        assert got is not None, f"{p}: python found {want['equation']}, js found nothing"
        norm = lambda eq: sorted(eq.replace(" ", "").replace("→", "+").split("+"))   # noqa: E731
        assert norm(got["equation"]) == norm(want["equation"]), \
            f"{p}: js={got['equation']!r} py={want['equation']!r}"
        planned += bool(got["plan"])
    print(f"ok   chem.js == chemistry engine: {len(raw)} formulas, {len(pairs)} reactant sets "
          f"({planned} with a 3D plan)")


def _scenes():
    """Frames with known content, 640x480 BGR."""
    def blank(v=255):
        return np.full((480, 640, 3), v, np.uint8)

    def text(img, s, org, scale=2.0, thick=5):
        cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (15, 15, 15), thick, cv2.LINE_AA)
        return img

    two = text(text(blank(), "CH4", (50, 260)), "H2O", (380, 260))
    rng = np.random.default_rng(0)
    desk = rng.integers(60, 110, (480, 640, 3), dtype=np.uint8)
    cv2.rectangle(desk, (150, 120), (520, 380), (40, 40, 40), -1)
    cv2.rectangle(desk, (160, 110), (510, 360), (245, 245, 245), -1)
    cv2.rectangle(desk, (160, 110), (510, 360), (90, 90, 90), 3)
    text(desk, "H2O", (230, 270), 2.4, 6)
    noise = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    blobs = blank(200)
    cv2.circle(blobs, (200, 240), 60, (30, 30, 30), -1)
    cv2.rectangle(blobs, (380, 180), (520, 320), (40, 40, 40), -1)
    shaded = (blank().astype(np.float32) * np.linspace(0.55, 1.0, 640)[None, :, None]).astype(np.uint8)
    text(shaded, "NaCl", (90, 200), 1.8, 4)
    text(shaded, "Fe2O3", (330, 360), 1.6, 4)
    return {"two formulas": two, "card on a desk": desk, "pure noise": noise,
            "solid blobs": blobs, "uneven light": shaded}


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / float(a[2] * a[3] + b[2] * b[3] - inter)


def test_detect(page):
    scenes = _scenes()
    payload = [{"name": k, "b": b64(cv2.cvtColor(v, cv2.COLOR_BGR2RGBA).tobytes())} for k, v in scenes.items()]
    out = page.evaluate("""async (items) => {
        %s
        const { FormulaDetector } = await import('/static/js/detect.js');
        const det = new FormulaDetector();
        return items.map((it) => {
          const rgba = fromB64(it.b);
          const t0 = performance.now();
          let boxes;
          for (let i = 0; i < 5; i++) boxes = det.detect(rgba, 640, 480);
          return { boxes, ms: (performance.now() - t0) / 5 };
        });
    }""" % JS_B64, payload)
    expect = {"two formulas": 2, "card on a desk": 1, "pure noise": 0, "solid blobs": 0, "uneven light": 2}
    times = []
    for (name, frame), res in zip(scenes.items(), out):
        py = ar_app.detect_formula_regions(frame)
        js = [(b["x"], b["y"], b["w"], b["h"]) for b in res["boxes"]]
        times.append(res["ms"])
        assert len(js) == expect[name], f"{name}: js found {len(js)} boxes, expected {expect[name]}"
        assert len(py) == len(js), f"{name}: js found {len(js)} boxes, python {len(py)}"
        for box in js:
            assert max(_iou(box, p) for p in py) > 0.6, f"{name}: js box {box} not matched by python {py}"
    print(f"ok   detect.js matches the desktop detector on {len(scenes)} scenes "
          f"(~{np.mean(times):.1f} ms per 640x480 frame in headless Chromium)")


if __name__ == "__main__":
    tests = [test_preprocess, test_decode, test_chem, test_detect]
    passed = 0
    with browser_page() as (page, _base):
        for fn in tests:
            try:
                fn(page)
            except Exception as exc:  # noqa: BLE001
                print(f"FAIL {fn.__name__}: {exc!r}"[:1500])
            else:
                passed += 1
        if page.problems:
            print("browser problems:", page.problems[:5])
    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)
