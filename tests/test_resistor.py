"""Resistor colour code and the trained band reader.

- the colour-code rules (which way round, what counts as a legal code);
- the JavaScript twin (static/js/assist/resistor.js) agrees with Python;
- the exported model reads fresh drawings, stays quiet when there is no
  resistor, and reads the real photos kept in tests/fixtures/resistors/.

The real-photo check is a floor that catches a broken model, not an accuracy
claim: seven studio photos say little about a head-worn camera in a classroom.
Measure your own with tests/eval_resistor_real.py.
"""
import os
import random

import pytest

from backend.resistor.bands import COLOURS, decode, format_ohms, value_of

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "backend", "static", "models", "resistor.onnx")
FIXTURES = os.path.join(ROOT, "tests", "fixtures", "resistors")


def test_colour_code_rules():
    r = decode(["yellow", "violet", "red", "gold"])
    assert (r["ohms"], r["tolerance"], r["ambiguous"]) == (4700, 5, False)
    assert decode(["gold", "red", "violet", "yellow"])["ohms"] == 4700, "held the other way round"
    assert decode(["red", "red", "brown"])["tolerance"] == 20, "three bands: no tolerance band"
    assert decode(["grey", "red", "black", "gold", "brown"])["ohms"] == pytest.approx(82)
    assert decode(["yellow", "violet", "gold", "gold"])["ohms"] == pytest.approx(4.7)
    assert decode(["red", "red", "red", "red"])["ambiguous"] is False, "a palindrome reads the same both ways"
    # 10 k 1 % reads as a legal, standard 120 ohm from the other end: say so instead of guessing
    both = decode(["brown", "black", "black", "red", "brown"])
    assert both["ambiguous"] and {both["ohms"], both["alternative"]["ohms"]} == {10000, 120}
    # a standard value beats a non-standard one
    assert decode(["brown", "red", "black", "brown"])["ohms"] == 12
    for bad in (["gold", "gold", "gold"], ["red", "red"], ["red"] * 6, ["black", "brown", "red", "gold"], []):
        assert decode(bad) is None, bad
    assert value_of(["black", "brown", "red"]) is None, "a value never starts with black"
    assert [format_ohms(v) for v in (4700, 1e6, 0.47, 82, 2.2e9, 1500)] == ["4.7 kΩ", "1 MΩ", "0.47 Ω", "82 Ω", "2.2 GΩ", "1.5 kΩ"]


def test_javascript_twin_matches_python():
    from tests.webharness import browser_page, playwright_available
    if not playwright_available():
        pytest.skip("playwright / uvicorn not installed")
    rng = random.Random(5)
    cases = [[rng.choice(COLOURS) for _ in range(rng.choice([3, 4, 4, 4, 5, 5, 2, 6]))] for _ in range(500)]
    with browser_page() as (page, _):
        got = page.evaluate("""async (cases) => {
            const { decodeBands, formatOhms, COLOURS } = await import('/static/js/assist/resistor.js');
            return { colours: COLOURS, out: cases.map((b) => { const r = decodeBands(b);
              return r && { ohms: r.ohms, tol: r.tolerance, amb: r.ambiguous, std: r.standard, bands: r.bands,
                            alt: r.alternative ? r.alternative.ohms : null, text: formatOhms(r.ohms) }; }) };
        }""", cases)
    assert got["colours"] == COLOURS
    legal = 0
    for bands, js in zip(cases, got["out"]):
        py = decode(bands)
        if py is None:
            assert js is None, bands
            continue
        legal += 1
        assert js["ohms"] == pytest.approx(py["ohms"], rel=1e-9) and js["tol"] == py["tolerance"], bands
        assert js["amb"] == py["ambiguous"] and js["std"] == py["standard"] and js["bands"] == py["bands"], bands
        assert js["alt"] == (pytest.approx(py["alternative"]["ohms"], rel=1e-9) if py["ambiguous"] else None), bands
        assert js["text"] == format_ohms(py["ohms"]), bands
    assert legal > 100


def _session():
    if not os.path.exists(MODEL):
        pytest.skip("resistor model missing: python -m backend.resistor.export")
    ort = pytest.importorskip("onnxruntime")
    return ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])


def test_reader_on_fresh_drawings():
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    from backend.resistor.render import render_sample
    from tests.eval_resistor_real import read
    session = _session()
    rng = np.random.default_rng(31337)
    right = parts = quiet = empties = 0
    for _ in range(400):
        img, labels = render_sample(rng, None)
        bands = read(session, img)
        if not labels:
            empties += 1
            quiet += not bands
            continue
        parts += 1
        want, got = decode([COLOURS[i - 1] for i in labels]), decode(bands) if 3 <= len(bands) <= 5 else None
        right += bool(got and abs(got["ohms"] - want["ohms"]) < 1e-9 * max(1, want["ohms"]) and got["tolerance"] == want["tolerance"])
    # v2 (8 epochs): 307/361 values and 39/39 empty pictures at this fixed seed.
    # Keep a little margin for numerical/rendering differences, but catch regressions.
    assert right / parts >= 0.82, f"only {right}/{parts} drawn resistors read correctly"
    assert quiet / empties >= 0.95, f"read a resistor into {empties - quiet}/{empties} pictures that have none"


def test_reader_on_real_photos():
    pytest.importorskip("cv2")
    from tests.eval_resistor_real import evaluate
    _session()
    if not os.path.isdir(FIXTURES):
        pytest.skip("no real photos in tests/fixtures/resistors")
    summary = evaluate(FIXTURES, verbose=False)
    ok, n = summary["as taken"]
    assert n >= 5
    # v2: 7/7 as taken, 51/56 across all eight views (each view at least 6/7).
    assert ok / n >= 0.85, f"only {ok}/{n} real photos read correctly as taken: {summary}"
    for label, (correct, count) in summary.items():
        assert correct / count >= 0.80, f"only {correct}/{count} real photos at {label}: {summary}"
    total_ok = sum(v[0] for v in summary.values())
    total = sum(v[1] for v in summary.values())
    assert total_ok / total >= 0.88, f"real photos under varied conditions: {summary}"
