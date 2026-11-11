"""End-to-end tests of the phone web app (/ar) in headless Chromium.

A synthetic "camera" image of paper cards on a desk stands in for the phone
camera (?src=...). Checks the whole chain on the device side: find the
writing -> read it -> show the molecule -> react when two formulas are close
-> play the animation through its phases.

Run: python tests/test_web_app.py
"""
import os
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

from tests.webharness import browser_page, playwright_available  # noqa: E402

if not playwright_available():
    print("SKIP: playwright / uvicorn not installed")
    from tests.webharness import skip_unavailable
    skip_unavailable("Optional dependency or generated asset unavailable; see message above")
try:
    from tests.scenes import save_scene  # noqa: E402
except ImportError:
    print("SKIP: opencv not installed")
    from tests.webharness import skip_unavailable
    skip_unavailable("Optional dependency or generated asset unavailable; see message above")

from backend.recognition.config import ONNX_PATH  # noqa: E402

BUNDLE = os.path.join(ROOT, "backend", "static", "data", "chem3d.json")
if not os.path.exists(BUNDLE):
    print("SKIP: chem3d.json not built (python -m backend.chemistry.build3d)")
    from tests.webharness import skip_unavailable
    skip_unavailable("Optional dependency or generated asset unavailable; see message above")

FAR = [(0.5, 0.34), (0.5, 0.74)]
NEAR = [(0.5, 0.56), (0.5, 0.70)]
STATE = "() => window.chemvision.state()"
READY = "() => window.chemvision && window.chemvision.geo"


def _open(cards, query, **scene):
    import zlib
    name = f"t_{zlib.crc32((str(cards) + str(scene)).encode()):08x}.png"   # stable across runs
    src = save_scene(name, cards, text_h=58, **scene)
    return browser_page(f"/ar?src={src}&autostart=1{query}")


def _wait(page, js, timeout=30000):
    page.wait_for_function(js, timeout=timeout)


def test_molecules_appear():
    with _open([("CH4", FAR[0]), ("H2O", FAR[1])], "&demo=CH4,H2O", seed=3) as (page, _):
        _wait(page, READY)
        _wait(page, "() => window.chemvision.state().molecules === 2")
        s = page.evaluate(STATE)
        assert sorted(t["formula"] for t in s["tracks"]) == ["CH4", "H2O"], s["tracks"]
        assert sorted(t["name"] for t in s["tracks"]) == ["Methane", "Water"]
        assert s["reaction"] is None, "cards this far apart must not react"
        labels = page.eval_on_selector_all(".lbl.mol b", "els => els.map(e => e.textContent)")
        assert sorted(labels) == ["CH₄", "H₂O"], labels
        assert not page.problems, page.problems[:3]
    print("ok   two cards -> two floating 3D molecules with names; no reaction while apart")


def test_reaction_plays_through():
    with _open([("CH4", NEAR[0]), ("O2", NEAR[1])], "&demo=CH4,O2", seed=3) as (page, _):
        _wait(page, READY)
        _wait(page, "() => { const r = window.chemvision.state().reaction; return r && r.plan; }")
        seen, captions, t0 = [], [], time.time()
        while time.time() - t0 < 40:
            r = page.evaluate(STATE)["reaction"]
            assert r is not None, "reaction ended while the cards were still together"
            if not seen or seen[-1] != r["phase"]:
                seen.append(r["phase"])
                captions.append(r["caption"])
            if r["phase"] == "done":
                break
            page.wait_for_timeout(150)
        assert seen[-4:] == ["break", "move", "form", "done"], f"phases seen: {seen}"
        text = " | ".join(captions)
        assert "Bonds break: 4 C–H, 2 O=O" in text, text
        assert "Bonds form" in text and "C=O" in text and "O–H" in text, text
        assert "C −4 → +4 (oxidised)" in text and "O 0 → −2 (reduced)" in text, text
        banner = page.inner_text("#banner .eq")
        assert banner == "CH₄ + 2 O₂ → CO₂ + 2 H₂O", banner
        meta = page.inner_text("#banner .meta")
        assert "releases" in meta and "890" in meta, meta
        assert page.evaluate("() => document.getElementById('caption').classList.contains('on')")
        assert not page.problems, page.problems[:3]
    print("ok   two cards together -> balanced equation, 3D animation through break/move/form, "
          "oxidation-state summary")


def test_unknown_pair_and_separation():
    with _open([("He", NEAR[0]), ("Ne", NEAR[1])], "&demo=He,Ne", seed=5) as (page, _):
        _wait(page, READY)
        _wait(page, "() => document.getElementById('hint').classList.contains('on')")
        hint = page.inner_text("#hint")
        assert "No known reaction" in hint and "He" in hint and "Ne" in hint, hint
        assert page.evaluate(STATE)["reaction"] is None
        assert not page.problems, page.problems[:3]
    print("ok   a pair with no known reaction is reported as such, nothing is invented")


def test_tap_to_decompose():
    with _open([("CaCO3", FAR[0]), ("H2O", FAR[1])], "&demo=CaCO3,H2O", seed=7) as (page, _):
        _wait(page, READY)
        _wait(page, "() => window.chemvision.state().molecules === 2")
        pos = page.evaluate("""() => { const a = window.chemvision;
            const t = a.tracker.tracks.find((x) => x.formula === 'CaCO3');
            return { x: (t.smooth.x + t.smooth.w / 2) * a.geo.k, y: (t.smooth.y + t.smooth.h / 2) * a.geo.k }; }""")
        page.mouse.click(pos["x"], pos["y"])
        _wait(page, "() => document.getElementById('sheet').classList.contains('on')")
        assert "Calcium carbonate" in page.inner_text("#sheet .nm")
        assert "can decompose" in page.inner_text("#sheet .more")
        page.click("#decompose")
        _wait(page, "() => { const r = window.chemvision.state().reaction; return r && r.key === 'CaCO3' && r.plan; }")
        assert "CaO" in page.inner_text("#banner .eq") and "absorbs" in page.inner_text("#banner .meta")
        assert not page.problems, page.problems[:3]
    print("ok   tapping a formula shows its details; 'Break it down' plays its decomposition")


def test_reads_handwriting_on_device():
    if not os.path.exists(ONNX_PATH):
        print("skip on-device reading: no exported model (python -m backend.recognition.export)")
        return
    # Freeze the image, not the font-rendering library: Pillow versions produce
    # different strokes and make this an accidental OCR benchmark otherwise.
    from pathlib import Path
    fixture = Path(ROOT) / "tests" / "fixtures" / "handwriting_cards.png"
    target = Path(ROOT) / "backend" / "static" / "test" / fixture.name
    target.parent.mkdir(exist_ok=True)
    target.write_bytes(fixture.read_bytes())
    with browser_page(f"/ar?src=/static/test/{fixture.name}") as (page, _):
        _wait(page, READY)
        _wait(page, "() => window.chemvision.state().tracks.filter((t) => t.formula).length === 2", 45000)
        s = page.evaluate(STATE)
        assert s["mode"] == "on-device", s["mode"]
        assert sorted(t["formula"] for t in s["tracks"]) == ["H2O", "NaCl"], s["tracks"]
        assert all(t["confidence"] >= 0.5 for t in s["tracks"])
        assert not page.problems, page.problems[:3]
    print(f"ok   handwriting read on the device: {[(t['formula'], t['confidence']) for t in s['tracks']]} "
          f"(find {s['stats']['detectMs']:.1f} ms, read {s['stats']['readMs']:.0f} ms per crop)")


if __name__ == "__main__":
    funcs = [test_molecules_appear, test_reaction_plays_through, test_unknown_pair_and_separation,
             test_tap_to_decompose, test_reads_handwriting_on_device]
    passed = 0
    for fn in funcs:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {fn.__name__}: {exc!r}"[:1200])
        else:
            passed += 1
    print(f"\n{passed}/{len(funcs)} passed")
    sys.exit(0 if passed == len(funcs) else 1)
