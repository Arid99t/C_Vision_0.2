"""Test the mobile web endpoints (/ar page + /frame) and the server-side
reaction wiring. Deterministic: detection uses drawn text, and the reaction
path is exercised via demo-labels (recognition accuracy is covered separately
by eval_recognizer.py). Run: python tests/test_webapp.py
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cv2
    import numpy as np
except ImportError:
    print("SKIP: opencv/numpy not installed")
    from tests.webharness import skip_unavailable
    skip_unavailable("Optional dependency or generated asset unavailable; see message above")

from fastapi.testclient import TestClient  # noqa: E402
from backend.app import app  # noqa: E402
from backend import pipeline  # noqa: E402
from clients.desktop.ar_app import Track, to_ascii  # noqa: E402

client = TestClient(app)


def _two_words_far_apart():
    """Two formulas with a wide gap so detection reliably returns two boxes."""
    canvas = np.full((360, 800, 3), 255, np.uint8)
    cv2.putText(canvas, "CH4", (70, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (10, 10, 10), 5)
    cv2.putText(canvas, "H2O", (520, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (10, 10, 10), 5)
    return cv2.imencode(".jpg", canvas)[1].tobytes()


def test_ar_page():
    r = client.get("/ar")
    assert r.status_code == 200 and "Chem_Vision AR" in r.text
    print("ok   /ar serves the web app page")


def test_frame_structure():
    # send a few frames (a stream) so boxes pass the persistence/hits gate
    img = _two_words_far_apart()
    data = None
    for _ in range(4):
        r = client.post("/frame", files={"image": ("f.jpg", img, "image/jpeg")})
        assert r.status_code == 200
        data = r.json()
    assert data["width"] == 800 and data["height"] == 360
    assert isinstance(data["tracks"], list) and len(data["tracks"]) >= 2
    print(f"ok   /frame detects + returns structure ({len(data['tracks'])} tracks)")


def test_reaction_wiring():
    # two recognized formulas placed close -> proximity should produce the reaction
    session = pipeline.build_session()
    session.tracker.tracks = [
        Track(0, (120, 140, 90, 50), formula="CH4", pretty="CH₄", confidence=1.0),
        Track(1, (230, 140, 90, 50), formula="H2O", pretty="H₂O", confidence=1.0),
    ]
    session._check_proximity(session.tracker.tracks)
    assert session.reaction and session.reaction.get("ok"), "no reaction produced"
    eq = to_ascii(session.reaction["equation"])
    assert "CO" in eq
    print(f"ok   proximity -> reaction wiring: {eq}")


if __name__ == "__main__":
    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in funcs:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {fn.__name__}: {exc!r}")
        else:
            passed += 1
    print(f"\n{passed}/{len(funcs)} passed")
    sys.exit(0 if passed == len(funcs) else 1)
