"""Headless test of the desktop client's vision/AR logic (no webcam, no display).

Exercises: blob detection -> tracking across frames -> demo-label assignment ->
proximity trigger -> chemistry. The HTTP layer is covered separately by
test_api.py, so here we stub the backend client to call the chemistry engine
directly. Run: python tests/test_client_headless.py
"""
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows cp1252 chokes on subscripts
except (AttributeError, ValueError):
    pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cv2
    import numpy as np
except ImportError:
    print("SKIP: opencv-python/numpy not installed")
    sys.exit(0)

from clients.desktop import ar_app  # noqa: E402
from backend.chemistry import react as chem_react  # noqa: E402


class StubClient:
    """Stand-in for BackendClient: no network, chemistry computed in-process."""
    recognizer_ok = False

    def react(self, reactants):
        return chem_react(reactants)

    def recognize(self, crop):
        return None


def _frame_with(formulas):
    """White frame with each (text, x) drawn as bold black ink."""
    img = np.full((360, 720, 3), 255, np.uint8)
    for text, x in formulas:
        cv2.putText(img, text, (x, 200), cv2.FONT_HERSHEY_SIMPLEX, 2.2,
                    (0, 0, 0), 6, cv2.LINE_AA)
    return img


def run():
    cfg = ar_app.AppConfig(demo_labels=["CH4", "H2O"], proximity_px=70.0)
    app = ar_app.ARApp(cfg)          # health check fails gracefully (server down)
    app.client = StubClient()

    # 1) detection finds two blobs on a frame
    first = _frame_with([("CH4", 80), ("H2O", 480)])
    boxes = ar_app.detect_formula_regions(first)
    assert len(boxes) == 2, f"expected 2 regions, got {len(boxes)}"
    print(f"ok   detection found {len(boxes)} formula regions")

    # 2) drive H2O leftward toward CH4 over several frames; labels must persist
    triggered = False
    out = first
    for step in range(11):
        h2o_x = 480 - step * 26       # 480 -> 220
        frame = _frame_with([("CH4", 80), ("H2O", h2o_x)])
        out = app.process_frame(frame)
        if app.reaction and app.reaction.get("ok"):
            triggered = True
            break

    labeled = [t for t in app.tracker.tracks if t.formula]
    assert len(labeled) >= 2, f"labels did not persist across frames: {labeled}"
    print(f"ok   tracking kept {len(labeled)} labeled formulas across motion")

    assert triggered, "proximity never triggered a reaction"
    eq = app.reaction["equation"]
    assert app.reaction["products"] == {"CO": 1, "H2": 3}
    print(f"ok   proximity triggered reaction: {eq}")

    cv2.imwrite(os.path.join(os.path.dirname(__file__), "..", "_client_preview.png"), out)
    print("ok   wrote annotated frame -> _client_preview.png")
    print("\n4/4 passed")


if __name__ == "__main__":
    run()
