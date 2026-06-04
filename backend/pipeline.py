"""Server-side frame processing for the mobile web client.

The phone can't run the model, so it streams camera frames here. We reuse the
exact same AR pipeline the desktop client uses (detection -> tracking ->
recognition -> stability -> proximity -> reaction) but with recognition and
chemistry called *in process* instead of over HTTP. A frame goes in; structured
boxes + the balanced reaction come out as plain data the browser overlays.

Reusing the desktop client's vision logic keeps behavior identical across
desktop and mobile. (The client module only depends on cv2/numpy/requests.)
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from clients.desktop.ar_app import ARApp, AppConfig, Tracker, to_ascii
from backend.chemistry import describe_formula, normalize_ocr
from backend.chemistry import react as chem_react
from backend.recognition import infer


class _InProcessClient:
    """Drop-in replacement for the desktop BackendClient that calls the model
    and chemistry engine directly (no HTTP)."""

    def __init__(self):
        self.recognizer_ok = infer.recognizer_available()

    def recognize(self, crop_bgr) -> Optional[dict]:
        if not self.recognizer_ok:
            return None
        out = infer.get_recognizer().predict(crop_bgr)
        formula = normalize_ocr(out["raw"])
        return {
            "raw": out["raw"],
            "formula": formula,
            "confidence": out["confidence"],
            "describe": describe_formula(formula) if formula else None,
        }

    def react(self, reactants):
        return chem_react(reactants)


def build_session() -> ARApp:
    """Build a stateful AR session (tracker + stability) for one phone stream.

    Tuned for the low frame rate of a network video stream: recognize every
    frame and confirm a label after 2 agreeing reads (instead of the desktop's
    8-frame interval, which would take many seconds at ~3 fps).
    """
    cfg = AppConfig(max_regions=3, recognize_interval=1, stable_reads=2,
                    min_confidence=0.5, proximity_px=70.0,
                    min_components=2, min_track_hits=2)
    app = ARApp.__new__(ARApp)          # bypass __init__'s HTTP health check
    app.cfg = cfg
    app.client = _InProcessClient()
    app.tracker = Tracker()
    app.frame_idx = 0
    app.reaction = None
    app.reaction_expires = 0.0
    app.debug = False
    app.last_mask = None
    app._demo_assigned = 0
    return app


def decode_frame(data: bytes):
    """Decode uploaded JPEG/PNG bytes into a BGR image array (or None)."""
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def process_to_state(session: ARApp, frame) -> dict:
    """Run one frame through the pipeline and return overlay data for the browser.

    Returns frame size, the tracked formula boxes (with confirmed labels +
    confidence), and the current reaction (if two formulas are close).
    """
    session.process_frame(frame)        # updates tracker + reaction in place
    h, w = frame.shape[:2]
    tracks = []
    for t in session.tracker.tracks:
        if t.hits < session.cfg.min_track_hits:   # not persisted enough -> hide
            continue
        x, y, bw, bh = t.box
        tracks.append({
            "x": int(x), "y": int(y), "w": int(bw), "h": int(bh),
            "label": (t.pretty or t.formula) if t.formula else None,
            "confidence": round(float(t.confidence), 2) if t.formula else None,
            "confirmed": t.formula is not None,
        })

    reaction = None
    if session.reaction and session.reaction.get("ok"):
        reaction = {
            "equation": session.reaction["equation"],
            "ascii": to_ascii(session.reaction["equation"]),
            "name": session.reaction.get("name"),
            "conditions": session.reaction.get("conditions"),
        }
    return {"width": int(w), "height": int(h), "tracks": tracks, "reaction": reaction}
