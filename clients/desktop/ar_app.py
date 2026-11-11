"""Chem_Vision desktop AR client.

Live webcam app:
  1. detect handwritten-formula blobs in the frame (ink, not paper edges),
  2. send each crop to the backend /recognize endpoint (the CV model),
  3. overlay the recognized formula on top of it ("display what's written"),
  4. when two recognized formulas are brought CLOSE together, call /react and
     overlay the balanced reaction ("CH4 + H2O -> CO + 3 H2").

Run the backend first:
    uvicorn backend.app:app --port 8000

Then:
    python -m clients.desktop.ar_app --server http://localhost:8000

Useful flags:
    --camera 0              webcam index
    --image path.png        run once on a still image instead of the webcam
    --demo-labels CH4,H2O   skip recognition; tag the first N detected blobs with
                            these labels (demo the AR flow without the model).
    --min-confidence 0.5    drop reads below this confidence
    --stable-reads 3        require this many identical reads before showing a label

Keys:  q quit · space pause · r re-read · d toggle detection-mask debug view
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

try:
    import cv2
    import numpy as np
    import requests
except ImportError as exc:  # pragma: no cover
    print(f"Missing dependency: {exc}. Install: pip install opencv-python numpy requests")
    raise SystemExit(1)


# --------------------------------------------------------------------------
# Backend client
# --------------------------------------------------------------------------
class BackendClient:
    def __init__(self, base_url: str, timeout: float = 4.0):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.recognizer_ok = self._check_health()

    def _check_health(self) -> bool:
        try:
            r = requests.get(f"{self.base}/health", timeout=self.timeout)
            return bool(r.json().get("recognizer_available"))
        except requests.RequestException:
            print(f"[warn] backend not reachable at {self.base}; chemistry/recognition disabled")
            return False

    def recognize(self, crop_bgr) -> Optional[dict]:
        ok, buf = cv2.imencode(".png", crop_bgr)
        if not ok:
            return None
        try:
            r = requests.post(f"{self.base}/recognize",
                              files={"image": ("crop.png", buf.tobytes(), "image/png")},
                              timeout=self.timeout)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        return None

    def react(self, reactants: list[str]) -> Optional[dict]:
        try:
            r = requests.post(f"{self.base}/react", json={"reactants": reactants},
                              timeout=self.timeout)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        return None


# --------------------------------------------------------------------------
# Detection — isolate dark ink, not paper edges / shadows
# --------------------------------------------------------------------------
def _detection_mask(gray):
    """Binary mask of ink strokes, robust to paper borders and uneven light.

    Black-hat morphology highlights dark features (ink) smaller than the kernel
    on a lighter background, which naturally suppresses the large bright paper
    and its shadowed edges. We add an absolute floor so faint noise isn't
    amplified by Otsu in a low-contrast (dim) scene.
    """
    h_img = gray.shape[0]
    k = max(15, h_img // 28) | 1  # odd kernel ~ character height
    bh_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, bh_kernel)
    _, mask = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask[blackhat < 18] = 0  # absolute floor: kill faint shadow/noise response
    return mask


def _nms(boxes, iou_thresh=0.3):
    """Keep larger boxes, drop ones overlapping an already-kept box."""
    keep = []
    for b in sorted(boxes, key=lambda b: b[2] * b[3], reverse=True):
        if all(_iou(b, k) < iou_thresh for k in keep):
            keep.append(b)
    return keep


def _char_like_components(mask_roi, box_w, box_h) -> int:
    """Count character-shaped connected components inside a candidate box.

    Real handwritten formulas are made of several roughly line-height strokes
    (the characters). A random dark blob on the background is one solid shape,
    and fine texture is many tiny specks — neither produces multiple
    character-sized components. This is the main "is it actually text?" filter.
    """
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask_roi, connectivity=8)
    box_area = float(box_w * box_h)
    count = 0
    for i in range(1, n):                    # skip background label 0
        _, _, w, h, area = stats[i]
        tall_enough = h >= 0.33 * box_h      # spans much of the writing line
        sized = 0.008 * box_area <= area <= 0.85 * box_area  # not noise, not the whole box
        if tall_enough and sized:
            count += 1
    return count


def detect_formula_regions(frame_bgr, max_regions=3, return_mask=False,
                           min_components=2):
    """Find candidate handwritten-formula bounding boxes (x, y, w, h).

    `min_components` requires at least this many character-shaped parts in a
    region, which rejects non-text blobs/backgrounds. Lower it to 1 if you write
    very short or fully-connected (cursive) formulas, at the cost of more false
    positives.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    h_img, w_img = gray.shape
    frame_area = float(w_img * h_img)

    mask = _detection_mask(gray)
    # merge characters horizontally into formula-sized blobs (gap ~ stroke run)
    merge_w = max(15, w_img // 50)
    merge_k = cv2.getStructuringElement(cv2.MORPH_RECT, (merge_w, 9))
    merged = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, merge_k, iterations=2)
    merged = cv2.morphologyEx(merged, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    m = 5
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < 0.0008 * frame_area or area > 0.5 * frame_area:
            continue
        if w < 24 or h < 14:
            continue
        if not (0.7 <= w / float(h) <= 18):       # formulas are roughly horizontal
            continue
        if x <= m or y <= m or x + w >= w_img - m or y + h >= h_img - m:
            continue                              # hugging the frame border -> paper/hand
        roi = mask[y:y + h, x:x + w]
        density = float((roi > 0).mean())
        if not (0.04 <= density <= 0.6):          # text has moderate ink density
            continue
        if _char_like_components(roi, w, h) < min_components:
            continue                              # not text-like -> reject
        boxes.append((x, y, w, h))

    boxes = _nms(boxes, 0.3)[:max_regions]
    return (boxes, merged) if return_mask else boxes


# --------------------------------------------------------------------------
# Tracking + per-track recognition stability
# --------------------------------------------------------------------------
@dataclass
class Track:
    tid: int
    box: tuple
    formula: Optional[str] = None      # confirmed label (None until stable)
    pretty: Optional[str] = None
    confidence: float = 0.0
    reads: list = field(default_factory=list)  # recent (formula, conf, pretty) votes
    last_seen: int = 0
    last_recognized: int = -999
    misses: int = 0
    hits: int = 1                              # frames this track has matched a box

    def centroid(self):
        x, y, w, h = self.box
        return (x + w / 2, y + h / 2)


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


class Tracker:
    """Greedy IoU tracker so a formula keeps its label across frames."""

    def __init__(self, iou_thresh=0.1, max_misses=4):
        self.tracks: list[Track] = []
        self.iou_thresh = iou_thresh
        self.max_misses = max_misses
        self._next_id = 0

    def update(self, boxes, frame_idx) -> list[Track]:
        unmatched = set(range(len(boxes)))
        for tr in self.tracks:
            best, best_iou = -1, self.iou_thresh
            for i in unmatched:
                v = _iou(tr.box, boxes[i])
                if v >= best_iou:
                    best, best_iou = i, v
            if best >= 0:
                tr.box = boxes[best]
                tr.last_seen = frame_idx
                tr.misses = 0
                tr.hits += 1
                unmatched.discard(best)
            else:
                tr.misses += 1
        for i in unmatched:
            self.tracks.append(Track(self._next_id, boxes[i], last_seen=frame_idx))
            self._next_id += 1
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        return self.tracks


def box_gap(a, b) -> float:
    """Shortest gap between two axis-aligned boxes (0 if overlapping)."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    dx = max(bx - (ax + aw), ax - (bx + bw), 0)
    dy = max(by - (ay + ah), ay - (by + bh), 0)
    return (dx ** 2 + dy ** 2) ** 0.5


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
GREEN, YELLOW, RED, WHITE = (0, 220, 0), (0, 220, 220), (0, 0, 255), (255, 255, 255)

# OpenCV's Hershey fonts are ASCII-only: subscripts and the reaction arrow would
# render as garbage. Map them back to plain ASCII before drawing.
_SUB_TO_DIGIT = {ord(c): str(i) for i, c in enumerate("₀₁₂₃₄₅₆₇₈₉")}


def to_ascii(text: str) -> str:
    if not text:
        return text
    return text.translate(_SUB_TO_DIGIT).replace("→", "->")


def draw_label(frame, text, org, color=GREEN, scale=0.7, bg=True):
    text = to_ascii(text)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(text, font, scale, 2)
    x, y = org
    if bg:
        cv2.rectangle(frame, (x - 3, y - th - 6), (x + tw + 3, y + base), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font, scale, color, 2, cv2.LINE_AA)


def render(frame, tracks, reaction, server_ok, recognizer_ok):
    for tr in tracks:
        x, y, w, h = tr.box
        if tr.formula:                              # confirmed
            cv2.rectangle(frame, (x, y), (x + w, y + h), GREEN, 2)
            conf = f" {tr.confidence:.0%}" if tr.confidence else ""
            draw_label(frame, f"{tr.pretty or tr.formula}{conf}", (x, max(20, y - 8)))
        else:                                       # still reading / unconfirmed
            cv2.rectangle(frame, (x, y), (x + w, y + h), YELLOW, 1)
            draw_label(frame, "reading...", (x, max(20, y - 8)), YELLOW, 0.5)

    status = ("backend:OK" if server_ok else "backend:DOWN")
    status += "  model:OK" if recognizer_ok else "  model:not-trained"
    draw_label(frame, status, (10, frame.shape[0] - 12),
               GREEN if server_ok else RED, 0.55)

    if reaction and reaction.get("ok"):
        eq = reaction["equation"]
        name = reaction.get("name") or ""
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), (30, 30, 30), -1)
        draw_label(frame, eq, (12, 34), GREEN, 0.9, bg=False)
        if name:
            draw_label(frame, name, (12, 60), WHITE, 0.6, bg=False)
    return frame


# --------------------------------------------------------------------------
# Main app
# --------------------------------------------------------------------------
@dataclass
class AppConfig:
    server: str = "http://localhost:8000"
    camera: int = 0
    image: Optional[str] = None
    proximity_px: float = 70.0
    recognize_interval: int = 8       # frames between recognition attempts per track
    min_confidence: float = 0.5
    stable_reads: int = 3             # identical reads needed to confirm a label
    max_regions: int = 3
    min_components: int = 2           # char-shaped parts required to call a blob text
    min_track_hits: int = 3           # frames a box must persist before it's shown
    demo_labels: list[str] = field(default_factory=list)


class ARApp:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.client = BackendClient(cfg.server)
        self.tracker = Tracker()
        self.frame_idx = 0
        self.reaction: Optional[dict] = None
        self.reaction_expires = 0.0
        self.debug = False
        self.last_mask = None
        self._demo_assigned = 0

    def _maybe_recognize(self, frame, tr: Track, force=False):
        # demo mode: assign provided labels to the first detected blobs
        if self.cfg.demo_labels and tr.formula is None:
            if self._demo_assigned < len(self.cfg.demo_labels):
                tr.formula = self.cfg.demo_labels[self._demo_assigned]
                tr.pretty = tr.formula
                tr.confidence = 1.0
                self._demo_assigned += 1
            return
        if not self.client.recognizer_ok:
            return
        due = self.frame_idx - tr.last_recognized >= self.cfg.recognize_interval
        if not (force or due):
            return
        x, y, w, h = tr.box
        pad = 6
        crop = frame[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]
        if crop.size == 0:
            return
        res = self.client.recognize(crop)
        tr.last_recognized = self.frame_idx
        if not res:
            return
        formula = res.get("formula")
        conf = res.get("confidence", 0.0)
        valid = (res.get("describe") or {}).get("valid", False)
        # reject low-confidence or chemically-invalid reads outright
        if not formula or conf < self.cfg.min_confidence or not valid or len(formula) < 2:
            return
        # stability gate: majority vote over a sliding window of recent reads.
        # Confirmation still needs `stable_reads` agreeing reads, but unlike
        # requiring N *identical consecutive* reads, one flickered misread no
        # longer resets progress — it's simply outvoted.
        pretty = (res.get("describe") or {}).get("pretty") or formula
        tr.reads.append((formula, conf, pretty))
        tr.reads = tr.reads[-(2 * self.cfg.stable_reads - 1):]
        votes: dict[str, list] = {}
        for f, c, p in tr.reads:
            votes.setdefault(f, []).append((c, p))
        winner, wvotes = max(votes.items(),
                             key=lambda kv: (len(kv[1]), sum(c for c, _ in kv[1])))
        if len(wvotes) >= self.cfg.stable_reads:
            tr.formula = winner
            tr.pretty = wvotes[-1][1]
            tr.confidence = sum(c for c, _ in wvotes) / len(wvotes)

    def _check_proximity(self, tracks):
        labeled = [t for t in tracks if t.formula]
        if len(labeled) < 2:
            return
        best = None
        for i in range(len(labeled)):
            for j in range(i + 1, len(labeled)):
                gap = box_gap(labeled[i].box, labeled[j].box)
                if best is None or gap < best[0]:
                    best = (gap, labeled[i], labeled[j])
        gap, a, b = best
        if gap <= self.cfg.proximity_px:
            res = self.client.react([a.formula, b.formula])
            if res is not None:
                self.reaction = res
                self.reaction_expires = time.time() + 4.0

    def process_frame(self, frame):
        self.frame_idx += 1
        boxes, self.last_mask = detect_formula_regions(
            frame, self.cfg.max_regions, return_mask=True,
            min_components=self.cfg.min_components)
        tracks = self.tracker.update(boxes, self.frame_idx)
        # only act on boxes that have persisted a few frames -> ignores the
        # one-frame flicker that random backgrounds produce
        visible = [t for t in tracks if t.hits >= self.cfg.min_track_hits]
        for tr in visible:
            self._maybe_recognize(frame, tr)
        self._check_proximity(visible)
        if self.reaction and time.time() > self.reaction_expires:
            self.reaction = None
        return render(frame, visible, self.reaction,
                      self.client.recognizer_ok or bool(self.cfg.demo_labels),
                      self.client.recognizer_ok)

    # --- run modes --------------------------------------------------------
    def _show_debug(self):
        if self.debug and self.last_mask is not None:
            cv2.imshow("Chem_Vision mask (debug)", self.last_mask)

    def run_image(self):
        frame = cv2.imread(self.cfg.image)
        if frame is None:
            print(f"Could not read image: {self.cfg.image}")
            return
        # recognize needs a few frames to confirm; loop a few times on the still
        out = frame
        for _ in range(self.cfg.stable_reads + 1):
            out = self.process_frame(frame.copy())
        cv2.imshow("Chem_Vision (image)", out)
        print("Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def run_camera(self):
        cap = cv2.VideoCapture(self.cfg.camera)
        if not cap.isOpened():
            print(f"Could not open camera {self.cfg.camera}")
            return
        paused = False
        print("Running. Keys: q quit | space pause | r re-read | d debug mask")
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    break
                out = self.process_frame(frame)
                cv2.imshow("Chem_Vision AR", out)
                self._show_debug()
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                paused = not paused
            if key == ord("r"):
                for tr in self.tracker.tracks:
                    tr.formula = None
                    tr.reads.clear()
                self._demo_assigned = 0
            if key == ord("d"):
                self.debug = not self.debug
                if not self.debug:
                    cv2.destroyWindow("Chem_Vision mask (debug)")
        cap.release()
        cv2.destroyAllWindows()

    def run(self):
        if self.cfg.image:
            self.run_image()
        else:
            self.run_camera()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Chem_Vision desktop AR client")
    p.add_argument("--server", default="http://localhost:8000")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--image", default=None, help="run once on a still image")
    p.add_argument("--proximity-px", type=float, default=70.0)
    p.add_argument("--recognize-interval", type=int, default=8)
    p.add_argument("--min-confidence", type=float, default=0.5)
    p.add_argument("--stable-reads", type=int, default=3)
    p.add_argument("--max-regions", type=int, default=3)
    p.add_argument("--min-components", type=int, default=2,
                   help="char-shaped parts required to accept a blob as text (1 = lenient)")
    p.add_argument("--min-track-hits", type=int, default=3,
                   help="frames a box must persist before it's shown")
    p.add_argument("--demo-labels", default="",
                   help="comma-separated labels to tag detected blobs (skips recognition)")
    a = p.parse_args(argv)
    return AppConfig(
        server=a.server, camera=a.camera, image=a.image,
        proximity_px=a.proximity_px, recognize_interval=a.recognize_interval,
        min_confidence=a.min_confidence, stable_reads=a.stable_reads,
        max_regions=a.max_regions, min_components=a.min_components,
        min_track_hits=a.min_track_hits,
        demo_labels=[s.strip() for s in a.demo_labels.split(",") if s.strip()],
    )


if __name__ == "__main__":
    ARApp(parse_args()).run()
