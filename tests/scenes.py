"""Synthetic camera scenes for the browser tests: paper cards with handwritten
formulas lying on a desk, as a phone would see them.

    frame = make_scene([("CH4", (0.30, 0.42)), ("H2O", (0.70, 0.58))])

Handwriting comes from the held-out fonts of the training-data generator when
they are installed, otherwise from an OpenCV font (still dark ink on paper).
"""
from __future__ import annotations

import os
import random
import subprocess
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

STATIC_TEST_DIR = os.path.join(ROOT, "backend", "static", "test")


def _ink(text: str, rng: random.Random, height: int) -> np.ndarray:
    """Ink map (float32, 1 = ink) of `text`, `height` pixels tall."""
    try:
        from backend.recognition.fontbank import FontBank
        from backend.recognition.synthv2 import render_font
        bank = FontBank("test")
        for _ in range(20):
            ink = render_font(text, rng, bank.pick(rng))
            if ink is not None:
                break
        else:
            raise RuntimeError("render failed")
    except Exception:  # noqa: BLE001  (fonts not installed)
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 3.0, 7)
        img = np.zeros((th + base + 20, tw + 20), np.uint8)
        cv2.putText(img, text, (10, th + 10), cv2.FONT_HERSHEY_SIMPLEX, 3.0, 255, 7, cv2.LINE_AA)
        ys, xs = np.nonzero(img)
        ink = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32) / 255.0
    scale = height / ink.shape[0]
    return cv2.resize(ink, None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)


def desk(size, seed=0) -> np.ndarray:
    """A plain mid-tone desk with soft shading (BGR uint8)."""
    w, h = size
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = 105 + 22 * np.sin(xx / w * 2.3 + 0.4) + 14 * (yy / h)
    img = np.stack([base * 0.72, base * 0.86, base], axis=2)          # warm wood tone
    img += rng.normal(0, 3.0, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def paste_card(frame, text, centre, text_h=70, angle=0.0, seed=0, pen=(40, 28, 20)):
    """Draw one paper card with `text` on it. `centre` is (x, y) as a fraction of
    the frame. Returns the card's text box (x, y, w, h) in pixels."""
    rng = random.Random(seed)
    H, W = frame.shape[:2]
    ink = _ink(text, rng, text_h)
    ih, iw = ink.shape
    cw, ch = iw + int(1.5 * text_h), ih + int(1.3 * text_h)
    card = np.full((ch, cw, 3), 244, np.float32)
    card[..., 0] -= 6                                                     # slightly warm paper
    y0, x0 = (ch - ih) // 2, (cw - iw) // 2
    for c in range(3):
        card[y0:y0 + ih, x0:x0 + iw, c] = card[y0:y0 + ih, x0:x0 + iw, c] * (1 - ink) + pen[c] * ink
    mask = np.ones((ch, cw), np.float32)
    m = cv2.getRotationMatrix2D((cw / 2, ch / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(cw * cos + ch * sin) + 2, int(cw * sin + ch * cos) + 2
    m[0, 2] += nw / 2 - cw / 2
    m[1, 2] += nh / 2 - ch / 2
    card = cv2.warpAffine(card, m, (nw, nh), flags=cv2.INTER_LINEAR)
    mask = cv2.warpAffine(mask, m, (nw, nh), flags=cv2.INTER_LINEAR)
    cx, cy = int(centre[0] * W), int(centre[1] * H)
    left, top = cx - nw // 2, cy - nh // 2
    fx0, fy0, fx1, fy1 = max(0, left), max(0, top), min(W, left + nw), min(H, top + nh)
    if fx1 <= fx0 or fy1 <= fy0:
        return None
    sub = (slice(fy0 - top, fy1 - top), slice(fx0 - left, fx1 - left))
    region = frame[fy0:fy1, fx0:fx1].astype(np.float32)
    shadow = cv2.GaussianBlur(mask, (0, 0), 6)[sub]
    region *= (1 - 0.35 * np.roll(shadow, 5, axis=0))[..., None]          # soft drop shadow
    a = mask[sub][..., None]
    frame[fy0:fy1, fx0:fx1] = np.clip(region * (1 - a) + card[sub] * a, 0, 255).astype(np.uint8)
    return (cx - iw // 2, cy - ih // 2, iw, ih)


def make_scene(cards, size=(720, 1280), text_h=70, seed=0, angles=None):
    """`cards` = [(formula, (x_frac, y_frac)), ...] -> (BGR frame, [text boxes])."""
    frame = desk(size, seed)
    boxes = []
    for i, (text, centre) in enumerate(cards):
        boxes.append(paste_card(frame, text, centre, text_h, (angles or [0] * len(cards))[i], seed + 17 * i))
    return frame, boxes


def save_scene(name: str, cards, **kw) -> str:
    """Write a scene into backend/static/test/ and return its URL path."""
    os.makedirs(STATIC_TEST_DIR, exist_ok=True)
    frame, _ = make_scene(cards, **kw)
    cv2.imwrite(os.path.join(STATIC_TEST_DIR, name), frame)
    return "/static/test/" + name


def save_clip(name: str, frames, fps: int = 15) -> str:
    """Encode BGR frames as a WebM (VP8) clip with ffmpeg; returns its URL path."""
    os.makedirs(STATIC_TEST_DIR, exist_ok=True)
    h, w = frames[0].shape[:2]
    out = os.path.join(STATIC_TEST_DIR, name)
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libvpx", "-b:v", "2M",
         "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    for fr in frames:
        proc.stdin.write(fr.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed to encode the clip")
    return "/static/test/" + name
