"""Camera-crop cleanup so live crops look like the training distribution.

The model trains on synthetic images: dark ink on a flat, near-white page.
A webcam crop instead has uneven illumination, low contrast and slanted
writing. This module normalizes those away with three steps that are all
close to identity on already-clean input (so it is safe to apply everywhere,
including on synthetic eval images):

  1. flat-field: estimate the paper brightness per region (block-wise high
     percentile, since ink is a dark minority) and divide it out,
  2. contrast stretch: map the darkest ink toward black,
  3. deskew: remove the dominant slant via image moments (the classic
     shear-correction trick), clamped so a bad estimate can't destroy the crop.

numpy + PIL only — no torch, no cv2 — so it runs wherever inference runs.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def _flat_field(g: np.ndarray) -> np.ndarray:
    """Divide out slowly-varying background brightness (paper + lighting)."""
    h, w = g.shape
    bs = max(8, h // 3)                       # block ~ a third of the crop height
    pad_h, pad_w = (-h) % bs, (-w) % bs
    padded = np.pad(g, ((0, pad_h), (0, pad_w)), mode="edge")
    hh, ww = padded.shape
    blocks = (padded.reshape(hh // bs, bs, ww // bs, bs)
                    .transpose(0, 2, 1, 3)
                    .reshape(hh // bs, ww // bs, bs * bs))
    # 90th percentile per block ~ paper brightness even when ink is present
    bg_small = np.percentile(blocks, 90, axis=2).astype(np.float32)
    bg = Image.fromarray(bg_small).resize((ww, hh), Image.BILINEAR)
    bg = np.asarray(bg, np.float32)[:h, :w]
    return np.clip(g / np.maximum(bg, 1.0) * 255.0, 0, 255)


def _stretch_contrast(g: np.ndarray) -> np.ndarray:
    """Pull the darkest ink toward 0 so faint pencil reads like bold marker."""
    lo = float(np.percentile(g, 2))
    if 255.0 - lo < 40.0:                     # no real ink contrast -> leave alone
        return g
    return np.clip((g - lo) * (255.0 / (255.0 - lo)), 0, 255)


def _deskew(g: np.ndarray) -> np.ndarray:
    """Shear-correct slanted writing using second-order image moments."""
    h, w = g.shape
    ink = 255.0 - g                           # weight = darkness
    total = float(ink.sum())
    if total <= 0:
        return g
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    ybar = float((ink * ys).sum() / total)
    xbar = float((ink * xs).sum() / total)
    mu11 = float((ink * (xs - xbar) * (ys - ybar)).sum() / total)
    mu02 = float((ink * (ys - ybar) ** 2).sum() / total)
    if mu02 < 1e-3:
        return g
    skew = float(np.clip(mu11 / mu02, -0.6, 0.6))
    if abs(skew) < 0.05:                      # already upright
        return g
    pil = Image.fromarray(g.astype(np.uint8), "L")
    # x' = x + skew*(y - ybar) shears the strokes back to vertical
    pil = pil.transform(pil.size, Image.AFFINE, (1, skew, -skew * ybar, 0, 1, 0),
                        resample=Image.BILINEAR, fillcolor=255)
    return np.asarray(pil, np.float32)


def clean_crop(pil_gray: Image.Image) -> Image.Image:
    """Normalize one grayscale formula crop for recognition."""
    g = np.asarray(pil_gray, np.float32)
    if g.ndim != 2 or g.shape[0] < 8 or g.shape[1] < 8:
        return pil_gray
    g = _flat_field(g)
    g = _stretch_contrast(g)
    g = _deskew(g)
    return Image.fromarray(g.astype(np.uint8), "L")


# ==========================================================================
# v2 pipeline — used by the mobile recognizer (model.MobileRec).
#
# `prepare_crop` turns ANY grayscale crop of one formula into the exact
# (32, W) image the model reads. Every step is explicit array arithmetic (no
# PIL / OpenCV resampling), because backend/static/js/preprocess.js mirrors
# this function line by line: the phone must feed the model the same pixels it
# was trained on. If you change anything here, change the JS twin, regenerate
# the training data and retrain. tests/test_preprocess_parity.py checks the two
# implementations against each other.
#
# Steps: bound the size -> flat-field -> contrast stretch -> level a clearly
# tilted line -> crop tight to the ink with a fixed margin -> height 32.
# The tight crop makes the input independent of how generous the detector's
# box was, which the v1 pipeline (trained on fixed synthetic margins) was not.
# ==========================================================================
import math                      # noqa: E402
from functools import lru_cache  # noqa: E402

from .config import IMG_HEIGHT, MAX_WIDTH, MIN_WIDTH  # noqa: E402

WORK_HEIGHT = 64        # taller crops are area-downscaled to this first
INK_THRESHOLD = 200.0   # on the smoothed, cleaned image: paper ~255, strokes well below
MARGIN_FRAC = 0.12      # white margin kept around the ink (fraction of ink height)
ROT_MIN_ELONG = 6.0     # trust the tilt estimate only for clearly elongated ink
ROT_MIN_DEG = 5.0       # below this the model copes on its own
ROT_MAX_DEG = 40.0      # beyond this the estimate is more likely wrong than right


def _round_levels(g: np.ndarray) -> np.ndarray:
    """Round half up to integer gray levels 0..255 (JS: Math.floor(v + 0.5))."""
    return np.clip(np.floor(g + 0.5), 0, 255).astype(np.int64)


def _hist_percentile(levels: np.ndarray, p: float) -> int:
    """Smallest level whose cumulative count reaches ceil(p * n)."""
    hist = np.bincount(levels.ravel(), minlength=256)
    target = max(1, int(math.ceil(p * levels.size)))
    return int(np.searchsorted(np.cumsum(hist), target))


@lru_cache(maxsize=32)   # keep this small: a matrix can be >1 MB and sizes rarely repeat
def _resize_matrix(n_in: int, n_out: int) -> np.ndarray:
    """(n_out, n_in) resampling weights: box filter to shrink, bilinear to grow."""
    if n_out == n_in:
        return np.eye(n_in)
    scale = n_in / n_out
    o = np.arange(n_out, dtype=np.float64)
    m = np.zeros((n_out, n_in))
    if n_out < n_in:
        start, end = o * scale, (o + 1.0) * scale
        i = np.arange(n_in, dtype=np.float64)[None, :]
        m = np.clip(np.minimum(end[:, None], i + 1.0) - np.maximum(start[:, None], i),
                    0.0, None) / scale
    else:
        src = (o + 0.5) * scale - 0.5
        i0 = np.floor(src)
        f = src - i0
        a = np.clip(i0, 0, n_in - 1).astype(np.int64)
        b = np.clip(i0 + 1, 0, n_in - 1).astype(np.int64)
        rows = np.arange(n_out)
        np.add.at(m, (rows, a), 1.0 - f)
        np.add.at(m, (rows, b), f)
    m.setflags(write=False)
    return m


def resize_gray(g: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Separable resize of a float image (see _resize_matrix for the filters)."""
    h, w = g.shape
    return _resize_matrix(h, out_h) @ g @ _resize_matrix(w, out_w).T


def _window_filter(g: np.ndarray, k: int, use_max: bool) -> np.ndarray:
    """Separable k x k sliding max (or min); windows are clipped at the border."""
    r = k // 2
    fill = -np.inf if use_max else np.inf
    fn = np.max if use_max else np.min
    a = np.pad(g, ((0, 0), (r, r)), constant_values=fill)
    a = fn(np.lib.stride_tricks.sliding_window_view(a, k, axis=1), axis=2)
    a = np.pad(a, ((r, r), (0, 0)), constant_values=fill)
    return fn(np.lib.stride_tricks.sliding_window_view(a, k, axis=0), axis=2)


def _box3(g: np.ndarray) -> np.ndarray:
    """3 x 3 mean with replicated edges (tames single-pixel glints and noise)."""
    p = np.pad(g, 1, mode="edge")
    h, w = g.shape
    out = np.zeros_like(g)
    for dy in range(3):
        for dx in range(3):
            out += p[dy:dy + h, dx:dx + w]
    return out / 9.0


def _flat_field2(g: np.ndarray) -> np.ndarray:
    """Divide out the paper brightness.

    The paper is estimated with a grayscale closing: strokes are thin in at
    least one direction, so a window wider than any stroke "paints over" the
    ink with the surrounding paper, while broad shading and hard shadow edges
    survive in the estimate and are divided away.
    """
    k = max(9, int(0.28 * g.shape[0] + 0.5)) | 1
    paper = _window_filter(_window_filter(_box3(g), k, True), k, False)
    return np.clip(g / np.maximum(paper, 1.0) * 255.0, 0.0, 255.0)


def _stretch_contrast2(g: np.ndarray) -> np.ndarray:
    """Map the ink level to 0 and the paper level to 255."""
    levels = _round_levels(g)
    lo, hi = _hist_percentile(levels, 0.02), _hist_percentile(levels, 0.90)
    if hi - lo < 20:                  # no real ink contrast -> leave alone
        return g
    return np.clip((g - lo) * (255.0 / (hi - lo)), 0.0, 255.0)


def _bilinear_sample(g: np.ndarray, xs: np.ndarray, ys: np.ndarray, fill: float) -> np.ndarray:
    h, w = g.shape
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    fx, fy = xs - x0, ys - y0

    def at(yy, xx):
        ok = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
        v = np.full(xx.shape, fill)
        v[ok] = g[yy[ok], xx[ok]]
        return v

    return (at(y0, x0) * (1 - fx) * (1 - fy) + at(y0, x0 + 1) * fx * (1 - fy)
            + at(y0 + 1, x0) * (1 - fx) * fy + at(y0 + 1, x0 + 1) * fx * fy)


def estimate_tilt(mask: np.ndarray):
    """(theta_radians, elongation) of the ink's principal axis, or None."""
    ys, xs = np.nonzero(mask)
    if xs.size < 30:
        return None
    xbar, ybar = xs.mean(), ys.mean()
    dx, dy = xs - xbar, ys - ybar
    mu20, mu02, mu11 = (dx * dx).mean(), (dy * dy).mean(), (dx * dy).mean()
    common = math.sqrt(4.0 * mu11 * mu11 + (mu20 - mu02) ** 2)
    l1, l2 = (mu20 + mu02 + common) / 2.0, (mu20 + mu02 - common) / 2.0
    if l2 <= 1e-6:
        return None
    return 0.5 * math.atan2(2.0 * mu11, mu20 - mu02), l1 / l2


def _level(g: np.ndarray, theta: float) -> np.ndarray:
    """Rotate so an ink axis tilted by `theta` becomes horizontal (white fill)."""
    h, w = g.shape
    c, s = math.cos(theta), math.sin(theta)
    w2 = int(math.ceil(w * abs(c) + h * abs(s)))
    h2 = int(math.ceil(w * abs(s) + h * abs(c)))
    cx, cy, cx2, cy2 = (w - 1) / 2.0, (h - 1) / 2.0, (w2 - 1) / 2.0, (h2 - 1) / 2.0
    X, Y = np.meshgrid(np.arange(w2) - cx2, np.arange(h2) - cy2)
    return _bilinear_sample(g, c * X - s * Y + cx, s * X + c * Y + cy, 255.0)


def _ink_bbox(mask: np.ndarray):
    """(x0, y0, x1, y1) inclusive box of the writing.

    A row/column counts as writing when it holds at least two dark pixels but
    is not dark almost end to end: that skips lone noise pixels and the ruled
    or grid lines of notebook paper, which would otherwise stretch the box.
    """
    h, w = mask.shape
    r, c = mask.sum(1), mask.sum(0)
    rows = np.nonzero((r >= 2) & (r <= 0.85 * w))[0]
    cols = np.nonzero((c >= 2) & (c <= 0.85 * h))[0]
    if rows.size == 0 or cols.size == 0:
        return None
    return int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1])


def prepare_crop(gray: np.ndarray) -> np.ndarray:
    """Any (h, w) grayscale crop of ONE formula -> (32, W) uint8 model image.

    Dark ink on lighter paper is assumed (same assumption as the detector).
    """
    g = np.asarray(gray, np.float64)
    if g.ndim != 2:
        raise ValueError("prepare_crop expects a 2-D grayscale array")
    h, w = g.shape
    if h > WORK_HEIGHT:
        nw = max(1, int(w * WORK_HEIGHT / h + 0.5))
        g = resize_gray(g, WORK_HEIGHT, nw)
        h, w = g.shape

    if h >= 8 and w >= 8:
        g = _stretch_contrast2(_flat_field2(g))
        # Geometry is measured on a 3x3-smoothed copy: sensor noise amplified by
        # the contrast stretch would otherwise pass for ink and inflate the box.
        tilt = estimate_tilt(_box3(g) < INK_THRESHOLD)
        if tilt is not None:
            theta, elong = tilt
            deg = abs(math.degrees(theta))
            if elong >= ROT_MIN_ELONG and ROT_MIN_DEG <= deg <= ROT_MAX_DEG:
                g = _level(g, theta)
                h, w = g.shape
        box = _ink_bbox(_box3(g) < INK_THRESHOLD)
        if box is not None:
            x0, y0, x1, y1 = box
            m = max(2, int(MARGIN_FRAC * (y1 - y0 + 1) + 0.5))
            ch, cw = (y1 - y0 + 1) + 2 * m, (x1 - x0 + 1) + 2 * m
            canvas = np.full((ch, cw), 255.0)
            sy0, sy1 = max(0, y0 - m), min(h, y1 + m + 1)
            sx0, sx1 = max(0, x0 - m), min(w, x1 + m + 1)
            dy, dx = sy0 - (y0 - m), sx0 - (x0 - m)
            canvas[dy:dy + (sy1 - sy0), dx:dx + (sx1 - sx0)] = g[sy0:sy1, sx0:sx1]
            g = canvas
            h, w = g.shape

    out_w = min(MAX_WIDTH, max(MIN_WIDTH, int(w * IMG_HEIGHT / h + 0.5)))
    out = resize_gray(g, IMG_HEIGHT, out_w)
    return _round_levels(out).astype(np.uint8)


def to_model_input(arr_uint8: np.ndarray) -> np.ndarray:
    """(32, W) uint8 -> float32 in [-1, 1]; the normalization the models train with."""
    return ((arr_uint8.astype(np.float32) / 255.0) - 0.5) / 0.5
