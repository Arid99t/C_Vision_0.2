"""Synthetic training data, v2: camera-realistic crops of handwritten formulas.

The v1 generator (synth.py) rendered a formula in one font on a clean white
page. Models trained on it read fonts well but stumble on what a phone camera
actually delivers. v2 changes three things:

  1. HANDWRITING. Half the samples are composed from individual EMNIST
     characters written by real people (glyphs.py), with natural proportions:
     small lowercase letters, lowered subscripts, uneven baseline and spacing,
     one pen width per formula. The other half use ~100 handwriting fonts
     (fontbank.py), drawn character by character with jitter.
  2. CAMERA. Each formula is put through a model of a handheld phone shot:
     tilt, shear, perspective, uneven lighting, shadows, ruled paper, faint
     pencil to bold marker, defocus and motion blur, sensor noise, JPEG, and a
     text size anywhere from 16 to ~100 pixels tall.
  3. CROP. The result is cropped the way the detector crops (ink box plus a
     variable pad) and then run through preprocess.prepare_crop — the very
     function inference uses — so training and deployment see the same pixels.

Needs numpy, Pillow and OpenCV; no torch, so data workers stay light.

Preview:  python -m backend.recognition.synthv2      (writes synth_preview/)
"""
from __future__ import annotations

import json
import math
import os
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import preprocess
from .fontbank import FontBank, load_font
from .glyphs import GlyphBank

cv2.setNumThreads(1)   # samples are tiny; worker processes provide the parallelism

# --------------------------------------------------------------------------
# What gets written: label sampling
# --------------------------------------------------------------------------
_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "chemistry", "data")

_EXTRA_KNOWN = """
H2O CO2 CO CH4 NH3 O2 H2 N2 Cl2 HCl NaCl NaOH H2SO4 HNO3 H3PO4 H2CO3 CaCO3 CaO
Ca(OH)2 Fe2O3 Fe3O4 FeO MgO ZnCl2 ZnO C2H5OH CH3OH C6H12O6 C12H22O11 C3H8 C4H10
C2H6 C2H4 C2H2 C6H6 C8H18 C5H12 C6H14 CH3COOH NH4Cl NH4NO3 (NH4)2SO4 Na2SO4
Na2CO3 NaHCO3 KOH KCl KBr KI KNO3 KMnO4 K2Cr2O7 K2SO4 AgNO3 AgCl CuSO4 CuO Cu2O
CuCl2 Cu(NO3)2 BaSO4 BaCl2 Ba(OH)2 Al2O3 AlCl3 Al(OH)3 Al2(SO4)3 SiO2 SO2 SO3
NO NO2 N2O N2O4 N2O5 HF HBr HI H2S H2O2 PCl3 PCl5 P2O5 P4O10 CCl4 CHCl3 CaCl2
CaSO4 Ca3(PO4)2 MgCl2 MgSO4 Mg(OH)2 FeCl3 FeCl2 FeSO4 Fe(OH)3 FeS ZnSO4 ZnS
Zn(NO3)2 PbO PbO2 Pb(NO3)2 PbCl2 MnO2 Na2O K2O Li2O LiOH LiCl NaNO3 NaClO KClO3
HClO HClO4 HCN NaF CaF2 SnCl2 HgO TiO2 CS2 O3 S8 P4 NaBr NaI Na2S K2CO3 MgCO3
ZnCO3 CuCO3 NH4OH CH3Cl C3H6 C4H8 C2H5Cl C6H5OH C7H8 C10H8 C3H7OH HCOOH CH2O
C3H6O C2H4O2 C6H8O6
""".split()

_SHORT = """
H2 O2 N2 Cl2 F2 Br2 I2 Na K Ca Mg Fe Zn Al Cu Ag Au Pb Sn Hg Li Ba Si S P C Ni
Co Mn Cr Ti He Ne Ar HF HI NO CO KI O3 H N O B F I
""".split()

_CORE = ["H", "C", "N", "O", "S", "Cl", "Na", "K", "Ca", "Fe", "P", "Mg"]
_COMMON_EL = _CORE + ["Zn", "Al", "Cu", "Br", "I", "F", "Si", "Mn", "Ag", "Ba", "Pb",
                      "Li", "Cr", "Ni", "Hg", "Sn", "Co", "B", "Ti", "Au", "He", "Ne", "Ar"]
_ALL_ELEMENTS = """
H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn
Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La
Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po
At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg
Cn Nh Fl Mc Lv Ts Og
""".split()


def _load_known() -> list:
    known = set(_EXTRA_KNOWN)
    try:
        with open(os.path.join(_DATA, "compounds.json"), encoding="utf-8") as f:
            known.update(json.load(f).keys())
        with open(os.path.join(_DATA, "reactions.json"), encoding="utf-8") as f:
            for entry in json.load(f).get("reactions", []):
                known.update(entry.get("reactants", []))
                known.update(entry.get("products", []))
    except FileNotFoundError:
        pass
    return sorted(known)


KNOWN = _load_known()


def _count(rng) -> str:
    r = rng.random()
    if r < 0.44:
        return ""
    if r < 0.68:
        return "2"
    if r < 0.80:
        return "3"
    if r < 0.88:
        return "4"
    if r < 0.96:
        return str(rng.randint(5, 9))
    return str(rng.randint(10, 22))


def _element(rng) -> str:
    r = rng.random()
    if r < 0.45:
        return rng.choice(_CORE)
    if r < 0.88:
        return rng.choice(_COMMON_EL)
    return rng.choice(_ALL_ELEMENTS)


def random_formula(rng) -> str:
    """A grammatically valid formula that need not exist: forces the model to
    read characters instead of memorising a list of known compounds."""
    n = rng.choice((1, 2, 2, 2, 3, 3, 4))
    parts, grouped = [], False
    for _ in range(n):
        if n > 1 and not grouped and rng.random() < 0.12:
            inner = _element(rng) + _count(rng)
            if rng.random() < 0.8:
                inner += _element(rng) + _count(rng)
            parts.append(f"({inner}){rng.choice('234')}")
            grouped = True
        else:
            parts.append(_element(rng) + _count(rng))
    return "".join(parts)


def _single(rng) -> str:
    r = rng.random()
    if r < 0.45:
        return rng.choice(KNOWN)
    if r < 0.65:
        return rng.choice(_SHORT)
    return random_formula(rng)


def sample_label(rng) -> str:
    """Mix of known compounds, short symbols, random valid formulas and 'A + B'."""
    r = rng.random()
    if r < 0.32:
        return rng.choice(KNOWN)
    if r < 0.46:
        return rng.choice(_SHORT)
    if r < 0.86:
        return random_formula(rng)
    line = f"{_single(rng)} + {_single(rng)}"
    return line if len(line) <= 20 else rng.choice(KNOWN)


def tokenize(text: str) -> list:
    """[(char, role)] with role in upper|lower|sub|paren|plus|space.
    'sub' marks a count (a digit following an element or a closing bracket)."""
    out, anchor, in_count = [], False, False
    for ch in text:
        if ch.isdigit():
            is_count = anchor or in_count
            out.append((ch, "sub" if is_count else "upper"))
            in_count, anchor = is_count, False
            continue
        in_count = False
        if ch.isupper():
            role = "upper"
        elif ch.islower():
            role = "lower"
        elif ch in "()":
            role = "paren"
        elif ch == "+":
            role = "plus"
        else:
            role = "space"
        anchor = ch.isalpha() or ch == ")"
        out.append((ch, role))
    return out


# --------------------------------------------------------------------------
# Rendering: an "ink map" (float32, 1.0 = full ink) tight around the writing
# --------------------------------------------------------------------------
def _tight(a: np.ndarray, thr: float = 0.2):
    ys, xs = np.nonzero(a > thr)
    if ys.size == 0:
        return None
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _ellipse(k: int):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def render_font(text: str, rng, font_path: str):
    """Draw `text` with one font, character by character, with jitter."""
    size = rng.randint(40, 60)
    sub_style = rng.random() < 0.6
    sub_scale = rng.uniform(0.5, 0.72)
    drop = rng.uniform(0.10, 0.36) * size
    kern = rng.uniform(-0.03, 0.10) * size
    img = Image.new("L", (int(len(text) * size * 1.25 + 4 * size), int(size * 2.8)), 0)
    draw = ImageDraw.Draw(img)
    x, base = float(size), size * 1.7
    for ch, role in tokenize(text):
        if ch == " ":
            x += size * rng.uniform(0.2, 0.5)
            continue
        lowered = role == "sub" and sub_style
        px = size * (sub_scale if lowered else 1.0) * rng.uniform(0.94, 1.06)
        font = load_font(font_path, max(10, int(round(px))))
        y = base + (drop if lowered else 0.0) + rng.gauss(0, 0.02 * size)
        draw.text((x, y), ch, fill=255, font=font, anchor="ls")
        x += font.getlength(ch) + kern * rng.uniform(0.5, 1.5)
    ink = _tight(np.asarray(img, np.float32) / 255.0)
    if ink is None:
        return None
    r = rng.random()                       # pen weight: bolder or finer strokes
    if r < 0.30:
        ink = cv2.dilate(ink, _ellipse(3 if rng.random() < 0.7 else 5))
    elif r < 0.45 and size >= 50:
        ink = cv2.erode(ink, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
        if float(ink.max()) < 0.5:
            return None
    return ink


_X_HEIGHT = set("acemnorsuvwxz")
_DESCENDER = set("gjpqy")


def _rotate_small(bm: np.ndarray, deg: float) -> np.ndarray:
    if abs(deg) < 0.5:
        return bm
    h, w = bm.shape
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), deg, 1.0)
    c, s = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(w * c + h * s) + 1, int(w * s + h * c) + 1
    m[0, 2] += nw / 2.0 - w / 2.0
    m[1, 2] += nh / 2.0 - h / 2.0
    return cv2.warpAffine(bm, m, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=0)


def _prep_glyph(g: np.ndarray, target_h: float, pen: float, rng):
    """Scale one EMNIST glyph to `target_h` and give it the formula's pen width."""
    ink = _tight(g.astype(np.float32) / 255.0, 0.25)
    if ink is None:
        return None
    h0, w0 = ink.shape
    th = max(6, int(round(target_h)))
    tw = max(3, int(round(w0 * th / h0 * rng.uniform(0.85, 1.15))))
    bm = cv2.resize(ink, (tw, th), interpolation=cv2.INTER_CUBIC if th >= h0 else cv2.INTER_AREA)
    bm = np.clip(bm, 0.0, 1.0)
    solid = (bm > 0.5).astype(np.uint8)
    # A zero border gives the distance transform an edge to measure from: the
    # glyph is cropped tight, so its strokes touch (or even fill) the bitmap.
    dist = cv2.distanceTransform(cv2.copyMakeBorder(solid, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0),
                                 cv2.DIST_L2, 3)[1:-1, 1:-1]
    vals = dist[solid > 0]
    if vals.size:
        width = min(2.0 * float(np.percentile(vals, 95)), float(min(th, tw)))
        diff = width - pen
        if diff >= 1.0:                    # thinner pen than the glyph was written with
            k = min(int(round(diff)), int(width - 1.2))
            if k >= 1:
                thin = cv2.erode(bm, _ellipse(k + 1))
                if int((thin > 0.5).sum()) >= 0.3 * int(solid.sum()):
                    bm = thin              # keep it only if the strokes survived
        elif diff <= -1.0:                 # bolder pen
            bm = cv2.dilate(bm, _ellipse(int(round(-diff)) + 1))
    return _rotate_small(bm, rng.gauss(0, 4.0))


def _draw_symbol(ch: str, cap: float, pen: float, rng) -> np.ndarray:
    """Hand-drawn '(' ')' '+', which EMNIST lacks, with the formula's pen width."""
    t = max(1, int(round(pen)))
    pad = t + 3
    if ch == "+":
        s = cap * rng.uniform(0.42, 0.66)
        n = int(s + 2 * pad)
        c = n / 2.0
        img = np.zeros((n, n), np.uint8)
        a1, a2 = math.radians(rng.gauss(0, 6)), math.radians(90 + rng.gauss(0, 6))
        for a in (a1, a2):
            r1, r2 = s / 2 * rng.uniform(0.85, 1.05), s / 2 * rng.uniform(0.85, 1.05)
            p1 = (int(round(c - r1 * math.cos(a))), int(round(c - r1 * math.sin(a))))
            p2 = (int(round(c + r2 * math.cos(a))), int(round(c + r2 * math.sin(a))))
            cv2.line(img, p1, p2, 255, t, cv2.LINE_AA)
        return img.astype(np.float32) / 255.0
    h = cap * rng.uniform(1.0, 1.3)
    bulge = h * rng.uniform(0.12, 0.30)
    tilt = h * rng.gauss(0, 0.04)
    img = np.zeros((int(h + 2 * pad), int(bulge + abs(tilt) + 2 * pad) + 2), np.uint8)
    pts = []
    for a in np.linspace(-1.0, 1.0, 24):
        x = bulge * (a * a if ch == "(" else 1.0 - a * a) + tilt * a
        pts.append([pad + x + (abs(tilt) if tilt < 0 else 0.0) * 0 + abs(tilt), pad + (a + 1) / 2 * h])
    pts = np.array(pts, np.float64)
    pts[:, 0] -= pts[:, 0].min() - pad
    cv2.polylines(img, [np.round(pts).astype(np.int32)], False, 255, t, cv2.LINE_AA)
    return img.astype(np.float32) / 255.0


def render_glyphs(text: str, rng, bank: GlyphBank):
    """Compose `text` from individual handwritten characters."""
    cap = float(rng.randint(38, 58))                 # capital-letter height in px
    pen = cap * rng.uniform(0.04, 0.13)              # one pen for the whole formula
    sub_style = rng.random() < 0.65
    gap = rng.uniform(0.0, 0.14) * cap
    items, x = [], 0.0                               # (bitmap, x, y_top), baseline at y=0
    for ch, role in tokenize(text):
        if role == "space":
            x += cap * rng.uniform(0.25, 0.55)
            continue
        jitter = rng.gauss(0, 0.025 * cap)
        if role == "plus":
            bm = _draw_symbol(ch, cap, pen, rng)
            top = -0.45 * cap - bm.shape[0] / 2.0 + jitter
        elif role == "paren":
            bm = _draw_symbol(ch, cap, pen, rng)
            top = -0.45 * cap - bm.shape[0] / 2.0 + jitter
        else:
            bottom = 0.0
            if role == "sub" and sub_style:
                hgt = cap * rng.uniform(0.42, 0.66)
                bottom = cap * rng.uniform(0.08, 0.34)
            elif role == "lower" and ch in _X_HEIGHT:
                hgt = cap * rng.uniform(0.52, 0.72)
            elif role == "lower" and ch in _DESCENDER:
                hgt = cap * rng.uniform(0.75, 0.95)
                bottom = cap * rng.uniform(0.18, 0.32)
            elif role == "lower" and ch == "i":
                hgt = cap * rng.uniform(0.70, 0.90)
            elif role == "lower":
                hgt = cap * rng.uniform(0.85, 1.02)
            else:
                hgt = cap * rng.uniform(0.92, 1.06)
            bm = _prep_glyph(bank.sample(ch, rng), hgt, pen, rng)
            if bm is None:
                return None
            top = bottom - bm.shape[0] + jitter
        items.append((bm, x, top))
        x += bm.shape[1] + gap * rng.uniform(0.3, 1.6) - (0.06 * cap if rng.random() < 0.08 else 0.0)
    if not items:
        return None
    y_min = min(t for _, _, t in items)
    y_max = max(t + b.shape[0] for b, _, t in items)
    x_max = max(xx + b.shape[1] for b, xx, _ in items)
    canvas = np.zeros((int(math.ceil(y_max - y_min)) + 2, int(math.ceil(x_max)) + 2), np.float32)
    for bm, xx, top in items:
        yi, xi = int(round(top - y_min)), max(0, int(round(xx)))
        h, w = bm.shape
        h, w = min(h, canvas.shape[0] - yi), min(w, canvas.shape[1] - xi)
        if h > 0 and w > 0:
            np.maximum(canvas[yi:yi + h, xi:xi + w], bm[:h, :w], out=canvas[yi:yi + h, xi:xi + w])
    return _tight(canvas)


# --------------------------------------------------------------------------
# Camera model: ink map -> grayscale crop as a phone would deliver it
# --------------------------------------------------------------------------
def camera(ink: np.ndarray, rng, nprng, level: float = 1.0):
    """Photograph an ink map. `level` scales how harsh the conditions are
    (1.0 = training distribution, 0 = ideal scan)."""
    h0, w0 = ink.shape
    mg = int(0.35 * h0) + 6
    canvas = cv2.copyMakeBorder(ink, mg, mg, mg, mg, cv2.BORDER_CONSTANT, value=0)
    H, W = canvas.shape

    # 1) pose: tilt, slant, perspective
    if rng.random() < 0.78:
        ang = max(-10.0, min(10.0, rng.gauss(0, 3.5))) * level
    else:
        ang = rng.uniform(-30.0, 30.0) * level           # card held at an angle
    shear = max(-0.4, min(0.4, rng.gauss(0, 0.13))) * level
    rot = np.vstack([cv2.getRotationMatrix2D((W / 2.0, H / 2.0), ang, 1.0), [0, 0, 1]])
    shr = np.array([[1.0, shear, -shear * H / 2.0], [0, 1, 0], [0, 0, 1]])
    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    jit = 0.05 * level
    dst = src + np.float32([[rng.uniform(-jit, jit) * W, rng.uniform(-jit, jit) * H] for _ in range(4)])
    m = cv2.getPerspectiveTransform(src, dst).astype(np.float64) @ rot @ shr
    corners = cv2.perspectiveTransform(src.reshape(1, 4, 2), m).reshape(4, 2)
    lo, hi = corners.min(0), corners.max(0)
    m = np.array([[1, 0, -lo[0] + 2], [0, 1, -lo[1] + 2], [0, 0, 1.0]]) @ m
    size = (int(hi[0] - lo[0]) + 5, int(hi[1] - lo[1]) + 5)
    warped = cv2.warpPerspective(canvas, m, size, flags=cv2.INTER_LINEAR, borderValue=0)

    # 2) distance: how many pixels tall the writing is in the camera frame
    rows = np.nonzero(warped.max(axis=1) > 0.3)[0]
    if rows.size == 0:
        return None
    text_h = float(rows[-1] - rows[0] + 1)
    cam_h = math.exp(rng.uniform(math.log(16.0), math.log(96.0)))
    scale = min(cam_h / max(8.0, text_h), 640.0 / warped.shape[1])
    cam = cv2.resize(warped, None, fx=scale, fy=scale,
                     interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    cam = np.clip(cam, 0.0, 1.0)
    ch, cw = cam.shape
    if ch < 6 or cw < 6:
        return None

    # 3) light: paper tone, uneven illumination, ruled lines, a soft shadow
    xx = np.linspace(-1.0, 1.0, cw, dtype=np.float32)[None, :]
    yy = np.linspace(-1.0, 1.0, ch, dtype=np.float32)[:, None]
    g = 0.22 * level
    paper = rng.uniform(150.0, 252.0) * (1.0 + rng.uniform(-g, g) * xx + rng.uniform(-g, g) * yy
                                        + rng.uniform(-0.15, 0.15) * level * (xx * xx + yy * yy))
    if rng.random() < 0.12 * level:
        period = max(6.0, text_h * scale * rng.uniform(0.7, 1.5))
        dark = rng.uniform(0.05, 0.20)
        half = 0.6 + rng.random()
        line = np.abs(((np.arange(ch) + rng.uniform(0, period)) % period) - period / 2.0) < half
        paper = paper * (1.0 - dark * line[:, None].astype(np.float32))
        if rng.random() < 0.35:
            vline = np.abs(((np.arange(cw) + rng.uniform(0, period)) % period) - period / 2.0) < half
            paper = paper * (1.0 - dark * vline[None, :].astype(np.float32))
    if rng.random() < 0.15 * level:
        a = rng.uniform(0, math.tau)
        edge = (math.cos(a) * xx + math.sin(a) * yy - rng.uniform(-0.6, 0.6)) / rng.uniform(0.05, 0.4)
        paper = paper * (1.0 - rng.uniform(0.10, 0.35) / (1.0 + np.exp(-edge)))
    paper = np.clip(paper, 50.0, 255.0)
    alpha = rng.uniform(0.55, 1.0) if level > 0 else 1.0
    tone = min(rng.uniform(0.0, 95.0), max(0.0, float(paper.min()) - 30.0 / alpha))
    img = paper * (1.0 - alpha * cam) + tone * alpha * cam

    # 4) optics: defocus and hand shake
    sig = abs(rng.gauss(0, 0.55)) * level
    if rng.random() < 0.25 * level:
        sig += rng.uniform(0.4, 1.3)
    sig = min(sig, 0.25 + 0.04 * text_h * scale)     # small writing can't take much blur
    if sig > 0.25:
        img = cv2.GaussianBlur(img, (0, 0), sig)
    if rng.random() < 0.18 * level:
        k = rng.choice([s for s in (3, 5, 7) if s <= max(3.0, 0.18 * text_h * scale)])
        kern = np.zeros((k, k), np.float32)
        a = rng.uniform(0, math.pi)
        c = (k - 1) / 2.0
        d = (k - 1) / 2.0
        cv2.line(kern, (int(round(c - d * math.cos(a))), int(round(c - d * math.sin(a)))),
                 (int(round(c + d * math.cos(a))), int(round(c + d * math.sin(a)))), 1.0, 1)
        img = cv2.filter2D(img, -1, kern / max(1e-6, float(kern.sum())))

    # 5) sensor: noise, in-camera sharpening, JPEG
    noise = rng.uniform(0.0, 7.0) * level
    if noise > 0.05:
        img = img + nprng.normal(0.0, noise, img.shape).astype(np.float32)
    if rng.random() < 0.15 * level:
        img = img + rng.uniform(0.4, 1.2) * (img - cv2.GaussianBlur(img, (0, 0), 1.0))
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.5 * level:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(30, 92)])
        if ok:
            img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)

    # Reject shots nobody could read (and the detector would never box): the
    # strokes must still stand out from the paper after everything above.
    strokes, blank = img[cam > 0.4], img[cam < 0.05]
    if strokes.size < 12 or blank.size < 12:
        return None
    if float(np.median(blank)) - float(np.median(strokes)) < 28.0:
        return None

    # 6) the detector's crop: ink box plus an uneven pad (occasionally cutting in)
    ys, xs = np.nonzero(cam > 0.25)
    if ys.size == 0:
        return None
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    bh = y1 - y0 + 1

    def pad():
        if rng.random() < 0.06:
            return -int(rng.uniform(0.0, 0.04) * bh)
        return max(1, int(rng.uniform(0.03, 0.30) * bh))

    top, bottom, left, right = pad(), pad(), pad(), pad()
    crop = img[max(0, y0 - top):min(ch, y1 + 1 + bottom), max(0, x0 - left):min(cw, x1 + 1 + right)]
    return crop if crop.shape[0] >= 6 and crop.shape[1] >= 6 else None


def augment_photo(gray: np.ndarray, rng, nprng) -> np.ndarray:
    """Lightly "re-photograph" a REAL crop (uint8 grayscale) so a few dozen
    hand-collected samples yield many varied training views: small tilt, slant
    and zoom, a lighting change, a little blur, noise and JPEG."""
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), rng.gauss(0, 3.0), rng.uniform(0.88, 1.12))
    shear = rng.gauss(0, 0.07)
    m[0, 1] += shear
    m[0, 2] -= shear * h / 2.0
    img = cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    img = img * rng.uniform(0.75, 1.1) * (1.0 + rng.uniform(-0.15, 0.15) * xx
                                          + rng.uniform(-0.15, 0.15) * yy)
    if rng.random() < 0.35:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.0))
    img = img + nprng.normal(0.0, rng.uniform(0.0, 5.0), img.shape).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(40, 92)])
        if ok:
            img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return img


# --------------------------------------------------------------------------
# Sample factory
# --------------------------------------------------------------------------
class Synth:
    """Draws (model_image, label) samples. `split='test'` switches to held-out
    fonts and the EMNIST test glyphs, i.e. handwriting the model never saw."""

    def __init__(self, split: str = "train", seed: int = 0, font_ratio: float = 0.5,
                 level: float = 1.0):
        self.rng = random.Random(seed)
        self.nprng = np.random.default_rng(seed)
        self.fonts = FontBank(split)
        self.glyphs = GlyphBank(split)
        self.font_ratio = font_ratio
        self.level = level

    def crop(self, text: str | None = None, mode: str | None = None):
        """One raw camera crop (uint8 grayscale, any size) with its label and mode."""
        for _ in range(20):
            label = text if text is not None else sample_label(self.rng)
            how = mode or ("font" if self.rng.random() < self.font_ratio else "glyph")
            if how == "font":
                ink = render_font(label, self.rng, self.fonts.pick(self.rng))
            else:
                ink = render_glyphs(label, self.rng, self.glyphs)
            if ink is None or ink.shape[0] < 8 or ink.shape[1] < 4:
                continue
            shot = camera(ink, self.rng, self.nprng, self.level)
            if shot is not None:
                return shot, label, how
        raise RuntimeError("could not render a sample")

    def sample(self, text: str | None = None, mode: str | None = None):
        """One training sample: ((32, W) uint8 model image, label, mode)."""
        shot, label, how = self.crop(text, mode)
        return preprocess.prepare_crop(shot), label, how


def contact_sheet(samples, cols: int = 4, cell_w: int = 300) -> Image.Image:
    """Grid of (image, caption) pairs for eyeballing the generator."""
    cell_h = 60
    rows = (len(samples) + cols - 1) // cols
    sheet = Image.new("L", (cols * cell_w, rows * cell_h), 128)
    draw = ImageDraw.Draw(sheet)
    for i, (arr, cap) in enumerate(samples):
        x, y = (i % cols) * cell_w, (i // cols) * cell_h
        im = Image.fromarray(arr)
        if im.height != 32:
            im = im.resize((max(1, int(im.width * 32 / im.height)), 32))
        sheet.paste(im.crop((0, 0, min(im.width, cell_w - 6), 32)), (x + 3, y + 3))
        draw.text((x + 3, y + 40), cap, fill=255)
    return sheet


if __name__ == "__main__":
    os.makedirs("synth_preview", exist_ok=True)
    for split in ("train", "test"):
        gen = Synth(split, seed=1)
        shots = []
        for _ in range(40):
            arr, label, how = gen.sample()
            shots.append((arr, f"{label}  [{how}]"))
        contact_sheet(shots).save(f"synth_preview/v2_{split}.png")
    print("wrote synth_preview/v2_train.png and v2_test.png")
