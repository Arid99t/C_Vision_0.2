"""Training pictures for the resistor colour-band reader.

Two sources, because drawings alone do not transfer to real parts (the first
model scored 94 % on drawings and 0 of 8 on real photos):

  drawn      an axial resistor carrying a known colour code, built to look like
             the real thing: dog-bone body with bands sitting on the end caps,
             paint colours spread over what real makers use (measured from
             photos), sharp highlight streaks along the glossy body, glittery
             gold and silver, thick tinned leads; placed over a real photo, a
             workbench or white paper, then pushed through a camera model.
  real       a few labelled photos (tools/fetch_resistor_photos.py), cut out of
             their white background and re-photographed many ways: other
             backgrounds, sizes, tilts, lighting, blur.

The label is the band colours left to right AS SEEN, because the reader cannot
know which way round the part is held; bands.decode() sorts that out.

    python -m backend.resistor.render --bank              background photo bank (once)
    python -m backend.resistor.render --make 120000       write the training set to <work>/resistor/data
    python -m backend.resistor.render --preview p.png     a sheet of samples to look at
"""
from __future__ import annotations

import argparse
import glob
import os
import zipfile

import cv2
import numpy as np

from ..recognition.config import WORK_DIR
from .bands import COLOURS, E24, E24_3, E96

H, W = 48, 160            # model input (height, width)
SS = 2                    # supersampling while drawing
MAX_BANDS = 5
DATA_DIR = os.path.join(WORK_DIR, "resistor", "data")
REAL_DIR = os.path.join(WORK_DIR, "resistor", "real")
BANK_PATH = os.path.join(WORK_DIR, "resistor", "bg_bank.npy")
COCO_ZIP = os.path.join(WORK_DIR, "coco", "val2017.zip")

# Paint colours (sRGB). Each list spans what different makers and lights give
# for that colour, measured from photos; a sample blends two of them and wanders.
PAINT = {
    "black": [(10, 8, 10), (40, 36, 36), (55, 50, 48), (28, 26, 30)],
    "brown": [(112, 62, 34), (127, 30, 11), (120, 56, 38), (90, 48, 46), (150, 95, 85), (100, 50, 50), (80, 44, 40)],
    "red": [(196, 36, 34), (200, 30, 40), (192, 85, 95), (170, 40, 60), (215, 60, 60), (160, 30, 45)],
    "orange": [(236, 116, 26), (211, 105, 66), (235, 90, 60), (240, 130, 60), (228, 100, 80)],
    "yellow": [(238, 208, 38), (224, 191, 17), (235, 204, 92), (220, 180, 40), (210, 170, 20)],
    "green": [(40, 132, 62), (44, 93, 54), (46, 162, 44), (60, 120, 80), (30, 100, 60)],
    "blue": [(34, 70, 178), (40, 110, 170), (60, 120, 180), (20, 60, 150), (50, 100, 190)],
    "violet": [(126, 58, 158), (72, 84, 179), (175, 132, 172), (110, 90, 130), (120, 80, 150), (95, 80, 120)],
    "grey": [(128, 128, 128), (122, 104, 97), (105, 105, 115), (150, 150, 150), (95, 95, 100)],
    "white": [(234, 234, 230), (219, 215, 207), (245, 245, 240)],
    "gold": [(186, 146, 64), (135, 109, 72), (190, 170, 140), (201, 177, 75), (175, 140, 95), (160, 125, 80), (190, 150, 110)],
    "silver": [(184, 187, 192), (160, 160, 165), (200, 200, 205), (170, 172, 178)],
}
BODIES = [  # (weight, colour, wander)
    (0.20, (192, 152, 128), 20),   # carbon film: tan
    (0.20, (214, 196, 168), 18),   #   cream
    (0.10, (214, 182, 130), 20),   #   beige
    (0.10, (10, 95, 225), 22),     # metal film: vivid blue
    (0.12, (96, 150, 200), 24),    #   blue
    (0.10, (105, 168, 207), 20),   #   light blue
    (0.05, (64, 112, 172), 20),    #   darker blue
    (0.04, (84, 168, 180), 20),    #   teal
    (0.03, (150, 190, 142), 20),   # light green
    (0.03, (220, 164, 142), 18),   # salmon
    (0.03, (168, 170, 160), 18),   # grey
]
DESK = [(236, 232, 224), (200, 170, 130), (120, 86, 58), (60, 62, 66), (30, 110, 70), (44, 78, 130),
        (250, 250, 250), (20, 20, 22), (190, 196, 204), (224, 190, 160), (160, 120, 96)]


# ------------------------------------------------------------- backgrounds ---
def build_bank(count: int = 900, size: tuple[int, int] = (144, 256), seed: int = 7) -> str:
    """Cut a compact bank of real-photo backgrounds out of the COCO val images."""
    if not os.path.exists(COCO_ZIP):
        raise FileNotFoundError(f"{COCO_ZIP} not found; training will use drawn backgrounds only")
    rng = np.random.default_rng(seed)
    with zipfile.ZipFile(COCO_ZIP) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".jpg")]
        picks = rng.choice(len(names), size=min(count, len(names)), replace=False)
        bank = np.zeros((len(picks), size[0], size[1], 3), np.uint8)
        for i, k in enumerate(picks):
            img = cv2.imdecode(np.frombuffer(z.read(names[k]), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            bank[i] = cv2.cvtColor(cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_AREA),
                                   cv2.COLOR_BGR2RGB)
    os.makedirs(os.path.dirname(BANK_PATH), exist_ok=True)
    np.save(BANK_PATH, bank)
    return BANK_PATH


def load_bank():
    return np.load(BANK_PATH, mmap_mode="r") if os.path.exists(BANK_PATH) else None


def _jitter(rng, base, wander):
    gain = rng.uniform(0.88, 1.12)
    c = np.asarray(base, np.float32) * gain + rng.uniform(-wander, wander, 3) * 0.6
    return np.clip(c, 0, 255) / 255.0


def _paint(rng, name):
    anchors = PAINT[name]
    a, b = anchors[rng.integers(len(anchors))], anchors[rng.integers(len(anchors))]
    t = rng.random()
    return _jitter(rng, np.asarray(a, np.float32) * (1 - t) + np.asarray(b, np.float32) * t, 12)


def _background(rng, bank, h, w):
    kind = rng.random()
    if kind < 0.24:                                       # white paper / photo table
        bg = np.ones((h, w, 3), np.float32) * _jitter(rng, (246, 246, 244), 8)
        if rng.random() < 0.5:
            t = np.linspace(0, 1, w, dtype=np.float32)[None, :, None]
            bg = bg * (1 - rng.uniform(0, 0.12) * t)
    elif bank is not None and kind < 0.66:
        src = bank[rng.integers(len(bank))]
        sh, sw = src.shape[:2]
        cw = int(rng.uniform(0.25, 1.0) * sw)
        ch = max(8, min(sh, int(cw * h / w)))
        x0, y0 = rng.integers(0, sw - cw + 1), rng.integers(0, sh - ch + 1)
        bg = cv2.resize(np.ascontiguousarray(src[y0:y0 + ch, x0:x0 + cw]), (w, h),
                        interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    elif kind < 0.78:                                     # breadboard
        bg = np.ones((h, w, 3), np.float32) * _jitter(rng, (238, 234, 222), 14)
        pitch = rng.uniform(13, 24)
        ox, oy = rng.uniform(0, pitch, 2)
        hole = _jitter(rng, (52, 50, 48), 18)
        r = max(1, int(pitch * rng.uniform(0.16, 0.24)))
        for y in np.arange(oy, h, pitch):
            for x in np.arange(ox, w, pitch):
                cv2.rectangle(bg, (int(x - r), int(y - r)), (int(x + r), int(y + r)), hole.tolist(), -1)
    else:                                                 # desk: two-tone gradient + mottling
        a = _jitter(rng, DESK[rng.integers(len(DESK))], 20)
        b = _jitter(rng, DESK[rng.integers(len(DESK))], 20) if rng.random() < 0.4 else a * rng.uniform(0.7, 1.1)
        t = np.linspace(0, 1, w, dtype=np.float32)[None, :, None] if rng.random() < 0.5 \
            else np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
        bg = a * (1 - t) + np.clip(b, 0, 1) * t + np.zeros((h, w, 3), np.float32)
        low = cv2.resize(rng.normal(0, 0.05, (h // 8 + 1, w // 8 + 1, 1)).astype(np.float32), (w, h))
        bg = bg + low[..., None]
    sigma = rng.uniform(0, 3.2)                           # the background is usually out of focus
    if sigma > 0.4:
        bg = cv2.GaussianBlur(bg, (0, 0), sigma)
    return np.clip(bg, 0, 1)


def _distractors(rng, img):
    """Things that are not resistors: wires, blobs, bars."""
    h, w = img.shape[:2]
    for _ in range(rng.integers(0, 4)):
        colour = rng.uniform(0, 1, 3).tolist()
        kind = rng.integers(3)
        p = (int(rng.uniform(0, w)), int(rng.uniform(0, h)))
        q = (int(rng.uniform(0, w)), int(rng.uniform(0, h)))
        if kind == 0:
            cv2.line(img, p, q, colour, int(rng.uniform(2, 9)), cv2.LINE_AA)
        elif kind == 1:
            cv2.ellipse(img, p, (int(rng.uniform(6, 40)), int(rng.uniform(6, 30))), rng.uniform(0, 180),
                        0, 360, colour, -1, cv2.LINE_AA)
        else:
            cv2.rectangle(img, p, (p[0] + int(rng.uniform(8, 80)), p[1] + int(rng.uniform(6, 40))), colour, -1)


# ---------------------------------------------------------------- the part ---
def _norm(weights):
    a = np.asarray(weights, np.float64)
    return a / a.sum()


def sample_bands(rng) -> list[str]:
    """A colour code. Mostly real catalogue values (so the common ones are seen
    often), with a share of arbitrary codes so the reader does not simply
    memorise the catalogue."""
    n = int(rng.choice([4, 5, 3], p=[0.56, 0.38, 0.06]))
    mults = COLOURS[:10] + ["gold", "silver"]
    if rng.random() < 0.7:
        if n == 5:
            sig = int(rng.choice(E96 if rng.random() < 0.7 else E24_3))
            digits = [COLOURS[int(d)] for d in f"{sig:03d}"]
            mult = str(rng.choice(mults, p=_norm([10, 14, 14, 12, 8, 4, 2, 0.5, 0.2, 0.1, 6, 3])))
            tol = str(rng.choice(["brown", "red", "green", "blue", "violet", "gold"],
                                 p=[0.68, 0.12, 0.07, 0.05, 0.03, 0.05]))
        else:
            sig = int(rng.choice(E24))
            digits = [COLOURS[int(d)] for d in f"{sig:02d}"]
            mult = str(rng.choice(mults, p=_norm([10, 14, 16, 14, 12, 8, 4, 1, 0.3, 0.2, 6, 3])))
            tol = str(rng.choice(["gold", "silver", "brown", "red"], p=[0.74, 0.1, 0.09, 0.07]))
    else:
        k = 3 if n == 5 else 2
        digits = [COLOURS[int(rng.integers(1, 10))]] + [COLOURS[int(rng.integers(0, 10))] for _ in range(k - 1)]
        mult = str(rng.choice(mults))
        tol = str(rng.choice(["brown", "red", "green", "blue", "violet", "grey", "gold", "silver"]))
    return digits + [mult] + ([tol] if n >= 4 else [])


def _band_layout(rng, n):
    """Start and width of each band as fractions of the body length. Carbon-film
    parts carry wide bands, the first often on the end cap; precision parts
    carry thin ones."""
    thin = n == 5 or rng.random() < 0.25
    if thin:
        u, bw, gap = rng.uniform(0.05, 0.16), rng.uniform(0.035, 0.075), rng.uniform(0.04, 0.085)
    else:
        u, bw, gap = rng.uniform(0.04, 0.2), rng.uniform(0.07, 0.125), rng.uniform(0.045, 0.1)
    lead = n - 1 if n >= 4 else n
    budget = 0.80 if n >= 4 else 0.94                       # room the leading bands may use
    need = lead * bw + (lead - 1) * gap
    if u + need > budget:                                   # squeeze to fit
        f = (budget - u) / need
        bw, gap = bw * f, gap * f
    spans = []
    for _ in range(lead):
        w = bw * rng.uniform(0.85, 1.15)
        spans.append((u, w))
        u += w + gap * rng.uniform(0.85, 1.2)
    if n >= 4:
        w = bw * rng.uniform(0.9, 1.35)
        if rng.random() < 0.25:                             # evenly spaced: direction is hard to tell
            start = u
        else:                                               # tolerance band set apart, on the far cap
            start = max(u + gap * 0.4, 1 - rng.uniform(0.05, 0.22) - w)
        spans.append((min(start, 0.985 - w), w))
    return spans


def _draw_part(rng, bands, length, dia):
    """The resistor lying along +x, RGBA float, with leads. Returns (rgba, centre)."""
    lead = int(length * rng.uniform(0.5, 1.6))
    pad = 6
    L = int(length)
    ww, hh = L + 2 * lead + 2 * pad, int(dia) + 2 * pad
    cy = hh / 2.0
    x0 = pad + lead
    ys = np.arange(hh, dtype=np.float32)[:, None]

    # body profile: radius along the length. Dog-bone: fat end caps, a waist between.
    xs = np.arange(L, dtype=np.float32)
    u = (xs + 0.5) / length
    profile = np.ones_like(u)
    if rng.random() < 0.7:
        waist = rng.uniform(0.68, 0.94)
        cap, soft = rng.uniform(0.15, 0.28), rng.uniform(0.04, 0.11)
        edge = np.minimum(u, 1 - u)
        t = np.clip((cap + soft - edge) / soft, 0, 1)
        profile = waist + (1 - waist) * (0.5 - 0.5 * np.cos(np.pi * t))
    end = np.minimum(xs + 0.5, length - xs - 0.5) / (dia * rng.uniform(0.25, 0.55))
    profile = profile * np.sqrt(np.clip(1 - np.clip(1 - end, 0, 1) ** 2, 0, 1))   # rounded ends
    radius = profile * dia / 2

    # leads: tinned wire, lighter along the middle; a cone of coating runs a little way up each
    waist_r = float(np.median(radius))
    lw = max(1.5, 2 * waist_r * rng.uniform(0.15, 0.34))
    wire = _jitter(rng, (186, 188, 190), 24)
    v = (ys - cy) / (lw / 2)
    rgb = np.zeros((hh, ww, 3), np.float32) + wire * (0.5 + 0.5 * np.sqrt(np.clip(1 - v * v, 0, 1)))[..., None]
    alpha = np.zeros((hh, ww), np.float32) + np.clip(1.2 - np.abs(v), 0, 1)

    weights = np.array([b[0] for b in BODIES])
    _, base, wander = BODIES[int(rng.choice(len(BODIES), p=weights / weights.sum()))]
    body_colour = _jitter(rng, base, wander)
    column = np.tile(body_colour, (L, 1))
    glitter = np.zeros(L, np.float32)
    matte = np.ones(L, np.float32)
    for name, (start, width) in zip(bands, _band_layout(rng, len(bands))):
        a, b = max(0, int(start * length)), min(L, int(np.ceil((start + width) * length)))
        column[a:b] = _paint(rng, name)
        if name in ("gold", "silver"):
            glitter[a:b] = rng.uniform(0.12, 0.4)
        matte[a:b] = rng.uniform(0.55, 1.0)                # paint is often less glossy than the lacquer
    if rng.random() < 0.6:                                 # soften band edges a little
        k = int(rng.integers(1, 3)) * 2 + 1
        column = cv2.blur(column[None], (k, 1))[0]

    # cylinder shading: darker towards the rim, soft sheen, one or two sharp highlight streaks
    vv = (ys - cy) / np.maximum(radius[None, :], 1e-3)               # (hh, L)
    inside = np.clip((1 - np.abs(vv)) * np.maximum(radius[None, :], 1.0), 0, 1)
    floor = rng.uniform(0.42, 0.68)
    lambert = floor + (1 - floor) * np.sqrt(np.clip(1 - vv * vv, 0, 1))
    flip = -1.0 if rng.random() < 0.2 else 1.0                        # light from below sometimes
    sheen = np.exp(-0.5 * ((vv - flip * rng.uniform(-0.62, -0.15)) / rng.uniform(0.12, 0.3)) ** 2) * rng.uniform(0.0, 0.5)
    for _ in range(int(rng.choice([0, 1, 2], p=[0.3, 0.35, 0.35]))):
        pos, thick, strength = flip * rng.uniform(-0.72, 0.5), rng.uniform(0.03, 0.1), rng.uniform(0.35, 1.0)
        env = cv2.resize(rng.uniform(0, 1, (1, max(2, L // int(rng.integers(8, 40)) + 1))).astype(np.float32), (L, 1))[0]
        env = np.clip((env - rng.uniform(0.1, 0.5)) * 3, 0, 1)        # broken into blobs along the body
        sheen = np.maximum(sheen, np.exp(-0.5 * ((vv - pos) / thick) ** 2) * strength * env[None, :])
    sheen = sheen * matte[None, :]
    texture = column[None, :, :] * np.ones((hh, 1, 1), np.float32)
    if glitter.any():                                                 # metallic flake: dark and bright specks
        speck = rng.normal(0, 1, (hh, L)).astype(np.float32) * glitter[None, :]
        texture = np.clip(texture * (1 + speck[..., None]), 0, 1)
        sheen = np.maximum(sheen, (rng.random((hh, L)) < 0.06 * (glitter[None, :] > 0)).astype(np.float32) * 0.8)
    if rng.random() < 0.7:                                            # bands bow round the cylinder / wobble
        bow = rng.uniform(-0.1, 0.1) * dia
        wob = rng.normal(0, rng.uniform(0, 0.8), hh).cumsum() * 0.3
        for r in range(hh):
            shift = int(round(bow * (vv[r, L // 2] ** 2) + wob[r]))
            if shift:
                texture[r] = np.roll(texture[r], shift, axis=0)
    body = texture * lambert[..., None]
    body = body + (1 - body) * sheen[..., None]
    sl = slice(x0, x0 + L)
    # coating that creeps up the leads
    for side in (0, 1):
        n_cone = int(length * rng.uniform(0.0, 0.1))
        for i in range(n_cone):
            x = x0 - 1 - i if side == 0 else x0 + L + i
            if 0 <= x < ww:
                r_c = lw / 2 + (waist_r * 0.55 - lw / 2) * (1 - i / max(1, n_cone)) ** 2
                m = np.clip(r_c - np.abs(ys[:, 0] - cy) + 0.5, 0, 1)
                rgb[:, x] = rgb[:, x] * (1 - m[:, None]) + (body_colour * 0.9) * m[:, None]
                alpha[:, x] = np.maximum(alpha[:, x], m)
    rgb[:, sl] = rgb[:, sl] * (1 - inside[..., None]) + body * inside[..., None]
    alpha[:, sl] = np.maximum(alpha[:, sl], inside)
    return np.dstack([np.clip(rgb, 0, 1), alpha]), (x0 + length / 2.0, cy)


def _place(rng, img, part, centre, length, dia, max_tilt=15.0):
    """Rotate and paste a part (RGBA) into the picture, with a soft shadow."""
    h, w = img.shape[:2]
    pcx, pcy = centre
    angle = rng.uniform(-max_tilt, max_tilt)
    margin = length / 2 * np.cos(np.radians(angle)) + 3
    cx = rng.uniform(margin, w - margin) if w > 2 * margin else w / 2
    tilt = abs(np.sin(np.radians(angle))) * length / 2 + dia / 2
    cy = rng.uniform(min(tilt, h / 2), max(h - tilt, h / 2))
    m = cv2.getRotationMatrix2D((pcx, pcy), angle, 1.0)
    m[0, 2] += cx - pcx
    m[1, 2] += cy - pcy
    warped = cv2.warpAffine(part, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    a = warped[..., 3:4]
    if rng.random() < 0.55:                               # soft contact shadow
        off = rng.uniform(2, 9)
        sh = cv2.GaussianBlur(np.roll(a[..., 0], int(off), axis=0), (0, 0), rng.uniform(2, 6))
        img = img * (1 - rng.uniform(0.15, 0.45) * sh[..., None])
    return img * (1 - a) + warped[..., :3] * a


def render_sample(rng, bank=None, bands=None):
    """One drawn training picture. Returns (uint8 RGB image H x W x 3, [class indices]).
    Pass `bands` (colour names) to draw that particular resistor, the right way round."""
    h, w = H * SS, W * SS
    img = _background(rng, bank, h, w)
    labels: list[int] = []
    fixed = bands is not None
    if not fixed and rng.random() < 0.15:
        _distractors(rng, img)
    if fixed or rng.random() >= 0.12:                     # 12 % of pictures hold no resistor
        bands = list(bands) if fixed else sample_bands(rng)
        length = rng.uniform(0.34, 0.9) * w
        dia = min(length * rng.uniform(0.26, 0.44), h * 0.86)
        part, centre = _draw_part(rng, bands, length, dia)
        img = _place(rng, img, part, centre, length, dia)
        labels = [1 + COLOURS.index(b) for b in bands]
        if not fixed and rng.random() < 0.5:              # held the other way round
            img = img[:, ::-1]
            labels = labels[::-1]
        if rng.random() < 0.5:
            img = img[::-1]
    elif rng.random() < 0.7:
        _distractors(rng, img)
    return _camera(rng, np.ascontiguousarray(img)), labels


def _camera(rng, img, strength=1.0):
    """What a small camera does to the scene."""
    h, w = img.shape[:2]
    ramp = np.linspace(rng.uniform(0.72, 1.0), rng.uniform(1.0, 1.28), w, dtype=np.float32)[None, :, None]
    if rng.random() < 0.5:
        ramp = ramp[:, ::-1]
    img = img * (1 + (ramp - 1) * strength)
    img = img * (1 + (rng.uniform(0.8, 1.2, 3).astype(np.float32) - 1) * strength)        # white-balance error
    gain = rng.uniform(0.75, 1.25) if rng.random() < 0.85 else rng.uniform(1.2, 1.6)      # sometimes overexposed
    img = np.clip(img * (1 + (gain - 1) * strength), 0, 1) ** (1 + (rng.uniform(0.72, 1.4) - 1) * strength)
    grey = img.mean(axis=2, keepdims=True)
    img = grey + (img - grey) * (1 + (rng.uniform(0.65, 1.25) - 1) * strength)            # saturation
    sigma = rng.uniform(0, 2.0)
    if sigma > 0.3:
        img = cv2.GaussianBlur(img, (0, 0), sigma)
    if rng.random() < 0.15:                                            # motion blur
        k = int(rng.integers(2, 5)) * 2 + 1
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2, :] = 1.0 / k
        if rng.random() < 0.4:
            kernel = kernel.T
        img = cv2.filter2D(img, -1, kernel)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.25:                                            # a lower-resolution sensor
        f = rng.uniform(0.5, 0.85)
        img = cv2.resize(cv2.resize(img, (int(W * f), int(H * f)), interpolation=cv2.INTER_AREA), (W, H),
                         interpolation=cv2.INTER_LINEAR)
    img = img + rng.normal(0, rng.uniform(0, 0.035), img.shape).astype(np.float32)
    out = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    if rng.random() < 0.3:
        ok, enc = cv2.imencode(".jpg", out[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(35, 85))])
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1]
    return np.ascontiguousarray(out)


# ------------------------------------------------------------- real photos ---
def load_real(split: str = "train") -> list[tuple[np.ndarray, list[int]]]:
    """Labelled real crops as (RGBA float picture, [class indices]). The white
    photo-table background becomes transparent, so the part can be put anywhere."""
    out = []
    for path in sorted(glob.glob(os.path.join(REAL_DIR, split, "*.jpg"))):
        names = [b for b in os.path.basename(path).split("_")[0].split("-") if b in COLOURS]
        bgr = cv2.imread(path)
        if bgr is None or not 3 <= len(names) <= MAX_BANDS:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # background = near-white pixels connected to the border
        white = (rgb.min(axis=2) > 222).astype(np.uint8)
        mask = np.zeros((rgb.shape[0] + 2, rgb.shape[1] + 2), np.uint8)
        flood = white.copy()
        for x in range(0, rgb.shape[1], 4):
            for y in (0, rgb.shape[0] - 1):
                if flood[y, x] == 1:
                    cv2.floodFill(flood, mask, (x, y), 2)
        keep = cv2.erode((flood != 2).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2)   # drop the pale fringe
        alpha = cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 1.0)
        out.append((np.dstack([rgb.astype(np.float32) / 255.0, alpha]), [1 + COLOURS.index(b) for b in names]))
    return out


def augment_real(rng, real, bank=None):
    """One training picture from a real photo: new background, size, tilt, side
    and lighting. Returns (uint8 RGB H x W x 3, [class indices])."""
    part, labels = real[rng.integers(len(real))]
    labels = list(labels)
    h, w = H * SS, W * SS
    img = _background(rng, bank, h, w)
    scale = rng.uniform(0.45, 1.25) * (w / part.shape[1])               # the crops hold the body at 62 % of the width
    scale_y = scale * rng.uniform(0.9, 1.1)
    pw, ph = max(8, int(part.shape[1] * scale)), max(4, int(part.shape[0] * scale_y))
    small = cv2.resize(part, (pw, ph), interpolation=cv2.INTER_AREA)
    angle = rng.uniform(-14, 14)
    m = cv2.getRotationMatrix2D((pw / 2, ph / 2), angle, 1.0)
    m[0, 2] += w / 2 - pw / 2 + rng.uniform(-0.12, 0.12) * w
    m[1, 2] += h / 2 - ph / 2 + rng.uniform(-0.14, 0.14) * h
    warped = cv2.warpAffine(small, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    a = warped[..., 3:4]
    if rng.random() < 0.3:                                            # keep the original white table sometimes
        img = np.ones_like(img) * rng.uniform(0.9, 1.0)
    img = img * (1 - a) + warped[..., :3] * a
    if rng.random() < 0.5:
        img = img[:, ::-1]
        labels = labels[::-1]
    if rng.random() < 0.5:
        img = img[::-1]
    return _camera(rng, np.ascontiguousarray(img), strength=0.7), labels


# ------------------------------------------------------- training set on disk ---
def _chunk(args):
    index, count, seed, real_share = args
    rng = np.random.default_rng((seed, index))
    bank = load_bank()
    real = load_real("train") if real_share > 0 else []
    x = np.zeros((count, H, W, 3), np.uint8)
    y = np.zeros((count, MAX_BANDS), np.int64)
    n = np.zeros(count, np.int64)
    for i in range(count):
        img, labels = augment_real(rng, real, bank) if real and rng.random() < real_share else render_sample(rng, bank)
        x[i] = img
        y[i, :len(labels)] = labels
        n[i] = len(labels)
    path = os.path.join(DATA_DIR, f"chunk_{index:04d}.npz")
    np.savez(path, x=x, y=y, n=n)
    return path


def make_dataset(total: int, chunk: int = 5000, seed: int = 20, real_share: float = 0.2, workers: int = 4) -> list[str]:
    """Write the training set as chunks. The last chunk is kept for validation
    (drawn pictures only, so its score is comparable from run to run)."""
    import multiprocessing as mp
    os.makedirs(DATA_DIR, exist_ok=True)
    for old in glob.glob(os.path.join(DATA_DIR, "chunk_*.npz")):
        os.remove(old)
    count = max(2, total // chunk)
    jobs = [(i, chunk, seed, real_share if i < count - 1 else 0.0) for i in range(count)]
    with mp.Pool(workers) as pool:
        paths = []
        for path in pool.imap_unordered(_chunk, jobs):
            paths.append(path)
            print(f"  {len(paths)}/{count} chunks", flush=True)
    return sorted(paths)


def preview(path: str, rows: int = 12, cols: int = 4, seed: int = 1) -> None:
    rng = np.random.default_rng(seed)
    bank = load_bank()
    real = load_real("train")
    sheet = np.zeros((rows * (H + 14), cols * W, 3), np.uint8)
    for i in range(rows * cols):
        img, labels = augment_real(rng, real, bank) if real and i % cols == cols - 1 else render_sample(rng, bank)
        y, x = (i // cols) * (H + 14), (i % cols) * W
        sheet[y:y + H, x:x + W] = img
        text = " ".join(COLOURS[k - 1][:3] for k in labels) or "(none)"
        cv2.putText(sheet, text, (x + 2, y + H + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1,
                    cv2.LINE_AA)
    cv2.imwrite(path, cv2.resize(sheet[..., ::-1], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", action="store_true", help="build the real-photo background bank")
    ap.add_argument("--make", type=int, help="write this many training pictures to disk")
    ap.add_argument("--real-share", type=float, default=0.2, help="share of pictures made from real photos")
    ap.add_argument("--preview", help="write a sheet of samples to this PNG (right column: from real photos)")
    args = ap.parse_args()
    if args.bank:
        print("background bank:", build_bank())
    if args.preview:
        preview(args.preview)
        print("wrote", args.preview)
    if args.make:
        print(f"real training photos: {len(load_real('train'))}")
        files = make_dataset(args.make, real_share=args.real_share)
        print(f"{len(files)} chunks in {DATA_DIR}")
