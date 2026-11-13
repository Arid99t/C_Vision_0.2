"""Test pictures for the Assist app: drawn "packages", "chips" and "boards".

These are stand-ins rendered with system fonts, good for checking that the
pipeline works end to end and stays working. They are NOT photos: passing
these tests says nothing about accuracy on real packaging.
"""
from __future__ import annotations

import os

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from tests.webharness import ROOT

OUT_DIR = os.path.join(ROOT, "backend", "static", "test")
_FONT_DIRS = [r"C:\Windows\Fonts", "/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/liberation",
              "/System/Library/Fonts/Supplemental", "/Library/Fonts"]
_BOLD = ["arialbd.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "Arial Bold.ttf"]
_PLAIN = ["arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf", "Arial.ttf"]
_MONO = ["consolab.ttf", "courbd.ttf", "DejaVuSansMono-Bold.ttf", "LiberationMono-Bold.ttf", "Courier New Bold.ttf"]


def _font(names, size):
    for d in _FONT_DIRS:
        for n in names:
            path = os.path.join(d, n)
            if os.path.exists(path):
                return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


def fonts_available() -> bool:
    return any(os.path.exists(os.path.join(d, n)) for d in _FONT_DIRS for n in _BOLD)


def _pack(draw, box, colour, lines, ink="white"):
    """A package: a rounded block of colour with centred lines of text.
    lines: [(text, size, bold?)]"""
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=18, fill=colour)
    total = sum(int(size * 1.25) for _, size, _ in lines)
    y = y0 + (y1 - y0 - total) // 2
    for text, size, bold in lines:
        f = _font(_BOLD if bold else _PLAIN, size)
        w = draw.textlength(text, font=f)
        draw.text((x0 + (x1 - x0 - w) / 2, y), text, font=f, fill=ink)
        y += int(size * 1.25)


def packages(left, right=None, size=(1280, 720), name=None, blur=0.6) -> str:
    """One or two packages on a kitchen-counter-coloured background.
    left/right: (colour, [(text, size, bold)], ink)"""
    img = Image.new("RGB", size, (206, 198, 186))
    d = ImageDraw.Draw(img)
    w, h = size
    if right is None:
        _pack(d, (w // 2 - 220, 90, w // 2 + 220, h - 90), left[0], left[1], left[2] if len(left) > 2 else "white")
    else:
        _pack(d, (90, 90, w // 2 - 110, h - 90), left[0], left[1], left[2] if len(left) > 2 else "white")
        _pack(d, (w // 2 + 110, 90, w - 90, h - 90), right[0], right[1], right[2] if len(right) > 2 else "white")
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    return _save(img, name)


def chip(marking: str, second_line: str = "", name=None, size=(1280, 720), scale=0.5) -> str:
    """A DIP chip seen from above, light-grey marking on a dark body. At the
    default scale it fits the Electronics box (about a quarter of the frame)."""
    img = Image.new("RGB", size, (58, 96, 70))
    d = ImageDraw.Draw(img)
    w, h = size

    def k(v):
        return int(round(v * scale))

    bx0, by0, bx1, by1 = w // 2 - k(250), h // 2 - k(95), w // 2 + k(250), h // 2 + k(95)
    for i in range(8):
        x = bx0 + k(30) + i * k(63)
        d.rectangle((x, by0 - k(34), x + k(22), by0), fill=(200, 200, 205))
        d.rectangle((x, by1, x + k(22), by1 + k(34)), fill=(200, 200, 205))
    d.rounded_rectangle((bx0, by0, bx1, by1), radius=k(8), fill=(34, 34, 38))
    d.ellipse((bx0 + k(12), h // 2 - k(16), bx0 + k(44), h // 2 + k(16)), fill=(22, 22, 24))
    f = _font(_MONO, k(62))
    tw = d.textlength(marking, font=f)
    d.text((w // 2 - tw / 2 + k(16), h // 2 - k(74 if second_line else 36)), marking, font=f, fill=(196, 196, 200))
    if second_line:
        f2 = _font(_MONO, k(40))
        tw = d.textlength(second_line, font=f2)
        d.text((w // 2 - tw / 2 + k(16), h // 2 + k(14)), second_line, font=f2, fill=(170, 170, 176))
    return _save(img.filter(ImageFilter.GaussianBlur(0.5)), name)


def board(lines, name=None, size=(1280, 720)) -> str:
    """A whiteboard / slide with a few lines of printed text."""
    img = Image.new("RGB", size, (244, 244, 240))
    d = ImageDraw.Draw(img)
    y = 110
    for i, text in enumerate(lines):
        f = _font(_BOLD if i == 0 else _PLAIN, 64 if i == 0 else 48)
        d.text((110, y), text, font=f, fill=(28, 34, 52))
        y += 110 if i == 0 else 84
    return _save(img.filter(ImageFilter.GaussianBlur(0.5)), name)


def _save(img, name) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    img.save(os.path.join(OUT_DIR, name))
    return "/static/test/" + name


def resistor_frame(bands, seed=3, name=None, size=(1280, 720)) -> str:
    """A camera frame with one drawn resistor in the middle, the size the
    Electronics box expects (the box is 1/8 of the frame's long side)."""
    import numpy as np

    from backend.resistor.render import load_bank, render_sample
    rng = np.random.default_rng(seed)
    patch, _ = render_sample(rng, load_bank(), bands=bands)
    w, h = size
    pw = round(0.125 * max(w, h))
    ph = round(pw * patch.shape[0] / patch.shape[1])
    img = Image.new("RGB", size, (96, 100, 104))
    img.paste(Image.fromarray(patch).resize((pw, ph), Image.LANCZOS), ((w - pw) // 2, (h - ph) // 2))
    return _save(img, name)


# EAN-13 drawn from the published bit patterns (kept separate from the reader's
# own tables in static/js/assist/barcode.js, so the tests are not circular)
_EAN_L = ["0001101", "0011001", "0010011", "0111101", "0100011", "0110001", "0101111", "0111011", "0110111", "0001011"]
_EAN_PARITY = ["LLLLLL", "LLGLGG", "LLGGLG", "LLGGGL", "LGLLGG", "LGGLLG", "LGGGLL", "LGLGLG", "LGLGGL", "LGGLGL"]


def ean13_check(first12: str) -> str:
    """The 13-digit code for 12 digits (adds the check digit)."""
    total = sum(int(d) * (3 if i % 2 else 1) for i, d in enumerate(first12))
    return first12 + str((10 - total % 10) % 10)


def ean13_bits(code: str) -> str:
    """The 95 bars (1) and gaps (0) of an EAN-13 code."""
    def right(d):
        return "".join("1" if b == "0" else "0" for b in _EAN_L[int(d)])
    bits = "101"
    for d, p in zip(code[1:7], _EAN_PARITY[int(code[0])]):
        bits += _EAN_L[int(d)] if p == "L" else right(d)[::-1]
    bits += "01010" + "".join(right(d) for d in code[7:])
    return bits + "101"


def barcode_scene(code: str, right=None, module: int = 4, name=None, size=(1280, 720), sideways=False) -> str:
    """A package showing its barcode (left or alone), optionally beside a second
    package with printed words. right: (colour, [(text, size, bold)], ink)"""
    img = Image.new("RGB", size, (206, 198, 186))
    d = ImageDraw.Draw(img)
    w, h = size
    box = (90, 90, w // 2 - 110, h - 90) if right else (w // 2 - 260, 90, w // 2 + 260, h - 90)
    d.rounded_rectangle(box, radius=18, fill=(232, 228, 214))
    bits = ean13_bits(code)
    bw, bh = len(bits) * module, 150
    label = Image.new("RGB", (bw + 22 * module, bh + 54), "white")
    ld = ImageDraw.Draw(label)
    for i, b in enumerate(bits):
        if b == "1":
            ld.rectangle((11 * module + i * module, 12, 11 * module + (i + 1) * module - 1, 12 + bh), fill=(12, 12, 14))
    ld.text((11 * module, bh + 18), code, font=_font(_PLAIN, 26), fill=(20, 20, 20))
    if sideways:
        label = label.rotate(90, expand=True)
    img.paste(label, ((box[0] + box[2] - label.width) // 2, (box[1] + box[3] - label.height) // 2))
    if right:
        _pack(d, (w // 2 + 110, 90, w - 90, h - 90), right[0], right[1], right[2] if len(right) > 2 else "white")
    return _save(img.filter(ImageFilter.GaussianBlur(0.6)), name)
