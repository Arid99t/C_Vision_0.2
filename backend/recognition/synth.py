"""Synthetic handwritten-formula data generator.

Training a handwriting recognizer needs labeled images. Collecting thousands by
hand is slow, so we bootstrap with synthetic data: render formula strings using
handwriting-style fonts and apply aggressive augmentation (rotation, shear,
stroke-width changes, warping, noise, blur) so the model generalizes to real
ink. You can later mix in real labeled samples via `dataset.FolderDataset` to
fine-tune.

Everything here uses only PIL + numpy (no torch), so it can also be used to
pre-render a dataset to disk.
"""
from __future__ import annotations

import glob
import os
import random
import string
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from .config import IMG_HEIGHT

# --- label vocabulary used to synthesize formulas -------------------------
_ELEMENTS = ["H", "C", "N", "O", "S", "P", "Cl", "Na", "K", "Ca", "Mg", "Fe",
             "Zn", "Al", "Cu", "Br", "I", "F", "Si", "Mn"]

# real formulas anchor the distribution toward things the app actually sees
_REAL = ["H2O", "CO2", "CO", "CH4", "O2", "H2", "N2", "Cl2", "NH3", "HCl",
         "NaCl", "NaOH", "H2SO4", "CaCO3", "CaO", "Fe2O3", "MgO", "ZnCl2",
         "C2H5OH", "C6H12O6", "C3H8", "H2CO3", "NH4Cl", "Na2SO4", "Ca(OH)2"]


def random_formula() -> str:
    """Return a plausible formula (or short equation) label string."""
    r = random.random()
    if r < 0.55:
        return random.choice(_REAL)
    if r < 0.8:
        # random element(count) chain
        n = random.randint(1, 3)
        parts = []
        for _ in range(n):
            el = random.choice(_ELEMENTS)
            cnt = random.choice(["", "", "2", "3", "4", "5", "6"])
            parts.append(el + cnt)
        return "".join(parts)
    # a two-term equation fragment "A + B"
    return f"{random.choice(_REAL)} + {random.choice(_REAL)}"


# --- font discovery -------------------------------------------------------
# Probe string: every character class we ever synthesize. A font that renders
# all of these will render any formula we generate.
_PROBE = string.ascii_letters + string.digits + "()+ "

# Symbol/icon fonts that exist on Windows but have no text glyphs (would raise
# "Bitmap missing for glyph"). Skipped by name before the slower render check.
_SYMBOL_HINTS = ("wingding", "webding", "symbol", "marlett", "holomdl",
                 "segmdl2", "segoeicons", "seguiemj", "seguisym", "bssym",
                 "mssym", "mtextra", "opensym", "refspecialty", "msreference")


def _font_can_render(path: str) -> bool:
    """True if the font loads and renders the full probe without errors."""
    try:
        font = ImageFont.truetype(path, 32)
        img = Image.new("L", (len(_PROBE) * 34 + 20, 80), 255)
        ImageDraw.Draw(img).text((4, 4), _PROBE, fill=0, font=font)
        return True
    except Exception:  # OSError (missing glyph), font parse errors, etc.
        return False


def _discover_fonts() -> list[str]:
    """Find .ttf fonts that can actually render our characters, preferring
    handwriting/script styles. Symbol fonts are filtered out so the data
    generator can't crash on an unrenderable glyph."""
    search_dirs = [
        os.environ.get("WINDIR", "C:\\Windows") + "\\Fonts",
        "/usr/share/fonts",
        "/Library/Fonts",
        os.path.join(os.path.dirname(__file__), "fonts"),  # drop custom fonts here
    ]
    prefer = ("segoesc", "segoepr", "comic", "ink", "hand", "script", "bradhitc",
              "freescpt", "mistral", "lucida", "gabriola", "caveat", "patrick",
              "kalam", "shadows", "reenie")
    found, preferred = [], []
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for path in glob.glob(os.path.join(d, "*.ttf")):
            name = os.path.basename(path).lower()
            if any(h in name for h in _SYMBOL_HINTS):
                continue
            (preferred if any(p in name for p in prefer) else found).append(path)
    # validate: keep only fonts that render the whole charset
    usable = [p for p in (preferred + found) if _font_can_render(p)]
    return usable  # empty -> caller uses PIL default bitmap font


_FONTS = _discover_fonts()


def _load_font(size: int):
    if _FONTS:
        try:
            return ImageFont.truetype(random.choice(_FONTS), size)
        except OSError:
            pass
    return ImageFont.load_default()


# --- rendering + augmentation --------------------------------------------
def _sized_font(font, px):
    """Reload a TrueType font at a different pixel size (used for subscripts)."""
    try:
        return ImageFont.truetype(font.path, max(8, int(px)))
    except Exception:
        return font


def _advance(draw, ch, font):
    try:
        return font.getlength(ch)
    except AttributeError:  # very old PIL
        return draw.textsize(ch, font=font)[0]


def _render_inline(text, size, font):
    """All characters on one baseline at full size (e.g. 'H2O')."""
    tmp = ImageDraw.Draw(Image.new("L", (10, 10), 255))
    try:
        l, t, r, b = tmp.textbbox((0, 0), text, font=font)
        tw, th = r - l, b - t
    except AttributeError:
        tw, th = tmp.textsize(text, font=font)
        l = t = 0
    pad = max(6, size // 3)
    img = Image.new("L", (max(8, tw + 2 * pad), max(8, th + 2 * pad)), 255)
    ImageDraw.Draw(img).text((pad - l, pad - t), text, fill=0, font=font)
    return img


def _render_subscript(text, size, font):
    """Element counts drawn as smaller, lowered digits (the natural 'H₂O' way).

    The label stays plain ASCII ('H2O'); only the rendering changes, so the model
    learns to map subscript handwriting to the same string. Size/drop are
    randomized so it generalizes across how differently people write subscripts.
    """
    sub_font = _sized_font(font, size * random.uniform(0.55, 0.72))
    drop = int(size * random.uniform(0.22, 0.42))
    pad = max(6, size // 3)
    probe = ImageDraw.Draw(Image.new("L", (10, 10), 255))

    layout, x, in_sub, prev_anchor = [], float(pad), False, False
    for ch in text:
        is_count = ch.isdigit() and (prev_anchor or in_sub)  # a count after element/')'
        in_sub = is_count
        f = sub_font if is_count else font
        layout.append((ch, f, x, drop if is_count else 0))
        x += _advance(probe, ch, f)
        prev_anchor = ch.isalpha() or ch == ")"

    total_w = int(x) + pad
    baseline = pad + size
    img = Image.new("L", (max(8, total_w), max(8, size + drop + 2 * pad)), 255)
    d = ImageDraw.Draw(img)
    for ch, f, cx, off in layout:
        d.text((int(cx), baseline + off), ch, fill=0, font=f, anchor="ls")
    return img


def _render(text: str, size: int) -> Image.Image:
    """Render `text`, ~half the time using subscript-style counts. Never raises:
    falls back to inline, then the default bitmap font, then a blank canvas."""
    chosen = _load_font(size)
    attempts = []
    if random.random() < 0.5 and getattr(chosen, "path", None):
        attempts.append((_render_subscript, chosen))
    attempts.append((_render_inline, chosen))
    attempts.append((_render_inline, ImageFont.load_default()))
    for renderer, fnt in attempts:
        try:
            return renderer(text, size, fnt)
        except OSError:  # "Bitmap missing for glyph" -> next fallback
            continue
    return Image.new("L", (max(8, len(text) * size // 2), size + 8), 255)


def _augment(img: Image.Image) -> Image.Image:
    w, h = img.size
    # random affine: rotation + shear + scale jitter (simulates slanted writing)
    angle = random.uniform(-6, 6)
    img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=255, expand=True)
    shear = random.uniform(-0.25, 0.25)
    img = img.transform(img.size, Image.AFFINE, (1, shear, 0, 0, 1, 0),
                        resample=Image.BILINEAR, fillcolor=255)
    # stroke thickness: erode (thinner) or dilate (thicker) via min/max filter
    if random.random() < 0.5:
        flt = ImageFilter.MinFilter(3) if random.random() < 0.5 else ImageFilter.MaxFilter(3)
        img = img.filter(flt)
    # slight blur
    if random.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.4, 1.2)))
    # gaussian noise
    arr = np.asarray(img, dtype=np.float32)
    arr += np.random.normal(0, random.uniform(2, 14), arr.shape)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "L")


def to_fixed_height(img: Image.Image, height: int = IMG_HEIGHT) -> np.ndarray:
    """Resize to fixed height keeping aspect ratio; return (H, W) uint8 array."""
    w, h = img.size
    new_w = max(8, int(round(w * height / h)))
    img = img.resize((new_w, height), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def make_sample(text: str | None = None) -> tuple[np.ndarray, str]:
    """Generate one (image, label). Image is a fixed-height (H, W) uint8 array."""
    if text is None:
        text = random_formula()
    size = random.randint(28, 48)
    img = _augment(_render(text, size))
    return to_fixed_height(img), text


def fonts_available() -> int:
    return len(_FONTS)


if __name__ == "__main__":  # quick visual smoke test -> saves a few samples
    os.makedirs("synth_preview", exist_ok=True)
    print(f"Discovered {fonts_available()} fonts")
    for i in range(8):
        arr, label = make_sample()
        Image.fromarray(arr, "L").save(f"synth_preview/{i}_{label.replace(' ', '_')}.png")
    print("Wrote samples to ./synth_preview/")
