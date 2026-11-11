"""Font pool for the v2 synthetic data generator.

Three sources, merged into one validated manifest:

  * open-licensed handwriting fonts downloaded from Google Fonts into
    config.FONT_DIR (outside the repo),
  * handwriting-style and plain fonts that ship with Windows,
  * anything the user drops into backend/recognition/fonts/.

A set of the downloaded families is HELD OUT: never used for training, only by
the benchmark, so "unseen handwriting style" accuracy is measured honestly.

Each font is checked before use: it must render the whole charset, and its
lowercase letters must be real lowercase (small-caps fonts would teach the
model that "CL" and "Cl" look the same).

Build:  python -m backend.recognition.fontbank          (download + validate)
"""
from __future__ import annotations

import glob
import json
import os
import re
import string
import urllib.parse
import urllib.request
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import FONT_DIR

_HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(FONT_DIR, "manifest.json")
PROBE = string.ascii_letters + string.digits + "()+"

# Google Fonts handwriting families (all OFL / Apache licensed).
TRAIN_FAMILIES = [
    "Caveat", "Patrick Hand", "Indie Flower", "Shadows Into Light", "Kalam",
    "Gloria Hallelujah", "Architects Daughter", "Handlee", "Covered By Your Grace",
    "Reenie Beanie", "Rock Salt", "Gochi Hand", "Just Another Hand", "Coming Soon",
    "Schoolbell", "Annie Use Your Telescope", "Sue Ellen Francisco",
    "Waiting for the Sunrise", "Zeyada", "Homemade Apple", "Nothing You Could Do",
    "Over the Rainbow", "Swanky and Moo Moo", "Give You Glory", "Loved by the King",
    "Dekko", "Neucha", "Short Stack", "Walter Turncoat", "Crafty Girls", "Sunshiney",
    "Itim", "Mali", "Sriracha", "Pangolin", "Delius", "Chilanka", "Kristi",
    "La Belle Aurore", "Mansalva", "Fuzzy Bubbles", "Playpen Sans", "Shantell Sans",
    "Comic Neue", "Sedgwick Ave", "Rancho", "Caveat Brush", "Just Me Again Down Here",
    "Vibur", "The Girl Next Door", "Dawning of a New Day", "Shadows Into Light Two",
    "Bad Script", "Grape Nuts", "Delicious Handrawn", "Edu SA Beginner",
    "Edu VIC WA NT Beginner", "Edu NSW ACT Foundation", "Nerko One", "Mynerve",
    "Beth Ellen", "Gaegu", "Delius Swash Caps", "Oooh Baby", "Solitreo",
]
# held out: only the benchmark ever renders these
TEST_FAMILIES = [
    "Cedarville Cursive", "Edu QLD Beginner", "Edu TAS Beginner", "Kavivanar",
    "Borel", "Square Peg", "Ole", "Licorice", "Finger Paint", "Marck Script",
    "Playwrite US Modern", "Playwrite GB S", "Sedgwick Ave Display", "Send Flowers",
    "Life Savers",
]

# Windows fonts, matched on the lower-cased file name.
_WIN_HAND = ("segoepr", "segoesc", "comic", "inkfree", "bradhitc", "freescpt",
             "lhandw", "mistral", "kristen", "tempsitc", "vinerhand", "vineritc",
             "pristina", "maian", "mvboli", "papyrus", "gabriola")
_WIN_PLAIN = ("arial.ttf", "calibri.ttf", "verdana.ttf", "tahoma.ttf", "times.ttf",
              "georgia.ttf", "cambria", "consola.ttf", "segoeui.ttf", "trebuc.ttf",
              "candara.ttf", "corbel.ttf", "cour.ttf", "framd.ttf", "gothic.ttf")


def _css_ttf_url(family: str) -> str | None:
    """Ask the Google Fonts CSS API for the plain TTF of `family`."""
    url = "https://fonts.googleapis.com/css2?family=" + urllib.parse.quote_plus(family)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/4.0"})
    try:
        css = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")
    except Exception:
        return None
    m = re.search(r"url\((https://[^)]+\.ttf)\)", css)
    return m.group(1) if m else None


def download(families=None, verbose=True) -> int:
    """Fetch every listed family that is not on disk yet. Returns the number fetched."""
    os.makedirs(FONT_DIR, exist_ok=True)
    got = 0
    for fam in families or (TRAIN_FAMILIES + TEST_FAMILIES):
        dest = os.path.join(FONT_DIR, fam.replace(" ", "") + ".ttf")
        if os.path.exists(dest):
            continue
        url = _css_ttf_url(fam)
        if not url:
            if verbose:
                print(f"  skip (not found): {fam}")
            continue
        try:
            data = urllib.request.urlopen(url, timeout=60).read()
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"  skip ({exc}): {fam}")
            continue
        with open(dest, "wb") as f:
            f.write(data)
        got += 1
    return got


def _norm_glyph(font, ch: str, n: int = 24):
    """Binary n x n thumbnail of one glyph, plus its pixel height (None if empty)."""
    img = Image.new("L", (96, 96), 0)
    ImageDraw.Draw(img).text((20, 70), ch, fill=255, font=font, anchor="ls")
    a = np.asarray(img)
    ys, xs = np.nonzero(a > 80)
    if ys.size < 6:
        return None, 0
    crop = Image.fromarray(a[ys.min():ys.max() + 1, xs.min():xs.max() + 1])
    return np.asarray(crop.resize((n, n), Image.BILINEAR)) > 80, int(ys.max() - ys.min() + 1)


def _iou(a, b) -> float:
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 0.0


def is_usable(path: str) -> bool:
    """True if the font can draw the charset with genuine lowercase letters."""
    try:
        font = ImageFont.truetype(path, 44)
        img = Image.new("L", (44 * len(PROBE), 110), 0)
        ImageDraw.Draw(img).text((6, 80), PROBE, fill=255, font=font, anchor="ls")
    except Exception:
        return False
    thumbs = {}
    for ch in "aAeEgGnNrR128QO0+(":
        t, h = _norm_glyph(font, ch)
        if t is None:
            return False
        thumbs[ch] = (t, h)
    # missing-glyph boxes all look alike: unrelated characters must differ
    if _iou(thumbs["1"][0], thumbs["8"][0]) > 0.85 or _iou(thumbs["2"][0], thumbs["Q"][0]) > 0.85:
        return False
    pairs = ("aA", "eE", "gG", "nN", "rR")
    same_shape = np.mean([_iou(thumbs[p[0]][0], thumbs[p[1]][0]) for p in pairs])
    if same_shape > 0.55:
        return False                      # small-caps / caps-only font
    return True


def build_manifest(verbose=True) -> dict:
    """Validate every candidate font and write the train/test manifest."""
    windir = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    test_files = {f.replace(" ", "").lower() + ".ttf" for f in TEST_FAMILIES}

    groups = {"hand": [], "plain": [], "test": []}
    for path in sorted(glob.glob(os.path.join(FONT_DIR, "*.ttf"))):
        key = "test" if os.path.basename(path).lower() in test_files else "hand"
        groups[key].append(path)
    for path in sorted(glob.glob(os.path.join(_HERE, "fonts", "*.ttf"))):
        groups["hand"].append(path)
    if os.path.isdir(windir):
        for path in sorted(glob.glob(os.path.join(windir, "*.ttf"))):
            name = os.path.basename(path).lower()
            if any(name.startswith(h) for h in _WIN_HAND):
                groups["hand"].append(path)
            elif any(name.startswith(p) for p in _WIN_PLAIN):
                groups["plain"].append(path)

    manifest = {k: [p for p in v if is_usable(p)] for k, v in groups.items()}
    os.makedirs(FONT_DIR, exist_ok=True)
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    if verbose:
        for k, v in groups.items():
            print(f"{k:>5}: {len(manifest[k])} usable of {len(v)} candidates")
    return manifest


@lru_cache(maxsize=1024)
def load_font(path: str, px: int):
    return ImageFont.truetype(path, px)


class FontBank:
    """Picks a font file for one sample. split='test' uses only held-out families."""

    def __init__(self, split: str = "train", plain_ratio: float = 0.15):
        if not os.path.exists(MANIFEST):
            raise FileNotFoundError(
                "Font manifest not built. Run: python -m backend.recognition.fontbank")
        with open(MANIFEST, encoding="utf-8") as f:
            m = json.load(f)
        self.split = split
        self.plain_ratio = plain_ratio
        self.hand = [p for p in m["hand"] if os.path.exists(p)]
        self.plain = [p for p in m["plain"] if os.path.exists(p)]
        self.test = [p for p in m["test"] if os.path.exists(p)]
        if split == "test" and not self.test:
            raise RuntimeError("No held-out test fonts available.")
        if split != "test" and not self.hand:
            raise RuntimeError("No training fonts available.")

    def pick(self, rng) -> str:
        if self.split == "test":
            return rng.choice(self.test)
        if self.plain and rng.random() < self.plain_ratio:
            return rng.choice(self.plain)
        return rng.choice(self.hand)


if __name__ == "__main__":
    n = download()
    print(f"downloaded {n} new font files into {FONT_DIR}")
    build_manifest()
