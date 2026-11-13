"""Resistor colour code: the band alphabet and how bands turn into a value.

No heavy imports, so the data generator, the trainer, the exporter and the
tests can all share it. The JavaScript twin is static/js/assist/resistor.js
(`decodeBands`); tests/test_resistor.py keeps the two in step.
"""
from __future__ import annotations

# index 0 is the CTC blank; the model's classes are BLANK + COLOURS
COLOURS = ["black", "brown", "red", "orange", "yellow", "green", "blue",
           "violet", "grey", "white", "gold", "silver"]
BLANK = 0
CLASSES = 1 + len(COLOURS)

DIGIT = {c: i for i, c in enumerate(COLOURS[:10])}
MULTIPLIER = {**{c: 10.0 ** i for i, c in enumerate(COLOURS[:10])}, "gold": 0.1, "silver": 0.01}
TOLERANCE = {"brown": 1.0, "red": 2.0, "green": 0.5, "blue": 0.25, "violet": 0.1,
             "grey": 0.05, "gold": 5.0, "silver": 10.0}

E12 = [10, 12, 15, 18, 22, 27, 33, 39, 47, 56, 68, 82]
E24 = [10, 11, 12, 13, 15, 16, 18, 20, 22, 24, 27, 30, 33, 36, 39, 43, 47, 51, 56, 62, 68, 75, 82, 91]
E96 = [100, 102, 105, 107, 110, 113, 115, 118, 121, 124, 127, 130, 133, 137, 140, 143, 147, 150,
       154, 158, 162, 165, 169, 174, 178, 182, 187, 191, 196, 200, 205, 210, 215, 221, 226, 232,
       237, 243, 249, 255, 261, 267, 274, 280, 287, 294, 301, 309, 316, 324, 332, 340, 348, 357,
       365, 374, 383, 392, 402, 412, 422, 432, 442, 453, 464, 475, 487, 499, 511, 523, 536, 549,
       562, 576, 590, 604, 619, 634, 649, 665, 681, 698, 715, 732, 750, 768, 787, 806, 825, 845,
       866, 887, 909, 931, 953, 976]
# 5-band parts are also sold in the E24 values written with three digits
E24_3 = [v * 10 for v in E24]


def value_of(bands: list[str]) -> dict | None:
    """Read bands in the order given. Returns None when that order is not a
    legal colour code (so the caller can try the other direction).

      3 bands  digit digit multiplier              (tolerance 20 %)
      4 bands  digit digit multiplier tolerance
      5 bands  digit digit digit multiplier tolerance
    """
    n = len(bands)
    if n not in (3, 4, 5):
        return None
    digits = bands[:3] if n == 5 else bands[:2]
    mult = bands[3] if n == 5 else bands[2]
    tol = bands[-1] if n >= 4 else None
    if any(d not in DIGIT for d in digits) or mult not in MULTIPLIER:
        return None
    if DIGIT[digits[0]] == 0:                      # a value never starts with black
        return None
    if tol is not None and tol not in TOLERANCE:
        return None
    significand = 0
    for d in digits:
        significand = significand * 10 + DIGIT[d]
    series = E96 + E24_3 if n == 5 else E24
    return {
        "ohms": significand * MULTIPLIER[mult],
        "tolerance": TOLERANCE[tol] if tol else 20.0,
        "standard": significand in series,       # a value that is actually manufactured
        "bands": list(bands),
    }


def decode(bands: list[str]) -> dict | None:
    """Bands as seen left-to-right -> the most plausible reading.

    A resistor can be held either way round, so both directions are tried:
    a direction must be a legal code; a standard (E-series) value beats a
    non-standard one; a gold/silver band can only be at the far end. If both
    directions are equally plausible the result is marked ambiguous and
    carries the alternative.
    """
    forward, backward = value_of(bands), value_of(bands[::-1])
    if len(bands) >= 4:
        # gold/silver are never the first band
        if bands[0] in ("gold", "silver"):
            forward = None
        if bands[-1] in ("gold", "silver"):
            backward = None
    options = [o for o in (forward, backward) if o]
    if not options:
        return None
    if len(options) == 2 and options[0]["bands"] == options[1]["bands"]:
        options = options[:1]                      # a palindrome reads the same both ways
    options.sort(key=lambda o: not o["standard"])  # stable: standard values first
    best = dict(options[0])
    if len(options) == 2 and options[0]["standard"] == options[1]["standard"]:
        best["ambiguous"] = True
        best["alternative"] = options[1]
    else:
        best["ambiguous"] = False
    return best


def format_ohms(ohms: float) -> str:
    for scale, unit in ((1e9, "GΩ"), (1e6, "MΩ"), (1e3, "kΩ")):
        if ohms >= scale:
            return f"{ohms / scale:.3g} {unit}"
    return f"{ohms:.3g} Ω"
