"""Character vocabulary for the recognizer.

The model is a sequence recognizer (CTC), so it emits one character at a time.
Index 0 is reserved as the CTC "blank". Everything the model can output for a
handwritten chemistry formula lives here: digits, element letters (upper- and
lower-case for two-letter symbols like Ca/Cl), parentheses, '+' and space for
full equation lines.
"""

# order matters: index 0 is CTC blank, real chars start at 1
BLANK = "<blank>"

_DIGITS = "0123456789"
_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_LOWER = "abcdefghijklmnopqrstuvwxyz"
_SYMBOLS = "()+ "

CHARS = _DIGITS + _UPPER + _LOWER + _SYMBOLS

# index<->char maps (0 == blank)
ITOS = [BLANK] + list(CHARS)
STOI = {c: i for i, c in enumerate(ITOS)}

NUM_CLASSES = len(ITOS)  # includes blank


def encode(text: str) -> list[int]:
    """Map a label string to class indices (skips unknown chars)."""
    return [STOI[c] for c in text if c in STOI]


def decode_greedy(indices) -> str:
    """CTC greedy decode: collapse repeats, then drop blanks.

    `indices` is the per-timestep argmax sequence (ints).
    """
    out = []
    prev = -1
    for idx in indices:
        idx = int(idx)
        if idx != prev and idx != 0:  # not a repeat, not blank
            out.append(ITOS[idx])
        prev = idx
    return "".join(out)
