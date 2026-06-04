"""Chemical-formula parsing and OCR-output normalization.

Two jobs:
  1. `normalize_ocr` cleans raw text coming out of the handwriting model
     (unicode subscripts, stray whitespace, the classic 0/O confusion) into a
     canonical ASCII formula such as "H2O" or "Ca(OH)2".
  2. `parse_formula` turns a canonical formula into an element->count mapping,
     returning None when the string is not a valid formula. That None is also
     what `normalize_ocr` uses to decide whether a correction candidate is good.
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from itertools import product
from typing import Optional

# --- periodic table -------------------------------------------------------
# Full set of IUPAC element symbols. Parsing validates against this so that
# OCR noise like "Xz" is rejected instead of silently parsed.
ELEMENTS = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al",
    "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe",
    "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm",
    "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",
    "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
    "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
}

# unicode subscript/superscript digits -> ascii
_SUB_TO_ASCII = {ord(c): str(i) for i, c in enumerate("₀₁₂₃₄₅₆₇₈₉")}
_SUP_TO_ASCII = {ord(c): d for c, d in zip("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻", "0123456789+-")}
_ASCII_TO_SUB = {str(i): c for i, c in enumerate("₀₁₂₃₄₅₆₇₈₉")}

# characters that handwriting OCR routinely swaps for one another.
# Each entry maps a seen char -> the alternatives worth trying during cleanup.
_AMBIGUOUS = {"0": "0O", "O": "0O"}

_TOKEN_RE = re.compile(r"[A-Z][a-z]?|\d+|\(|\)")

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def to_subscripts(formula: str) -> str:
    """Render a formula's digits as unicode subscripts for display (H2O -> H₂O)."""
    return re.sub(r"\d", lambda m: _ASCII_TO_SUB[m.group()], formula)


def parse_formula(formula: str) -> Optional[dict]:
    """Parse a canonical formula into {element: count}, or None if invalid.

    Supports nested parentheses and multipliers, e.g. ``Ca(OH)2`` ->
    ``{"Ca": 1, "O": 2, "H": 2}`` and ``Mg(NO3)2`` correctly.
    """
    if not formula:
        return None
    tokens = _TOKEN_RE.findall(formula)
    # reject anything the tokenizer could not fully consume (stray symbols)
    if "".join(tokens) != formula:
        return None

    stack: list[defaultdict] = [defaultdict(int)]
    i, n = 0, len(tokens)
    while i < n:
        tok = tokens[i]
        if tok == "(":
            stack.append(defaultdict(int))
            i += 1
        elif tok == ")":
            i += 1
            mult = 1
            if i < n and tokens[i].isdigit():
                mult = int(tokens[i])
                i += 1
            group = stack.pop()
            if not stack:  # unbalanced ')'
                return None
            for el, c in group.items():
                stack[-1][el] += c * mult
        elif tok.isdigit():
            # a number with no element/group in front of it is invalid
            return None
        else:
            if tok not in ELEMENTS:
                return None
            i += 1
            count = 1
            if i < n and tokens[i].isdigit():
                count = int(tokens[i])
                i += 1
            stack[-1][tok] += count

    if len(stack) != 1:  # unbalanced '('
        return None
    return dict(stack[0])


def normalize_ocr(text: Optional[str]) -> Optional[str]:
    """Clean raw recognizer output into a canonical formula string.

    Steps: unicode sub/superscripts -> ascii, drop whitespace, then resolve the
    classic 0/O ambiguity. Plain text can't tell ``H2O`` (water) from ``H20``
    (20 hydrogens) on its own, so we lean on a dictionary of known compounds:
    we generate every 0/O substitution that still parses, and prefer one that
    spells a *known* compound. Failing that we keep the form closest to the
    original (fewest changes), which leaves real molecules like C10H8 alone.

    The trained recognizer is the real fix for 0/O — it draws different glyphs;
    this is just a safety net for noisy output.
    """
    if text is None:
        return None
    t = text.translate(_SUB_TO_ASCII).translate(_SUP_TO_ASCII)
    t = re.sub(r"\s+", "", t)
    if not t:
        return t

    # positions of ambiguous chars; cap the search space so pathological input
    # can't explode the 2**k combinations.
    positions = [i for i, c in enumerate(t) if c in _AMBIGUOUS][:10]
    choices = [_AMBIGUOUS[t[p]] for p in positions]

    parseable: list[tuple[str, int]] = []  # (candidate, #changes-from-original)
    for combo in product(*choices):
        cand = list(t)
        for pos, ch in zip(positions, combo):
            cand[pos] = ch
        cand_s = "".join(cand)
        if parse_formula(cand_s) is not None:
            changes = sum(1 for pos in positions if cand_s[pos] != t[pos])
            parseable.append((cand_s, changes))

    if not parseable:
        return t  # nothing parses; hand back the cleaned text for display

    known = [c for c in parseable if c[0] in KNOWN_FORMULAS]
    pool = known if known else parseable
    pool.sort(key=lambda c: c[1])  # fewest changes wins ties
    return pool[0][0]


# --- friendly names -------------------------------------------------------
def _load_compounds() -> dict:
    path = os.path.join(_DATA_DIR, "compounds.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


_COMPOUNDS = _load_compounds()


def _load_known_formulas() -> set:
    """Canonical formula strings we recognize, used to disambiguate 0/O.

    Pulls from compounds.json plus every reactant/product in reactions.json,
    read directly to avoid an import cycle with the reactions module.
    """
    known = set(_COMPOUNDS.keys())
    try:
        with open(os.path.join(_DATA_DIR, "reactions.json"), encoding="utf-8") as f:
            data = json.load(f)
        for entry in data.get("reactions", []):
            known.update(entry.get("reactants", []))
            known.update(entry.get("products", []))
    except FileNotFoundError:
        pass
    return known


KNOWN_FORMULAS = _load_known_formulas()


def describe_formula(formula: str) -> dict:
    """Return a display description of a single formula.

    Includes the canonical form, subscript form, element composition and a
    common name when known. Used by the clients to "display what's written".
    """
    canonical = normalize_ocr(formula)
    counts = parse_formula(canonical) if canonical else None
    return {
        "input": formula,
        "formula": canonical,
        "pretty": to_subscripts(canonical) if canonical else formula,
        "valid": counts is not None,
        "composition": counts or {},
        "name": _COMPOUNDS.get(canonical),
    }
