"""Curated reaction lookup.

Given a set of reactant formulas, decide *what* the products are. This is the
deliberately-scoped part of the app: we never try to predict products for
arbitrary reactants (that is unreliable). Instead we match against a curated
database plus a couple of well-defined rules (e.g. hydrocarbon combustion).
Once products are known, `balancer` computes the coefficients.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from .formula import canonical_formula, normalize_ocr, parse_formula

_DATA = os.path.join(os.path.dirname(__file__), "data", "reactions.json")


def _norm(formula: str) -> str:
    """Clean OCR output, then map alternative spellings (C2H6O) to the main one."""
    return canonical_formula(normalize_ocr(formula))


def _load_table() -> dict:
    with open(_DATA, encoding="utf-8") as f:
        data = json.load(f)
    table = {}
    for entry in data.get("reactions", []):
        # normalize keys so "H20" written by hand still matches "H2O"
        key = frozenset(_norm(r) for r in entry["reactants"])
        if key in table:
            raise ValueError(f"reactions.json lists {sorted(key)} twice")
        table[key] = entry
    return table


_TABLE = _load_table()


def _is_hydrocarbon(counts: Optional[dict]) -> bool:
    """True for C/H or C/H/O molecules (things that burn to CO2 + H2O)."""
    if not counts:
        return False
    return (set(counts).issubset({"C", "H", "O"}) and "C" in counts and "H" in counts
            and 4 * counts["C"] + counts["H"] - 2 * counts.get("O", 0) > 0)


def find_reaction(reactants: list[str]) -> Optional[dict]:
    """Return a reaction record for the given reactants, or None if unknown.

    The record always carries normalized ``reactants`` and a ``products`` list;
    database hits also include ``name``/``conditions``/``note``.
    """
    norm = [_norm(r) for r in reactants if r]
    norm = list(dict.fromkeys(r for r in norm if r))
    if not norm:
        return None

    key = frozenset(norm)
    if key in _TABLE:
        rec = dict(_TABLE[key])
        rec["reactants"] = norm
        rec["source"] = "database"
        return rec

    # --- rule: generic hydrocarbon combustion --------------------------------
    if len(norm) == 2 and "O2" in norm:
        other = next(r for r in norm if r != "O2")
        counts = parse_formula(other)
        if _is_hydrocarbon(counts):
            return {
                "reactants": norm,
                "products": ["CO2", "H2O"],
                "name": "Combustion (hydrocarbon + O₂)",
                "source": "rule",
            }

    return None
