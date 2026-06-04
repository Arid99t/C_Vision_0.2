"""Equation balancing and the top-level `react` entry point.

`balance` is a thin wrapper over chempy's linear-algebra balancer. `react`
ties everything together: look up products for the reactants, balance, and
format a human-readable equation with unicode subscripts.
"""
from __future__ import annotations

import re
from typing import Optional

from chempy import balance_stoichiometry

from .formula import normalize_ocr, to_subscripts
from .reactions import find_reaction


def _coerce(species) -> dict:
    """chempy returns sympy Integer/Rational coefficients; convert to native
    Python ints (or floats for the rare fractional case) so the result is
    JSON-serializable."""
    out = {}
    for formula, coeff in species.items():
        ival = int(coeff)
        out[formula] = ival if ival == coeff else float(coeff)
    return out


def balance(reactants: list[str], products: list[str]) -> tuple[dict, dict]:
    """Balance ``reactants -> products``; returns ({formula: coeff}, {...}).

    Raises ValueError if the system cannot be balanced (chempy raises its own
    exceptions, which we normalize to ValueError for the API layer).
    """
    try:
        reac, prod = balance_stoichiometry(set(reactants), set(products))
    except Exception as exc:  # chempy raises ValueError / linsolve failures
        raise ValueError(str(exc)) from exc
    return _coerce(reac), _coerce(prod)


def _format_side(species: dict) -> str:
    parts = []
    for formula, coeff in species.items():
        prefix = "" if coeff == 1 else f"{coeff} "
        parts.append(f"{prefix}{to_subscripts(formula)}")
    return " + ".join(parts)


def format_equation(reactants: dict, products: dict) -> str:
    """Build 'a A + b B → c C + d D' with subscripts and coefficients."""
    return f"{_format_side(reactants)} → {_format_side(products)}"


def react(reactants: list[str]) -> dict:
    """High-level: given reactant formulas, return the balanced reaction.

    Always returns a dict with ``ok``. On success it carries ``equation``
    (display string), ``reactants``/``products`` (coefficient maps), plus the
    reaction ``name``/``conditions``/``source``. On failure ``ok`` is False and
    ``reason`` explains why (unknown reactants or unbalanceable).
    """
    norm = [normalize_ocr(r) for r in reactants if r]
    info = find_reaction(reactants)
    if info is None:
        return {
            "ok": False,
            "reason": "No known reaction for these reactants.",
            "reactants": norm,
        }

    try:
        reac, prod = balance(info["reactants"], info["products"])
    except ValueError as exc:
        return {
            "ok": False,
            "reason": f"Could not balance reaction: {exc}",
            "reactants": info["reactants"],
            "products": info["products"],
        }

    return {
        "ok": True,
        "equation": format_equation(reac, prod),
        "reactants": reac,
        "products": prod,
        "name": info.get("name"),
        "conditions": info.get("conditions"),
        "note": info.get("note"),
        "source": info.get("source"),
    }
