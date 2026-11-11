"""Generate a 3D atom layout for a formula, for the AR clients' molecule/atom
visualization (Bohr-style atoms + orbiting electrons, animated merging into
products on reaction).

This is a **stylized, illustrative** layout generator, not a structure
predictor: real molecular geometry (bond angles, lone-pair effects, ring
structures) is a much harder problem than this app's chemistry engine takes
on elsewhere (see the README's "what's real vs. what's scoped" section). The
rule here is simple and generic so it works for *any* valid formula:

  - 1 atom            -> a single sphere.
  - 2 atoms           -> linear (matches real diatomics like O2, HCl).
  - a single non-H     -> that atom is treated as the "central" atom (this is
    element with        often chemically correct for small molecules: O in
    count 1, +          H2O, C in CH4/CO2, N in NH3, S in H2SO4) with the
    <=3 distinct         remaining atoms placed around it using a VSEPR-lite
    elements             shape picked by how many surround it (linear /
                         trigonal planar / tetrahedral / a generic even
                         sphere distribution for 5+).
  - anything else      -> a simple zig-zag chain in the formula's written
    (e.g. NaOH,          atom order. This does NOT reflect real connectivity
    C2H5OH, CaCO3)       for branched/ring molecules; it exists so the
                         visualization always has *something* reasonable to
                         show rather than requiring a curated structure per
                         compound.

Electron shell counts (for the orbiting-electron animation) are the neutral
atom's real textbook shell configuration from `elements.py`; formal charge is
not modeled.
"""
from __future__ import annotations

import math
import re
from typing import Optional

from .elements import electron_shells, element_color, element_radius
from .formula import ELEMENTS, parse_formula

_TOKEN_RE = re.compile(r"[A-Z][a-z]?|\d+|\(|\)")


def _expand_atoms(formula: str) -> Optional[list[str]]:
    """Flatten a formula into its atoms IN WRITTEN ORDER (unlike parse_formula,
    which aggregates into counts and loses order). Mirrors parse_formula's
    grammar exactly so anything parse_formula accepts, this accepts too."""
    if parse_formula(formula) is None:
        return None
    tokens = _TOKEN_RE.findall(formula)
    if "".join(tokens) != formula:
        return None

    stack: list[list[str]] = [[]]
    i, n = 0, len(tokens)
    while i < n:
        tok = tokens[i]
        if tok == "(":
            stack.append([])
            i += 1
        elif tok == ")":
            i += 1
            mult = 1
            if i < n and tokens[i].isdigit():
                mult = int(tokens[i])
                i += 1
            group = stack.pop()
            if not stack:
                return None
            stack[-1].extend(group * mult)
        elif tok.isdigit():
            return None
        else:
            if tok not in ELEMENTS:
                return None
            i += 1
            count = 1
            if i < n and tokens[i].isdigit():
                count = int(tokens[i])
                i += 1
            stack[-1].extend([tok] * count)

    if len(stack) != 1:
        return None
    return stack[0]


# Curated shape overrides for molecules where the generic "n ligands -> a
# regular n-gon/tetrahedron" rule gives a chemically wrong answer because it
# ignores lone pairs on the central atom (the classic VSEPR effect: water's 2
# lone pairs bend its H-O-H angle to ~104.5 deg instead of 180; ammonia's 1
# lone pair makes it pyramidal, not flat trigonal). Same "curate the common
# cases, fall back to a generic rule" approach this project already uses for
# reactions.json / compounds.json.
_SHAPE_OVERRIDES = {"H2O": "bent", "NH3": "pyramidal"}


def _overridden_vectors(kind: str, n: int) -> Optional[list[tuple[float, float, float]]]:
    if kind == "bent" and n == 2:
        half = math.radians(104.5) / 2
        return [(math.cos(half), math.sin(half), 0.0),
                (math.cos(half), -math.sin(half), 0.0)]
    if kind == "pyramidal" and n == 3:
        return _unit_vectors(4)[:3]   # 3 of a tetrahedron's 4 vertices
    return None


def _unit_vectors(n: int) -> list[tuple[float, float, float]]:
    """Unit vectors for `n` substituents around a central atom (VSEPR-lite)."""
    if n == 1:
        return [(1.0, 0.0, 0.0)]
    if n == 2:
        return [(1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)]
    if n == 3:
        return [(math.cos(a), math.sin(a), 0.0)
                for a in (math.tau * k / 3 for k in range(3))]
    if n == 4:
        raw = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
        return [tuple(c / math.sqrt(3) for c in v) for v in raw]
    # 5+: generic even distribution on a sphere (Fibonacci sphere)
    pts = []
    ga = math.pi * (3.0 - math.sqrt(5.0))  # golden angle
    for k in range(n):
        y = 1 - 2 * (k + 0.5) / n
        r = math.sqrt(max(0.0, 1 - y * y))
        theta = ga * k
        pts.append((r * math.cos(theta), y, r * math.sin(theta)))
    return pts


def _atom(symbol: str, x: float, y: float, z: float) -> dict:
    return {
        "element": symbol,
        "x": round(x, 4), "y": round(y, 4), "z": round(z, 4),
        "radius": element_radius(symbol),
        "color": element_color(symbol),
        "shells": electron_shells(symbol),
    }


def molecule_geometry(formula: str) -> Optional[dict]:
    """Build {'atoms': [...], 'bonds': [[i,j],...], 'shape': str} for a valid
    canonical formula, or None if the formula doesn't parse."""
    counts = parse_formula(formula)
    atoms_ordered = _expand_atoms(formula)
    if counts is None or atoms_ordered is None:
        return None

    n_atoms = len(atoms_ordered)
    if n_atoms == 0:
        return None

    if n_atoms == 1:
        el = atoms_ordered[0]
        return {"atoms": [_atom(el, 0, 0, 0)], "bonds": [], "shape": "single"}

    if n_atoms == 2:
        a, b = atoms_ordered
        d = element_radius(a) + element_radius(b) + 0.4
        return {
            "atoms": [_atom(a, -d / 2, 0, 0), _atom(b, d / 2, 0, 0)],
            "bonds": [[0, 1]], "shape": "linear",
        }

    # central-atom case: exactly one non-hydrogen element with count 1, in a
    # formula simple enough (few distinct elements) that "central + ligands"
    # is a reasonable read (H2O, CH4, CO2, NH3, H2SO4, ...).
    central_candidates = [el for el, c in counts.items() if c == 1 and el != "H"]
    if len(central_candidates) == 1 and len(counts) <= 3:
        central = central_candidates[0]
        ligands = [el for el in atoms_ordered if el != central]
        override = _SHAPE_OVERRIDES.get(formula)
        vecs = _overridden_vectors(override, len(ligands)) if override else None
        if vecs is None:
            vecs = _unit_vectors(len(ligands))
            override = {2: "linear", 3: "trigonal", 4: "tetrahedral"}.get(len(ligands), "sphere")
        atoms = [_atom(central, 0, 0, 0)]
        bonds = []
        for idx, (el, v) in enumerate(zip(ligands, vecs), start=1):
            bond_len = element_radius(central) + element_radius(el) + 0.4
            atoms.append(_atom(el, v[0] * bond_len, v[1] * bond_len, v[2] * bond_len))
            bonds.append([0, idx])
        return {"atoms": atoms, "bonds": bonds, "shape": override}

    # fallback: zig-zag chain in written order (no claim to real connectivity)
    atoms, bonds = [], []
    x = 0.0
    for idx, el in enumerate(atoms_ordered):
        y = 0.35 if idx % 2 else -0.35
        atoms.append(_atom(el, x, y if idx else 0.0, 0.0))
        if idx > 0:
            bonds.append([idx - 1, idx])
        x += element_radius(el) * 2 + 0.5
    return {"atoms": atoms, "bonds": bonds, "shape": "chain"}
