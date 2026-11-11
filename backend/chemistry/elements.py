"""Periodic-table data for the 3D atom visualization: atomic number, CPK/Jmol
display color, a relative covalent radius, and electron-shell occupancy.

This is display data, not a chemistry engine — it feeds the Bohr-style atom
models (nucleus + electron shells) the AR clients render, not any reaction
logic. Two simplifications, both standard for teaching visualizations:

  1. Shell counts are for the *neutral* atom (ignores ionization/charge), and
     are the classic "2, 8, 8, 18, 18, 32" teaching model, not full quantum
     subshell notation. Elements 1-54 (H through Xe) use their real, textbook
     shell configuration (including the well-known Cr/Cu-style exceptions);
     anything heavier falls back to filling shell capacities in order, which
     is an approximation (it misses the real Aufbau exceptions) but those
     elements never appear in this app's chemistry anyway.
  2. Colors/radii are the standard Jmol/CPK convention (public scientific
     convention, not this project's invention). Elements without a specific
     entry get a neutral gray placeholder.
"""
from __future__ import annotations

# Same 118 symbols, in atomic-number order, as backend.chemistry.formula.ELEMENTS
# (kept as an explicit ordered list here since atomic number = position + 1).
_ORDERED_SYMBOLS = [
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
]
ATOMIC_NUMBER = {sym: i + 1 for i, sym in enumerate(_ORDERED_SYMBOLS)}

# Real shell occupancy (K,L,M,N,O,P,...) for elements 1-54, textbook values
# (includes the Cr/Cu/Nb/Mo/Ru/Rh-style early-shift exceptions).
_REAL_SHELLS = {
    "H": [1], "He": [2], "Li": [2, 1], "Be": [2, 2], "B": [2, 3], "C": [2, 4],
    "N": [2, 5], "O": [2, 6], "F": [2, 7], "Ne": [2, 8], "Na": [2, 8, 1],
    "Mg": [2, 8, 2], "Al": [2, 8, 3], "Si": [2, 8, 4], "P": [2, 8, 5],
    "S": [2, 8, 6], "Cl": [2, 8, 7], "Ar": [2, 8, 8], "K": [2, 8, 8, 1],
    "Ca": [2, 8, 8, 2], "Sc": [2, 8, 9, 2], "Ti": [2, 8, 10, 2],
    "V": [2, 8, 11, 2], "Cr": [2, 8, 13, 1], "Mn": [2, 8, 13, 2],
    "Fe": [2, 8, 14, 2], "Co": [2, 8, 15, 2], "Ni": [2, 8, 16, 2],
    "Cu": [2, 8, 18, 1], "Zn": [2, 8, 18, 2], "Ga": [2, 8, 18, 3],
    "Ge": [2, 8, 18, 4], "As": [2, 8, 18, 5], "Se": [2, 8, 18, 6],
    "Br": [2, 8, 18, 7], "Kr": [2, 8, 18, 8], "Rb": [2, 8, 18, 8, 1],
    "Sr": [2, 8, 18, 8, 2], "Y": [2, 8, 18, 9, 2], "Zr": [2, 8, 18, 10, 2],
    "Nb": [2, 8, 18, 12, 1], "Mo": [2, 8, 18, 13, 1], "Tc": [2, 8, 18, 13, 2],
    "Ru": [2, 8, 18, 15, 1], "Rh": [2, 8, 18, 16, 1], "Pd": [2, 8, 18, 18],
    "Ag": [2, 8, 18, 18, 1], "Cd": [2, 8, 18, 18, 2], "In": [2, 8, 18, 18, 3],
    "Sn": [2, 8, 18, 18, 4], "Sb": [2, 8, 18, 18, 5], "Te": [2, 8, 18, 18, 6],
    "I": [2, 8, 18, 18, 7], "Xe": [2, 8, 18, 18, 8],
}
_GENERIC_CAPACITIES = [2, 8, 8, 18, 18, 32, 32, 32]  # fallback for Z > 54


def electron_shells(symbol: str) -> list[int]:
    """Electrons per shell for the neutral atom, outermost shell last."""
    if symbol in _REAL_SHELLS:
        return list(_REAL_SHELLS[symbol])
    z = ATOMIC_NUMBER.get(symbol)
    if z is None:
        return [2]  # unknown symbol: shouldn't happen if formula validated first
    shells, remaining = [], z
    for cap in _GENERIC_CAPACITIES:
        if remaining <= 0:
            break
        take = min(cap, remaining)
        shells.append(take)
        remaining -= take
    return shells or [z]


# Jmol/CPK element colors (hex strings, "#rrggbb") for the elements this app's
# chemistry actually uses; anything else gets NEUTRAL_COLOR.
NEUTRAL_COLOR = "#c8c8c8"
_COLORS = {
    "H": "#ffffff", "He": "#d9ffff", "Li": "#cc80ff", "Be": "#c2ff00",
    "B": "#ffb5b5", "C": "#909090", "N": "#3050f8", "O": "#ff0d0d",
    "F": "#90e050", "Ne": "#b3e3f5", "Na": "#ab5cf2", "Mg": "#8aff00",
    "Al": "#bfa6a6", "Si": "#f0c8a0", "P": "#ff8000", "S": "#ffff30",
    "Cl": "#1ff01f", "Ar": "#80d1e3", "K": "#8f40d4", "Ca": "#3dff00",
    "Mn": "#9c7ac7", "Fe": "#e06633", "Co": "#f090a0", "Ni": "#50d050",
    "Cu": "#c88033", "Zn": "#7d80b0", "Br": "#a62929", "I": "#940094",
    "Ba": "#00c900", "Ag": "#c0c0c0", "Au": "#ffd123", "Hg": "#b8b8d0",
    "Pb": "#575961", "Ti": "#bfc2c7", "Cr": "#8a99c7", "Sn": "#668080",
}

# Relative covalent radius (angstrom-ish units; only the relative scale
# matters since the renderer applies its own scale factor).
_RADII = {
    "H": 0.32, "He": 0.46, "Li": 1.33, "Be": 1.02, "B": 0.85, "C": 0.75,
    "N": 0.71, "O": 0.66, "F": 0.64, "Ne": 0.67, "Na": 1.55, "Mg": 1.39,
    "Al": 1.26, "Si": 1.16, "P": 1.11, "S": 1.03, "Cl": 0.99, "Ar": 0.96,
    "K": 1.96, "Ca": 1.71, "Mn": 1.39, "Fe": 1.25, "Co": 1.26, "Ni": 1.21,
    "Cu": 1.38, "Zn": 1.31, "Br": 1.14, "I": 1.33, "Ba": 1.98, "Ag": 1.53,
    "Au": 1.44, "Hg": 1.49, "Pb": 1.46, "Ti": 1.60, "Cr": 1.39, "Sn": 1.39,
}
_DEFAULT_RADIUS = 1.10


def element_color(symbol: str) -> str:
    return _COLORS.get(symbol, NEUTRAL_COLOR)


def element_radius(symbol: str) -> float:
    return _RADII.get(symbol, _DEFAULT_RADIUS)


# --------------------------------------------------------------------------
# Data for the real-structure 3D views and reaction animations
# (mol3d.py / reaction3d.py). Textbook values.
# --------------------------------------------------------------------------

# Pauling electronegativity. Decides which atom "owns" the electrons of a bond
# (oxidation states) and which way a shared pair is drawn leaning.
ELECTRONEGATIVITY = {
    "H": 2.20, "Li": 0.98, "Be": 1.57, "B": 2.04, "C": 2.55, "N": 3.04, "O": 3.44,
    "F": 3.98, "Na": 0.93, "Mg": 1.31, "Al": 1.61, "Si": 1.90, "P": 2.19, "S": 2.58,
    "Cl": 3.16, "K": 0.82, "Ca": 1.00, "Ti": 1.54, "Cr": 1.66, "Mn": 1.55, "Fe": 1.83,
    "Co": 1.88, "Ni": 1.91, "Cu": 1.90, "Zn": 1.65, "Br": 2.96, "Ag": 1.93, "Sn": 1.96,
    "I": 2.66, "Ba": 0.89, "Au": 2.54, "Hg": 2.00, "Pb": 2.33,
}

# Valence electrons drawn as dots. Main-group elements: the group number. For
# the transition metals the app uses, the electrons they give up in their
# highest oxidation state that appears in the library (a display convention:
# "the electrons that take part in bonding").
VALENCE_ELECTRONS = {
    "H": 1, "He": 2, "Li": 1, "Be": 2, "B": 3, "C": 4, "N": 5, "O": 6, "F": 7, "Ne": 8,
    "Na": 1, "Mg": 2, "Al": 3, "Si": 4, "P": 5, "S": 6, "Cl": 7, "Ar": 8,
    "K": 1, "Ca": 2, "Br": 7, "I": 7, "Ba": 2, "Sn": 4, "Pb": 4,
    "Ti": 4, "Cr": 6, "Mn": 7, "Fe": 3, "Co": 3, "Ni": 2, "Cu": 2, "Zn": 2,
    "Ag": 1, "Au": 3, "Hg": 2,
}

# Shannon ionic radii (angstrom) for the common charge of each monatomic ion.
IONIC_RADIUS = {
    ("Li", 1): 0.76, ("Na", 1): 1.02, ("K", 1): 1.38, ("Ag", 1): 1.15, ("Cu", 1): 0.77,
    ("Mg", 2): 0.72, ("Ca", 2): 1.00, ("Ba", 2): 1.35, ("Zn", 2): 0.74, ("Cu", 2): 0.73,
    ("Fe", 2): 0.78, ("Pb", 2): 1.19, ("Sn", 2): 1.18, ("Hg", 2): 1.02, ("Ni", 2): 0.69,
    ("Co", 2): 0.75, ("Mn", 2): 0.83,
    ("Al", 3): 0.54, ("Fe", 3): 0.65, ("Cr", 3): 0.62, ("Co", 3): 0.61, ("Au", 3): 0.85,
    ("Mn", 4): 0.53, ("Pb", 4): 0.78, ("Ti", 4): 0.61, ("Sn", 4): 0.69,
    ("F", -1): 1.33, ("Cl", -1): 1.81, ("Br", -1): 1.96, ("I", -1): 2.20,
    ("O", -2): 1.40, ("S", -2): 1.84,
}


def electronegativity(symbol: str) -> float:
    return ELECTRONEGATIVITY.get(symbol, 1.8)


def valence_electrons(symbol: str) -> int:
    return VALENCE_ELECTRONS.get(symbol, 2)


def ionic_radius(symbol: str, charge: int) -> float:
    """Radius of a monatomic ion; falls back to a scaled covalent radius."""
    if (symbol, charge) in IONIC_RADIUS:
        return IONIC_RADIUS[(symbol, charge)]
    return element_radius(symbol) * (0.6 if charge > 0 else 1.5)
