"""Chemistry engine: formula parsing, OCR normalization, reaction lookup, balancing."""
from .formula import normalize_ocr, parse_formula, describe_formula, to_subscripts
from .geometry import molecule_geometry
from .reactions import find_reaction
from .balancer import react, balance, format_equation

__all__ = [
    "normalize_ocr",
    "parse_formula",
    "describe_formula",
    "to_subscripts",
    "molecule_geometry",
    "find_reaction",
    "react",
    "balance",
    "format_equation",
]
