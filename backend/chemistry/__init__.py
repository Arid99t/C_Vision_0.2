"""Chemistry engine: formula parsing, OCR normalization, reaction lookup, balancing."""
from .formula import normalize_ocr, parse_formula, describe_formula, to_subscripts
from .reactions import find_reaction
from .balancer import react, balance, format_equation

__all__ = [
    "normalize_ocr",
    "parse_formula",
    "describe_formula",
    "to_subscripts",
    "find_reaction",
    "react",
    "balance",
    "format_equation",
]
