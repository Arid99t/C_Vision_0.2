"""Tests for the chemistry engine. Run: python -m pytest tests/ -q
(or just: python tests/test_chemistry.py for a dependency-free check)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.chemistry import (  # noqa: E402
    normalize_ocr,
    parse_formula,
    describe_formula,
    react,
    balance,
)


def test_parse_basic():
    assert parse_formula("H2O") == {"H": 2, "O": 1}
    assert parse_formula("CH4") == {"C": 1, "H": 4}
    assert parse_formula("CO2") == {"C": 1, "O": 2}


def test_parse_parentheses():
    assert parse_formula("Ca(OH)2") == {"Ca": 1, "O": 2, "H": 2}
    assert parse_formula("Mg(NO3)2") == {"Mg": 1, "N": 2, "O": 6}


def test_parse_invalid():
    assert parse_formula("Xz9") is None
    assert parse_formula("2H") is None      # leading count
    assert parse_formula("H2O)") is None    # unbalanced paren


def test_normalize_unicode_subscripts():
    assert normalize_ocr("H₂O") == "H2O"
    assert normalize_ocr("CH₄") == "CH4"


def test_normalize_zero_oh_confusion():
    # "H20" (zero) should become "H2O" (oxygen)
    assert normalize_ocr("H20") == "H2O"
    # but a genuine count keeps its zero
    assert normalize_ocr("C10H8") == "C10H8"


def test_normalize_whitespace():
    assert normalize_ocr("  C O 2 ") == "CO2"


def test_describe():
    d = describe_formula("H2O")
    assert d["valid"] is True
    assert d["name"] == "Water"
    assert d["pretty"] == "H₂O"


def test_balance_combustion():
    reac, prod = balance(["CH4", "O2"], ["CO2", "H2O"])
    assert reac == {"CH4": 1, "O2": 2}
    assert prod == {"CO2": 1, "H2O": 2}


def test_react_steam_reforming():
    r = react(["CH4", "H2O"])
    assert r["ok"] is True
    assert r["name"] == "Steam reforming of methane"
    # CH4 + H2O -> CO + 3 H2
    assert r["products"] == {"CO": 1, "H2": 3}


def test_react_unknown():
    r = react(["He", "Ne"])
    assert r["ok"] is False


def test_react_hydrocarbon_rule():
    # propane combustion via the generic rule (not in the DB explicitly)
    r = react(["C3H8", "O2"])
    assert r["ok"] is True
    assert r["source"] == "rule"
    assert r["products"] == {"CO2": 3, "H2O": 4}


if __name__ == "__main__":
    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in funcs:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {fn.__name__}: {exc}")
        else:
            passed += 1
            print(f"ok   {fn.__name__}")
    print(f"\n{passed}/{len(funcs)} passed")
    sys.exit(0 if passed == len(funcs) else 1)
