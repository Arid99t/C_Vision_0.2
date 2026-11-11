"""Regression checks for malformed input, API errors and heat provenance."""
import pytest
from fastapi.testclient import TestClient

from backend.app import app
from backend.chemistry import parse_formula, react
from backend.chemistry.reactions import _TABLE
from backend.chemistry.thermochemistry import _DATA

INVALID = ["()", "H()", "(())", "H0", "H01", "(OH)0", "H٢", "H10001",
           "(C100H100)100", "H" * 257, "H" + "9" * 5000, ")", "((H)"]


def test_invalid_formula_parity(page):
    assert all(parse_formula(f) is None for f in INVALID)
    parsed = page.evaluate("""async formulas => {
        const {loadChem} = await import('/static/js/chem.js');
        const chem = await loadChem();
        return formulas.map(f => chem.parseFormula(f));
    }""", INVALID)
    assert parsed == [None] * len(INVALID)
    assert parse_formula("K4(ON(SO3)2)2") == {"K": 4, "O": 14, "N": 2, "S": 4}


def test_duplicate_oxygen_and_oxygen_rich_fuel():
    assert react(["O2", "O2"])["ok"] is False
    assert react(["CH2O9", "O2"])["ok"] is False
    result = react(["C9H20", "O2", "O2"])
    assert result["ok"] is True
    assert result["reactants"] == {"C9H20": 1, "O2": 14}


@pytest.mark.parametrize("reactants,products", [([], ["H2O"]), (["H2"], []),
    (["()"], ["H2O"]), (["H0"], ["H2"]), (["H2", "O2"], ["H2"])])
def test_bad_balance_is_client_error(reactants, products):
    with TestClient(app) as client:
        response = client.post("/balance", json={"reactants": reactants, "products": products})
        assert response.status_code == 422, response.text


def test_checked_heats_and_estimates_are_distinct():
    for key in _DATA["reactions"]:
        result = react(key.split("|"))
        heat = result["thermochemistry"]
        assert result["ok"] and heat["status"] == "reference_checked"
        assert heat["sources"] and heat["temperature_K"] == 298.15
        assert abs(result["dH"] - heat["calculated_kJ"]) <= 1
    assert react(["CaCO3"])["thermochemistry"]["status"] == "unverified_estimate"
    # Every curated reaction must still balance after stricter validation.
    for record in _TABLE.values():
        assert react(record["reactants"])["ok"], record
