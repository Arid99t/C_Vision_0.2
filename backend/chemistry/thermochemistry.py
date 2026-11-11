"""Source and phase information for the spot-checked reaction enthalpies."""
import json
from pathlib import Path

_DATA = json.loads((Path(__file__).parent / "data" / "thermochemistry.json").read_text(encoding="utf-8"))


def describe_heat(reactants: dict, products: dict, estimate) -> dict | None:
    if estimate is None:
        return None
    record = _DATA["reactions"].get("|".join(sorted(reactants)))
    if record is None:
        return {"status": "unverified_estimate"}
    phases = record["phases"]

    def total(side):
        return sum(n * _DATA["formation_kJ_mol"][f"{f}({phases[f]})"] for f, n in side.items())

    calculated = total(products) - total(reactants)
    if abs(calculated - estimate) > 1:
        raise ValueError(f"Reaction heat {estimate} disagrees with reference calculation {calculated}")
    return {"status": "reference_checked", "calculated_kJ": round(calculated, 3),
            "temperature_K": _DATA["temperature_K"], "phases": phases,
            "sources": [_DATA["sources"][s] for s in record["sources"]],
            "checked_on": _DATA["checked_on"]}
