"""Tests for the 3D molecule geometry generator (backend/chemistry/geometry.py).
Run: python tests/test_geometry.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.chemistry.elements import ATOMIC_NUMBER, electron_shells  # noqa: E402
from backend.chemistry.geometry import molecule_geometry  # noqa: E402


def _dist(a, b):
    return math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2 + (a["z"] - b["z"]) ** 2)


def test_invalid_formula_returns_none():
    assert molecule_geometry("Xz9") is None
    assert molecule_geometry("") is None
    print("ok   invalid formulas return None")


def test_single_atom():
    g = molecule_geometry("Fe")
    assert len(g["atoms"]) == 1 and g["bonds"] == [] and g["shape"] == "single"
    print("ok   single atom -> one atom, no bonds")


def test_diatomic_linear():
    g = molecule_geometry("O2")
    assert g["shape"] == "linear"
    assert len(g["atoms"]) == 2 and g["bonds"] == [[0, 1]]
    assert g["atoms"][0]["x"] != g["atoms"][1]["x"]   # actually separated
    print("ok   diatomic (O2) is linear with one bond")


def test_water_is_bent_not_linear():
    g = molecule_geometry("H2O")
    assert g["shape"] == "bent"
    o = g["atoms"][0]
    h1, h2 = g["atoms"][1], g["atoms"][2]
    assert o["element"] == "O" and h1["element"] == h2["element"] == "H"
    # angle H-O-H should be near the real ~104.5 deg, and NOT ~180 (linear)
    v1 = (h1["x"] - o["x"], h1["y"] - o["y"], h1["z"] - o["z"])
    v2 = (h2["x"] - o["x"], h2["y"] - o["y"], h2["z"] - o["z"])
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(a * a for a in v2))
    angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, dot / (n1 * n2)))))
    assert 95 < angle_deg < 115, f"H-O-H angle {angle_deg} not bent"
    print(f"ok   water is bent (H-O-H = {angle_deg:.1f} deg), not linear")


def test_ammonia_is_pyramidal():
    g = molecule_geometry("NH3")
    assert g["shape"] == "pyramidal"
    assert len(g["atoms"]) == 4 and len(g["bonds"]) == 3
    # the 3 H atoms must NOT be coplanar with N at the origin (i.e. not flat
    # trigonal planar) -> the tetrahedron-vertex subset has nonzero z spread
    zs = {round(a["z"], 3) for a in g["atoms"][1:]}
    assert len(zs) > 1, "ammonia H atoms are coplanar (flat), should be pyramidal"
    print("ok   ammonia is pyramidal (not flat trigonal)")


def test_methane_tetrahedral_equal_bond_lengths():
    g = molecule_geometry("CH4")
    assert g["shape"] == "tetrahedral"
    c = g["atoms"][0]
    dists = [_dist(c, h) for h in g["atoms"][1:]]
    assert len(g["atoms"]) == 5 and len(g["bonds"]) == 4
    assert max(dists) - min(dists) < 1e-6, "C-H bond lengths should all be equal"
    print("ok   methane is tetrahedral with 4 equal C-H bonds")


def test_co2_linear():
    g = molecule_geometry("CO2")
    assert g["shape"] == "linear"
    c, o1, o2 = g["atoms"]
    # O-C-O angle should be ~180
    v1 = (o1["x"] - c["x"], o1["y"] - c["y"])
    v2 = (o2["x"] - c["x"], o2["y"] - c["y"])
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, dot / (n1 * n2)))))
    assert angle_deg > 170, f"O-C-O angle {angle_deg} should be ~180"
    print("ok   CO2 is linear (O-C-O ~180 deg)")


def test_chain_fallback_used_for_complex_molecules():
    g = molecule_geometry("NaOH")
    assert g["shape"] == "chain"
    assert len(g["atoms"]) == 3 and len(g["bonds"]) == 2
    g2 = molecule_geometry("C2H5OH")
    assert g2["shape"] in ("chain", "sphere")   # not claiming real ethanol structure
    print("ok   complex/multi-element formulas fall back to a chain layout")


def test_all_atoms_have_valid_shell_and_color_data():
    for formula in ["H2SO4", "CaCO3", "Na2SO4", "C6H12O6", "ZnCl2", "Fe2O3"]:
        g = molecule_geometry(formula)
        assert g is not None, formula
        for atom in g["atoms"]:
            assert atom["element"] in ATOMIC_NUMBER
            assert isinstance(atom["shells"], list) and sum(atom["shells"]) > 0
            assert atom["color"].startswith("#") and len(atom["color"]) == 7
            assert atom["radius"] > 0
    print("ok   every atom across varied formulas has valid shell/color/radius data")


def test_shell_counts_match_atomic_number_for_light_elements():
    # sanity: for elements 1-54 (real table), shells sum to atomic number
    # (neutral-atom electron count == atomic number)
    for sym in ["H", "C", "N", "O", "Na", "Cl", "Ca", "Fe", "Cu", "I"]:
        z = ATOMIC_NUMBER[sym]
        assert sum(electron_shells(sym)) == z, f"{sym}: shells don't sum to Z={z}"
    print("ok   real-table shell counts sum to the atomic number")


if __name__ == "__main__":
    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in funcs:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {fn.__name__}: {exc!r}")
        else:
            passed += 1
    print(f"\n{passed}/{len(funcs)} passed")
    sys.exit(0 if passed == len(funcs) else 1)
