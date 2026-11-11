"""Tests for the real-structure 3D library and the reaction animation plans.

Reads the built bundle (backend/static/data/chem3d.json), so RDKit is not
needed to run them; rebuild the bundle with
    .venv\\Scripts\\python.exe -m backend.chemistry.build3d
Run: python tests/test_chem3d.py
"""
import json
import math
import os
import sys
import warnings
from collections import Counter

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from backend.chemistry import parse_formula, react  # noqa: E402
from backend.chemistry import reaction3d  # noqa: E402
from backend.chemistry.elements import valence_electrons  # noqa: E402

BUNDLE_PATH = os.path.join(ROOT, "backend", "static", "data", "chem3d.json")
if not os.path.exists(BUNDLE_PATH):
    print("SKIP: chem3d.json not built (python -m backend.chemistry.build3d)")
    from tests.webharness import skip_unavailable
    skip_unavailable("Optional dependency or generated asset unavailable; see message above")
with open(BUNDLE_PATH, encoding="utf-8") as f:
    B = json.load(f)
MOLS, RX = B["molecules"], B["reactions"]


def _pos(t):
    return np.array([a[1:4] for a in t["atoms"]], float)


def _angle(t, i, j, k):
    p = _pos(t)
    u, v = p[i] - p[j], p[k] - p[j]
    return math.degrees(math.acos(float(u @ v) / (np.linalg.norm(u) * np.linalg.norm(v))))


def test_library_is_consistent():
    for formula, t in MOLS.items():
        counts = Counter(a[0] for a in t["atoms"])
        assert dict(counts) == parse_formula(formula), f"{formula}: wrong composition"
        assert sum(a[4] for a in t["atoms"]) == 0, f"{formula}: not neutral"
        n = len(t["atoms"])
        assert all(0 <= i < n and 0 <= j < n and o in (1, 2, 3) for i, j, o in t["bonds"])
        # every valence electron is accounted for: lone ones + two per bond order
        shown = sum(t["lone"]) + 2 * sum(o for _, _, o in t["bonds"])
        owned = sum(valence_electrons(a[0]) - a[4] for a in t["atoms"])
        assert shown == owned, f"{formula}: {shown} electrons drawn, {owned} expected"
        assert sum(t["ox"]) == 0, f"{formula}: oxidation states do not sum to zero"
        p = _pos(t)
        for i in range(n):
            for j in range(i + 1, n):
                assert np.linalg.norm(p[i] - p[j]) > 0.55, f"{formula}: atoms {i},{j} overlap"
    print(f"ok   {len(MOLS)} structures: composition, charge, electron count, no overlaps")


def test_textbook_shapes():
    w = MOLS["H2O"]
    o = [i for i, a in enumerate(w["atoms"]) if a[0] == "O"][0]
    h = [i for i, a in enumerate(w["atoms"]) if a[0] == "H"]
    assert 98 < _angle(w, h[0], o, h[1]) < 112, "water should be bent (~104.5 deg)"

    c = MOLS["CO2"]
    ci = [i for i, a in enumerate(c["atoms"]) if a[0] == "C"][0]
    oi = [i for i, a in enumerate(c["atoms"]) if a[0] == "O"]
    assert _angle(c, oi[0], ci, oi[1]) > 172, "carbon dioxide should be linear"
    assert sorted(o for _, _, o in c["bonds"]) == [2, 2], "CO2 has two double bonds"

    m = MOLS["CH4"]
    ci = [i for i, a in enumerate(m["atoms"]) if a[0] == "C"][0]
    hs = [i for i, a in enumerate(m["atoms"]) if a[0] == "H"]
    angles = [_angle(m, hs[i], ci, hs[j]) for i in range(4) for j in range(i + 1, 4)]
    assert all(103 < a < 116 for a in angles), f"methane should be tetrahedral: {angles}"

    a3 = MOLS["NH3"]
    ni = [i for i, a in enumerate(a3["atoms"]) if a[0] == "N"][0]
    hs = [i for i, a in enumerate(a3["atoms"]) if a[0] == "H"]
    assert 100 < _angle(a3, hs[0], ni, hs[1]) < 114, "ammonia should be pyramidal"
    assert a3["lone"][ni] == 2, "ammonia's nitrogen carries one lone pair"

    bz = _pos(MOLS["C6H6"])
    assert np.linalg.svd(bz - bz.mean(0), compute_uv=False)[-1] < 0.2, "benzene should be flat"

    assert MOLS["NaCl"]["kind"] == "ionic" and len(MOLS["NaCl"]["ionic"]) == 1
    charges = sorted(a[4] for a in MOLS["NaCl"]["atoms"])
    assert charges == [-1, 1], "NaCl should be Na+ and Cl-"
    assert MOLS["O2"]["ox"] == [0, 0] and sorted(MOLS["H2O"]["ox"]) == [-2, 1, 1]
    print("ok   textbook shapes: bent water, linear CO2, tetrahedral CH4, pyramidal NH3, flat benzene")


def test_every_plan_conserves_atoms_and_electrons():
    planned = 0
    for key, r in RX.items():
        p = r.get("plan")
        if not p:
            continue
        planned += 1
        n = len(p["atoms"])
        want = Counter()
        for f, c in p["reactants"]:
            for el, k in parse_formula(f).items():
                want[el] += k * c
        made = Counter()
        for f, c in p["products"]:
            for el, k in parse_formula(f).items():
                made[el] += k * c
        assert want == made, f"{key}: equation is not balanced"
        assert Counter(a[0] for a in p["atoms"]) == want, f"{key}: atoms do not match the equation"
        for a, b, o in p["bonds_r"] + p["bonds_p"]:
            assert 0 <= a < b < n and o in (1, 2, 3), f"{key}: bad bond {a}-{b}"
        assert p["stats"]["unmatched"] == 0, f"{key}: electrons not conserved"
        # every electron leaves a real place and arrives at a real place, once
        src = Counter(tuple(m[0:4]) for m in p["electrons"])
        dst = Counter(tuple(m[4:8]) for m in p["electrons"])
        assert max(src.values()) == 1 and max(dst.values()) == 1, f"{key}: electron used twice"
        total = sum(valence_electrons(a[0]) for a in p["atoms"])
        assert len(p["electrons"]) == total, f"{key}: {len(p['electrons'])} electrons tracked, {total} exist"
        assert sum(a[13] for a in p["atoms"]) == sum(a[14] for a in p["atoms"]) == 0, \
            f"{key}: oxidation states are not conserved"
        assert sum(a[7] for a in p["atoms"]) == sum(a[8] for a in p["atoms"]) == 0, f"{key}: charge"
    print(f"ok   {planned} reaction plans conserve atoms, charge, electrons and oxidation number")


def test_known_reactions_read_right():
    p = RX["HCl|NaOH"]["plan"]
    assert p["stats"]["bonds_broken"] == 1 and p["stats"]["bonds_formed"] == 1
    assert p["stats"]["transferred"] == 0 and not p["redox"], "neutralization is not redox"

    p = RX["Cl2|Na"]["plan"]
    assert p["stats"]["transferred"] == 2, "each sodium hands one electron to chlorine"
    assert {(r["el"], r["from"], r["to"]) for r in p["redox"]} == {("Na", 0, 1), ("Cl", 0, -1)}

    p = RX["CH4|O2"]["plan"]
    assert {(r["el"], r["from"], r["to"]) for r in p["redox"]} == {("C", -4, 4), ("O", 0, -2)}
    assert p["stats"]["bonds_broken"] == 8 and p["stats"]["bonds_formed"] == 8

    p = RX["Fe|O2"]["plan"]
    assert p["stats"]["transferred"] == 12, "4 Fe -> 4 Fe3+ releases 12 electrons"

    # the O-H bond of ethanol's hydroxyl survives hydration in reverse: C2H4 + H2O
    p = RX["C2H4|H2O"]["plan"]
    assert p["stats"]["bonds_broken"] <= 2, "hydration should break only one O-H and one C=C pi bond"
    print("ok   neutralization, salt formation, combustion and rusting are described correctly")


def test_planner_runs_live():
    """Plan a reaction from scratch with the library in the bundle (no RDKit)."""
    res = react(["N2", "H2"])
    plan = reaction3d.plan_reaction(list(res["reactants"].items()), list(res["products"].items()),
                                    MOLS, seed=1)
    assert plan["stats"]["atoms"] == 8 and plan["stats"]["unmatched"] == 0
    assert {(r["el"], r["from"], r["to"]) for r in plan["redox"]} == {("N", 0, -3), ("H", 0, 1)}
    start = np.array([a[1:4] for a in plan["atoms"]])
    end = np.array([a[4:7] for a in plan["atoms"]])
    for pts in (start, end):                      # no two atoms land on the same spot
        d = np.linalg.norm(pts[:, None] - pts[None], axis=2) + np.eye(len(pts)) * 9
        assert d.min() > 0.5
    print("ok   planner builds a fresh plan (Haber process) from the library")


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
