"""Real 3D structures for the compounds the app can show.

`geometry.py` draws stylised Bohr atoms from a formula alone. This module is
the accurate counterpart: for every compound in data/structures.json it builds
the actual connectivity and a 3D shape

  * molecules     RDKit distance-geometry embedding + force-field relaxation,
                  so water is bent, methane tetrahedral, benzene a flat ring;
  * ionic         each ion is built on its own, then the ions are packed into
    compounds     one formula unit by minimising a simple electrostatic energy
                  (opposite charges touch, like charges keep apart);
  * elements      a single atom.

and records, per atom, what the reaction animation needs: formal charge,
oxidation state and how many non-bonding valence electrons it carries.

RDKit is needed only here, at build time (python -m backend.chemistry.build3d);
the result is plain JSON that the server and the web app just read.
"""
from __future__ import annotations

import json
import math
import os
import zlib
from collections import Counter

import numpy as np

from .elements import (element_color, element_radius, electronegativity, ionic_radius,
                       valence_electrons)
from .formula import parse_formula

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
STRUCTURES_PATH = os.path.join(_DATA, "structures.json")


def load_structures() -> dict:
    with open(STRUCTURES_PATH, encoding="utf-8") as f:
        return json.load(f)["structures"]


def composition_key(counts: dict) -> str:
    """Order-independent key for a composition, e.g. {'C':2,'H':6,'O':1} -> 'C2H6O1'."""
    return "".join(f"{el}{counts[el]}" for el in sorted(counts))


# --------------------------------------------------------------------------
# single fragments (one molecule or one ion)
# --------------------------------------------------------------------------
def _embed(mol) -> np.ndarray:
    """3D coordinates (n, 3) in angstrom for one connected RDKit molecule."""
    from rdkit.Chem import AllChem

    n = mol.GetNumAtoms()
    if n == 1:
        return np.zeros((1, 3))
    params = AllChem.ETKDGv3()
    params.randomSeed = 7
    if AllChem.EmbedMolecule(mol, params) < 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) < 0:
            raise RuntimeError("3D embedding failed")
    embedded = np.array(mol.GetConformer().GetPositions(), dtype=np.float64)
    try:
        # Relax with a force field only when it has parameters for EVERY atom:
        # with a missing atom type (e.g. Cr in dichromate) UFF still runs, on
        # garbage parameters, and collapses the structure.
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
        elif AllChem.UFFHasAllMoleculeParams(mol):
            AllChem.UFFOptimizeMolecule(mol, maxIters=2000)
    except Exception:  # noqa: BLE001  keep the embedded geometry
        return embedded
    relaxed = np.array(mol.GetConformer().GetPositions(), dtype=np.float64)
    return relaxed if np.isfinite(relaxed).all() else embedded


def _pca_align(pos: np.ndarray) -> np.ndarray:
    """Centre on the centroid and turn the widest extent along x, then y."""
    pos = pos - pos.mean(axis=0)
    if len(pos) < 2:
        return pos
    _, _, vt = np.linalg.svd(pos, full_matrices=True)    # always a full 3x3 basis
    if np.linalg.det(vt) < 0:          # keep a proper rotation (no mirror image)
        vt[2] *= -1.0
    return pos @ vt.T


# --------------------------------------------------------------------------
# packing ions into a formula unit
# --------------------------------------------------------------------------
def _rot(v: np.ndarray) -> np.ndarray:
    """Rotation matrix from a rotation vector (Rodrigues)."""
    theta = float(np.linalg.norm(v))
    if theta < 1e-9:
        return np.eye(3)
    k = v / theta
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(theta) * kx + (1 - math.cos(theta)) * (kx @ kx)


def _pack(frags: list, seed: int) -> list:
    """Place rigid ions. Each frag: {'pos': (n,3), 'q': (n,), 'sig': (n,)}.
    Returns world coordinates per fragment (fragment 0 stays where it is)."""
    from scipy.optimize import minimize

    m = len(frags)
    if m == 1:
        return [frags[0]["pos"]]
    owner = np.concatenate([np.full(len(f["pos"]), i) for i, f in enumerate(frags)])
    q = np.concatenate([f["q"] for f in frags])
    sig = np.concatenate([f["sig"] for f in frags])
    iu, ju = np.triu_indices(len(owner), 1)
    keep = owner[iu] != owner[ju]
    iu, ju = iu[keep], ju[keep]
    qq = q[iu] * q[ju]
    contact = sig[iu] + sig[ju]
    rep = np.maximum(np.abs(qq), 0.25) * contact ** 8 / 9.0   # minimum at r = contact

    def world(x):
        out = [frags[0]["pos"]]
        for i in range(1, m):
            t, v = x[(i - 1) * 6:(i - 1) * 6 + 3], x[(i - 1) * 6 + 3:i * 6]
            out.append(frags[i]["pos"] @ _rot(v).T + t)
        return out

    def energy(x):
        p = np.concatenate(world(x))
        r = np.maximum(np.linalg.norm(p[iu] - p[ju], axis=1), 0.3)
        centres = x.reshape(m - 1, 6)[:, :3]
        return float((qq / r + rep / r ** 9).sum() + 1e-3 * (centres ** 2).sum())

    rng = np.random.default_rng(seed)
    best, best_x = None, None
    for _ in range(8 if m > 4 else 16):       # random restarts; keep the lowest energy
        x0 = np.zeros((m - 1, 6))
        d = rng.normal(size=(m - 1, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        x0[:, :3] = d * rng.uniform(2.5, 6.0, size=(m - 1, 1))
        x0[:, 3:] = rng.normal(size=(m - 1, 3))
        res = minimize(energy, x0.ravel(), method="L-BFGS-B")
        if best is None or res.fun < best - 1e-6:
            best, best_x = res.fun, res.x
    return world(best_x)


# --------------------------------------------------------------------------
# one library entry -> template
# --------------------------------------------------------------------------
def build_template(formula: str, entry: dict) -> dict:
    """Build the 3D template of one compound. Raises ValueError with a clear
    message if the SMILES is invalid or does not match the formula."""
    from rdkit import Chem

    counts = parse_formula(formula)
    if counts is None:
        raise ValueError(f"{formula}: not a parseable formula")
    mol = Chem.MolFromSmiles(entry["smiles"])
    if mol is None:
        raise ValueError(f"{formula}: invalid SMILES {entry['smiles']!r}")
    mol = Chem.AddHs(mol)
    Chem.Kekulize(mol, clearAromaticFlags=True)          # integer bond orders
    have = Counter(a.GetSymbol() for a in mol.GetAtoms())
    if dict(have) != counts:
        raise ValueError(f"{formula}: SMILES {entry['smiles']!r} has composition {dict(have)}")

    n = mol.GetNumAtoms()
    els = [a.GetSymbol() for a in mol.GetAtoms()]
    fc = [a.GetFormalCharge() for a in mol.GetAtoms()]
    bonds = sorted((min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                    max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                    int(round(b.GetBondTypeAsDouble()))) for b in mol.GetBonds())
    neighbours = [[] for _ in range(n)]
    for i, j, order in bonds:
        neighbours[i].append((j, order))
        neighbours[j].append((i, order))

    frag_idx = Chem.GetMolFrags(mol)
    frag_mols = Chem.GetMolFrags(mol, asMols=True)
    frag_of = [0] * n
    frags = []
    for fi, (idx, fm) in enumerate(zip(frag_idx, frag_mols)):
        pos = _embed(fm)
        pos = pos - pos.mean(axis=0)
        net = sum(fc[i] for i in idx)
        pack_q = np.array([float(fc[i]) for i in idx])
        if len(idx) > 1 and net < 0:
            # spread an anion's charge over its terminal O/S atoms (resonance)
            terminal = [k for k, i in enumerate(idx)
                        if els[i] in ("O", "S")
                        and sum(1 for j, _ in neighbours[i] if els[j] != "H") == 1
                        and not any(els[j] == "H" for j, _ in neighbours[i])]
            if terminal:
                pack_q[:] = 0.0
                pack_q[terminal] = net / len(terminal)
        if len(idx) == 1:
            sig = np.array([ionic_radius(els[idx[0]], fc[idx[0]]) if fc[idx[0]]
                            else element_radius(els[idx[0]])])
        else:
            sig = np.array([{"H": 0.6, "O": 1.30, "N": 1.50}.get(els[i], 1.70) for i in idx])
        frags.append({"pos": pos, "q": pack_q, "sig": sig, "idx": idx})
        for i in idx:
            frag_of[i] = fi

    # the biggest / most highly charged ion goes in the middle
    frags.sort(key=lambda f: (-len(f["idx"]), -abs(float(f["q"].sum()))))
    placed = _pack(frags, seed=zlib.crc32(formula.encode()))
    pos = np.zeros((n, 3))
    for f, world in zip(frags, placed):
        pos[list(f["idx"])] = world
    pos = _pca_align(pos)

    # electrostatic contacts between ions, drawn as dashed links
    ionic = []
    if len(frags) > 1:
        pq, ps = np.zeros(n), np.zeros(n)
        for f in frags:
            pq[list(f["idx"])] = f["q"]
            ps[list(f["idx"])] = f["sig"]
        for i in range(n):
            for j in range(i + 1, n):
                if frag_of[i] != frag_of[j] and pq[i] * pq[j] < 0:
                    if np.linalg.norm(pos[i] - pos[j]) <= 1.25 * (ps[i] + ps[j]):
                        ionic.append([i, j])

    atoms, lone, ox = [], [], []
    for i in range(n):
        order_sum = sum(o for _, o in neighbours[i])
        lone.append(max(0, valence_electrons(els[i]) - fc[i] - order_sum))
        state = fc[i]
        for j, o in neighbours[i]:
            diff = electronegativity(els[j]) - electronegativity(els[i])
            if els[j] != els[i] and abs(diff) > 1e-9:
                state += o if diff > 0 else -o
        ox.append(state)
        if not neighbours[i] and fc[i]:
            radius = 0.10 + 0.5 * ionic_radius(els[i], fc[i])       # monatomic ion
        elif not neighbours[i]:
            radius = 0.15 + 0.42 * element_radius(els[i])           # lone neutral atom
        else:
            radius = 0.26 + 0.24 * element_radius(els[i])           # ball-and-stick
        atoms.append([els[i], round(float(pos[i, 0]), 3), round(float(pos[i, 1]), 3),
                      round(float(pos[i, 2]), 3), fc[i], round(radius, 3)])

    extent = max(float(np.linalg.norm(pos[i])) + atoms[i][5] for i in range(n))
    kind = "atom" if n == 1 else ("ionic" if len(frags) > 1 else "molecule")
    return {"formula": formula, "name": entry.get("name"), "note": entry.get("note"),
            "kind": kind, "atoms": atoms, "bonds": [list(b) for b in bonds], "ionic": ionic,
            "frag": frag_of, "lone": lone, "ox": ox, "radius": round(extent, 3)}


def check_template(t: dict) -> list:
    """Sanity problems with a built template (empty list = fine)."""
    problems = []
    pos = np.array([a[1:4] for a in t["atoms"]])
    n = len(pos)
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(pos[i] - pos[j]) < 0.55:
                problems.append(f"atoms {i} and {j} overlap")
    for i, j, _ in t["bonds"]:
        d = float(np.linalg.norm(pos[i] - pos[j]))
        if not 0.6 <= d <= 3.2:
            problems.append(f"bond {i}-{j} has length {d:.2f} A")
    if sum(a[4] for a in t["atoms"]) != 0:
        problems.append("net charge is not zero")
    return problems


def build_library(verbose: bool = False) -> dict:
    """{'molecules': {formula: template}, 'aliases': {written form: formula},
    'by_composition': {composition key: formula}, 'elements': {...}}"""
    structures = load_structures()
    molecules, aliases, by_comp = {}, {}, {}
    for formula, entry in structures.items():
        t = build_template(formula, entry)
        problems = check_template(t)
        if problems:
            raise ValueError(f"{formula}: " + "; ".join(problems))
        molecules[formula] = t
        by_comp.setdefault(composition_key(parse_formula(formula)), formula)
        for alias in entry.get("aliases", []):
            if parse_formula(alias) != parse_formula(formula):
                raise ValueError(f"{formula}: alias {alias} has a different composition")
            aliases[alias] = formula
        if verbose:
            print(f"  {formula:<12} {t['kind']:<9} {len(t['atoms']):>3} atoms  {t['name']}")
    used = sorted({a[0] for t in molecules.values() for a in t["atoms"]})
    elements = {el: {"color": element_color(el), "en": electronegativity(el)} for el in used}
    return {"molecules": molecules, "aliases": aliases, "by_composition": by_comp,
            "elements": elements}
