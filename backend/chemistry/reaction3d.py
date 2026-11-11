"""Plan the 3D animation of a balanced reaction.

Given the reactant and product molecules (3D templates from mol3d.py) and
their coefficients, work out everything the web app needs to animate it:

  1. ATOM MAP. Every reactant atom is assigned the product atom it becomes.
     Among all element-preserving assignments we search for one that changes
     as few bonds as possible (the "minimum chemical distance" principle), so
     an O-H group that survives the reaction is shown surviving.
  2. BONDS. From the map: which bonds break, which form, which change order.
  3. ELECTRONS. Every valence electron is tracked from where it sits before
     (a lone electron on an atom, or one of the electrons of a bond) to where
     it sits after. Electrons stay with an atom they already belonged to
     whenever possible; the ones that cannot are genuine transfers between
     atoms (e.g. Na -> Cl) and are flagged so the app can highlight them.
  4. OXIDATION STATES before and after, for the summary line.
  5. LAYOUT. Where each molecule sits at the start and at the end, with
     product molecules turned so atoms travel as little as possible.

This is a schematic of WHAT changes, not a step-by-step reaction mechanism:
real mechanisms go through intermediates this does not model.

Pure numpy/scipy; no RDKit.
"""
from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment

GAP = 1.1          # empty space between neighbouring molecules, angstrom
BOND_SYMBOL = {1: "–", 2: "=", 3: "≡"}
SUPERSCRIPT = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")


class Side:
    """One side of the equation expanded into individual molecules and atoms."""

    def __init__(self, species: list, library: dict):
        self.species = species                      # [(formula, coefficient)]
        self.mols = []                              # [{'formula', 'atoms': [idx], 'radius'}]
        self.el, self.fc, self.lone, self.ox, self.rad = [], [], [], [], []
        self.local = []                             # template coordinates per atom
        self.mol_of = []
        self.bonds = {}                             # {(a, b): order}, a < b
        self.ionic = []
        for formula, coef in species:
            t = library[formula]
            for _ in range(int(coef)):
                base = len(self.el)
                self.mols.append({"formula": formula, "radius": t["radius"],
                                  "atoms": list(range(base, base + len(t["atoms"])))})
                for k, a in enumerate(t["atoms"]):
                    self.el.append(a[0])
                    self.local.append(np.array(a[1:4], dtype=np.float64))
                    self.fc.append(a[4])
                    self.rad.append(a[5])
                    self.lone.append(t["lone"][k])
                    self.ox.append(t["ox"][k])
                    self.mol_of.append(len(self.mols) - 1)
                for i, j, order in t["bonds"]:
                    self.bonds[(base + i, base + j)] = order
                for i, j in t["ionic"]:
                    self.ionic.append((base + i, base + j))
        self.n = len(self.el)
        self.adj = [dict() for _ in range(self.n)]
        for (a, b), order in self.bonds.items():
            self.adj[a][b] = order
            self.adj[b][a] = order


# --------------------------------------------------------------------------
# 1. atom mapping
# --------------------------------------------------------------------------
def _overlap(R: Side, P: Side, perm: list) -> int:
    return sum(min(o, P.adj[perm[a]].get(perm[b], 0)) for (a, b), o in R.bonds.items())


def map_atoms(R: Side, P: Side, seed: int = 0) -> list:
    """perm[a] = index of the product atom that reactant atom `a` becomes."""
    if Counter(R.el) != Counter(P.el):
        raise ValueError("the two sides do not contain the same atoms")
    rng = random.Random(seed)
    groups_r, groups_p = {}, {}
    for i, e in enumerate(R.el):
        groups_r.setdefault(e, []).append(i)
    for i, e in enumerate(P.el):
        groups_p.setdefault(e, []).append(i)

    def contrib(perm, a):
        pa = P.adj[perm[a]]
        return sum(min(o, pa.get(perm[b], 0)) for b, o in R.adj[a].items())

    restarts = 60 if R.n <= 30 else (24 if R.n <= 70 else 10)
    best_perm, best_score = None, -1
    for _ in range(restarts):
        perm = [0] * R.n
        for e, atoms in groups_r.items():
            targets = groups_p[e][:]
            rng.shuffle(targets)
            for a, t in zip(atoms, targets):
                perm[a] = t
        improved, sweeps = True, 0
        while improved and sweeps < 60:
            improved, sweeps = False, sweeps + 1
            for atoms in groups_r.values():
                for i in range(len(atoms)):
                    a = atoms[i]
                    for j in range(i + 1, len(atoms)):
                        b = atoms[j]
                        before = contrib(perm, a) + contrib(perm, b)
                        perm[a], perm[b] = perm[b], perm[a]
                        after = contrib(perm, a) + contrib(perm, b)
                        if after > before or (after == before and rng.random() < 0.05):
                            improved = improved or after > before
                        else:
                            perm[a], perm[b] = perm[b], perm[a]
        score = _overlap(R, P, perm)
        if score > best_score:
            best_score, best_perm = score, perm[:]
    return best_perm


# --------------------------------------------------------------------------
# 5. layout (needed before the distance-based refinement of the map)
# --------------------------------------------------------------------------
def _grid(radii: list, cols: int | None = None) -> np.ndarray:
    """Centres (m, 3) of molecules laid out on a compact grid in the x-y plane."""
    m = len(radii)
    if m == 0:
        return np.zeros((0, 3))
    cell = 2.0 * max(radii) + GAP
    cols = cols or max(1, int(math.ceil(math.sqrt(1.7 * m))))
    rows = int(math.ceil(m / cols))
    out = np.zeros((m, 3))
    for k in range(m):
        r, c = divmod(k, cols)
        in_row = min(cols, m - r * cols)
        out[k] = [(c - (in_row - 1) / 2.0) * cell, ((rows - 1) / 2.0 - r) * cell,
                  0.35 * cell * (((r + c) % 2) - 0.5)]       # stagger in depth
    return out


def _width(centres: np.ndarray, radii: list) -> float:
    if len(centres) == 0:
        return 0.0
    return float((centres[:, 0] + radii).max() - (centres[:, 0] - radii).min())


def layout_reactants(R: Side) -> np.ndarray:
    """One block of molecules per reactant species, side by side."""
    blocks, start = [], 0
    for _, coef in R.species:
        radii = [m["radius"] for m in R.mols[start:start + int(coef)]]
        c = _grid(radii, cols=max(1, int(math.ceil(math.sqrt(int(coef))))))
        blocks.append((c, np.array(radii)))
        start += int(coef)
    total = sum(_width(c, r) for c, r in blocks) + 2.2 * GAP * (len(blocks) - 1)
    x, out = -total / 2.0, []
    for c, r in blocks:
        w = _width(c, r)
        shift = x - float((c[:, 0] - r).min())
        out.append(c + [shift, 0.0, 0.0])
        x += w + 2.2 * GAP
    return np.concatenate(out)


def layout_products(P: Side) -> np.ndarray:
    return _grid([m["radius"] for m in P.mols])


def _spin(local_xy: np.ndarray, target_xy: np.ndarray) -> float:
    """In-plane angle that best turns `local_xy` onto `target_xy` (both centred)."""
    s = float((local_xy[:, 0] * target_xy[:, 1] - local_xy[:, 1] * target_xy[:, 0]).sum())
    c = float((local_xy[:, 0] * target_xy[:, 0] + local_xy[:, 1] * target_xy[:, 1]).sum())
    return math.atan2(s, c) if abs(s) + abs(c) > 1e-9 else 0.0


def _rotz(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def place(R: Side, P: Side, perm: list):
    """World coordinates: start[a] for reactant atoms, end[p] for product atoms.
    Product molecules of one species swap grid places and spin in-plane so the
    atoms have the shortest way to go."""
    centres_r = layout_reactants(R)
    start = np.array([centres_r[R.mol_of[a]] + R.local[a] for a in range(R.n)])
    inv = [0] * R.n
    for a, p in enumerate(perm):
        inv[p] = a
    slots = layout_products(P)
    centres_p = slots.copy()
    # molecules of the same species are interchangeable: give each the nearest slot
    by_formula = {}
    for mi, m in enumerate(P.mols):
        by_formula.setdefault(m["formula"], []).append(mi)
    for members in by_formula.values():
        if len(members) > 1:
            src = np.array([start[[inv[p] for p in P.mols[mi]["atoms"]]].mean(axis=0)
                            for mi in members])
            cost = ((src[:, None, :] - slots[members][None, :, :]) ** 2).sum(axis=2)
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                centres_p[members[r]] = slots[members[c]]
    end = np.zeros((P.n, 3))
    spins = []
    for mi, m in enumerate(P.mols):
        atoms = m["atoms"]
        local = np.array([P.local[p] for p in atoms])
        src = start[[inv[p] for p in atoms]]
        theta = _spin(local[:, :2] - local[:, :2].mean(axis=0),
                      src[:, :2] - src[:, :2].mean(axis=0)) if len(atoms) > 1 else 0.0
        spins.append(theta)
        end[atoms] = local @ _rotz(theta).T + centres_p[mi]
    return start, end, centres_r, centres_p, spins


def refine_by_distance(R: Side, P: Side, perm: list, start: np.ndarray, end: np.ndarray) -> list:
    """Among maps with the same bond overlap, prefer shorter journeys: swap the
    targets of two like atoms whenever that keeps every bond and saves distance."""
    perm = perm[:]
    groups = {}
    for i, e in enumerate(R.el):
        groups.setdefault(e, []).append(i)

    def contrib(a):
        pa = P.adj[perm[a]]
        return sum(min(o, pa.get(perm[b], 0)) for b, o in R.adj[a].items())

    for _ in range(4):
        changed = False
        for atoms in groups.values():
            for i in range(len(atoms)):
                a = atoms[i]
                for j in range(i + 1, len(atoms)):
                    b = atoms[j]
                    d_now = (((start[a] - end[perm[a]]) ** 2).sum()
                             + ((start[b] - end[perm[b]]) ** 2).sum())
                    d_new = (((start[a] - end[perm[b]]) ** 2).sum()
                             + ((start[b] - end[perm[a]]) ** 2).sum())
                    if d_new >= d_now - 1e-9:
                        continue
                    before = contrib(a) + contrib(b)
                    perm[a], perm[b] = perm[b], perm[a]
                    if contrib(a) + contrib(b) < before:
                        perm[a], perm[b] = perm[b], perm[a]
                    else:
                        changed = True
        if not changed:
            break
    return perm


# --------------------------------------------------------------------------
# 3. electrons
# --------------------------------------------------------------------------
def track_electrons(R: Side, P: Side, perm: list, start: np.ndarray, end_u: np.ndarray):
    """Follow every valence electron. Atoms are numbered as on the reactant
    side throughout ("unified" numbering; `end_u[a]` is where atom a ends up).

    Returns (moves, stats). A move is
        [from_kind, a, b, slot, to_kind, a, b, slot, flag]
    with kind 0 = non-bonding electron on atom a (b = -1) and kind 1 = one of
    the electrons of the bond a-b; `slot` numbers the electrons of that place;
    flag 0 = stays put, 1 = moves but stays with an atom it belonged to,
    2 = transferred to a different atom.
    """
    lone_p = [P.lone[perm[a]] for a in range(R.n)]
    bonds_p = {}
    inv = [0] * R.n
    for a, p in enumerate(perm):
        inv[p] = a
    for (p, q), o in P.bonds.items():
        a, b = sorted((inv[p], inv[q]))
        bonds_p[(a, b)] = o

    moves, sources, sinks = [], [], []
    for a in range(R.n):
        keep = min(R.lone[a], lone_p[a])
        for k in range(keep):
            moves.append([0, a, -1, k, 0, a, -1, k, 0])
        sources += [(0, a, -1, k) for k in range(keep, R.lone[a])]
        sinks += [(0, a, -1, k) for k in range(keep, lone_p[a])]
    for (a, b) in sorted(set(R.bonds) | set(bonds_p)):
        n_r, n_p = 2 * R.bonds.get((a, b), 0), 2 * bonds_p.get((a, b), 0)
        keep = min(n_r, n_p)
        for k in range(keep):
            moves.append([1, a, b, k, 1, a, b, k, 0])
        sources += [(1, a, b, k) for k in range(keep, n_r)]
        sinks += [(1, a, b, k) for k in range(keep, n_p)]

    unmatched = abs(len(sources) - len(sinks))
    if sources and sinks:
        def where(slot, coords):
            kind, a, b, _ = slot
            return coords[a] if kind == 0 else (coords[a] + coords[b]) / 2.0

        def atoms_of(slot):
            return {slot[1]} if slot[0] == 0 else {slot[1], slot[2]}

        span = float(np.linalg.norm(np.ptp(np.vstack([start, end_u]), axis=0))) + 1e-6
        cost = np.zeros((len(sources), len(sinks)))
        for i, s in enumerate(sources):
            sa, sp = atoms_of(s), where(s, start)
            for j, d in enumerate(sinks):
                shared = bool(sa & atoms_of(d))
                cost[i, j] = (0.0 if shared else 1.0) + 0.2 * float(
                    np.linalg.norm(sp - where(d, end_u))) / span
        rows, cols = linear_sum_assignment(cost)
        for i, j in zip(rows, cols):
            s, d = sources[i], sinks[j]
            shared = bool(atoms_of(s) & atoms_of(d))
            moves.append([*s, *d, 1 if shared else 2])
    transferred = sum(1 for m in moves if m[8] == 2)
    return moves, {"electrons": len(moves), "moved": sum(1 for m in moves if m[8]),
                   "transferred": transferred, "unmatched": unmatched}, bonds_p


# --------------------------------------------------------------------------
# captions
# --------------------------------------------------------------------------
_HALOGENS = {"F", "Cl", "Br", "I"}


def _bond_pair(x: str, y: str) -> tuple:
    """Order two bonded elements the way chemists write the bond:
    C-H, O-H, N-H, H-Cl, C=O, S=O, N-O ..."""
    if x == y:
        return x, y
    if "H" in (x, y):
        other = y if x == "H" else x
        return ("H", other) if other in _HALOGENS else (other, "H")
    if "C" in (x, y):
        return ("C", y if x == "C" else x)
    from .elements import electronegativity
    return (x, y) if electronegativity(x) <= electronegativity(y) else (y, x)


def _bond_label(x: str, y: str, order: int) -> str:
    a, b = _bond_pair(x, y)
    return f"{a}{BOND_SYMBOL.get(order, '–')}{b}"


def _bond_names(bonds: dict, el: list) -> str:
    counts = Counter()
    for (a, b), o in bonds.items():
        counts[_bond_label(el[a], el[b], o)] += 1
    return ", ".join(f"{n} {name}" for name, n in sorted(counts.items(), key=lambda kv: -kv[1]))


def _signed(n: int) -> str:
    return f"+{n}" if n > 0 else (f"−{-n}" if n < 0 else "0")


def _ion(el: str, q: int) -> str:
    mag = "" if abs(q) == 1 else str(abs(q))
    return el + (mag + ("+" if q > 0 else "-")).translate(SUPERSCRIPT)


def plan_reaction(reactants: list, products: list, library: dict, seed: int = 0) -> dict:
    """Full animation plan. `reactants` / `products` are [(formula, coefficient)]
    with integer coefficients; every formula must be a key of `library`."""
    R, P = Side(reactants, library), Side(products, library)
    perm = map_atoms(R, P, seed)
    start, end, centres_r, centres_p, spins = place(R, P, perm)
    for _ in range(2):                       # shorter journeys, then re-place
        perm = refine_by_distance(R, P, perm, start, end)
        start, end, centres_r, centres_p, spins = place(R, P, perm)
    end_u = np.array([end[perm[a]] for a in range(R.n)])
    moves, estats, bonds_p = track_electrons(R, P, perm, start, end_u)

    broken = {k: o for k, o in R.bonds.items() if k not in bonds_p}
    formed = {k: o for k, o in bonds_p.items() if k not in R.bonds}
    changed = {k: (R.bonds[k], bonds_p[k]) for k in R.bonds
               if k in bonds_p and bonds_p[k] != R.bonds[k]}

    inv = [0] * R.n
    for a, p in enumerate(perm):
        inv[p] = a
    ionic_p = [sorted((inv[p], inv[q])) for p, q in P.ionic]

    # oxidation-state changes, grouped by element
    redox = Counter()
    for a in range(R.n):
        before, after = R.ox[a], P.ox[perm[a]]
        if before != after:
            redox[(R.el[a], before, after)] += 1
    redox_list = [{"el": e, "from": b, "to": a, "n": n}
                  for (e, b, a), n in sorted(redox.items(), key=lambda kv: (kv[0][2] < kv[0][1], kv[0][0]))]

    # which atoms hand electrons to which (for the "electron transfer" caption)
    flows = Counter()
    for m in moves:
        if m[8] == 2:
            src = R.el[m[1]]
            dst_atoms = [m[5]] if m[4] == 0 else [m[5], m[6]]
            flows[(src, "/".join(sorted({R.el[x] for x in dst_atoms})))] += 1

    steps = []
    if broken:
        steps.append({"phase": "break", "text": "Bonds break: " + _bond_names(broken, R.el)})
    elif R.ionic:
        steps.append({"phase": "break", "text": "The ions separate"})
    else:
        steps.append({"phase": "break", "text": "The particles collide"})
    move_text = "Atoms and electrons rearrange"
    if flows:
        parts = [f"{n} e⁻ from {s} to {d}" for (s, d), n in flows.most_common(3)]
        move_text = "Electrons transfer: " + ", ".join(parts)
    steps.append({"phase": "move", "text": move_text})
    form_bits = []
    if formed:
        form_bits.append("Bonds form: " + _bond_names(formed, R.el))
    if changed:
        ch = Counter()
        for (a, b), (o1, o2) in changed.items():
            ch[f"{_bond_label(R.el[a], R.el[b], o1)} → {_bond_label(R.el[a], R.el[b], o2)}"] += 1
        form_bits.append("Bonds change: " + ", ".join(f"{n} {k}" for k, n in ch.items()))
    if ionic_p:
        ions = Counter()
        for a in range(R.n):
            p = perm[a]
            if not P.adj[p] and P.fc[p]:
                ions[_ion(P.el[p], P.fc[p])] += 1
        if ions:
            form_bits.append("Ions attract: " + " ".join(sorted(ions)))
    steps.append({"phase": "form", "text": " · ".join(form_bits) if form_bits else "The products form"})
    if redox_list:
        ox = [r for r in redox_list if r["to"] > r["from"]]
        red = [r for r in redox_list if r["to"] < r["from"]]
        bits = []
        if ox:
            bits.append(", ".join(f"{r['el']} {_signed(r['from'])} → {_signed(r['to'])}" for r in ox)
                        + " (oxidised)")
        if red:
            bits.append(", ".join(f"{r['el']} {_signed(r['from'])} → {_signed(r['to'])}" for r in red)
                        + " (reduced)")
        steps.append({"phase": "done", "text": "Oxidation states: " + "; ".join(bits)})
    else:
        steps.append({"phase": "done", "text": "No oxidation states change (not a redox reaction)"})

    lo = np.minimum(start.min(axis=0), end.min(axis=0))
    hi = np.maximum(start.max(axis=0), end.max(axis=0))
    pad = max(max(R.rad), max(P.rad)) + 0.4

    def r3(v):
        return [round(float(x), 3) for x in v]

    atoms = []
    for a in range(R.n):
        p = perm[a]
        atoms.append([R.el[a], *r3(start[a]), *r3(end_u[a]), R.fc[a], P.fc[p],
                      R.rad[a], P.rad[p], R.mol_of[a], P.mol_of[p], R.ox[a], P.ox[p]])
    return {
        "reactants": [[f, int(c)] for f, c in reactants],
        "products": [[f, int(c)] for f, c in products],
        # [el, sx, sy, sz, ex, ey, ez, charge0, charge1, radius0, radius1, mol0, mol1, ox0, ox1]
        "atoms": atoms,
        "r_mols": [{"f": m["formula"], "c": r3(centres_r[i])} for i, m in enumerate(R.mols)],
        "p_mols": [{"f": m["formula"], "c": r3(centres_p[i])} for i, m in enumerate(P.mols)],
        "bonds_r": [[a, b, o] for (a, b), o in sorted(R.bonds.items())],
        "bonds_p": [[a, b, o] for (a, b), o in sorted(bonds_p.items())],
        "ionic_r": [list(p) for p in R.ionic],
        "ionic_p": ionic_p,
        "electrons": moves,
        "redox": redox_list,
        "steps": steps,
        "stats": {"atoms": R.n, "bonds_broken": int(sum(broken.values())),
                  "bonds_formed": int(sum(formed.values())), **estats},
        "box": [r3(lo - pad), r3(hi + pad)],
    }
