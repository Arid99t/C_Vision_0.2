"""Build the 3D chemistry bundle the web app plays back.

    .venv\\Scripts\\python.exe -m backend.chemistry.build3d          (needs RDKit)

Reads   data/structures.json   real structure of every known compound
        data/reactions.json    curated reactions
Writes  backend/static/data/chem3d.json with

  molecules   3D template per compound (atoms, bonds, charges, lone electrons)
  reactions   for every curated reaction, and for the combustion of every
              C/H/O compound in the library: the balanced equation plus a full
              animation plan (atom map, bonds broken/formed, where each valence
              electron goes, oxidation-state changes) from reaction3d.py
  lookups     aliases, known formulas, element colours

Structures whose SMILES did not change are reused from the previous bundle
(embedding and ion packing are the slow part); pass --force to rebuild all.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import zlib

from . import mol3d, reaction3d
from .balancer import react
from .formula import ELEMENTS, KNOWN_FORMULAS, parse_formula
from .reactions import _TABLE, _is_hydrocarbon

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(os.path.dirname(_HERE), "static", "data", "chem3d.json")
MAX_ATOMS = 130      # reactions bigger than this are listed without an animation plan


def reaction_key(reactants) -> str:
    """Lookup key for a set of reactant formulas (order-independent)."""
    return "|".join(sorted(set(reactants)))


def _load_previous() -> dict:
    try:
        with open(OUT_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def build(force: bool = False, verbose: bool = True) -> dict:
    t0 = time.time()
    structures = mol3d.load_structures()
    previous = {} if force else _load_previous().get("molecules", {})

    molecules, aliases, by_comp, rebuilt = {}, {}, {}, 0
    for formula, entry in structures.items():
        old = previous.get(formula)
        if old and old.get("smiles") == entry["smiles"]:
            t = old
        else:
            t = mol3d.build_template(formula, entry)
            t["smiles"] = entry["smiles"]
            rebuilt += 1
        t["name"], t["note"] = entry.get("name"), entry.get("note")
        problems = mol3d.check_template(t)
        if problems:
            raise ValueError(f"{formula}: " + "; ".join(problems))
        molecules[formula] = t
        by_comp.setdefault(mol3d.composition_key(parse_formula(formula)), formula)
        for alias in entry.get("aliases", []):
            if parse_formula(alias) != parse_formula(formula):
                raise ValueError(f"{formula}: alias {alias} has a different composition")
            aliases[alias] = formula
    if verbose:
        print(f"structures: {len(molecules)} ({rebuilt} built, {len(molecules) - rebuilt} reused) "
              f"in {time.time() - t0:.0f}s")

    # every curated reaction + combustion of each C/H/O compound in the library
    wanted = [list(key) for key in _TABLE]
    seen = {reaction_key(r) for r in wanted}
    for formula in molecules:
        if formula != "O2" and _is_hydrocarbon(parse_formula(formula)):
            pair = [formula, "O2"]
            if reaction_key(pair) not in seen:
                wanted.append(pair)
                seen.add(reaction_key(pair))

    reactions, text_only, failed = {}, [], []
    t1 = time.time()
    for reactants in wanted:
        key = reaction_key(reactants)
        res = react(reactants)
        if not res.get("ok"):
            failed.append((key, res.get("reason")))
            continue
        coefs = list(res["reactants"].items()) + list(res["products"].items())
        if any(c != int(c) or c <= 0 for _, c in coefs):
            failed.append((key, f"non-integer or zero coefficient: {dict(coefs)}"))
            continue
        entry = {"key": key, "equation": res["equation"], "name": res.get("name"),
                 "conditions": res.get("conditions"), "note": res.get("note"),
                 "dH": res.get("dH"), "source": res.get("source"),
                 "thermochemistry": res.get("thermochemistry"),
                 "reactants": [[f, int(c)] for f, c in res["reactants"].items()],
                 "products": [[f, int(c)] for f, c in res["products"].items()]}
        missing = [f for f, _ in coefs if f not in molecules]
        n_atoms = sum(int(c) * len(molecules[f]["atoms"]) for f, c in res["reactants"].items()
                      if f in molecules)
        if missing or n_atoms > MAX_ATOMS:
            text_only.append((key, f"no structure for {missing}" if missing
                              else f"{n_atoms} atoms"))
        else:
            plan = reaction3d.plan_reaction(entry["reactants"], entry["products"], molecules,
                                            seed=zlib.crc32(key.encode()))
            if plan["stats"]["unmatched"]:
                raise ValueError(f"{key}: electron count differs between the two sides")
            entry["plan"] = plan
        reactions[key] = entry
    if verbose:
        planned = sum(1 for r in reactions.values() if "plan" in r)
        print(f"reactions: {len(reactions)} balanced, {planned} with a 3D plan "
              f"in {time.time() - t1:.0f}s")
        for key, why in text_only:
            print(f"  text only   {key}: {why}")
        for key, why in failed:
            print(f"  NOT USABLE  {key}: {why}")

    if failed:
        raise ValueError(f"Refusing to publish an incomplete chemistry bundle: {failed}")
    used = sorted({a[0] for t in molecules.values() for a in t["atoms"]})
    bundle = {
        "version": 1,
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elements": {el: {"color": mol3d.element_color(el),
                          "en": mol3d.electronegativity(el)} for el in used},
        "all_elements": sorted(ELEMENTS),
        "known": sorted(KNOWN_FORMULAS | set(molecules)),
        "aliases": aliases,
        "by_composition": by_comp,
        "molecules": molecules,
        "reactions": reactions,
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(bundle, f, separators=(",", ":"), ensure_ascii=False)
    if verbose:
        print(f"wrote {OUT_PATH} ({os.path.getsize(OUT_PATH) / 1e6:.2f} MB)")
    return bundle


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build the 3D chemistry bundle")
    p.add_argument("--force", action="store_true", help="rebuild every structure")
    build(force=p.parse_args().force)
