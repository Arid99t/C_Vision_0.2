"""The Assist app's authored data must be internally consistent.

safety.json and parts.json are edited by hand; these checks catch the mistakes
that would otherwise only show up as a silent non-match on the device: a rule
whose tag no substance carries, a word claimed by two substances, two formulas
the reader cannot tell apart, a pinout with the wrong number of pins.
"""
import json
import re
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "backend" / "static" / "data"
SAFETY = json.loads((DATA / "safety.json").read_text(encoding="utf-8"))
PARTS = json.loads((DATA / "parts.json").read_text(encoding="utf-8"))["parts"]
LEVELS = {"danger", "caution", "info"}
KINDS = {"chemical", "food", "drink", "medicine", "other", "label"}
HAZARDS = {"flammable", "oxidiser", "corrosive", "toxic", "irritant", "health", "environment"}
PPE = {"gloves", "goggles", "lab coat"}


def formula_key(text: str) -> str:
    """Mirror of formulaKey() in static/js/assist/kb.js."""
    return re.sub(r"[lI|]", "1", text.replace("0", "O")).upper()


def tokens(phrase: str) -> tuple:
    return tuple(t for t in re.split(r"[^a-z0-9']+", phrase.lower()) if t)


def test_substances_are_well_formed():
    ids = [s["id"] for s in SAFETY["substances"]]
    assert len(ids) == len(set(ids)), "duplicate substance id"
    for s in SAFETY["substances"]:
        assert s["kind"] in KINDS, s["id"]
        assert s.get("words") or s.get("formulas"), f"{s['id']} can never be recognised"
        assert set(s.get("hazards", [])) <= HAZARDS, s["id"]
        assert set(s.get("ppe", [])) <= PPE, s["id"]
        assert set(s.get("allergens", [])) <= set(SAFETY["allergens"]), s["id"]
        for w in s.get("words", []):
            assert w == w.lower() and tokens(w), f"{s['id']}: words must be lower case: {w!r}"
        if s["kind"] == "label":
            assert s.get("removes"), f"label {s['id']} does nothing"
        else:
            assert s.get("name")


def test_no_word_or_formula_names_two_things():
    seen = {}
    for s in SAFETY["substances"]:
        for w in s.get("words", []):
            key = tokens(w)
            assert seen.setdefault(key, s["id"]) == s["id"], f"{w!r} is claimed by {seen[key]} and {s['id']}"
    formulas = {}
    for s in SAFETY["substances"]:
        for f in s.get("formulas", []):
            key = formula_key(f)
            assert formulas.setdefault(key, s["id"]) == s["id"], f"{f} collides with {formulas[key]} once misreads are allowed"


def test_every_rule_can_fire_and_explains_itself():
    tags = {t for s in SAFETY["substances"] for t in s.get("tags", [])}
    tags |= {"kind:" + s["kind"] for s in SAFETY["substances"]}
    ids = [r["id"] for r in SAFETY["rules"]]
    assert len(ids) == len(set(ids)), "duplicate rule id"
    for r in SAFETY["rules"]:
        assert r["level"] in LEVELS, r["id"]
        assert r["a"] in tags and r["b"] in tags, f"rule {r['id']} uses a tag no substance has"
        for field in ("title", "why", "do", "source"):
            assert r.get(field, "").strip(), f"rule {r['id']} is missing {field}"
        assert len(r["title"]) <= 44, f"rule {r['id']}: title too long for the glasses display"
    used = {r["a"] for r in SAFETY["rules"]} | {r["b"] for r in SAFETY["rules"]}
    removable = {x for s in SAFETY["substances"] for x in s.get("removes", [])}
    for x in removable:
        assert x in tags or (x.startswith("allergen:") and x[9:] in SAFETY["allergens"]), f"'removes' names unknown {x}"
    unused = {t for t in tags if not t.startswith("kind:")} - used
    assert not unused, f"tags that no rule uses (typo?): {sorted(unused)}"


def test_the_headline_cases():
    """The pairs people actually ask about must be in the list at the right level."""
    by_tag = {}
    for s in SAFETY["substances"]:
        for t in s.get("tags", []):
            by_tag.setdefault(s["id"], set()).add(t)

    def level(a, b):
        ta, tb = by_tag.get(a, set()), by_tag.get(b, set())
        order = {"danger": 3, "caution": 2, "info": 1}
        hits = [r["level"] for r in SAFETY["rules"]
                if (r["a"] in ta and r["b"] in tb) or (r["b"] in ta and r["a"] in tb)]
        return max(hits, key=order.get) if hits else None

    assert level("bleach", "ammonia") == "danger"
    assert level("bleach", "toilet-cleaner") == "danger"
    assert level("bleach", "vinegar") == "danger"
    assert level("bleach", "rubbing-alcohol") == "danger"
    assert level("drain-cleaner", "toilet-cleaner") == "danger"
    assert level("hydrogen-peroxide", "vinegar") == "caution"
    assert level("alcohol-drink", "energy-drink") == "caution"
    assert level("alcohol-drink", "antihistamine") == "danger"
    assert level("paracetamol", "cold-flu") == "caution"
    assert level("grapefruit", "statin") == "caution"
    assert level("milk", "iron-supplement") == "info"
    assert level("vinegar", "baking-soda") == "info"
    assert level("milk", "coffee") is None
    assert level("sodium", "water") == "danger"


def test_parts_are_well_formed():
    pins = {"DIP-4": 4, "DIP-6": 6, "DIP-8": 8, "DIP-14": 14, "DIP-16": 16, "DIP-20": 20, "DIP-28": 28,
            "TO-92": 3, "TO-220": 3}
    ids = [p["id"] for p in PARTS]
    assert len(ids) == len(set(ids))
    for p in PARTS:
        re.compile(p["re"])
        assert p["re"].startswith("^") and p["re"].endswith("$"), f"{p['id']}: pattern must match the whole marking"
        assert len(p["pins"]) == pins[p["package"]], f"{p['id']}: {len(p['pins'])} pins for {p['package']}"
        for field in ("name", "about", "supply", "tip"):
            assert p[field].strip(), f"{p['id']} is missing {field}"
        if p["package"].startswith("DIP"):
            assert any(re.search(r"GND|VSS|V-|CATHODE|EMITTER", x) for x in p["pins"]), f"{p['id']} has no ground pin"


def test_markings_find_exactly_one_part():
    def find(marking):
        return [p["id"] for p in PARTS if re.match(p["re"], marking)]

    known = {"NE555P": "555", "LM555CN": "555", "SN74HC595N": "74595", "LM358N": "358", "CD4017BE": "4017",
             "ATMEGA328PPU": "ATmega328", "L7805CV": "78xx", "LM7912CT": "79xx", "BC547B": "BC547",
             "2N3904": "2N3904", "2N3906": "2N3904", "74LS00": "7400", "SN7404N": "7404", "HEF4011BP": "4011", "ULN2003AN": "ULN2003",
             "L293D": "L293D", "MAX232CPE": "MAX232", "IRFZ44N": "IRF", "TIP120": "TIP", "LM317T": "LM317",
             "DS18B20": "DS18B20", "78L05": "78Lxx", "74HC4051": "4051", "TL072CP": "TL072", "LM324N": "324",
             "MC74HC00AN": "7400", "LM1117T": "1117", "LM1117T33": "1117", "24LC256": "24LC256", "PC817": "PC817", "74HC14": "7414",
             "74HC164": "74164", "UA741CP": "741", "LM386N1": "386", "ATTINY8520PU": "ATtiny85", "2N7000": "2N7000"}
    for marking, want in known.items():
        assert find(marking) == [want], f"{marking} -> {find(marking)}"
    # date codes, country of origin and bare numbers must not be taken for a part
    for junk in ("2148", "G4", "PHILIPPINES", "MALAYSIA", "358", "555", "0A12", "TEXAS", "CHINA", "1117", "4017"):
        assert not find(junk), f"{junk} wrongly matches {find(junk)}"


def test_different_packages_do_not_inherit_a_through_hole_pinout():
    # 2N7002 is SOT-23, not the TO-92 2N7000; 1117 suffixes identify
    # different packages. Similar transistor numbers can reverse E and C.
    # Unknown is preferable to a confident diagram for the wrong component.
    for marking in ("2N7002", "AMS1117", "LM1117", "LM1117IMP33", "LM1117DT33",
                    "2N2222A", "PN2222A", "P2N2222A", "S8050", "SS8550"):
        assert not any(re.fullmatch(p["re"], marking) for p in PARTS), marking
