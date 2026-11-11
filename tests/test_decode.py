"""Unit tests for the chemistry-constrained CTC beam search (decode.py).
Run: python tests/test_decode.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from backend.recognition import decode  # noqa: E402
from backend.recognition.charset import ITOS, STOI  # noqa: E402


def test_grammar():
    valid_complete = ["H2O", "CO2", "NaCl", "Ca(OH)2", "C6H12O6", "Mg(NO3)2",
                      "H", "C10H8", "Fe2O3", "CH4 + H2O", "H2+O2", "KMnO4",
                      "Al2(SO4)3", "H20"]  # H20 = 20 H; 0/O is fixed downstream
    valid_prefix_only = ["M", "A", "Ca(", "Ca(OH", "Al2(SO4",
                         "CH4 ", "CH4 +", "CH4 + "]
    invalid = ["Hl", "hCl", "2H", "H02", "()", "(H", "Ca)", "Xz", "H2 O",
               "CH4 ++", "CH4  + H2O", "Ca()2", "N4a", "Q", "JCl"]

    for s in valid_complete:
        st = decode.valid_prefix(s)
        assert st is not None and decode.is_complete(st), f"should be complete: {s!r}"
    for s in valid_prefix_only:
        st = decode.valid_prefix(s)
        assert st is not None, f"should be a valid prefix: {s!r}"
        assert not decode.is_complete(st), f"should NOT be complete: {s!r}"
    for s in invalid:
        st = decode.valid_prefix(s)
        assert st is None or not decode.is_complete(st), f"should be invalid: {s!r}"
    # these must be rejected even as prefixes (dead ends the beam should prune)
    for s in ["Hl", "hCl", "2H", "()", "Ca)", "N4a", "Q", "JCl"]:
        assert decode.valid_prefix(s) is None, f"should be pruned early: {s!r}"
    print("ok   grammar accepts/rejects the right strings")


def _logits_for(chars, tweak=None):
    """(T, C) log-prob matrix that strongly emits `chars` in order, with a
    blank between each. `tweak` overrides individual (t, char) probabilities."""
    T = 2 * len(chars) + 1
    probs = np.full((T, len(ITOS)), 1e-6)
    for t in range(T):
        if t % 2 == 0:
            probs[t, 0] = 1.0                      # blank
        else:
            probs[t, STOI[chars[(t - 1) // 2]]] = 1.0
    if tweak:
        for (t, ch), p in tweak.items():
            probs[t, STOI[ch]] = p
    probs /= probs.sum(1, keepdims=True)
    return np.log(probs)


def test_beam_search_prefers_valid():
    # unambiguous case: reads exactly what's there
    lp = _logits_for(list("H2O"))
    text, conf = decode.beam_search(lp, ITOS)
    assert text == "H2O", f"got {text!r}"
    assert conf > 0.9

    # ambiguous case: 'HCl' where the 'C' timestep slightly prefers 'l'.
    # Greedy would read 'Hl?l' territory; 'Hll'/'Hl' are not chemistry, so the
    # constrained search must recover 'HCl'.
    lp = _logits_for(list("HCl"), tweak={(3, "l"): 1.4, (3, "C"): 1.0})
    greedy = "".join(ITOS[i] for i in lp.argmax(1) if i != 0)
    assert "Hl" in greedy, f"tweak failed to fool greedy: {greedy!r}"
    text, _ = decode.beam_search(lp, ITOS)
    assert text == "HCl", f"got {text!r}"
    print("ok   beam search recovers valid formulas greedy gets wrong")


def test_beam_search_no_valid():
    # only '(' has support -> every beam is an incomplete prefix -> None
    lp = _logits_for(list("((("))
    assert decode.beam_search(lp, ITOS) is None
    # lowercase gibberish: whatever valid beam noise assembles must have ~0 conf
    res = decode.beam_search(_logits_for(list("abc")), ITOS)
    assert res is None or res[1] < 0.1, f"junk got confidence {res}"
    print("ok   nothing valid -> None or near-zero confidence")


if __name__ == "__main__":
    test_grammar()
    test_beam_search_prefers_valid()
    test_beam_search_no_valid()
    print("\n3/3 passed")
