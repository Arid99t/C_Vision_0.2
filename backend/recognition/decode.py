"""Chemistry-constrained CTC beam-search decoding.

Greedy CTC decoding picks the single most likely character per timestep, so one
ambiguous stroke turns `H2SO4` into `H2O4` or `HCl` into `NHl` even though the
model assigned real probability to the correct string. This module runs a CTC
prefix beam search whose beams are constrained by a *formula grammar*: a prefix
survives only if it can still grow into a valid formula line (element symbols
from the periodic table, counts that don't start with 0, balanced parentheses,
optional `A + B` separators). Illegal strings like `Hl`, `N4a` or `H20`
(zero-count) are pruned before they can win, so the search redistributes their
probability to the chemically possible readings.

Pure numpy — no torch — so it can be unit-tested and reused anywhere.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from backend.chemistry.formula import ELEMENTS

_ONE_LETTER = {e for e in ELEMENTS if len(e) == 1}
_FIRST_LETTERS = {e[0] for e in ELEMENTS}  # no element starts with J or Q

# --------------------------------------------------------------------------
# Formula-line grammar as an incremental automaton.
#
# A state is the immutable tuple (sep, depth, done, upper, prev):
#   sep    None while inside a formula, else the separator progress:
#          'sp1' = space after a complete formula, 'plus' = '+' seen,
#          'sp2' = space after '+'.
#   depth  current parenthesis nesting depth
#   done   tuple of bools, one per depth: has a completed unit at that depth
#   upper  a pending uppercase letter that may still become a 2-letter symbol
#   prev   what was last completed: 'start' | 'elem' | 'close' | 'digit'
# --------------------------------------------------------------------------
INITIAL = (None, 0, (False,), None, "start")


def _set_done(done: tuple, depth: int) -> tuple:
    if done[depth]:
        return done
    return done[:depth] + (True,) + done[depth + 1:]


def _start_formula(ch: str):
    """State for the first character of a fresh formula (after a separator)."""
    if ch.isupper() and ch in _FIRST_LETTERS:
        return (None, 0, (False,), ch, "start")
    if ch == "(":
        return (None, 1, (False, False), None, "start")
    return None


def advance(state: tuple, ch: str) -> Optional[tuple]:
    """Feed one character; return the next state, or None if now invalid."""
    sep, depth, done, upper, prev = state

    if sep is not None:                       # between formulas: 'A + B'
        if sep == "sp1":
            return (("plus", depth, done, None, prev) if ch == "+" else None)
        if sep == "plus":
            if ch == " ":
                return ("sp2", depth, done, None, prev)
            return _start_formula(ch)         # also accept 'A+B' without spaces
        return _start_formula(ch)             # sp2 -> formula must start

    # resolve a pending uppercase letter first
    if upper is not None:
        if ch.islower():
            if upper + ch in ELEMENTS:        # two-letter symbol completed
                return (None, depth, _set_done(done, depth), None, "elem")
            return None
        if upper not in _ONE_LETTER:          # must stand alone but can't
            return None
        done = _set_done(done, depth)         # one-letter element completed
        prev = "elem"
    elif ch.islower():
        return None                           # lowercase needs an uppercase first

    if ch.isupper():
        if ch not in _FIRST_LETTERS:
            return None                       # no element starts with this letter
        return (None, depth, done, ch, prev)
    if ch.isdigit():
        if prev in ("elem", "close"):
            return None if ch == "0" else (None, depth, done, None, "digit")
        if prev == "digit":
            return (None, depth, done, None, "digit")
        return None                           # count with nothing to count
    if ch == "(":
        return (None, depth + 1, done + (False,), None, "start")
    if ch == ")":
        if depth == 0 or not done[depth] or prev == "start":
            return None
        return (None, depth - 1, _set_done(done[:-1], depth - 1), None, "close")
    if ch in (" ", "+"):
        complete = depth == 0 and done[0] and prev in ("elem", "close", "digit")
        if not complete:
            return None
        return ("sp1" if ch == " " else "plus", 0, (False,), None, "start")
    return None


def is_complete(state: tuple) -> bool:
    """True if the string scanned so far is a complete, valid formula line."""
    sep, depth, done, upper, prev = state
    if sep is not None or depth != 0:
        return False
    if upper is not None:
        return upper in _ONE_LETTER           # pending letter closes the formula
    return done[0] and prev in ("elem", "close", "digit")


def valid_prefix(text: str) -> Optional[tuple]:
    """Scan a whole string; return its state, or None if it can't be extended
    into a valid formula line. Convenience wrapper for tests."""
    state = INITIAL
    for ch in text:
        state = advance(state, ch)
        if state is None:
            return None
    return state


# --------------------------------------------------------------------------
# CTC prefix beam search (Hannun et al.) with grammar-pruned extensions.
# --------------------------------------------------------------------------
_NEG = -1.0e30  # ~log(0)


def beam_search(logp: np.ndarray, itos: list,
                beam_width: int = 8, topk_chars: int = 6):
    """Decode a (T, C) log-prob matrix into the best *valid* formula line.

    Returns (text, confidence) where confidence is the per-timestep geometric
    mean of the winning beam's probability, or None if no beam survives as a
    complete formula (caller should fall back to greedy decoding).
    """
    cands = beam_candidates(logp, itos, beam_width, topk_chars, limit=1)
    if not cands:
        return None
    text, total = cands[0]
    return text, math.exp(total / logp.shape[0])


# How much a reading that spells a KNOWN compound is preferred over one that
# is merely well-formed, in log-probability units: e^2 ~ 7.4x. Handwriting is
# full of look-alikes that only context can settle (CO / Co, HF / Hf, Cl / C1);
# people write real compounds far more often than not, so when two readings are
# close the known one wins. A clearly better unknown reading still wins.
LEXICON_BONUS = 2.0


def choose_reading(cands: list, is_known, bonus: float = LEXICON_BONUS):
    """Pick from beam_candidates() output [(text, total_logp), ...] using the
    lexicon prior. `is_known(text) -> bool`. Returns (text, total_logp) or None."""
    if not cands:
        return None
    return max(cands, key=lambda c: c[1] + (bonus if is_known(c[0]) else 0.0))


def beam_candidates(logp: np.ndarray, itos: list, beam_width: int = 8,
                    topk_chars: int = 6, limit: int = 5) -> list:
    """Up to `limit` complete, valid formula lines as (text, total log-probability),
    most probable first. Empty when no beam ends as a complete formula."""
    T, C = logp.shape
    k = min(topk_chars, C - 1)
    # beam: prefix (tuple of char indices) -> [logP(ends in blank),
    #                                          logP(ends in non-blank), grammar state]
    beams = {(): [0.0, _NEG, INITIAL]}

    for t in range(T):
        step = logp[t]
        # only consider the k most likely characters this timestep (plus blank)
        cand = np.argpartition(step[1:], -k)[-k:] + 1
        new: dict = {}

        for prefix, (pb, pnb, gs) in beams.items():
            p_tot = np.logaddexp(pb, pnb)

            # stay on the same prefix: emit blank, or repeat the last char
            entry = new.get(prefix)
            if entry is None:
                entry = new[prefix] = [_NEG, _NEG, gs]
            entry[0] = np.logaddexp(entry[0], p_tot + step[0])
            if prefix:
                entry[1] = np.logaddexp(entry[1], pnb + step[prefix[-1]])

            # extend with a new character (grammar-checked)
            for ci in cand:
                ci = int(ci)
                ngs = advance(gs, itos[ci])
                if ngs is None:
                    continue                  # chemically impossible -> prune
                ext = prefix + (ci,)
                # a repeated char needs a blank in between -> only pb flows
                src = pb if (prefix and ci == prefix[-1]) else p_tot
                e = new.get(ext)
                if e is None:
                    e = new[ext] = [_NEG, _NEG, ngs]
                e[1] = np.logaddexp(e[1], src + step[ci])

        beams = dict(sorted(new.items(),
                            key=lambda kv: -np.logaddexp(kv[1][0], kv[1][1]))
                     [:beam_width])

    ranked = sorted(beams.items(),
                    key=lambda kv: -np.logaddexp(kv[1][0], kv[1][1]))
    out = []
    for prefix, (pb, pnb, gs) in ranked:
        if prefix and is_complete(gs):
            out.append(("".join(itos[i] for i in prefix), float(np.logaddexp(pb, pnb))))
            if len(out) >= limit:
                break
    return out
