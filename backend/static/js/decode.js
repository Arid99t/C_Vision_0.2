// Chemistry-constrained CTC decoding — JavaScript twin of
// backend/recognition/decode.py.
//
// The recognizer emits, for every image column, a probability for each
// character. A plain "take the best character per column" read turns one
// ambiguous stroke into nonsense (H2SO4 -> H2O4). Here a beam search keeps only
// strings that can still become a VALID formula line (real element symbols,
// counts that don't start with 0, balanced brackets, optional "A + B"), and
// when two readings are close it prefers one that spells a known compound.

const NEG = -1.0e30;

const isLower = (c) => c >= 'a' && c <= 'z';
const isUpper = (c) => c >= 'A' && c <= 'Z';
const isDigit = (c) => c >= '0' && c <= '9';

function logAddExp(a, b) {
  const m = a > b ? a : b;
  return m + Math.log1p(Math.exp(-Math.abs(a - b)));
}

/** Formula-line grammar as an incremental automaton over immutable states. */
export function makeGrammar(elementSymbols) {
  const ELEMENTS = new Set(elementSymbols);
  const ONE_LETTER = new Set([...ELEMENTS].filter((e) => e.length === 1));
  const FIRST = new Set([...ELEMENTS].map((e) => e[0]));
  // state: {sep, depth, done[], upper, prev}; prev in start|elem|close|digit
  const INITIAL = Object.freeze({ sep: null, depth: 0, done: [false], upper: null, prev: 'start' });
  const st = (sep, depth, done, upper, prev) => ({ sep, depth, done, upper, prev });

  function setDone(done, depth) {
    if (done[depth]) return done;
    const d = done.slice();
    d[depth] = true;
    return d;
  }

  function startFormula(ch) {
    if (isUpper(ch) && FIRST.has(ch)) return st(null, 0, [false], ch, 'start');
    if (ch === '(') return st(null, 1, [false, false], null, 'start');
    return null;
  }

  /** Feed one character; the next state, or null if the text is now invalid. */
  function advance(state, ch) {
    let { sep, depth, done, upper, prev } = state;
    if (sep !== null) {                                 // between formulas: "A + B"
      if (sep === 'sp1') return ch === '+' ? st('plus', depth, done, null, prev) : null;
      if (sep === 'plus') {
        if (ch === ' ') return st('sp2', depth, done, null, prev);
        return startFormula(ch);                        // also accept "A+B"
      }
      return startFormula(ch);
    }
    if (upper !== null) {                               // resolve a pending capital first
      if (isLower(ch)) {
        return ELEMENTS.has(upper + ch) ? st(null, depth, setDone(done, depth), null, 'elem') : null;
      }
      if (!ONE_LETTER.has(upper)) return null;
      done = setDone(done, depth);
      prev = 'elem';
    } else if (isLower(ch)) {
      return null;                                      // lowercase needs a capital first
    }
    if (isUpper(ch)) return FIRST.has(ch) ? st(null, depth, done, ch, prev) : null;
    if (isDigit(ch)) {
      if (prev === 'elem' || prev === 'close') return ch === '0' ? null : st(null, depth, done, null, 'digit');
      if (prev === 'digit') return st(null, depth, done, null, 'digit');
      return null;                                      // a count with nothing to count
    }
    if (ch === '(') return st(null, depth + 1, done.concat([false]), null, 'start');
    if (ch === ')') {
      if (depth === 0 || !done[depth] || prev === 'start') return null;
      return st(null, depth - 1, setDone(done.slice(0, -1), depth - 1), null, 'close');
    }
    if (ch === ' ' || ch === '+') {
      const complete = depth === 0 && done[0] && (prev === 'elem' || prev === 'close' || prev === 'digit');
      if (!complete) return null;
      return st(ch === ' ' ? 'sp1' : 'plus', 0, [false], null, 'start');
    }
    return null;
  }

  /** True if the text scanned so far is a complete, valid formula line. */
  function isComplete(state) {
    if (state.sep !== null || state.depth !== 0) return false;
    if (state.upper !== null) return ONE_LETTER.has(state.upper);
    return state.done[0] && (state.prev === 'elem' || state.prev === 'close' || state.prev === 'digit');
  }

  function validPrefix(text) {
    let s = INITIAL;
    for (const ch of text) {
      s = advance(s, ch);
      if (s === null) return null;
    }
    return s;
  }

  return { INITIAL, advance, isComplete, validPrefix };
}

/** Plain per-column read: collapse repeats, drop blanks. */
export function greedy(logp, T, C, itos) {
  let out = '', prev = -1;
  for (let t = 0; t < T; t++) {
    let best = 0, bv = logp[t * C];
    for (let c = 1; c < C; c++) {
      const v = logp[t * C + c];
      if (v > bv) { bv = v; best = c; }
    }
    if (best !== prev && best !== 0) out += itos[best];
    prev = best;
  }
  return out;
}

/** Sum over columns of the best log-probability (the single best path). */
export function bestPathLogp(logp, T, C) {
  let total = 0;
  for (let t = 0; t < T; t++) {
    let bv = logp[t * C];
    for (let c = 1; c < C; c++) if (logp[t * C + c] > bv) bv = logp[t * C + c];
    total += bv;
  }
  return total;
}

/**
 * CTC prefix beam search with grammar-pruned extensions.
 * @param logp Float32Array of T*C log-probabilities, row-major; class 0 = blank
 * @returns up to `limit` complete valid lines [{text, logp}], most probable first
 */
export function beamCandidates(logp, T, C, itos, grammar,
  { beamWidth = 8, topk = 6, limit = 5 } = {}) {
  const k = Math.min(topk, C - 1);
  let beams = new Map();
  beams.set('', { prefix: [], pb: 0.0, pnb: NEG, gs: grammar.INITIAL });
  const cand = new Int32Array(k);

  for (let t = 0; t < T; t++) {
    const off = t * C;
    // the k most likely characters this column (blank excluded)
    for (let i = 0; i < k; i++) cand[i] = -1;
    for (let c = 1; c < C; c++) {
      const v = logp[off + c];
      let pos = k;
      while (pos > 0 && (cand[pos - 1] < 0 || logp[off + cand[pos - 1]] < v)) pos--;
      if (pos < k) {
        for (let j = k - 1; j > pos; j--) cand[j] = cand[j - 1];
        cand[pos] = c;
      }
    }
    const next = new Map();
    for (const [key, b] of beams) {
      const pTot = logAddExp(b.pb, b.pnb);
      const last = b.prefix.length ? b.prefix[b.prefix.length - 1] : -1;
      // stay on the same prefix: emit blank, or repeat the last character
      let e = next.get(key);
      if (!e) { e = { prefix: b.prefix, pb: NEG, pnb: NEG, gs: b.gs }; next.set(key, e); }
      e.pb = logAddExp(e.pb, pTot + logp[off]);
      if (last >= 0) e.pnb = logAddExp(e.pnb, b.pnb + logp[off + last]);
      // extend with a new character, if the grammar allows it
      for (let i = 0; i < k; i++) {
        const ci = cand[i];
        if (ci < 0) continue;
        const ngs = grammar.advance(b.gs, itos[ci]);
        if (ngs === null) continue;                    // chemically impossible -> prune
        const extKey = key + ',' + ci;
        // a repeated character needs a blank in between -> only pb flows
        const src = ci === last ? b.pb : pTot;
        let x = next.get(extKey);
        if (!x) { x = { prefix: b.prefix.concat(ci), pb: NEG, pnb: NEG, gs: ngs }; next.set(extKey, x); }
        x.pnb = logAddExp(x.pnb, src + logp[off + ci]);
      }
    }
    const ranked = [...next.entries()].sort(
      (a, b) => logAddExp(b[1].pb, b[1].pnb) - logAddExp(a[1].pb, a[1].pnb));
    beams = new Map(ranked.slice(0, beamWidth));
  }

  const out = [];
  const final = [...beams.values()].sort(
    (a, b) => logAddExp(b.pb, b.pnb) - logAddExp(a.pb, a.pnb));
  for (const b of final) {
    if (b.prefix.length && grammar.isComplete(b.gs)) {
      out.push({ text: b.prefix.map((i) => itos[i]).join(''), logp: logAddExp(b.pb, b.pnb) });
      if (out.length >= limit) break;
    }
  }
  return out;
}

/** Apply the lexicon prior: a reading that spells a known compound gets `bonus`
 *  (in log-probability units) over one that is merely well-formed. */
export function chooseReading(cands, isKnown, bonus = 2.0) {
  let best = null, bestScore = -Infinity;
  for (const c of cands) {
    const score = c.logp + (isKnown(c.text) ? bonus : 0);
    if (score > bestScore) { bestScore = score; best = c; }
  }
  return best;
}
