// Safety knowledge base: label text -> what the thing is -> is this pair safe?
//
// Pure logic over static/data/safety.json (no camera, no DOM), so it can be
// tested by itself:
//
//   findInLines(lines)   words and formulas on a label -> hits (which substances)
//   profile(ids)         substances on one package     -> its tags, hazards, allergens
//   judge(a, b)          two packages                  -> danger / caution / info / none + why
//
// Matching is deliberately conservative. A short word must match exactly; a
// longer one may differ by a letter (the text reader does misread); the longest
// phrase wins ("almond milk" is not "milk"); and a word that follows "do not
// mix with" or is followed by "free" is not counted as the contents.

const NEGATORS_BEFORE = new Set(['not', 'never', 'avoid', 'without', 'no', 'non']);
const LEVEL_ORDER = { danger: 3, caution: 2, info: 1, none: 0 };

/** Lower-case word tokens; digits that are really letters are mapped back. */
export function tokenize(text) {
  const out = [];
  for (const raw of text.toLowerCase().normalize('NFKD').replace(/[̀-ͯ]/g, '').split(/[^a-z0-9']+/)) {
    let t = raw.replace(/'/g, '');
    if (!t) continue;
    const letters = t.replace(/[0-9]/g, '').length;
    if (letters >= 3 && letters < t.length && letters / t.length >= 0.6) {
      t = t.replace(/0/g, 'o').replace(/1/g, 'l').replace(/5/g, 's');      // "b1each" -> "bleach"
    }
    out.push(t);
  }
  return out;
}

/** Formula comparison key that ignores the usual misreads (0/O, 1/l/I, case). */
export function formulaKey(text) {
  return text.replace(/[\s.,;:]+$/g, '').replace(/[₀-₉]/g, (c) => String(c.charCodeAt(0) - 0x2080))
    .replace(/0/g, 'O').replace(/[lI|]/g, '1').toUpperCase();
}

function tokenMatches(token, word) {
  if (token === word) return 2;                               // exact
  const n = word.length;
  if (n < 7 || token[0] !== word[0]) return 0;
  if (n < 10) {                                               // one wrong letter, same length
    if (token.length !== n) return 0;
    let diff = 0;
    for (let i = 0; i < n; i++) if (token[i] !== word[i] && ++diff > 1) return 0;
    return 1;
  }
  return Math.abs(token.length - n) <= 2 && editDistance(token, word, 2) <= 2 ? 1 : 0;
}

function editDistance(a, b, limit) {
  let prev = Array.from({ length: b.length + 1 }, (_, i) => i);
  for (let i = 1; i <= a.length; i++) {
    const cur = [i];
    let best = i;
    for (let j = 1; j <= b.length; j++) {
      cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
      if (cur[j] < best) best = cur[j];
    }
    if (best > limit) return limit + 1;
    prev = cur;
  }
  return prev[b.length];
}

export class SafetyKB {
  constructor(data) {
    this.data = data;
    this.subs = new Map(data.substances.map((s) => [s.id, s]));
    this.phrases = [];                                        // {words:[...], sub, chars}
    this.formulas = new Map();                                // key -> substance
    for (const s of data.substances) {
      for (const w of s.words || []) {
        const words = tokenize(w);
        if (words.length) this.phrases.push({ words, sub: s, chars: words.join('').length });
      }
      for (const f of s.formulas || []) this.formulas.set(formulaKey(f), s);
    }
    // longest phrases first, so "peanut butter" is tried before "butter"
    this.phrases.sort((a, b) => b.words.length - a.words.length || b.chars - a.chars);
    this.byFirst = new Map();
    for (const p of this.phrases) {
      const k = p.words[0][0];
      if (!this.byFirst.has(k)) this.byFirst.set(k, []);
      this.byFirst.get(k).push(p);
    }
  }

  static async load(url = '/static/data/safety.json') {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error('safety data missing');
    return new SafetyKB(await resp.json());
  }

  /** A formula (printed or handwritten) -> substance, or null. */
  formulaSub(text) {
    return this.formulas.get(formulaKey(text)) || null;
  }

  /**
   * Find substances named on a label.
   * @param lines [{text, conf, lineHeight, x, y, w, h}] in reading order
   * @returns hits [{sub, exact, conf, size, line, via: 'word' | 'formula'}]
   */
  findInLines(lines) {
    const tokens = [], hits = [];
    lines.forEach((line, li) => {
      for (const t of tokenize(line.text)) tokens.push({ t, li });
      // printed formulas: case and brackets matter, so look at the raw words
      for (const raw of line.text.split(/\s+/)) {
        const w = raw.replace(/^[^A-Za-z0-9(]+|[^A-Za-z0-9)]+$/g, '');
        if (w.length >= 3 && /[A-Z]/.test(w) && /^[A-Za-z0-9()]+$/.test(w) && !/^[a-z]+$/.test(w)) {
          const sub = this.formulaSub(w);
          if (sub) hits.push({ sub, exact: true, conf: line.conf, size: line.lineHeight, line: li, via: 'formula' });
        }
      }
    });
    return hits.concat(this.matchTokens(tokens, lines));
  }

  matchTokens(tokens, lines) {
    const hits = [];
    const used = new Uint8Array(tokens.length);
    const candidates = [];
    for (let i = 0; i < tokens.length; i++) {
      const list = this.byFirst.get(tokens[i].t[0]);
      if (!list) continue;
      for (const p of list) {
        if (i + p.words.length > tokens.length) continue;
        let quality = 2, ok = true;
        for (let k = 0; k < p.words.length; k++) {
          const q = tokenMatches(tokens[i + k].t, p.words[k]);
          if (!q) { ok = false; break; }
          if (q < quality) quality = q;
          // a phrase may wrap onto the next line, but not jump further
          if (k && tokens[i + k].li - tokens[i + k - 1].li > 1) { ok = false; break; }
        }
        if (ok) candidates.push({ i, p, quality });
      }
    }
    // longer phrases claim their words first
    candidates.sort((a, b) => b.p.words.length - a.p.words.length || b.quality - a.quality || b.p.chars - a.p.chars);
    for (const c of candidates) {
      const n = c.p.words.length;
      let free = true;
      for (let k = 0; k < n; k++) if (used[c.i + k]) { free = false; break; }
      if (!free) continue;
      for (let k = 0; k < n; k++) used[c.i + k] = 1;
      if (c.p.sub.kind !== 'label' && this.negated(tokens, c.i, n)) continue;
      const line = lines[tokens[c.i].li];
      hits.push({ sub: c.p.sub, exact: c.quality === 2, conf: line.conf, size: line.lineHeight,
        line: tokens[c.i].li, via: 'word' });
    }
    return hits;
  }

  /** "do not mix with bleach", "peanut free": the word is there, the stuff is not. */
  negated(tokens, i, n) {
    const li = tokens[i].li;
    const after = tokens[i + n];
    if (after && after.li === li && after.t === 'free') return true;
    for (let k = i - 1; k >= 0 && k >= i - 5; k--) {
      if (tokens[k].li !== li && tokens[k].li !== li - 1) break;
      if (NEGATORS_BEFORE.has(tokens[k].t)) return true;
    }
    return false;
  }

  /** What one package is, from the substances found on it. */
  profile(ids) {
    const subs = [...new Set(ids)].map((id) => this.subs.get(id)).filter(Boolean);
    const removed = new Set(subs.flatMap((s) => s.removes || []));
    // "isopropyl ALCOHOL" is not a drink and "alcohol free beer" is not alcohol: a substance
    // whose every tag has been cancelled by something else on the label is not there at all
    const real = subs.filter((s) => s.kind !== 'label' && !((s.tags || []).length && s.tags.every((t) => removed.has(t))));
    const tags = new Set(), allergens = new Set(), hazards = new Set(), ppe = new Set(), kinds = new Set();
    for (const s of real) {
      (s.tags || []).forEach((t) => tags.add(t));
      (s.allergens || []).forEach((a) => allergens.add(a));
      (s.hazards || []).forEach((h) => hazards.add(h));
      (s.ppe || []).forEach((p) => ppe.add(p));
      kinds.add(s.kind);
    }
    for (const s of subs) {
      for (const r of s.removes || []) {
        if (r.startsWith('allergen:')) allergens.delete(r.slice(9)); else tags.delete(r);
      }
    }
    kinds.forEach((k) => tags.add('kind:' + k));
    return { subs: real, labels: subs.filter((s) => s.kind === 'label'), tags, allergens: [...allergens],
      hazards: [...hazards], ppe: [...ppe], kinds: [...kinds],
      notes: real.map((s) => s.note).filter(Boolean) };
  }

  /**
   * Are these two packages safe together?
   * @returns {level, headline, findings: [rule, ...]} — the worst rule first.
   *   level 'none' means "nothing in this list", never "proven safe".
   */
  judge(a, b) {
    const found = [];
    for (const r of this.data.rules) {
      if ((a.tags.has(r.a) && b.tags.has(r.b)) || (a.tags.has(r.b) && b.tags.has(r.a))) found.push(r);
    }
    found.sort((x, y) => LEVEL_ORDER[y.level] - LEVEL_ORDER[x.level]);
    const groups = new Set(), findings = [];
    for (const r of found) {                                 // one rule per group: the worst
      if (r.group) { if (groups.has(r.group)) continue; groups.add(r.group); }
      findings.push(r);
    }
    const level = findings.length ? findings[0].level : 'none';
    return { level, headline: this.data.levels[level], findings };
  }
}

export { LEVEL_ORDER };
