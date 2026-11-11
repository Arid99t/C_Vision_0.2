// On-device chemistry: formula parsing, OCR clean-up and reaction lookup.
//
// JavaScript twin of backend/chemistry/formula.py + reactions.py, driven by the
// pre-built bundle (static/data/chem3d.json), which already holds every
// balanced reaction with its 3D animation plan. Nothing here needs the server.

const SUB = '₀₁₂₃₄₅₆₇₈₉';
const SUP = '⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻';
const SUP_ASCII = '0123456789+-';
const TOKEN_RE = /[A-Z][a-z]?|\d+|\(|\)/g;

export function toSubscripts(formula) {
  return formula.replace(/\d/g, (d) => SUB[+d]);
}

export class Chem {
  constructor(bundle) {
    this.bundle = bundle;
    this.elements = new Set(bundle.all_elements);
    this.known = new Set(bundle.known);
    this.aliases = bundle.aliases || {};
    this.molecules = bundle.molecules || {};
    this.reactions = bundle.reactions || {};
    this.byComposition = bundle.by_composition || {};
  }

  /** Canonical formula -> {element: count}, or null if it is not a valid formula. */
  parseFormula(formula) {
    if (!formula || formula.length > 256) return null;
    const tokens = formula.match(TOKEN_RE) || [];
    if (tokens.join('') !== formula) return null;       // stray symbols
    const stack = [new Map()];
    let i = 0;
    const n = tokens.length;
    const isNum = (t) => t !== undefined && /^\d+$/.test(t);
    while (i < n) {
      const tok = tokens[i];
      if (tok === '(') {
        stack.push(new Map());
        i++;
      } else if (tok === ')') {
        i++;
        let mult = 1;
        if (i < n && isNum(tokens[i])) {
          if (tokens[i].startsWith('0') || tokens[i].length > 5) return null;
          mult = parseInt(tokens[i], 10); i++;
        }
        const group = stack.pop();
        if (!stack.length || !group.size) return null;  // unbalanced ')' or empty group
        const top = stack[stack.length - 1];
        for (const [el, c] of group) top.set(el, (top.get(el) || 0) + c * mult);
      } else if (isNum(tok)) {
        return null;                                    // a number with nothing in front
      } else {
        if (!this.elements.has(tok)) return null;
        i++;
        let count = 1;
        if (i < n && isNum(tokens[i])) {
          if (tokens[i].startsWith('0') || tokens[i].length > 5) return null;
          count = parseInt(tokens[i], 10); i++;
        }
        const top = stack[stack.length - 1];
        top.set(tok, (top.get(tok) || 0) + count);
      }
      if ([...stack[stack.length - 1].values()].reduce((a, b) => a + b, 0) > 10000) return null;
    }
    if (stack.length !== 1 || !stack[0].size) return null;
    return Object.fromEntries(stack[0]);
  }

  /** Clean raw recognizer text into a canonical formula: unicode digits ->
   *  ASCII, no whitespace, and the classic 0/O mix-up resolved toward a known
   *  compound (else toward the reading with the fewest changes). */
  normalizeOcr(text) {
    if (text === null || text === undefined) return null;
    let t = '';
    for (const ch of text) {
      const a = SUB.indexOf(ch), b = SUP.indexOf(ch);
      t += a >= 0 ? String(a) : (b >= 0 ? SUP_ASCII[b] : ch);
    }
    t = t.replace(/\s+/g, '');
    if (!t) return t;
    if (t.length > 256) return t;
    const positions = [];
    for (let i = 0; i < t.length && positions.length < 10; i++) {
      if (t[i] === '0' || t[i] === 'O') positions.push(i);
    }
    const parseable = [];
    const total = 1 << positions.length;
    for (let mask = 0; mask < total; mask++) {
      const cand = t.split('');
      let changes = 0;
      // same order as itertools.product("0O", ...): last position varies fastest
      for (let p = 0; p < positions.length; p++) {
        const bit = (mask >> (positions.length - 1 - p)) & 1;
        const ch = bit ? 'O' : '0';
        if (ch !== t[positions[p]]) changes++;
        cand[positions[p]] = ch;
      }
      const s = cand.join('');
      if (this.parseFormula(s) !== null) parseable.push([s, changes]);
    }
    if (!parseable.length) return t;
    const known = parseable.filter((c) => this.known.has(c[0]));
    const pool = known.length ? known : parseable;
    let best = pool[0];
    for (const c of pool) if (c[1] < best[1]) best = c;
    return best[0];
  }

  /** Another spelling of a known compound -> its main spelling (C2H6O -> C2H5OH). */
  canonical(formula) {
    return this.aliases[formula] || formula;
  }

  /** True if every formula on a (possibly "A + B") line is a known compound. */
  isKnownLine(text) {
    return text.split('+').every((part) => {
      const p = part.trim();
      return p.length > 0 && this.known.has(this.normalizeOcr(p));
    });
  }

  /** 3D template for a formula: exact spelling, alias, or same composition. */
  template(formula) {
    const f = this.canonical(formula);
    if (this.molecules[f]) return this.molecules[f];
    const counts = this.parseFormula(f);
    if (!counts) return null;
    const key = Object.keys(counts).sort().map((el) => el + counts[el]).join('');
    const same = this.byComposition[key];
    return same ? this.molecules[same] : null;
  }

  /** Display description of one formula ("what's written"). */
  describe(raw) {
    const formula = this.normalizeOcr(raw);
    const counts = formula ? this.parseFormula(formula) : null;
    const tpl = counts ? this.template(formula) : null;
    return {
      input: raw, formula,
      pretty: formula ? toSubscripts(formula) : raw,
      valid: counts !== null,
      composition: counts || {},
      name: tpl ? tpl.name : null,
      template: tpl,
    };
  }

  /** Lookup key for a set of reactant formulas (order-independent). */
  reactionKey(formulas) {
    const norm = formulas.map((f) => this.canonical(this.normalizeOcr(f))).filter(Boolean);
    return [...new Set(norm)].sort().join('|');
  }

  /**
   * The reaction between the given reactants, or null. Entries from the bundle
   * carry `.plan` (the 3D animation); combustion of a C/H/O compound that is
   * not in the library is still balanced here, without a plan.
   */
  reactionFor(formulas) {
    const key = this.reactionKey(formulas);
    if (!key) return null;
    if (this.reactions[key]) return this.reactions[key];
    const parts = key.split('|');
    if (parts.length === 2 && parts.includes('O2')) {
      const fuel = parts.find((p) => p !== 'O2');
      const c = fuel ? this.parseFormula(fuel) : null;
      if (c && c.C && c.H && Object.keys(c).every((el) => 'CHO'.includes(el))) {
        return this._combustion(fuel, c, key);
      }
    }
    return null;
  }

  /** Decomposition of a single compound, if one is known. */
  decompositionOf(formula) {
    const r = this.reactions[this.reactionKey([formula])];
    return r || null;
  }

  _combustion(fuel, c, key) {
    // CxHyOz + (x + y/4 - z/2) O2 -> x CO2 + y/2 H2O, cleared of fractions
    const x = c.C, y = c.H, z = c.O || 0;
    let coef = [4, 4 * x + y - 2 * z, 4 * x, 2 * y];      // fuel, O2, CO2, H2O (times 4)
    if (coef[1] <= 0) return null;
    const gcd = (a, b) => (b ? gcd(b, a % b) : a);
    const g = coef.reduce(gcd);
    coef = coef.map((v) => v / g);
    const side = (pairs) => pairs.map(([f, n]) => (n === 1 ? '' : n + ' ') + toSubscripts(f)).join(' + ');
    const reactants = [[fuel, coef[0]], ['O2', coef[1]]];
    const products = [['CO2', coef[2]], ['H2O', coef[3]]];
    return {
      key, equation: `${side(reactants)} → ${side(products)}`,
      name: 'Combustion (hydrocarbon + O₂)', conditions: null, note: null, dH: null,
      source: 'rule', reactants, products,
    };
  }
}

export async function loadChem(url = '/static/data/chem3d.json') {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`could not load ${url}: ${resp.status}`);
  return new Chem(await resp.json());
}
