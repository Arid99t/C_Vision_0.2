// Barcodes (EAN-13 / UPC-A), read on the device from the camera picture.
//
// The words on a package are often stylised, curved round a can or in another
// language; its barcode is made to be read by a machine. So Safety mode also
// looks for one and, when it finds it, asks what the product is:
//
//   scanLine()     one row of pixels -> 13 digits, or null
//   BarcodeFinder  looks along a few rows and columns of the camera picture and
//                  only believes a code it has read twice
//   ProductBook    code -> product: first the local list (static/data/products.json,
//                  which a school can fill with its own items), then, only if the
//                  user switched it on, Open Food Facts over the internet
//
// EAN-13: start guard, six digits, centre guard, six digits, end guard. Each
// digit is two bars and two gaps, seven units wide in total. The first of the
// 13 digits is not drawn: it is carried by which of two patterns (L or G) each
// left-hand digit uses.

// bar/gap widths of the digits 0-9 in the "L" pattern (gap, bar, gap, bar)
const L = [[3, 2, 1, 1], [2, 2, 2, 1], [2, 1, 2, 2], [1, 4, 1, 1], [1, 1, 3, 2], [1, 2, 3, 1], [1, 1, 1, 4], [1, 3, 1, 2],
  [1, 2, 1, 3], [3, 1, 1, 2]];
// which left-hand digits use the mirrored "G" pattern, for each hidden first digit
const FIRST = ['000000', '001011', '001101', '001110', '010011', '011001', '011100', '010101', '010110', '011010'];

/** Does this 13-digit string have a correct check digit? */
export function validEan13(code) {
  if (!/^\d{13}$/.test(code)) return false;
  let sum = 0;
  for (let i = 0; i < 12; i++) sum += Number(code[i]) * (i % 2 ? 3 : 1);
  return (10 - (sum % 10)) % 10 === Number(code[12]);
}

/** Four run widths -> the digit they are closest to, and how far off. */
function matchDigit(runs, start, mirrored) {
  const total = runs[start] + runs[start + 1] + runs[start + 2] + runs[start + 3];
  if (!(total > 0)) return null;
  let best = -1, bestErr = Infinity;
  for (let d = 0; d < 10; d++) {
    let err = 0;
    for (let k = 0; k < 4; k++) {
      const want = mirrored ? L[d][3 - k] : L[d][k];
      const diff = (runs[start + k] * 7) / total - want;
      err += diff * diff;
    }
    if (err < bestErr) { bestErr = err; best = d; }
  }
  return { digit: best, err: bestErr };
}

/** Decode from a list of alternating run widths, runs[i] being the first bar of the start guard. */
function decodeAt(runs, i) {
  if (i + 59 > runs.length) return null;
  const unit = (runs[i] + runs[i + 1] + runs[i + 2]) / 3;
  for (let k = 0; k < 3; k++) if (runs[i + k] < 0.4 * unit || runs[i + k] > 1.9 * unit) return null;
  let digits = '', parity = '', err = 0;
  for (let d = 0; d < 6; d++) {
    const at = i + 3 + 4 * d;
    const plain = matchDigit(runs, at, false), mirror = matchDigit(runs, at, true);
    if (!plain) return null;
    const pick = plain.err <= mirror.err ? plain : mirror;
    digits += pick.digit; parity += pick === plain ? '0' : '1'; err += pick.err;
  }
  const first = FIRST.indexOf(parity);
  if (first < 0) return null;
  for (let d = 0; d < 6; d++) {
    const m = matchDigit(runs, i + 3 + 24 + 5 + 4 * d, false);
    if (!m) return null;
    digits += m.digit; err += m.err;
  }
  if (err / 12 > 0.55) return null;                                  // too far from any clean pattern
  const code = first + digits;
  return validEan13(code) ? code : null;
}

/**
 * Read a barcode along one line of grey levels (0-255, any direction).
 * @returns the 13 digits, or null
 */
export function scanLine(line) {
  const n = line.length;
  if (n < 120) return null;
  // local contrast threshold: midway between the darkest and lightest nearby
  const win = Math.max(16, Math.round(n / 14));
  const runs = [];
  let dark = false, runStart = 0, lo = 255, hi = 0;
  const mins = new Uint8Array(n), maxs = new Uint8Array(n);
  for (let x = 0; x < n; x++) {                                        // sliding min / max (simple, n is small)
    if (x % win === 0) { lo = 255; hi = 0; for (let k = x; k < Math.min(n, x + win); k++) { if (line[k] < lo) lo = line[k]; if (line[k] > hi) hi = line[k]; } }
    mins[x] = lo; maxs[x] = hi;
  }
  for (let x = 0; x < n; x++) {
    const a = Math.max(0, x - win), b = Math.min(n - 1, x + win);
    const l = Math.min(mins[a], mins[x], mins[b]), h = Math.max(maxs[a], maxs[x], maxs[b]);
    const isDark = h - l > 50 && line[x] < (l + h) / 2;
    if (isDark !== dark) { runs.push(x - runStart); runStart = x; dark = isDark; }
  }
  runs.push(n - runStart);
  // runs[0] is light; odd indices are bars
  for (const order of [runs, runs.slice().reverse()]) {
    const offset = order === runs ? 1 : (runs.length % 2 ? 1 : 0);    // index of the first bar in this direction
    for (let i = offset; i + 59 <= order.length; i += 2) {
      if (i > 0 && order[i - 1] < 2.2 * order[i]) continue;           // needs a quiet margin before the guard
      const code = decodeAt(order, i);
      if (code) return code;
    }
  }
  return null;
}

/** Looks for a barcode in the camera picture; reports a code once two reads agree. */
export class BarcodeFinder {
  constructor() {
    this.canvas = document.createElement('canvas');
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
    this.last = null;                 // {code, at} most recent single read
    this.reads = 0;
  }

  /**
   * @param region {sx, sy, sw, sh} of the source to search
   * @returns a confirmed 13-digit code, or null
   */
  scan(source, region, now) {
    const w = Math.min(1280, Math.round(region.sw)), h = Math.round((w * region.sh) / region.sw);
    if (this.canvas.width !== w || this.canvas.height !== h) { this.canvas.width = w; this.canvas.height = h; }
    this.ctx.drawImage(source, region.sx, region.sy, region.sw, region.sh, 0, 0, w, h);
    const tally = new Map();
    const note = (code) => { if (code) tally.set(code, (tally.get(code) || 0) + 1); };
    for (const f of [0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8]) {              // rows first: packs are usually held upright
      const d = this.ctx.getImageData(0, Math.round(f * h) - 1, w, 3).data;
      const line = new Uint8Array(w);
      for (let x = 0; x < w; x++) line[x] = (d[x * 4 + 1] + d[(w + x) * 4 + 1] + d[(2 * w + x) * 4 + 1]) / 3;
      note(scanLine(line));
    }
    if (!tally.size) {
      for (const f of [0.5, 0.35, 0.65]) {                              // then columns: a pack on its side
        const d = this.ctx.getImageData(Math.round(f * w) - 1, 0, 3, h).data;
        const line = new Uint8Array(h);
        for (let y = 0; y < h; y++) line[y] = (d[y * 12 + 1] + d[y * 12 + 5] + d[y * 12 + 9]) / 3;
        note(scanLine(line));
      }
    }
    this.reads++;
    let best = null;
    for (const [code, count] of tally) if (!best || count > best.count) best = { code, count };
    if (!best) return null;
    const confirmed = best.count >= 2 || (this.last && this.last.code === best.code && now - this.last.at < 4000);
    this.last = { code: best.code, at: now };
    return confirmed ? best.code : null;
  }
}

// Open Food Facts names for the allergens this app flags
const OFF_ALLERGENS = { 'en:milk': 'milk', 'en:eggs': 'eggs', 'en:peanuts': 'peanuts', 'en:nuts': 'tree nuts', 'en:soybeans': 'soy',
  'en:gluten': 'wheat', 'en:fish': 'fish', 'en:crustaceans': 'shellfish', 'en:molluscs': 'shellfish', 'en:sesame-seeds': 'sesame' };
const CACHE_KEY = 'cvision-assist-products';

/** Code -> product. Local list first; the internet only if switched on. */
export class ProductBook {
  constructor(local) {
    this.local = local;               // {code: {name, subs: [...], allergens?: [...]}}
    this.cache = {};
    try { this.cache = JSON.parse(localStorage.getItem(CACHE_KEY) || '{}'); } catch { /* private mode */ }
    this.pending = new Map();
  }

  static async load(url = '/static/data/products.json') {
    try {
      const resp = await fetch(url);
      return new ProductBook(resp.ok ? (await resp.json()).products || {} : {});
    } catch { return new ProductBook({}); }
  }

  /**
   * @returns {name, subs?, text?, allergens?, source: 'local' | 'online' | 'saved'} or null if unknown.
   *   `text` is label text to run through the safety matcher (online products).
   */
  async find(code, online) {
    // UPC-A codes are EAN-13 with a leading zero: accept the list written either way
    const hit = this.local[code] || (code[0] === '0' ? this.local[code.slice(1)] : null);
    if (hit) return { ...hit, source: 'local' };
    if (this.cache[code]) return { ...this.cache[code], source: 'saved' };
    if (!online) return null;
    if (!this.pending.has(code)) this.pending.set(code, this.lookup(code).finally(() => this.pending.delete(code)));
    return this.pending.get(code);
  }

  async lookup(code) {
    const fields = 'product_name,product_name_en,generic_name,generic_name_en,brands,categories,ingredients_text,ingredients_text_en,allergens_tags,traces_tags';
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 8000);
    try {
      const resp = await fetch(`https://world.openfoodfacts.org/api/v2/product/${code}.json?fields=${fields}`,
        { signal: ctrl.signal, credentials: 'omit', referrerPolicy: 'no-referrer' });
      if (!resp.ok) return null;
      const p = (await resp.json()).product;
      if (!p) return null;
      const name = p.product_name_en || p.product_name || p.generic_name_en || p.generic_name || p.brands || `Product ${code}`;
      const product = {
        name: String(name).slice(0, 60),
        // what the label would say, biggest print first: name, kind of product, then ingredients
        text: [[p.product_name_en, p.product_name, p.generic_name_en, p.generic_name].filter(Boolean).join(' . '),
          String(p.categories || ''), String(p.ingredients_text_en || p.ingredients_text || '')],
        allergens: [...new Set((p.allergens_tags || []).map((t) => OFF_ALLERGENS[t]).filter(Boolean))],
      };
      this.cache[code] = product;
      try { localStorage.setItem(CACHE_KEY, JSON.stringify(this.cache)); } catch { /* full or private */ }
      return { ...product, source: 'online' };
    } catch { return null; } finally { clearTimeout(timer); }
  }
}
