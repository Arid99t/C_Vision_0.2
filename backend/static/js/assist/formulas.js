// Handwritten chemical formulas as a second source for Safety mode.
//
// Reuses the Chem_Vision AR pipeline unchanged (detect.js finds dark writing on
// a light label, recognizer.js reads it, chem.js knows what is a formula), at a
// gentler pace: the label on a reagent bottle does not move much. A formula is
// offered to the safety tray only after two agreeing reads.

import { loadChem, toSubscripts } from '../chem.js';
import { FormulaDetector } from '../detect.js';
import { rgbaToGray } from '../preprocess.js';
import { Tracker, vote } from '../track.js';

const EDGE = 384, EVERY_MS = 220, MIN_CONF = 0.5, MIN_HITS = 2;

export class FormulaSource {
  constructor(chem, recognizer) {
    this.chem = chem;
    this.recognizer = recognizer;
    this.detector = new FormulaDetector();
    this.tracker = new Tracker();
    this.proc = document.createElement('canvas');
    this.pctx = this.proc.getContext('2d', { willReadFrequently: true });
    this.crop = document.createElement('canvas');
    this.cctx = this.crop.getContext('2d', { willReadFrequently: true });
    this.last = 0;
    this.reading = false;
    this.f = 1;                    // detector pixels -> source pixels
    this.origin = { x: 0, y: 0 };
    this.changed = false;          // a formula appeared or went since the last look
    this.known = '';
    this.printed = [];             // lines the printed-text reader saw last (set by Safety mode)
    this.kb = null;
  }

  static async load() {
    const chem = await loadChem();
    const { Recognizer } = await import('../recognizer.js');
    return new FormulaSource(chem, await Recognizer.load(chem));
  }

  reset() { this.tracker.tracks = []; this.known = ''; }

  tick(app, now) {
    if (now - this.last < EVERY_MS) return;
    this.last = now;
    const g = app.geo;
    const pw = g.sw >= g.sh ? EDGE : Math.round((EDGE * g.sw) / g.sh), ph = g.sw >= g.sh ? Math.round((EDGE * g.sh) / g.sw) : EDGE;
    const w = Math.max(32, pw & ~1), h = Math.max(32, ph & ~1);
    if (this.proc.width !== w || this.proc.height !== h) { this.proc.width = w; this.proc.height = h; }
    this.pctx.drawImage(app.source, g.sx, g.sy, g.sw, g.sh, 0, 0, w, h);
    const rgba = this.pctx.getImageData(0, 0, w, h).data;
    this.tracker.update(this.detector.detect(rgba, w, h, { maxRegions: 3 }), now);
    this.f = g.sw / w;
    this.origin = { x: g.sx, y: g.sy };
    this.pump(app, now);
    const known = this.tracker.tracks.filter((t) => t.formula).map((t) => t.id + t.formula).join(',');
    if (known !== this.known) { this.known = known; this.changed = true; }
  }

  pump(app, now) {
    if (this.reading) return;
    let pick = null;
    for (const t of this.tracker.tracks) {
      if (t.hits < MIN_HITS || this.isPrintedWords(this.sourceBox(t))) continue;
      const due = t.formula ? now - t.lastRead >= 2500 : (t.rejects || 0) < 4 || now - t.lastRead >= 3000;
      if (due && (!pick || (!t.formula && pick.formula) || (!!t.formula === !!pick.formula && t.lastRead < pick.lastRead))) pick = t;
    }
    if (!pick) return;
    this.reading = true;
    pick.lastRead = now;
    const generation = app.generation;
    this.read(app, pick).then((res) => {
      if (!res || generation !== app.generation || !this.tracker.tracks.includes(pick)) return;
      if (!res.valid || !res.formula || res.formula.length < 2 || res.confidence < MIN_CONF) {
        pick.rejects = (pick.rejects || 0) + 1;
        return;
      }
      pick.rejects = 0;
      vote(pick, res.formula, res.confidence, 2);
    }).catch((err) => console.warn('formula read failed:', err.message)).finally(() => { this.reading = false; });
  }

  async read(app, track) {
    const b = track.box, f = this.f, [vw, vh] = app.sourceSize();
    const pad = Math.max(3, 0.12 * b.h);
    const x0 = Math.max(0, (b.x - pad) * f + this.origin.x), y0 = Math.max(0, (b.y - pad) * f + this.origin.y);
    const x1 = Math.min(vw, (b.x + b.w + pad) * f + this.origin.x), y1 = Math.min(vh, (b.y + b.h + pad) * f + this.origin.y);
    const sw = x1 - x0, sh = y1 - y0;
    if (sw < 4 || sh < 4) return null;
    let th = Math.max(8, Math.min(96, Math.round(sh)));
    let tw = Math.max(8, Math.round((sw * th) / sh));
    if (tw > 640) { th = Math.max(8, Math.round((th * 640) / tw)); tw = 640; }
    this.crop.width = tw; this.crop.height = th;
    this.cctx.imageSmoothingQuality = 'high';
    this.cctx.drawImage(app.source, x0, y0, sw, sh, 0, 0, tw, th);
    return this.recognizer.read(rgbaToGray(this.cctx.getImageData(0, 0, tw, th).data, tw * th), th, tw);
  }

  sourceBox(t) {
    return { x: t.smooth.x * this.f + this.origin.x, y: t.smooth.y * this.f + this.origin.y,
      w: t.smooth.w * this.f, h: t.smooth.h * this.f };
  }

  /**
   * Confirmed formulas as tray clusters. A formula the safety list does not
   * know still gets a card if the chemistry data can name it. Writing that the
   * printed-text reader has already read as ordinary words is skipped: it is
   * package text, not a formula.
   */
  sightings(kb, printedLines) {
    this.changed = false;
    this.kb = kb;
    this.printed = printedLines;
    const out = [];
    for (const t of this.tracker.tracks) {
      if (!t.formula || t.hits < MIN_HITS) continue;
      const box = this.sourceBox(t);
      if (this.isPrintedWords(box)) continue;
      let sub = kb.formulaSub(t.formula);
      if (!sub) {
        const d = this.chem.describe(t.formula);
        if (!d.name) continue;
        const id = 'formula:' + d.formula;
        sub = kb.subs.get(id);
        if (!sub) {
          sub = { id, name: `${d.name} (${d.pretty || toSubscripts(d.formula)})`, kind: 'chemical', icon: 'flask', tags: [],
            hazards: [], ppe: [], note: 'This app has no hazard data for this chemical. Treat it with care and ask your teacher.' };
          kb.subs.set(id, sub);
        }
      }
      out.push({ box, hits: [{ sub, exact: true, conf: Math.max(0.9, t.confidence), size: box.h, via: 'hand', formula: t.formula }] });
    }
    return out;
  }

  /** Is this writing something the printed-text reader already read as ordinary
   *  words? (Package text, not a formula: leave it to the label matcher.) A short
   *  scrawl the text reader could not make sense of is still tried as a formula. */
  isPrintedWords(box) {
    return this.printed.some((l) => {
      if (l.conf < 0.8 || overlap(box, l) <= 0.3 || this.kb?.formulaSub(l.text)) return false;
      const compact = l.text.replace(/\s+/g, '');
      if (this.chem.parseFormula(this.chem.normalizeOcr(compact)) !== null) return false;
      return compact.replace(/[^A-Za-z]/g, '').length >= 5 || (this.kb ? this.kb.findInLines([l]).length > 0 : false);
    });
  }

  draw(c, g) {
    c.font = '600 11px system-ui, sans-serif';
    for (const t of this.tracker.tracks) {
      if (t.hits < MIN_HITS || t.formula || (t.rejects || 0) >= 4) continue;
      const b = this.sourceBox(t);
      if (this.isPrintedWords(b)) continue;
      c.fillStyle = 'rgba(255,225,90,.95)';
      c.fillText('reading…', (b.x - g.sx) * g.scale, (b.y + b.h - g.sy) * g.scale + 13);
    }
  }
}

function overlap(a, b) {
  const ix = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
  const iy = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
  return (ix * iy) / Math.max(1, Math.min(a.w * a.h, b.w * b.h));
}
