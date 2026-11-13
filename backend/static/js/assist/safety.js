// Safety mode: "what is this, and is it safe next to that?"
//
//   text reader -> lines on the labels in view
//   splitClusters -> which lines belong to which package (left / right)
//   kb.findInLines -> what each package contains
//   barcode.js -> or, more surely, the package's barcode looked up in a product list
//   ItemTray -> remembers the last two things shown (A and B), so they can be
//               shown side by side OR one after the other
//   kb.judge -> danger / caution / info / no known problem, and why
//
// A substance is believed once it has been read twice, or once with a clear
// exact match, so a single misread does not put a warning on screen.

import { BarcodeFinder, ProductBook } from './barcode.js';
import { HAZARD, KIND, LEVEL_ICON, PPE, UI } from './icons.js';
import { LEVEL_ORDER, SafetyKB } from './kb.js';
import { readingOrder } from './ocr.js';

const HOLD_MS = 30000;          // an item stays in the tray this long after it was last seen
const PENDING_MS = 4000;        // an unconfirmed sighting is forgotten after this
const STALE_MS = 9000;          // a substance not re-read for this long (while its item was) is dropped
const SLOT_COLOUR = { A: '#5fd4ff', B: '#c69bff' };

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

/** Lines -> one or two groups, split at the widest empty vertical band. */
export function splitClusters(lines, frameW) {
  if (!lines.length) return [];
  const spans = lines.map((l) => [l.x, l.x + l.w]).sort((a, b) => a[0] - b[0]);
  let end = spans[0][1], gapAt = null, gapW = 0;
  for (const [a, b] of spans) {
    if (a - end > gapW) { gapW = a - end; gapAt = (a + end) / 2; }
    if (b > end) end = b;
  }
  if (gapAt === null || gapW < 0.06 * frameW) return [readingOrder(lines)];
  return [readingOrder(lines.filter((l) => l.cx < gapAt)), readingOrder(lines.filter((l) => l.cx >= gapAt))]
    .filter((g) => g.length);
}

function bounds(lines) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const l of lines) {
    x0 = Math.min(x0, l.x); y0 = Math.min(y0, l.y); x1 = Math.max(x1, l.x + l.w); y1 = Math.max(y1, l.y + l.h);
  }
  return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
}

function near(a, b) {
  const ix = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
  const iy = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
  const inter = ix * iy, union = a.w * a.h + b.w * b.h - inter;
  if (union > 0 && inter / union > 0.15) return true;
  const dx = a.x + a.w / 2 - (b.x + b.w / 2), dy = a.y + a.h / 2 - (b.y + b.h / 2);
  return Math.hypot(dx, dy) < 0.35 * Math.max(a.w, b.w);
}

/** The last two things shown to the camera. */
export class ItemTray {
  constructor(kb) {
    this.kb = kb;
    this.items = [];              // confirmed, at most two, each with .slot 'A' | 'B'
    this.pending = [];            // seen but not yet believed
    this.nextId = 1;
  }

  clear() { this.items = []; this.pending = []; }

  /**
   * @param clusters [{box, hits: [{sub, exact, conf, size}]}] — what this pass saw
   */
  update(clusters, now) {
    const seenNow = new Set();
    const pool = [...this.items, ...this.pending];
    const ordered = clusters.filter((c) => c.hits.some((h) => h.sub.kind !== 'label'))
      .sort((a, b) => b.box.w * b.box.h - a.box.w * a.box.h);
    for (const c of ordered) {
      let best = null, bestScore = 0;
      for (const it of pool) {
        if (seenNow.has(it)) continue;                       // two clusters in one pass are two things
        const shared = c.hits.filter((h) => it.subs.has(h.sub.id)).length;
        const close = it.box && now - it.lastSeen < 2500 && near(it.box, c.box);
        // the same place only counts for something still unconfirmed: a confirmed item that
        // shares nothing with this label has been swapped for a different package
        const score = (shared ? 2 + shared : 0) + (close && (shared || !it.confirmed) ? 2 : 0);
        if (score > bestScore) { best = it; bestScore = score; }
      }
      if (!best) {
        best = { id: this.nextId++, subs: new Map(), box: null, lastSeen: now, firstSeen: now, confirmed: false,
          slot: null, profile: null, name: '', formula: null };
        this.pending.push(best);
        pool.push(best);
      }
      seenNow.add(best);
      this.merge(best, c, now);
    }
    // promote what is now believed
    for (const it of [...this.pending]) {
      if (!it.confirmed) continue;
      if (this.items.length >= 2) {
        const victim = this.items.filter((x) => !seenNow.has(x)).sort((a, b) => a.lastSeen - b.lastSeen)[0];
        if (!victim) continue;                               // three things in view: keep the two we have
        this.items.splice(this.items.indexOf(victim), 1);
      }
      it.slot = this.items.some((x) => x.slot === 'A') ? 'B' : 'A';
      this.items.push(it);
      this.pending.splice(this.pending.indexOf(it), 1);
    }
    this.pending = this.pending.filter((it) => now - it.lastSeen <= PENDING_MS);
    this.items = this.items.filter((it) => now - it.lastSeen <= HOLD_MS);
    this.items.sort((a, b) => (a.slot < b.slot ? -1 : 1));
    return seenNow;
  }

  merge(item, cluster, now) {
    for (const h of cluster.hits) {
      const e = item.subs.get(h.sub.id) || { seen: 0, confirmed: false, size: 0, last: now };
      e.seen++; e.last = now; e.size = Math.max(e.size, h.size || 0);
      if ((h.exact && h.conf >= 0.85) || e.seen >= 2) e.confirmed = true;
      item.subs.set(h.sub.id, e);
      if (h.formula) item.formula = h.formula;
    }
    item.box = cluster.box; item.lastSeen = now;
    // forget substances that stopped being read while the package stayed in view
    const real = [...item.subs].filter(([id, e]) => e.confirmed && this.kb.subs.get(id)?.kind !== 'label');
    for (const [id, e] of [...item.subs]) {
      if (now - e.last > STALE_MS && !(real.length === 1 && real[0][0] === id)) item.subs.delete(id);
    }
    const ids = [...item.subs].filter(([, e]) => e.confirmed).map(([id]) => id);
    item.profile = this.kb.profile(ids);
    item.confirmed = item.profile.subs.length > 0;
    if (item.confirmed) {
      const main = item.profile.subs.slice().sort((a, b) => item.subs.get(b.id).size - item.subs.get(a.id).size)[0];
      item.name = main.name; item.icon = main.icon || 'unknown'; item.kind = main.kind;
    }
  }
}

export class SafetyMode {
  constructor(app) {
    this.app = app;
    this.id = 'safety';
    this.kb = null;
    this.tray = null;
    this.reader = null;
    this.formulas = null;            // optional handwritten-formula source
    this.visible = new Set();        // items seen in the latest pass
    this.verdict = null;
    this.passes = 0;
    this.lastLines = [];
    this.sig = '';
    this.unchanged = 0;
    this.detailsOpen = false;
    this.barcodes = new BarcodeFinder();
    this.book = null;                // barcode -> product
    this.lastScan = 0;
    this.codes = new Map();          // code -> when it was last dealt with
    this.products = [];              // codes recognised this session (for the tests)
    this.revision = 0;
  }

  async enter(root) {
    this.root = root;
    const revision = ++this.revision, generation = this.app.generation;
    const current = () => revision === this.revision && generation === this.app.generation && this.app.mode === this;
    root.innerHTML = `
      <div class="slots"><div class="slot" id="slotA"></div><div class="slot" id="slotB"></div></div>
      <div class="panel verdict" id="verdict" data-level="idle"></div>`;
    this.app.setActions([
      { id: 'details', label: 'Details', icon: UI.list, run: () => this.openDetails() },
      { id: 'clear', label: 'Clear', icon: UI.clear, run: () => this.clear() },
      { id: 'allergy', label: 'Allergies', icon: UI.allergy, run: () => this.openAllergies() },
      { id: 'online', label: 'Online lookup', icon: UI.barcode, on: !!this.app.settings.onlineLookup, run: () => this.toggleOnline() },
    ]);
    this.render(true);
    this.app.busy('Loading the label reader…');
    try {
      if (!this.kb) this.kb = await SafetyKB.load();
      if (!current()) return;
      if (!this.tray) this.tray = new ItemTray(this.kb);
      if (!this.book) this.book = await ProductBook.load();
      if (!current()) return;
      this.reader = await this.app.textReader();
      if (!current()) return;
      if (this.app.opts.handwriting && !this.formulas) {
        try {
          const { FormulaSource } = await import('./formulas.js');
          this.formulas = await FormulaSource.load();
        } catch (err) { console.warn('handwritten formulas unavailable:', err.message); }
      }
    } finally { if (current()) this.app.busy(null); }
    if (current()) this.render(true);
  }

  leave() { this.revision++; this.formulas?.reset(); }

  toggleOnline() {
    const on = !this.app.settings.onlineLookup;
    this.app.saveSettings({ onlineLookup: on });
    this.app.updateAction('online', { on });
    this.codes.clear();
    this.app.toast(on ? 'Online lookup on: barcode numbers go to Open Food Facts' : 'Online lookup off: local product list only', 3200);
  }

  clear() {
    this.revision++;
    this.codes.clear();
    this.products = [];
    this.lastLines = [];
    this.formulas?.reset();
    this.tray?.clear();
    this.visible = new Set();
    this.verdict = null;
    this.unchanged = 0;
    this.render(true);
    this.app.toast('Cleared');
  }

  // ----------------------------------------------------------------- loop ---
  tick(now) {
    if (!this.reader || !this.kb) return;
    if (this.formulas) this.formulas.tick(this.app, now);
    if (this.book && now - this.lastScan > 330) { this.lastScan = now; this.lookForBarcode(now); }
    // a steady view that has already been read twice with the same result: read rarely
    const idle = this.unchanged >= 2 && !this.app.motion.changed;
    if (this.reader.due(now, idle ? 4000 : 700)) this.pass(now);
    else if (this.formulas?.changed && !this.reader.busy) this.absorb(now, false);
    if (this.tray.items.some((it) => now - it.lastSeen > HOLD_MS)) { this.tray.update([], now); this.judge(); this.render(); }
  }

  async pass(now) {
    const g = this.app.geo;
    const generation = this.app.generation;
    const revision = this.revision;
    this.app.motion.mark();
    let res;
    try {
      res = await this.reader.read(this.app.source, { sx: g.sx, sy: g.sy, sw: g.sw, sh: g.sh },
        { workEdge: 960, detEdge: 480, maxLines: 6 });
    } catch (err) { console.warn('label read failed:', err.message); return; }
    if (!res || revision !== this.revision || generation !== this.app.generation || this.app.mode !== this) return;
    this.app.stats.readMs = res.totalMs;
    this.passes++;
    this.lastLines = res.lines.filter((l) => l.conf >= 0.6);
    this.absorb(performance.now());
  }

  /** A barcode in view? Look the product up and put it in the tray. */
  lookForBarcode(now) {
    const g = this.app.geo;
    let code = null;
    try { code = this.barcodes.scan(this.app.source, { sx: g.sx, sy: g.sy, sw: g.sw, sh: g.sh }, now); } catch { return; }
    if (!code || now - (this.codes.get(code) ?? -1e9) < 6000) return;
    this.codes.set(code, now);
    const generation = this.app.generation;
    const revision = this.revision;
    this.book.find(code, !!this.app.settings.onlineLookup).then((product) => {
      if (revision !== this.revision || generation !== this.app.generation || this.app.mode !== this) return;
      if (!product) {
        this.app.toast(this.app.settings.onlineLookup ? `Barcode ${code}: product not found`
          : `Barcode ${code} is not in the local list. Online lookup is off.`, 3200);
        return;
      }
      const sub = this.productSubstance(code, product);
      if (!sub) { this.app.toast(`${product.name}: nothing in the safety list`, 2600); return; }
      const again = this.products.some((x) => x.code === code);          // still in view: just keep it alive
      if (!again) this.products.push({ code, name: product.name, source: product.source });
      const t = performance.now();
      // (A pack read first by its words and then by its barcode shows up as two items.
      //  That is left alone on purpose: guessing that two sightings are the same pack
      //  could throw away a real second item and with it a real warning.)
      const box = { x: g.sx + g.sw * 0.3, y: g.sy + g.sh * 0.3, w: g.sw * 0.4, h: g.sh * 0.4 };
      const seen = this.tray.update([{ box, hits: [{ sub, exact: true, conf: 1, size: 100 }] }], t);
      this.visible = new Set([...this.visible, ...seen]);
      this.judge();
      this.render();
      if (!again) this.app.toast(`Barcode: ${product.name}`, 2200);
    }).catch((err) => console.warn('barcode lookup failed:', err.message));
  }

  /** A looked-up product as one substance the tray and the rules understand. */
  productSubstance(code, product) {
    const id = 'barcode:' + code;
    if (this.kb.subs.has(id)) return this.kb.subs.get(id);
    let ids = product.subs || [];
    if (!ids.length && product.text) {
      // name and category are the big print; the ingredient list is small print
      const lines = product.text.map((text, i) => ({ text, conf: 1, lineHeight: [100, 60, 30][i] || 30, x: 0, y: i * 100, w: 100, h: 30 }));
      ids = this.kb.findInLines(lines).map((h) => h.sub.id);
    }
    const p = this.kb.profile(ids);
    const allergens = [...new Set([...p.allergens, ...(product.allergens || [])])];
    if (!p.subs.length && !allergens.length) return null;
    const kind = p.kinds.find((k) => k === 'chemical') || p.kinds.find((k) => k === 'medicine') || p.kinds[0] || 'food';
    const sub = { id, name: product.name, kind, icon: p.subs[0]?.icon || 'box', tags: [...p.tags].filter((t) => !t.startsWith('kind:')),
      allergens, hazards: p.hazards, ppe: p.ppe, note: p.notes[0] };
    this.kb.subs.set(id, sub);
    return sub;
  }

  /** Turn the latest lines (and handwritten formulas) into tray updates.
   *  `fresh` is false when only the handwritten formulas changed: the printed
   *  lines are then old news and must not be counted a second time. */
  absorb(now, fresh = true) {
    const clusters = [];
    for (const group of fresh ? splitClusters(this.lastLines, this.app.geo.sw) : []) {
      const hits = this.kb.findInLines(group);
      const real = hits.filter((h) => h.sub.kind !== 'label');
      if (!real.length) continue;
      // the product name is the big print; ignore substances that only appear in small print
      const biggest = Math.max(...real.map((h) => h.size));
      clusters.push({ box: bounds(group), hits: hits.filter((h) => h.sub.kind === 'label' || h.size >= 0.5 * biggest) });
    }
    for (const f of this.formulas ? this.formulas.sightings(this.kb, this.lastLines) : []) clusters.push(f);
    if (fresh) {
      const sig = clusters.map((c) => c.hits.map((h) => h.sub.id).sort().join('+')).sort().join('|');
      this.unchanged = sig === this.sig ? this.unchanged + 1 : 0;
      this.sig = sig;
    }
    const seen = this.tray.update(clusters, now);
    this.visible = fresh ? seen : new Set([...this.visible, ...seen]);
    this.judge();
    this.render();
  }

  judge() {
    const [a, b] = this.tray.items;
    if (!a || !b) { this.verdict = null; return; }
    const v = this.kb.judge(a.profile, b.profile);
    // two formulas the chemistry engine knows a reaction for: say so
    const chem = this.formulas?.chem;
    if (chem && a.formula && b.formula && LEVEL_ORDER[v.level] < LEVEL_ORDER.caution) {
      const rx = chem.reactionFor([a.formula, b.formula]);
      if (rx) {
        const heat = rx.dH == null ? '' : rx.dH < 0 ? ` It releases about ${Math.abs(rx.dH)} kJ of heat.` : ` It absorbs about ${rx.dH} kJ of heat.`;
        v.findings.unshift({ id: 'chem-reaction', level: 'caution', title: 'These react',
          why: `${rx.equation}${rx.name ? ' (' + rx.name + ')' : ''}.${heat}`,
          do: 'Only combine them as part of a planned experiment, with the right protection.',
          source: 'This app\'s reaction list' });
        v.level = 'caution'; v.headline = this.kb.data.levels.caution;
      }
    }
    this.verdict = v;
  }

  // --------------------------------------------------------------- output ---
  draw(c, g, now) {
    for (const it of this.tray?.items || []) {
      if (!it.box || now - it.lastSeen > 1800) continue;
      const x = (it.box.x - g.sx) * g.scale, y = (it.box.y - g.sy) * g.scale;
      const w = it.box.w * g.scale, h = it.box.h * g.scale, pad = 8;
      this.app.brackets(c, x - pad, y - pad, w + 2 * pad, h + 2 * pad, SLOT_COLOUR[it.slot], it.slot);
    }
    this.formulas?.draw(c, g);
  }

  slotHtml(item, letter) {
    if (!item) {
      return `<div class="slot-empty"><span class="badge" style="--c:${SLOT_COLOUR[letter]}">${letter}</span>
        <span>${letter === 'A' ? 'Show a label' : 'Show a second item'}</span></div>`;
    }
    const p = item.profile, mine = this.app.settings.allergens;
    const hazards = p.hazards.map((h) => `<i class="ghs" title="${esc(HAZARD[h]?.label || h)}">${HAZARD[h]?.icon || ''}</i>`).join('');
    const ppe = p.ppe.map((x) => `<i class="ppe" title="${esc(PPE[x]?.label || x)}">${PPE[x]?.icon || ''}</i>`).join('');
    const alarm = p.allergens.filter((a) => mine.includes(a));
    const allergens = alarm.length
      ? `<span class="chip alarm">${UI.allergy} Your allergy: ${esc(alarm.join(', '))}</span>`
      : p.allergens.length ? `<span class="chip">Contains ${esc(p.allergens.join(', '))}</span>` : '';
    const more = p.subs.length > 1 ? `<span class="chip">+ ${esc(p.subs.filter((s) => s.name !== item.name).map((s) => s.name).join(', '))}</span>` : '';
    const live = this.visible.has(item) ? ' live' : '';
    return `<div class="slot-card${live}${alarm.length ? ' alarm' : ''}" style="--c:${SLOT_COLOUR[letter]}">
      <span class="badge">${letter}</span><span class="kind">${KIND[item.icon] || KIND.unknown}</span>
      <div class="who"><b>${esc(item.name)}</b>
        <div class="marks">${hazards}${ppe}${allergens}${more}</div></div></div>`;
  }

  render(force = false) {
    if (!this.root) return;
    const [a, b] = this.tray ? this.tray.items : [];
    const A = this.tray?.items.find((x) => x.slot === 'A'), B = this.tray?.items.find((x) => x.slot === 'B');
    const v = this.verdict;
    const key = JSON.stringify([A?.id, A?.name, A?.profile?.subs.map((s) => s.id), this.visible.has(A),
      B?.id, B?.name, B?.profile?.subs.map((s) => s.id), this.visible.has(B), v?.level, v?.findings.map((f) => f.id),
      !!this.reader, this.app.settings.allergens]);
    if (!force && key === this.renderKey) return;
    this.renderKey = key;
    this.root.querySelector('#slotA').innerHTML = this.slotHtml(A, 'A');
    this.root.querySelector('#slotB').innerHTML = this.slotHtml(B, 'B');
    const el = this.root.querySelector('#verdict');
    if (v) {
      const top = v.findings[0];
      el.dataset.level = v.level;
      el.innerHTML = `<div class="v-head"><span class="v-icon">${LEVEL_ICON[v.level]}</span>
          <div><div class="v-level">${esc(v.headline)}</div><div class="v-title">${esc(top ? top.title
    : `${a.name} + ${b.name}`)}</div></div></div>
        <p class="v-why">${esc(top ? top.why
    : 'Nothing in this app\'s list says these are a problem together. That is not a guarantee: read both labels.')}</p>
        ${top ? `<p class="v-do">${esc(top.do)}</p>` : ''}
        ${v.findings.length > 1 ? `<p class="v-more">+ ${v.findings.length - 1} more in Details</p>` : ''}`;
    } else {
      const one = a || b;
      el.dataset.level = 'idle';
      el.innerHTML = !this.reader
        ? '<p class="v-why">Getting ready…</p>'
        : one
          ? `<div class="v-head"><span class="v-icon dim">${UI.safety}</span><div><div class="v-level">One item so far</div>
              <div class="v-title">${esc(one.name)}</div></div></div>
             <p class="v-why">${esc(one.profile.notes[0] || 'Show a second item to check whether the two are safe together.')}</p>`
          : `<div class="v-head"><span class="v-icon dim">${UI.safety}</span><div><div class="v-level">Safety check</div>
              <div class="v-title">Point at a label</div></div></div>
             <p class="v-why">Show two products or chemicals, together or one after the other, to see if they are safe together.</p>`;
    }
  }

  openDetails() {
    const items = this.tray?.items || [];
    const v = this.verdict;
    let html = '';
    if (v) {
      html += `<div class="d-verdict" data-level="${v.level}"><span class="v-icon">${LEVEL_ICON[v.level]}</span><b>${esc(v.headline)}</b></div>`;
      if (!v.findings.length) html += '<p>Nothing in this app\'s list says these two are a problem together. That is not a guarantee: read both labels.</p>';
      for (const f of v.findings) {
        html += `<div class="finding" data-level="${f.level}"><h4><span class="v-icon">${LEVEL_ICON[f.level]}</span>${esc(f.title)}</h4>
          <p>${esc(f.why)}</p><p class="do">${esc(f.do)}</p><p class="src">Source: ${esc(f.source)}</p></div>`;
      }
    }
    for (const it of items) {
      const p = it.profile;
      html += `<div class="d-item"><h4><span class="badge" style="--c:${SLOT_COLOUR[it.slot]}">${it.slot}</span>${esc(p.subs.map((s) => s.name).join(' + '))}</h4>`;
      if (p.hazards.length) html += `<p class="marks">${p.hazards.map((h) => `<span class="chip"><i class="ghs">${HAZARD[h].icon}</i>${esc(HAZARD[h].label)}</span>`).join('')}</p>`;
      if (p.ppe.length) html += `<p class="marks">${p.ppe.map((x) => `<span class="chip"><i class="ppe">${PPE[x].icon}</i>${esc(PPE[x].label)}</span>`).join('')}</p>`;
      if (p.allergens.length) html += `<p>Allergens: ${esc(p.allergens.join(', '))}</p>`;
      for (const n of p.notes) html += `<p>${esc(n)}</p>`;
      html += '</div>';
    }
    if (!items.length) html = '<p>Nothing has been recognised yet. Point the camera at the front of a package, or at a chemical formula on a bottle.</p>';
    html += '<p class="fine">General safety information from a short curated list. It is not complete and is not medical or professional advice. Always read the label.</p>';
    this.app.openSheet('Safety details', html);
  }

  openAllergies() {
    const all = this.kb?.data.allergens || [];
    const mine = new Set(this.app.settings.allergens);
    const html = `<p>Tick what you are allergic to. Packages that mention it are flagged.</p>
      <div class="checks">${all.map((a) => `<button class="check${mine.has(a) ? ' on' : ''}" data-nav data-allergen="${esc(a)}"
        role="checkbox" aria-checked="${mine.has(a)}">${esc(a)}</button>`).join('')}</div>
      <p class="fine">This only looks for allergen words printed large on the front. It cannot read ingredient lists or "may contain" warnings. Always check the pack.</p>`;
    const sheet = this.app.openSheet('My allergies', html);
    sheet.querySelectorAll('[data-allergen]').forEach((btn) => btn.addEventListener('click', () => {
      const a = btn.dataset.allergen;
      const set = new Set(this.app.settings.allergens);
      if (set.has(a)) set.delete(a); else set.add(a);
      this.app.saveSettings({ allergens: [...set] });
      btn.classList.toggle('on', set.has(a));
      btn.setAttribute('aria-checked', set.has(a));
      this.render(true);
    }));
  }

  state() {
    return {
      ready: !!this.reader, passes: this.passes, products: this.products, scans: this.barcodes.reads,
      lines: this.lastLines.map((l) => l.text),
      items: (this.tray?.items || []).map((it) => ({ slot: it.slot, name: it.name, subs: it.profile.subs.map((s) => s.id),
        tags: [...it.profile.tags], allergens: it.profile.allergens, formula: it.formula })),
      verdict: this.verdict ? { level: this.verdict.level, rules: this.verdict.findings.map((f) => f.id) } : null,
    };
  }
}
