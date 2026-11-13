// Electronics mode: two helpers for someone with a soldering iron in one hand.
//
//   Resistor  hold the resistor inside the box -> its value from the colour bands
//             (model trained in backend/resistor/, see resistor.js)
//   Chip      point at the printing on a chip or transistor -> what it is and
//             its pinout (text reader + static/data/parts.json)
//
// The view is digitally zoomed so a small part is easy to aim at, and the
// camera is asked for more pixels while this mode is open.

import { UI } from './icons.js';
import { BandVote, CSS, decodeBands, formatOhms, ResistorReader } from './resistor.js';

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const POWER = /^(VCC|VDD|V\+|VS|\+VS|AVCC|VS \(motor\)|VSS \(logic 5 V\)|VBAT|IN)$/;
const GROUND = /^(GND|VSS|V-|VEE|DGND|AGND|GND \/ ADJ)$/;

/** Marking text -> the part it names, or null. Tries each printed word, and
 *  the whole line run together ("LM 358 N"), with the usual misreads undone. */
export function findPart(parts, lines) {
  const sorted = [...lines].sort((a, b) => b.lineHeight - a.lineHeight);
  for (const line of sorted) {
    const words = line.text.toUpperCase().split(/\s+/).map((w) => w.replace(/[^A-Z0-9]/g, '')).filter(Boolean);
    const candidates = [...words, words.join(''), ...words.slice(0, -1).map((w, i) => w + words[i + 1])];
    for (const c of candidates) {
      if (c.length < 3) continue;
      for (const v of new Set([c, c.replace(/O/g, '0'), c.replace(/O/g, '0').replace(/I/g, '1'), c.replace(/^8C/, 'BC')])) {
        for (const p of parts) {
          const m = p.regex.exec(v);
          if (m) return { part: p, marking: v, groups: m.slice(1), conf: line.conf };
        }
      }
    }
  }
  return null;
}

/** "~RESET" in running text -> RESET with a bar over it (active-low). */
const bar = (html) => html.replace(/~([A-Z0-9]+)/g, '<span style="text-decoration:overline">$1</span>');

const fill = (text, groups) => String(text).replace(/\{(\d)\}/g, (_, k) => {
  const g = groups?.[Number(k) - 1];
  return g === undefined ? 'xx' : String(Number(g));
});

function pinLabel(name, x, y, anchor) {
  const colour = POWER.test(name) ? '#ff9a9a' : GROUND.test(name) ? '#8fc0ff' : '#eef3ff';
  const parts = name.split(' ').map((w) => (w.startsWith('~')
    ? `<tspan text-decoration="overline">${esc(w.slice(1))}</tspan>` : esc(w))).join(' ');
  return `<text x="${x}" y="${y}" text-anchor="${anchor}" fill="${colour}" font-size="12" font-weight="600" dominant-baseline="middle">${parts}</text>`;
}

/** A drawing of the part with every pin named. */
export function pinoutSvg(part) {
  const pins = part.pins, n = pins.length;
  if (part.package.startsWith('DIP')) {
    const rows = n / 2, rh = 24, bw = 96, lw = 132, x0 = lw + 16, y0 = 16, bh = rows * rh + 8;
    let s = `<rect x="${x0}" y="${y0}" width="${bw}" height="${bh}" rx="7" fill="#1b2130" stroke="#6b7a99" stroke-width="1.5"/>
      <path d="M${x0 + bw / 2 - 11} ${y0} a11 11 0 0 0 22 0" fill="#0b0f18" stroke="#6b7a99" stroke-width="1.5"/>
      <circle cx="${x0 + 12}" cy="${y0 + 14}" r="3.5" fill="#6b7a99"/>`;
    for (let i = 0; i < rows; i++) {
      const y = y0 + 4 + rh * (i + 0.5), right = n - 1 - i;
      s += `<rect x="${x0 - 16}" y="${y - 4}" width="16" height="8" rx="2" fill="#c9ced8"/>
        <rect x="${x0 + bw}" y="${y - 4}" width="16" height="8" rx="2" fill="#c9ced8"/>
        <text x="${x0 + 22}" y="${y}" fill="#93a4c4" font-size="11" dominant-baseline="middle">${i + 1}</text>
        <text x="${x0 + bw - 8}" y="${y}" fill="#93a4c4" font-size="11" text-anchor="end" dominant-baseline="middle">${right + 1}</text>`
        + pinLabel(pins[i], x0 - 22, y, 'end') + pinLabel(pins[right], x0 + bw + 22, y, 'start');
    }
    return `<svg viewBox="0 0 ${2 * lw + bw + 32} ${bh + 32}" role="img" aria-label="${esc(part.name)} pinout">${s}</svg>`;
  }
  // three-legged parts, seen from the front
  const tab = part.package === 'TO-220';
  const cx = [70, 140, 210];
  let s = tab
    ? `<rect x="62" y="6" width="156" height="50" rx="4" fill="#aeb4c0"/><circle cx="140" cy="28" r="10" fill="#0b0f18"/>
       <rect x="62" y="50" width="156" height="70" rx="5" fill="#1b2130" stroke="#6b7a99" stroke-width="1.5"/>`
    : `<path d="M82 120 V64 a58 50 0 0 1 116 0 V120 z" fill="#1b2130" stroke="#6b7a99" stroke-width="1.5"/>
       <path d="M92 76 H188" stroke="#3a455c" stroke-width="2"/>`;
  s += `<text x="140" y="${tab ? 92 : 100}" fill="#93a4c4" font-size="11" text-anchor="middle">${tab ? 'printed side' : 'flat side'} facing you</text>`;
  pins.forEach((p, i) => {
    const x = tab ? cx[i] : 104 + i * 36, stagger = i === 1 ? 26 : 0;
    s += `<rect x="${x - 3.5}" y="120" width="7" height="${46 + stagger}" fill="#c9ced8"/>
      <text x="${x}" y="${178 + stagger}" fill="#93a4c4" font-size="11" text-anchor="middle">${i + 1}</text>`
      + pinLabel(p, tab ? x : 140 + (i - 1) * 86, 196 + stagger, 'middle');
  });
  return `<svg viewBox="0 0 280 236" role="img" aria-label="${esc(part.name)} pinout">${s}</svg>`;
}

export class ElectronicsMode {
  constructor(app) {
    this.app = app;
    this.id = 'electronics';
    this.sub = app.opts.sub === 'chip' ? 'chip' : 'resistor';
    this.parts = null;
    this.resistor = null;
    this.reader = null;
    this.vote = new BandVote();
    this.value = null;            // {reading, at}
    this.chip = null;             // {part, marking, groups, at}
    this.chipVotes = [];
    this.lastRead = 0;
    this.lines = [];
    this.reads = 0;
    this.working = false;
    this.revision = 0;
  }

  async enter(root) {
    this.root = root;
    root.innerHTML = '<div class="hint-pill" id="eHint"></div><div class="panel result" id="eResult"></div>';
    this.app.setActions([
      { id: 'resistor', label: 'Resistor', icon: UI.resistor, run: () => this.setSub('resistor') },
      { id: 'chip', label: 'Chip', icon: UI.chip, run: () => this.setSub('chip') },
      { id: 'pinout', label: 'Pinout', icon: UI.list, run: () => (this.chip ? this.showPinout(this.chip.part, this.chip.groups) : this.browse()) },
      { id: 'browse', label: 'All parts', icon: UI.text, run: () => this.browse() },
    ]);
    this.moreCamera(true);
    await this.setSub(this.sub, true);
  }

  leave() {
    this.revision++;
    this.working = false;
    this.moreCamera(false);
    this.vote.clear();
    this.value = null;
    this.chip = null;
    this.chipVotes = [];
  }

  back() { return false; }

  /** Ask the camera for more pixels while small parts are being read. */
  moreCamera(on) {
    const track = this.app.video.srcObject?.getVideoTracks?.()[0];
    if (!track?.applyConstraints) return;
    const want = on ? { width: { ideal: 1920 }, height: { ideal: 1080 } } : { width: { ideal: 1280 }, height: { ideal: 720 } };
    track.applyConstraints(want).then(() => setTimeout(() => this.app.layout(), 300)).catch(() => {});
  }

  async setSub(sub, force = false) {
    if (!force && sub === this.sub) return;
    this.sub = sub;
    const revision = ++this.revision, generation = this.app.generation;
    const current = () => revision === this.revision && generation === this.app.generation && this.app.mode === this;
    this.error = null;
    this.working = false;
    this.value = null;
    this.chip = null;
    this.chipVotes = [];
    this.seeing = false;
    this.app.updateAction('resistor', { on: sub === 'resistor' });
    this.app.updateAction('chip', { on: sub === 'chip' });
    this.vote.clear();
    this.zoomFor = '';
    this.render(true);
    try {
      if (!this.parts) {
        const data = await fetch('/static/data/parts.json').then((r) => { if (!r.ok) throw new Error('parts list missing'); return r.json(); });
        this.parts = data.parts.map((p) => ({ ...p, regex: new RegExp(p.re) }));
      }
      if (!current()) return;
      if (sub === 'resistor' && !this.resistor) {
        this.app.busy('Loading the resistor reader…');
        this.resistor = await ResistorReader.load();
      }
      if (sub === 'chip') {
        this.app.busy('Loading the text reader…');
        // Another mode may have replaced a failed shared reader since our last visit.
        this.reader = await this.app.textReader();
      }
    } catch (err) {
      if (current()) this.error = err.message || String(err);
    } finally { if (current()) this.app.busy(null); }
    if (current()) this.render(true);
  }

  // -------------------------------------------------------------- geometry ---
  /** The box the part goes in, in source pixels. */
  box() {
    const g = this.app.geo;
    const cx = g.vw / 2, cy = g.vh / 2;
    if (this.sub === 'resistor') {
      const w = Math.round(0.125 * Math.max(g.vw, g.vh)), h = Math.round((w * 48) / 160);
      return { sx: cx - w / 2, sy: cy - h / 2, sw: w, sh: h };
    }
    // about 3 cm across with the part 10 cm from the camera: a DIP chip fills it
    const w = Math.min(g.vw, Math.round(0.24 * Math.max(g.vw, g.vh))), h = Math.round(w * 0.56);
    return { sx: cx - w / 2, sy: cy - h / 2, sw: w, sh: h };
  }

  /** Zoom so the box spans a good part of the screen whatever its shape. */
  fitZoom() {
    const g = this.app.geo;
    const key = `${this.sub}|${g.W}x${g.H}|${g.vw}x${g.vh}`;
    if (key === this.zoomFor) return;
    this.zoomFor = key;
    const b = this.box(), base = Math.max(g.W / g.vw, g.H / g.vh);
    const tall = this.sub === 'chip' ? 0.5 : 0.34;           // share of the screen height the box may take
    const fit = Math.min((0.62 * g.W) / (b.sw * base), (tall * g.H) / (b.sh * base));
    this.app.setZoom(Math.max(1, Math.min(6, fit)));
  }

  // ------------------------------------------------------------------ loop ---
  tick(now) {
    this.fitZoom();
    if (this.sub === 'resistor') this.tickResistor(now); else this.tickChip(now);
    if (now - (this.renderedAt || 0) > 400) this.render();
  }

  tickResistor(now) {
    if (!this.resistor || this.resistor.busy || now - this.lastRead < 170) return;
    this.lastRead = now;
    const generation = this.app.generation;
    const revision = this.revision;
    this.resistor.read(this.app.source, this.box()).then((r) => {
      if (!r || revision !== this.revision || generation !== this.app.generation || this.app.mode !== this || this.sub !== 'resistor') return;
      this.reads++;
      const ok = r.bands.length >= 3 && r.bands.length <= 5 && r.conf >= 0.55;
      this.seeing = r.bands.length > 0;
      const agreed = this.vote.push(ok ? r.bands : null);
      const reading = agreed && decodeBands(agreed);
      if (reading) this.value = { reading, at: performance.now() };
      this.render();
    }).catch((err) => console.warn('resistor read failed:', err.message));
  }

  tickChip(now) {
    if (!this.reader || !this.parts) return;
    const idle = this.chip && !this.app.motion.changed;
    if (!this.reader.due(now, idle ? 3000 : 600)) return;
    const generation = this.app.generation;
    const revision = this.revision;
    this.app.motion.mark();
    this.working = true;
    this.reader.read(this.app.source, this.box(), { workEdge: 640, detEdge: 320, maxLines: 4, maxUpscale: 2.5, minHeight: 10 })
      .then((res) => {
        if (!res || revision !== this.revision || generation !== this.app.generation || this.app.mode !== this || this.sub !== 'chip') return;
        this.app.stats.readMs = res.totalMs;
        this.reads++;
        this.lines = res.lines.filter((l) => l.conf >= 0.6);
        const hit = findPart(this.parts, this.lines);
        // believe a part after two reads, or one very clear one
        this.chipVotes.push(hit ? hit.part.id : '');
        if (this.chipVotes.length > 3) this.chipVotes.shift();
        if (hit && (hit.conf >= 0.92 || this.chipVotes.filter((v) => v === hit.part.id).length >= 2)) {
          this.chip = { ...hit, at: performance.now() };
        }
        this.render();
      })
      .catch((err) => console.warn('chip read failed:', err.message))
      .finally(() => { if (revision === this.revision) this.working = false; });
  }

  // ---------------------------------------------------------------- output ---
  draw(c, g, now) {
    const b = this.box();
    const x = (b.sx - g.sx) * g.scale, y = (b.sy - g.sy) * g.scale, w = b.sw * g.scale, h = b.sh * g.scale;
    const fresh = this.sub === 'resistor' ? this.value && now - this.value.at < 900 : this.chip && now - this.chip.at < 1500;
    // dim everything outside the box so the eye goes to it
    c.fillStyle = 'rgba(3,5,10,.45)';
    c.beginPath(); c.rect(0, 0, g.W, g.H); c.rect(x, y, w, h); c.fill('evenodd');
    this.app.brackets(c, x, y, w, h, fresh ? '#4dffa6' : '#ffffff');
  }

  render(force = false) {
    if (!this.root) return;
    const now = performance.now();
    this.renderedAt = now;
    // Keep a reading briefly for inspection, then retire it when the part is no longer seen.
    if (this.value && now - this.value.at > 5000) { this.value = null; this.vote.clear(); }
    if (this.chip && now - this.chip.at > 7000) { this.chip = null; this.chipVotes = []; }
    let hint, html;
    if (this.error) {
      hint = 'Not available';
      html = `<div class="grow"><div class="big" style="font-size:1.1rem">Could not load</div><div class="sub">${esc(this.error)}</div></div>`;
    } else if (this.sub === 'resistor') {
      const v = this.value;
      hint = !this.resistor ? 'Loading…' : v && now - v.at < 900 ? 'Got it' : this.seeing ? 'Hold steady…' : 'Fill the box with the resistor';
      if (v) {
        const r = v.reading, stale = now - v.at > 2500;
        const strip = r.bands.map((b, i) => `${r.bands.length >= 4 && i === r.bands.length - 1 ? '<i class="gap"></i>' : ''}<i style="background:${CSS[b]}" title="${b}"></i>`).join('');
        const alt = r.ambiguous ? ` · or ${formatOhms(r.alternative.ohms)} read from the other end` : '';
        const odd = r.standard ? '' : ' · not a standard value, check it';
        html = `<div class="bands">${strip}</div><div class="grow"><div class="big"${stale ? ' style="opacity:.6"' : ''}>${formatOhms(r.ohms)}
          <span style="font-size:.55em;color:var(--dim)">±${r.tolerance}%</span></div>
          <div class="sub">${esc(r.bands.join(' · '))}${esc(alt)}${odd}</div></div>`;
      } else {
        html = `<span class="v-icon dim">${UI.resistor}</span><div class="grow"><div class="big" style="font-size:1.15rem">Resistor reader</div>
          <div class="sub">Hold the resistor lengthwise inside the box, close to the camera, in good light.</div></div>`;
      }
    } else {
      const ch = this.chip;
      hint = !this.reader ? 'Loading…' : ch && now - ch.at < 1500 ? 'Got it' : 'Point at the printing on the part';
      if (ch) {
        html = `<span class="v-icon dim">${UI.chip}</span><div class="grow"><div class="big" style="font-size:1.25rem;white-space:normal">${esc(fill(ch.part.name, ch.groups))}</div>
          <div class="sub">${esc(ch.marking)} · ${esc(ch.part.package)} · ${esc(fill(ch.part.about, ch.groups))}</div></div>`;
      } else {
        const seen = this.lines.length ? `Read “${esc(this.lines[0].text)}”: not in the parts list. ` : '';
        html = `<span class="v-icon dim">${UI.chip}</span><div class="grow"><div class="big" style="font-size:1.15rem">Chip finder</div>
          <div class="sub">${seen}Point at the printing on a chip or transistor, or open All parts.</div></div>`;
      }
    }
    const key = hint + html;
    if (!force && key === this.renderKey) return;
    this.renderKey = key;
    this.root.querySelector('#eHint').textContent = hint;
    this.root.querySelector('#eResult').innerHTML = html;
    this.app.updateAction('pinout', { on: this.sub === 'chip' && !!this.chip });
  }

  showPinout(part, groups) {
    const html = `<div class="pinout">${pinoutSvg(part)}<dl class="facts">
      <dt>What it is</dt><dd>${esc(fill(part.about, groups))}</dd>
      <dt>Package</dt><dd>${esc(part.package)}</dd>
      <dt>Supply</dt><dd>${esc(part.supply)}</dd>
      <dt>Watch out</dt><dd>${bar(esc(part.tip))}</dd></dl></div>
      <p class="fine">Pin 1 is beside the notch or dot. Red = supply, blue = ground, a bar over a name = active-low.
        Check the maker's datasheet before relying on this: second-source parts occasionally differ.</p>`;
    this.app.openSheet(fill(part.name, groups), html);
  }

  browse() {
    if (!this.parts) return;
    const rows = [...this.parts].sort((a, b) => a.name.localeCompare(b.name, 'en', { numeric: true }))
      .map((p) => `<button class="row" data-nav data-part="${esc(p.id)}"><b>${esc(fill(p.name, null))}</b><span>${esc(p.package)}</span></button>`).join('');
    const sheet = this.app.openSheet('All parts', `<div class="rows">${rows}</div>`);
    sheet.querySelectorAll('[data-part]').forEach((b) => b.addEventListener('click', () => {
      this.showPinout(this.parts.find((p) => p.id === b.dataset.part), null);
    }));
  }

  state() {
    const r = this.value?.reading;
    return { sub: this.sub, reads: this.reads, ready: this.sub === 'resistor' ? !!this.resistor : !!this.reader,
      resistor: r ? { ohms: r.ohms, tolerance: r.tolerance, bands: r.bands, text: formatOhms(r.ohms), ambiguous: r.ambiguous } : null,
      chip: this.chip ? { id: this.chip.part.id, marking: this.chip.marking, name: fill(this.chip.part.name, this.chip.groups) } : null,
      lines: this.lines.map((l) => l.text), error: this.error || null };
  }
}
