// C Vision Assist — hands-free helper for smart glasses and phones.
//
// Three modes share one camera view and one shell:
//   safety.js        what is this product / chemical, and is it safe with that one?
//   electronics.js   resistor colour bands and chip pinouts
//   access.js        live captions and reading text aloud
//
// Built for a small display with no touchscreen (Vuzix M400: 640x360, a
// touchpad that sends arrow keys and Enter): every control is reachable with
// Left/Right/Enter, Back closes or pauses. It also works by touch on a phone.
// Everything heavy runs in workers and is loaded only for the mode in use.
//
// URL options (demos and tests):
//   ?mode=safety|electronics|access   start in this mode
//   ?autostart=1                      skip the start screen
//   ?src=<image or video url>         use a file instead of the camera
//   ?hand=0                           Safety: do not look for handwritten formulas
//   ?sub=chip|resistor|read|captions  sub-mode inside Electronics / Access
//   ?audio=<wav url>                  Access: caption this file instead of the microphone
//   ?threads=2                        use at most this many CPU threads for the models (cooler, slower)

import { UI } from './icons.js';

self.ORT_THREADS = 1;       // the page itself only runs tiny models; workers take the rest

const params = new URLSearchParams(location.search);
const $ = (id) => document.getElementById(id);
const SETTINGS_KEY = 'cvision-assist';

const MODES = [
  { id: 'safety', title: 'Safety', icon: UI.safety, blurb: 'Is this safe with that? Chemicals, cleaners, food, medicines.',
    load: () => import('./safety.js').then((m) => m.SafetyMode) },
  { id: 'electronics', title: 'Electronics', icon: UI.electronics, blurb: 'Read resistor colour bands and look up chip pinouts.',
    load: () => import('./electronics.js').then((m) => m.ElectronicsMode) },
  { id: 'access', title: 'Access', icon: UI.access, blurb: 'Live captions of speech, and text read aloud.',
    load: () => import('./access.js').then((m) => m.AccessMode) },
];

class Assist {
  constructor() {
    this.stage = $('stage');
    this.video = $('cam');
    this.still = $('still');
    this.hud = $('hud');
    this.hctx = this.hud.getContext('2d');
    this.modes = new Map();            // id -> mode instance (created on first use)
    this.mode = null;
    this.source = null;
    this.geo = null;
    this.zoom = 1;
    this.running = false;
    this.starting = false;
    this.generation = 0;
    this.modeTransition = 0;
    this.actions = [];
    this.stats = { fps: 0, readMs: 0 };
    this.opts = { handwriting: params.get('hand') !== '0', sub: params.get('sub'), audio: params.get('audio') };
    this.settings = { allergens: [], captionSize: 1, lastMode: 'safety', onlineLookup: false };
    try { Object.assign(this.settings, JSON.parse(localStorage.getItem(SETTINGS_KEY) || '{}')); } catch { /* private mode */ }
    this.reader = null;
    this.motion = new Motion();
    this.lastTick = 0;
    this.lastFrame = 0;
    this.toastTimer = 0;
  }

  // ------------------------------------------------------------------ setup ---
  init() {
    const want = params.get('mode');
    const first = MODES.some((m) => m.id === want) ? want : this.settings.lastMode;
    $('modes').innerHTML = MODES.map((m) => `<button class="mode-card" data-nav data-mode="${m.id}">${m.icon}
      <div><b>${m.title}</b><span>${m.blurb}</span></div></button>`).join('');
    $('tabs').innerHTML = MODES.map((m) => `<button class="tab" data-nav role="tab" data-mode="${m.id}">${m.icon}<span>${m.title}</span></button>`).join('');
    $('modes').querySelectorAll('[data-mode]').forEach((b) => b.addEventListener('click', () => this.start(b.dataset.mode)));
    $('tabs').querySelectorAll('[data-mode]').forEach((b) => b.addEventListener('click', () => this.setMode(b.dataset.mode)));
    $('sheetClose').addEventListener('click', () => this.closeSheet());
    document.addEventListener('keydown', (e) => this.onKey(e));
    // the system Back button (Android, and a WebView wrapper that calls goBack): close / pause, do not leave
    window.addEventListener('popstate', () => {
      if (!this.running) return;
      this.back();
      if (this.running) history.pushState({ assist: 1 }, '');
    });
    window.addEventListener('resize', () => this.layout());
    window.addEventListener('orientationchange', () => setTimeout(() => this.layout(), 300));
    window.addEventListener('pagehide', () => this.stop());
    document.addEventListener('visibilitychange', () => { if (document.hidden) this.stop(); });
    $('load').textContent = '';
    this.focusFirst($('modes').querySelector(`[data-mode="${first}"]`));
    if (params.get('autostart') === '1' || params.get('src')) this.start(first);
  }

  saveSettings(patch) {
    Object.assign(this.settings, patch);
    try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(this.settings)); } catch { /* private mode */ }
  }

  // ----------------------------------------------------------------- camera ---
  async start(modeId) {
    if (this.running || this.starting) return;
    this.starting = true;
    const generation = ++this.generation;
    $('err').textContent = '';
    $('load').textContent = 'Starting the camera…';
    const src = params.get('src');
    try {
      if (src && /\.(png|jpe?g|webp|gif|bmp)(\?|$)/i.test(src)) {
        this.still.src = src;
        await this.still.decode();
        if (generation !== this.generation) return;
        this.still.style.display = 'block';
        this.video.style.display = 'none';
        this.source = this.still;
      } else if (src) {
        this.video.src = src;
        this.video.loop = true;
        await this.video.play();
        if (generation !== this.generation) return;
        this.source = this.video;
      } else {
        if (!navigator.mediaDevices?.getUserMedia) throw new Error('this browser has no camera access');
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
        if (generation !== this.generation) { stream.getTracks().forEach((t) => t.stop()); return; }
        this.video.srcObject = stream;
        await this.video.play();
        if (generation !== this.generation) return;
        this.source = this.video;
        // the camera can be taken away (another app, sleep, unplugged): say so instead of freezing
        stream.getVideoTracks()[0]?.addEventListener('ended', () => {
          if (!this.running || generation !== this.generation) return;
          this.stop();
          $('err').textContent = 'The camera stopped. Choose a mode to start again.';
        });
      }
    } catch (err) {
      if (generation !== this.generation) return;
      this.video.srcObject?.getTracks().forEach((t) => t.stop());
      this.video.srcObject = null;
      $('err').textContent = `Camera error: ${err.message || err}. Open this page over HTTPS and allow camera access.`;
      $('load').textContent = '';
      return;
    } finally {
      if (generation === this.generation) this.starting = false;
    }
    if (generation !== this.generation) return;
    this.running = true;
    $('start').style.display = 'none';
    $('load').textContent = '';
    try { history.pushState({ assist: 1 }, ''); } catch { /* sandboxed: Back simply leaves */ }
    this.layout();
    try {
      const lock = await navigator.wakeLock?.request('screen');
      if (generation !== this.generation) { lock?.release?.().catch(() => {}); return; }
      this.wakeLock = lock;
    } catch { /* not allowed: fine */ }
    if (generation !== this.generation) return;
    this.lastFrame = performance.now();
    this.raf = requestAnimationFrame((t) => this.frame(t));
    await this.setMode(modeId);
    if (this.running && !this.navScope().includes(document.activeElement)) this.focusFirst($('tabs').querySelector('.tab.on'));
  }

  stop() {
    if (!this.running && !this.starting) return;
    this.generation++;
    this.running = false;
    this.starting = false;
    this.modeTransition++;
    this.switching = false;
    cancelAnimationFrame(this.raf);
    this.mode?.leave?.();
    this.mode = null;
    this.video.srcObject?.getTracks().forEach((t) => t.stop());
    this.video.srcObject = null;
    this.video.pause();
    this.source = null;
    this.geo = null;
    this.setZoom(1);
    this.closeSheet();
    this.busy(null);
    $('mode').innerHTML = '';
    $('actions').innerHTML = '';
    this.wakeLock?.release?.().catch(() => {});
    this.wakeLock = null;
    $('start').style.display = 'flex';
    this.focusFirst($('modes').querySelector(`[data-mode="${this.settings.lastMode}"]`));
  }

  async setMode(id) {
    if (!this.running || this.switching || this.mode?.id === id) return;
    const def = MODES.find((m) => m.id === id);
    if (!def) return;
    this.switching = true;
    const generation = this.generation;
    const transition = ++this.modeTransition;
    const current = () => generation === this.generation && transition === this.modeTransition && this.running;
    try {
      this.mode?.leave?.();
      this.mode = null;
      this.closeSheet();
      this.setZoom(1);
      $('mode').innerHTML = '';
      $('actions').innerHTML = '';
      $('tabs').querySelectorAll('.tab').forEach((t) => {
        t.classList.toggle('on', t.dataset.mode === id);
        t.setAttribute('aria-selected', t.dataset.mode === id);
      });
      this.saveSettings({ lastMode: id });
      let mode = this.modes.get(id);
      if (!mode) {
        this.busy(`Loading ${def.title}…`);
        const Mode = await def.load();
        mode = new Mode(this);
        this.modes.set(id, mode);
      }
      if (!current()) return;
      const root = document.createElement('div');
      root.className = 'mode-root';
      root.dataset.mode = id;
      $('mode').append(root);
      this.mode = mode;
      await mode.enter(root);
    } catch (err) {
      if (!current()) return;
      console.error(err);
      this.busy(null);
      $('mode').innerHTML = `<div class="panel dock" style="padding:.7rem .9rem"><b>${def.title} could not start</b>
        <p style="margin:.3rem 0 0;font-size:.82rem;color:#ffb1b1">${String(err.message || err).replace(/</g, '&lt;')}</p>
        <p style="margin:.3rem 0 0;font-size:.76rem;color:#a9b6cf">Check that the models are installed (tools/fetch_assist_models.py), then switch tabs to retry.</p></div>`;
      this.mode = null;
    } finally {
      if (current()) { this.switching = false; this.busy(null); }
    }
  }

  /** Digital zoom: the middle of the picture fills the screen (Electronics uses it). */
  setZoom(z) {
    this.zoom = z;
    const t = z === 1 ? '' : `scale(${z})`;
    this.video.style.transform = t;
    this.still.style.transform = t;
    this.layout();
  }

  sourceSize() {
    const s = this.source;
    return s === this.still ? [s.naturalWidth, s.naturalHeight] : [s.videoWidth, s.videoHeight];
  }

  /** How the source maps onto the screen (object-fit: cover, then zoom). */
  layout() {
    if (!this.source) return;
    const [vw, vh] = this.sourceSize();
    if (!vw || !vh) return;
    const W = this.stage.clientWidth, H = this.stage.clientHeight;
    const scale = Math.max(W / vw, H / vh) * this.zoom;
    const sw = W / scale, sh = H / scale;
    this.geo = { W, H, vw, vh, scale, sx: (vw - sw) / 2, sy: (vh - sh) / 2, sw, sh };
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (this.hud.width !== Math.round(W * dpr) || this.hud.height !== Math.round(H * dpr)) {
      this.hud.width = Math.round(W * dpr); this.hud.height = Math.round(H * dpr);
    }
    this.hctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  // ------------------------------------------------------------------- loop ---
  frame(now) {
    if (!this.running) return;
    this.raf = requestAnimationFrame((t) => this.frame(t));
    if (now - this.lastTick < 60) return;                 // ~15 updates a second is plenty for a HUD
    const dt = (now - this.lastFrame) / 1000;
    this.lastFrame = now; this.lastTick = now;
    if (dt > 0) this.stats.fps += (1 / dt - this.stats.fps) * 0.15;
    if (!this.geo) this.layout();
    const ready = this.geo && (this.source === this.still || this.video.readyState >= 2);
    const c = this.hctx;
    if (this.geo) c.clearRect(0, 0, this.geo.W, this.geo.H);
    if (ready && this.mode) {
      if (this.reader?.broken && !this.readerLoading && 'reader' in this.mode) {
        this.textReader().then((r) => { if (this.mode) this.mode.reader = r; }).catch(() => {});
      }
      this.motion.update(this.source, now);
      try {
        this.mode.tick(now);
        this.mode.draw?.(c, this.geo, now);
      } catch (err) { console.error('mode error:', err); }
    }
    this.updateStatus(now);
  }

  updateStatus(now) {
    if (now - (this.statusAt || 0) < 500) return;
    this.statusAt = now;
    const working = !!(this.reader?.busy || this.mode?.working);
    $('status').classList.toggle('work', working);
    const read = this.stats.readMs ? ` · read ${(this.stats.readMs / 1000).toFixed(1)} s` : '';
    $('status').querySelector('span').textContent = `on-device${read}`;
  }

  // ----------------------------------------------------------- shared tools ---
  /** The printed-text reader, shared by all modes and loaded on first use. */
  async textReader() {
    if (this.reader?.broken) { this.reader.dispose(); this.reader = null; }
    if (!this.reader) {
      this.readerLoading ||= import('./ocr.js').then((m) => m.TextReader.create());
      try { this.reader = await this.readerLoading; } finally { this.readerLoading = null; }
    }
    return this.reader;
  }

  setActions(list) {
    this.actions = list;
    const bar = $('actions');
    bar.innerHTML = `<button class="act" id="pause" data-nav title="Pause">${UI.pause}<span>Pause</span></button>`
      + list.map((a) => `<button class="act${a.on ? ' on' : ''}" data-nav data-act="${a.id}" title="${a.label}">${a.icon || ''}<span>${a.label}</span></button>`).join('');
    bar.querySelector('#pause').addEventListener('click', () => this.stop());
    for (const a of list) bar.querySelector(`[data-act="${a.id}"]`).addEventListener('click', () => a.run());
    // coming from the start screen the cursor has nowhere to be: put it on the mode's main button
    if (!this.navScope().includes(document.activeElement)) this.focusFirst(bar.querySelector('[data-act]'));
  }

  /** Update one action button's label / highlighted state without rebuilding the bar. */
  updateAction(id, { label, on, icon } = {}) {
    const b = $('actions').querySelector(`[data-act="${id}"]`);
    if (!b) return;
    if (on !== undefined) b.classList.toggle('on', on);
    if (label !== undefined) { b.querySelector('span').textContent = label; b.title = label; }
    if (icon !== undefined) { b.querySelector('svg')?.remove(); b.insertAdjacentHTML('afterbegin', icon); }
  }

  busy(text) {
    $('busy').hidden = !text;
    if (text) $('busy').textContent = text;
  }

  toast(text, ms = 1600) {
    const t = $('toast');
    t.textContent = text;
    t.classList.add('on');
    clearTimeout(this.toastTimer);
    this.toastTimer = setTimeout(() => t.classList.remove('on'), ms);
  }

  openSheet(title, html) {
    $('sheetTitle').textContent = title;
    $('sheetBody').innerHTML = html;
    $('sheetBody').scrollTop = 0;
    $('sheet').hidden = false;
    this.focusFirst($('sheetBody').querySelector('[data-nav]') || $('sheetClose'));
    return $('sheetBody');
  }

  closeSheet() {
    if ($('sheet').hidden) return false;
    $('sheet').hidden = true;
    $('sheetBody').innerHTML = '';
    this.focusFirst($('actions').querySelector('[data-nav]') || $('tabs').querySelector('.tab.on'));
    return true;
  }

  /** Corner brackets around something recognised, with an optional tag. */
  brackets(c, x, y, w, h, colour, tag) {
    const L = Math.max(6, Math.min(18, w / 4, h / 3));
    c.lineJoin = 'round'; c.lineCap = 'round';
    for (const [stroke, width] of [['rgba(0,0,0,.55)', 5], [colour, 2.5]]) {
      c.strokeStyle = stroke; c.lineWidth = width;
      c.beginPath();
      for (const [cx, cy, dx, dy] of [[x, y, 1, 1], [x + w, y, -1, 1], [x, y + h, 1, -1], [x + w, y + h, -1, -1]]) {
        c.moveTo(cx + dx * L, cy); c.lineTo(cx, cy); c.lineTo(cx, cy + dy * L);
      }
      c.stroke();
    }
    if (tag) {
      c.font = '800 13px system-ui, sans-serif';
      const tw = c.measureText(tag).width + 12, ty = Math.max(2, y - 22);
      c.fillStyle = colour;
      c.beginPath(); c.roundRect?.(x, ty, tw, 19, 5); if (!c.roundRect) c.rect(x, ty, tw, 19); c.fill();
      c.fillStyle = '#06101c';
      c.fillText(tag, x + 6, ty + 14);
    }
  }

  // ------------------------------------------------- touchpad / keyboard nav ---
  /** The controls the cursor can currently reach, in order. */
  navScope() {
    const root = !$('sheet').hidden ? $('sheet') : $('start').style.display !== 'none' ? $('start') : this.stage;
    return [...root.querySelectorAll('[data-nav]')].filter((el) => !el.disabled && el.offsetParent !== null
      && (root !== this.stage || !el.closest('#sheet')));
  }

  focusFirst(el) { el?.focus({ preventScroll: false }); }

  moveFocus(step) {
    const list = this.navScope();
    if (!list.length) return;
    const i = list.indexOf(document.activeElement);
    const next = list[i < 0 ? (step > 0 ? 0 : list.length - 1) : (i + step + list.length) % list.length];
    next.focus();
    next.scrollIntoView?.({ block: 'nearest', inline: 'nearest' });
  }

  back() {
    if (this.closeSheet()) return;
    if (this.mode?.back?.()) return;
    if (this.running) this.stop();
  }

  onKey(e) {
    if (e.altKey || e.ctrlKey || e.metaKey) return;
    if (this.mode?.onKey?.(e)) { e.preventDefault(); return; }
    // a sheet with nothing to select but Close: the arrows scroll the text instead
    if (!$('sheet').hidden && this.navScope().length <= 1 && e.key.startsWith('Arrow')) {
      const body = $('sheetBody');
      body.scrollBy({ top: (e.key === 'ArrowRight' || e.key === 'ArrowDown' ? 0.6 : -0.6) * body.clientHeight, behavior: 'smooth' });
      e.preventDefault();
      return;
    }
    switch (e.key) {
      case 'ArrowRight': case 'ArrowDown': this.moveFocus(1); break;
      case 'ArrowLeft': case 'ArrowUp': this.moveFocus(-1); break;
      case 'Escape': case 'Backspace': case 'BrowserBack': case 'GoBack': this.back(); break;
      case 'Enter': case ' ':
        if (document.activeElement?.matches?.('[data-nav]')) return;      // the browser clicks it
        this.navScope()[0]?.focus(); break;
      case '1': case '2': case '3':
        if (this.running) this.setMode(MODES[Number(e.key) - 1].id); else return;
        break;
      default: return;
    }
    e.preventDefault();
  }

  /** Snapshot for the browser tests. */
  state() {
    return { running: this.running, mode: this.mode?.id || null, zoom: this.zoom, stats: { ...this.stats },
      sheet: !$('sheet').hidden, actions: this.actions.map((a) => a.id),
      focus: document.activeElement?.dataset?.act || document.activeElement?.dataset?.mode || document.activeElement?.id || null,
      detail: this.mode?.state?.() || null };
  }
}

/** Has the view changed since the last time a mode looked? A 32x18 thumbnail
 *  compared now and then: cheap enough to run on the glasses all the time. */
class Motion {
  constructor() {
    this.canvas = document.createElement('canvas');
    this.canvas.width = 32; this.canvas.height = 18;
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
    this.now = null; this.ref = null; this.at = 0; this.changed = true;
  }

  update(source, now) {
    if (now - this.at < 400) return;
    this.at = now;
    try {
      this.ctx.drawImage(source, 0, 0, 32, 18);
      const d = this.ctx.getImageData(0, 0, 32, 18).data;
      const g = new Uint8Array(32 * 18);
      for (let i = 0; i < g.length; i++) g[i] = (d[i * 4] * 77 + d[i * 4 + 1] * 150 + d[i * 4 + 2] * 29) >> 8;
      this.now = g;
      if (!this.ref) { this.changed = true; return; }
      let sum = 0;
      for (let i = 0; i < g.length; i++) sum += Math.abs(g[i] - this.ref[i]);
      this.changed = sum / g.length > 6;
    } catch { this.changed = true; }
  }

  /** Remember the current view as "already looked at". */
  mark() { this.ref = this.now; this.changed = false; }
}

const app = new Assist();
window.assist = app;
app.init();
