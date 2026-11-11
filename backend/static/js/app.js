// Chem_Vision AR — the phone app. Everything runs on the device:
//
//   camera frame -> detect.js (where is handwriting?) -> track.js (which is which)
//                -> recognizer.js (what does it say?)  -> chem.js (is it a formula?
//                   do two of them react?)             -> mol3d.js (show it in 3D)
//                -> objects.js (optional: what else is in the room?)
//
// The server only hands out these files. If the on-device formula model is
// missing, reading falls back to the server's /recognize endpoint.
//
// URL options (handy for demos and tests):
//   ?src=<image or video url>   use a file instead of the camera
//   ?autostart=1                skip the start screen
//   ?demo=Na,Cl2                label the writing found with these formulas instead of reading it
//   ?objects=1                  start with object labels on
//   ?objedge=480                object model input size (smaller = faster, misses small things)
//   ?edge=448  ?conf=0.5        detection size / minimum read confidence

import { loadChem } from './chem.js';
import { FormulaDetector } from './detect.js';
import { boxGap, Tracker, vote } from './track.js';
import { rgbaToGray } from './preprocess.js';
import { pretty, Scene3D } from './mol3d.js';

const params = new URLSearchParams(location.search);
function optionNumber(name, fallback, min, max) {
  const raw = params.get(name), value = Number(raw);
  return raw !== null && raw.trim() && Number.isFinite(value)
    ? Math.max(min, Math.min(max, value)) : fallback;
}
const CFG = {
  detectEdge: Math.round(optionNumber('edge', 448, 192, 960)),
  detectEveryMs: 45,                           // detect at most ~22 times a second
  minConfidence: optionNumber('conf', 0.5, 0, 1),
  stableReads: 2,                              // agreeing reads needed to show a label
  minHits: 2,                                  // frames a box must persist before it counts
  rereadMs: 1500,                              // re-check a confirmed label this often
  giveUpAfter: 4,                              // failed reads before a blob stops being shown
  proximity: 0.11,                             // "close together": gap / long edge
  releaseFactor: 1.9,                          // ...and this much further apart to let go
  holdMs: 900,                                 // grace period after the cards separate
  demo: params.get('demo') ? params.get('demo').split(',').map((s) => s.trim()).filter(Boolean) : null,
};

const $ = (id) => document.getElementById(id);
const PHASE = { gather: '1 · Reactants', break: '2 · Bonds break', move: '3 · Rearranging',
  form: '4 · Bonds form', done: '5 · Result' };

class App {
  constructor() {
    this.stage = $('stage');
    this.video = $('cam');
    this.still = $('still');
    this.hud = $('hud');
    this.hctx = this.hud.getContext('2d');
    this.proc = document.createElement('canvas');
    this.pctx = this.proc.getContext('2d', { willReadFrequently: true });
    this.crop = document.createElement('canvas');
    this.cctx = this.crop.getContext('2d', { willReadFrequently: true });
    this.detector = new FormulaDetector();
    this.tracker = new Tracker();
    this.source = null;                 // <video> or <img> being analysed
    this.geo = null;                    // mapping between source, detector and screen
    this.reading = false;
    this.reaction = null;               // {rx, ids, lastClose, solo}
    this.objects = null;
    this.objectList = [];
    this.on = { formulas: true, objects: params.get('objects') === '1', electrons: true };
    this.stats = { fps: 0, detectMs: 0, readMs: 0, reads: 0, mode: '…', objMs: 0 };
    this.lastDetect = 0;
    this.lastFrame = 0;
    this.hintUntil = 0;
    this.selected = null;
    this.demoNext = 0;
    this.running = false;
    this.starting = false;
    this.generation = 0;
    this.animationFrame = null;
  }

  // ------------------------------------------------------------- loading ---
  async init() {
    const say = (t) => { $('load').textContent = t; };
    try {
      say('Loading chemistry…');
      this.chem = await loadChem();
      this.scene = new Scene3D($('gl'), this.chem, $('labels'));
      const health = await fetch('/health').then((r) => r.json()).catch(() => ({}));
      say('Loading the formula reader…');
      try {
        const { Recognizer } = await import('./recognizer.js');
        this.recognizer = await Recognizer.load(this.chem);
        this.stats.mode = 'on-device';
      } catch (err) {
        console.warn('on-device reader unavailable:', err.message);
        this.recognizer = null;
        this.stats.mode = 'no reader';
        $('fallback').hidden = !health.recognizer_available;
        $('privacy').textContent = health.recognizer_available
          ? 'The on-device reader could not load. Enable server recognition below to send formula crops to this server.'
          : 'The formula reader is unavailable. Check the model and runtime files, then reload.';
      }
      $('tObjects').hidden = !health.on_device?.objects;
      if ($('tObjects').hidden) this.on.objects = false;
      say('');
      $('go').disabled = false;
      $('go').textContent = 'Start camera';
      this.bindUi();
      if (this.on.objects) this.loadObjects();
      if (params.get('autostart') === '1' || params.get('src')) this.start();
    } catch (err) {
      $('err').textContent = 'Could not load the app: ' + (err.message || err);
    }
  }

  bindUi() {
    $('go').addEventListener('click', () => this.start());
    $('pause').addEventListener('click', () => this.stop());
    $('serverFallback').addEventListener('change', (event) => {
      this.stats.mode = event.target.checked ? 'server' : 'no reader';
    });
    const toggle = (id, key, after) => $(id).addEventListener('click', () => {
      this.on[key] = !this.on[key];
      $(id).classList.toggle('on', this.on[key]);
      if (after) after();
    });
    toggle('tFormulas', 'formulas', () => { if (!this.on.formulas) { this.tracker.tracks = []; this.endReaction(); } });
    toggle('tObjects', 'objects', () => {
      if (this.on.objects) this.loadObjects();
      else { this.objectList = []; if (this.objects) this.objects.tracks = []; }
    });
    toggle('tElectrons', 'electrons', () => { this.scene.showElectrons = this.on.electrons; });
    $('tObjects').classList.toggle('on', this.on.objects);
    $('replay').addEventListener('click', () => this.scene.player?.restart());
    $('closeSheet').addEventListener('click', () => this.closeSheet());
    $('decompose').addEventListener('click', () => this.decomposeSelected());
    this.stage.addEventListener('pointerdown', (e) => this.onTap(e));
    window.addEventListener('resize', () => this.layout());
    window.addEventListener('orientationchange', () => setTimeout(() => this.layout(), 300));
    window.addEventListener('pagehide', () => this.stop());
    document.addEventListener('visibilitychange', () => { if (document.hidden) this.stop(); });
  }

  // -------------------------------------------------------------- source ---
  async start() {
    if (this.running || this.starting) return;
    this.starting = true;
    $('go').disabled = true;
    const generation = ++this.generation;
    $('err').textContent = '';
    const src = params.get('src');
    try {
      if (src && /\.(png|jpe?g|webp|gif|bmp)(\?|$)/i.test(src)) {
        this.still.src = src;
        await this.still.decode();
        this.still.style.display = 'block';
        this.video.style.display = 'none';
        this.source = this.still;
      } else if (src) {
        this.video.src = src;
        this.video.loop = true;
        await this.video.play();
        this.source = this.video;
      } else {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false });
        if (generation !== this.generation) { stream.getTracks().forEach((t) => t.stop()); return; }
        this.video.srcObject = stream;
        await this.video.play();
        this.source = this.video;
      }
    } catch (err) {
      this.video.srcObject?.getTracks().forEach((t) => t.stop());
      this.video.srcObject = null;
      $('err').textContent = 'Camera error: ' + (err.message || err)
        + '. Open this page over HTTPS and allow camera access.';
      return;
    } finally {
      this.starting = false;
      $('go').disabled = false;
    }
    if (generation !== this.generation) return;
    this.running = true;
    $('start').style.display = 'none';
    this.layout();
    this.lastFrame = performance.now();
    this.animationFrame = requestAnimationFrame((t) => this.frame(t));
  }

  stop() {
    this.generation++;
    this.running = false;
    cancelAnimationFrame(this.animationFrame);
    this.video.srcObject?.getTracks().forEach((t) => t.stop());
    this.video.srcObject = null;
    this.video.pause();
    this.source = null;
    this.geo = null;
    this.tracker.tracks = [];
    this.objectList = [];
    if (this.objects) this.objects.tracks = [];
    this.demoNext = 0;
    this.endReaction();
    this.closeSheet();
    this.scene?.syncMolecules([]);
    $('start').style.display = 'flex';
  }

  sourceSize() {
    const s = this.source;
    return s === this.still ? [s.naturalWidth, s.naturalHeight] : [s.videoWidth, s.videoHeight];
  }

  /** Work out how the source maps onto the screen (object-fit: cover) and onto
   *  the small frame the detector sees. */
  layout() {
    if (!this.source) return;
    const [vw, vh] = this.sourceSize();
    if (!vw || !vh) return;
    const W = this.stage.clientWidth, H = this.stage.clientHeight;
    const scale = Math.max(W / vw, H / vh);
    const sw = W / scale, sh = H / scale;
    let pw, ph;
    if (W >= H) { pw = CFG.detectEdge; ph = Math.round((CFG.detectEdge * H) / W); }
    else { ph = CFG.detectEdge; pw = Math.round((CFG.detectEdge * W) / H); }
    pw = Math.max(32, pw & ~1); ph = Math.max(32, ph & ~1);
    this.geo = { W, H, vw, vh, scale, sx: (vw - sw) / 2, sy: (vh - sh) / 2, sw, sh, pw, ph, k: W / pw };
    this.proc.width = pw; this.proc.height = ph;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.hud.width = Math.round(W * dpr); this.hud.height = Math.round(H * dpr);
    this.hctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.scene.resize(W, H, dpr);
  }

  // ---------------------------------------------------------------- loop ---
  frame(now) {
    if (!this.running) return;
    const dt = Math.min(0.1, (now - this.lastFrame) / 1000);
    this.lastFrame = now;
    if (dt > 0) this.stats.fps += (1 / dt - this.stats.fps) * 0.08;
    if (!this.geo) this.layout();
    const ready = this.geo && (this.source === this.still || this.video.readyState >= 2);
    if (ready) {
      if (this.on.formulas && now - this.lastDetect >= CFG.detectEveryMs) {
        this.lastDetect = now;
        this.detect(now);
        this.pumpReads(now);
        this.updateReaction(now);
      }
      if (this.on.objects && this.objects) this.pumpObjects(now);
      this.drawHud(now);
      this.syncScene();
    }
    if (this.scene) this.scene.update(dt);
    this.updatePanels(now);
    this.animationFrame = requestAnimationFrame((t) => this.frame(t));
  }

  detect(now) {
    const g = this.geo;
    this.pctx.drawImage(this.source, g.sx, g.sy, g.sw, g.sh, 0, 0, g.pw, g.ph);
    const rgba = this.pctx.getImageData(0, 0, g.pw, g.ph).data;
    const t0 = performance.now();
    const boxes = this.detector.detect(rgba, g.pw, g.ph, { maxRegions: 4 });
    this.stats.detectMs += (performance.now() - t0 - this.stats.detectMs) * 0.15;
    this.tracker.update(boxes, now);
  }

  // ------------------------------------------------------------- reading ---
  /** ?demo=CH4,H2O : label the detected writing with these formulas (top to
   *  bottom, left to right) instead of reading it — to demo or test a reaction. */
  assignDemo() {
    const waiting = this.tracker.tracks.filter((t) => t.hits >= CFG.minHits && !t.formula)
      .sort((a, b) => (a.box.y - b.box.y) || (a.box.x - b.box.x));
    for (const t of waiting) {
      if (this.demoNext >= CFG.demo.length) return;
      const d = this.chem.describe(CFG.demo[this.demoNext++]);
      Object.assign(t, { formula: d.formula, confidence: 1, pretty: d.pretty, name: d.name,
        template: d.template, composition: d.composition });
    }
  }

  pumpReads(now) {
    if (CFG.demo) { this.assignDemo(); return; }
    if (this.reading || this.stats.mode === 'no reader') return;
    const locked = this.reaction ? this.reaction.ids : [];
    let pick = null;
    for (const t of this.tracker.tracks) {
      if (t.hits < CFG.minHits || locked.includes(t.id)) continue;
      // confirmed labels are re-checked now and then; things that keep failing
      // to read as a formula (a cabinet, a keyboard) are retried only rarely
      const due = t.formula ? now - t.lastRead >= CFG.rereadMs
        : (t.rejects || 0) < CFG.giveUpAfter || now - t.lastRead >= 2500;
      if (!due) continue;
      // unconfirmed tracks first, then whoever has waited longest
      if (!pick || (!t.formula && pick.formula) || (!!t.formula === !!pick.formula && t.lastRead < pick.lastRead)) {
        pick = t;
      }
    }
    if (!pick) return;
    this.reading = true;
    const generation = this.generation;
    pick.lastRead = now;
    const t0 = performance.now();
    this.readTrack(pick)
      .then((res) => {
        this.stats.readMs += (performance.now() - t0 - this.stats.readMs) * 0.2;
        this.stats.reads++;
        if (res && generation === this.generation && this.on.formulas
            && this.tracker.tracks.includes(pick)) this.applyRead(pick, res);
      })
      .catch((err) => console.warn('read failed:', err))
      .finally(() => { this.reading = false; });
  }

  /** Cut the track's writing out of the full-resolution frame and read it. */
  async readTrack(track) {
    const g = this.geo, b = track.box;
    const pad = Math.max(3, 0.12 * b.h);
    const f = g.sw / g.pw;                                  // detector px -> source px
    const x0 = Math.max(0, (b.x - pad) * f + g.sx), y0 = Math.max(0, (b.y - pad) * f + g.sy);
    const x1 = Math.min(g.vw, (b.x + b.w + pad) * f + g.sx), y1 = Math.min(g.vh, (b.y + b.h + pad) * f + g.sy);
    const sw = x1 - x0, sh = y1 - y0;
    if (sw < 4 || sh < 4) return null;
    let th = Math.max(8, Math.min(96, Math.round(sh)));
    let tw = Math.max(8, Math.round((sw * th) / sh));
    if (tw > 640) { th = Math.max(8, Math.round((th * 640) / tw)); tw = 640; }
    this.crop.width = tw; this.crop.height = th;
    this.cctx.imageSmoothingQuality = 'high';
    this.cctx.drawImage(this.source, x0, y0, sw, sh, 0, 0, tw, th);
    if (this.recognizer) {
      const gray = rgbaToGray(this.cctx.getImageData(0, 0, tw, th).data, tw * th);
      return this.recognizer.read(gray, th, tw);
    }
    // no on-device model: let the server read the crop
    const blob = await new Promise((res) => this.crop.toBlob(res, 'image/png'));
    const fd = new FormData();
    fd.append('image', blob, 'crop.png');
    const resp = await fetch('/recognize', { method: 'POST', body: fd });
    if (!resp.ok) return null;
    const j = await resp.json();
    return { raw: j.raw, formula: j.formula, confidence: j.confidence, valid: !!j.describe?.valid };
  }

  applyRead(track, res) {
    track.lastRaw = res.raw;
    track.lastConf = res.confidence;
    if (!res.valid || !res.formula || res.formula.length < 2 || res.confidence < CFG.minConfidence) {
      track.rejects = (track.rejects || 0) + 1;
      return;
    }
    track.rejects = 0;
    if (vote(track, res.formula, res.confidence, CFG.stableReads)) {
      const d = this.chem.describe(track.formula);
      track.pretty = d.pretty;
      track.name = d.name;
      track.template = d.template;
      track.composition = d.composition;
    }
  }

  // ----------------------------------------------------------- reactions ---
  updateReaction(now) {
    const g = this.geo;
    const near = CFG.proximity * Math.max(g.pw, g.ph);
    const labeled = this.tracker.tracks.filter((t) => t.formula && t.hits >= CFG.minHits);
    if (this.reaction) {
      const r = this.reaction;
      const alive = r.ids.map((id) => labeled.find((t) => t.id === id)).filter(Boolean);
      if (r.solo) {
        if (alive.length) r.lastClose = now;
        if (now - r.lastClose > 3000) this.endReaction();
        return;
      }
      if (alive.length === 2 && boxGap(alive[0].box, alive[1].box) <= near * CFG.releaseFactor) r.lastClose = now;
      const playing = this.scene.player && this.scene.player.phase !== 'done';
      // let go once the cards have been apart for a moment (but never mid-animation)
      if (now - r.lastClose > (playing ? 6000 : CFG.holdMs)) this.endReaction();
      return;
    }
    let best = null;
    for (let i = 0; i < labeled.length; i++) {
      for (let j = i + 1; j < labeled.length; j++) {
        const gap = boxGap(labeled[i].box, labeled[j].box);
        if (gap <= near && (!best || gap < best.gap)) best = { gap, a: labeled[i], b: labeled[j] };
      }
    }
    if (!best) return;
    const rx = this.chem.reactionFor([best.a.formula, best.b.formula]);
    if (!rx) {
      this.hint(`No known reaction between ${pretty(best.a.formula)} and ${pretty(best.b.formula)}`, now);
      return;
    }
    this.beginReaction(rx, [best.a.id, best.b.id], now, false);
  }

  beginReaction(rx, ids, now, solo) {
    this.reaction = { rx, ids, lastClose: now, solo };
    this.closeSheet();
    if (rx.plan) this.scene.startReaction(rx, { x: this.geo.W / 2, y: this.geo.H * 0.45 });
    const b = $('banner');
    b.querySelector('.eq').textContent = rx.equation;
    b.querySelector('.nm').textContent = rx.name || '';
    const meta = [];
    if (rx.conditions) meta.push(`<span>${rx.conditions}</span>`);
    if (rx.dH != null) {
      meta.push(rx.dH < 0 ? `<span class="hot">releases ≈ ${Math.abs(rx.dH)} kJ of heat</span>`
        : `<span class="cold">absorbs ≈ ${rx.dH} kJ of heat</span>`);
      if (rx.thermochemistry?.status === 'reference_checked') {
        const phases = Object.entries(rx.thermochemistry.phases).map(([f, p]) => `${f}(${p})`).join(', ');
        meta.push(`<span>25 °C reference · ${phases}</span>`);
      } else meta.push('<span>unverified heat estimate</span>');
    }
    if (rx.note) meta.push(`<span>${rx.note}</span>`);
    if (!rx.plan) meta.push('<span>balanced equation only (no 3D structure for these)</span>');
    b.querySelector('.meta').innerHTML = meta.join('');
    b.classList.add('on');
    $('shade').classList.toggle('on', !!rx.plan);
  }

  endReaction() {
    if (!this.reaction) return;
    this.reaction = null;
    this.scene.endReaction();
    $('banner').classList.remove('on');
    $('caption').classList.remove('on');
    $('shade').classList.remove('on');
  }

  hint(text, now) {
    $('hint').textContent = text;
    $('hint').classList.add('on');
    this.hintUntil = now + 1800;
  }

  // ------------------------------------------------------------- objects ---
  async loadObjects() {
    if (this.objects || this.loadingObjects) return;
    this.loadingObjects = true;
    $('tObjects').disabled = true;
    $('tObjects').textContent = 'Loading…';
    try {
      const response = await fetch('/static/models/objects.json');
      if (!response.ok) throw new Error('object model metadata missing');
      const meta = await response.json();
      const { ObjectSpotter } = await import('./objects.js');
      this.objects = await ObjectSpotter.load(meta);
      this.objects.longEdge = Math.round(optionNumber('objedge', this.objects.longEdge, 192, 960));
    } catch (err) {
      this.on.objects = false;
      $('tObjects').classList.remove('on');
      this.hint('Object labels could not load. Tap Objects to retry.', performance.now());
      console.warn('object spotter unavailable:', err.message);
    } finally {
      this.loadingObjects = false;
      $('tObjects').disabled = false;
      $('tObjects').textContent = 'Objects';
    }
  }

  pumpObjects(now) {
    const o = this.objects;
    if (o.busy || now - o.lastRun < o.intervalMs) return;
    const g = this.geo;
    const generation = this.generation;
    o.run(this.source, g.sx, g.sy, g.sw, g.sh, now).then((list) => {
      if (this.on.objects && generation === this.generation) this.objectList = list;
      else o.tracks = [];
      this.stats.objMs = o.lastMs;
    }).catch((err) => console.warn('object spotting failed:', err));
  }

  // -------------------------------------------------------------- output ---
  /** Formulas as floating molecules; the two taking part in a reaction step aside. */
  syncScene() {
    const g = this.geo, k = g.k;
    const busy = this.reaction && this.reaction.rx.plan ? this.reaction.ids : [];
    const list = [];
    if (this.on.formulas) {
      for (const t of this.tracker.tracks) {
        if (!t.formula || t.hits < CFG.minHits) continue;
        list.push({ id: t.id, formula: t.formula, template: t.template, composition: t.composition,
          x: (t.smooth.x + t.smooth.w / 2) * k, y: t.smooth.y * k, size: t.smooth.h * k,
          hidden: busy.includes(t.id) });
      }
    }
    this.scene.syncMolecules(list);
  }

  drawHud(now) {
    const c = this.hctx, g = this.geo, k = g.k;
    c.clearRect(0, 0, g.W, g.H);
    c.lineJoin = 'round';
    // room objects: a quiet outline and a name tag
    if (this.on.objects) {
      c.font = '600 12px system-ui, sans-serif';
      for (const o of this.objectList) {
        const x = (o.x - g.sx) * g.scale, y = (o.y - g.sy) * g.scale, w = o.w * g.scale, h = o.h * g.scale;
        c.strokeStyle = `hsla(${o.hue}, 90%, 68%, .9)`;
        c.lineWidth = 2;
        c.strokeRect(x, y, w, h);
        const text = `${o.label} ${Math.round(o.score * 100)}%`;
        const tw = c.measureText(text).width + 10;
        const ty = y > 20 ? y - 19 : y + 1;
        c.fillStyle = `hsla(${o.hue}, 70%, 22%, .88)`;
        c.fillRect(x - 1, ty, tw, 18);
        c.fillStyle = '#fff';
        c.fillText(text, x + 4, ty + 13);
      }
    }
    if (!this.on.formulas) return;
    for (const t of this.tracker.tracks) {
      if (t.hits < CFG.minHits) continue;
      if (!t.formula && (t.rejects || 0) >= CFG.giveUpAfter) continue;   // not a formula: stay quiet
      const x = t.smooth.x * k, y = t.smooth.y * k, w = t.smooth.w * k, h = t.smooth.h * k;
      const ok = !!t.formula, L = Math.min(14, w / 3, h / 2);
      c.strokeStyle = ok ? 'rgba(77,255,166,.95)' : 'rgba(255,225,90,.85)';
      c.lineWidth = ok ? 2.5 : 1.5;
      c.beginPath();                                       // corner brackets
      for (const [cx, cy, dx, dy] of [[x, y, 1, 1], [x + w, y, -1, 1], [x, y + h, 1, -1], [x + w, y + h, -1, -1]]) {
        c.moveTo(cx + dx * L, cy); c.lineTo(cx, cy); c.lineTo(cx, cy + dy * L);
      }
      c.stroke();
      if (!ok) {
        c.font = '600 11px system-ui, sans-serif';
        c.fillStyle = 'rgba(255,225,90,.95)';
        c.fillText('reading…', x, y + h + 13);
      } else if (this.selected === t.id) {
        c.strokeStyle = 'rgba(255,255,255,.8)';
        c.lineWidth = 1;
        c.strokeRect(x - 3, y - 3, w + 6, h + 6);
      }
    }
  }

  updatePanels(now) {
    if (this.hintUntil && now > this.hintUntil) { $('hint').classList.remove('on'); this.hintUntil = 0; }
    const p = this.scene?.player;
    const cap = $('caption');
    if (p && this.reaction) {
      cap.classList.add('on');
      cap.querySelector('.step').textContent = PHASE[p.phase];
      cap.querySelector('.txt').textContent = p.caption;
      cap.querySelector('.bar i').style.width = (p.progress * 100).toFixed(1) + '%';
    } else {
      cap.classList.remove('on');
    }
    const s = this.stats;
    const obj = this.on.objects && this.objects ? ` · objects ${s.objMs.toFixed(0)} ms` : '';
    $('status').innerHTML = `${s.mode} · ${s.fps.toFixed(0)} fps<br>find ${s.detectMs.toFixed(1)} ms · read ${s.readMs.toFixed(0)} ms${obj}`;
  }

  // ----------------------------------------------------------------- taps ---
  onTap(e) {
    if (e.target.closest('button') || e.target.closest('#sheet') || !this.geo) return;
    const k = this.geo.k, r = this.stage.getBoundingClientRect();
    const px = (e.clientX - r.left) / k, py = (e.clientY - r.top) / k;
    const hit = this.tracker.tracks.find((t) => t.formula
      && px >= t.smooth.x - 12 && px <= t.smooth.x + t.smooth.w + 12
      && py >= t.smooth.y - 70 && py <= t.smooth.y + t.smooth.h + 12);
    if (!hit) { this.closeSheet(); return; }
    this.selected = hit.id;
    const sheet = $('sheet');
    sheet.querySelector('h3').textContent = hit.pretty || pretty(hit.formula);
    sheet.querySelector('.nm').textContent = hit.name || 'Not in the structure library';
    const comp = Object.entries(hit.composition || {}).map(([el, n]) => `${n} × ${el}`).join(', ');
    const dec = this.chem.decompositionOf(hit.formula);
    sheet.querySelector('.more').textContent = comp + (dec ? ` · can decompose: ${dec.equation}` : '');
    $('decompose').style.display = dec ? '' : 'none';
    sheet.classList.add('on');
  }

  closeSheet() {
    this.selected = null;
    $('sheet').classList.remove('on');
  }

  decomposeSelected() {
    const t = this.tracker.tracks.find((x) => x.id === this.selected);
    const rx = t && this.chem.decompositionOf(t.formula);
    if (rx) this.beginReaction(rx, [t.id], performance.now(), true);
  }

  /** Snapshot of what the app currently sees (used by the browser tests). */
  state() {
    const p = this.scene?.player;
    return {
      mode: this.stats.mode,
      tracks: this.tracker.tracks.map((t) => ({ id: t.id, formula: t.formula, name: t.name,
        confidence: +t.confidence.toFixed(3), hits: t.hits, raw: t.lastRaw || null,
        rawConfidence: t.lastConf == null ? null : +t.lastConf.toFixed(3),
        box: { x: t.box.x, y: t.box.y, w: t.box.w, h: t.box.h } })),
      reaction: this.reaction ? { key: this.reaction.rx.key, equation: this.reaction.rx.equation,
        phase: p ? p.phase : null, caption: p ? p.caption : null, plan: !!this.reaction.rx.plan } : null,
      molecules: this.scene ? this.scene.views.size : 0,
      objects: this.objectList.map((o) => ({ label: o.label, score: +o.score.toFixed(3) })),
      stats: { ...this.stats, reads: this.stats.reads },
    };
  }
}

const app = new App();
window.chemvision = app;
app.init();
