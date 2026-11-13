// Main-thread side of the printed-text reader (the work happens in
// ocr.worker.js). Hands the worker one picture at a time and paces itself:
// it never queues work, and it backs off when a pass is slow, so a weak device
// spends at most about half its time reading.

const BASE = '/static/models/assist/';

/** A worker script's address, carrying the page's ?threads= setting along. */
export function workerUrl(file) {
  const url = new URL(file, import.meta.url);
  const threads = new URLSearchParams(location.search).get('threads');
  if (threads) url.searchParams.set('threads', threads);
  return url;
}

export class TextReader {
  constructor(worker, threads) {
    this.worker = worker;
    this.threads = threads;
    this.busy = false;
    this.lastMs = 0;                 // duration of the last pass
    this.lastEnd = -1e9;
    this.level = 0;                  // 0 full quality .. 2 lightest (raised when passes are slow)
    this.canvas = document.createElement('canvas');
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
    this.pending = new Map();
    this.nextId = 1;
    this.failures = 0;               // passes in a row that failed or timed out
    this.broken = false;             // set after repeated failures: the app then builds a new reader
    worker.onmessage = (e) => this.onMessage(e.data);
    worker.onerror = (e) => this.failAll(new Error(e.message || 'text reader crashed'));
  }

  static async create() {
    const worker = new Worker(workerUrl('./ocr.worker.js'), { type: 'module' });
    let threads;
    try { threads = await new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('text reader took too long to load')), 90000);
      worker.onmessage = (e) => {
        clearTimeout(timer);
        if (e.data.type === 'ready') resolve(e.data.threads);
        else reject(new Error(e.data.message || 'text reader failed to load'));
      };
      worker.onerror = (e) => { clearTimeout(timer); reject(new Error(e.message || 'text reader failed to load')); };
      worker.postMessage({ type: 'init', base: new URL(BASE, location.href).href });
    }); } catch (err) { worker.terminate(); throw err; }
    return new TextReader(worker, threads);
  }

  onMessage(m) {
    const waiter = this.pending.get(m.id);
    if (!waiter) return;
    this.pending.delete(m.id);
    if (m.type === 'result') waiter.resolve(m); else waiter.reject(new Error(m.message || 'text reader error'));
  }

  failAll(err) {
    for (const w of this.pending.values()) w.reject(err);
    this.pending.clear();
    this.busy = false;
  }

  /** Is it time for another pass? Leaves the CPU idle at least as long as the
   *  last pass took (and never less than `minGap` ms). */
  due(now, minGap = 700) {
    return !this.busy && !this.disposed && now - this.lastEnd >= Math.max(minGap, this.lastMs);
  }

  /**
   * Read the text in a region of a video/image/canvas.
   * @param region {sx, sy, sw, sh} in source pixels
   * @param opts   {workEdge, detEdge, maxLines, minHeight}
   * @returns lines [{text, conf, x, y, w, h (upright bounds), cx, cy, angle, lineHeight}] in source pixels
   */
  async read(source, region, opts = {}) {
    if (this.busy || this.disposed) return null;
    this.busy = true;
    try {
      const lighter = [1, 0.85, 0.7][this.level];
      const workEdge = Math.round((opts.workEdge || 960) * lighter);
      const scale = Math.min(opts.maxUpscale || 1, workEdge / Math.max(region.sw, region.sh));
      const w = Math.max(32, Math.round(region.sw * scale)), h = Math.max(32, Math.round(region.sh * scale));
      if (this.canvas.width !== w || this.canvas.height !== h) { this.canvas.width = w; this.canvas.height = h; }
      this.ctx.imageSmoothingQuality = 'high';
      this.ctx.drawImage(source, region.sx, region.sy, region.sw, region.sh, 0, 0, w, h);
      const data = this.ctx.getImageData(0, 0, w, h).data;
      const id = this.nextId++;
      const started = performance.now();
      const reply = await new Promise((resolve, reject) => {
        const timer = setTimeout(() => { this.pending.delete(id); reject(new Error('text reader timed out')); }, 25000);
        this.pending.set(id, { resolve: (v) => { clearTimeout(timer); resolve(v); },
          reject: (e) => { clearTimeout(timer); reject(e); } });
        this.worker.postMessage({ type: 'run', id, data: data.buffer, width: w, height: h, opts: {
          detEdge: Math.round(((opts.detEdge || 480) * lighter) / 32) * 32,
          maxLines: Math.max(3, Math.round((opts.maxLines || 8) * lighter)),
          minHeight: opts.minHeight || 9,
        } }, [data.buffer]);
      });
      this.failures = 0;
      this.lastMs = performance.now() - started;
      this.lastEnd = performance.now();
      // adapt: step down when a pass takes over 1.6 s, back up when comfortably fast
      if (this.lastMs > 1600 && this.level < 2) this.level++;
      else if (this.lastMs < 450 && this.level > 0) this.level--;
      const lines = reply.lines.map((l) => ({
        text: l.text, conf: l.conf, angle: l.angle,
        x: region.sx + l.x / scale, y: region.sy + l.y / scale, w: l.bw / scale, h: l.bh / scale,
        cx: region.sx + l.cx / scale, cy: region.sy + l.cy / scale, lineHeight: l.h / scale, length: l.w / scale,
      }));
      return { lines, ms: reply.ms, found: reply.found, totalMs: this.lastMs };
    } catch (err) {
      this.lastEnd = performance.now();
      if (++this.failures >= 2) this.broken = true;
      throw err;
    } finally {
      this.busy = false;
    }
  }

  dispose() {
    this.disposed = true;
    this.worker.terminate();
    this.failAll(new Error('text reader closed'));
  }
}

/** Lines in reading order: top to bottom, then left to right within a row. */
export function readingOrder(lines) {
  const sorted = [...lines].sort((a, b) => a.cy - b.cy);
  const rows = [];
  for (const l of sorted) {
    const row = rows.find((r) => Math.abs(r.cy - l.cy) < 0.6 * Math.min(r.h, l.lineHeight));
    if (row) { row.items.push(l); row.cy = (row.cy * (row.items.length - 1) + l.cy) / row.items.length; }
    else rows.push({ cy: l.cy, h: l.lineHeight, items: [l] });
  }
  return rows.flatMap((r) => r.items.sort((a, b) => a.cx - b.cx));
}
