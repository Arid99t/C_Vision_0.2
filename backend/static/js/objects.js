// Room-object spotting on the device.
//
// JavaScript twin of backend/objects/detect.py: same letterboxing, same score
// merging (several word prompts describe one on-screen label; the label takes
// its best prompt), same suppression — so the accuracy measured by
// backend/objects/evaluate.py is the accuracy of what runs here.
//
// The model (static/models/objects.onnx) is an open-vocabulary detector with
// the vocabulary of backend/objects/vocabulary.json baked in. It is told what
// to look for in words, so it can also be wrong in ways a purpose-trained
// model would not be: labels are only shown once an object has been seen
// consistently for several runs, and each label has its own score threshold.

import { createSession, ort } from './ortenv.js';

const PAD = 114 / 255;

function iou(a, b) {
  const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]), y2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return union > 0 ? inter / union : 0;
}

/**
 * Raw model output -> detections, best first.
 * @param data  Float32Array of (4 + P) * N values laid out [channel][candidate]
 * @param scale input pixels per source pixel (to map boxes back)
 * @returns [{label, score, box:[x1,y1,x2,y2] in source pixels, show, hue}]
 */
export function decode(data, P, N, meta, scale, floor = null) {
  const labels = meta.labels, L = labels.length;
  const cand = [];
  for (let i = 0; i < N; i++) {
    let best = -1, bestScore = 0;
    for (let l = 0; l < L; l++) {
      const pr = labels[l].prompts;
      let s = 0;
      for (let k = 0; k < pr.length; k++) {
        const v = data[(4 + pr[k]) * N + i];
        if (v > s) s = v;
      }
      if (s > bestScore) { bestScore = s; best = l; }
    }
    if (best < 0) continue;
    if (bestScore >= (floor === null ? labels[best].threshold : floor)) cand.push([bestScore, best, i]);
  }
  cand.sort((a, b) => b[0] - a[0]);
  if (cand.length > 400) cand.length = 400;
  const same = meta.nms_iou ?? 0.5, other = meta.cross_iou ?? 0.8;
  const out = [];
  for (const [score, l, i] of cand) {                   // greedy suppression, best first
    const cx = data[i], cy = data[N + i], w = data[2 * N + i], h = data[3 * N + i];
    const box = [(cx - w / 2) / scale, (cy - h / 2) / scale, (cx + w / 2) / scale, (cy + h / 2) / scale];
    let ok = true;
    for (const o of out) {
      if (iou(box, o.box) >= (o.l === l ? same : other)) { ok = false; break; }
    }
    if (ok) out.push({ l, label: labels[l].label, score, box, show: labels[l].show !== false, hue: labels[l].hue });
  }
  return out;
}

export class ObjectSpotter {
  constructor(session, meta) {
    this.session = session;
    this.meta = meta;
    this.inputName = session.inputNames[0];
    this.outputName = session.outputNames[0];
    this.canvas = document.createElement('canvas');
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
    this.busy = false;
    this.lastRun = -1e9;
    this.lastMs = 0;
    this.intervalMs = 350;          // pause between runs (adapts to how long a run takes)
    this.longEdge = meta.long_edge || 640;
    this.tracks = [];               // objects seen recently, for stable labels
    this.nextId = 1;
  }

  static async load(meta, base = '/static/models/') {
    return new ObjectSpotter(await createSession(base + (meta.file || 'objects.onnx')), meta);
  }

  /** Letterbox a region of `source` into the model's input tensor. */
  prepare(source, sx, sy, sw, sh) {
    const stride = this.meta.stride || 32;
    const scale = this.longEdge / Math.max(sw, sh);
    const nw = Math.max(1, Math.round(sw * scale)), nh = Math.max(1, Math.round(sh * scale));
    const pw = Math.ceil(nw / stride) * stride, ph = Math.ceil(nh / stride) * stride;
    if (this.canvas.width !== pw || this.canvas.height !== ph) {
      this.canvas.width = pw; this.canvas.height = ph;
    }
    this.ctx.fillStyle = 'rgb(114,114,114)';
    this.ctx.fillRect(0, 0, pw, ph);
    this.ctx.drawImage(source, sx, sy, sw, sh, 0, 0, nw, nh);
    const rgba = this.ctx.getImageData(0, 0, pw, ph).data;
    const n = pw * ph, x = new Float32Array(3 * n);
    for (let i = 0, j = 0; i < n; i++, j += 4) {
      x[i] = rgba[j] / 255; x[n + i] = rgba[j + 1] / 255; x[2 * n + i] = rgba[j + 2] / 255;
    }
    return { tensor: new ort.Tensor('float32', x, [1, 3, ph, pw]), scale };
  }

  /** One detection pass over a region of the source; boxes in source pixels. */
  async detect(source, sx, sy, sw, sh, floor = null) {
    const { tensor, scale } = this.prepare(source, sx, sy, sw, sh);
    const out = (await this.session.run({ [this.inputName]: tensor }))[this.outputName];
    const P = out.dims[1] - 4, N = out.dims[2];
    const dets = decode(out.data, P, N, this.meta, scale, floor);
    for (const d of dets) { d.box[0] += sx; d.box[1] += sy; d.box[2] += sx; d.box[3] += sy; }
    return dets;
  }

  /**
   * Detect, then update the stable object list. Returns what should be drawn:
   * [{x, y, w, h, label, score, hue}] in source pixels.
   */
  async run(source, sx, sy, sw, sh, now) {
    this.busy = true;
    const t0 = performance.now();
    try {
      const dets = await this.detect(source, sx, sy, sw, sh);
      this.lastMs = performance.now() - t0;
      this.lastRun = now;
      // leave the device at least as much time idle as a run took
      this.intervalMs = Math.max(250, Math.min(1500, this.lastMs * 1.2));
      return this.track(dets.filter((d) => d.show));
    } finally {
      this.lastRun = now; // failures also wait before retrying
      this.busy = false;
    }
  }

  /** Keep an object on screen only while it is seen consistently with one label:
   *  shown after 2 hits in a row with the same label, dropped after 2 misses. */
  track(dets) {
    const used = new Set();
    for (const t of this.tracks) {
      let best = -1, bestIou = 0.3;
      dets.forEach((d, i) => {
        if (used.has(i)) return;
        const v = iou(t.box, d.box);
        if (v > bestIou) { bestIou = v; best = i; }
      });
      if (best < 0) { t.misses++; continue; }
      const d = dets[best];
      used.add(best);
      t.misses = 0;
      for (let k = 0; k < 4; k++) t.box[k] += (d.box[k] - t.box[k]) * 0.6;
      if (d.label === t.label) {
        t.hits++;
        t.score += (d.score - t.score) * 0.5;
      } else if (--t.hits <= 0) {            // a different label has to win repeatedly
        t.label = d.label; t.hue = d.hue; t.score = d.score; t.hits = 1;
      }
    }
    dets.forEach((d, i) => {
      if (!used.has(i)) {
        this.tracks.push({ id: this.nextId++, box: d.box.slice(), label: d.label, hue: d.hue,
          score: d.score, hits: 1, misses: 0 });
      }
    });
    this.tracks = this.tracks.filter((t) => t.misses < 2);
    return this.tracks.filter((t) => t.hits >= 2).map((t) => ({
      x: t.box[0], y: t.box[1], w: t.box[2] - t.box[0], h: t.box[3] - t.box[1],
      label: t.label, score: t.score, hue: t.hue, id: t.id }));
  }
}
