// Printed-text reader, off the main thread.
//
// Two small pretrained models (PP-OCRv3, see tools/fetch_assist_models.py):
//   detector    picture -> "how likely is this pixel part of a line of text"
//   recognizer  one cut-out line, 48 px high -> characters (CTC)
// plus the glue between them, written here: threshold the detector's map,
// label the connected blobs, fit a rotated box to each, cut the line out of the
// full-resolution picture, read it.
//
// Runs in a Web Worker so a slow device stutters in the worker, not in the
// camera view. The page sends one RGBA picture per request and gets back the
// lines of text with their boxes.
//
//   -> {type:'init', base}                      <- {type:'ready', threads}
//   -> {type:'run', id, data, width, height, opts}
//                                               <- {type:'result', id, lines, ms}
//   opts: detEdge   long edge of the detector input (multiple of 32)
//         maxLines  read at most this many lines (the biggest text first)
//         minHeight skip lines shorter than this many pixels

import { createSession, ort, threads } from '../ortenv.js';

const CHARS = '0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~!"#$%&\'()*+,-./  ';
const DET_MEAN = [0.485, 0.456, 0.406], DET_STD = [0.229, 0.224, 0.225];   // applied to B, G, R as trained
const DET_THRESH = 0.3, BOX_THRESH = 0.55, UNCLIP = 1.6;
const REC_H = 48, REC_MAX_W = 480;

let det = null, rec = null;

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'init') {
      [det, rec] = await Promise.all([createSession(m.base + 'ocr_det.onnx'), createSession(m.base + 'ocr_rec.onnx')]);
      self.postMessage({ type: 'ready', threads });
    } else if (m.type === 'run') {
      const out = await run(new Uint8ClampedArray(m.data), m.width, m.height, m.opts || {});
      self.postMessage({ type: 'result', id: m.id, ...out });
    } else if (m.type === 'release') {
      await det?.release(); await rec?.release();
      det = rec = null;
      self.postMessage({ type: 'released' });
    }
  } catch (err) {
    self.postMessage({ type: 'error', id: m.id, message: String(err?.message || err) });
  }
};

async function run(rgba, w, h, opts) {
  const t0 = performance.now();
  const boxes = await detect(rgba, w, h, opts.detEdge || 480);
  const t1 = performance.now();
  const minH = opts.minHeight || 9;
  const picked = boxes.filter((b) => b.h >= minH)
    .sort((a, b) => b.h * Math.sqrt(b.w) - a.h * Math.sqrt(a.w))
    .slice(0, opts.maxLines || 8);
  const lines = [];
  for (const b of picked) {
    const r = await readLine(rgba, w, h, b);
    if (r.text) lines.push({ ...b, text: r.text, conf: r.conf });
  }
  const t2 = performance.now();
  return { lines, found: boxes.length, ms: { det: t1 - t0, rec: t2 - t1, total: t2 - t0 } };
}

// ---------------------------------------------------------------- detect ---
async function detect(rgba, w, h, edge) {
  const scale = Math.min(2, edge / Math.max(w, h));
  const dw = Math.max(32, Math.round((w * scale) / 32) * 32), dh = Math.max(32, Math.round((h * scale) / 32) * 32);
  const input = new Float32Array(3 * dw * dh);
  const fx = w / dw, fy = h / dh, plane = dw * dh;
  for (let y = 0; y < dh; y++) {
    // 2x2 box average: enough anti-aliasing for the up-to-2x reduction used here
    const sy0 = Math.min(h - 1, Math.floor((y + 0.25) * fy)), sy1 = Math.min(h - 1, Math.floor((y + 0.75) * fy));
    for (let x = 0; x < dw; x++) {
      const sx0 = Math.min(w - 1, Math.floor((x + 0.25) * fx)), sx1 = Math.min(w - 1, Math.floor((x + 0.75) * fx));
      const a = (sy0 * w + sx0) * 4, b = (sy0 * w + sx1) * 4, c = (sy1 * w + sx0) * 4, d = (sy1 * w + sx1) * 4;
      const r = (rgba[a] + rgba[b] + rgba[c] + rgba[d]) / 1020;
      const g = (rgba[a + 1] + rgba[b + 1] + rgba[c + 1] + rgba[d + 1]) / 1020;
      const bl = (rgba[a + 2] + rgba[b + 2] + rgba[c + 2] + rgba[d + 2]) / 1020;
      const i = y * dw + x;
      input[i] = (bl - DET_MEAN[0]) / DET_STD[0];
      input[plane + i] = (g - DET_MEAN[1]) / DET_STD[1];
      input[2 * plane + i] = (r - DET_MEAN[2]) / DET_STD[2];
    }
  }
  const out = await det.run({ [det.inputNames[0]]: new ort.Tensor('float32', input, [1, 3, dh, dw]) });
  const prob = out[det.outputNames[0]].data;
  return boxesFromMap(prob, dw, dh, fx, fy);
}

/** Probability map -> rotated text boxes, in source-picture pixels. */
export function boxesFromMap(prob, dw, dh, fx, fy) {
  const n = dw * dh;
  const seen = new Uint8Array(n);
  const stack = new Int32Array(n);
  const boxes = [];
  for (let start = 0; start < n; start++) {
    if (seen[start] || prob[start] < DET_THRESH) continue;
    // flood fill one blob (8-connected), keeping its pixels
    let top = 0, count = 0, sum = 0, sx = 0, sy = 0;
    const pixels = [];
    stack[top++] = start; seen[start] = 1;
    while (top) {
      const p = stack[--top];
      const x = p % dw, y = (p - x) / dw;
      pixels.push(p); count++; sum += prob[p]; sx += x; sy += y;
      for (let dy = -1; dy <= 1; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= dh) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= dw) continue;
          const q = yy * dw + xx;
          if (!seen[q] && prob[q] >= DET_THRESH) { seen[q] = 1; stack[top++] = q; }
        }
      }
    }
    if (count < 8 || sum / count < BOX_THRESH) continue;
    // principal axis of the blob = direction of the line of text
    const mx = sx / count, my = sy / count;
    let cxx = 0, cxy = 0, cyy = 0;
    for (const p of pixels) {
      const x = (p % dw) - mx, y = Math.floor(p / dw) - my;
      cxx += x * x; cxy += x * y; cyy += y * y;
    }
    const spread = Math.sqrt((cxx - cyy) * (cxx - cyy) + 4 * cxy * cxy);
    const major = (cxx + cyy + spread) / 2, minor = Math.max(1e-6, (cxx + cyy - spread) / 2);
    let theta = major / minor > 3 ? 0.5 * Math.atan2(2 * cxy, cxx - cyy) : 0;   // short blobs: keep upright
    let ux = Math.cos(theta), uy = Math.sin(theta);
    // extent along and across the axis
    let u0 = Infinity, u1 = -Infinity, v0 = Infinity, v1 = -Infinity;
    for (const p of pixels) {
      const x = (p % dw) - mx, y = Math.floor(p / dw) - my;
      const u = x * ux + y * uy, v = -x * uy + y * ux;
      if (u < u0) u0 = u; if (u > u1) u1 = u; if (v < v0) v0 = v; if (v > v1) v1 = v;
    }
    let bw = u1 - u0 + 1, bh = v1 - v0 + 1;
    const grow = (bw * bh * UNCLIP) / (2 * (bw + bh));           // the detector shrinks text; grow it back
    const cu = (u0 + u1) / 2, cv = (v0 + v1) / 2;
    const cx = mx + cu * ux - cv * uy, cy = my + cu * uy + cv * ux;
    bw += 2 * grow; bh += 2 * grow;
    if (Math.min(bw, bh) < 5) continue;
    // text that runs up the page: read it bottom-to-top, like turning the pack
    if (Math.abs(theta) > 1.2 && uy > 0) { ux = -ux; uy = -uy; theta += theta > 0 ? -Math.PI : Math.PI; }
    // to source pixels (the two axes may be scaled slightly differently)
    const k = (fx + fy) / 2;
    const b = { cx: (cx + 0.5) * fx, cy: (cy + 0.5) * fy, w: bw * k, h: bh * k, angle: theta, score: sum / count };
    const hx = (b.w / 2) * Math.abs(ux) + (b.h / 2) * Math.abs(uy), hy = (b.w / 2) * Math.abs(uy) + (b.h / 2) * Math.abs(ux);
    b.x = b.cx - hx; b.y = b.cy - hy; b.bw = 2 * hx; b.bh = 2 * hy;   // axis-aligned bounds
    boxes.push(b);
  }
  return boxes;
}

// ------------------------------------------------------------- recognise ---
async function readLine(rgba, w, h, b) {
  const tw = Math.max(16, Math.min(REC_MAX_W, Math.round((REC_H * b.w) / b.h)));
  const input = new Float32Array(3 * REC_H * tw);
  const ux = Math.cos(b.angle), uy = Math.sin(b.angle);
  const stepU = b.w / tw, stepV = b.h / REC_H, plane = REC_H * tw;
  for (let j = 0; j < REC_H; j++) {
    const v = (j + 0.5) * stepV - b.h / 2;
    for (let i = 0; i < tw; i++) {
      const u = (i + 0.5) * stepU - b.w / 2;
      let x = b.cx + u * ux - v * uy - 0.5, y = b.cy + u * uy + v * ux - 0.5;
      x = x < 0 ? 0 : x > w - 1 ? w - 1 : x; y = y < 0 ? 0 : y > h - 1 ? h - 1 : y;
      const x0 = x | 0, y0 = y | 0, x1 = x0 + 1 < w ? x0 + 1 : x0, y1 = y0 + 1 < h ? y0 + 1 : y0;
      const ax = x - x0, ay = y - y0;
      const p00 = (y0 * w + x0) * 4, p01 = (y0 * w + x1) * 4, p10 = (y1 * w + x0) * 4, p11 = (y1 * w + x1) * 4;
      const w00 = (1 - ax) * (1 - ay), w01 = ax * (1 - ay), w10 = (1 - ax) * ay, w11 = ax * ay;
      const o = j * tw + i;
      for (let c = 0; c < 3; c++) {                         // model wants B, G, R
        const s = 2 - c;
        const val = rgba[p00 + s] * w00 + rgba[p01 + s] * w01 + rgba[p10 + s] * w10 + rgba[p11 + s] * w11;
        input[c * plane + o] = val / 127.5 - 1;
      }
    }
  }
  const out = await rec.run({ [rec.inputNames[0]]: new ort.Tensor('float32', input, [1, 3, REC_H, tw]) });
  const t = out[rec.outputNames[0]];
  return ctcText(t.data, t.dims[1], t.dims[2]);
}

/** Best-path CTC decode of (T x C) probabilities. */
export function ctcText(p, T, C) {
  let text = '', prev = 0, sum = 0, kept = 0;
  for (let t = 0; t < T; t++) {
    let best = 0, bv = -1;
    const o = t * C;
    for (let c = 0; c < C; c++) if (p[o + c] > bv) { bv = p[o + c]; best = c; }
    if (best !== 0 && best !== prev) { text += CHARS[best - 1] ?? ''; sum += bv; kept++; }
    prev = best;
  }
  return { text: text.trim(), conf: kept ? sum / kept : 0 };
}
