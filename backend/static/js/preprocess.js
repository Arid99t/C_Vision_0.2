// JavaScript twin of backend/recognition/preprocess.py (the v2 "prepare_crop"
// pipeline). The recognizer is trained on images produced by the Python
// version, so this file must do the SAME arithmetic, step for step:
//
//   bound the size -> flat-field -> contrast stretch -> level a tilted line
//   -> crop tight to the ink with a fixed margin -> height 32
//
// If you change one side, change the other, regenerate the training data and
// retrain. tests/test_web_parity.py runs both on the same crops and compares.

export const PREP = {
  imgHeight: 32, minWidth: 16, maxWidth: 384,
  workHeight: 64, inkThreshold: 200, marginFrac: 0.12,
  rotMinElong: 6, rotMinDeg: 5, rotMaxDeg: 40,
};

/** Adopt the constants exported with the model (models/formula_rec.json). */
export function configure(meta) {
  if (!meta) return;
  PREP.imgHeight = meta.img_height ?? PREP.imgHeight;
  PREP.minWidth = meta.min_width ?? PREP.minWidth;
  PREP.maxWidth = meta.max_width ?? PREP.maxWidth;
  const p = meta.preprocess || {};
  PREP.workHeight = p.work_height ?? PREP.workHeight;
  PREP.inkThreshold = p.ink_threshold ?? PREP.inkThreshold;
  PREP.marginFrac = p.margin_frac ?? PREP.marginFrac;
  PREP.rotMinElong = p.rot_min_elong ?? PREP.rotMinElong;
  PREP.rotMinDeg = p.rot_min_deg ?? PREP.rotMinDeg;
  PREP.rotMaxDeg = p.rot_max_deg ?? PREP.rotMaxDeg;
}

/** RGBA pixels -> grayscale, same integer formula as Pillow's convert("L"). */
export function rgbaToGray(rgba, n) {
  const g = new Float64Array(n);
  for (let i = 0, j = 0; i < n; i++, j += 4) {
    g[i] = (rgba[j] * 19595 + rgba[j + 1] * 38470 + rgba[j + 2] * 7471 + 32768) >> 16;
  }
  return g;
}

function roundLevel(v) {
  const r = Math.floor(v + 0.5);
  return r < 0 ? 0 : (r > 255 ? 255 : r);
}

/** Smallest level whose cumulative count reaches ceil(p * n). */
function histPercentiles(g, ps) {
  const hist = new Uint32Array(256);
  for (let i = 0; i < g.length; i++) hist[roundLevel(g[i])]++;
  return ps.map((p) => {
    const target = Math.max(1, Math.ceil(p * g.length));
    let cum = 0;
    for (let v = 0; v < 256; v++) {
      cum += hist[v];
      if (cum >= target) return v;
    }
    return 255;
  });
}

/** Sparse rows of the (nOut x nIn) resampling matrix: box filter to shrink,
 *  bilinear to grow. Returns [{start, weights}] per output index. */
function resizeRows(nIn, nOut) {
  const rows = new Array(nOut);
  if (nOut === nIn) {
    for (let o = 0; o < nOut; o++) rows[o] = { start: o, weights: [1.0] };
    return rows;
  }
  const scale = nIn / nOut;
  if (nOut < nIn) {
    for (let o = 0; o < nOut; o++) {
      const start = o * scale, end = (o + 1.0) * scale;
      const i0 = Math.floor(start), i1 = Math.min(nIn, Math.ceil(end));
      const weights = [];
      for (let i = i0; i < i1; i++) {
        const ov = Math.min(end, i + 1.0) - Math.max(start, i);
        weights.push((ov > 0 ? ov : 0) / scale);
      }
      rows[o] = { start: i0, weights };
    }
  } else {
    for (let o = 0; o < nOut; o++) {
      const src = (o + 0.5) * scale - 0.5;
      const i0 = Math.floor(src), f = src - i0;
      const a = Math.min(Math.max(i0, 0), nIn - 1);
      const b = Math.min(Math.max(i0 + 1, 0), nIn - 1);
      if (a === b) rows[o] = { start: a, weights: [(1.0 - f) + f] };
      else rows[o] = { start: a, weights: [1.0 - f, f] };
    }
  }
  return rows;
}

/** Separable resize of a float image: (Wy @ g) @ Wx^T. */
export function resizeGray(g, h, w, outH, outW) {
  const ry = resizeRows(h, outH), rx = resizeRows(w, outW);
  const tmp = new Float64Array(outH * w);
  for (let o = 0; o < outH; o++) {
    const { start, weights } = ry[o];
    const base = o * w;
    for (let k = 0; k < weights.length; k++) {
      const wk = weights[k], src = (start + k) * w;
      for (let x = 0; x < w; x++) tmp[base + x] += wk * g[src + x];
    }
  }
  const out = new Float64Array(outH * outW);
  for (let y = 0; y < outH; y++) {
    const base = y * w, ob = y * outW;
    for (let o = 0; o < outW; o++) {
      const { start, weights } = rx[o];
      let acc = 0;
      for (let k = 0; k < weights.length; k++) acc += weights[k] * tmp[base + start + k];
      out[ob + o] = acc;
    }
  }
  return out;
}

/** 3 x 3 mean with replicated edges. */
function box3(g, h, w) {
  const out = new Float64Array(h * w);
  for (let dy = 0; dy < 3; dy++) {
    for (let dx = 0; dx < 3; dx++) {
      for (let y = 0; y < h; y++) {
        const sy = Math.min(h - 1, Math.max(0, y + dy - 1)) * w;
        const oy = y * w;
        for (let x = 0; x < w; x++) {
          out[oy + x] += g[sy + Math.min(w - 1, Math.max(0, x + dx - 1))];
        }
      }
    }
  }
  for (let i = 0; i < out.length; i++) out[i] /= 9.0;
  return out;
}

/** Separable k x k sliding max (or min); windows are clipped at the border. */
function windowFilter(g, h, w, k, useMax) {
  const r = k >> 1;
  const tmp = new Float64Array(h * w), out = new Float64Array(h * w);
  for (let y = 0; y < h; y++) {
    const base = y * w;
    for (let x = 0; x < w; x++) {
      const x0 = Math.max(0, x - r), x1 = Math.min(w - 1, x + r);
      let v = g[base + x0];
      for (let i = x0 + 1; i <= x1; i++) {
        const t = g[base + i];
        if (useMax ? t > v : t < v) v = t;
      }
      tmp[base + x] = v;
    }
  }
  for (let y = 0; y < h; y++) {
    const y0 = Math.max(0, y - r), y1 = Math.min(h - 1, y + r);
    for (let x = 0; x < w; x++) {
      let v = tmp[y0 * w + x];
      for (let i = y0 + 1; i <= y1; i++) {
        const t = tmp[i * w + x];
        if (useMax ? t > v : t < v) v = t;
      }
      out[y * w + x] = v;
    }
  }
  return out;
}

/** Divide out the paper brightness (paper estimated by a grayscale closing). */
function flatField(g, h, w) {
  const k = Math.max(9, Math.floor(0.28 * h + 0.5)) | 1;
  const paper = windowFilter(windowFilter(box3(g, h, w), h, w, k, true), h, w, k, false);
  const out = new Float64Array(h * w);
  for (let i = 0; i < out.length; i++) {
    const v = (g[i] / Math.max(paper[i], 1.0)) * 255.0;
    out[i] = v < 0 ? 0 : (v > 255 ? 255 : v);
  }
  return out;
}

/** Map the ink level to 0 and the paper level to 255. */
function stretchContrast(g) {
  const [lo, hi] = histPercentiles(g, [0.02, 0.90]);
  if (hi - lo < 20) return g;
  const f = 255.0 / (hi - lo);
  const out = new Float64Array(g.length);
  for (let i = 0; i < g.length; i++) {
    const v = (g[i] - lo) * f;
    out[i] = v < 0 ? 0 : (v > 255 ? 255 : v);
  }
  return out;
}

/** {theta, elong} of the ink's principal axis, or null. mask: Uint8Array. */
export function estimateTilt(mask, h, w) {
  let n = 0, sx = 0, sy = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (mask[y * w + x]) { n++; sx += x; sy += y; }
    }
  }
  if (n < 30) return null;
  const xbar = sx / n, ybar = sy / n;
  let m20 = 0, m02 = 0, m11 = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (mask[y * w + x]) {
        const dx = x - xbar, dy = y - ybar;
        m20 += dx * dx; m02 += dy * dy; m11 += dx * dy;
      }
    }
  }
  m20 /= n; m02 /= n; m11 /= n;
  const common = Math.sqrt(4.0 * m11 * m11 + (m20 - m02) * (m20 - m02));
  const l1 = (m20 + m02 + common) / 2.0, l2 = (m20 + m02 - common) / 2.0;
  if (l2 <= 1e-6) return null;
  return { theta: 0.5 * Math.atan2(2.0 * m11, m20 - m02), elong: l1 / l2 };
}

/** Rotate so an ink axis tilted by theta becomes horizontal (white fill). */
function level(g, h, w, theta) {
  const c = Math.cos(theta), s = Math.sin(theta);
  const w2 = Math.ceil(w * Math.abs(c) + h * Math.abs(s));
  const h2 = Math.ceil(w * Math.abs(s) + h * Math.abs(c));
  const cx = (w - 1) / 2.0, cy = (h - 1) / 2.0, cx2 = (w2 - 1) / 2.0, cy2 = (h2 - 1) / 2.0;
  const out = new Float64Array(h2 * w2);
  const at = (yy, xx) => (xx >= 0 && xx < w && yy >= 0 && yy < h ? g[yy * w + xx] : 255.0);
  for (let i = 0; i < h2; i++) {
    const Y = i - cy2;
    for (let j = 0; j < w2; j++) {
      const X = j - cx2;
      const xs = c * X - s * Y + cx, ys = s * X + c * Y + cy;
      const x0 = Math.floor(xs), y0 = Math.floor(ys);
      const fx = xs - x0, fy = ys - y0;
      out[i * w2 + j] = at(y0, x0) * (1 - fx) * (1 - fy) + at(y0, x0 + 1) * fx * (1 - fy)
        + at(y0 + 1, x0) * (1 - fx) * fy + at(y0 + 1, x0 + 1) * fx * fy;
    }
  }
  return { g: out, h: h2, w: w2 };
}

function inkMask(g, h, w) {
  const sm = box3(g, h, w);
  const mask = new Uint8Array(h * w);
  for (let i = 0; i < mask.length; i++) mask[i] = sm[i] < PREP.inkThreshold ? 1 : 0;
  return mask;
}

/** [x0, y0, x1, y1] (inclusive) of the writing, or null. Rows/columns that are
 *  dark almost end to end (ruled lines) or hold a lone dark pixel don't count. */
function inkBbox(mask, h, w) {
  const rows = new Uint32Array(h), cols = new Uint32Array(w);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (mask[y * w + x]) { rows[y]++; cols[x]++; }
    }
  }
  let y0 = -1, y1 = -1, x0 = -1, x1 = -1;
  for (let y = 0; y < h; y++) {
    if (rows[y] >= 2 && rows[y] <= 0.85 * w) { if (y0 < 0) y0 = y; y1 = y; }
  }
  for (let x = 0; x < w; x++) {
    if (cols[x] >= 2 && cols[x] <= 0.85 * h) { if (x0 < 0) x0 = x; x1 = x; }
  }
  return y0 < 0 || x0 < 0 ? null : [x0, y0, x1, y1];
}

/**
 * Any (h x w) grayscale crop of ONE formula -> the (32 x W) image the model reads.
 * @param gray Float64Array | Uint8Array, row-major, 0..255, dark ink on lighter paper
 * @returns {{data: Uint8Array, width: number, height: number}}
 */
export function prepareCrop(gray, h, w) {
  let g = gray instanceof Float64Array ? gray : Float64Array.from(gray);
  if (h > PREP.workHeight) {
    const nw = Math.max(1, Math.floor((w * PREP.workHeight) / h + 0.5));
    g = resizeGray(g, h, w, PREP.workHeight, nw);
    h = PREP.workHeight; w = nw;
  }
  if (h >= 8 && w >= 8) {
    g = stretchContrast(flatField(g, h, w));
    const tilt = estimateTilt(inkMask(g, h, w), h, w);
    if (tilt) {
      const deg = Math.abs((tilt.theta * 180) / Math.PI);
      if (tilt.elong >= PREP.rotMinElong && deg >= PREP.rotMinDeg && deg <= PREP.rotMaxDeg) {
        const r = level(g, h, w, tilt.theta);
        g = r.g; h = r.h; w = r.w;
      }
    }
    const box = inkBbox(inkMask(g, h, w), h, w);
    if (box) {
      const [x0, y0, x1, y1] = box;
      const m = Math.max(2, Math.floor(PREP.marginFrac * (y1 - y0 + 1) + 0.5));
      const ch = (y1 - y0 + 1) + 2 * m, cw = (x1 - x0 + 1) + 2 * m;
      const canvas = new Float64Array(ch * cw).fill(255.0);
      const sy0 = Math.max(0, y0 - m), sy1 = Math.min(h, y1 + m + 1);
      const sx0 = Math.max(0, x0 - m), sx1 = Math.min(w, x1 + m + 1);
      const dy = sy0 - (y0 - m), dx = sx0 - (x0 - m);
      for (let y = sy0; y < sy1; y++) {
        const src = y * w, dst = (dy + y - sy0) * cw + dx;
        for (let x = sx0; x < sx1; x++) canvas[dst + x - sx0] = g[src + x];
      }
      g = canvas; h = ch; w = cw;
    }
  }
  const outH = PREP.imgHeight;
  const outW = Math.min(PREP.maxWidth, Math.max(PREP.minWidth,
    Math.floor((w * outH) / h + 0.5)));
  const res = resizeGray(g, h, w, outH, outW);
  const data = new Uint8Array(outH * outW);
  for (let i = 0; i < data.length; i++) data[i] = roundLevel(res[i]);
  return { data, width: outW, height: outH };
}

/** (32 x W) uint8 -> Float32Array in [-1, 1], the model's input normalization. */
export function toModelInput(data) {
  const x = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) x[i] = (data[i] / 255.0 - 0.5) / 0.5;
  return x;
}
