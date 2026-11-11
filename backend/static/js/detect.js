// Find handwritten-formula regions in a camera frame, on the device.
//
// JavaScript port of detect_formula_regions() in clients/desktop/ar_app.py —
// same idea, same filters, sized relative to the frame so it works at the
// reduced resolution used here:
//
//   1. black-hat: ink is a dark feature narrower than the kernel on lighter
//      paper, so (closing - image) lights up strokes and ignores the page,
//      its edges and soft shadows;
//   2. Otsu threshold with an absolute floor -> ink mask;
//   3. close the mask horizontally so characters merge into formula blobs;
//   4. keep blobs that look like a line of text: sensible size and shape, not
//      touching the border, moderate ink density, and made of at least
//      `minComponents` character-sized pieces.
//
// Assumes dark ink on a lighter background (same assumption as the recognizer).
// Everything is O(1) work per pixel per pass on preallocated typed arrays, so
// a frame costs a few milliseconds.

// --- grayscale max/min filters (van Herk / Gil-Werman), windows clipped ------
function filterRows(src, dst, w, h, k, isMax, g, hb) {
  const r = k >> 1;
  for (let y = 0; y < h; y++) {
    const o = y * w;
    if (isMax) {
      for (let x = 0; x < w; x++) {
        const v = src[o + x];
        g[x] = x % k === 0 || v > g[x - 1] ? v : g[x - 1];
      }
      for (let x = w - 1; x >= 0; x--) {
        const v = src[o + x];
        hb[x] = x === w - 1 || (x + 1) % k === 0 || v > hb[x + 1] ? v : hb[x + 1];
      }
    } else {
      for (let x = 0; x < w; x++) {
        const v = src[o + x];
        g[x] = x % k === 0 || v < g[x - 1] ? v : g[x - 1];
      }
      for (let x = w - 1; x >= 0; x--) {
        const v = src[o + x];
        hb[x] = x === w - 1 || (x + 1) % k === 0 || v < hb[x + 1] ? v : hb[x + 1];
      }
    }
    const lastBlock = Math.floor((w - 1) / k);
    for (let x = 0; x < w; x++) {
      const a = x - r, b = x + r;
      let v;
      if (a < 0) v = g[b < w ? b : w - 1];                       // inside the first block
      else if (b >= w) {
        v = hb[a];
        if (Math.floor(a / k) !== lastBlock) {
          const t = g[w - 1];
          if (isMax ? t > v : t < v) v = t;
        }
      } else {
        v = hb[a];
        const t = g[b];
        if (isMax ? t > v : t < v) v = t;
      }
      dst[o + x] = v;
    }
  }
}

function filterCols(src, dst, w, h, k, isMax, G, H) {
  const r = k >> 1;
  for (let y = 0; y < h; y++) {                                   // prefix within blocks
    const o = y * w, up = o - w, start = y % k === 0;
    for (let x = 0; x < w; x++) {
      const v = src[o + x];
      if (start) G[o + x] = v;
      else {
        const p = G[up + x];
        G[o + x] = (isMax ? v > p : v < p) ? v : p;
      }
    }
  }
  for (let y = h - 1; y >= 0; y--) {                              // suffix within blocks
    const o = y * w, dn = o + w, start = y === h - 1 || (y + 1) % k === 0;
    for (let x = 0; x < w; x++) {
      const v = src[o + x];
      if (start) H[o + x] = v;
      else {
        const p = H[dn + x];
        H[o + x] = (isMax ? v > p : v < p) ? v : p;
      }
    }
  }
  const lastBlock = Math.floor((h - 1) / k);
  for (let y = 0; y < h; y++) {
    const o = y * w, a = y - r, b = y + r;
    if (a < 0) {
      const gb = (b < h ? b : h - 1) * w;
      for (let x = 0; x < w; x++) dst[o + x] = G[gb + x];
    } else if (b >= h) {
      const ha = a * w, gl = (h - 1) * w, two = Math.floor(a / k) !== lastBlock;
      for (let x = 0; x < w; x++) {
        let v = H[ha + x];
        if (two) {
          const t = G[gl + x];
          if (isMax ? t > v : t < v) v = t;
        }
        dst[o + x] = v;
      }
    } else {
      const ha = a * w, gb = b * w;
      for (let x = 0; x < w; x++) {
        const v = H[ha + x], t = G[gb + x];
        dst[o + x] = (isMax ? t > v : t < v) ? t : v;
      }
    }
  }
}

// --- binary dilation / erosion with a kw x kh rectangle via running counts ---
function binaryRect(src, dst, w, h, kw, kh, dilate, tmp, cnt) {
  const rx = kw >> 1, ry = kh >> 1;
  for (let y = 0; y < h; y++) {                                   // horizontal
    const o = y * w;
    let c = 0;
    for (let x = 0; x <= rx && x < w; x++) c += src[o + x];
    for (let x = 0; x < w; x++) {
      const lo = x - rx, hi = x + rx;
      const len = (hi < w ? hi : w - 1) - (lo > 0 ? lo : 0) + 1;
      tmp[o + x] = dilate ? (c > 0 ? 1 : 0) : (c === len ? 1 : 0);
      if (hi + 1 < w) c += src[o + hi + 1];
      if (lo >= 0) c -= src[o + lo];
    }
  }
  cnt.fill(0, 0, w);                                              // vertical
  for (let y = 0; y <= ry && y < h; y++) {
    const o = y * w;
    for (let x = 0; x < w; x++) cnt[x] += tmp[o + x];
  }
  for (let y = 0; y < h; y++) {
    const o = y * w, lo = y - ry, hi = y + ry;
    const len = (hi < h ? hi : h - 1) - (lo > 0 ? lo : 0) + 1;
    if (dilate) for (let x = 0; x < w; x++) dst[o + x] = cnt[x] > 0 ? 1 : 0;
    else for (let x = 0; x < w; x++) dst[o + x] = cnt[x] === len ? 1 : 0;
    if (hi + 1 < h) {
      const a = (hi + 1) * w;
      for (let x = 0; x < w; x++) cnt[x] += tmp[a + x];
    }
    if (lo >= 0) {
      const a = lo * w;
      for (let x = 0; x < w; x++) cnt[x] -= tmp[a + x];
    }
  }
}

/**
 * 8-connected components of a 0/1 image: bounding boxes and pixel counts.
 * Classic two-pass labelling with union-find, on preallocated typed arrays
 * (`labels`: one int per pixel, `parent`: one per provisional label).
 * `bin` is read with row stride `stride` starting at `base`, so the same
 * routine labels a whole image or just a rectangle of one.
 */
function components(bin, w, h, labels, parent, stride = w, base = 0) {
  if (stride !== w || base !== 0) return componentsRoi(bin, w, h, labels, parent, stride, base);
  let next = 1;
  const find = (a) => {
    while (parent[a] !== a) { parent[a] = parent[parent[a]]; a = parent[a]; }
    return a;
  };
  const union = (a, b) => {
    const ra = find(a), rb = find(b);
    if (ra === rb) return ra;
    if (ra < rb) { parent[rb] = ra; return ra; }
    parent[ra] = rb;
    return rb;
  };
  for (let y = 0; y < h; y++) {
    const o = y * w;
    for (let x = 0; x < w; x++) {
      const i = o + x;
      if (!bin[i]) { labels[i] = 0; continue; }
      let lab = x > 0 ? labels[i - 1] : 0, l;
      if (y > 0) {
        const up = i - w;
        if (x > 0 && (l = labels[up - 1])) lab = lab ? union(lab, l) : l;
        if ((l = labels[up])) lab = lab ? union(lab, l) : l;
        if (x < w - 1 && (l = labels[up + 1])) lab = lab ? union(lab, l) : l;
      }
      if (!lab) { lab = next; parent[next] = next; next++; }
      labels[i] = lab;
    }
  }
  // second pass: gather statistics per root label
  const slot = new Int32Array(next).fill(-1);
  const out = [];
  for (let y = 0; y < h; y++) {
    const o = y * w;
    for (let x = 0; x < w; x++) {
      const l = labels[o + x];
      if (!l) continue;
      const root = find(l);
      let k = slot[root];
      if (k < 0) { k = slot[root] = out.length; out.push({ x0: x, y0: y, x1: x, y1: y, area: 0 }); }
      const c = out[k];
      if (x < c.x0) c.x0 = x;
      else if (x > c.x1) c.x1 = x;
      c.y1 = y;
      c.area++;
    }
  }
  return out;
}

/** components() for a w x h window of a larger image (row stride, start offset). */
function componentsRoi(bin, w, h, labels, parent, stride, base) {
  let next = 1;
  const find = (a) => {
    while (parent[a] !== a) { parent[a] = parent[parent[a]]; a = parent[a]; }
    return a;
  };
  const union = (a, b) => {
    const ra = find(a), rb = find(b);
    if (ra === rb) return ra;
    if (ra < rb) { parent[rb] = ra; return ra; }
    parent[ra] = rb;
    return rb;
  };
  for (let y = 0; y < h; y++) {
    const o = y * w, src = base + y * stride;
    for (let x = 0; x < w; x++) {
      const i = o + x;
      if (!bin[src + x]) { labels[i] = 0; continue; }
      let lab = x > 0 ? labels[i - 1] : 0, l;
      if (y > 0) {
        const up = i - w;
        if (x > 0 && (l = labels[up - 1])) lab = lab ? union(lab, l) : l;
        if ((l = labels[up])) lab = lab ? union(lab, l) : l;
        if (x < w - 1 && (l = labels[up + 1])) lab = lab ? union(lab, l) : l;
      }
      if (!lab) { lab = next; parent[next] = next; next++; }
      labels[i] = lab;
    }
  }
  const slot = new Int32Array(next).fill(-1);
  const out = [];
  for (let y = 0; y < h; y++) {
    const o = y * w;
    for (let x = 0; x < w; x++) {
      const l = labels[o + x];
      if (!l) continue;
      const root = find(l);
      let k = slot[root];
      if (k < 0) { k = slot[root] = out.length; out.push({ x0: x, y0: y, x1: x, y1: y, area: 0 }); }
      const c = out[k];
      if (x < c.x0) c.x0 = x;
      else if (x > c.x1) c.x1 = x;
      c.y1 = y;
      c.area++;
    }
  }
  return out;
}

export function iou(a, b) {
  const x1 = Math.max(a.x, b.x), y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w), y2 = Math.min(a.y + a.h, b.y + b.h);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = a.w * a.h + b.w * b.h - inter;
  return union ? inter / union : 0;
}

export class FormulaDetector {
  constructor() {
    this.size = 0;
  }

  _alloc(w, h) {
    if (this.w === w && this.h === h) return;
    this.w = w;
    this.h = h;
    const n = w * h, w2 = w >> 1, h2 = h >> 1, n2 = w2 * h2;
    this.gray = new Uint8Array(n);
    this.blur = new Uint8Array(n);
    this.tmp = new Uint8Array(n);
    this.mask = new Uint8Array(n);                 // full-resolution ink mask
    for (const name of ['small', 'a', 'b', 'G', 'H', 'close', 'maskS', 'merged']) {
      this[name] = new Uint8Array(n2);             // half-resolution work images
    }
    this.rowG = new Uint8Array(w2);
    this.rowH = new Uint8Array(w2);
    this.cnt = new Int32Array(w2);
    this.labels = new Int32Array(n);               // big enough for a full-res window
    this.parent = new Int32Array((n >> 1) + 16);   // more than the most labels possible
  }

  /**
   * @param rgba   RGBA pixels of the (already downscaled) frame
   * @returns [{x, y, w, h}] in the coordinates of that frame, largest first
   */
  detect(rgba, w, h, { maxRegions = 3, minComponents = 2 } = {}) {
    this._alloc(w, h);
    const n = w * h, w2 = w >> 1, h2 = h >> 1, n2 = w2 * h2;
    const { gray, blur, tmp, mask, small, a, b, G, H, close, maskS, merged, rowG, rowH, cnt } = this;
    const s = Math.max(w, h) / 640;                 // sizes below were tuned at 640 px

    for (let i = 0, j = 0; i < n; i++, j += 4) {
      gray[i] = (rgba[j] * 77 + rgba[j + 1] * 150 + rgba[j + 2] * 29) >> 8;
    }
    // light smoothing: separable [1 2 1] / 4
    for (let y = 0; y < h; y++) {
      const o = y * w;
      tmp[o] = (3 * gray[o] + gray[o + 1] + 2) >> 2;
      for (let x = 1; x < w - 1; x++) tmp[o + x] = (gray[o + x - 1] + 2 * gray[o + x] + gray[o + x + 1] + 2) >> 2;
      tmp[o + w - 1] = (gray[o + w - 2] + 3 * gray[o + w - 1] + 2) >> 2;
    }
    for (let y = 0; y < h; y++) {
      const up = (y > 0 ? y - 1 : 0) * w, dn = (y < h - 1 ? y + 1 : y) * w, o = y * w;
      for (let x = 0; x < w; x++) blur[o + x] = (tmp[up + x] + 2 * tmp[o + x] + tmp[dn + x] + 2) >> 2;
    }

    // The paper brightness (a grayscale closing with a kernel ~ character
    // height) varies slowly, so it is estimated at half resolution...
    for (let y = 0; y < h2; y++) {
      const o = 2 * y * w, d = y * w2;
      for (let x = 0; x < w2; x++) {
        const i = o + 2 * x;
        small[d + x] = (blur[i] + blur[i + 1] + blur[i + w] + blur[i + w + 1] + 2) >> 2;
      }
    }
    const k = Math.max(5, Math.floor(h / 56)) | 1;
    filterRows(small, a, w2, h2, k, true, rowG, rowH);
    filterCols(a, b, w2, h2, k, true, G, H);            // b = dilation
    filterRows(b, a, w2, h2, k, false, rowG, rowH);
    filterCols(a, close, w2, h2, k, false, G, H);       // close = closing
    // ...while the black-hat response (closing - image) and the ink mask keep
    // the full resolution, so thin strokes survive.
    const hist = new Uint32Array(256);
    for (let y = 0; y < h; y++) {
      const o = y * w, c = (y >> 1 < h2 ? y >> 1 : h2 - 1) * w2;
      for (let x = 0; x < w; x++) {
        const xs = x >> 1;
        const v = close[c + (xs < w2 ? xs : w2 - 1)] - blur[o + x];
        const r = v > 0 ? v : 0;
        tmp[o + x] = r;
        hist[r]++;
      }
    }
    // Otsu threshold with an absolute floor (kills faint shadow/noise response)
    let sum = 0;
    for (let v = 0; v < 256; v++) sum += v * hist[v];
    let wB = 0, sumB = 0, best = -1, thr = 0;
    for (let v = 0; v < 256; v++) {
      wB += hist[v];
      if (!wB) continue;
      const wF = n - wB;
      if (!wF) break;
      sumB += v * hist[v];
      const mB = sumB / wB, mF = (sum - sumB) / wF;
      const between = wB * wF * (mB - mF) * (mB - mF);
      if (between > best) { best = between; thr = v; }
    }
    const cut = Math.max(thr + 1, 18);
    for (let i = 0; i < n; i++) mask[i] = tmp[i] >= cut ? 1 : 0;

    // Blob analysis at half resolution: a pixel is ink if any of its 2x2 is.
    for (let y = 0; y < h2; y++) {
      const o = 2 * y * w, d = y * w2;
      for (let x = 0; x < w2; x++) {
        const i = o + 2 * x;
        maskS[d + x] = mask[i] | mask[i + 1] | mask[i + w] | mask[i + w + 1];
      }
    }
    // merge characters into formula-sized blobs, then drop specks
    const mw = (2 * Math.max(2, Math.round(7.5 * s)) - 1) | 1;
    const mh = (2 * Math.max(2, Math.round(4.5 * s)) - 1) | 1;
    binaryRect(maskS, b, w2, h2, mw, mh, true, a, cnt);
    binaryRect(b, merged, w2, h2, mw, mh, false, a, cnt);
    binaryRect(merged, b, w2, h2, 3, 3, false, a, cnt);
    binaryRect(b, merged, w2, h2, 3, 3, true, a, cnt);

    const blobs = components(merged, w2, h2, this.labels, this.parent);
    const frameArea = n, margin = Math.max(2, Math.round(5 * s));
    const candidates = [];
    for (const c of blobs) {                            // back to full-resolution pixels
      const x0 = 2 * c.x0, y0 = 2 * c.y0, x1 = 2 * c.x1 + 1, y1 = 2 * c.y1 + 1;
      const bw = x1 - x0 + 1, bh = y1 - y0 + 1, area = bw * bh;
      if (area < 0.0008 * frameArea || area > 0.5 * frameArea) continue;
      if (bw < 24 * s || bh < 14 * s) continue;
      const aspect = bw / bh;
      if (aspect < 0.7 || aspect > 18) continue;        // formulas are roughly horizontal
      if (x0 <= margin || y0 <= margin || x1 >= w - 1 - margin || y1 >= h - 1 - margin) {
        continue;                                       // hugging the border -> paper edge / hand
      }
      candidates.push({ x0, y0, x1, y1, bw, bh, area });
    }
    const boxes = [];
    for (const { x0, y0, x1, y1, bw, bh, area } of candidates) {
      let ink = 0;
      for (let y = y0; y <= y1; y++) {
        const o = y * w;
        for (let x = x0; x <= x1; x++) ink += mask[o + x];
      }
      const density = ink / area;
      if (density < 0.04 || density > 0.6) continue;    // text has moderate ink density
      // "Is it actually text?": count character-sized pieces of ink inside the
      // box, on the full-resolution mask (only this small window is labelled).
      let pieces = 0;
      for (const ch of components(mask, bw, bh, this.labels, this.parent, w, y0 * w + x0)) {
        const chH = ch.y1 - ch.y0 + 1;
        if (chH >= 0.33 * bh && ch.area >= 0.008 * area && ch.area <= 0.85 * area) pieces++;
      }
      if (pieces < minComponents) continue;             // not text-like
      boxes.push({ x: x0, y: y0, w: bw, h: bh, pieces });
    }
    boxes.sort((p, q) => q.w * q.h - p.w * p.h);
    const keep = [];
    for (const bx of boxes) {
      if (keep.every((kb) => iou(bx, kb) < 0.3)) keep.push(bx);
      if (keep.length >= maxRegions) break;
    }
    return keep;
  }
}
