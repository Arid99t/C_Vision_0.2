// Resistor colour-band reader, on the device.
//
// The model (static/models/resistor.onnx, trained in backend/resistor/) looks at
// a 160x48 colour picture and names the bands left to right. Everything after
// that is the colour code itself, kept in step with backend/resistor/bands.py
// by tests/test_resistor.py:
//
//   decodeBands()  bands as seen -> ohms and tolerance, working out which way
//                  round the resistor is held
//   BandVote       several agreeing reads before a value is shown

import { createSession, ort } from '../ortenv.js';

export const COLOURS = ['black', 'brown', 'red', 'orange', 'yellow', 'green', 'blue', 'violet', 'grey', 'white', 'gold', 'silver'];
export const CSS = { black: '#1a1716', brown: '#7a4423', red: '#d02824', orange: '#f07a1c', yellow: '#f2d32a', green: '#2c8c42',
  blue: '#2a4cc0', violet: '#8640aa', grey: '#868686', white: '#f0f0ec', gold: '#c59a3e', silver: '#c2c5cb' };

const DIGIT = Object.fromEntries(COLOURS.slice(0, 10).map((c, i) => [c, i]));
const MULTIPLIER = { ...Object.fromEntries(COLOURS.slice(0, 10).map((c, i) => [c, 10 ** i])), gold: 0.1, silver: 0.01 };
const TOLERANCE = { brown: 1, red: 2, green: 0.5, blue: 0.25, violet: 0.1, grey: 0.05, gold: 5, silver: 10 };
const E24 = [10, 11, 12, 13, 15, 16, 18, 20, 22, 24, 27, 30, 33, 36, 39, 43, 47, 51, 56, 62, 68, 75, 82, 91];
const E96 = [100, 102, 105, 107, 110, 113, 115, 118, 121, 124, 127, 130, 133, 137, 140, 143, 147, 150, 154, 158, 162, 165,
  169, 174, 178, 182, 187, 191, 196, 200, 205, 210, 215, 221, 226, 232, 237, 243, 249, 255, 261, 267, 274, 280, 287, 294,
  301, 309, 316, 324, 332, 340, 348, 357, 365, 374, 383, 392, 402, 412, 422, 432, 442, 453, 464, 475, 487, 499, 511, 523,
  536, 549, 562, 576, 590, 604, 619, 634, 649, 665, 681, 698, 715, 732, 750, 768, 787, 806, 825, 845, 866, 887, 909, 931,
  953, 976];
const SERIES5 = new Set([...E96, ...E24.map((v) => v * 10)]), SERIES4 = new Set(E24);

/** Read bands in the order given; null if that order is not a legal code. */
export function valueOf(bands) {
  const n = bands.length;
  if (n < 3 || n > 5) return null;
  const digits = n === 5 ? bands.slice(0, 3) : bands.slice(0, 2);
  const mult = n === 5 ? bands[3] : bands[2];
  const tol = n >= 4 ? bands[n - 1] : null;
  if (digits.some((d) => !(d in DIGIT)) || !(mult in MULTIPLIER)) return null;
  if (DIGIT[digits[0]] === 0) return null;
  if (tol !== null && !(tol in TOLERANCE)) return null;
  const significand = digits.reduce((acc, d) => acc * 10 + DIGIT[d], 0);
  return { ohms: significand * MULTIPLIER[mult], tolerance: tol ? TOLERANCE[tol] : 20,
    standard: (n === 5 ? SERIES5 : SERIES4).has(significand), bands: [...bands] };
}

/** Bands as seen left-to-right -> the most plausible reading (either direction). */
export function decodeBands(bands) {
  let forward = valueOf(bands), backward = valueOf([...bands].reverse());
  if (bands.length >= 4) {
    if (bands[0] === 'gold' || bands[0] === 'silver') forward = null;
    if (bands[bands.length - 1] === 'gold' || bands[bands.length - 1] === 'silver') backward = null;
  }
  let options = [forward, backward].filter(Boolean);
  if (!options.length) return null;
  if (options.length === 2 && options[0].bands.join() === options[1].bands.join()) options = options.slice(0, 1);
  options.sort((a, b) => Number(!a.standard) - Number(!b.standard));
  const best = { ...options[0] };
  best.ambiguous = options.length === 2 && options[0].standard === options[1].standard;
  if (best.ambiguous) best.alternative = options[1];
  return best;
}

export function formatOhms(ohms) {
  const trim = (v) => String(Number(v.toPrecision(3)));
  if (ohms >= 1e9) return trim(ohms / 1e9) + ' GΩ';
  if (ohms >= 1e6) return trim(ohms / 1e6) + ' MΩ';
  if (ohms >= 1e3) return trim(ohms / 1e3) + ' kΩ';
  return trim(ohms) + ' Ω';
}

export class ResistorReader {
  constructor(session, meta) {
    this.session = session;
    this.meta = meta;
    this.h = meta.input.height; this.w = meta.input.width;
    this.canvas = document.createElement('canvas');
    this.canvas.width = this.w; this.canvas.height = this.h;
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
    this.busy = false;
    this.ms = 0;
  }

  static async load(base = '/static/models/') {
    const resp = await fetch(base + 'resistor.json');
    if (!resp.ok) throw new Error('resistor model missing (run python -m backend.resistor.export)');
    const meta = await resp.json();
    return new ResistorReader(await createSession(base + (meta.file || 'resistor.onnx')), meta);
  }

  /**
   * Name the bands in a region of the camera picture.
   * @returns {bands: [colour names, left to right], probs: [0..1 each], conf: weakest band}
   */
  async read(source, region) {
    if (this.busy) return null;
    this.busy = true;
    try {
      const t0 = performance.now();
      const { w, h } = this;
      this.ctx.imageSmoothingQuality = 'high';
      this.ctx.drawImage(source, region.sx, region.sy, region.sw, region.sh, 0, 0, w, h);
      const rgba = this.ctx.getImageData(0, 0, w, h).data;
      const input = new Float32Array(3 * w * h), plane = w * h;
      for (let i = 0; i < plane; i++) {
        input[i] = rgba[i * 4] / 127.5 - 1;
        input[plane + i] = rgba[i * 4 + 1] / 127.5 - 1;
        input[2 * plane + i] = rgba[i * 4 + 2] / 127.5 - 1;
      }
      const out = await this.session.run({ [this.session.inputNames[0]]: new ort.Tensor('float32', input, [1, 3, h, w]) });
      const lp = out[this.session.outputNames[0]];
      const res = readBands(lp.data, lp.dims[0], lp.dims[lp.dims.length - 1]);
      this.ms = performance.now() - t0;
      return res;
    } finally { this.busy = false; }
  }
}

/** CTC best path over (T x C) log-probabilities -> bands with their confidence. */
export function readBands(logp, T, C) {
  const bands = [], probs = [];
  let prev = 0;
  for (let t = 0; t < T; t++) {
    let best = 0, bv = -Infinity;
    for (let c = 0; c < C; c++) { const v = logp[t * C + c]; if (v > bv) { bv = v; best = c; } }
    if (best !== 0 && best !== prev) { bands.push(COLOURS[best - 1]); probs.push(Math.exp(bv)); }
    else if (best !== 0 && probs.length) probs[probs.length - 1] = Math.max(probs[probs.length - 1], Math.exp(bv));
    prev = best;
  }
  return { bands, probs, conf: probs.length ? Math.min(...probs) : 0 };
}

/** A value is shown only when `need` of the last `window` reads agree on the
 *  same bands (in either direction). */
export class BandVote {
  constructor(window = 6, need = 3) { this.window = window; this.need = need; this.reads = []; }

  clear() { this.reads = []; }

  push(bands) {
    const key = bands && bands.length ? [bands.join(','), [...bands].reverse().join(',')].sort()[0] : '';
    this.reads.push({ key, bands });
    if (this.reads.length > this.window) this.reads.shift();
    const tally = new Map();
    for (const r of this.reads) if (r.key) tally.set(r.key, (tally.get(r.key) || 0) + 1);
    let top = null;
    for (const [k, n] of tally) if (!top || n > top.n) top = { k, n };
    if (!top || top.n < this.need) return null;
    return [...this.reads].reverse().find((r) => r.key === top.k).bands;
  }

  /** How many of the recent reads saw nothing at all. */
  get empties() { return this.reads.filter((r) => !r.key).length; }
}
