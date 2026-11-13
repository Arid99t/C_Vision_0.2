// Live captions, off the main thread: microphone samples in, text out.
//
// Speech-to-text is the pretrained Moonshine "tiny" model (encoder + decoder,
// see tools/fetch_assist_models.py). It reads a whole utterance at a time, so
// this worker first cuts the audio stream into utterances at the pauses:
//
//   samples -> 30 ms frames -> loudness against a running noise floor
//           -> "speech started" / "speech ended" (or 9 s, whichever is first)
//           -> final text for the utterance
//   and, while someone is still talking and the worker has nothing else to do,
//   a provisional text of the utterance so far.
//
// It never queues provisional work, so a slow device simply shows fewer
// in-progress updates; finished utterances are always transcribed, in order.
//
//   -> {type:'init', base}                       <- {type:'ready', threads}
//   -> {type:'audio', samples: Float32Array, rate}
//   -> {type:'flush'}  (end of input: finish the current utterance)
//   -> {type:'reset'}
//   <- {type:'state', speaking}                  <- {type:'partial'|'final', text, ms, seconds}

import { createSession, ort, threads } from '../ortenv.js';

const SR = 16000, FRAME = 480;                       // 30 ms
const START_FRAMES = 3, END_FRAMES = 15;             // 90 ms of speech starts, 450 ms of quiet ends
const MAX_SECONDS = 9, MIN_SECONDS = 0.35, PREROLL = 10;     // frames kept from before speech began
const PARTIAL_EVERY = 1.3;                           // seconds of new audio before a provisional pass

let enc = null, dec = null, vocab = null, pastNames = [], special = new Set();
let carry = new Float32Array(0);                     // samples not yet framed
let resamplePos = 0;
let noise = 0.004, speaking = false, loudRun = 0, quietRun = 0;
let preroll = [], segment = [], segmentLen = 0, sincePartial = 0;
const finals = [];                                   // finished utterances waiting to be transcribed
let busy = false, lastPartialLen = 0;

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'init') {
      [enc, dec, vocab] = await Promise.all([
        createSession(m.base + 'asr_encoder.onnx'), createSession(m.base + 'asr_decoder.onnx'),
        fetch(m.base + 'asr_vocab.json').then((r) => { if (!r.ok) throw new Error('speech vocabulary missing'); return r.json(); }),
      ]);
      pastNames = dec.inputNames.filter((n) => n.startsWith('past_key_values'));
      special = new Set(vocab.special);
      self.postMessage({ type: 'ready', threads });
    } else if (m.type === 'audio') {
      feed(m.rate === SR ? m.samples : resample(m.samples, m.rate));
    } else if (m.type === 'flush') {
      endSegment();
      pump();
    } else if (m.type === 'reset') {
      carry = new Float32Array(0); speaking = false; loudRun = quietRun = 0;
      preroll = []; segment = []; segmentLen = 0; finals.length = 0; sincePartial = 0;
    }
  } catch (err) {
    self.postMessage({ type: 'error', message: String(err?.message || err) });
  }
};

function resample(x, rate) {
  const step = rate / SR, n = Math.floor((x.length - resamplePos) / step);
  const out = new Float32Array(Math.max(0, n));
  let pos = resamplePos;
  for (let i = 0; i < n; i++, pos += step) {
    const k = Math.floor(pos), f = pos - k;
    out[i] = x[k] * (1 - f) + (x[k + 1] ?? x[k]) * f;
  }
  resamplePos = pos - x.length;                       // carry the fractional position into the next block
  if (resamplePos < 0) resamplePos = 0;
  return out;
}

// ------------------------------------------------------ cutting utterances ---
function feed(samples) {
  const all = new Float32Array(carry.length + samples.length);
  all.set(carry); all.set(samples, carry.length);
  let i = 0;
  for (; i + FRAME <= all.length; i += FRAME) frame(all.subarray(i, i + FRAME));
  carry = all.slice(i);
  pump();
}

function frame(f) {
  let sum = 0;
  for (let i = 0; i < FRAME; i++) sum += f[i] * f[i];
  const rms = Math.sqrt(sum / FRAME);
  const loud = rms > Math.max(0.01, noise * 3.2);
  if (!speaking) {
    noise += (rms - noise) * (rms < noise ? 0.3 : 0.03);        // floor follows quiet quickly, noise slowly
    preroll.push(f.slice());
    if (preroll.length > PREROLL) preroll.shift();
    loudRun = loud ? loudRun + 1 : 0;
    if (loudRun >= START_FRAMES) {
      speaking = true; quietRun = 0; loudRun = 0;
      segment = preroll; segmentLen = preroll.length * FRAME; preroll = [];
      sincePartial = 0; lastPartialLen = 0;
      self.postMessage({ type: 'state', speaking: true });
    }
    return;
  }
  segment.push(f.slice());
  segmentLen += FRAME; sincePartial += FRAME;
  quietRun = loud ? 0 : quietRun + 1;
  if (quietRun >= END_FRAMES || segmentLen >= MAX_SECONDS * SR) endSegment();
}

function endSegment() {
  if (!speaking) return;
  speaking = false;
  self.postMessage({ type: 'state', speaking: false });
  if (segmentLen >= MIN_SECONDS * SR) finals.push(join(segment, segmentLen));
  segment = []; segmentLen = 0; quietRun = 0;
}

function join(frames, len) {
  const out = new Float32Array(len);
  let o = 0;
  for (const f of frames) { out.set(f, o); o += f.length; }
  return out;
}

// ----------------------------------------------------------- transcribing ---
async function pump() {
  if (busy || !enc) return;
  busy = true;
  try {
    while (finals.length) {
      const audio = finals.shift();
      const r = await transcribe(audio);
      self.postMessage({ type: 'final', text: r.text, ms: r.ms, seconds: audio.length / SR, waiting: finals.length });
    }
    if (speaking && sincePartial >= PARTIAL_EVERY * SR && segmentLen > lastPartialLen) {
      sincePartial = 0; lastPartialLen = segmentLen;
      const audio = join(segment, segmentLen);
      const r = await transcribe(audio);
      if (speaking || finals.length) self.postMessage({ type: 'partial', text: r.text, ms: r.ms, seconds: audio.length / SR });
    }
  } catch (err) {
    self.postMessage({ type: 'error', message: String(err?.message || err) });
  } finally {
    busy = false;
  }
  if (finals.length) pump();
}

async function transcribe(audio) {
  const t0 = performance.now();
  const h = (await enc.run({ [enc.inputNames[0]]: new ort.Tensor('float32', audio, [1, audio.length]) }))[enc.outputNames[0]];
  const feeds = { input_ids: new ort.Tensor('int64', BigInt64Array.from([BigInt(vocab.bos)]), [1, 1]),
    encoder_hidden_states: h, use_cache_branch: new ort.Tensor('bool', [false], [1]) };
  for (const n of pastNames) feeds[n] = new ort.Tensor('float32', new Float32Array(0), [1, 8, 0, 36]);
  const limit = Math.ceil((audio.length / SR) * vocab.tokens_per_second) + 2;
  const tokens = [];
  let repeats = 0;
  for (let step = 0; step < limit; step++) {
    const out = await dec.run(feeds);
    const logits = out.logits.data, C = out.logits.dims[2], base = logits.length - C;
    let best = 0, bv = -Infinity;
    for (let c = 0; c < C; c++) if (logits[base + c] > bv) { bv = logits[base + c]; best = c; }
    if (best === vocab.eos) break;
    repeats = best === tokens[tokens.length - 1] ? repeats + 1 : 0;
    if (repeats >= 4) break;                           // stuck repeating: stop
    tokens.push(best);
    for (const n of pastNames) {
      // cross-attention keys never change after the first step; self-attention ones grow
      if (n.includes('.encoder.')) { if (step === 0) feeds[n] = out[n.replace('past_key_values', 'present')]; }
      else feeds[n] = out[n.replace('past_key_values', 'present')];
    }
    feeds.input_ids = new ort.Tensor('int64', BigInt64Array.from([BigInt(best)]), [1, 1]);
    feeds.use_cache_branch = new ort.Tensor('bool', [true], [1]);
  }
  let text = '';
  for (const t of tokens) if (!special.has(t)) text += vocab.tokens[t] ?? '';
  text = text.replace(/▁/g, ' ').replace(/\s+/g, ' ').trim();
  if (!/[a-z0-9]/i.test(text)) text = '';
  return { text, ms: performance.now() - t0 };
}
