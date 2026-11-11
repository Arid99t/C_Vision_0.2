// On-device formula recognizer: crop -> prepared image -> ONNX model -> text.
//
// The same three stages as backend/recognition/infer.py, running in the
// browser: preprocess.js prepares the crop exactly like training did, the
// exported model (static/models/formula_rec.onnx) scores each image column,
// and decode.js turns the scores into the most plausible VALID formula.

import { createSession, ort } from './ortenv.js';
import { configure, prepareCrop, toModelInput } from './preprocess.js';
import { beamCandidates, bestPathLogp, chooseReading, greedy, makeGrammar } from './decode.js';

export class Recognizer {
  constructor(session, meta, chem) {
    this.session = session;
    this.meta = meta;
    this.chem = chem;
    this.itos = meta.charset;
    this.classes = meta.charset.length;
    this.grammar = makeGrammar(chem.bundle.all_elements);
    this.decodeOpts = {
      beamWidth: meta.decode?.beam_width ?? 8,
      topk: meta.decode?.topk_chars ?? 6,
    };
    this.bonus = meta.decode?.lexicon_bonus ?? 2.0;
    this.inputName = session.inputNames[0];
    this.outputName = session.outputNames[0];
  }

  static async load(chem, base = '/static/models/') {
    const resp = await fetch(base + 'formula_rec.json');
    if (!resp.ok) throw new Error('formula model metadata missing (run backend.recognition.export)');
    const meta = await resp.json();
    configure(meta);
    const file = meta.files?.int8 || meta.files?.float32 || 'formula_rec.onnx';
    return new Recognizer(await createSession(base + file), meta, chem);
  }

  /**
   * Read one grayscale crop of a formula.
   * @param gray  Float64Array | Uint8Array, row-major (h x w), 0..255
   * @returns {raw, formula, confidence, greedy, valid, ms}
   *   raw         best reading (chemistry-constrained; prefers known compounds)
   *   formula     raw after 0/O clean-up — the canonical formula
   *   confidence  the model's probability for that whole string, 0..1
   *   valid       whether `formula` parses as a chemical formula
   */
  async read(gray, h, w) {
    const t0 = performance.now();
    const prep = prepareCrop(gray, h, w);
    const tensor = new ort.Tensor('float32', toModelInput(prep.data), [1, 1, prep.height, prep.width]);
    const out = await this.session.run({ [this.inputName]: tensor });
    const lp = out[this.outputName];
    const T = lp.dims[0], C = lp.dims[2];
    const result = this.decode(lp.data, T, C);
    result.ms = performance.now() - t0;
    result.prepared = prep;
    return result;
  }

  /** Scores (T x C log-probabilities) -> reading. Split out so it can be tested. */
  decode(logp, T, C) {
    const plain = greedy(logp, T, C, this.itos);
    const cands = beamCandidates(logp, T, C, this.itos, this.grammar, this.decodeOpts);
    const best = chooseReading(cands, (text) => this.chem.isKnownLine(text), this.bonus);
    const raw = best ? best.text : plain;
    const conf = Math.min(1, Math.exp(best ? best.logp : bestPathLogp(logp, T, C)));
    const formula = this.chem.normalizeOcr(raw);
    return {
      raw, formula, confidence: conf, greedy: plain,
      valid: !!formula && this.chem.parseFormula(formula) !== null,
    };
  }
}
