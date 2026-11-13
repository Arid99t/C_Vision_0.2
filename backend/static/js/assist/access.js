// Access mode: two aids for following a lesson.
//
//   Captions  what is being said, as large text (speech-to-text on the device,
//             see asr.worker.js). Nothing is sent anywhere.
//   Read      point at a board, slide or page, press Read: the text is shown
//             large and, where the device has a voice, read aloud.
//
// The microphone is opened only while Captions is showing and released the
// moment you leave it.

import { UI } from './icons.js';
import { readingOrder, workerUrl } from './ocr.js';

const BASE = '/static/models/assist/';
const SIZES = [1, 1.3, 1.6];
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

function voiceAvailable() {
  if (window.AssistNative?.speak) {
    try {
      return typeof window.AssistNative.isSpeechAvailable === 'function'
        ? !!window.AssistNative.isSpeechAvailable() : true;
    } catch { return false; }
  }
  return 'speechSynthesis' in window;
}

/** Rough letter size of a line, from its length and how many characters it
 *  holds. (More dependable than the height of the detected box.) */
function letterSize(l) {
  const n = l.text.replace(/ /g, '').length;
  return n >= 4 ? l.length / (0.5 * l.text.length) : 0.6 * l.lineHeight;
}

/** Lines of text -> paragraphs: a new one where the letters change size (a
 *  heading) or the gap to the previous line is clearly more than a line. */
export function paragraphs(lines) {
  const ordered = readingOrder(lines);
  const out = [];
  let prev = null;
  for (const l of ordered) {
    const a = prev ? letterSize(prev) : 0, b = letterSize(l);
    const sameRow = prev && Math.abs(l.cy - prev.cy) < 0.5 * Math.min(prev.lineHeight, l.lineHeight);
    const heading = prev && !sameRow && Math.abs(a - b) > 0.25 * Math.max(a, b);
    const gap = prev && !sameRow && l.cy - prev.cy > 2.3 * Math.min(a, b);
    if (!prev || heading || gap) out.push(l.text);
    else out[out.length - 1] += ' ' + l.text;
    prev = l;
  }
  return out;
}

export class AccessMode {
  constructor(app) {
    this.app = app;
    this.id = 'access';
    this.sub = app.opts.sub === 'read' ? 'read' : 'captions';
    this.worker = null;
    this.listening = false;
    this.speaking = false;
    this.finals = [];
    this.partial = '';
    this.lastMs = 0;
    this.lag = 0;
    this.level = 0;
    this.text = [];                // paragraphs last read
    this.readerOpen = false;
    this.voice = voiceAvailable();
    this.speakOn = true;
    this.working = false;
    this.error = null;
    this.captionGeneration = 0;
    this.readGeneration = 0;
    // Native TTS initializes asynchronously and may have no installed offline voice.
    window.addEventListener('assist-native-speech', () => {
      this.voice = voiceAvailable();
      if (!this.voice) this.stopSpeaking();
      if (this.app.mode === this) { this.setActions(); this.render(); }
    });
  }

  async enter(root) {
    this.root = root;
    this.voice = voiceAvailable();
    root.style.setProperty('--cap', this.app.settings.captionSize || 1);
    this.setActions();
    await this.setSub(this.sub, true);
  }

  setActions() {
    this.app.setActions([
      { id: 'captions', label: 'Captions', icon: UI.captions, on: this.sub === 'captions', run: () => this.setSub('captions') },
      { id: 'read', label: 'Read', icon: UI.read, on: this.sub === 'read', run: () => (this.sub === 'read' ? this.readNow() : this.setSub('read')) },
      { id: 'size', label: 'Text size', icon: UI.text, run: () => this.cycleSize() },
      ...(this.voice ? [{ id: 'speak', label: this.speakOn ? 'Voice on' : 'Voice off', icon: UI.speak, on: this.speakOn, run: () => this.toggleSpeak() }] : []),
    ]);
  }

  leave() {
    this.readGeneration++;
    this.stopCaptions();
    this.stopSpeaking();
    this.readerOpen = false;
  }

  back() {
    if (this.readerOpen) { this.closeReader(); return true; }
    return false;
  }

  async setSub(sub, force = false) {
    if (!force && sub === this.sub) return;
    this.sub = sub;
    const readGeneration = ++this.readGeneration;
    this.error = null;
    this.app.updateAction('captions', { on: sub === 'captions' });
    this.app.updateAction('read', { on: sub === 'read', label: sub === 'read' ? 'Read now' : 'Read' });
    if (sub === 'captions') {
      this.stopSpeaking();
      this.readerOpen = false;
      this.render();
      await this.startCaptions();
    } else {
      this.stopCaptions();
      this.render();
      try {
        this.app.busy('Loading the text reader…');
        this.reader = await this.app.textReader();
      } catch (err) {
        if (readGeneration === this.readGeneration) this.error = err.message || String(err);
      } finally {
        if (readGeneration === this.readGeneration && this.app.mode === this) this.app.busy(null);
      }
    }
    if (readGeneration === this.readGeneration && this.app.mode === this) this.render();
  }

  cycleSize() {
    const i = SIZES.indexOf(this.app.settings.captionSize || 1);
    const next = SIZES[(i + 1) % SIZES.length];
    this.app.saveSettings({ captionSize: next });
    this.root.style.setProperty('--cap', next);
    this.app.toast(['Normal text', 'Large text', 'Largest text'][SIZES.indexOf(next)]);
  }

  toggleSpeak() {
    this.speakOn = !this.speakOn;
    if (!this.speakOn) this.stopSpeaking();
    this.app.updateAction('speak', { on: this.speakOn, label: this.speakOn ? 'Voice on' : 'Voice off' });
  }

  // -------------------------------------------------------------- captions ---
  async startCaptions() {
    if (this.listening || this.startingCaptions) return;
    this.startingCaptions = true;
    const generation = this.app.generation;
    const captionGeneration = ++this.captionGeneration;
    const current = () => captionGeneration === this.captionGeneration && generation === this.app.generation
      && this.sub === 'captions' && this.app.mode === this;
    try {
      if (!this.worker) {
        this.app.busy('Loading captions (about 50 MB the first time)…');
        const worker = await new Promise((resolve, reject) => {
          const w = new Worker(workerUrl('./asr.worker.js'), { type: 'module' });
          const finish = (error) => {
            clearTimeout(timer);
            if (this.captionLoader?.worker === w) this.captionLoader = null;
            if (error) { w.terminate(); reject(error); } else resolve(w);
          };
          const timer = setTimeout(() => finish(new Error('captions took too long to load')), 120000);
          this.captionLoader = { worker: w, cancel: () => finish(new Error('captions closed')) };
          w.onmessage = (e) => {
            if (e.data.type === 'ready') finish(); else finish(new Error(e.data.message || 'captions failed to load'));
          };
          w.onerror = (e) => finish(new Error(e.message || 'captions failed to load'));
          try { w.postMessage({ type: 'init', base: new URL(BASE, location.href).href }); } catch (err) { finish(err); }
        });
        if (!current()) { worker.terminate(); return; }
        this.worker = worker;
        worker.onmessage = (e) => { if (current() && this.listening) this.onAsr(e.data); };
        worker.onerror = (e) => {
          if (!current()) return;
          this.stopCaptions();
          this.error = e.message || 'Captions stopped. Reopen Captions to try again.';
          this.app.busy(null);
          this.render();
        };
      }
      if (!current()) return;
      this.app.busy(null);
      this.worker.postMessage({ type: 'reset' });
      if (this.app.opts.audio) await this.playFile(this.app.opts.audio, current);
      else await this.openMic(current);
      if (!current()) return;
      this.listening = true;
    } catch (err) {
      if (current()) {
        this.stopCaptions();
        this.error = /denied|NotAllowed/i.test(String(err)) ? 'Microphone access was refused. Allow it and reopen Captions.'
          : (err.message || String(err));
        this.app.busy(null);
        this.render();
      }
    } finally {
      if (current()) {
        this.startingCaptions = false;
        this.app.busy(null);
        this.render();
      }
    }
  }

  async openMic(current) {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true }, video: false });
    // Permission can arrive after Back, pause or a mode change. Release that
    // late stream before constructing an audio graph or recording any samples.
    if (!current()) { stream.getTracks().forEach((t) => t.stop()); return; }
    this.stream = stream;
    const Ctx = window.AudioContext || window.webkitAudioContext;
    try { this.audio = new Ctx({ sampleRate: 16000 }); } catch { this.audio = new Ctx(); }
    const audio = this.audio, rate = audio.sampleRate;
    const src = audio.createMediaStreamSource(stream);
    const send = (block) => { if (current()) this.onBlock(block, rate); };
    if (audio.audioWorklet) {
      await audio.audioWorklet.addModule(new URL('./audio-tap.js', import.meta.url));
      if (!current()) return;
      const tap = new AudioWorkletNode(audio, 'tap');
      tap.port.onmessage = (e) => send(e.data);
      src.connect(tap);
    } else {                                              // older engines
      const node = audio.createScriptProcessor(4096, 1, 1);
      node.onaudioprocess = (e) => send(new Float32Array(e.inputBuffer.getChannelData(0)));
      src.connect(node); node.connect(audio.destination);
    }
    if (current() && audio.state === 'suspended') await audio.resume();
  }

  /** Caption a sound file instead of the microphone (demos and tests). */
  async playFile(url, current) {
    const data = await fetch(url).then((r) => { if (!r.ok) throw new Error('audio file not found'); return r.arrayBuffer(); });
    if (!current()) return;
    const ctx = new (window.OfflineAudioContext || window.webkitOfflineAudioContext)(1, 16000, 16000);
    const samples = (await ctx.decodeAudioData(data)).getChannelData(0);
    if (!current()) return;
    const padded = new Float32Array(samples.length + 16000);           // a second of quiet so the last utterance ends
    padded.set(samples);
    let i = 0;
    this.fileTimer = setInterval(() => {
      if (i >= padded.length) { clearInterval(this.fileTimer); this.worker?.postMessage({ type: 'flush' }); return; }
      this.onBlock(padded.slice(i, i + 1600), 16000);
      i += 1600;
    }, 20);
  }

  onBlock(block, rate) {
    let sum = 0;
    for (let i = 0; i < block.length; i += 4) sum += block[i] * block[i];
    this.level = Math.min(1, Math.sqrt(sum / (block.length / 4)) * 6);
    this.worker?.postMessage({ type: 'audio', samples: block, rate }, [block.buffer]);
  }

  stopCaptions() {
    this.captionGeneration++;
    this.startingCaptions = false;
    this.captionLoader?.cancel();
    clearInterval(this.fileTimer);
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    this.audio?.close?.().catch(() => {});
    this.audio = null;
    this.listening = false;
    this.speaking = false;
    this.working = false;
    this.partial = '';
    this.level = 0;
    // Captions are the largest model: release it even when only changing to Read.
    this.worker?.terminate();
    this.worker = null;
  }

  onAsr(m) {
    if (m.type === 'state') this.speaking = m.speaking;
    else if (m.type === 'partial') { if (m.text) this.partial = m.text; this.lastMs = m.ms; }
    else if (m.type === 'final') {
      this.partial = '';
      this.lastMs = m.ms;
      this.lag = m.ms / 1000 / Math.max(0.5, m.seconds);      // processing time per second of speech
      if (m.text) { this.finals.push(m.text); if (this.finals.length > 4) this.finals.shift(); }
    } else if (m.type === 'error') this.error = m.message;
    this.working = this.speaking;
    this.render();
  }

  // ------------------------------------------------------------------ read ---
  async readNow() {
    if (!this.reader || this.reading || !this.app.geo || this.sub !== 'read') return;
    this.reading = true;
    this.stopSpeaking();
    this.app.busy('Reading…');
    const g = this.app.geo, generation = this.app.generation;
    const readGeneration = this.readGeneration;
    const current = () => readGeneration === this.readGeneration && generation === this.app.generation
      && this.app.mode === this && this.sub === 'read';
    try {
      const res = await this.reader.read(this.app.source, { sx: g.sx, sy: g.sy, sw: g.sw, sh: g.sh },
        { workEdge: 1280, detEdge: 640, maxLines: 16, minHeight: 10 });
      if (!res || !current()) return;
      this.app.stats.readMs = res.totalMs;
      this.text = paragraphs(res.lines.filter((l) => l.conf >= 0.5));
      this.readerOpen = true;
      this.render();
      if (this.text.length && this.speakOn) this.speak(this.text);
      if (!this.text.length) this.app.toast('No text found. Move closer and try again.', 2400);
    } catch (err) {
      if (current()) { this.error = err.message || String(err); this.render(); }
    } finally { this.reading = false; if (current()) this.app.busy(null); }
  }

  speak(parts) {
    if (!voiceAvailable()) return;
    if (window.AssistNative?.speak) { window.AssistNative.speak(parts.join('\n\n')); return; }
    if (!('speechSynthesis' in window)) return;
    speechSynthesis.cancel();
    parts.forEach((p, i) => {
      const u = new SpeechSynthesisUtterance(p);
      u.rate = 0.95;
      u.onstart = () => { this.nowIndex = i; this.renderReader(); };
      u.onend = () => { if (i === parts.length - 1) { this.nowIndex = -1; this.renderReader(); } };
      speechSynthesis.speak(u);
    });
  }

  stopSpeaking() {
    this.nowIndex = -1;
    try { window.AssistNative?.stopSpeaking?.(); if ('speechSynthesis' in window) speechSynthesis.cancel(); } catch { /* no voice */ }
  }

  closeReader() {
    this.stopSpeaking();
    this.readerOpen = false;
    this.render();
  }

  // ---------------------------------------------------------------- output ---
  tick(now) {
    if (this.sub === 'captions' && now - (this.meterAt || 0) > 90) {
      this.meterAt = now;
      const bar = this.root?.querySelector('.meter i');
      if (bar) bar.style.width = Math.round(this.level * 100) + '%';
      this.level *= 0.7;
    }
  }

  render() {
    if (!this.root) return;
    if (this.sub === 'captions') {
      const state = this.error ? 'Not available' : !this.worker ? 'Loading…' : !this.listening ? 'Starting…'
        : this.speaking ? 'Hearing speech' : 'Listening';
      const old = this.finals.slice(-3).map((t) => `<p class="old">${esc(t)}</p>`).join('');
      const live = this.partial ? `<p class="live">${esc(this.partial)}</p>` : '';
      const empty = this.error ? `<p class="old">${esc(this.error)}</p>`
        : '<p class="old" style="font-size:.7em">Speech near you will appear here.</p>';
      // only worth saying when the device cannot keep up with the speaker
      const lag = this.lag > 0.8 ? '<span style="margin-left:auto">device is slow: captions lag</span>' : '';
      this.root.innerHTML = `<div class="captions"><div class="state">${UI.mic}<span>${state}</span>
        <span class="meter"><i></i></span>${lag}</div><div class="text" aria-live="polite">${old || live ? old + live : empty}</div></div>`;
    } else if (this.readerOpen) {
      this.root.innerHTML = '<div class="reader" id="readerBox"></div>';
      this.renderReader();
    } else {
      this.root.innerHTML = `<div class="hint-pill">${this.reader ? 'Point at the text, then choose Read now' : 'Loading…'}</div>
        <div class="panel result"><span class="v-icon dim">${UI.read}</span><div class="grow">
        <div class="big" style="font-size:1.15rem;white-space:normal">Read text ${this.voice ? 'aloud' : 'in large print'}</div>
        <div class="sub">${this.error ? esc(this.error) : 'Works on printed text: boards, slides, pages, labels. Hold still while it reads.'}
        ${this.voice ? '' : ' This device has no voice, so the text is shown only.'}</div></div></div>`;
    }
    this.app.updateAction('read', { on: this.sub === 'read', label: this.sub === 'read' ? (this.readerOpen ? 'Read again' : 'Read now') : 'Read' });
  }

  renderReader() {
    const box = this.root?.querySelector('#readerBox');
    if (!box) return;
    box.innerHTML = this.text.length
      ? this.text.map((p, i) => `<p class="${i === this.nowIndex ? 'now' : ''}">${esc(p)}</p>`).join('')
      : '<p style="color:var(--dim)">No text found. Move closer, hold still, and choose Read again.</p>';
    box.querySelector('.now')?.scrollIntoView({ block: 'nearest' });
  }

  onKey(e) {
    // while text is on screen, up/down scroll it
    if (this.readerOpen && (e.key === 'ArrowUp' || e.key === 'ArrowDown') && this.app.navScope().length && $sheetHidden()) {
      const box = this.root.querySelector('#readerBox');
      box?.scrollBy({ top: (e.key === 'ArrowDown' ? 0.6 : -0.6) * box.clientHeight, behavior: 'smooth' });
      return true;
    }
    return false;
  }

  state() {
    return { sub: this.sub, voice: this.voice, error: this.error,
      captions: { ready: !!this.worker, listening: this.listening, finals: [...this.finals], partial: this.partial, lastMs: this.lastMs },
      read: { open: this.readerOpen, text: [...this.text] } };
  }
}

const $sheetHidden = () => document.getElementById('sheet').hidden;
