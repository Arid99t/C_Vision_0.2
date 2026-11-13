"""How heavy is the Assist app? Times each on-device step in headless Chromium.

    python tests/bench_assist.py --json-out assist-benchmark.json
                                            (run on an otherwise idle PC)

Prints, for default and single-thread OCR/captions (resistor always one thread,
matching the Assist app):
  label pass      one Safety pass over two packages (detector + reading 6 lines)
  chip pass       one Electronics pass over the small box around a chip
  page pass       one Read pass over a board with four lines
  resistor read   one colour-band read
  captions        final-utterance inference time per second of segmented audio

These are desktop headless-Chromium measurements on synthetic scenes and a
synthetic voice, not accuracy or real-device results. Load times use local
files and may benefit from the OS cache. Vision reports warmed medians;
captions use one complete clip, excluding model load, real-time audio capture
and the wait for a speaker to pause. No phone/glasses speed is inferred.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tests import assist_scenes as sc  # noqa: E402
from tests.webharness import browser_page  # noqa: E402

JS = """
async ({ cleaners, chip, board, resistor, speech }) => {
  // Match /assist: its small page-thread models use one thread, while OCR
  // and ASR workers follow the default or explicit ?threads= setting.
  self.ORT_THREADS = 1;
  const { TextReader } = await import('/static/js/assist/ocr.js');
  const load = async (url) => { const img = new Image(); img.src = url; await img.decode(); return img; };
  const med = (a) => { const s = a.slice().sort((x, y) => x - y), m = Math.floor(s.length / 2);
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
  const timed = async (promise, ms, label) => {
    let timer;
    try { return await Promise.race([promise, new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error(label + ' timed out')), ms);
    })]); } finally { clearTimeout(timer); }
  };
  const out = { isolated: self.crossOriginIsolated, cores: navigator.hardwareConcurrency,
    userAgent: navigator.userAgent, executionProvider: 'wasm' };

  let t0 = performance.now();
  const reader = await TextReader.create();
  let res = null, worker = null;
  try {
  out.textLoadMs = performance.now() - t0;
  out.threads = reader.threads;
  const pass = async (url, region, opts) => {
    const img = await load(url);
    const r = region(img);
    await reader.read(img, r, opts);                                       // warm-up
    const runs = [];
    for (let i = 0; i < 5; i++) {
      reader.level = 0;
      const start = performance.now(), result = await reader.read(img, r, opts);
      if (!result?.lines.length) throw new Error('OCR benchmark returned no text for ' + url);
      runs.push({ ...result, elapsedMs: performance.now() - start });
    }
    return { total: med(runs.map((x) => x.elapsedMs)), det: med(runs.map((x) => x.ms.det)), rec: med(runs.map((x) => x.ms.rec)),
             lines: runs[0].lines.length, samples: runs.length, warmups: 1,
             lineCounts: runs.map((x) => x.lines.length), elapsedMs: runs.map((x) => x.elapsedMs) };
  };
  const whole = (img) => ({ sx: 0, sy: 0, sw: img.naturalWidth, sh: img.naturalHeight });
  out.label = await pass(cleaners, whole, { workEdge: 960, detEdge: 480, maxLines: 6 });
  out.chip = await pass(chip, (img) => { const w = Math.round(0.24 * img.naturalWidth), h = Math.round(w * 0.56);
    return { sx: (img.naturalWidth - w) / 2, sy: (img.naturalHeight - h) / 2, sw: w, sh: h }; },
    { workEdge: 640, detEdge: 320, maxLines: 4, maxUpscale: 2.5, minHeight: 10 });
  out.page = await pass(board, whole, { workEdge: 1280, detEdge: 640, maxLines: 16, minHeight: 10 });

  const { ResistorReader } = await import('/static/js/assist/resistor.js');
  out.resistorThreads = (await import('/static/js/ortenv.js')).threads;
  t0 = performance.now();
  res = await timed(ResistorReader.load(), 90000, 'Resistor model load');
  out.resistorLoadMs = performance.now() - t0;
  const rimg = await load(resistor);
  const w = Math.round(0.125 * rimg.naturalWidth), h = Math.round(w * 0.3);
  const box = { sx: (rimg.naturalWidth - w) / 2, sy: (rimg.naturalHeight - h) / 2, sw: w, sh: h };
  const times = [];
  for (let i = 0; i < 25; i++) {
    const result = await timed(res.read(rimg, box), 25000, 'Resistor read');
    if (!result?.bands.length) throw new Error('Resistor benchmark returned no bands');
    times.push(res.ms);
  }
  out.resistorMs = med(times.slice(5));
  out.resistor = { samples: 20, warmups: 5, elapsedMs: times.slice(5), epoch: res.meta.epoch };

  // One complete buffer queues every utterance before the worker starts
  // transcription, making waiting===0 a reliable end-of-clip signal.
  const { workerUrl } = await import('/static/js/assist/ocr.js');
  worker = new Worker(workerUrl('./asr.worker.js'), { type: 'module' });
  t0 = performance.now();
  const ready = await timed(new Promise((resolve, reject) => {
    worker.onmessage = (e) => {
      if (e.data.type === 'ready') resolve(e.data);
      else if (e.data.type === 'error') reject(new Error(e.data.message || 'Captions failed to load'));
    };
    worker.onerror = (e) => reject(new Error(e.message || 'Captions worker crashed'));
    worker.onmessageerror = () => reject(new Error('Captions worker message could not be decoded'));
    worker.postMessage({ type: 'init', base: new URL('/static/models/assist/', location.href).href });
  }), 90000, 'Captions model load');
  out.speechLoadMs = performance.now() - t0;
  out.speechThreads = ready.threads;
  const response = await fetch(speech);
  if (!response.ok) throw new Error('Speech fixture missing: ' + response.status);
  const buf = await response.arrayBuffer();
  const audio = (await new OfflineAudioContext(1, 16000, 16000).decodeAudioData(buf)).getChannelData(0);
  const finals = [];
  const done = new Promise((resolve, reject) => {
    worker.onmessage = (e) => {
      const m = e.data;
      if (m.type === 'error') { reject(new Error(m.message || 'Captions failed')); return; }
      if (m.type !== 'final') return;
      if (!m.text?.trim() || !(m.seconds > 0) || !Number.isFinite(m.ms) || m.ms < 0) {
        reject(new Error('Captions returned an empty or invalid final result')); return;
      }
      finals.push(m);
      if (m.waiting === 0) {
        if (finals.length < 2) reject(new Error('Speech fixture produced fewer than two utterances'));
        else resolve();
      }
    };
    worker.onerror = (e) => reject(new Error(e.message || 'Captions worker crashed'));
    worker.onmessageerror = () => reject(new Error('Captions worker message could not be decoded'));
  });
  const padded = new Float32Array(audio.length + 16000); padded.set(audio);
  t0 = performance.now();
  worker.postMessage({ type: 'audio', samples: padded, rate: 16000 }, [padded.buffer]);
  worker.postMessage({ type: 'flush' });
  await timed(done, 90000, 'Captions transcription');
  out.speech = { utterances: finals.length, seconds: finals.reduce((a, f) => a + f.seconds, 0), ms: finals.reduce((a, f) => a + f.ms, 0),
                 elapsedMs: performance.now() - t0, inputSeconds: audio.length / 16000,
                 text: finals.map((f) => f.text).join(' '), segments: finals };
  return out;
  } finally {
    reader.dispose();
    worker?.terminate();
    await res?.session.release();
  }
}
"""


def run(threads: int | None, scene_urls: dict) -> dict:
    query = f"?threads={threads}" if threads else ""
    with browser_page("/static/blank.html" + query) as (page, _):
        result = page.evaluate(JS, scene_urls)
        if page.problems:
            raise RuntimeError("Browser benchmark errors: " + "\n".join(page.problems))
        result["chromium"] = page.context.browser.version
        result["requestedThreads"] = threads
        return result


def scenes() -> dict:
    shutil.copy(os.path.join(ROOT, "tests", "fixtures", "speech_sample.wav"), os.path.join(sc.OUT_DIR, "speech_sample.wav"))
    return {
        "cleaners": sc.packages(((30, 90, 200), [("ULTRA", 60, True), ("BLEACH", 120, True), ("Kills 99.9% of germs", 34, False)]),
                                ((250, 250, 250), [("Glass Cleaner", 76, True), ("with AMMONIA", 64, True),
                                                   ("Streak free shine", 34, False)], (20, 60, 160)), name="bench_cleaners.png"),
        "chip": sc.chip("SN74HC595N", "2148 G4", name="bench_chip.png"),
        "board": sc.board(["Photosynthesis", "Plants make glucose from", "carbon dioxide and water", "using energy from sunlight."],
                          name="bench_board.png"),
        "resistor": sc.resistor_frame(["yellow", "violet", "red", "gold"], name="bench_resistor.png"),
        "speech": "/static/test/speech_sample.wav",
    }


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="also print the full JSON report")
    ap.add_argument("--json-out", type=Path, help="save a reproducible JSON report (only after every benchmark succeeds)")
    args = ap.parse_args()
    started = datetime.now(timezone.utc).isoformat()
    os.makedirs(sc.OUT_DIR, exist_ok=True)
    scene_urls = scenes()
    rows = []
    for threads in (None, 1):
        r = run(threads, scene_urls)
        rows.append(r)
        s = r["speech"]
        print(f"\n{r['threads']} OCR / {r['speechThreads']} caption / {r['resistorThreads']} resistor threads "
              f"(cross-origin isolated: {r['isolated']}, cores: {r['cores']})")
        print(f"  load: text reader {r['textLoadMs']:.0f} ms, resistor reader {r['resistorLoadMs']:.0f} ms, captions {r['speechLoadMs']:.0f} ms")
        for key, name in (("label", "label pass"), ("chip", "chip pass"), ("page", "page pass")):
            p = r[key]
            print(f"  {name:<14} {p['total']:6.0f} ms   (find text {p['det']:.0f} ms + read {p['lines']} lines {p['rec']:.0f} ms)")
        print(f"  resistor read  {r['resistorMs']:6.1f} ms")
        print(f"  captions       {s['ms'] / s['seconds']:6.0f} ms per second of segmented audio "
              f"({s['utterances']} utterances, {s['seconds']:.1f} s; {s['elapsedMs']:.0f} ms elapsed): \"{s['text']}\"")
    asset_paths = [Path(ROOT) / "backend" / "static" / url.removeprefix("/static/") for url in scene_urls.values()]
    asset_paths.extend((Path(ROOT) / "backend" / "static" / "models" / "assist").glob("*"))
    asset_paths.extend(Path(ROOT) / "backend" / "static" / "models" / name for name in ("resistor.onnx", "resistor.json"))
    assets = {}
    for path in sorted(asset_paths):
        if path.is_file():
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            assets[path.relative_to(ROOT).as_posix()] = {"bytes": path.stat().st_size, "sha256": digest}
    report = {
        "schemaVersion": 1, "startedUtc": started, "finishedUtc": datetime.now(timezone.utc).isoformat(),
        "scope": "Desktop headless Chromium; generated scenes and synthetic speech. No physical-device measurements.",
        "environment": {"platform": platform.platform(), "machine": platform.machine(),
                        "processor": platform.processor(), "python": sys.version, "logicalCpus": os.cpu_count()},
        "method": {"vision": "Median of warmed reads; OCR totals include canvas preparation and worker round-trip.",
                   "captions": "One complete clip; summed final-segment inference milliseconds / segmented audio seconds. "
                               "Excludes model load, live capture, pause detection wait and provisional captions.",
                   "load": "Local-file loads; OS cache is not cleared."},
        "assets": assets, "runs": rows,
    }
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(encoded, encoding="utf-8")
        print(f"\nSaved {args.json_out}")
    if args.json:
        print(encoded, end="")
