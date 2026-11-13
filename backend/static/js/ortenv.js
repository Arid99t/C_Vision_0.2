// One shared ONNX Runtime Web instance for every on-device model.
//
// The WebAssembly build is vendored under /static/vendor/ort/ (no CDN, works
// offline on a LAN). Threads are used when the page is cross-origin isolated,
// which backend/app.py arranges with its COOP/COEP headers.

import * as ort from '../vendor/ort/ort.wasm.min.mjs';

const cores = navigator.hardwareConcurrency || 2;
// A page that runs several runtimes at once (the Assist app: one on the page
// for its tiny models, one in each worker) can cap this one by setting
// self.ORT_THREADS before importing, so they do not fight over the same cores.
// ?threads=N on the page (passed on to its workers) caps them all: fewer threads run cooler.
const asked = Number(new URLSearchParams(self.location?.search || '').get('threads'));
const wanted = asked >= 1 ? Math.min(asked, Number.isFinite(self.ORT_THREADS) ? self.ORT_THREADS : 4)
  : Number.isFinite(self.ORT_THREADS) ? self.ORT_THREADS : Math.min(4, cores - 1);
ort.env.wasm.wasmPaths = new URL('../vendor/ort/', import.meta.url).href;
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.max(1, Math.min(4, wanted)) : 1;
ort.env.logLevel = 'error';

export const threads = ort.env.wasm.numThreads;
export { ort };

/** Create a session from a model URL, with a readable error if it fails. */
export async function createSession(url) {
  try {
    return await ort.InferenceSession.create(url, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
  } catch (err) {
    throw new Error(`could not load model ${url}: ${err.message || err}`);
  }
}
