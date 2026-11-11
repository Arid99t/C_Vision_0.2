// One shared ONNX Runtime Web instance for every on-device model.
//
// The WebAssembly build is vendored under /static/vendor/ort/ (no CDN, works
// offline on a LAN). Threads are used when the page is cross-origin isolated,
// which backend/app.py arranges with its COOP/COEP headers.

import * as ort from '../vendor/ort/ort.wasm.min.mjs';

const cores = navigator.hardwareConcurrency || 2;
ort.env.wasm.wasmPaths = new URL('../vendor/ort/', import.meta.url).href;
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.max(1, Math.min(4, cores - 1)) : 1;
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
