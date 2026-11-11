# Chem_Vision on Android

The Android client is a **web app that runs entirely on the phone**. The PC only
hands out the files (page, scripts, models, 3D chemistry data); the camera frames
stay on the phone in the default on-device mode, with no per-frame network
round trip. Server fallback requires an explicit choice to upload formula crops.

```
┌──────────────────────────── Android phone (Chrome) ───────────────────────────┐
│  camera ─▶ detect.js      where is handwriting?          ~7 ms / frame *       │
│           track.js        which blob is which, stable labels                   │
│           recognizer.js   what does it say?  (ONNX model, WebAssembly) ~10 ms * │
│           chem.js         is it a formula? do two of them react?               │
│           mol3d.js        3D molecules + reaction animation (Three.js / WebGL) │
│           objects.js      optional: label room objects (ONNX model)            │
└───────────────────────────────────────────────────────────────────────────────┘
        ▲ files only                    * measured in headless Chromium on the dev laptop
        │
   backend/app.py  (FastAPI)   GET /ar   GET /static/...
```

## Run it

1. On the PC, once after cloning: `python tools/fetch_web_runtime.py`
   (downloads the WebAssembly runtime, 15 MB, into `backend/static/vendor/ort/`).
2. Start the backend: `uvicorn backend.app:app --host 0.0.0.0 --port 8000`
3. Browsers only give camera access to **HTTPS** pages. The easy way:
   `ngrok http 8000` → open the `https://…ngrok-free.app/ar` link on the phone.
   (LAN-only alternative: in Chrome on the phone, add `http://<PC-IP>:8000` under
   `chrome://flags/#unsafely-treat-insecure-origin-as-secure`.)
4. Tap **Start camera**. The first load downloads about 20 MB (runtime + models,
   or about 30 MB with the object model); after that the browser cache is used.

Controls: **Formulas** / **Objects** / **e⁻** (show electrons) toggles at the
bottom. **Pause** releases the camera and returns to the start screen. Switching
to another tab/app also stops it. The optional object model loads on the first
tap of **Objects**, so it does not delay formula startup.
Tap a recognised formula for its details; if it can decompose, **Break
it down** plays that reaction.

## What you see

- A recognised formula gets its **real 3D structure** floating above it: correct
  shape (bent water, tetrahedral methane, flat benzene), double/triple bonds,
  ions with their charges and relative sizes, and the valence electrons as dots
  (yellow = shared in a bond, cyan = lone).
- Bring two formulas together and, if the pair is in the reaction list, the
  molecules react: bonds that break glow red and fade, atoms travel to their
  places in the products, electrons move with them (electrons handed from one
  atom to another are pink), new bonds flash green, and the caption explains each
  step and ends with the oxidation-state changes.

It is a schematic of **what** changes, not a reaction mechanism.

## Handy URL options

| Option | Effect |
|---|---|
| `/ar?src=/static/test/x.png` | use an image or video file instead of the camera |
| `/ar?demo=Na,Cl2` | label the writing found with these formulas instead of reading it (demo a reaction with any two scribbles) |
| `/ar?objects=1` | start with object labels on |
| `/ar?conf=0.4` | minimum confidence for a read to count (default 0.5) |
| `/ar?objedge=480` | smaller object-model input for slower phones |
| `/ar?edge=384` | detection frame size (smaller = faster, misses smaller writing) |

## If the phone is too slow or too old

If the on-device reader cannot load, the start screen offers **Use server
recognition (send formula crops)**. It is off by default. Enable it to find
writing locally and send the small crops to `/recognize`. The original fully server-side pipeline (`POST /frame`) is still
there for other clients.

## A native app later

The models are plain ONNX files with documented inputs and outputs
(`backend/static/models/*.json`), and the preprocessing/decoding is specified by
two mirrored implementations (Python and JavaScript) kept in step by tests. A
native Kotlin or Flutter app can load the same files with ONNX Runtime Mobile and
port the roughly 600 lines of `detect.js`, `preprocess.js` and `decode.js`. A
quicker route to an installable app is wrapping this page in a WebView or
Trusted Web Activity.

---

## API contract (shared with the desktop client)

Base URL: `http://<host>:8000`

### `GET /health`
```json
{ "status": "ok", "recognizer_available": true, "torch_available": true,
  "on_device": { "recognizer": true, "chemistry_3d": true, "objects": true } }
```

### `POST /recognize`  — multipart form, field `image` (PNG/JPEG of ONE formula crop)
```json
{
  "raw": "H2O",
  "formula": "H2O",
  "confidence": 0.93,
  "describe": { "formula": "H2O", "pretty": "H₂O", "name": "Water",
                "valid": true, "composition": { "H": 2, "O": 1 },
                "geometry": { "atoms": [...], "bonds": [[0,1],[0,2]], "shape": "bent" } }
}
```
`confidence` is the model's probability for the whole string (0–1). Returns
**503** if no model is available. `geometry` is the older stylised Bohr-model
layout (see "3D atom models" in the root README); the web app uses the real
structures in `static/data/chem3d.json` instead.

### `POST /react`  — JSON
Request: `{ "reactants": ["CH4", "H2O"] }`
```json
{
  "ok": true,
  "equation": "CH₄ + H₂O → CO + 3 H₂",
  "reactants": { "CH4": 1, "H2O": 1 },
  "products":  { "CO": 1, "H2": 3 },
  "name": "Steam reforming of methane",
  "conditions": "700–1100 °C, Ni catalyst",
  "dH": 206,
  "source": "database",
  "geometry": { "CH4": {...}, "H2O": {...}, "CO": {...}, "H2": {...} }
}
```
`dH` is the approximate standard enthalpy change in kJ for the equation as
balanced (negative = releases heat), or `null`. On unknown reactants:
`{ "ok": false, "reason": "No known reaction ...", "reactants": [...] }`

> Note: render `equation`/`pretty` only where unicode subscripts are supported.
> For canvases without subscript glyphs, use the ASCII `formula` fields or map
> `₀–₉ → 0–9` and `→ → ->` (the desktop client's `to_ascii` does exactly this).

### `POST /describe`  — JSON
Request: `{ "formula": "H20" }`  →  normalizes the 0/O and returns the `describe` block above.

### `POST /balance`  — JSON
Request: `{ "reactants": ["Fe","O2"], "products": ["Fe2O3"] }`
```json
{ "ok": true, "reactants": { "Fe": 4, "O2": 3 }, "products": { "Fe2O3": 2 } }
```

## Final phone check

1. Run `python tools/doctor.py`, start the server and open its HTTPS `/ar` address.
2. Allow the camera; confirm the status says **on-device** and a 3D molecule
   appears over a clear H2O card. Try CH4 and O2 together, then move them apart.
3. Tap CaCO3 and try **Break it down**, then **Replay**.
4. Enable **Objects**. Record the find/read/object timings and whether the view
   remains responsive; try `?objedge=480` if needed.
5. Rotate the phone. Tap **Pause**, confirm the camera indicator stops, and
   restart. Switch apps and return; the start screen should be shown again.
6. Record the phone model, browser version, lighting and formula samples.
   Collect about 30 labelled crops with the desktop capture tool and run
   `python tests/eval_real.py` before judging handwriting accuracy.

Physical-device and real-handwriting testing are intentionally deferred to the
owner. Headless Chromium and the synthetic regression image are not substitutes.
