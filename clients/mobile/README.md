# Chem_Vision — Mobile (Android) client plan

The mobile app is a **thin client** of the same backend the desktop app uses
(your chosen "both via a shared backend" architecture). It does the camera +
AR overlay locally and calls the Python backend over HTTP for recognition and
chemistry. No ML runs on the phone in v1 (it can later, see "On-device" below).

---

## Option A — Zero-install web app (IMPLEMENTED ✅)

The fastest way to run on a phone today: a camera web page served by the backend
at **`/ar`**. The phone streams frames to the PC, which runs the full pipeline
(`backend/pipeline.py`) and returns boxes + the reaction; the browser overlays
them on the live video. No app install, no Android toolchain.

**Run it:**
1. Start the backend on the PC: `uvicorn backend.app:app --host 0.0.0.0 --port 8000`
2. Browsers require **HTTPS** for camera access, so expose it securely with ngrok:
   `ngrok http 8000` → gives an `https://…ngrok-free.app` URL.
3. On the phone, open **`https://…ngrok-free.app/ar`** in Chrome → tap **Start camera**.

(LAN-only alternative without ngrok: run uvicorn with a self-signed cert, or add
`http://<PC-IP>:8000` under `chrome://flags/#unsafely-treat-insecure-origin-as-secure`.)

This reuses everything. The native app below is the longer-term, fully on-device path.

---

```
┌─────────────────────────── Android phone ───────────────────────────┐
│  CameraX preview  ─▶  frame  ─▶  detect formula blobs (OpenCV/ML Kit) │
│        │                                   │                          │
│        ▼                                   ▼                          │
│   AR overlay (Canvas)  ◀──  recognized text + balanced reaction       │
└───────────────────────────────│──────────────────────────────────────┘
                                 │  HTTPS (JSON / multipart)
                                 ▼
                  Python backend (FastAPI)  ──  backend/app.py
                  /recognize  /react  /describe  /balance
```

## Recommended stack

| Concern            | Choice                                                   |
|--------------------|----------------------------------------------------------|
| App framework      | **Flutter** (one codebase; can add iOS later)            |
| Camera             | `camera` plugin (live frames) or `image_picker` (MVP)    |
| Blob detection     | Google **ML Kit** text detection, or port the OpenCV blob detector via `opencv_dart` |
| Overlay            | `CustomPaint` on top of the camera preview               |
| HTTP               | `http` or `dio`                                          |
| Proximity trigger  | Same geometry as desktop: gap between two detected boxes |

> Alternative: **Unity + AR Foundation (ARCore)** if you want true world-anchored
> 3D AR (molecules floating in space). Heavier, but stronger "AR" feel. The
> backend contract below is identical.

## Build order (mirrors how the desktop client was built)

1. Camera preview + draw a static rectangle overlay (prove the render loop).
2. Detect formula blobs per frame; draw their boxes.
3. POST each crop to `/recognize`; show the returned formula above the box.
4. Compute the gap between the two nearest labeled boxes; when it drops below a
   threshold, POST both formulas to `/react` and show the banner.
5. Polish: debounce recognition, cache labels per tracked box, handle 503.

Point the app at your machine's LAN IP (e.g. `http://192.168.1.20:8000`) and run
the backend with `--host 0.0.0.0`. For real deployment, put the backend behind
HTTPS and lock down CORS in `backend/app.py`.

## On-device recognition (later)

Export the trained CRNN to **ONNX** / **TFLite** and run it with
`onnxruntime`/`tflite_flutter` to remove the network round-trip. The chemistry
engine can also be reimplemented in Dart, or kept server-side. Keep the same
`/react` JSON shape so either path works.

---

## API contract (shared with the desktop client)

Base URL: `http://<host>:8000`

### `GET /health`
```json
{ "status": "ok", "recognizer_available": true, "torch_available": true }
```

### `POST /recognize`  — multipart form, field `image` (PNG/JPEG of ONE formula crop)
```json
{
  "raw": "H2O",
  "formula": "H2O",
  "confidence": 0.93,
  "describe": { "formula": "H2O", "pretty": "H₂O", "name": "Water",
                "valid": true, "composition": { "H": 2, "O": 1 } }
}
```
Returns **503** if no model is trained yet.

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
  "source": "database"
}
```
On unknown reactants: `{ "ok": false, "reason": "No known reaction ...", "reactants": [...] }`

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
