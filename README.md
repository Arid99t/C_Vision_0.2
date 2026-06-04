# Chem_Vision

An **AR chemistry app**: point a camera at handwritten chemical formulas, the
app reads them and shows what's written, and when you **bring two formulas
close together** it computes and overlays the balanced reaction.

> Example: write `CH₄` on one card and `H₂O` on another. Move them near each
> other and the app overlays **`CH₄ + H₂O → CO + 3 H₂`** ("Steam reforming of
> methane").

It runs as a **shared Python backend** (computer-vision model + chemistry
engine) with thin clients: a **Windows desktop** AR app today and an
**Android** app on the same API (see [`clients/mobile/`](clients/mobile/README.md)).

---

## What's real vs. what's scoped

Be aware of the honest engineering boundaries:

- **Reading handwriting** uses a custom CRNN+CTC model you train (synthetic data
  generator included; mix in real samples for accuracy). Handwriting OCR is
  never perfect — confidence is shown on screen.
- **Reactions** come from a **curated database + automatic balancing**, not
  open-ended prediction. Predicting products for *arbitrary* reactants is a
  research-grade, often-ambiguous problem, so the app answers reliably for known
  reaction types and says "no known reaction" otherwise. Add your own in
  [`backend/chemistry/data/reactions.json`](backend/chemistry/data/reactions.json).

---

## Architecture

```
                 camera frame
                      │
        ┌─────────────▼──────────────┐         clients/desktop/ar_app.py
        │  detect formula blobs       │         (OpenCV: webcam + AR overlay)
        │  track them across frames   │
        └─────┬───────────────┬───────┘
              │ crop          │ two boxes close?
              ▼               ▼
   POST /recognize       POST /react            ┌──────────── backend/ ────────────┐
              │               │   ───────────▶  │  recognition/  CRNN+CTC model     │
              ▼               ▼                  │  chemistry/    parser + balancer  │
   "CH4" (+conf)     "CH4 + H2O → CO + 3 H2"     │  app.py        FastAPI            │
              │               │                  └───────────────────────────────────┘
              └──── overlaid on the live video ───┘
```

## Repo layout

```
backend/
  app.py                     FastAPI server (the shared backend)
  chemistry/
    formula.py               parse formulas, normalize OCR output (0/O, subscripts)
    reactions.py             curated reaction lookup + combustion rule
    balancer.py              chempy balancing + the react() entry point
    data/reactions.json      <- add reactions here
    data/compounds.json      <- common names ("H2O" -> "Water")
  recognition/
    model.py                 CRNN (CNN + BiLSTM + CTC)
    charset.py               character vocabulary + CTC decode
    synth.py                 synthetic handwritten-formula generator
    dataset.py               datasets + CTC collation
    train.py                 training loop  ->  weights/crnn.pt
    infer.py                 inference wrapper (graceful fallback if untrained)
    fonts/                   drop handwriting .ttf here to improve training data
clients/
  desktop/ar_app.py          Windows/desktop AR client
  mobile/README.md           Android (Flutter) plan + API contract
tests/
  test_chemistry.py          11 tests   (chemistry engine)
  test_api.py                6 tests    (HTTP endpoints)
  test_client_headless.py    detection/tracking/proximity without a webcam
```

---

## Quickstart

### 0. Install
```powershell
# from the project root (d:\Chem_Vision)
C:\anaconda3\python.exe -m pip install -r requirements-core.txt      # chemistry + API
C:\anaconda3\python.exe -m pip install -r requirements-ml.txt        # torch (recognition)
C:\anaconda3\python.exe -m pip install -r requirements-desktop.txt   # opencv (desktop client)
```

### 1. Run the backend
```powershell
C:\anaconda3\python.exe -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```
Open http://localhost:8000/docs for interactive API docs.

### 2. Try the desktop AR app — *no trained model needed yet*
Use `--demo-labels` to tag the first two detected blobs so you can see the full
AR → reaction flow immediately:
```powershell
C:\anaconda3\python.exe -m clients.desktop.ar_app --server http://localhost:8000 --demo-labels CH4,H2O
```
Hold up two cards with dark writing on light paper, bring them together → the
reaction banner appears. Keys: `q` quit · `space` pause · `r` re-read.

Test detection on a still image instead of a webcam:
```powershell
C:\anaconda3\python.exe -m clients.desktop.ar_app --image path\to\photo.jpg
```

### 3. Train the recognition model, then drop `--demo-labels`
```powershell
# (optional) add handwriting fonts to backend/recognition/fonts/ first
C:\anaconda3\python.exe -m backend.recognition.train --epochs 8 --steps 800 --batch 32
```
This writes `backend/recognition/weights/crnn.pt`. Restart the backend and the
`/recognize` endpoint goes live; now the desktop app reads real handwriting:
```powershell
C:\anaconda3\python.exe -m clients.desktop.ar_app --server http://localhost:8000
```

### 4. Run the tests
```powershell
C:\anaconda3\python.exe tests\test_chemistry.py
C:\anaconda3\python.exe tests\test_api.py
C:\anaconda3\python.exe tests\test_client_headless.py
```

---

## How the "bring two formulas close" trigger works

Each frame the client detects formula blobs, tracks them across frames (so a
formula keeps its recognized label), and measures the **gap between the two
nearest labeled boxes**. When that gap drops below `--proximity-px` (default 70),
it sends both formulas to `/react` and overlays the balanced equation for a few
seconds. See [`clients/desktop/ar_app.py`](clients/desktop/ar_app.py).

## Improving recognition accuracy

The synthetic model learns from fonts, so it can misread *your* specific letters
(e.g. a handwritten `Na` read as `N4`). The fix is fine-tuning on your own ink.

1. **Capture your handwriting** with the built-in tool — write a formula, hold
   it to the webcam, press SPACE to save the auto-cropped sample:
   ```powershell
   C:\anaconda3\python.exe -m clients.desktop.capture --label H2
   # press SPACE for each shot; press n to switch label (O2, Na, Cl2, CO2 ...)
   ```
   Aim for ~20-40 samples across the formulas/letters you use. Keys: `SPACE`
   save · `n` new label · `u` undo · `q` quit. Saves to `data\real\`.
2. **Fine-tune** on them (the real samples are auto-augmented and oversampled):
   ```powershell
   C:\anaconda3\python.exe -m backend.recognition.train --resume `
       --real-dir data\real\images --real-labels data\real\labels.tsv
   ```
3. **Add handwriting fonts** to `backend/recognition/fonts/` (e.g. Google's
   *Caveat*, *Patrick Hand*) before training for extra variety.
4. Use a GPU build of PyTorch for much faster training.

## Adding reactions

Append to [`backend/chemistry/data/reactions.json`](backend/chemistry/data/reactions.json):
```json
{ "reactants": ["N2", "O2"], "products": ["NO"],
  "name": "Formation of nitric oxide", "conditions": "high temperature" }
```
Reactants are matched as an unordered set; coefficients are computed
automatically. Restart the backend to pick up changes.

## Limitations & roadmap

- Subscripts/charges in handwriting are hard; the model + the 0/O normalizer
  help, but verify low-confidence reads.
- The blob detector assumes dark ink on light paper; tune thresholds in
  `detect_formula_regions` for your lighting.
- Mobile app is specced but not built — see [`clients/mobile/`](clients/mobile/README.md).
- Next steps: on-device model (ONNX/TFLite), world-anchored 3D molecule AR
  (ARCore/Unity), larger reaction DB, reaction conditions/energetics.
```
