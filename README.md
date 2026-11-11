# Chem_Vision

An **AR chemistry app**: point a camera at handwritten chemical formulas, the
app reads them and shows each one as a **3D molecule**, and when you **bring two
formulas close together** it plays the reaction in 3D — bonds breaking and
forming, electrons moving — and shows the balanced equation.

> Example: write `CH₄` on one card and `O₂` on another. Move them together and
> the app shows **`CH₄ + 2 O₂ → CO₂ + 2 H₂O`**, animates the four C–H and two
> O=O bonds breaking and the C=O and O–H bonds forming, and ends with
> "C −4 → +4 (oxidised); O 0 → −2 (reduced) · releases ≈ 890 kJ".

It can also label everyday **room objects** (laptop, monitor, bottle, cup, …).

On **Android** the whole thing runs **on the phone** in the browser — the PC only
serves the files. There is also a **Windows desktop** client that talks to the
Python backend. See [`clients/mobile/`](clients/mobile/README.md).

---

## What's real vs. what's scoped

Be aware of the honest engineering boundaries:

- **Reading handwriting** uses a small CTC model trained on synthetic data
  (handwriting fonts + real handwritten characters, pushed through a camera
  model). It has **not been tested on real handwriting from a real phone yet**;
  the numbers below are synthetic benchmarks. Handwriting OCR is never perfect —
  a label is only shown after repeated agreeing reads above a confidence gate.
  Fine-tuning on your own handwriting is supported (see below).
- **Reactions** come from a **curated list (about 170) + automatic balancing**,
  plus a combustion rule for C/H/O compounds — not open-ended prediction. For
  anything else the app says "no known reaction". Add your own in
  [`backend/chemistry/data/reactions.json`](backend/chemistry/data/reactions.json).
- **3D molecules** are real structures for the 182 compounds in
  [`structures.json`](backend/chemistry/data/structures.json) (built with RDKit;
  ionic compounds are shown as one formula unit of ions). A valid formula that
  is not in the library is shown as its atoms only, with no bonds claimed.
- **The reaction animation is a schematic of what changes** — which bonds break
  and form, where each valence electron ends up, oxidation states. It is not a
  step-by-step reaction mechanism.
- **Object labels are not error-free**, and cannot be. Measured accuracy is in
  the table below; labels that tested as unreliable are not shown at all.

---

## Architecture

```
 Android phone (Chrome)  — everything on the device          backend/static/
 ┌──────────────────────────────────────────────────────┐
 │ camera ─▶ js/detect.js      find handwriting          │   models/formula_rec.onnx
 │           js/track.js       keep labels stable        │   models/objects.onnx
 │           js/recognizer.js  read it (ONNX / WASM)     │   data/chem3d.json
 │           js/chem.js        formula? reaction?        │   vendor/three, vendor/ort
 │           js/mol3d.js       3D molecules + reactions  │
 │           js/objects.js     room objects (optional)   │
 └───────────────────────▲──────────────────────────────┘
                         │ files only (GET /ar, /static/...)
 ┌───────────────────────┴──────────────────────────────┐
 │ backend/app.py (FastAPI)                              │◀── clients/desktop/ar_app.py
 │   recognition/  model, training, export, inference    │    (OpenCV webcam client:
 │   chemistry/    parser, balancer, 3D structures,      │     POST /recognize, /react)
 │                 reaction animation planner            │
 │   objects/      object model build / evaluate / tune  │
 └──────────────────────────────────────────────────────┘
```

The JavaScript vision code is a line-by-line twin of the Python it replaces;
browser tests run both on the same inputs and compare (the crop preparation
matches pixel for pixel).

## Repo layout

```
backend/
  app.py                     FastAPI server; serves the phone app and the API
  pipeline.py                server-side frame pipeline (POST /frame, fallback)
  chemistry/
    formula.py               parse formulas, normalize OCR output (0/O, subscripts)
    reactions.py             curated reaction lookup + combustion rule
    balancer.py              chempy balancing + the react() entry point
    mol3d.py                 real 3D structures (RDKit + ion packing)
    reaction3d.py            reaction animation planner (atom map, bonds, electrons)
    build3d.py               builds static/data/chem3d.json from the two data files
    elements.py, geometry.py element data; older stylised Bohr-model layout (API field)
    data/reactions.json      <- add reactions here
    data/structures.json     <- add compounds (name + SMILES) here
    data/compounds.json      extra common names
  recognition/
    model.py                 MobileRec (0.57 M params) and the original CRNN
    synthv2.py               training data: handwritten glyphs + fonts + camera model
    glyphs.py, fontbank.py   EMNIST glyph bank; handwriting font pool
    gen_data.py              pre-generates training/benchmark sets to disk
    preprocess.py            crop clean-up (prepare_crop; twin: static/js/preprocess.js)
    decode.py                chemistry-constrained beam search (twin: static/js/decode.js)
    train.py                 training loop        ->  weights/rec_mobile.pt
    export.py                ONNX export          ->  static/models/formula_rec.onnx
    infer.py                 inference wrapper (onnxruntime, or PyTorch)
    synth.py, dataset.py     original v1 generator / datasets (kept for the CRNN)
  objects/
    vocabulary.json          <- what to look for, in words
    build.py                 exports the object model   ->  static/models/objects.onnx
    evaluate.py              measures accuracy, calibrates thresholds
    personalize.py           tunes a model to photos of your own room
    detect.py                reference decoder (twin: static/js/objects.js)
  static/
    ar.html, js/             the phone web app
    data/chem3d.json         molecules + reaction animation plans (built)
    models/                  exported models (+ .json describing each)
    vendor/                  Three.js (kept in git), ONNX Runtime Web (fetched)
clients/
  desktop/ar_app.py          Windows/desktop AR client
  desktop/capture.py         collect labelled samples of your handwriting
  mobile/README.md           Android guide + API contract
tools/fetch_web_runtime.py   downloads ONNX Runtime Web into static/vendor/ort
tests/                       see "Tests" below
```

---

## Quickstart

### 0. Install
```powershell
# from the project root (Python 3.11+ recommended)
python -m venv .venv-runtime
.\.venv-runtime\Scripts\Activate.ps1
python -m pip install -r requirements-core.txt      # API + chemistry + running the model
python -m pip install -r requirements-desktop.txt   # opencv (desktop client)
python tools\fetch_web_runtime.py                   # WebAssembly runtime for the phone (15 MB)
python tools\doctor.py                              # check assets, dependencies and model loading
```
Training and rebuilding need more: `requirements-ml.txt` (PyTorch) and
`requirements-build.txt` (RDKit, Ultralytics, Playwright — install the latter
into a separate venv, see the note in that file).

### 1. Run the backend
```powershell
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```
Open http://localhost:8000/docs for interactive API docs.

### 2. Android phone
`ngrok http 8000`, then open `https://…ngrok-free.app/ar` on the phone and tap
**Start camera** (browsers require HTTPS for the camera). Details, options and
troubleshooting: [`clients/mobile/README.md`](clients/mobile/README.md).

The optional object model loads only when **Objects** is enabled. **Pause**
releases the camera; switching away from the page also stops it. If on-device
recognition cannot load, server reading stays off until you enable the clearly
labelled crop-upload option on the start screen.

No handwriting handy? `…/ar?demo=Na,Cl2` labels any two scribbles as Na and Cl₂
so you can watch a reaction. On a PC, `http://localhost:8000/ar` works with a
webcam too.

### 3. Desktop AR app
```powershell
python -m clients.desktop.ar_app --server http://localhost:8000
```
Hold up two cards with dark writing on light paper, bring them together → the
reaction banner appears. Keys: `q` quit · `space` pause · `r` re-read · `d`
debug mask. `--demo-labels CH4,H2O` skips recognition; `--image photo.jpg` runs
on a still.

---

## The handwriting model

`weights/rec_mobile.pt` → `static/models/formula_rec.onnx` is a
depthwise-separable CNN + small bidirectional GRU with CTC, built for phones.
Compared with the original CRNN (`weights/crnn.pt`, kept and still loadable):

|                                   | original CRNN | mobile model |
|-----------------------------------|--------------:|-------------:|
| parameters                        | 8.74 M        | **0.57 M**   |
| file size                         | 35.0 MB       | **2.3 MB**   |
| time per crop, 1 CPU thread       | 23.9 ms       | **1.3 ms**   |
| unseen handwriting fonts, camera conditions | 14.7 % | **78.7 %** |
| handwritten characters (EMNIST test), camera conditions | 8.2 % | **88.3 %** |
| same, mild conditions             | 13.3 %        | **87.3 %**   |
| original generator (fonts on a clean page) | **81.3 %** | 78.2 % |

Accuracy = whole formula exactly right, 600 samples per set, each model through
its own full pipeline (`python tests\bench_recognizer.py`). With the confidence
gate at 0.5 the mobile model accepts 79–91 % of reads and 91–95 % of those are
right; the clients additionally require two agreeing reads before showing a
label. In headless Chromium on the development laptop one on-device read
(prepare + model + decode) takes about 8 ms and finding the writing about 7 ms
per frame.

Read these numbers for what they are:

- They are **synthetic**. The three "camera" sets come from the same generator
  family the mobile model trained on (with fonts and handwritten characters it
  never saw), so they favour it; the old model was never trained for tilted,
  tightly cropped, unevenly lit input. On the old model's own test set the two
  are about level.
- Nothing here has been measured on **real handwriting** or on a **real Android
  phone**. Do that first: `python tests\eval_real.py` (below).
- An 8-bit quantized copy was tried and discarded: it lost 6 points of accuracy
  and was not faster.

### Check and improve it on your own handwriting
```powershell
# 1. collect ~30 samples: write a formula, hold it to the webcam, press SPACE (n = new label)
python -m clients.desktop.capture --label H2O
# 2. see how both models do on them
python tests\eval_real.py
# 3. if needed, fine-tune on them and re-export for the phone
python -m backend.recognition.train --resume --real-dir data\real\images --real-labels data\real\labels.tsv
python -m backend.recognition.export
```
The fine-tuning path is covered by a capture round-trip and a temporary
two-step training/export smoke test (`tests/test_training_smoke.py`). The
exported outputs are compared with PyTorch at seven input widths. This checks
the workflow using stand-in images; real handwriting accuracy remains unmeasured.

### Retrain from scratch
```powershell
# training data lives outside the repo; set CHEMVISION_WORK or it uses D:\ChemVision_work / .\work
# download https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip into <work>\emnist\ first (560 MB)
python -m backend.recognition.glyphs       # handwritten glyph bank
python -m backend.recognition.fontbank     # ~80 handwriting fonts (Google Fonts)
python -m backend.recognition.gen_data --name train --n 480000 --seed 1
python -m backend.recognition.gen_data --name val --n 4000 --seed 5
python -m backend.recognition.gen_data --name bench_fonts  --split test --mode font  --n 3000 --seed 7
python -m backend.recognition.gen_data --name bench_glyphs --split test --mode glyph --n 3000 --seed 8
python -m backend.recognition.train        # ~7 min per epoch on a GTX 1650
python -m backend.recognition.export
```
Data generation takes about 30 minutes on 4 cores. Do not pass `--amp` on GTX
16xx cards (it produces NaNs there).

### How a read is decided
1. `detect` finds dark-ink blobs that look like a line of text.
2. `prepare_crop` flattens the lighting, stretches contrast, levels a tilted
   line, crops tight to the ink and scales to 32 px height.
3. The model scores every image column; a **chemistry-constrained beam search**
   keeps only strings that can be valid formulas, and prefers a known compound
   when two readings are close (CO vs Co, HF vs Hf).
4. The read counts only if it parses and its confidence (the model's probability
   for the whole string) is at least 0.5; a label appears after two agreeing
   reads and is re-checked every 1.5 s.

---

## 3D molecules and reactions

Everything the phone draws is precomputed into `backend/static/data/chem3d.json`:

```powershell
.venv\Scripts\python.exe -m backend.chemistry.build3d      # needs RDKit; ~2 min first time
```

- **Add a compound**: an entry in `structures.json` with a name and a SMILES
  string (ions written separately, e.g. `[Na+].[Cl-]`). The build checks that
  the SMILES really has the composition of its formula.
- **Add a reaction**: an entry in `reactions.json` (reactants, products, name;
  optional conditions, note, `dH`). Coefficients are computed; a single reactant
  is a decomposition (in the app: tap the formula → **Break it down**).

For each reaction the planner (`reaction3d.py`) finds the atom-to-atom
assignment that changes the fewest bonds, then tracks every valence electron
from where it sits before to where it sits after. Electrons that stay with an
atom they belonged to are drawn yellow (shared) or cyan (lone); electrons handed
to a different atom — a genuine transfer, as in Na → Cl — are pink. The tests
check that all 197 plans conserve atoms, charge, electrons and oxidation number.

`dH` is in kJ for the smallest whole-number balanced equation. Nine values
were spot-checked using NIST formation enthalpies; their phase assumptions,
source URLs and calculations are recorded in
[`thermochemistry.json`](backend/chemistry/data/thermochemistry.json).
The app labels these as 25 °C reference values, separate from the process
conditions. Other recorded heats are explicitly labelled **unverified heat
estimate**. This is a spot-check, not a full scientific audit.

## Room objects

Tap **Objects** in the app. The model is YOLOE-26 (open-vocabulary: it is told
what to find in words) with the vocabulary of
[`backend/objects/vocabulary.json`](backend/objects/vocabulary.json) baked in.

```powershell
.venv\Scripts\python.exe -m backend.objects.build              # nano: 10 MB, the default
.venv\Scripts\python.exe -m backend.objects.build --size s     # small: 39 MB, more accurate, ~2.5x slower
.venv\Scripts\python.exe -m backend.objects.evaluate           # measure + calibrate (needs COCO val2017 + LVIS, see file)
```

Measured on 1,200 LVIS/COCO validation photos, with each label's threshold set
for about 90 % precision ("when it names it, how often is it right" / "how many
of them it finds"):

| label | nano (default) | small (`--size s`) |
|---|---|---|
| laptop | 92 % / 69 % | 92 % / 90 % |
| mouse | 91 % / 66 % | 92 % / 75 % |
| monitor / TV | 91 % / 50 % | 91 % / 65 % |
| keyboard | 92 % / 41 % | 90 % / 50 % |
| plushie | 90 % / 33 % | 90 % / 43 % |
| bottle | 91 % / 27 % | 91 % / 23 % |
| cup | 90 % / 22 % | 90 % / 27 % |
| table / desk | 90 % / 21 % | 93 % / 8 % |
| fan | 86 % / 29 % | 82 % / 43 % |
| light | 88 % / 2 % | 85 % / 10 % |
| toy, card, book | **not shown** — measured precision 10–58 % | same |
| pc, cables, extension lead, LED light, notebook | **not measured** (no ground truth in LVIS); shown at a default threshold | same |

So: it is right about nine times in ten for the things it does name, it misses
a lot, and it is not usable for toys and cards as it stands. These are web
photos, not your room. In headless Chromium on the development laptop one pass
of the nano model takes about 180 ms at 640 px (120 ms at 480 px:
`/ar?objedge=480`).

**To get close to "no mistakes" in your own room**, train on your own photos:

```powershell
.venv\Scripts\python.exe -m backend.objects.personalize label --photos D:\room_photos   # a large model labels them
#   -> check the previews, delete/fix wrong ones
.venv\Scripts\python.exe -m backend.objects.personalize train --epochs 60               # trains + installs the small model
```
This pipeline has only been smoke-tested (36 stand-in photos, 2 epochs): it
runs end to end and the result loads in the app, but it has not been used on a
real room.

**Licence**: the optional YOLOE / Ultralytics model uses AGPL-3.0 or an
Ultralytics Enterprise licence. Review the upstream terms before distribution;
excluding model binaries from Git does not remove their obligations. The owner
has not selected a licence for the original app code. Dependency and asset
notices are in [`THIRD_PARTY_NOTICES.txt`](THIRD_PARTY_NOTICES.txt), including
links for the model and training inputs.

---

## Tests

Install the test tools and browser, then run all checks with one command:

```powershell
python -m pip install -r requirements-test.txt
python -m playwright install chromium
python -m pytest tests -q
```

Training checks additionally need `requirements-ml.txt` and the training
checkpoint; they skip in a runtime-only install. The optional object-model
test skips when its locally built model is absent. The individual commands
below still work. Synthetic camera inputs do not establish real-world accuracy.

```powershell
# engine and API (no browser needed)
python tests\test_chemistry.py           # 11  parser, balancer, reactions
python tests\test_api.py                 # 6   HTTP endpoints
python tests\test_decode.py              # 3   constrained beam search
python tests\test_geometry.py            # 10  Bohr-model layout (API field)
python tests\test_chem3d.py              # 5   182 structures, 197 reaction plans
python tests\test_client_headless.py     # 4   desktop detection/tracking/proximity
python tests\test_detection_realistic.py # 2   paper edges / noise rejected
python tests\test_capture.py             # 4   capture -> fine-tune dataset
python tests\test_webapp.py              # 3   /ar page + /frame fallback

# phone app in headless Chromium (needs playwright)
python tests\test_web_parity.py          # 4   JS twins == Python (prepare, decode, chem, detect)
python tests\test_web_recognizer.py      # 1   browser reads == server reads
python tests\test_web_objects.py         # 2   object decoder parity + timing
python tests\test_web_app.py             # 5   molecules, reaction, tap, on-device reading

# measurements
python tests\bench_recognizer.py         # old vs new recognizer
python tests\eval_real.py                # your own handwriting samples
python -m tests.bench_latency            # desktop pipeline latency
```

## Limitations & roadmap

- Not yet run on a real Android phone: speeds above are from desktop Chromium.
  Expect a mid-range phone to be several times slower, most noticeably for the
  object model.
- Not yet evaluated on real handwriting.
- The detector assumes dark ink on light paper, at least two characters, and
  roughly horizontal writing; a single letter such as `C` is not picked up.
- Reactions are limited to the curated list plus combustion.
- Object labels: see the table. Personalising to your room is the way forward.
- Possible next steps: native wrapper (WebView / TWA), world-anchored AR
  (WebXR / ARCore), reaction conditions as interactive controls, more compounds.
