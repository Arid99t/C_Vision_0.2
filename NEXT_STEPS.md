# Completion and remaining acceptance checks

Updated 2026-11-11. Work is in `C:\Users\MY PC\Desktop\Chem_Vision`.
The earlier uncommitted work was preserved. After the review, the owner requested
publication to `https://github.com/Arid99t/C_Vision_0.2`. The older
`D:\Chem_Vision` copy has not been changed; no application deployment was made.

## Completed in this pass

- Reviewed the backend, browser app, desktop pipeline, model tooling, tests and
  previous handoff. Verified the ONNX Runtime Web files against pinned SHA-256
  checksums.
- Installed core and test requirements in a new isolated `.venv-verify`, without
  inherited system packages. Added the missing OpenCV dependency for `/frame`.
  `pip check` reports no broken requirements.
- Added `tools/doctor.py` to check dependencies, asset integrity, model loading
  and chemistry before starting the server.
- Fixed Python/JavaScript formula parsing: reject empty groups, zero or
  leading-zero counts, unsupported digits and excessive sizes. Atom expansion
  uses the same validation. Bad balance requests return a client error.
- Fixed duplicate reactants and prevented zero/negative oxygen coefficients in
  the automatic combustion rule.
- Object models now load on demand. Invalid numeric URL options use defaults
  or are bounded. Added Pause/resume, camera release when leaving the page,
  duplicate-start protection and guards against stale asynchronous results.
- Corrected fallback privacy: crops are uploaded only after selecting the
  start screen's server-recognition option.
- Added provenance for nine NIST-checked heat values, including phases and
  reference temperature. Other heats are labelled unverified. This spot-check
  is not a scientific audit of every reaction and condition.
- Rebuilt 182 structures and 197 reaction animation plans. The build fails
  instead of silently publishing an incomplete reaction set.
- Integrated older script-only checks into pytest. Added regressions for
  parsing, HTTP errors, camera lifecycle, lazy loading and server opt-in.
  Missing optional models/dependencies now skip without aborting collection.
- Added a temporary fine-tune/export smoke test with stand-in images. It checks
  real-crop mixing and ONNX/PyTorch agreement at seven widths without changing
  production weights or claiming real-handwriting accuracy.
- Froze a synthetic handwriting regression image so rendering-library changes
  cannot silently alter the end-to-end test input.
- Added upstream licence texts, dependency notices, portable installation
  instructions and a phone acceptance checklist.

## Verification results

- Clean Python 3.13 environment: **62 passed, 2 skipped** (the two skips need
  PyTorch/training assets), including all browser model/parity checks.
- Existing development environment: **63 passed**, plus the separate
  fine-tune/export smoke test **1 passed**.
- Runtime checksums, `tools/doctor.py`, `pip check`, and `git diff --check` pass.
- No physical phone or real-handwriting results are claimed.

## Running and checking this checkout

The verified runtime is separate from the older build `.venv` and Anaconda:

```powershell
.venv-verify\Scripts\python.exe tools\doctor.py
.venv-verify\Scripts\python.exe -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/ar` on the PC. Use HTTPS on the phone, as described
in `clients/mobile/README.md`. New installations should follow the README's
`.venv-runtime` setup rather than a machine-specific Python path.

```powershell
python -m pip install -r requirements-test.txt
python -m playwright install chromium
python -m pytest tests -q
```

Training/capture dataset checks require PyTorch and a training checkpoint and
skip in a runtime-only install. The existing Anaconda installation emits a
native gmpy2 DLL diagnostic unless `MPMATH_NOGMPY=1` and
`SYMPY_GROUND_TYPES=python` are set; the clean environment has no such issue.
Anaconda packages were not modified.

## Deferred by the owner

The owner explicitly chose to do phone and real-handwriting testing later.

1. Run the phone checklist in `clients/mobile/README.md`. Record phone model,
   browser, orientation, FPS and find/read/object timings.
2. Collect labelled handwriting crops and run `python tests/eval_real.py`.
   Fine-tune only if the measured result warrants it. Synthetic tests do not
   establish real-world accuracy.
3. Optional room personalisation requires photos of the actual room. Existing
   object thresholds and production weights were retained.

## Distribution decisions and current scope

- A licence for original code remains the owner's decision. See
  `THIRD_PARTY_NOTICES.txt`; the optional Ultralytics model has its own terms.
  This pass does not select a licence on the owner's behalf.
- The published source includes regression fixtures, the formula model,
  chemistry bundle and vendored Three.js files. Environments,
  private samples, downloaded runtime binaries and object weights remain
  ignored. GitHub publication was explicitly requested by the owner.
- Single-character camera detection remains unsupported; use multi-character
  formula cards. Desktop overlays remain text-based; open `/ar` in a desktop
  browser for the 3D view.
- APK packaging, offline service-worker caching, WebGPU and world-anchored AR
  are optional future features outside the existing browser-app release scope.
- Keep `D:\ChemVision_work` if retraining is planned. No training data, old
  weights, logs or user files were deleted.
