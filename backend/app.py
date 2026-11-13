"""Chem_Vision HTTP API — the shared backend for all clients.

Endpoints:
  GET  /ar                the phone web app (runs the models ON the phone)
  GET  /static/...        its scripts, models and 3D chemistry data
  GET  /health            liveness + what is available
  POST /recognize         image upload -> recognized formula text (the CV model)
  POST /describe          {formula} -> composition + common name ("what's written")
  POST /react             {reactants:[...]} -> balanced reaction (the "bring close" result)
  POST /balance           {reactants, products} -> balanced equation
  POST /frame             one camera frame -> boxes + reaction (server-side fallback)

Run:  uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import io
import mimetypes
import os
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from .chemistry import balance, describe_formula, molecule_geometry, normalize_ocr, react
from .recognition import infer
from .recognition.config import ONNX_PATH

# Windows takes MIME types from the registry, which often has none (or the
# wrong one) for these. Browsers refuse ES modules and streaming WebAssembly
# that arrive with the wrong type, so pin them.
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("application/json", ".json")

app = FastAPI(title="Chem_Vision API", version="0.2.0")

# permissive CORS so the desktop window and the mobile app can both call in.
# Tighten `allow_origins` before any real deployment.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=2048)   # models/wasm/JSON compress well


@app.middleware("http")
async def _web_app_headers(request: Request, call_next):
    response = await call_next(request)
    # Cross-origin isolation lets the on-device models use WebAssembly threads
    # (SharedArrayBuffer). Everything the page loads is served from here, so
    # the stricter embedding policy costs nothing.
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    if request.url.path.startswith(("/static/", "/ar", "/assist")):
        # always revalidate: an updated script or model shows up at once, while
        # unchanged files cost only a tiny "304 Not Modified"
        response.headers["Cache-Control"] = "no-cache"
    return response


# the web app's scripts, vendored libraries, exported models and 3D data
_STATIC = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=_STATIC), name="static")


def _describe_with_geometry(formula: str) -> dict:
    """describe_formula() plus 3D atom-layout data for the AR clients' molecule
    visualization (kept out of formula.py so the chemistry engine stays
    geometry-agnostic; see chemistry/geometry.py)."""
    d = describe_formula(formula)
    d["geometry"] = molecule_geometry(d["formula"]) if d["valid"] else None
    return d


# --- request models -------------------------------------------------------
class DescribeRequest(BaseModel):
    formula: str


class ReactRequest(BaseModel):
    reactants: List[str]


class BalanceRequest(BaseModel):
    reactants: List[str]
    products: List[str]


# --- endpoints ------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "recognizer_available": infer.recognizer_available(),
        "torch_available": infer.torch_available(),
        # what the phone web app (/ar) can run by itself, without this server's help
        "on_device": {
            "recognizer": os.path.exists(ONNX_PATH) and os.path.exists(
                os.path.join(_STATIC, "models", "formula_rec.json")),
            "chemistry_3d": os.path.exists(os.path.join(_STATIC, "data", "chem3d.json")),
            "objects": all(os.path.exists(os.path.join(_STATIC, "models", f"objects.{ext}"))
                           for ext in ("onnx", "json")),
        },
        # what the Assist app (/assist) has installed
        "assist": {
            "text_reader": _has_models("assist/ocr_det.onnx", "assist/ocr_rec.onnx"),
            "captions": _has_models("assist/asr_encoder.onnx", "assist/asr_decoder.onnx", "assist/asr_vocab.json"),
            "resistor_reader": _has_models("resistor.onnx", "resistor.json"),
        },
    }


def _has_models(*names: str) -> bool:
    return all(os.path.exists(os.path.join(_STATIC, "models", n)) for n in names)


@app.post("/recognize")
async def recognize(image: UploadFile = File(...)):
    """Run the handwriting model on one cropped formula image.

    Returns the raw model text, the normalized canonical formula, a confidence
    score, and a description. 503 if no trained model is available yet.
    """
    if not infer.recognizer_available():
        raise HTTPException(
            status_code=503,
            detail="Recognition model unavailable. Install requirements-core.txt and "
                   "restore the exported formula_rec.onnx model, or train and export it.",
        )
    try:
        data = await image.read()
        pil = Image.open(io.BytesIO(data)).convert("L")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Bad image: {exc}")

    result = infer.get_recognizer().predict(np.asarray(pil))
    formula = normalize_ocr(result["raw"])
    return {
        "raw": result["raw"],
        "formula": formula,
        "confidence": result["confidence"],
        "describe": _describe_with_geometry(formula) if formula else None,
    }


@app.post("/describe")
def describe(req: DescribeRequest):
    return _describe_with_geometry(req.formula)


@app.post("/react")
def do_react(req: ReactRequest):
    if not req.reactants:
        raise HTTPException(status_code=400, detail="No reactants provided.")
    return react(req.reactants)


@app.post("/balance")
def do_balance(req: BalanceRequest):
    try:
        reac, prod = balance(req.reactants, req.products)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"ok": True, "reactants": reac, "products": prod}


# --- mobile web client ----------------------------------------------------
_SESSION = None  # one stateful AR session for the phone stream (single user)


def _get_session():
    global _SESSION
    if _SESSION is None:
        from . import pipeline  # lazy: needs opencv + the client vision utils
        _SESSION = pipeline.build_session()
    return _SESSION


@app.get("/ar", response_class=HTMLResponse)
def ar_page():
    """The phone camera web app. Open this in Chrome on the phone."""
    with open(os.path.join(_STATIC, "ar.html"), encoding="utf-8") as f:
        return f.read()


@app.get("/assist", response_class=HTMLResponse)
def assist_page():
    """The hands-free Assist app for smart glasses and phones: safety check,
    electronics helper, captions and read-aloud. Runs on the device."""
    with open(os.path.join(_STATIC, "assist.html"), encoding="utf-8") as f:
        return f.read()


@app.post("/frame")
async def frame(image: UploadFile = File(...)):
    """Process one camera frame from the web client; return overlay boxes + reaction."""
    from . import pipeline
    data = await image.read()
    img = pipeline.decode_frame(data)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode frame.")
    return pipeline.process_to_state(_get_session(), img)


@app.get("/")
def root():
    return {"name": "Chem_Vision API", "docs": "/docs",
            "health": "/health", "mobile_app": "/ar", "assist_app": "/assist"}
