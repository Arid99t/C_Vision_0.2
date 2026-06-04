"""Chem_Vision HTTP API — the shared backend for all clients.

Endpoints:
  GET  /health            liveness + whether recognition is available
  POST /recognize         image upload -> recognized formula text (the CV model)
  POST /describe          {formula} -> composition + common name ("what's written")
  POST /react             {reactants:[...]} -> balanced reaction (the "bring close" result)
  POST /balance           {reactants, products} -> balanced equation

Run:  uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import io
import os
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from PIL import Image
from pydantic import BaseModel

from .chemistry import balance, describe_formula, normalize_ocr, react
from .recognition import infer

app = FastAPI(title="Chem_Vision API", version="0.1.0")

# permissive CORS so the desktop window and the mobile app can both call in.
# Tighten `allow_origins` before any real deployment.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


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
    }


@app.post("/recognize")
async def recognize(image: UploadFile = File(...)):
    """Run the handwriting model on one cropped formula image.

    Returns the raw model text, the normalized canonical formula, a confidence
    score, and a description. 503 if no trained model is available yet.
    """
    if not infer.recognizer_available():
        raise HTTPException(
            status_code=503,
            detail="Recognition model unavailable. Install torch and train: "
                   "python -m backend.recognition.train",
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
        "describe": describe_formula(formula) if formula else None,
    }


@app.post("/describe")
def describe(req: DescribeRequest):
    return describe_formula(req.formula)


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
_STATIC = os.path.join(os.path.dirname(__file__), "static")
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
            "health": "/health", "mobile_app": "/ar"}
