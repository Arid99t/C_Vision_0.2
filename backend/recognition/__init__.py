"""Handwritten-chemistry recognition.

  model.py        MobileRec (phone-sized) and the original CRNN, both CTC
  synthv2.py      camera-realistic training data (handwritten glyphs + fonts)
  preprocess.py   crop clean-up; prepare_crop is mirrored in static/js/preprocess.js
  decode.py       chemistry-constrained beam search (mirrored in static/js/decode.js)
  train.py        training; export.py -> ONNX for the server and the phone
  infer.py        inference wrapper (onnxruntime or PyTorch)
"""
import os

# Anaconda ships Intel MKL's libiomp5md.dll and PyTorch bundles its own copy.
# On Windows both get loaded, triggering "OMP: Error #15 ... already initialized"
# which aborts the process. This is the standard, widely-used workaround; it must
# be set BEFORE torch is imported, and this package __init__ runs first for every
# torch entry point (train / infer / model). setdefault respects a user override.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
