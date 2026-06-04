"""Handwritten-chemistry recognition: CRNN + CTC model, synthetic data, training, inference."""
import os

# Anaconda ships Intel MKL's libiomp5md.dll and PyTorch bundles its own copy.
# On Windows both get loaded, triggering "OMP: Error #15 ... already initialized"
# which aborts the process. This is the standard, widely-used workaround; it must
# be set BEFORE torch is imported, and this package __init__ runs first for every
# torch entry point (train / infer / model). setdefault respects a user override.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
