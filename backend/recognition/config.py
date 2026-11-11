"""Shared recognizer constants kept free of heavy deps (no torch import).

Lets the synthetic data generator and inference preprocessing share settings
without dragging in torch.
"""
import os

IMG_HEIGHT = 32   # fixed input height for the recognizers; width is variable
MIN_WIDTH = 16    # narrowest model input (a single character)
MAX_WIDTH = 384   # widest model input (long "A + B" lines get squeezed to this)

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))

# --- trained weights -------------------------------------------------------
WEIGHTS_DIR = os.path.join(_HERE, "weights")
# legacy desktop model: VGG-style CRNN + BiLSTM (8.7 M parameters)
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "crnn.pt")
# mobile model: depthwise-separable CNN + small BiGRU (see model.MobileRec)
MOBILE_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "rec_mobile.pt")

# --- exported models served to the on-device web app ------------------------
STATIC_DIR = os.path.join(os.path.dirname(_HERE), "static")
MODELS_DIR = os.path.join(STATIC_DIR, "models")
ONNX_PATH = os.path.join(MODELS_DIR, "formula_rec.onnx")            # float32
ONNX_INT8_PATH = os.path.join(MODELS_DIR, "formula_rec.int8.onnx")  # quantized
REC_META_PATH = os.path.join(MODELS_DIR, "formula_rec.json")        # charset etc.

# --- bulky, regenerable training data lives OUTSIDE the repo ----------------
# Override with the CHEMVISION_WORK environment variable. The default prefers a
# roomy data drive when one exists and falls back to <repo>/work otherwise.
_DEFAULT_WORK = r"D:\ChemVision_work"
WORK_DIR = os.environ.get("CHEMVISION_WORK") or (
    _DEFAULT_WORK if os.path.isdir(_DEFAULT_WORK) else os.path.join(REPO_ROOT, "work"))
EMNIST_ZIP = os.path.join(WORK_DIR, "emnist", "gzip.zip")
GLYPH_DIR = os.path.join(WORK_DIR, "emnist")       # compact glyph bank (.npy)
FONT_DIR = os.path.join(WORK_DIR, "fonts")         # downloaded handwriting fonts
RECDATA_DIR = os.path.join(WORK_DIR, "recdata")    # pre-generated training sets
