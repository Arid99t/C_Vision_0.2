"""Shared recognizer constants kept free of heavy deps (no torch import).

Lets the synthetic data generator and inference preprocessing share settings
without dragging in torch.
"""
import os

IMG_HEIGHT = 32  # fixed input height for the CRNN; width is variable

# default location for trained weights + charset snapshot
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "crnn.pt")
