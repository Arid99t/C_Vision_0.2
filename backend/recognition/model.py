"""CRNN (CNN + BiLSTM + CTC) for handwritten formula recognition.

A standard convolutional-recurrent text recognizer (Shi et al., 2015): the CNN
turns a fixed-height grayscale image into a sequence of feature columns, a
bidirectional LSTM models the sequence, and a linear head produces per-timestep
class logits for CTC. Image height is fixed at 32 px; width is variable, which
is what lets one model read formulas of different lengths.

For input (B, 1, 32, W) the output sequence length is T = W/4 - 1.
"""
from __future__ import annotations

import torch.nn as nn

from .charset import NUM_CLASSES
from .config import IMG_HEIGHT  # noqa: F401  (re-exported for convenience)


class CRNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, rnn_hidden: int = 256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),                                   # 32 -> 16,  W -> W/2
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),                                   # 16 -> 8,   W/2 -> W/4
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),                 # 8 -> 4,    width kept
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),                 # 4 -> 2,    width kept
            nn.Conv2d(512, 512, 2, 1, 0), nn.ReLU(True),          # 2 -> 1
        )
        self.rnn = nn.LSTM(512, rnn_hidden, num_layers=2,
                           bidirectional=True, batch_first=False)
        self.fc = nn.Linear(rnn_hidden * 2, num_classes)

    def forward(self, x):
        feat = self.cnn(x)                      # (B, 512, 1, T)
        b, c, h, w = feat.size()
        assert h == 1, f"expected collapsed height 1, got {h} (input height must be 32)"
        feat = feat.squeeze(2)                  # (B, 512, T)
        feat = feat.permute(2, 0, 1)            # (T, B, 512)
        seq, _ = self.rnn(feat)
        logits = self.fc(seq)                   # (T, B, num_classes)
        return logits.log_softmax(2)


def build_model(device: str = "cpu") -> CRNN:
    return CRNN().to(device)
