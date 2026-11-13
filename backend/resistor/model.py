"""The resistor band reader: a small CNN that turns a 48x160 colour picture
into 40 columns of band-colour scores, read out with CTC (like the formula
reader, but in colour and with 12 "letters").

About 0.26 M parameters; one read is a few milliseconds on a phone-class CPU.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .bands import CLASSES


class _Block(nn.Module):
    """Depthwise-separable conv + norm + activation, optional pooling."""

    def __init__(self, cin: int, cout: int, pool: tuple[int, int] | None):
        super().__init__()
        layers = [nn.Conv2d(cin, cin, 3, padding=1, groups=cin, bias=False), nn.BatchNorm2d(cin), nn.ReLU6(True),
                  nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU6(True)]
        if pool:
            layers.append(nn.MaxPool2d(pool))
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)


class BandReader(nn.Module):
    def __init__(self, classes: int = CLASSES, width: int = 40, hidden: int = 80):
        super().__init__()
        w = width
        self.stem = nn.Sequential(nn.Conv2d(3, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU6(True),
                                  nn.MaxPool2d(2))                       # 24 x 80
        self.features = nn.Sequential(
            _Block(w, w * 2, (2, 2)),                                    # 12 x 40
            _Block(w * 2, w * 3, (2, 1)),                                #  6 x 40
            _Block(w * 3, w * 4, (2, 1)),                                #  3 x 40
            _Block(w * 4, w * 4, None),
        )
        self.squeeze = nn.Conv2d(w * 4, w * 4, (3, 1), bias=False)       #  1 x 40
        self.norm = nn.BatchNorm2d(w * 4)
        self.rnn = nn.GRU(w * 4, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(hidden * 2, classes)

    def forward(self, x):
        """x: (N, 3, 48, 160) in [-1, 1]  ->  log-probabilities (T=40, N, classes)."""
        f = self.features(self.stem(x))
        f = torch.relu(self.norm(self.squeeze(f))).squeeze(2).transpose(1, 2)    # (N, T, C)
        out, _ = self.rnn(f)
        return torch.log_softmax(self.head(self.drop(out)), dim=2).transpose(0, 1)


def greedy(logp: torch.Tensor) -> list[list[int]]:
    """CTC best path for a batch: (T, N, C) -> class indices per sample."""
    best = logp.argmax(2).transpose(0, 1).tolist()
    out = []
    for row in best:
        seq, prev = [], 0
        for k in row:
            if k != prev and k != 0:
                seq.append(k)
            prev = k
        out.append(seq)
    return out
