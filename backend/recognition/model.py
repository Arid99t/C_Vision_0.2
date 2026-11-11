"""Formula recognizers: image of one handwritten formula -> per-column character
scores, decoded with CTC.

Two architectures share the same contract — input (B, 1, 32, W) in [-1, 1],
output (T, B, num_classes) log-probabilities:

  * ``MobileRec`` (default, "mobile"): depthwise-separable CNN + a small
    bidirectional GRU. ~0.6 M parameters and ~70 M multiply-adds for a
    128-pixel-wide crop. Built to run on a phone (ONNX / WebAssembly), and its
    output is twice as dense along the width (T = W/2), which keeps narrow
    neighbours such as the "11" in C12H22O11 apart.

  * ``CRNN`` ("crnn"): the original VGG-style CRNN (Shi et al., 2015) with a
    two-layer BiLSTM. 8.7 M parameters, ~0.9 G multiply-adds for the same crop,
    T = W/4 + 1. Kept so existing ``crnn.pt`` checkpoints still load.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .charset import NUM_CLASSES
from .config import IMG_HEIGHT  # noqa: F401  (re-exported for convenience)


# --------------------------------------------------------------------------
# Mobile recognizer
# --------------------------------------------------------------------------
class _SqueezeExcite(nn.Module):
    def __init__(self, ch: int, reduction: int = 4):
        super().__init__()
        self.fc1 = nn.Conv2d(ch, ch // reduction, 1)
        self.fc2 = nn.Conv2d(ch // reduction, ch, 1)
        self.act = nn.ReLU(inplace=True)
        self.gate = nn.Hardsigmoid()

    def forward(self, x):
        s = x.mean(dim=(2, 3), keepdim=True)
        return x * self.gate(self.fc2(self.act(self.fc1(s))))


class _DSBlock(nn.Module):
    """Depthwise k x k conv -> BN -> ReLU -> (squeeze-excite) -> 1x1 conv -> BN -> ReLU,
    with a residual connection when the shape is unchanged."""

    def __init__(self, cin: int, cout: int, k: int = 3, stride=(1, 1), se: bool = False):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, k, stride, k // 2, groups=cin, bias=False)
        self.bn1 = nn.BatchNorm2d(cin)
        self.se = _SqueezeExcite(cin) if se else None
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)
        self.residual = cin == cout and tuple(stride) == (1, 1)

    def forward(self, x):
        y = self.act(self.bn1(self.dw(x)))
        if self.se is not None:
            y = self.se(y)
        y = self.act(self.bn2(self.pw(y)))
        return x + y if self.residual else y


MOBILE_WIDTHS = {
    "base": (24, 48, 96, 192, 288, 96),      # c0..c4, GRU hidden
    "small": (16, 32, 64, 128, 192, 64),
}


class MobileRec(nn.Module):
    arch = "mobile"

    def __init__(self, num_classes: int = NUM_CLASSES, size: str = "base", dropout: float = 0.1):
        super().__init__()
        c0, c1, c2, c3, c4, hid = MOBILE_WIDTHS[size]
        self.size = size
        self.stem = nn.Sequential(nn.Conv2d(1, c0, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(c0), nn.ReLU(inplace=True))   # 32 x W
        self.features = nn.Sequential(
            _DSBlock(c0, c1, 3, (2, 2)),             # 16 x W/2
            _DSBlock(c1, c2, 3, (2, 1)),             #  8 x W/2
            _DSBlock(c2, c2, 3),
            _DSBlock(c2, c3, 3, (2, 1)),             #  4 x W/2
            _DSBlock(c3, c3, 5),
            _DSBlock(c3, c3, 5),
            _DSBlock(c3, c4, 5, (2, 1), se=True),    #  2 x W/2
            _DSBlock(c4, c4, 5, se=True),
        )
        self.rnn = nn.GRU(c4, hid, num_layers=1, bidirectional=True, batch_first=False)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hid * 2, num_classes)

    @staticmethod
    def out_len(width):
        """Output timesteps for an input `width` (int or tensor): ceil(W / 2)."""
        return (width + 1) // 2

    def forward(self, x):
        f = self.features(self.stem(x))          # (B, C, 2, T)
        f = f.mean(dim=2)                        # collapse height -> (B, C, T)
        f = f.permute(2, 0, 1)                   # (T, B, C)
        seq, _ = self.rnn(f)
        return self.fc(self.drop(seq)).log_softmax(2)


# --------------------------------------------------------------------------
# Legacy desktop recognizer
# --------------------------------------------------------------------------
class CRNN(nn.Module):
    arch = "crnn"

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

    @staticmethod
    def out_len(width):
        return width // 4 + 1

    def forward(self, x):
        feat = self.cnn(x)                      # (B, 512, 1, T)
        b, c, h, w = feat.size()
        assert h == 1, f"expected collapsed height 1, got {h} (input height must be 32)"
        feat = feat.squeeze(2)                  # (B, 512, T)
        feat = feat.permute(2, 0, 1)            # (T, B, 512)
        seq, _ = self.rnn(feat)
        logits = self.fc(seq)                   # (T, B, num_classes)
        return logits.log_softmax(2)


def build_model(device: str = "cpu", arch: str = "crnn", size: str = "base") -> nn.Module:
    """Build a recognizer. `arch` is "mobile" or "crnn" (the default, so code
    written against the original single-model API keeps working)."""
    if arch == "mobile":
        return MobileRec(size=size).to(device)
    if arch == "crnn":
        return CRNN().to(device)
    raise ValueError(f"unknown recognizer architecture: {arch!r}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def load_checkpoint(path: str, device: str = "cpu"):
    """Load a checkpoint written by train.py; returns (model in eval mode, ckpt dict).
    Checkpoints without an 'arch' entry are the original CRNN."""
    ckpt = torch.load(path, map_location=device)
    arch = ckpt.get("arch", "crnn")
    model = build_model(device, arch, ckpt.get("size", "base"))
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt
