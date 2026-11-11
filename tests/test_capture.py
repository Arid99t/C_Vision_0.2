"""Headless test of the capture->save->load path (no webcam).
Run: python tests/test_capture.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cv2
    import numpy as np
except ImportError:
    print("SKIP: opencv/numpy not installed")
    from tests.webharness import skip_unavailable
    skip_unavailable("Optional dependency or generated asset unavailable; see message above")

from clients.desktop import capture            # noqa: E402


def _frame(text):
    img = np.full((360, 640, 3), 255, np.uint8)
    cv2.putText(img, text, (220, 210), cv2.FONT_HERSHEY_SIMPLEX, 2.4, (10, 10, 10), 6)
    return img


def test_capture_roundtrip():
    if __name__ != "__main__":
        import pytest
        pytest.importorskip("torch", reason="Training dataset test requires requirements-ml.txt")
    from backend.recognition.dataset import FolderDataset
    with tempfile.TemporaryDirectory() as tmp:
        img_dir = os.path.join(tmp, "images")
        labels = os.path.join(tmp, "labels.tsv")

        f1 = capture.save_sample(_frame("H2"), "H2", img_dir, labels)
        f2 = capture.save_sample(_frame("Na"), "Na", img_dir, labels)
        assert f1 and f2, "save_sample returned None (detection failed)"
        assert os.path.exists(os.path.join(img_dir, f1)), "image not written"
        print(f"ok   saved two samples: {f1}, {f2}")

        lines = [l.strip() for l in open(labels, encoding="utf-8") if l.strip()]
        assert lines == [f"{f1}\tH2", f"{f2}\tNa"], f"bad labels.tsv: {lines}"
        print("ok   labels.tsv written correctly")

        # undo removes the last one
        removed = capture._undo(img_dir, labels)
        assert removed == f2 and capture._count(labels) == 1
        print(f"ok   undo removed last sample ({removed})")

        # re-add and load through the fine-tune dataset
        capture.save_sample(_frame("CO2"), "CO2", img_dir, labels)
        ds = FolderDataset(img_dir, labels, augment=False)
        assert len(ds) == 2, f"expected 2 items, got {len(ds)}"
        tensor, target, label = ds[0]
        assert tensor.shape[1] == 32 and tensor.ndim == 3, f"bad tensor {tensor.shape}"
        print(f"ok   FolderDataset loaded {len(ds)} items; sample shape {tuple(tensor.shape)}")

    print("\n4/4 passed")


if __name__ == "__main__":
    test_capture_roundtrip()
