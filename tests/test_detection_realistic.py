"""Detection robustness check: a 'paper card' with a dark border + shadow on a
noisy background should yield ONE clean formula box, not boxes on the paper edge.
Run: python tests/test_detection_realistic.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cv2
    import numpy as np
except ImportError:
    print("SKIP: opencv/numpy not installed")
    sys.exit(0)

from clients.desktop import ar_app  # noqa: E402


def make_scene():
    # noisy grayish background (like a desk in dim light)
    bg = np.random.randint(60, 110, (480, 640, 3), np.uint8)
    # a white paper card with a visible dark border + drop shadow
    cv2.rectangle(bg, (150, 120), (520, 380), (40, 40, 40), -1)        # shadow
    cv2.rectangle(bg, (160, 110), (510, 360), (245, 245, 245), -1)     # paper
    cv2.rectangle(bg, (160, 110), (510, 360), (90, 90, 90), 3)         # dark border
    # the handwritten formula in the middle
    cv2.putText(bg, "H2O", (230, 270), cv2.FONT_HERSHEY_SIMPLEX, 2.4,
                (10, 10, 10), 6, cv2.LINE_AA)
    return bg


def run():
    scene = make_scene()
    boxes, mask = ar_app.detect_formula_regions(scene, return_mask=True)
    print(f"detected {len(boxes)} box(es): {boxes}")

    # exactly one region, and it should sit on the writing (center ~ (300, 250)),
    # not on the paper border (which spans x in [160,510], y in [110,360]).
    assert len(boxes) == 1, f"expected 1 clean box, got {len(boxes)} (paper edge leaking?)"
    x, y, w, h = boxes[0]
    cx, cy = x + w / 2, y + h / 2
    assert 220 < cx < 380 and 210 < cy < 300, f"box not on the writing: center=({cx},{cy})"
    # the box should be much smaller than the paper (not the whole card)
    assert w < 320 and h < 160, f"box too big, likely grabbed the card: {w}x{h}"
    print("ok   single box localized on the writing, paper edge rejected")

    # non-text backgrounds should NOT produce detections (the user's complaint)
    rng = np.random.default_rng(0)
    false_positives = 0
    for _ in range(8):
        noise = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)   # pure noise
        false_positives += len(ar_app.detect_formula_regions(noise))
    # a few dark blobs on a plain background (objects, not text)
    blobby = np.full((480, 640, 3), 200, np.uint8)
    cv2.circle(blobby, (200, 240), 60, (30, 30, 30), -1)
    cv2.rectangle(blobby, (380, 180), (520, 320), (40, 40, 40), -1)
    fp_blobs = len(ar_app.detect_formula_regions(blobby))
    print(f"     false positives: {false_positives} on noise, {fp_blobs} on solid blobs")
    assert false_positives == 0, "detector fired on random noise"
    assert fp_blobs == 0, "detector fired on solid non-text blobs"
    print("ok   rejects noise and solid non-text blobs (text-likeness filter)")
    print("\n2/2 passed")


if __name__ == "__main__":
    run()
