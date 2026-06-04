"""Measure per-frame pipeline latency for the Chem_Vision AR pipeline.

"End-to-end" here = frame-in-hand -> overlay-ready, i.e. one call to
ARApp.process_frame (detect -> track -> recognize -> react -> render). It does
NOT include camera capture/USB transfer or display refresh ("glass-to-glass"),
which you can only measure by filming a stopwatch on screen.

Run (use the real interpreter; bare `python` is a broken stub on this box):
    C:\\anaconda3\\python.exe -m tests.bench_latency                  # synthetic frame
    C:\\anaconda3\\python.exe -m tests.bench_latency --image some.png  # a real photo
    C:\\anaconda3\\python.exe -m tests.bench_latency --camera 0 --frames 300

Reports median + p95 latency (ms) for the full frame and each stage, plus the
FPS implied by the median. Warms up first so one-time model/CUDA init doesn't
skew the numbers.

Quote the MEDIAN as your typical per-frame latency and P95 as the worst case.
Don't quote the mean: recognition only fires every `--recognize-interval`
frames, so the distribution is bimodal and the mean is meaningless.
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from backend.pipeline import build_session
from backend.recognition import infer
from clients.desktop import ar_app


def synthetic_frame(w=640, h=480):
    """A light 'paper' frame with two dark formulas, enough to drive detection."""
    frame = np.full((h, w, 3), 245, np.uint8)
    cv2.putText(frame, "CH4", (60, 250), cv2.FONT_HERSHEY_SIMPLEX, 3, (20, 20, 20), 6)
    cv2.putText(frame, "H2O", (370, 250), cv2.FONT_HERSHEY_SIMPLEX, 3, (20, 20, 20), 6)
    return frame


def stats(samples_ms):
    a = np.asarray(samples_ms, float)
    return dict(median=float(np.median(a)), p95=float(np.percentile(a, 95)),
                lo=float(a.min()), hi=float(a.max()))


def line(name, s):
    return (f"{name:<20} median {s['median']:6.1f} ms   p95 {s['p95']:6.1f} ms"
            f"   (min {s['lo']:.1f} / max {s['hi']:.1f})")


def frame_source(args):
    """Yield raw BGR frames forever (still image / synthetic) or until camera EOF."""
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            raise SystemExit(f"Could not read image: {args.image}")
        while True:
            yield img
    elif args.camera is not None:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            raise SystemExit(f"Could not open camera {args.camera}")
        try:
            while True:
                ok, fr = cap.read()
                if not ok:
                    return
                yield fr
        finally:
            cap.release()
    else:
        base = synthetic_frame()
        while True:
            yield base


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image")
    p.add_argument("--camera", type=int)
    p.add_argument("--frames", type=int, default=200)
    p.add_argument("--warmup", type=int, default=15)
    p.add_argument("--recognize-interval", type=int, default=8,
                   help="frames between recognitions per track "
                        "(8 = desktop client; 1 = mobile/streaming, recognizes every frame)")
    args = p.parse_args()

    session = build_session()
    session.cfg.recognize_interval = args.recognize_interval
    dev = session.client.recognizer_ok and infer.get_recognizer().device or "n/a (model not trained)"
    print(f"recognizer: {'available' if session.client.recognizer_ok else 'NOT available'}"
          f"   device: {dev}   recognize_interval: {args.recognize_interval}")

    src = frame_source(args)

    for _ in range(args.warmup):              # never time the first inferences
        session.process_frame(next(src).copy())

    total, detect = [], []
    for _ in range(args.frames):
        frame = next(src).copy()

        t0 = time.perf_counter()              # detection stage in isolation
        ar_app.detect_formula_regions(frame, session.cfg.max_regions,
                                      min_components=session.cfg.min_components)
        detect.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()              # the real end-to-end per-frame number
        session.process_frame(frame)
        total.append((time.perf_counter() - t0) * 1000)

    t = stats(total)
    print()
    print(line("FULL FRAME", t))
    print(line("  detection", stats(detect)))

    if session.client.recognizer_ok:          # isolate the spiky model stage
        rec = infer.get_recognizer()
        crop = synthetic_frame()[170:300, 40:240]
        for _ in range(5):
            rec.predict(crop)
        infer_ms = [(_t(rec, crop)) for _ in range(50)]
        print(line("  model inference", stats(infer_ms)))

    print()
    print(f"implied FPS (1000 / median full frame) = {1000.0 / t['median']:.1f}")


def _t(rec, crop):
    t0 = time.perf_counter()
    rec.predict(crop)
    return (time.perf_counter() - t0) * 1000


if __name__ == "__main__":
    main()
