"""Measure the object model on labelled photos, and calibrate its thresholds.

    .venv\\Scripts\\python.exe -m backend.objects.evaluate --images 1500

Uses the LVIS v1 validation annotations (1,203 everyday categories, drawn on
COCO photos) for the images that are in COCO val2017. Both downloads live in
<work dir>/coco:

    http://images.cocodataset.org/zips/val2017.zip            (778 MB)
    https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip  (61 MB)

For every on-screen label that LVIS has ground truth for, this reports
precision (when it says "cup", how often is it a cup?) and recall (how many of
the cups does it find?) at IoU 0.5, using the very decoder the phone runs
(detect.py). It then picks, per label, the lowest score threshold that reaches
the target precision and writes them to thresholds.json for build.py.

LVIS is "federated": each photo is only checked for some categories. A
detection is counted as wrong only where LVIS verified the category absent or
labelled it exhaustively; elsewhere it is ignored.

Caveat: these are web photos, not your room. Treat the numbers as a rough guide.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import time
import zipfile

import numpy as np

from backend.recognition.config import WORK_DIR

from .build import MIN_SHOW_PRECISION, THRESHOLDS_PATH
from .detect import META_PATH, MODEL_PATH, ObjectDetector, iou_one_to_many

COCO_DIR = os.path.join(WORK_DIR, "coco")
REPORT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "accuracy.json")

# on-screen label -> LVIS category names that count as that label
LVIS_MAP = {
    "table": ["table", "desk", "dining_table", "coffee_table", "kitchen_table"],
    "laptop": ["laptop_computer"],
    "mouse": ["mouse_(computer_equipment)"],
    "keyboard": ["computer_keyboard"],
    "monitor": ["monitor_(computer_equipment) computer_monitor", "television_set"],
    "toy": ["toy", "doll"],
    "plushie": ["teddy_bear"],
    "notebook": ["notebook"],
    "book": ["book", "paperback_book", "hardback_book"],
    "card": ["card", "business_card", "identity_card", "postcard"],
    "bottle": ["bottle", "water_bottle", "wine_bottle", "beer_bottle", "thermos_bottle"],
    "cup": ["cup", "mug", "teacup"],
    "fan": ["fan"],
    "light": ["lamp", "lightbulb", "table_lamp"],
    "phone": ["cellular_telephone", "telephone"],
    "pen": ["pen", "pencil"],
    "headphones": ["headset", "earphone"],
    "chair": ["chair", "armchair", "folding_chair", "rocking_chair", "deck_chair"],
    "bag": ["backpack", "handbag", "shoulder_bag", "tote_bag", "duffel_bag"],
    "clock": ["clock", "wall_clock", "alarm_clock"],
    "remote": ["remote_control"],
    "speaker": ["speaker_(stero_equipment)"],
    "glasses": ["spectacles", "sunglasses"],
    "box": ["box"],
    "scissors": ["scissors"],
    "tablet": ["tablet_computer"],
    "person": ["person"],
}


def load_lvis():
    path = os.path.join(COCO_DIR, "lvis_v1_val.json.zip")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing (see this module's docstring for the URL)")
    with zipfile.ZipFile(path) as z:
        return json.load(z.open(z.namelist()[0]))


def evaluate(n_images: int = 1500, target_precision: float = 0.9, seed: int = 0,
             long_edge: int | None = None, write: bool = True, name: str = "objects") -> dict:
    model_path = os.path.join(os.path.dirname(MODEL_PATH), name + ".onnx")
    meta_path = os.path.join(os.path.dirname(META_PATH), name + ".json")
    det = ObjectDetector(model_path, meta_path)
    meta = det.meta
    lvis = load_lvis()
    name_to_id = {c["name"]: c["id"] for c in lvis["categories"]}
    label_cats = {}
    for label, names in LVIS_MAP.items():
        ids = {name_to_id[n] for n in names if n in name_to_id}
        if ids and any(lab["label"] == label for lab in meta["labels"]):
            label_cats[label] = ids
    cat_label = {c: lab for lab, ids in label_cats.items() for c in ids}

    zpath = os.path.join(COCO_DIR, "val2017.zip")
    zf = zipfile.ZipFile(zpath)
    have = set(zf.namelist())
    images = []
    for im in lvis["images"]:
        member = "val2017/" + im["coco_url"].rsplit("/", 1)[-1]
        if "val2017" in im["coco_url"] and member in have:
            decided = set(im.get("neg_category_ids", [])) | set()
            images.append((im, member, decided))
    gts = {}
    for a in lvis["annotations"]:
        if a["category_id"] in cat_label:
            gts.setdefault(a["image_id"], []).append(a)
    # photos that say something about at least one of our labels
    relevant = [it for it in images
                if it[0]["id"] in gts or (it[2] & set(cat_label))]
    random.Random(seed).shuffle(relevant)
    relevant = relevant[:n_images]

    from PIL import Image
    records = {lab: [] for lab in label_cats}        # (score, is_true_positive)
    n_gt = {lab: 0 for lab in label_cats}
    spent, t0 = 0.0, time.time()
    for k, (im, member, neg) in enumerate(relevant):
        rgb = np.asarray(Image.open(io.BytesIO(zf.read(member))).convert("RGB"))
        t1 = time.perf_counter()
        dets = det.detect(rgb, long_edge=long_edge, floor=0.05)
        spent += time.perf_counter() - t1
        not_ex = set(im.get("not_exhaustive_category_ids", []))
        by_label = {}
        for a in gts.get(im["id"], []):
            x, y, w, h = a["bbox"]
            by_label.setdefault(cat_label[a["category_id"]], []).append([x, y, x + w, y + h])
        for lab, cats in label_cats.items():
            truth = np.array(by_label.get(lab, []), dtype=np.float64).reshape(-1, 4)
            n_gt[lab] += len(truth)
            mine = [d for d in dets if d["label"] == lab]
            if not mine:
                continue
            # may unmatched detections be called wrong here?
            judged = bool(cats & neg) or (len(truth) > 0 and not (cats & not_ex))
            used = np.zeros(len(truth), bool)
            for d in mine:                                   # already best-first
                hit = -1
                if len(truth):
                    ious = iou_one_to_many(np.array(d["box"]), truth)
                    ious[used] = 0
                    if ious.max() >= 0.5:
                        hit = int(ious.argmax())
                if hit >= 0:
                    used[hit] = True
                    records[lab].append((d["score"], True))
                elif judged:
                    records[lab].append((d["score"], False))
        if (k + 1) % 250 == 0:
            print(f"  {k + 1}/{len(relevant)} images, {1000 * spent / (k + 1):.0f} ms per image", flush=True)

    rows, thresholds = [], {}
    for lab in label_cats:
        rec = sorted(records[lab], key=lambda r: -r[0])
        if not rec or n_gt[lab] == 0:
            continue
        scores = np.array([r[0] for r in rec])
        tp = np.cumsum([r[1] for r in rec])
        fp = np.cumsum([not r[1] for r in rec])
        precision, recall = tp / np.maximum(tp + fp, 1), tp / n_gt[lab]
        # average precision (area under the precision envelope)
        env = np.maximum.accumulate(precision[::-1])[::-1]
        ap = float(np.sum(np.diff(np.concatenate([[0.0], recall])) * env))
        ok = np.nonzero((precision >= target_precision) & (tp >= 5))[0]
        if ok.size:
            i = int(ok[-1])                                  # lowest threshold that still qualifies
            reached = True
        else:
            # The target is out of reach for this label. Prefer being right over
            # finding more: take the most precise operating point that still has
            # a handful of correct detections behind it.
            enough = np.nonzero(tp >= 5)[0]
            pool = enough if enough.size else np.arange(len(scores))
            i = int(pool[np.argmax(precision[pool])])
            reached = False
        thr = float(max(0.3, round(float(scores[i]) - 1e-4, 3)))
        thresholds[lab] = {"threshold": thr, "precision": round(float(precision[i]), 3),
                           "recall": round(float(recall[i]), 3)}
        default = np.nonzero(scores >= 0.35)[0]
        rows.append({"label": lab, "objects": int(n_gt[lab]), "ap50": round(ap, 3),
                     "threshold": thr, "precision": round(float(precision[i]), 3),
                     "recall": round(float(recall[i]), 3), "target_reached": reached,
                     "precision_at_0.35": round(float(precision[default[-1]]), 3) if default.size else None,
                     "recall_at_0.35": round(float(recall[default[-1]]), 3) if default.size else None})
    rows.sort(key=lambda r: -r["ap50"])
    report = {"model": meta["model"], "images": len(relevant), "long_edge": long_edge or meta["long_edge"],
              "ms_per_image_cpu": round(1000 * spent / max(1, len(relevant)), 1),
              "target_precision": target_precision, "labels": rows,
              "not_measured": sorted(lab["label"] for lab in meta["labels"]
                                     if lab["label"] not in {r["label"] for r in rows}),
              "date": time.strftime("%Y-%m-%d")}
    print(f"\n{meta['model']} on {len(relevant)} LVIS/COCO val photos "
          f"({report['ms_per_image_cpu']:.0f} ms per image on this CPU, {time.time() - t0:.0f}s total)")
    print(f"{'label':<12}{'objects':>8}{'AP50':>7}{'thr':>7}{'prec':>7}{'recall':>8}   at 0.35: prec / recall")
    for r in rows:
        mark = "" if r["target_reached"] else "  (precision target not reachable)"
        d = (f"{r['precision_at_0.35']:.2f} / {r['recall_at_0.35']:.2f}"
             if r["precision_at_0.35"] is not None else "-")
        print(f"{r['label']:<12}{r['objects']:>8}{r['ap50']:>7.2f}{r['threshold']:>7.2f}"
              f"{r['precision']:>7.2f}{r['recall']:>8.2f}   {d}{mark}")
    print("no ground truth here for:", ", ".join(report["not_measured"]))
    if write:
        allthr = {}
        if os.path.exists(THRESHOLDS_PATH):
            with open(THRESHOLDS_PATH, encoding="utf-8") as f:
                allthr = json.load(f)
        allthr[meta["model"]] = thresholds
        with open(THRESHOLDS_PATH, "w", encoding="utf-8") as f:
            json.dump(allthr, f, indent=1)
        allrep = {}
        if os.path.exists(REPORT_PATH):
            with open(REPORT_PATH, encoding="utf-8") as f:
                allrep = json.load(f)
        allrep[meta["model"]] = report
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(allrep, f, indent=1)
        # apply the calibration to the deployed metadata right away
        from .build import apply_calibration
        apply_calibration(meta["labels"], thresholds)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1)
        hidden = [lab["label"] for lab in meta["labels"] if lab.get("hidden_reason")]
        if hidden:
            print("too unreliable to show (measured precision under "
                  f"{int(100 * MIN_SHOW_PRECISION)}%):", ", ".join(hidden))
        print(f"thresholds -> {THRESHOLDS_PATH} and {meta_path}; report -> {REPORT_PATH}")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate / calibrate the object model")
    p.add_argument("--images", type=int, default=1500)
    p.add_argument("--precision", type=float, default=0.9, help="target precision per label")
    p.add_argument("--long-edge", type=int, default=None)
    p.add_argument("--no-write", action="store_true")
    p.add_argument("--name", default="objects", help="model file stem in static/models (default: objects)")
    a = p.parse_args()
    if not os.path.exists(os.path.join(os.path.dirname(MODEL_PATH), a.name + ".onnx")):
        raise SystemExit("no object model built yet: python -m backend.objects.build")
    evaluate(a.images, a.precision, long_edge=a.long_edge, write=not a.no_write, name=a.name)
