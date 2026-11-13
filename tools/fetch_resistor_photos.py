"""Download real, labelled resistor photos for training and testing the band reader.

    python tools/fetch_resistor_photos.py

The reader is trained mostly on drawings (backend/resistor/render.py). Drawings
alone do not transfer to real parts, so a few real photos are mixed in, and
others are held back to measure how well it really works. This script fetches
them from Wikimedia Commons, turns each resistor level, and cuts it to the
10:3 box the app uses:

  <work>/resistor/real/train/   photos mixed into training
  <work>/resistor/real/test/    photos never trained on (python tests/eval_resistor_real.py --dir ...)

File names are the bands (left to right or right to left), then the source.
The photos are free-licensed but not public domain, so they are NOT copied into
this repository; ATTRIBUTION.txt next to them records author and licence.

Sources (Wikimedia Commons):
  "<value> 5% axial resistor.jpg"  (22 photos)  CC BY-SA 2.0
  "Electronic-Axial-Lead-Resistors-Array.jpg"   public domain  (also kept in tests/fixtures/resistors/)
  "Resistor metal film 0.5W 1% 82R.jpg"         CC0            (also kept in tests/fixtures/resistors/)
  "Metal film resistor.jpg"                     CC BY-SA 3.0
"""
from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.recognition.config import WORK_DIR  # noqa: E402

API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "resistor-photo-fetch/0.1 (training data script)"}
OUT = os.path.join(WORK_DIR, "resistor", "real")
RAW = os.path.join(WORK_DIR, "resistor", "real_raw")
CROP_W, CROP_H = 480, 144

DIGITS = ["black", "brown", "red", "orange", "yellow", "green", "blue", "violet", "grey", "white"]
MULT = {-2: "silver", -1: "gold", **{i: c for i, c in enumerate(DIGITS)}}

# the 5 % carbon-film series: Commons title -> ohms. Every third one is held back for testing.
SERIES = [("1.5 ohm", 1.5), ("4.7 ohms", 4.7), ("10 ohms", 10), ("47 ohms", 47), ("100 ohms", 100), ("220 ohms", 220),
          ("270 ohms", 270), ("330 ohms", 330), ("470 ohms", 470), ("560 ohms", 560), ("680 ohms", 680),
          ("1 kiloohm", 1e3), ("2.2 kiloohms", 2.2e3), ("3.3 kiloohms", 3.3e3), ("4.7 kiloohms", 4.7e3),
          ("6.8 kiloohms", 6.8e3), ("10 kiloohms", 1e4), ("22 kiloohms", 2.2e4), ("47 kiloohms", 4.7e4),
          ("100 kiloohms", 1e5), ("330 kiloohms", 3.3e5), ("1 megaohm", 1e6)]


def bands_4(ohms: float) -> list[str]:
    """Colour code of a two-digit, 5 % resistor."""
    exp = int(np.floor(np.log10(ohms) + 1e-9)) - 1
    sig = int(round(ohms / 10 ** exp))
    return [DIGITS[sig // 10], DIGITS[sig % 10], MULT[exp], "gold"]


def fetch(titles: list[str], width: int = 1600) -> dict:
    """Commons titles -> {title: (local path, licence, artist)}; downloads what is missing."""
    os.makedirs(RAW, exist_ok=True)
    out = {}
    for i in range(0, len(titles), 20):
        query = urllib.parse.urlencode({"action": "query", "titles": "|".join(titles[i:i + 20]), "prop": "imageinfo",
                                        "iiprop": "url|extmetadata", "iiurlwidth": width, "format": "json",
                                        "iiextmetadatafilter": "LicenseShortName|Artist"})
        with urllib.request.urlopen(urllib.request.Request(API + "?" + query, headers=UA), timeout=60) as r:
            pages = json.load(r)["query"]["pages"]
        for page in pages.values():
            if "imageinfo" not in page:
                print("not found on Commons:", page.get("title"))
                continue
            info = page["imageinfo"][0]
            meta = info.get("extmetadata", {})
            path = os.path.join(RAW, page["title"].split(":", 1)[1].replace(" ", "_").replace("%", "pc"))
            if not os.path.exists(path):
                with urllib.request.urlopen(urllib.request.Request(info.get("thumburl") or info["url"], headers=UA),
                                            timeout=90) as r, open(path, "wb") as f:
                    f.write(r.read())
            out[page["title"]] = (path, meta.get("LicenseShortName", {}).get("value", "?"),
                                  meta.get("Artist", {}).get("value", "?"))
    return out


def level_crop(bgr: np.ndarray, fill: float = 0.62) -> np.ndarray | None:
    """One resistor on a plain light background -> a level 10:3 crop with the
    body spanning `fill` of the width."""
    h, w = bgr.shape[:2]
    notwhite = (bgr.min(axis=2) < 205).astype(np.uint8)
    k = max(9, int(0.035 * w)) | 1                                   # wider than a lead, narrower than the body
    body = cv2.morphologyEx(notwhite, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(body)
    if n < 2:
        return None
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.nonzero(lab == big)
    pts = np.stack([xs, ys], 1).astype(np.float32)
    mean = pts.mean(0)
    _, vecs = np.linalg.eigh(np.cov((pts - mean).T))
    ax = vecs[:, 1]
    angle = np.degrees(np.arctan2(ax[1], ax[0]))
    if angle > 90:
        angle -= 180
    if angle < -90:
        angle += 180
    m = cv2.getRotationMatrix2D((float(mean[0]), float(mean[1])), angle, 1.0)
    rot = cv2.warpAffine(bgr, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
    rpts = (pts - mean) @ np.array([[np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
                                    [np.sin(np.radians(angle)), np.cos(np.radians(angle))]], np.float32)
    length = float(np.percentile(rpts[:, 0], 99.5) - np.percentile(rpts[:, 0], 0.5))
    cx = float(mean[0] + (np.percentile(rpts[:, 0], 99.5) + np.percentile(rpts[:, 0], 0.5)) / 2)
    cw = length / fill
    ch = cw * CROP_H / CROP_W
    m2 = np.float32([[CROP_W / cw, 0, CROP_W / 2 - cx * CROP_W / cw], [0, CROP_H / ch, CROP_H / 2 - mean[1] * CROP_H / ch]])
    crop = cv2.warpAffine(rot, m2, (CROP_W, CROP_H), flags=cv2.INTER_AREA, borderValue=(255, 255, 255))
    # a good crop has the leads running out through mostly-background sides; if the
    # sides are full of resistor the photo was framed too tightly to cut out
    side = np.concatenate([crop[:, :CROP_W // 12], crop[:, -CROP_W // 12:]], axis=1)
    if (side.min(axis=2) > 205).mean() < 0.45:
        return None
    return crop


def main() -> int:
    for sub in ("train", "test"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    credits = []

    titles = {f"File:{name} 5% axial resistor.jpg": ohms for name, ohms in SERIES}
    got = fetch(list(titles))
    for i, (title, ohms) in enumerate(titles.items()):
        if title not in got:
            continue
        path, lic, artist = got[title]
        crop = level_crop(cv2.imread(path))
        if crop is None:
            print("could not find the resistor in", title)
            continue
        sub = "test" if i % 3 == 1 else "train"
        name = "-".join(bands_4(ohms)) + f"_series-{ohms:g}.jpg"
        cv2.imwrite(os.path.join(OUT, sub, name), crop, [cv2.IMWRITE_JPEG_QUALITY, 93])
        credits.append(f"{sub}/{name}: \"{title[5:]}\", {artist}, {lic}, Wikimedia Commons")

    # other photographers: only ever used for testing. Boxes are x0, y0, x1, y1 on a 1920-wide view.
    others = fetch(["File:Electronic-Axial-Lead-Resistors-Array.jpg", "File:Resistor metal film 0.5W 1% 82R.jpg",
                    "File:Metal film resistor.jpg"])
    array = others.get("File:Electronic-Axial-Lead-Resistors-Array.jpg")
    if array:
        img = cv2.imread(array[0])
        k = img.shape[1] / 1920.0
        truth = ["gold-orange-white-orange", "yellow-violet-black-gold", "brown-green-red-gold",
                 "gold-brown-black-brown", "gold-red-red-red", "yellow-violet-orange-gold"]
        for cx, name in zip([270, 567, 836, 1080, 1362, 1626], truth):
            box = img[int(320 * k):int(1020 * k), int((cx - 105) * k):int((cx + 105) * k)]
            crop = cv2.resize(cv2.rotate(box, cv2.ROTATE_90_COUNTERCLOCKWISE), (CROP_W, CROP_H), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(OUT, "test", name + "_array.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 93])
            credits.append(f"test/{name}_array.jpg: \"Electronic-Axial-Lead-Resistors-Array.jpg\", {array[2]}, {array[1]}, Wikimedia Commons")
    for title, box, name in [("File:Resistor metal film 0.5W 1% 82R.jpg", (800, 110, 1160, 218), "grey-red-black-gold-brown_82R"),
                             ("File:Metal film resistor.jpg", (720, 20, 1180, 158), "violet-green-black-gold-brown_75R")]:
        if title not in others:
            continue
        img = cv2.imread(others[title][0])
        k = img.shape[1] / 1920.0
        x0, y0, x1, y1 = [int(v * k) for v in box]
        cv2.imwrite(os.path.join(OUT, "test", name + ".jpg"),
                    cv2.resize(img[y0:y1, x0:x1], (CROP_W, CROP_H), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 93])
        credits.append(f"test/{name}.jpg: \"{title[5:]}\", {others[title][2]}, {others[title][1]}, Wikimedia Commons")

    with open(os.path.join(OUT, "ATTRIBUTION.txt"), "w", encoding="utf-8") as f:
        f.write("Crops of photographs from Wikimedia Commons, each under the licence shown.\n"
                "Cropped, rotated and resized by tools/fetch_resistor_photos.py.\n\n" + "\n".join(credits) + "\n")
    for sub in ("train", "test"):
        print(f"{sub}: {len(os.listdir(os.path.join(OUT, sub)))} photos in {os.path.join(OUT, sub)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
