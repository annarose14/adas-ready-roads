"""Feasibility probe: UFLD lane-detection network in OpenCV 5 DNN on DEV photos only.
Compares 4 input variants (full/crop x normal/mirror) by how many ego-lane lines are found,
grouped by your DEV labels. Writes visual comparisons to probe/."""
import csv
import random
import shutil
import time
from pathlib import Path

import cv2 as cv
import numpy as np

from adas_pipeline import resize

MODEL = "models/ufld_culane18_288x800.onnx"
IN_W, IN_H, GRID = 800, 288, 200
ROW_ANCHOR = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
MIN_PTS = 6          # a lane counts as found if detected on >= 6 of 18 rows
COLOURS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]

net = cv.dnn.readNetFromONNX(MODEL, engine=cv.dnn.ENGINE_CLASSIC)


def detect(img, mirror=False):
    src = cv.flip(img, 1) if mirror else img
    rgb = cv.cvtColor(cv.resize(src, (IN_W, IN_H)), cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
    net.setInput(((rgb - MEAN) / STD).transpose(2, 0, 1)[None].copy())
    out = net.forward()[0][:, ::-1, :]                      # (201, 18, 4)
    e = np.exp(out[:-1] - out[:-1].max(axis=0, keepdims=True))
    prob = e / e.sum(axis=0, keepdims=True)
    loc = (prob * np.arange(1, GRID + 1)[:, None, None]).sum(axis=0)
    loc[out.argmax(axis=0) == GRID] = 0
    h, w = src.shape[:2]
    col_w = (IN_W - 1) / (GRID - 1)
    lanes = []
    for i in range(4):
        pts = []
        for k in range(18):
            if loc[k, i] > 0:
                x = int(loc[k, i] * col_w * w / IN_W) - 1
                y = int(h * ROW_ANCHOR[17 - k] / IN_H) - 1
                pts.append((w - 1 - x if mirror else x, y))
        lanes.append(pts)
    if mirror:
        lanes = lanes[::-1]                                  # keep left-to-right order
    return lanes


def crop_band(img):
    h, w = img.shape[:2]
    bottom = int(h * 0.92)
    top = max(0, bottom - int(w / 2.78))
    return img[top:bottom], top


def run_variant(img, crop, mirror):
    if crop:
        band, top = crop_band(img)
        lanes = [[(x, y + top) for x, y in l] for l in detect(band, mirror)]
    else:
        lanes = detect(img, mirror)
    ego = sum(len(lanes[i]) >= MIN_PTS for i in (1, 2))
    return lanes, ego


def draw(img, lanes, title):
    out = img.copy()
    for i, pts in enumerate(lanes):
        for p in pts:
            cv.circle(out, p, 6, COLOURS[i], -1)
    cv.rectangle(out, (0, 0), (out.shape[1], 50), (0, 0, 0), -1)
    cv.putText(out, title, (10, 35), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return cv.resize(out, (640, int(640 * out.shape[0] / out.shape[1])))


random.seed(1)
rows = [r for r in csv.DictReader(open("dev_labels.csv"))]
groups = {"ready": [r for r in rows if r["my_label"] == "ready"],
          "bad": [r for r in rows if r["my_label"] in ("degraded", "fail")],
          "none": [r for r in rows if r["my_label"] == "none"]}
sample = [(g, r) for g, rs in groups.items() for r in random.sample(rs, min(8, len(rs)))]

variants = [("full", False, False), ("full_mirror", False, True),
            ("crop", True, False), ("crop_mirror", True, True)]
OUT = Path("probe")
if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir()
scores = {v[0]: {g: [] for g in groups} for v in variants}
times = []
for g, r in sample:
    img = cv.imread(f"data/{r['area']}/{r['image_id']}.jpg")
    if img is None:
        continue
    img = resize(img)
    tiles = []
    for name, crop, mirror in variants:
        t = time.time()
        lanes, ego = run_variant(img, crop, mirror)
        times.append(time.time() - t)
        scores[name][g].append(ego)
        tiles.append(draw(img, lanes, f"{name}: ego lanes {ego}/2"))
    grid = cv.vconcat([cv.hconcat(tiles[:2]), cv.hconcat(tiles[2:])])
    cv.imwrite(str(OUT / f"{g}_{r['image_id']}.jpg"), grid)

print(f"OpenCV {cv.__version__} UFLD probe on DEV photos | avg {1000 * np.mean(times):.0f} ms per run")
print("Mean ego-lane lines found (0-2), by YOUR label:")
print(f"{'variant':14s}{'ready':>8s}{'degraded/fail':>15s}{'none':>8s}")
for name, _, _ in variants:
    s = scores[name]
    fmt = lambda v: f"{np.mean(v):.2f}" if v else "-"
    print(f"{name:14s}{fmt(s['ready']):>8s}{fmt(s['bad']):>15s}{fmt(s['none']):>8s}")
print("Visual comparisons in probe/  (each image: 4 variants)")
