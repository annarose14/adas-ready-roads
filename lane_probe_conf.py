"""Probe 2: does UFLD's own CONFIDENCE separate ready vs degraded/fail vs none? (DEV photos only)
Ego-lane confidence = mean over rows of P(lane present) for the two ego lanes, from the
network's softmax (1 - P(background)). Uses the crop variant."""
import csv
import random

import cv2 as cv
import numpy as np

from adas_pipeline import resize

MODEL = "models/ufld_culane18_288x800.onnx"
IN_W, IN_H = 800, 288
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
net = cv.dnn.readNetFromONNX(MODEL, engine=cv.dnn.ENGINE_CLASSIC)


def ego_conf(img, mirror):
    h, w = img.shape[:2]
    bottom = int(h * 0.92)
    band = img[max(0, bottom - int(w / 2.78)):bottom]
    if mirror:
        band = cv.flip(band, 1)
    rgb = cv.cvtColor(cv.resize(band, (IN_W, IN_H)), cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
    net.setInput(((rgb - MEAN) / STD).transpose(2, 0, 1)[None].copy())
    out = net.forward()[0]                                   # (201, 18, 4)
    e = np.exp(out - out.max(axis=0, keepdims=True))
    p = e / e.sum(axis=0, keepdims=True)
    present = 1.0 - p[-1]                                    # (18, 4) P(lane at this row)
    return float(present[:, 1:3].mean()), float(present[:, 1:3].min(axis=1).mean())


random.seed(1)
rows = list(csv.DictReader(open("dev_labels.csv")))
groups = {"ready": [r for r in rows if r["my_label"] == "ready"],
          "bad": [r for r in rows if r["my_label"] in ("degraded", "fail")],
          "none": [r for r in rows if r["my_label"] == "none"]}
res = {(g, m): [] for g in groups for m in (False, True)}
for g, rs in groups.items():
    for r in rs:
        img = cv.imread(f"data/{r['area']}/{r['image_id']}.jpg")
        if img is None:
            continue
        img = resize(img)
        for m in (False, True):
            res[(g, m)].append(ego_conf(img, m))

print("UFLD ego-lane confidence on ALL dev photos (crop variant)")
print(f"{'variant':10s}{'group':8s}{'n':>4s}{'mean conf':>12s}{'weaker-lane conf':>18s}")
for m in (False, True):
    for g in groups:
        v = res[(g, m)]
        if v:
            a = np.array(v)
            print(f"{'mirror' if m else 'normal':10s}{g:8s}{len(v):>4d}{a[:, 0].mean():>12.2f}{a[:, 1].mean():>18.2f}")
