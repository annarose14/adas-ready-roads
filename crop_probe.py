"""DEVELOPMENT probe: does an OpenCV 5 road focus help the VLM answer "are painted lane lines
visible?" Modes: full photo, road crop, road crop with non-road painted grey.
Data (development only; the fresh evaluation comes later on a new sample):
  1. former test suburbs: locations where both blind labelling passes agreed (marked or unmarked)
  2. the 48 location final set (already used, so development data now)
Same prompt and same photos in every mode (up to 4 road photos per location), majority vote.
Pre-agreed rule: continue only if a crop mode lifts pooled balanced accuracy by >= 5 points.
Writes probe_crop/ side by side images (full | crop | mask) for every location labelled 'no'."""
import csv
import math
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2 as cv
import numpy as np

from adas_pipeline import RoadAuditor, resize
from road_focus import ask_view, road_view
from vlm_tool import VLMBlocked

MODES = ["full", "crop", "mask"]
PHOTOS = 4
V4_ROAD = {"painted", "inferred", "no_lanes"}
OUT = Path("probe_crop")
ROOTS = [p for base in Path(".").glob("data*") if base.is_dir() and base.name != "data_agent"
         for p in base.iterdir() if p.is_dir()]


def find(image_id):
    for d in ROOTS:
        p = d / f"{image_id}.jpg"
        if p.exists():
            return p
    p = Path("data_agent") / f"{image_id}.jpg"
    return p if p.exists() else None


def load_dev2():
    def lab(path):
        return {r["location_id"]: r["label"] for r in csv.DictReader(open(path))}
    a, b = lab("location_labels_test.csv"), lab("location_labels_test_relabel.csv")
    m = lambda x: "yes" if x in ("readable", "at_risk", "not_readable") else "no" if x == "unmarked" else None
    truth = {k: m(a[k]) for k in a if k in b and m(a[k]) and m(a[k]) == m(b[k])}
    ids = defaultdict(list)
    for r in csv.DictReader(open("test_agent_frames.csv")):
        if r["segment_id"] in truth and r["origin"] == "original":
            ids[r["segment_id"]].append(r["image_id"])
    rng = random.Random(1)
    photos = {}
    for loc in sorted(truth):
        lst = ids.get(loc, [])
        rng.shuffle(lst)
        photos[loc] = [(i, p) for i, p in ((i, find(i)) for i in lst[:PHOTOS]) if p]
    return truth, photos


def load_final():
    truth = {r["location_id"]: r["label"] for r in csv.DictReader(open("finalv5_labels.csv"))
             if r["label"] in ("yes", "no")}
    fr = defaultdict(list)
    for r in csv.DictReader(open("finalv5_agent_frames.csv")):
        if r["segment_id"] in truth and r["status"] in V4_ROAD:
            fr[r["segment_id"]].append(r)
    photos = {}
    for loc in sorted(truth):
        rows = sorted(fr.get(loc, []), key=lambda r: r["origin"] != "original")
        got = []
        for r in rows:
            p = find(r["image_id"])
            if p:
                got.append((r["image_id"], p))
            if len(got) >= PHOTOS:
                break
        photos[loc] = got
    return truth, photos


def vote(answers):
    yes, no = answers.count("YES"), answers.count("NO")
    return "yes" if yes > no else "no" if no > yes else "unclear"


def kappa(ps):
    n = len(ps)
    if not n:
        return float("nan")
    po = sum(a == b for a, b in ps) / n
    ca, cb = Counter(a for a, _ in ps), Counter(b for _, b in ps)
    pe = sum(ca[c] * cb[c] for c in set(ca) | set(cb)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def summary(pairs):
    dec = [(t, p) for t, p in pairs if p in ("yes", "no")]
    rec = {c: (sum(t == p for t, p in dec if t == c), sum(1 for t, _ in dec if t == c)) for c in ("yes", "no")}
    parts = [k / n for k, n in rec.values() if n]
    bal = 100 * sum(parts) / len(parts) if parts else float("nan")
    agree = sum(t == p for t, p in dec)
    k = kappa(dec)
    return (f"agreement {agree}/{len(dec)} = {100 * agree / max(1, len(dec)):.0f}%  "
            f"balanced {bal:.0f}%  kappa {'n/a' if math.isnan(k) else f'{k:.2f}'}  "
            f"lines found {rec['yes'][0]}/{rec['yes'][1]}  no lines found {rec['no'][0]}/{rec['no'][1]}  "
            f"unclear {sum(p == 'unclear' for _, p in pairs)}"), bal


def sheet(loc, truth, photos, answers, auditor, tag):
    rows = []
    for (image_id, path), ans in zip(photos, answers):
        img = cv.imread(str(path))
        if img is None:
            continue
        img = resize(img)
        tiles = []
        for mode, a in zip(MODES, ans):
            v = img if mode == "full" else road_view(img, auditor, mode)
            if v is None:
                v = np.full((300, 300, 3), 40, np.uint8)
            v = cv.resize(v, (int(v.shape[1] * 300 / v.shape[0]), 300))
            v = cv.copyMakeBorder(v, 34, 0, 0, 8, cv.BORDER_CONSTANT, value=(255, 255, 255))
            cv.putText(v, f"{mode}: {a}", (6, 24), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            tiles.append(v)
        rows.append(cv.hconcat(tiles))
    if not rows:
        return
    width = max(r.shape[1] for r in rows)
    rows = [cv.copyMakeBorder(r, 0, 8, 0, width - r.shape[1], cv.BORDER_CONSTANT, value=(255, 255, 255)) for r in rows]
    cv.imwrite(str(OUT / f"{tag}_{loc}_truth-{truth}.jpg"), cv.vconcat(rows))


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    auditor = RoadAuditor()
    pooled = {m: [] for m in MODES}
    for tag, title, loader in [("dev2", "Former test suburbs (consistent labels)", load_dev2),
                               ("final", "Final set, 48 locations (reused as development)", load_final)]:
        truth, photos = loader()
        per = {m: [] for m in MODES}
        no_view = 0
        no_photos = 0
        for i, loc in enumerate(sorted(truth), 1):
            if not photos[loc]:
                no_photos += 1
            ans = [[ask_view(p, img_id, m, auditor) for m in MODES] for img_id, p in photos[loc]]
            no_view += sum(a[1] == "NO_ROAD" for a in ans)
            for j, m in enumerate(MODES):
                per[m].append((truth[loc], vote([a[j] for a in ans])))
            if truth[loc] == "no":
                sheet(loc, "no", photos[loc], ans, auditor, tag)
            if i % 10 == 0:
                print(f"  ...{tag} {i}/{len(truth)}", file=sys.stderr)
        print(f"\n== {title}: {len(truth)} locations, "
              f"{Counter(truth.values())['no']} labelled no; locations with no road photos: {no_photos}; "
              f"photos with no road view: {no_view} ==")
        for m in MODES:
            print(f"{m:5s} {summary(per[m])[0]}")
            pooled[m] += per[m]
    print("\n== POOLED (development) ==")
    base = summary(pooled["full"])[1]
    for m in MODES:
        text, bal = summary(pooled[m])
        print(f"{m:5s} {text}" + ("" if m == "full" else f"   balanced vs full: {bal - base:+.0f}"))
    print(f"\nRule: continue to the fresh test only if crop or mask is >= +5 balanced. "
          f"Side by side images for 'no' locations: {OUT}/")


if __name__ == "__main__":
    try:
        main()
    except VLMBlocked as e:
        sys.exit(f"STOPPED: AWS blocked Bedrock ({e}). Answers so far are cached; run again later.")
