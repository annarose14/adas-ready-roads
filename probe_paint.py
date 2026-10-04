"""Probe: does LANE NETWORK + PAINT VERIFICATION separate marked vs unmarked roads?
A) DEV photos by your photo labels (ready / degraded-fail / none)
B) Former TEST suburbs (now development data) by your LOCATION labels (readable / at_risk / unmarked)"""
import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

import cv2 as cv
import numpy as np

from adas_pipeline import RoadAuditor, quality_check, resize
from lane_features import LaneNet, lane_evidence

auditor, lanenet = RoadAuditor(), LaneNet()
roots = [p for base in ("data", "data_test") if Path(base).exists()
         for p in Path(base).iterdir() if p.is_dir()]


def find(image_id):
    for d in roots:
        p = d / f"{image_id}.jpg"
        if p.exists():
            return p
    p = Path("data_agent") / f"{image_id}.jpg"
    return p if p.exists() else None


def features(path):
    img = cv.imread(str(path))
    if img is None:
        return None
    img = resize(img)
    if quality_check(img) != "ok":
        return None
    labels = auditor.segment(img)
    lanes, conf = lanenet.detect(img)
    ev = lane_evidence(img, labels, lanes)
    return {"detected": sum(e["detected"] for e in ev),
            "painted": sum(e["is_painted"] for e in ev), "conf": conf}


def table(title, groups):
    print(f"\n{title}")
    print(f"{'group':14s}{'n':>5s}{'lanes detected':>16s}{'lanes PAINTED':>15s}{'% frames >=1 painted':>23s}")
    for g, fs in groups.items():
        if fs:
            print(f"{g:14s}{len(fs):>5d}{np.mean([f['detected'] for f in fs]):>16.2f}"
                  f"{np.mean([f['painted'] for f in fs]):>15.2f}"
                  f"{100 * np.mean([f['painted'] >= 1 for f in fs]):>22.0f}%")


# A) DEV photos
dev = defaultdict(list)
for r in csv.DictReader(open("dev_labels.csv")):
    g = {"ready": "ready", "degraded": "degraded/fail", "fail": "degraded/fail", "none": "none"}.get(r["my_label"])
    p = Path("data") / r["area"] / f"{r['image_id']}.jpg"
    if g and p.exists():
        f = features(p)
        if f:
            dev[g].append(f)
table("A) DEV photos (your photo labels)", dev)

# B) former TEST locations, frame-level and location-level
loc_label = {r["location_id"]: r["label"] for r in csv.DictReader(open("location_labels_test.csv"))}
frames_by_loc = defaultdict(list)
for r in csv.DictReader(open("test_agent_frames.csv")):
    if r["segment_id"] in loc_label:
        frames_by_loc[r["segment_id"]].append(r["image_id"])

frame_groups, loc_rate = defaultdict(list), defaultdict(list)
for loc, ids in frames_by_loc.items():
    feats = [f for f in (features(p) for p in (find(i) for i in ids) if p) if f]
    if not feats:
        continue
    frame_groups[loc_label[loc]] += feats
    loc_rate[loc_label[loc]].append(np.mean([f["painted"] >= 1 for f in feats]))
table("B) Former TEST frames, grouped by your LOCATION label", frame_groups)

print("\nB) Location level: share of a location's frames with >=1 painted ego lane")
for g, v in loc_rate.items():
    print(f"  {g:12s} locations={len(v):>3d}   mean={st.mean(v):.2f}   "
          f"values: " + " ".join(f"{x:.2f}" for x in sorted(v)))
