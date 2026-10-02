"""Collect a version's mistakes on the DEV set into review/ as side-by-side images
(left = original, right = what the detector saw). Usage: python review_misses.py output_v3b"""
import csv
import shutil
import sys
from pathlib import Path

import cv2 as cv

OUTDIR = Path(sys.argv[1] if len(sys.argv) > 1 else "output_v3b")
REVIEW = Path("review")
if REVIEW.exists():
    shutil.rmtree(REVIEW)
REVIEW.mkdir()

preds = {r["image_id"]: r for r in csv.DictReader(open(OUTDIR / "results.csv"))}
groups = {
    "1_bad_called_ready": lambda h, p: h in ("degraded", "fail") and p == "ready",
    "2_road_called_not_road": lambda h, p: h in ("ready", "degraded", "fail") and p == "not_road",
    "3_good_called_no_lanes": lambda h, p: h in ("ready", "degraded") and p == "no_lanes",
    "4_none_called_lanes": lambda h, p: h == "none" and p in ("ready", "degraded", "fail"),
    "5_good_called_fail": lambda h, p: h == "ready" and p == "fail",
}
counts = {g: 0 for g in groups}

for r in csv.DictReader(open("dev_labels.csv")):
    p = preds.get(r["image_id"])
    if not p:
        continue
    for group, test in groups.items():
        if test(r["my_label"], p["status"]):
            orig = cv.imread(f"data/{r['area']}/{r['image_id']}.jpg")
            seen = cv.imread(str(OUTDIR / r["area"] / f"{r['image_id']}.jpg"))
            if orig is None:
                continue
            if seen is None:
                seen = orig.copy()
            h, w = seen.shape[:2]
            orig = cv.resize(orig, (w, h))
            cv.putText(orig, f"YOU: {r['my_label'].upper()}", (20, 50),
                       cv.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
            (REVIEW / group).mkdir(exist_ok=True)
            cv.imwrite(str(REVIEW / group / f"{r['area']}_{r['image_id']}.jpg"), cv.hconcat([orig, seen]))
            counts[group] += 1

for g, n in counts.items():
    print(f"{g}: {n} images")
print("Open review/ in Finder (Gallery view, Cmd+4).")
