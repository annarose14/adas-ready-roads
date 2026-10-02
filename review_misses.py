"""Collect v2's mistakes on the DEV set into review/ as side-by-side images
(left = original, right = what v2 detected)."""
import csv
import shutil
from pathlib import Path

import cv2 as cv

REVIEW = Path("review")
if REVIEW.exists():
    shutil.rmtree(REVIEW)
REVIEW.mkdir()

preds = {r["image_id"]: r for r in csv.DictReader(open("output_v2/results.csv"))}
groups = {
    "1_bad_called_ready": lambda h, p: h in ("degraded", "fail") and p == "ready",
    "2_none_called_lanes": lambda h, p: h == "none" and p in ("ready", "degraded", "fail"),
    "3_good_called_fail": lambda h, p: h == "ready" and p == "fail",
}
counts = {g: 0 for g in groups}

for r in csv.DictReader(open("dev_labels.csv")):
    p = preds.get(r["image_id"])
    if not p:
        continue
    for group, test in groups.items():
        if test(r["my_label"], p["status"]):
            orig = cv.imread(f"data/{r['area']}/{r['image_id']}.jpg")
            seen = cv.imread(f"output_v2/{r['area']}/{r['image_id']}.jpg")
            if orig is None or seen is None:
                continue
            h, w = seen.shape[:2]
            orig = cv.resize(orig, (w, h))
            cv.putText(orig, f"YOU: {r['my_label'].upper()}", (20, 50),
                       cv.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
            side = cv.hconcat([orig, seen])
            (REVIEW / group).mkdir(exist_ok=True)
            cv.imwrite(str(REVIEW / group / f"{r['area']}_{r['image_id']}.jpg"), side)
            counts[group] += 1

for g, n in counts.items():
    print(f"{g}: {n} images")
print("Open the review/ folder in Finder (Gallery view, Cmd+4).")
