"""Re-run the frozen pipeline with the CLASSIC DNN engine on all images and compare
verdicts with output_final/results.csv (made with OpenCV 5's default/new engine)."""
import csv
from collections import Counter
from pathlib import Path

import cv2 as cv

from adas_pipeline import RoadAuditor, quality_check, resize

old = {r["image_id"]: r for r in csv.DictReader(open("output_final/results.csv"))}
auditor = RoadAuditor(engine="classic")
same, diff, changes = 0, [], Counter()
for path in sorted(Path("data").glob("*/*.jpg")):
    o = old.get(path.stem)
    if not o or o["status"] == "unusable":
        continue
    img = cv.imread(str(path))
    if img is None:
        continue
    img = resize(img)
    if quality_check(img) != "ok":
        continue
    r = auditor.analyse(img)
    if r["status"] == o["status"]:
        same += 1
    else:
        diff.append((path.stem, o["status"], r["status"], o["paint_contrast"], r["paint_contrast"]))
        changes[(o["status"], r["status"])] += 1

total = same + len(diff)
print(f"Engine: classic vs output_final (new engine). Usable images compared: {total}")
print(f"  Same verdict: {same}/{total} = {100 * same / total:.1f}%")
print(f"  Different:    {len(diff)}")
for (a, b), n in changes.most_common():
    print(f"    {a:9s} -> {b:9s} {n}")
for d in diff[:10]:
    print(f"    {d[0]}: {d[1]} -> {d[2]}  (contrast {d[3]} -> {d[4]})")
