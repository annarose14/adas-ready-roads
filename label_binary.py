"""Blind labelling for the FINAL test: "Are painted lane lines visible in most of these photos?"
Usage: python label_binary.py [sheets_dir] [out_csv] [--shuffle]
       defaults: blind_final labels_final.csv
Keys: y = yes, lines visible   n = no lines / essentially invisible   c = can't tell
      b = back   q = quit (re-run to continue)"""
import csv
import random
import sys
from pathlib import Path

import cv2 as cv

args = [a for a in sys.argv[1:] if not a.startswith("--")]
SHEETS = Path(args[0]) if len(args) > 0 else Path("blind_final")
OUT = Path(args[1]) if len(args) > 1 else Path("labels_final.csv")
KEYS = {ord("y"): "yes", ord("n"): "no", ord("c"): "cant_tell"}
MAX_W, MAX_H = 1500, 900

sheets = sorted(SHEETS.glob("*.jpg"))
if "--shuffle" in sys.argv:
    random.Random(7).shuffle(sheets)
labels = {}
if OUT.exists():
    labels = {r["location_id"]: r["label"] for r in csv.DictReader(open(OUT))}


def save():
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["location_id", "label"])
        for k in sorted(labels):
            w.writerow([k, labels[k]])


i = next((n for n, p in enumerate(sheets) if p.stem.split("_")[0] not in labels), len(sheets))
while i < len(sheets):
    path = sheets[i]
    loc = path.stem.split("_")[0]
    img = cv.imread(str(path))
    h, w = img.shape[:2]
    s = min(MAX_W / w, MAX_H / h, 1.0)
    img = cv.resize(img, (int(w * s), int(h * s)))
    bar = f"{i + 1}/{len(sheets)}  Painted lane lines visible in most photos?  y=yes  n=no  c=can't tell  b=back  q=quit"
    cv.rectangle(img, (0, img.shape[0] - 32), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
    cv.putText(img, bar, (8, img.shape[0] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv.imshow("final test labelling", img)
    key = cv.waitKey(0) & 0xFF
    if key == ord("q"):
        break
    if key == ord("b"):
        i = max(0, i - 1)
        labels.pop(sheets[i].stem.split("_")[0], None)
        save()
        continue
    if key in KEYS:
        labels[loc] = KEYS[key]
        save()
        i += 1
cv.destroyAllWindows()
save()
print(f"Labelled {len(labels)}/{len(sheets)} locations -> {OUT}")
