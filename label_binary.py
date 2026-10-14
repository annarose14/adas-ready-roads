"""Blind labelling for the FINAL test: "Are painted lane lines visible in most of these photos?"
Usage: python label_binary.py [sheets_dir] [out_csv] [--shuffle]
       defaults: blind_final labels_final.csv
Rule: yes = any painted lane line on the road the car drives on (centre line, dashes,
      double lines, edge lines) in most road photos. Crossings, parking bays, kerbs alone = no.
Keys: y = yes   n = no lines   c = can't tell   z = zoom in/out   b = back   q = quit
Opens full screen at full resolution (macOS Retina shows normal windows at half size)."""
import csv
import random
import sys
from pathlib import Path

import cv2 as cv

args = [a for a in sys.argv[1:] if not a.startswith("--")]
SHEETS = Path(args[0]) if len(args) > 0 else Path("blind_final")
OUT = Path(args[1]) if len(args) > 1 else Path("labels_final.csv")
KEYS = {ord("y"): "yes", ord("n"): "no", ord("c"): "cant_tell"}
WIN = "labelling"

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


def render(img, zoom, text):
    if zoom:
        # zoom into the left half of the sheet at double size: roads sit low in each photo
        h, w = img.shape[:2]
        img = cv.resize(img[:, : w // 2], None, fx=2, fy=2, interpolation=cv.INTER_LINEAR)
    img = img.copy()
    bar_h = 56
    cv.rectangle(img, (0, img.shape[0] - bar_h), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
    cv.putText(img, text, (14, img.shape[0] - 18), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return img


cv.namedWindow(WIN, cv.WINDOW_NORMAL)
cv.setWindowProperty(WIN, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

i = next((n for n, p in enumerate(sheets) if p.stem.split("_")[0] not in labels), len(sheets))
zoom = False
while i < len(sheets):
    path = sheets[i]
    loc = path.stem.split("_")[0]
    img = cv.imread(str(path))
    text = (f"{i + 1}/{len(sheets)}  Painted lane lines visible?  y=yes  n=no  c=can't tell  "
            f"z=zoom {'(left half)' if zoom else ''}  b=back  q=quit")
    cv.imshow(WIN, render(img, zoom, text))
    key = cv.waitKey(0) & 0xFF
    if key == ord("q"):
        break
    if key == ord("z"):
        zoom = not zoom
        continue
    if key == ord("b"):
        i = max(0, i - 1)
        labels.pop(sheets[i].stem.split("_")[0], None)
        save()
        zoom = False
        continue
    if key in KEYS:
        labels[loc] = KEYS[key]
        save()
        i += 1
        zoom = False
cv.destroyAllWindows()
save()
print(f"Labelled {len(labels)}/{len(sheets)} locations -> {OUT}")
