"""Contact sheets of the frames used for each location.
Usage:
  python view_segments.py                          HIGH-priority locations (annotated)
  python view_segments.py L0003 L0011              specific locations (annotated)
  python view_segments.py conflict                 all 'map conflict' locations (annotated)
  python view_segments.py --prefix final_ --blind all --out blind_final
        BLIND mode for labelling: raw photos only - no verdict, no reason, no overlays.
Images are searched in every data*/<area>/ folder and data_agent/."""
import argparse
import csv
import math
import shutil
from collections import defaultdict
from pathlib import Path

import cv2 as cv
import numpy as np

from adas_pipeline import RoadAuditor, _result, quality_check, resize

TILE_W, TILE_H, COLS = 640, 400, 3

ap = argparse.ArgumentParser()
ap.add_argument("ids", nargs="*")
ap.add_argument("--prefix", default="")
ap.add_argument("--blind", action="store_true")
ap.add_argument("--out", default="review_segments")
args = ap.parse_args()

OUT = Path(args.out)
segs = {r["segment_id"]: r for r in csv.DictReader(open(f"{args.prefix}segments_agent.csv"))}
frames = defaultdict(list)
for r in csv.DictReader(open(f"{args.prefix}agent_frames.csv")):
    frames[r["segment_id"]].append(r)

if not args.ids:
    chosen = [s for s, r in segs.items() if r["priority"] == "high"]
elif args.ids == ["conflict"]:
    chosen = [s for s, r in segs.items() if r["reason"].startswith("map conflict")]
elif args.ids == ["all"]:
    chosen = list(segs)
else:
    chosen = args.ids

roots = [p for base in Path(".").glob("data*") if base.is_dir() and base.name != "data_agent"
         for p in base.iterdir() if p.is_dir()]


def find_image(image_id):
    for d in roots:
        p = d / f"{image_id}.jpg"
        if p.exists():
            return p
    p = Path("data_agent") / f"{image_id}.jpg"
    return p if p.exists() else None


if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir()
auditor = None if args.blind else RoadAuditor()
made = 0
for sid in chosen:
    seg, fr = segs.get(sid), frames.get(sid, [])
    if not seg or not fr:
        continue
    tiles = []
    for f in fr:
        path = find_image(f["image_id"])
        if path is None:
            continue
        img = resize(cv.imread(str(path)))
        if args.blind:
            tile = cv.resize(img, (TILE_W, TILE_H))
            tag = f["image_id"]
        else:
            q = quality_check(img)
            r = auditor.analyse(img) if q == "ok" else _result("unusable", reason=q)
            tile = cv.resize(auditor.annotate(img, r), (TILE_W, TILE_H))
            tag = f"{f['origin']}  {f['image_id']}"
        cv.rectangle(tile, (0, TILE_H - 34), (TILE_W, TILE_H), (0, 0, 0), -1)
        cv.putText(tile, tag, (8, TILE_H - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        tiles.append(tile)
    if not tiles:
        continue
    rows = math.ceil(len(tiles) / COLS)
    blank = np.zeros((TILE_H, TILE_W, 3), np.uint8)
    tiles += [blank] * (rows * COLS - len(tiles))
    sheet = cv.vconcat([cv.hconcat(tiles[i * COLS:(i + 1) * COLS]) for i in range(rows)])
    header = np.full((60, sheet.shape[1], 3), 255, np.uint8)
    if args.blind:
        title = f"{sid}  {seg['road_name'] or 'unnamed road'}   ({len(tiles)} photos)"
    else:
        title = (f"{sid}  {seg['road_name'] or '?'} ({seg['highway'] or '?'})  ->  {seg['final_verdict'].upper()}"
                 f"  | painted {seg['n_lines_found']}/{seg['n_road_frames']}  | {seg['reason']}")
    cv.putText(header, title[:140], (10, 40), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    name = f"{sid}_{(seg['road_name'] or 'unknown').replace(' ', '_')}.jpg"
    cv.imwrite(str(OUT / name), cv.vconcat([header, sheet]))
    made += 1
print(f"{made} contact sheets written to {OUT}/ ({'BLIND - no verdicts' if args.blind else 'annotated'})")
