"""Contact sheets of all frames the agent used for chosen segments (default: HIGH priority).
Each tile = what the frozen pipeline saw (paint magenta, lanes green, vehicles blue).
Usage: python view_segments.py                -> all HIGH priority segments
       python view_segments.py S0006 S0007    -> specific segments
       python view_segments.py conflict       -> all 'map conflict' review segments"""
import csv
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2 as cv
import numpy as np

from adas_pipeline import RoadAuditor, _result, quality_check, resize

OUT = Path("review_segments")
TILE_W, TILE_H, COLS = 640, 400, 3

segs = {r["segment_id"]: r for r in csv.DictReader(open("segments_agent.csv"))}
frames = defaultdict(list)
for r in csv.DictReader(open("agent_frames.csv")):
    frames[r["segment_id"]].append(r)

args = sys.argv[1:]
if not args:
    chosen = [s for s, r in segs.items() if r["priority"] == "high"]
elif args == ["conflict"]:
    chosen = [s for s, r in segs.items() if r["reason"].startswith("map conflict")]
else:
    chosen = args

areas = [p.name for p in Path("data").iterdir() if p.is_dir()]


def find_image(image_id):
    for a in areas:
        p = Path("data") / a / f"{image_id}.jpg"
        if p.exists():
            return p
    p = Path("data_agent") / f"{image_id}.jpg"
    return p if p.exists() else None


if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir()
auditor = RoadAuditor()
for sid in chosen:
    seg, fr = segs.get(sid), frames.get(sid, [])
    if not seg or not fr:
        print(f"{sid}: not found")
        continue
    tiles = []
    for f in fr:
        path = find_image(f["image_id"])
        if path is None:
            continue
        img = resize(cv.imread(str(path)))
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
    title = (f"{sid}  {seg['road_name'] or '?'} ({seg['highway'] or '?'})  ->  {seg['final_verdict'].upper()}"
             f"  | lines found {seg['n_lines_found']}/{seg['n_road_frames']}  | {seg['reason']}")
    cv.putText(header, title[:140], (10, 40), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    name = f"{sid}_{(seg['road_name'] or 'unknown').replace(' ', '_')}.jpg"
    cv.imwrite(str(OUT / name), cv.vconcat([header, sheet]))
    print(f"{sid}: {seg['road_name']} - {len(fr)} frames -> {OUT / name}")
print("Open review_segments/ in Finder.")
