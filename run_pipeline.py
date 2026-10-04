"""Run the pipeline over <data_root>/<area>/*.jpg.
Usage: python run_pipeline.py [data_root] [out_dir]      (defaults: data output_final)
Writes <out_dir>/<area>/*.jpg (annotated) and <out_dir>/results.csv,
merging Mapillary metadata (GPS, date, heading, sequence, photographer credit)."""
import csv
import sys
from pathlib import Path

import cv2 as cv

from adas_pipeline import PIPELINE_VERSION, RoadAuditor, _result, quality_check, resize, to_record

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("output_final")
META_COLS = ["lon", "lat", "captured_at", "compass_angle", "sequence", "creator"]
RESULT_COLS = ["quality", "status", "reason", "lanes_detected", "lanes_painted", "lane_conf",
               "road_frac", "drivable_frac", "surface_frac", "ego_top"]


def load_metadata():
    meta = {}
    for f in DATA.glob("*/metadata.csv"):
        for r in csv.DictReader(open(f)):
            meta[r["image_id"]] = r
    return meta


def main():
    OUT.mkdir(exist_ok=True)
    meta = load_metadata()
    auditor = RoadAuditor()
    rows, counts = [], {}
    paths = sorted(DATA.glob("*/*.jpg"))
    for i, path in enumerate(paths, 1):
        area = path.parent.name
        img = cv.imread(str(path))
        if img is None:
            continue
        img = resize(img)
        q = quality_check(img)
        if q != "ok":
            r = _result("unusable")
        else:
            r = auditor.analyse(img)
            (OUT / area).mkdir(exist_ok=True)
            cv.imwrite(str(OUT / area / path.name), auditor.annotate(img, r))
        rec = to_record(r)
        rec["quality"] = q
        m = meta.get(path.stem, {})
        rows.append([area, path.stem] + [m.get(c, "") for c in META_COLS]
                    + [rec.get(c, "") for c in RESULT_COLS] + [PIPELINE_VERSION])
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        if i % 50 == 0:
            print(f"  ...{i}/{len(paths)}")

    with open(OUT / "results.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["area", "image_id"] + META_COLS + RESULT_COLS + ["pipeline_version"])
        wr.writerows(rows)

    print(f"OpenCV {cv.__version__} - {PIPELINE_VERSION} ({auditor.engine} engine) "
          f"processed {len(rows)} images from {DATA}/ -> {OUT}/")
    for k, v in sorted(counts.items()):
        print(f"  {k:10s} {v}")


if __name__ == "__main__":
    main()
