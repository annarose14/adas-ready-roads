"""ADAS-Ready Roads - fetch street-level images from Mapillary into <root>/<area>/.
Usage:
  python fetch_mapillary.py <area> <min_lon> <min_lat> <max_lon> <max_lat> [root] [max_images]
  e.g. python fetch_mapillary.py burwood 151.098 -33.882 151.110 -33.872 data_test 200
Defaults: root=data, max_images=100. If Mapillary says the box has too much data, the box
is split into 4 smaller boxes automatically. Images are CC BY-SA 4.0 - creator saved for credit.
Requires env var MAPILLARY_TOKEN."""
import csv
import os
import sys
import time
from pathlib import Path

import requests

API_URL = "https://graph.mapillary.com/images"
FIELDS = "id,thumb_2048_url,computed_geometry,captured_at,compass_angle,sequence,creator"


def query_bbox(token, bbox, limit, depth=0):
    """Return image records in bbox; split into quadrants if the API says it's too much data."""
    params = {"access_token": token, "fields": FIELDS,
              "bbox": ",".join(f"{v:.6f}" for v in bbox), "limit": limit}
    resp = requests.get(API_URL, params=params, timeout=90)
    if resp.status_code == 200:
        return resp.json().get("data", [])
    if "reduce the amount of data" in resp.text and depth < 3:
        x0, y0, x1, y1 = bbox
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        quads = [(x0, y0, xm, ym), (xm, y0, x1, ym), (x0, ym, xm, y1), (xm, ym, x1, y1)]
        print(f"  box too dense - splitting into 4 (level {depth + 1})")
        out, per = [], max(25, limit // 4)
        for q in quads:
            out += query_bbox(token, q, per, depth + 1)
            time.sleep(0.5)
        return out
    sys.exit(f"API error {resp.status_code}: {resp.text[:300]}")


def main():
    if len(sys.argv) < 6:
        sys.exit(__doc__)
    token = os.environ.get("MAPILLARY_TOKEN")
    if not token:
        sys.exit("Set MAPILLARY_TOKEN first: export MAPILLARY_TOKEN=\"$(cat ~/.mapillary_token | tr -d '[:space:]')\"")
    area = sys.argv[1]
    bbox = tuple(float(v) for v in sys.argv[2:6])
    root = Path(sys.argv[6]) if len(sys.argv) > 6 else Path("data")
    max_images = int(sys.argv[7]) if len(sys.argv) > 7 else 100

    out_dir = root / area
    out_dir.mkdir(parents=True, exist_ok=True)
    images = query_bbox(token, bbox, max_images)
    seen, unique = set(), []
    for img in images:
        if img["id"] not in seen:
            seen.add(img["id"])
            unique.append(img)
    unique = unique[:max_images]
    print(f"Found {len(unique)} images for {area}")

    with open(out_dir / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "file", "lon", "lat", "captured_at",
                         "compass_angle", "sequence", "creator"])
        for i, img in enumerate(unique, 1):
            url = img.get("thumb_2048_url")
            coords = (img.get("computed_geometry") or {}).get("coordinates", [None, None])
            if not url:
                continue
            file_name = f"{img['id']}.jpg"
            path = out_dir / file_name
            if not path.exists():
                r = requests.get(url, timeout=60)
                if r.status_code != 200:
                    continue
                path.write_bytes(r.content)
                time.sleep(0.2)
            creator = (img.get("creator") or {}).get("username", "")
            writer.writerow([img["id"], file_name, coords[0], coords[1], img.get("captured_at"),
                             img.get("compass_angle"), img.get("sequence"), creator])
            if i % 25 == 0:
                print(f"  {i}/{len(unique)} saved")
    print(f"Done: {out_dir}/")


if __name__ == "__main__":
    main()
