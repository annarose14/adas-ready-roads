"""
ADAS-Ready Roads - Step 1: fetch street-level images from Mapillary.
Set MAPILLARY_TOKEN env var first. Images are CC BY-SA 4.0 - creator saved for credit.
"""
import csv
import os
import sys
import time
from pathlib import Path

import requests

TOKEN = os.environ.get("MAPILLARY_TOKEN")
if not TOKEN:
    sys.exit("Set the MAPILLARY_TOKEN environment variable first.")

# Bounding box: min_lon, min_lat, max_lon, max_lat (keep it ~1 km x 1 km)

AREA_NAME = "princes_hwy"
BBOX = (151.163, -33.920, 151.170, -33.913)

MAX_IMAGES = 100
OUT_DIR = Path("data") / AREA_NAME
API_URL = "https://graph.mapillary.com/images"
FIELDS = "id,thumb_2048_url,computed_geometry,captured_at,compass_angle,sequence,creator"


def fetch_image_list():
    params = {
        "access_token": TOKEN,
        "fields": FIELDS,
        "bbox": ",".join(str(v) for v in BBOX),
        "limit": MAX_IMAGES,
    }
    resp = requests.get(API_URL, params=params, timeout=60)
    if resp.status_code != 200:
        sys.exit(f"API error {resp.status_code}: {resp.text[:300]}")
    return resp.json().get("data", [])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    images = fetch_image_list()
    print(f"Found {len(images)} images in {AREA_NAME}")

    with open(OUT_DIR / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "file", "lon", "lat", "captured_at",
                         "compass_angle", "sequence", "creator"])
        for i, img in enumerate(images, 1):
            url = img.get("thumb_2048_url")
            coords = (img.get("computed_geometry") or {}).get("coordinates", [None, None])
            if not url:
                continue
            file_name = f"{img['id']}.jpg"
            file_path = OUT_DIR / file_name
            if not file_path.exists():
                r = requests.get(url, timeout=60)
                if r.status_code != 200:
                    print(f"  skip {img['id']} (download failed)")
                    continue
                file_path.write_bytes(r.content)
                time.sleep(0.2)
            creator = (img.get("creator") or {}).get("username", "")
            writer.writerow([img["id"], file_name, coords[0], coords[1],
                             img.get("captured_at"), img.get("compass_angle"),
                             img.get("sequence"), creator])
            print(f"  [{i}/{len(images)}] saved {file_name}")

    print(f"\nDone. Images + metadata in {OUT_DIR}/")


if __name__ == "__main__":
    main()
