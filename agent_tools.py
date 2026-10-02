"""ADAS-Ready Roads - agent tools.
  A. fetch_frames(): get more Mapillary images near a point (prefer same sequence),
     analyse each with the FROZEN pipeline (adas_pipeline.py).
  B. road_context(): OpenStreetMap road type near a point -> should it have lane markings?
Data credits: Mapillary images CC BY-SA 4.0 (creator saved); OSM data (c) OpenStreetMap
contributors, ODbL."""
import csv
import json
import os
import time
from pathlib import Path

import cv2 as cv
import requests

from adas_pipeline import RoadAuditor, _result, quality_check, resize, to_record

MAPILLARY_URL = "https://graph.mapillary.com/images"
FIELDS = "id,thumb_2048_url,computed_geometry,captured_at,compass_angle,sequence,creator"
FETCH_DIR = Path("data_agent")
FETCH_META = FETCH_DIR / "metadata.csv"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OSM_CACHE = Path("osm_cache.json")
HEADERS = {"User-Agent": "ADAS-Ready-Roads student project (OpenCV AI Competition 2026)"}

# Road classes that normally carry lane markings in Australian cities
MARKED_CLASSES = {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
                  "secondary", "secondary_link", "tertiary", "tertiary_link"}
CLASS_RANK = ["motorway", "trunk", "primary", "secondary", "tertiary", "motorway_link",
              "trunk_link", "primary_link", "secondary_link", "tertiary_link",
              "unclassified", "residential", "living_street", "service", "track"]

_auditor = None


def get_auditor():
    global _auditor
    if _auditor is None:
        _auditor = RoadAuditor()
    return _auditor


# ---------------------------------------------------------------- Tool A
def fetch_frames(lat, lon, sequence=None, exclude=(), radius_deg=0.0005, limit=10):
    """Fetch up to `limit` new frames within ~radius of (lat, lon), analyse them.
    Returns (frames, status). Each frame is a dict with the same fields as results.csv."""
    token = os.environ.get("MAPILLARY_TOKEN")
    if not token:
        return [], "no_token"
    bbox = f"{lon - radius_deg},{lat - radius_deg},{lon + radius_deg},{lat + radius_deg}"
    try:
        resp = requests.get(MAPILLARY_URL, params={"access_token": token, "fields": FIELDS,
                                                   "bbox": bbox, "limit": 100}, timeout=60)
    except requests.RequestException as e:
        return [], f"network_error: {e}"
    if resp.status_code != 200:
        return [], f"api_error_{resp.status_code}"
    data = resp.json().get("data", [])
    same = [d for d in data if sequence and d.get("sequence") == sequence]
    other = [d for d in data if d not in same]
    candidates = [d for d in same + other if d["id"] not in set(exclude)]

    FETCH_DIR.mkdir(exist_ok=True)
    new_meta = not FETCH_META.exists()
    frames = []
    with open(FETCH_META, "a", newline="") as mf:
        mw = csv.writer(mf)
        if new_meta:
            mw.writerow(["image_id", "lon", "lat", "captured_at", "sequence", "creator"])
        for d in candidates:
            if len(frames) >= limit:
                break
            url = d.get("thumb_2048_url")
            coords = (d.get("computed_geometry") or {}).get("coordinates")
            if not url or not coords:
                continue
            path = FETCH_DIR / f"{d['id']}.jpg"
            if not path.exists():
                try:
                    r = requests.get(url, timeout=60)
                except requests.RequestException:
                    continue
                if r.status_code != 200:
                    continue
                path.write_bytes(r.content)
                time.sleep(0.2)
            img = cv.imread(str(path))
            if img is None:
                continue
            img = resize(img)
            q = quality_check(img)
            res = get_auditor().analyse(img) if q == "ok" else _result("unusable")
            rec = {k: ("" if v is None else str(v)) for k, v in to_record(res).items()}
            creator = (d.get("creator") or {}).get("username", "")
            rec.update(image_id=d["id"], lon=str(coords[0]), lat=str(coords[1]),
                       captured_at=str(d.get("captured_at", "")), sequence=d.get("sequence", ""),
                       creator=creator, quality=q, area="agent_fetch")
            mw.writerow([d["id"], coords[0], coords[1], d.get("captured_at", ""),
                         d.get("sequence", ""), creator])
            frames.append(rec)
    return frames, "ok"


# ---------------------------------------------------------------- Tool B
def _load_cache():
    return json.loads(OSM_CACHE.read_text()) if OSM_CACHE.exists() else {}


def road_context(lat, lon, radius_m=15):
    """OpenStreetMap ways within radius_m. Returns the most important road class and
    whether lane markings are expected. Cached on disk."""
    cache = _load_cache()
    key = f"{lat:.5f},{lon:.5f}"
    if key in cache:
        return cache[key]
    query = f'[out:json][timeout:25];way(around:{radius_m},{lat},{lon})[highway];out tags;'
    try:
        resp = requests.post(OVERPASS_URL, data={"data": query}, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        ways = [e.get("tags", {}) for e in resp.json().get("elements", [])]
    except (requests.RequestException, ValueError) as e:
        return {"highway": "", "expected_marked": None, "source": f"osm_error: {e}"}
    time.sleep(1.0)   # be polite to the public Overpass server

    roads = [t for t in ways if t.get("highway") in CLASS_RANK]
    if not roads:
        ctx = {"highway": "", "name": "", "lanes": "", "lane_markings": "",
               "expected_marked": None, "n_ways": len(ways), "source": "osm_no_road"}
    else:
        best = min(roads, key=lambda t: CLASS_RANK.index(t["highway"]))
        cls = best.get("highway", "")
        lanes = best.get("lanes", "")
        marking_tag = best.get("lane_markings", "")
        expected = cls in MARKED_CLASSES
        if lanes.isdigit() and int(lanes) >= 2:
            expected = True
        if marking_tag == "yes":
            expected = True
        if marking_tag == "no":
            expected = False
        ctx = {"highway": cls, "name": best.get("name", ""), "lanes": lanes,
               "lane_markings": marking_tag, "expected_marked": expected,
               "n_ways": len(ways), "source": "osm"}
    cache[key] = ctx
    OSM_CACHE.write_text(json.dumps(cache, indent=1))
    return ctx
