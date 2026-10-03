"""ADAS-Ready Roads - agent tools.
  A. fetch_frames(): get more Mapillary images near a point (prefer same sequence),
     only from VEHICLE-speed sequences (speed estimated from the API's own GPS/timestamps),
     analyse each with the FROZEN pipeline (adas_pipeline.py).
  B. road_context(): OpenStreetMap road type near a point -> should it have lane markings?
     Also flags JUNCTIONS (2+ differently named roads nearby). Retries with back-off when
     the public Overpass server is busy; failed lookups are never cached.
Data credits: Mapillary images CC BY-SA 4.0 (creator saved); OSM data (c) OpenStreetMap
contributors, ODbL."""
import csv
import json
import math
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path

import cv2 as cv
import requests

from adas_pipeline import RoadAuditor, _result, quality_check, resize, to_record

MAPILLARY_URL = "https://graph.mapillary.com/images"
FIELDS = "id,thumb_2048_url,computed_geometry,captured_at,compass_angle,sequence,creator"
FETCH_DIR = Path("data_agent")
FETCH_META = FETCH_DIR / "metadata.csv"
MIN_VEHICLE_KMH = 10       # other people's sequences slower than this are not used
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OSM_CACHE = Path("osm_cache.json")
CACHE_VERSION = 2          # v2 adds junction detection; older cache entries are re-queried
OSM_RADII = (15, 40)       # try close first, then wider (GPS drift / offset centrelines)
OSM_RETRIES = 3
OSM_BACKOFF_S = (5, 15, 30)
HEADERS = {"User-Agent": "ADAS-Ready Roads student project (OpenCV AI Competition 2026)"}

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


def _dist_m(lat1, lon1, lat2, lon2):
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) *
         math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return 6371000 * 2 * math.asin(math.sqrt(a))


def _sequence_speeds(items):
    """Median km/h per sequence from the API items' own GPS + capture times."""
    by_seq = defaultdict(list)
    for d in items:
        c = (d.get("computed_geometry") or {}).get("coordinates")
        t = d.get("captured_at")
        if c and t:
            t = float(t)
            by_seq[d.get("sequence")].append((t / 1000.0 if t > 1e11 else t, c[1], c[0]))
    speeds = {}
    for seq, pts in by_seq.items():
        pts.sort()
        vals = []
        for (t1, la1, lo1), (t2, la2, lo2) in zip(pts, pts[1:]):
            dt = t2 - t1
            if 0.5 <= dt <= 60:
                vals.append(_dist_m(la1, lo1, la2, lo2) / dt * 3.6)
        speeds[seq] = statistics.median(vals) if vals else None
    return speeds


# ---------------------------------------------------------------- Tool A
def fetch_frames(lat, lon, sequence=None, exclude=(), radius_deg=0.0005, limit=10):
    """Fetch up to `limit` NEW frames within ~radius of (lat, lon), analyse them.
    Same-sequence frames are always eligible; other sequences must be vehicle-speed.
    `exclude`: image ids already known (skipped). Returns (frames, status)."""
    token = os.environ.get("MAPILLARY_TOKEN")
    if not token:
        return [], "no_token (run: export MAPILLARY_TOKEN=...)"
    bbox = ",".join(f"{v:.6f}" for v in (lon - radius_deg, lat - radius_deg,
                                          lon + radius_deg, lat + radius_deg))
    try:
        resp = requests.get(MAPILLARY_URL, params={"access_token": token, "fields": FIELDS,
                                                   "bbox": bbox, "limit": 100}, timeout=60)
    except requests.RequestException as e:
        return [], f"network_error: {e}"
    if resp.status_code != 200:
        return [], f"api_error_{resp.status_code}: {resp.text[:300]}"
    data = resp.json().get("data", [])
    speeds = _sequence_speeds(data)
    excluded = set(exclude)
    same = [d for d in data if sequence and d.get("sequence") == sequence]
    other = [d for d in data if d not in same]
    other_vehicle = [d for d in other if (speeds.get(d.get("sequence")) or 0) >= MIN_VEHICLE_KMH]
    candidates = [d for d in same + other_vehicle if d["id"] not in excluded]

    FETCH_DIR.mkdir(exist_ok=True)
    new_meta = not FETCH_META.exists()
    frames = []
    with open(FETCH_META, "a", newline="") as mf:
        mw = csv.writer(mf)
        if new_meta:
            mw.writerow(["image_id", "lon", "lat", "captured_at", "compass_angle", "sequence", "creator"])
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
            compass = d.get("compass_angle", "")
            rec.update(image_id=d["id"], lon=str(coords[0]), lat=str(coords[1]),
                       captured_at=str(d.get("captured_at", "")),
                       compass_angle="" if compass is None else str(compass),
                       sequence=d.get("sequence", ""), creator=creator, quality=q,
                       area="agent_fetch")
            mw.writerow([d["id"], coords[0], coords[1], d.get("captured_at", ""), compass,
                         d.get("sequence", ""), creator])
            frames.append(rec)
    return frames, (f"ok ({len(data)} in area, {len(same)} same sequence, "
                    f"{len(other) - len(other_vehicle)} non-vehicle skipped, {len(candidates)} new)")


# ---------------------------------------------------------------- Tool B
def _load_cache():
    return json.loads(OSM_CACHE.read_text()) if OSM_CACHE.exists() else {}


def _query_osm(lat, lon, radius_m):
    """One Overpass query with retries/back-off. Raises on final failure."""
    query = f'[out:json][timeout:25];way(around:{radius_m},{lat},{lon})[highway];out tags;'
    last_err = None
    for attempt in range(OSM_RETRIES):
        try:
            resp = requests.post(OVERPASS_URL, data={"data": query}, headers=HEADERS, timeout=60)
            if resp.status_code in (429, 502, 503, 504):
                raise requests.HTTPError(f"{resp.status_code} server busy")
            resp.raise_for_status()
            time.sleep(1.0)   # be polite to the public Overpass server
            return [e.get("tags", {}) for e in resp.json().get("elements", [])]
        except (requests.RequestException, ValueError) as e:
            last_err = e
            time.sleep(OSM_BACKOFF_S[min(attempt, len(OSM_BACKOFF_S) - 1)])
    raise requests.RequestException(f"failed after {OSM_RETRIES} attempts: {last_err}")


def road_context(lat, lon):
    """OpenStreetMap ways near a point: most important road class, whether lane markings are
    expected (True / False / None = unknown), and whether this is a junction. Cached (successes only)."""
    cache = _load_cache()
    key = f"{lat:.5f},{lon:.5f}"
    hit = cache.get(key)
    if hit and hit.get("v") == CACHE_VERSION and hit.get("source") != "osm_no_road":
        return hit

    roads, radius_used, n_ways = [], None, 0
    try:
        for radius in OSM_RADII:
            ways = _query_osm(lat, lon, radius)
            n_ways = len(ways)
            roads = [t for t in ways if t.get("highway") in CLASS_RANK]
            if roads:
                radius_used = radius
                break
    except (requests.RequestException, ValueError) as e:
        return {"highway": "", "expected_marked": None, "junction": False,
                "source": f"osm_error: {e}"}

    if not roads:
        ctx = {"highway": "", "name": "", "lanes": "", "lane_markings": "",
               "expected_marked": None, "junction": False, "roads_nearby": "",
               "radius_m": OSM_RADII[-1], "n_ways": n_ways, "source": "osm_no_road",
               "v": CACHE_VERSION}
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
        names = sorted({t["name"] for t in roads if t.get("name")})
        ctx = {"highway": cls, "name": best.get("name", ""), "lanes": lanes,
               "lane_markings": marking_tag, "expected_marked": expected,
               "junction": len(names) >= 2, "roads_nearby": "; ".join(names),
               "radius_m": radius_used, "n_ways": n_ways, "source": "osm", "v": CACHE_VERSION}
    cache[key] = ctx
    OSM_CACHE.write_text(json.dumps(cache, indent=1))
    return ctx
