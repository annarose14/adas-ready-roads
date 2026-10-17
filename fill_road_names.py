"""Fill missing road names on the dashboard map, for display only.
Some test runs had no OpenStreetMap answers (the public servers were down), so their
locations show as "unnamed road". This asks OpenStreetMap's Nominatim service for the street
at each of those points and stores it as road_name, with road_name_source =
"osm lookup for display". Verdicts and evaluation files are not changed.
Nominatim allows at most one request per second; lookups are cached in road_names_cache.json.
Run after merge_geojson.py."""
import json
import time
from pathlib import Path

import requests

MAP = Path("dashboard/segments_agent.geojson")
CACHE = Path("road_names_cache.json")
URL = "https://nominatim.openstreetmap.org/reverse"
HEADERS = {"User-Agent": "ADAS-Ready Roads student project (OpenCV AI Competition 2026)"}

cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
data = json.loads(MAP.read_text())
todo = [f for f in data["features"] if not f["properties"].get("road_name")]
print(f"{len(todo)} locations without a road name")
filled = failed = 0
for i, f in enumerate(todo, 1):
    p = f["properties"]
    lat, lon = float(p["lat"]), float(p["lon"])
    key = f"{lat:.5f},{lon:.5f}"
    if key not in cache:
        try:
            r = requests.get(URL, params={"lat": lat, "lon": lon, "format": "jsonv2", "zoom": 17},
                             headers=HEADERS, timeout=15)
            r.raise_for_status()
            cache[key] = {"name": (r.json().get("address") or {}).get("road", "")}
            CACHE.write_text(json.dumps(cache, indent=1))
        except (requests.RequestException, ValueError) as e:
            failed += 1
            print(f"  lookup failed for {p['segment_id']}: {str(e)[:80]}")
            time.sleep(2)
            continue
        time.sleep(1.1)
    if cache[key].get("name"):
        p["road_name"] = cache[key]["name"]
        p["road_name_source"] = "osm lookup for display"
        filled += 1
    if i % 20 == 0:
        print(f"  ...{i}/{len(todo)}")
MAP.write_text(json.dumps(data))
print(f"Filled {filled}, still unnamed {len(todo) - filled}, failed lookups {failed}")
