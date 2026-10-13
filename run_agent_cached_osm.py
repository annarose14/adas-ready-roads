"""Run agent.py using only cached OSM answers (no live Overpass calls).
Used when the public Overpass servers are unavailable. Locations without a cached
answer are recorded as osm_error; the v5 verdict does not use OSM."""
import runpy
import sys

import agent_tools


def cached_only(lat, lon):
    hit = agent_tools._load_cache().get(f"{lat:.5f},{lon:.5f}")
    if hit and hit.get("v") == agent_tools.CACHE_VERSION:
        return hit
    return {"highway": "", "expected_marked": None, "junction": False,
            "source": "osm_error: skipped (Overpass servers unavailable)"}


agent_tools.road_context = cached_only
sys.argv = ["agent.py"] + sys.argv[1:]
runpy.run_path("agent.py", run_name="__main__")
