"""Merge v5 agent runs into one map file for the dashboard. Location ids are prefixed per run
so they stay unique; each feature records which evaluation set it came from."""
import json
from pathlib import Path

RUNS = [("main_v5_", "D", "development suburbs"),
        ("dev2v5_", "T", "development (former test) suburbs"),
        ("finalv5_", "F", "final test suburbs (re-used)"),
        ("final2_", "C", "clean test suburbs")]
features = []
for prefix, tag, label in RUNS:
    path = Path(f"{prefix}segments_agent.geojson")
    if not path.exists():
        print(f"skip {path} (not found)")
        continue
    data = json.loads(path.read_text())
    for f in data["features"]:
        p = f["properties"]
        p["segment_id"] = f"{tag}-{p['segment_id']}"
        p["dataset"] = label
        features.append(f)
    print(f"{path}: {len(data['features'])} locations")
Path("dashboard").mkdir(exist_ok=True)
Path("dashboard/segments_agent.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": features}))
print(f"Wrote dashboard/segments_agent.geojson with {len(features)} locations")
