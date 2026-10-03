"""ADAS-Ready Roads - location agent.
Usage: python agent.py [results_csv] [output_prefix]
       defaults: output_final/results.csv  ""      e.g.  python agent.py output_test/results.csv test_
1. Vehicle-speed frames only (pedestrian/handheld sequences excluded).
2. Segments (~100 m per Mapillary sequence) -> OpenStreetMap context for each.
3. MERGE overlapping segments on the same named road (centres within MERGE_M) into one
   LOCATION, so one place is never counted several times.
4. Per location: PERCEIVE (frozen OpenCV 5 pipeline results) -> ACT (fetch more vehicle
   frames when evidence is thin; heading filter; skip known images) -> RE-PERCEIVE ->
   DECIDE (junctions -> human review) -> PRIORITISE / ESCALATE.
Every step is logged to <prefix>agent_trace.jsonl; every frame used to <prefix>agent_frames.csv.
Outputs: <prefix>segments_agent.csv/.geojson, <prefix>review_queue.csv,
<prefix>agent_trace.jsonl, <prefix>agent_frames.csv  (rows are locations; ids L0001...)"""
import csv
import json
import math
import statistics as st
import sys
from collections import Counter
from pathlib import Path

import segments as seglib
from agent_tools import fetch_frames, road_context

RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output_final/results.csv")
PREFIX = sys.argv[2] if len(sys.argv) > 2 else ""
seglib.RESULTS = RESULTS

MERGE_M = 30                     # merge segments on the same road with centres this close
FETCH_RADII = (0.0004, 0.0008)   # ~45 m, then ~90 m
FETCH_LIMIT = 8                  # new frames per fetch round
MAX_FETCH_CALLS = 60             # global budget (API politeness + runtime)
MIN_CONFIDENT_FRAMES = 3         # fewer road frames -> try to fetch more
HEADING_TOL = 35                 # degrees; keep frames along the road (either direction)
MAJOR_ROADS = {"motorway", "trunk", "primary", "secondary"}

OUT_CSV = Path(f"{PREFIX}segments_agent.csv")
OUT_GEO = Path(f"{PREFIX}segments_agent.geojson")
OUT_QUEUE = Path(f"{PREFIX}review_queue.csv")
OUT_TRACE = Path(f"{PREFIX}agent_trace.jsonl")
OUT_FRAMES = Path(f"{PREFIX}agent_frames.csv")
FRAME_COLS = ["segment_id", "image_id", "origin", "status", "reason", "paint_contrast",
              "lane_sides", "lat", "lon", "compass_angle", "captured_at", "sequence", "creator"]


def prep(f):
    f["_lat"], f["_lon"] = float(f["lat"]), float(f["lon"])
    f["_t"] = seglib.to_seconds(f.get("captured_at"))
    return f


def mean_heading(frames):
    angles = [float(f["compass_angle"]) for f in frames if f.get("compass_angle") not in ("", None)]
    if not angles:
        return None
    x = sum(math.cos(math.radians(a)) for a in angles)
    y = sum(math.sin(math.radians(a)) for a in angles)
    return math.degrees(math.atan2(y, x)) % 360


def aligned(frame, heading, sequences):
    if frame.get("sequence") in sequences or heading is None:
        return True
    a = frame.get("compass_angle")
    if a in ("", None):
        return False
    d = abs((float(a) - heading + 180) % 360 - 180)
    return d <= HEADING_TOL or d >= 180 - HEADING_TOL


def run_location(loc_id, main_seq, sequences, frames, known_ids, budget, ctx, coords, n_members):
    trace = []

    def log(step, **kw):
        trace.append({"step": step, **kw})

    for f in frames:
        f.setdefault("_origin", "original")
    s = seglib.summarise(loc_id, main_seq, frames)
    log("perceive", merged_segments=n_members, sequences=len(sequences), n_frames=s["n_frames"],
        n_road_frames=s["n_road_frames"], detect_rate=s["detect_rate"],
        median_contrast=s["median_contrast"], verdict=s["verdict"])

    expected = ctx.get("expected_marked")
    junction = bool(ctx.get("junction"))
    log("road_context", highway=ctx.get("highway", ""), name=ctx.get("name", ""),
        lanes=ctx.get("lanes", ""), lane_markings=ctx.get("lane_markings", ""),
        expected_marked=expected, junction=junction, roads_nearby=ctx.get("roads_nearby", ""),
        source=ctx.get("source", ""))

    rounds = 0
    if expected is False:
        log("skip_fetch", why="OSM says no lane markings expected; extra frames would not change the decision")
    while (s["n_road_frames"] < MIN_CONFIDENT_FRAMES and expected is not False
           and rounds < len(FETCH_RADII) and budget["left"] > 0):
        heading = mean_heading(frames)
        new, status = fetch_frames(s["lat"], s["lon"], sequence=main_seq, exclude=known_ids,
                                   radius_deg=FETCH_RADII[rounds], limit=FETCH_LIMIT)
        known_ids.update(f["image_id"] for f in new)
        kept = [prep(f) for f in new if aligned(f, heading, sequences)]
        for f in kept:
            f["_origin"] = "fetched"
        before = (s["verdict"], s["n_road_frames"], s["detect_rate"])
        frames = frames + kept
        s = seglib.summarise(loc_id, main_seq, frames)
        rounds += 1
        budget["left"] -= 1
        log("act_fetch", round=rounds, radius_m=int(FETCH_RADII[rounds - 1] * 111000),
            api=status, fetched=len(new), kept_after_heading_filter=len(kept),
            before={"verdict": before[0], "road_frames": before[1], "detect_rate": before[2]},
            after={"verdict": s["verdict"], "road_frames": s["n_road_frames"],
                   "detect_rate": s["detect_rate"]})
        if not new:
            break

    hw = ctx.get("highway", "")
    if expected is False:
        if s["n_road_frames"] >= seglib.MIN_FRAMES and s["detect_rate"] >= seglib.READABLE_RATE:
            final, reason = "human_review", "map conflict: camera finds lines but OSM says unmarked"
        else:
            final, reason = "unmarked_by_design", "OSM: no lane markings expected; camera agrees"
    elif s["verdict"] == "insufficient_evidence":
        final, reason = "human_review", "insufficient evidence even after fetching more frames"
    elif s["n_review"] > s["n_road_frames"]:
        final, reason = "human_review", "most frames have unclear road surface (e.g. concrete)"
    elif junction and s["verdict"] in ("not_readable", "at_risk"):
        final, reason = "human_review", "junction: lane geometry unreliable for automated screening"
    elif expected is None and s["verdict"] == "not_readable":
        final, reason = "human_review", "lanes not found and road type unknown in OSM"
    else:
        final = s["verdict"]
        reason = {"adas_readable": "lane lines found consistently with good contrast",
                  "at_risk": "lane lines found inconsistently or faint",
                  "not_readable": "lane lines rarely found on a road expected to be marked"}[final]

    if final == "not_readable" and hw in MAJOR_ROADS:
        priority = "high"
    elif final in ("not_readable", "at_risk"):
        priority = "medium"
    elif final == "human_review":
        priority = "review"
    else:
        priority = "none"
    log("decide", final_verdict=final, reason=reason, priority=priority)

    s.update(initial_verdict=trace[0]["verdict"], final_verdict=final, reason=reason,
             priority=priority, highway=hw, road_name=ctx.get("name", ""),
             expected_marked=expected, junction=junction,
             roads_nearby=ctx.get("roads_nearby", ""), osm_source=ctx.get("source", ""),
             merged_segments=n_members, n_sequences=len(sequences),
             fetch_rounds=rounds, _coords=coords)
    frame_rows = [{"segment_id": loc_id, "image_id": f.get("image_id", ""),
                   "origin": f.get("_origin", "original"), "status": f.get("status", ""),
                   "reason": f.get("reason", ""), "paint_contrast": f.get("paint_contrast", ""),
                   "lane_sides": f.get("lane_sides", ""), "lat": f.get("lat", ""),
                   "lon": f.get("lon", ""), "compass_angle": f.get("compass_angle", ""),
                   "captured_at": f.get("captured_at", ""), "sequence": f.get("sequence", ""),
                   "creator": f.get("creator", "")} for f in frames]
    return s, trace, frame_rows


def main():
    frames, rep = seglib.load_vehicle_frames()
    known_ids = {r["image_id"] for r in csv.DictReader(open(RESULTS))}
    raw = seglib.build_segments(frames)

    print(f"Input: {RESULTS}   Output prefix: '{PREFIX}'")
    print(f"Vehicle filter: kept {rep['frames_kept']}/{rep['frames_in']} frames; dropped "
          f"{rep['seq_slow']} slow sequences; {rep['seq_unknown']} unknown-speed sequences kept")
    print(f"Looking up OpenStreetMap context for {len(raw)} segments...")
    raws = []
    for i, (seq, fs) in enumerate(raw, 1):
        lat = st.mean(f["_lat"] for f in fs)
        lon = st.mean(f["_lon"] for f in fs)
        raws.append({"seq": seq, "frames": fs, "lat": lat, "lon": lon, "ctx": road_context(lat, lon),
                     "coords": [[f["_lon"], f["_lat"]] for f in fs]})
        if i % 20 == 0:
            print(f"  ...{i}/{len(raw)}")

    clusters = []
    for r in sorted(raws, key=lambda r: -len(r["frames"])):
        name = r["ctx"].get("name") or ""
        target = None
        if name:
            for c in clusters:
                if c["name"] == name and seglib.dist_m(c["lat"], c["lon"], r["lat"], r["lon"]) <= MERGE_M:
                    target = c
                    break
        if target:
            target["members"].append(r)
        else:
            clusters.append({"name": name, "lat": r["lat"], "lon": r["lon"], "members": [r]})
    print(f"Merged {len(raws)} segments into {len(clusters)} locations")

    budget = {"left": MAX_FETCH_CALLS}
    results, traces, all_frames = [], [], []
    for i, c in enumerate(clusters, 1):
        loc_id = f"L{i:04d}"
        lead = c["members"][0]
        seen, fs = set(), []
        for m in c["members"]:
            for f in m["frames"]:
                if f["image_id"] not in seen:
                    seen.add(f["image_id"])
                    fs.append(f)
        sequences = {m["seq"] for m in c["members"]}
        s, trace, frame_rows = run_location(loc_id, lead["seq"], sequences, fs, known_ids, budget,
                                            lead["ctx"], lead["coords"], len(c["members"]))
        results.append(s)
        traces.append({"segment_id": loc_id, "trace": trace})
        all_frames += frame_rows
        if i % 10 == 0:
            print(f"  ...{i}/{len(clusters)} locations (fetch budget left: {budget['left']})")

    cols = [k for k in results[0] if not k.startswith("_")]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    with open(OUT_QUEUE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["segment_id", "road_name", "highway", "final_verdict",
                                          "priority", "reason", "frame_ids"], extrasaction="ignore")
        w.writeheader()
        w.writerows([s for s in results if s["priority"] in ("high", "review")])
    with open(OUT_FRAMES, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FRAME_COLS)
        w.writeheader()
        w.writerows(all_frames)
    with open(OUT_TRACE, "w") as f:
        for t in traces:
            f.write(json.dumps(t) + "\n")
    features = []
    for s in results:
        c = s["_coords"]
        geom = {"type": "LineString", "coordinates": c} if len(c) > 1 else {"type": "Point", "coordinates": c[0]}
        features.append({"type": "Feature", "geometry": geom,
                         "properties": {k: v for k, v in s.items() if not k.startswith("_")}})
    OUT_GEO.write_text(json.dumps({"type": "FeatureCollection", "features": features}))

    osm_errors = sum(str(s["osm_source"]).startswith("osm_error") for s in results)
    print(f"\nLocations: {len(results)}   Fetch calls used: {MAX_FETCH_CALLS - budget['left']}   "
          f"OSM failed: {osm_errors}   Junction locations: {sum(s['junction'] for s in results)}   "
          f"Frames saved: {len(all_frames)} ({sum(f['origin'] == 'fetched' for f in all_frames)} fetched)")
    print("\nFinal verdicts:")
    for v, n in Counter(s["final_verdict"] for s in results).most_common():
        print(f"  {v:20s} {n}")
    print("\nPriority:")
    for p, n in Counter(s["priority"] for s in results).most_common():
        print(f"  {p:8s} {n}")
    print(f"\nWrote {OUT_CSV}, {OUT_GEO}, {OUT_QUEUE}, {OUT_TRACE}, {OUT_FRAMES}")


if __name__ == "__main__":
    main()
