"""ADAS-Ready Roads - segment agent.
Loop per road segment: PERCEIVE (frozen OpenCV 5 pipeline results) -> CONTEXT (OpenStreetMap)
-> ACT (fetch more Mapillary frames when evidence is thin; reject cross-street frames by
heading; skip known images) -> RE-PERCEIVE -> DECIDE -> PRIORITISE / ESCALATE to human review.
Every step is logged to agent_trace.jsonl so decisions are auditable.
Outputs: segments_agent.csv, segments_agent.geojson, review_queue.csv, agent_trace.jsonl"""
import csv
import json
import math
from collections import Counter
from pathlib import Path

import segments as seglib
from agent_tools import fetch_frames, road_context

FETCH_RADII = (0.0004, 0.0008)   # ~45 m, then ~90 m
FETCH_LIMIT = 8                  # new frames per fetch round
MAX_FETCH_CALLS = 60             # global budget (API politeness + runtime)
MIN_CONFIDENT_FRAMES = 3         # fewer road frames -> try to fetch more
HEADING_TOL = 35                 # degrees; keep frames along the road (either direction)
MAJOR_ROADS = {"motorway", "trunk", "primary", "secondary"}

OUT_CSV = Path("segments_agent.csv")
OUT_GEO = Path("segments_agent.geojson")
OUT_QUEUE = Path("review_queue.csv")
OUT_TRACE = Path("agent_trace.jsonl")


def prep(f):
    f["_lat"], f["_lon"] = float(f["lat"]), float(f["lon"])
    f["_t"] = float(f["captured_at"]) if f.get("captured_at") else 0.0
    return f


def mean_heading(frames):
    angles = [float(f["compass_angle"]) for f in frames if f.get("compass_angle") not in ("", None)]
    if not angles:
        return None
    x = sum(math.cos(math.radians(a)) for a in angles)
    y = sum(math.sin(math.radians(a)) for a in angles)
    return math.degrees(math.atan2(y, x)) % 360


def aligned(frame, heading, sequence):
    if frame.get("sequence") == sequence or heading is None:
        return True
    a = frame.get("compass_angle")
    if a in ("", None):
        return False
    d = abs((float(a) - heading + 180) % 360 - 180)
    return d <= HEADING_TOL or d >= 180 - HEADING_TOL


def run_segment(seg_id, seq, frames, known_ids, budget):
    trace = []

    def log(step, **kw):
        trace.append({"step": step, **kw})

    s = seglib.summarise(seg_id, seq, frames)
    coords = s["_coords"]
    log("perceive", n_frames=s["n_frames"], n_road_frames=s["n_road_frames"],
        detect_rate=s["detect_rate"], median_contrast=s["median_contrast"], verdict=s["verdict"])

    ctx = road_context(s["lat"], s["lon"])
    expected = ctx.get("expected_marked")
    log("road_context", highway=ctx.get("highway", ""), name=ctx.get("name", ""),
        lanes=ctx.get("lanes", ""), lane_markings=ctx.get("lane_markings", ""),
        expected_marked=expected, source=ctx.get("source", ""))

    rounds = 0
    if expected is False:
        log("skip_fetch", why="OSM says no lane markings expected; extra frames would not change the decision")
    while (s["n_road_frames"] < MIN_CONFIDENT_FRAMES and expected is not False
           and rounds < len(FETCH_RADII) and budget["left"] > 0):
        heading = mean_heading(frames)
        new, status = fetch_frames(s["lat"], s["lon"], sequence=seq, exclude=known_ids,
                                   radius_deg=FETCH_RADII[rounds], limit=FETCH_LIMIT)
        known_ids.update(f["image_id"] for f in new)
        kept = [prep(f) for f in new if aligned(f, heading, seq)]
        before = (s["verdict"], s["n_road_frames"], s["detect_rate"])
        frames = frames + kept
        s = seglib.summarise(seg_id, seq, frames)
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
             expected_marked=expected, fetch_rounds=rounds, _coords=coords)
    return s, trace


def main():
    frames = seglib.load_frames()
    known_ids = {r["image_id"] for r in csv.DictReader(open(seglib.RESULTS))}
    raw = seglib.build_segments(frames)
    budget = {"left": MAX_FETCH_CALLS}
    results, traces = [], []
    for i, (seq, fs) in enumerate(raw, 1):
        seg_id = f"S{i:04d}"
        s, trace = run_segment(seg_id, seq, fs, known_ids, budget)
        results.append(s)
        traces.append({"segment_id": seg_id, "trace": trace})
        if i % 10 == 0:
            print(f"  ...{i}/{len(raw)} segments (fetch budget left: {budget['left']})")

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

    print(f"\nSegments: {len(results)}   Fetch calls used: {MAX_FETCH_CALLS - budget['left']}")
    print("\nFinal verdicts:")
    for v, n in Counter(s["final_verdict"] for s in results).most_common():
        print(f"  {v:20s} {n}")
    print("\nPriority:")
    for p, n in Counter(s["priority"] for s in results).most_common():
        print(f"  {p:8s} {n}")
    print("\nInitial -> final (how the agent changed decisions):")
    for (a, b), n in Counter((s["initial_verdict"], s["final_verdict"]) for s in results).most_common():
        print(f"  {a:22s} -> {b:20s} {n}")

    print("\nExample traces:")
    shown = set()
    for want in ["act_fetch", "skip_fetch", "high"]:
        for t, s in zip(traces, results):
            steps = [x["step"] for x in t["trace"]]
            hit = (want in steps) or (want == "high" and s["priority"] == "high")
            if hit and t["segment_id"] not in shown:
                shown.add(t["segment_id"])
                print(f"\n  {t['segment_id']} ({s['road_name'] or '?'}):")
                for step in t["trace"]:
                    print("    " + json.dumps(step))
                break
    print(f"\nWrote {OUT_CSV}, {OUT_GEO}, {OUT_QUEUE}, {OUT_TRACE}")


if __name__ == "__main__":
    main()
