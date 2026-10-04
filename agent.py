"""ADAS-Ready Roads - location agent (v5: OpenCV-gated vision-language evidence).
Usage: python agent.py [results_csv] [output_prefix]      defaults: output_final/results.csv ""
Per location:
  PERCEIVE   frozen OpenCV 5 pipeline results (quality gate, segmentation road gate, own-bonnet
             removal, lane network + paint check -> logged as opencv_painted_rate)
  CONTEXT    OpenStreetMap road class / lane_markings tag / junction
  ACT        fetch more vehicle-speed Mapillary frames if < 3 road frames (any road type)
  ASK        Tool C (Bedrock VLM) on the best OpenCV-approved road frames, ADAPTIVELY:
             ask 2; stop if they agree; otherwise keep asking up to 6
  DECIDE     lines visible -> ADAS-readable (OSM lane_markings=no -> map conflict review)
             no lines + marked road (OSM) -> NOT READABLE (HIGH on major roads)
             no lines + minor/unknown road -> unmarked by design
             split / unclear / no usable road frames -> human review
Every step logged to <prefix>agent_trace.jsonl; every frame (with VLM answer) to <prefix>agent_frames.csv."""
import csv
import json
import math
import statistics as st
import sys
from collections import Counter
from pathlib import Path

import segments as seglib
from agent_tools import fetch_frames, road_context
from vlm_tool import MODEL_ID as VLM_MODEL, VLMBlocked, ask_lines_visible

RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output_final/results.csv")
PREFIX = sys.argv[2] if len(sys.argv) > 2 else ""
seglib.RESULTS = RESULTS

MERGE_M = 30
FETCH_RADII = (0.0004, 0.0008)
FETCH_LIMIT = 8
MAX_FETCH_CALLS = 60
MIN_CONFIDENT_FRAMES = 3
HEADING_TOL = 35
VLM_MIN, VLM_MAX = 2, 6          # adaptive questioning: stop after 2 if unanimous, else up to 6
VLM_AGREE = 0.67                 # share of decided answers needed for a yes/no location verdict
MAJOR_ROADS = {"motorway", "trunk", "primary", "secondary"}

OUT_CSV = Path(f"{PREFIX}segments_agent.csv")
OUT_GEO = Path(f"{PREFIX}segments_agent.geojson")
OUT_QUEUE = Path(f"{PREFIX}review_queue.csv")
OUT_TRACE = Path(f"{PREFIX}agent_trace.jsonl")
OUT_FRAMES = Path(f"{PREFIX}agent_frames.csv")
FRAME_COLS = ["segment_id", "image_id", "origin", "status", "reason", "lanes_detected",
              "lanes_painted", "vlm", "lat", "lon", "compass_angle", "captured_at", "sequence", "creator"]

ROOTS = [p for base in Path(".").glob("data*") if base.is_dir() and base.name != "data_agent"
         for p in base.iterdir() if p.is_dir()]


def find_image(image_id):
    for d in ROOTS:
        p = d / f"{image_id}.jpg"
        if p.exists():
            return p
    p = Path("data_agent") / f"{image_id}.jpg"
    return p if p.exists() else None


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


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def ask_vlm(frames, log):
    """Adaptive evidence gathering with the VLM on the best OpenCV-approved road frames."""
    road = [f for f in frames if f.get("status") in seglib.ROADLIKE]
    road.sort(key=lambda f: (f.get("_origin") != "original", -_num(f.get("drivable_frac"))))
    answers = []
    for f in road:
        if len(answers) >= VLM_MAX:
            break
        p = find_image(f["image_id"])
        if p is None:
            continue
        a = ask_lines_visible(p, f["image_id"])
        f["_vlm"] = a
        answers.append(a)
        yes, no = answers.count("YES"), answers.count("NO")
        if len(answers) >= VLM_MIN and ((yes >= VLM_MIN and no == 0) or (no >= VLM_MIN and yes == 0)):
            break
    yes, no, unclear = answers.count("YES"), answers.count("NO"), answers.count("UNCLEAR")
    decided = yes + no
    if decided == 0:
        verdict = "unclear"
    elif yes / decided >= VLM_AGREE:
        verdict = "yes"
    elif no / decided >= VLM_AGREE:
        verdict = "no"
    else:
        verdict = "split"
    log("ask_vlm", model=VLM_MODEL, road_frames_available=len(road), asked=len(answers),
        yes=yes, no=no, unclear=unclear, verdict=verdict)
    return verdict, yes, no, len(answers)


def run_location(loc_id, main_seq, sequences, frames, known_ids, budget, ctx, coords, n_members):
    trace = []

    def log(step, **kw):
        trace.append({"step": step, **kw})

    for f in frames:
        f.setdefault("_origin", "original")
    s = seglib.summarise(loc_id, main_seq, frames)
    log("perceive", merged_segments=n_members, sequences=len(sequences), n_frames=s["n_frames"],
        n_road_frames=s["n_road_frames"], opencv_painted_rate=s["detect_rate"])

    expected = ctx.get("expected_marked")
    tag_no = ctx.get("lane_markings", "") == "no"
    log("road_context", highway=ctx.get("highway", ""), name=ctx.get("name", ""),
        lanes=ctx.get("lanes", ""), lane_markings=ctx.get("lane_markings", ""),
        expected_marked=expected, junction=bool(ctx.get("junction")),
        roads_nearby=ctx.get("roads_nearby", ""), source=ctx.get("source", ""))

    rounds = 0
    while s["n_road_frames"] < MIN_CONFIDENT_FRAMES and rounds < len(FETCH_RADII) and budget["left"] > 0:
        heading = mean_heading(frames)
        new, status = fetch_frames(s["lat"], s["lon"], sequence=main_seq, exclude=known_ids,
                                   radius_deg=FETCH_RADII[rounds], limit=FETCH_LIMIT)
        known_ids.update(f["image_id"] for f in new)
        kept = [prep(f) for f in new if aligned(f, heading, sequences)]
        for f in kept:
            f["_origin"] = "fetched"
        before = s["n_road_frames"]
        frames = frames + kept
        s = seglib.summarise(loc_id, main_seq, frames)
        rounds += 1
        budget["left"] -= 1
        log("act_fetch", round=rounds, radius_m=int(FETCH_RADII[rounds - 1] * 111000), api=status,
            fetched=len(new), kept_after_heading_filter=len(kept),
            road_frames_before=before, road_frames_after=s["n_road_frames"])
        if not new:
            break

    vlm, yes, no, asked = ask_vlm(frames, log)
    hw = ctx.get("highway", "")
    if vlm == "yes":
        if tag_no:
            final, reason = "human_review", "map conflict: lane lines visible but OSM says lane_markings=no"
        else:
            final, reason = "adas_readable", f"painted lane lines visible ({yes}/{asked} photos)"
    elif vlm == "no":
        if expected is True:
            final, reason = "not_readable", f"no visible lane lines ({no}/{asked} photos) on a road expected to be marked"
        else:
            final, reason = "unmarked_by_design", f"no visible lane lines ({no}/{asked} photos); minor or unclassified road"
    elif vlm == "split":
        final, reason = "human_review", f"camera evidence split ({yes} yes / {no} no)"
    else:
        final, reason = "human_review", "no usable road photos even after fetching more"

    if final == "not_readable" and hw in MAJOR_ROADS:
        priority = "high"
    elif final == "not_readable":
        priority = "medium"
    elif final == "human_review":
        priority = "review"
    else:
        priority = "none"
    log("decide", final_verdict=final, reason=reason, priority=priority)

    s.update(initial_verdict=s["verdict"], opencv_painted_rate=s["detect_rate"], vlm_verdict=vlm,
             vlm_yes=yes, vlm_no=no, vlm_asked=asked, final_verdict=final, reason=reason,
             priority=priority, highway=hw, road_name=ctx.get("name", ""), expected_marked=expected,
             junction=bool(ctx.get("junction")), roads_nearby=ctx.get("roads_nearby", ""),
             osm_source=ctx.get("source", ""), merged_segments=n_members,
             n_sequences=len(sequences), fetch_rounds=rounds, _coords=coords)
    frame_rows = [{"segment_id": loc_id, "image_id": f.get("image_id", ""),
                   "origin": f.get("_origin", "original"), "status": f.get("status", ""),
                   "reason": f.get("reason", ""), "lanes_detected": f.get("lanes_detected", ""),
                   "lanes_painted": f.get("lanes_painted", ""), "vlm": f.get("_vlm", ""),
                   "lat": f.get("lat", ""), "lon": f.get("lon", ""),
                   "compass_angle": f.get("compass_angle", ""), "captured_at": f.get("captured_at", ""),
                   "sequence": f.get("sequence", ""), "creator": f.get("creator", "")} for f in frames]
    return s, trace, frame_rows


def main():
    frames, rep = seglib.load_vehicle_frames()
    known_ids = {r["image_id"] for r in csv.DictReader(open(RESULTS))}
    raw = seglib.build_segments(frames)

    print(f"Input: {RESULTS}   Output prefix: '{PREFIX}'   VLM: {VLM_MODEL}")
    print(f"Vehicle filter: kept {rep['frames_kept']}/{rep['frames_in']} frames; dropped "
          f"{rep['seq_slow']} slow sequences; {rep['seq_unknown']} unknown-speed sequences kept")
    raws = []
    for i, (seq, fs) in enumerate(raw, 1):
        lat = st.mean(f["_lat"] for f in fs)
        lon = st.mean(f["_lon"] for f in fs)
        raws.append({"seq": seq, "frames": fs, "lat": lat, "lon": lon, "ctx": road_context(lat, lon),
                     "coords": [[f["_lon"], f["_lat"]] for f in fs]})
        if i % 20 == 0:
            print(f"  OSM ...{i}/{len(raw)}")

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
    try:
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
    except VLMBlocked as e:
        sys.exit(f"\nSTOPPED: AWS blocked Bedrock access ({e}). VLM answers so far are cached; re-run later.")

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
        w = csv.DictWriter(f, fieldnames=FRAME_COLS, extrasaction="ignore")
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

    calls = sum(s["vlm_asked"] for s in results)
    print(f"\nLocations: {len(results)}   Fetch calls: {MAX_FETCH_CALLS - budget['left']}   "
          f"VLM questions: {calls} (avg {calls / max(1, len(results)):.1f}/location)")
    print("Final verdicts: " + ", ".join(f"{v}={n}" for v, n in
                                         Counter(s["final_verdict"] for s in results).most_common()))
    print(f"Wrote {OUT_CSV}, {OUT_GEO}, {OUT_QUEUE}, {OUT_TRACE}, {OUT_FRAMES}")


if __name__ == "__main__":
    main()
