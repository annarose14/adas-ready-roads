"""ADAS-Ready Roads - segment builder.
Reframed question: "Can a camera-based lane finder detect the lane lines along this
stretch of road?" Frames from the frozen pipeline (output_final/results.csv) are chained
into ~100 m segments per Mapillary sequence; each segment gets a machine
lane-detectability rate and a verdict. Single frames are never judged alone.
Vehicle-only: sequences moving slower than MIN_VEHICLE_KMH (walking / handheld, e.g.
shop-window photos) are excluded - the question is what a CAR's camera sees.
Outputs (when run directly): segments.csv and segments.geojson"""
import csv
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

RESULTS = Path("output_final/results.csv")
OUT_CSV = Path("segments.csv")
OUT_GEO = Path("segments.geojson")
SEG_MAX_LEN_M = 100       # max segment span
GAP_M = 50                # start a new segment if consecutive frames are further apart
MIN_FRAMES = 2            # fewer usable road frames -> insufficient_evidence
READABLE_RATE = 0.70      # starting values - calibrate on DEV segment labels
AT_RISK_RATE = 0.40
GOOD_CONTRAST = 75        # matches pipeline READY_T
MIN_VEHICLE_KMH = 10      # median sequence speed below this = pedestrian/handheld -> excluded
MAX_PAIR_GAP_S = 60       # only use consecutive frames this close in time to estimate speed
FOUND = {"ready", "degraded", "fail"}          # at least one lane line detected
ROADLIKE = FOUND | {"no_lanes"}                # a road frame where lines were looked for


def dist_m(lat1, lon1, lat2, lon2):
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) *
         math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return 6371000 * 2 * math.asin(math.sqrt(a))


def to_seconds(v):
    """Mapillary captured_at is epoch milliseconds; accept seconds too."""
    if v in ("", None):
        return 0.0
    t = float(v)
    return t / 1000.0 if t > 1e11 else t


def load_frames():
    frames = []
    for r in csv.DictReader(open(RESULTS)):
        if r["status"] == "unusable" or not r["lat"] or not r["lon"]:
            continue
        r["_lat"], r["_lon"] = float(r["lat"]), float(r["lon"])
        r["_t"] = to_seconds(r.get("captured_at"))
        frames.append(r)
    return frames


def sequence_speeds(frames):
    """Median speed (km/h) of each sequence from consecutive frames close in time; None if unknown."""
    by_seq = defaultdict(list)
    for f in frames:
        by_seq[f.get("sequence", "")].append(f)
    speeds = {}
    for seq, fs in by_seq.items():
        fs = sorted(fs, key=lambda f: f["_t"])
        vals = []
        for a, b in zip(fs, fs[1:]):
            dt = b["_t"] - a["_t"]
            if 0.5 <= dt <= MAX_PAIR_GAP_S:
                vals.append(dist_m(a["_lat"], a["_lon"], b["_lat"], b["_lon"]) / dt * 3.6)
        speeds[seq] = round(st.median(vals), 1) if vals else None
    return speeds


def load_vehicle_frames():
    """Frames from vehicle-speed sequences only. Unknown-speed sequences are kept (flagged)."""
    frames = load_frames()
    speeds = sequence_speeds(frames)
    kept, dropped = [], []
    for f in frames:
        sp = speeds.get(f.get("sequence", ""))
        f["_speed"] = sp
        (dropped if sp is not None and sp < MIN_VEHICLE_KMH else kept).append(f)
    report = {"frames_in": len(frames), "frames_kept": len(kept), "frames_dropped": len(dropped),
              "seq_total": len(speeds),
              "seq_slow": sum(1 for v in speeds.values() if v is not None and v < MIN_VEHICLE_KMH),
              "seq_unknown": sum(1 for v in speeds.values() if v is None)}
    return kept, report


def build_segments(frames):
    by_seq = defaultdict(list)
    for f in frames:
        by_seq[f["sequence"]].append(f)
    segments = []
    for seq, fs in by_seq.items():
        fs.sort(key=lambda f: f["_t"])
        cur = [fs[0]]
        for prev, f in zip(fs, fs[1:]):
            gap = dist_m(prev["_lat"], prev["_lon"], f["_lat"], f["_lon"])
            span = dist_m(cur[0]["_lat"], cur[0]["_lon"], f["_lat"], f["_lon"])
            if gap > GAP_M or span > SEG_MAX_LEN_M:
                segments.append((seq, cur))
                cur = [f]
            else:
                cur.append(f)
        segments.append((seq, cur))
    return segments


def summarise(seg_id, seq, fs):
    road = [f for f in fs if f["status"] in ROADLIKE]
    found = [f for f in road if f["status"] in FOUND]
    n_road = len(road)
    rate = len(found) / n_road if n_road else 0.0
    both = sum(1 for f in found if str(f["lane_sides"]) == "2") / n_road if n_road else 0.0
    contrasts = [float(f["paint_contrast"]) for f in found if f["paint_contrast"] not in ("", None)]
    med = st.median(contrasts) if contrasts else 0.0

    if n_road < MIN_FRAMES:
        verdict = "insufficient_evidence"
    elif rate >= READABLE_RATE and med >= GOOD_CONTRAST:
        verdict = "adas_readable"
    elif rate >= AT_RISK_RATE:
        verdict = "at_risk"
    else:
        verdict = "not_readable"
    confidence = "high" if n_road >= 5 else "medium" if n_road >= 3 else "low"

    lat = st.mean(f["_lat"] for f in fs)
    lon = st.mean(f["_lon"] for f in fs)
    length = dist_m(fs[0]["_lat"], fs[0]["_lon"], fs[-1]["_lat"], fs[-1]["_lon"])
    return {
        "segment_id": seg_id, "sequence": seq, "area": fs[0]["area"],
        "lat": round(lat, 6), "lon": round(lon, 6), "length_m": round(length),
        "n_frames": len(fs), "n_road_frames": n_road, "n_lines_found": len(found),
        "detect_rate": round(rate, 2), "both_lines_rate": round(both, 2),
        "median_contrast": round(med, 1),
        "n_review": sum(f["status"] == "review" for f in fs),
        "n_not_road": sum(f["status"] == "not_road" for f in fs),
        "verdict": verdict, "confidence": confidence,
        "frame_ids": ";".join(f["image_id"] for f in fs),
        "_coords": [[f["_lon"], f["_lat"]] for f in fs],
    }


def main():
    frames, rep = load_vehicle_frames()
    print(f"Vehicle filter: kept {rep['frames_kept']}/{rep['frames_in']} frames; "
          f"{rep['seq_slow']} of {rep['seq_total']} sequences slower than {MIN_VEHICLE_KMH} km/h dropped; "
          f"{rep['seq_unknown']} sequences with unknown speed kept")
    segs = [summarise(f"S{i:04d}", seq, fs) for i, (seq, fs) in enumerate(build_segments(frames), 1)]
    cols = [k for k in segs[0] if not k.startswith("_")]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(segs)
    features = []
    for s in segs:
        coords = s["_coords"]
        geom = ({"type": "LineString", "coordinates": coords} if len(coords) > 1
                else {"type": "Point", "coordinates": coords[0]})
        features.append({"type": "Feature", "geometry": geom,
                         "properties": {k: v for k, v in s.items() if not k.startswith("_")}})
    OUT_GEO.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    print(f"Segments: {len(segs)}")
    for v in ["adas_readable", "at_risk", "not_readable", "insufficient_evidence"]:
        print(f"  {v:22s} {sum(s['verdict'] == v for s in segs)}")
    print(f"Wrote {OUT_CSV} and {OUT_GEO}")


if __name__ == "__main__":
    main()
