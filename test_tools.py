"""Quick test of the agent tools on a few real segments."""
import csv

from agent_tools import fetch_frames, road_context

segs = list(csv.DictReader(open("segments.csv")))
picks = []
for verdict in ["adas_readable", "not_readable", "insufficient_evidence"]:
    picks += [s for s in segs if s["verdict"] == verdict][:2]

print("=== Tool B: OpenStreetMap road context ===")
for s in picks:
    ctx = road_context(float(s["lat"]), float(s["lon"]))
    print(f"{s['segment_id']} {s['area']:12s} {s['verdict']:22s} -> {ctx.get('highway') or '?':10s} "
          f"'{ctx.get('name', '')}' lanes={ctx.get('lanes', '')} "
          f"lane_markings={ctx.get('lane_markings', '') or '-'} "
          f"expected={ctx.get('expected_marked')} radius={ctx.get('radius_m')} ({ctx.get('source')})")

print("\n=== Tool A: fetch more frames for one insufficient segment ===")
s = next(x for x in segs if x["verdict"] == "insufficient_evidence")
frames, status = fetch_frames(float(s["lat"]), float(s["lon"]), sequence=s["sequence"],
                              exclude=s["frame_ids"].split(";"), limit=5)
print(f"{s['segment_id']} ({s['area']}): {status}")
print(f"new frames analysed: {len(frames)}")
for f in frames:
    same = "same-seq" if f["sequence"] == s["sequence"] else "other-seq"
    print(f"  {f['image_id']}  {same:9s}  status={f['status']:9s}  contrast={f['paint_contrast']}")
