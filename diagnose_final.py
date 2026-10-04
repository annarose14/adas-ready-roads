"""Read-only failure analysis of the FINAL test (no tuning, no re-scoring).
For every disagreement, show which decision branch produced the system's answer."""
import csv
from collections import Counter

human = {r["location_id"]: r["label"] for r in csv.DictReader(open("labels_final.csv"))}
rows = {r["segment_id"]: r for r in csv.DictReader(open("final_segments_agent.csv"))}
SYS = {"adas_readable": "yes", "at_risk": "yes", "not_readable": "no",
       "unmarked_by_design": "no", "human_review": "review"}

print("Human YES but system NO - which branch?")
branch = Counter()
for k, h in sorted(human.items()):
    r = rows.get(k)
    if not r or h != "yes" or SYS[r["final_verdict"]] != "no":
        continue
    branch[(r["final_verdict"], r["expected_marked"])] += 1
    print(f"  {k}  {r['road_name'][:22]:22s} {r['highway']:12s} OSM_marked={r['expected_marked']:5s} "
          f"painted_rate={r['detect_rate']:>4s} road_frames={r['n_road_frames']:>2s} -> {r['final_verdict']}")
print("\nBy branch (verdict, OSM expected_marked):")
for (v, e), n in branch.most_common():
    print(f"  {v:20s} OSM_marked={e:5s}  {n}")

print("\nPainted-lane rate distribution by human label (all decided locations):")
for lab in ("yes", "no"):
    rates = sorted(float(rows[k]["detect_rate"]) for k, h in human.items()
                   if h == lab and k in rows and rows[k]["final_verdict"] != "human_review")
    print(f"  human {lab:3s}: n={len(rates):2d}  " + " ".join(f"{x:.2f}" for x in rates))
