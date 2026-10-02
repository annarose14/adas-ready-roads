"""Find the best road-gate threshold on the DEV set.
Usage: python tune_road.py <results.csv> <column>
e.g.   python tune_road.py output_v3b/results.csv drivable_frac"""
import csv
import sys

results_file = sys.argv[1] if len(sys.argv) > 1 else "output_v3b/results.csv"
column = sys.argv[2] if len(sys.argv) > 2 else "drivable_frac"

preds = {r["image_id"]: r for r in csv.DictReader(open(results_file))}
roads, nonroads = [], []
for r in csv.DictReader(open("dev_labels.csv")):
    p = preds.get(r["image_id"])
    if not p or p.get(column) in ("", None):
        continue
    frac = float(p[column])
    item = (frac, r["my_label"], p["area"], r["image_id"])
    if r["my_label"] in ("ready", "degraded", "fail"):
        roads.append(item)
    elif r["my_label"] == "none":
        nonroads.append(item)

print(f"{column} values on DEV (sorted):")
print(f"{'value':>6s}  {'YOU':10s} image")
for frac, lbl, area, iid in sorted(roads + nonroads):
    print(f"{frac:>6.2f}  {lbl:10s} {area}/{iid}.jpg")

print(f"\nReal roads: {len(roads)}   No-marking scenes: {len(nonroads)}")
print(f"{'threshold':>10s}{'roads lost':>12s}{'non-roads kept':>16s}")
for t in [x / 100 for x in range(0, 75, 5)]:
    lost = sum(f < t for f, *_ in roads)
    kept = sum(f >= t for f, *_ in nonroads)
    print(f"{t:>10.2f}{f'{lost}/{len(roads)}':>12s}{f'{kept}/{len(nonroads)}':>16s}")
