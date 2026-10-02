"""Find the best ROAD_MIN_FRAC on the DEV set: keep real roads, reject non-roads."""
import csv

preds = {r["image_id"]: r for r in csv.DictReader(open("output_v3a/results.csv"))}
roads, nonroads = [], []
for r in csv.DictReader(open("dev_labels.csv")):
    p = preds.get(r["image_id"])
    if not p or p["road_frac"] in ("", None):
        continue
    frac = float(p["road_frac"])
    if r["my_label"] in ("ready", "degraded", "fail"):
        roads.append((frac, r["my_label"], p["area"], r["image_id"]))
    elif r["my_label"] == "none":
        nonroads.append((frac, r["my_label"], p["area"], r["image_id"]))

print("road_frac values on DEV (sorted):")
print(f"{'frac':>6s}  {'YOU':10s} image")
for frac, lbl, area, iid in sorted(roads + nonroads):
    print(f"{frac:>6.2f}  {lbl:10s} {area}/{iid}.jpg")

print(f"\nReal roads: {len(roads)}   No-marking scenes: {len(nonroads)}")
print(f"{'threshold':>10s}{'roads lost':>12s}{'non-roads kept':>16s}")
for t in [x / 100 for x in range(0, 55, 5)]:
    lost = sum(f < t for f, *_ in roads)
    kept = sum(f >= t for f, *_ in nonroads)
    print(f"{t:>10.2f}{f'{lost}/{len(roads)}':>12s}{f'{kept}/{len(nonroads)}':>16s}")
